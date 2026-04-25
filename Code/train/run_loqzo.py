# -*- coding: utf-8 -*-
# =========================
# run_loqzo.py 最终独立版（完整文件）
#
# 说明：
# 1) 这是 LoQZO 的“完整独立入口”，不是简单包装器；
# 2) 它与 run.py 代码主体保持一致，避免再出现参数体系漂移；
# 3) 默认更适合 LoQZO：
#    - trainer_module 默认优先使用 trainer_loqzo
#    - prefer_local_model 默认 False，避免 1.3B/13B 误匹配
#    - 当 trainer=zo_lowbit / zo_lowbit_ft 且未显式关闭时，会自动打开 loqzo_enable
#
# 使用建议：
# - 如果跑 LoQZO，请优先使用这个文件；
# - 如果跑 QuZO / MeZO / FO，对应使用 run.py 更清晰；
# =========================

"""
run.py

这是 QuZO / MeZO / FO / QFT 等实验的统一入口。

这个文件的设计目标：
1. 统一项目路径（Data / Models / outputs）
2. 统一模型解析（优先显式 model_path，其次可选本地目录，最后回退 Hugging Face repo）
3. 统一训练、评估、日志与 checkpoint 行为
4. 尽量少改动你现有仓库的 trainer / tasks / utils 逻辑
5. 默认输出尽量干净：一个 train.log + checkpoint 目录

与仓库历史版本相比，这版重点修复：
- 不再用模糊规则把 opt-1.3b 误匹配到 opt-13b
- 默认不再自动写多余的 result json 文件（只有显式指定 result_file 才写）
- 训练结束后的评估支持 32-shot demonstration，更贴近 QuZO 论文里 decoder-only 模型的设定
- 默认关闭 wandb，避免无 TTY 环境报错
- 支持单卡 / 单进程多卡模型并行 / torchrun DDP / ZeRO-3 启动

建议配合：
- Code/script/quzo.sh
- Code/script/loqzo.sh
一起使用。
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForTokenClassification,
    EvalPrediction,
    HfArgumentParser,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

# =========================
# 路径初始化
# =========================
CURRENT_FILE = Path(__file__).resolve()
TRAIN_DIR = CURRENT_FILE.parent
CODE_ROOT = TRAIN_DIR.parent
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "Data"
DEFAULT_MODELS_ROOT = PROJECT_ROOT / "Models"
DEFAULT_BASE_MODELS_ROOT = DEFAULT_MODELS_ROOT / "base_models"
DEFAULT_QUANT_MANIFESTS_ROOT = DEFAULT_MODELS_ROOT / "quantized_manifests"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# qft / QZO 会加载 quant_cuda 扩展；自动加入 Code/quant/build/lib.*，
# 避免只靠 shell 脚本配置 PYTHONPATH 时才能运行。
QUANT_BUILD_ROOT = CODE_ROOT / "quant" / "build"
if QUANT_BUILD_ROOT.exists():
    for _p in sorted(QUANT_BUILD_ROOT.glob("lib.*")):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
if str(CODE_ROOT / "quant") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "quant"))

import tasks  # noqa: E402
from metrics import calculate_metric  # noqa: E402
from utils import (  # noqa: E402
    DataCollatorWithPaddingAndNesting,
    NondiffCollator,
    Prediction,
    SIGUSR1Callback,
    count_time,
    encode_prompt,
    forward_wrap_with_option_len,
    temp_seed,
    write_metrics_to_file,
)

logger = logging.getLogger(__name__)


# =========================
# 一些常用模型别名
# =========================
MODEL_ALIAS_REGISTRY: Dict[str, Dict[str, str]] = {
    "OPT-125M": {"hf": "facebook/opt-125m", "family": "opt"},
    "OPT-1.3B": {"hf": "facebook/opt-1.3b", "family": "opt"},
    "OPT-2.7B": {"hf": "facebook/opt-2.7b", "family": "opt"},
    "OPT-13B": {"hf": "facebook/opt-13b", "family": "opt"},
    "OPT-30B": {"hf": "facebook/opt-30b", "family": "opt"},
    "Llama2-7B": {"hf": "meta-llama/Llama-2-7b-hf", "family": "llama"},
    "Llama2-13B": {"hf": "meta-llama/Llama-2-13b-hf", "family": "llama"},
    "Llama3-8B": {"hf": "meta-llama/Meta-Llama-3-8B", "family": "llama"},
    "Mistral-7B": {"hf": "mistralai/Mistral-7B-v0.3", "family": "mistral"},
    "Mistral-7B-Instruct": {"hf": "mistralai/Mistral-7B-Instruct-v0.3", "family": "mistral"},
}

# 任务名规范化表：允许用户用大小写不敏感或常见别名来写任务名。
# 最终都会映射成 tasks.py 中真正存在的类名前缀，避免任务名写错导致找不到类。
TASK_ALIAS_REGISTRY: Dict[str, str] = {
    "sst2": "SST2",
    "rte": "RTE",
    "cb": "CB",
    "boolq": "BoolQ",
    "wsc": "WSC",
    "wic": "WIC",
    "multirc": "MultiRC",
    "copa": "Copa",
    "record": "ReCoRD",
    "squad": "SQuAD",
    "drop": "DROP",
    "winogrande": "WinoGrande",
    "wikitext": "WikiText",
}


def normalize_task_name(task_name: str) -> str:
    key = str(task_name).strip().replace("_", "").replace("-", "").lower()
    return TASK_ALIAS_REGISTRY.get(key, task_name)


def _normalize_name(text: str) -> str:
    """把字符串规整成便于比较的形式，同时保留小数点，避免 1.3B / 13B 混淆。"""
    s = str(text).lower().strip()
    s = s.replace("/", "-").replace("\\", "-").replace("_", "-")
    s = re.sub(r"[^a-z0-9.\-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _safe_local_name_patterns(hf_name: str):
    return [
        hf_name,
        hf_name.replace("/", "__"),
        hf_name.replace("/", "--"),
        Path(hf_name).name,
    ]


def _looks_like_hf_model_dir(path: Path) -> bool:
    """判断一个目录是否像 Hugging Face 模型目录。"""
    if not path.is_dir():
        return False
    indicator_files = [
        "config.json",
        "tokenizer_config.json",
        "generation_config.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
    ]
    return any((path / name).exists() for name in indicator_files)


class WarningLimiter(logging.Filter):
    """限制重复 warning 的数量，避免日志被刷屏。"""

    def __init__(self, target_text: str, max_times: int = 5):
        super().__init__()
        self.target_text = target_text
        self.max_times = max_times
        self.count = 0

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if self.target_text in msg:
            self.count += 1
            if self.count > self.max_times:
                return False
            if self.count == self.max_times:
                record.msg = f"{self.target_text}（后续相同警告已省略）"
        return True


class EpochProgressCallback(TrainerCallback):
    """在 log 文件里显式记录每个 epoch 的开始与结束。"""

    def __init__(self) -> None:
        self._epoch_total = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._epoch_total = state.num_train_epochs
        logger.info("开始训练：总步数=%s，总 epoch≈%s", state.max_steps, state.num_train_epochs)
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch) + 1 if state.epoch is not None else 1
        logger.info("========== Epoch %d/%s 开始 (epoch_progress=%.4f) ==========", current_epoch, self._epoch_total, state.epoch or 0.0)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        logger.info("========== Epoch %d 结束 (global_step=%d) ==========", current_epoch, state.global_step)
        return control


def setup_logging(output_dir: Union[str, Path], log_filename: str = "train.log") -> Path:
    """配置日志。

    - 默认行为：同时输出到终端和 output_dir/log_filename。
    - 如果 shell 脚本已经通过 tee 把 stdout/stderr 重定向到同一个日志文件（LOQZO_SHELL_TEE=1），
      那么这里只保留 StreamHandler，避免同一条日志在文件里重复两份。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_filename

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(WarningLimiter("Exceed max length", max_times=3))
    root_logger.addHandler(stream_handler)

    if os.environ.get("LOQZO_SHELL_TEE", "0") != "1":
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(WarningLimiter("Exceed max length", max_times=3))
        root_logger.addHandler(file_handler)

    logging.getLogger(__name__).setLevel(logging.INFO)
    return log_path


def configure_reporting_and_wandb(args) -> None:
    """统一处理 wandb/report_to。默认关闭。"""
    report_to = getattr(args, "report_to", None)
    report_to_text = str(report_to).lower()
    if getattr(args, "use_wandb", False) or report_to_text == "wandb":
        # 显式开启 wandb
        os.environ.pop("WANDB_DISABLED", None)
        if getattr(args, "wandb_api_key", None):
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
        if isinstance(report_to, str):
            args.report_to = [report_to]
    else:
        os.environ["WANDB_DISABLED"] = "true"
        args.report_to = []


def set_seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class OurArguments(TrainingArguments):
    # -------------------- 路径相关 --------------------
    output_dir: str = str(DEFAULT_OUTPUT_ROOT)
    project_root: str = str(PROJECT_ROOT)
    data_root: str = str(DEFAULT_DATA_ROOT)
    models_root: str = str(DEFAULT_MODELS_ROOT)
    base_models_root: str = str(DEFAULT_BASE_MODELS_ROOT)
    quantized_manifests_root: str = str(DEFAULT_QUANT_MANIFESTS_ROOT)

    # -------------------- 模型加载相关 --------------------
    model_name: str = "facebook/opt-125m"
    model_path: Optional[str] = None
    prefer_local_model: bool = False
    local_files_only: bool = False
    hf_token: Optional[str] = None

    load_float16: bool = False
    load_bfloat16: bool = False
    load_int8: bool = False
    load_int4: bool = False
    max_length: int = 2048
    no_auto_device: bool = False
    force_disable_bnb: bool = False

    # -------------------- GPU / 启动相关 --------------------
    gpu_id: Optional[int] = None
    gpu_ids: Optional[str] = None
    launch_mode: str = "single"  # single / model_parallel / ddp / zero3

    # -------------------- 分布式 / torchrun 兼容 --------------------
    distributed: bool = False
    slurm: bool = False
    local_rank: int = -1
    local_world_size: int = -1
    delayed_start: bool = False

    # -------------------- 数据与任务 --------------------
    task_name: str = "SST2"
    num_train: int = 0
    num_dev: Optional[int] = None
    num_eval: Optional[int] = None
    num_train_sets: Optional[int] = None
    train_set_seed: Optional[int] = None
    result_file: Optional[str] = None
    write_result_json: bool = False

    # 训练后评估时是否使用 few-shot demonstrations（更接近 QuZO 论文的 decoder-only 设定）
    use_eval_demos_after_training: bool = True
    eval_num_demos: int = 32
    eval_demo_seed: int = 0

    # -------------------- 训练器选择 --------------------
    trainer: str = "regular"  # none / regular / zo / zo_lowbit / zo_lowbit_ft
    trainer_module: Optional[str] = None
    tuning_type: str = "ft"  # ft / lora / prefix / loretta_rep / qft

    # -------------------- 零阶优化 --------------------
    zo_eps: float = 1e-3
    num_pertub: int = 1
    perturb_bits: int = 4
    quantized_perturb_ours: bool = False
    mask_ratio: int = 0

    # -------------------- FO/梯度量化兼容参数 --------------------
    # 这四个参数主要是为了兼容旧版 quzo.sh / loqzo.sh。
    # 当前这版 run.py 默认不会主动使用它们；如果后续 trainer 需要，可再接上具体逻辑。
    fo_quant_grad: bool = False
    fo_quant_bits: int = 8
    use_grad_pre_hook_quant: bool = False
    grad_pre_hook_bits: int = 8

    # -------------------- prompt / calibration --------------------
    sfc: bool = False
    icl_sfc: bool = False
    only_train_option: bool = True
    train_as_classification: bool = False
    non_diff: bool = False

    # -------------------- Prefix Tuning --------------------
    prefix_tuning: bool = False
    num_prefix: int = 5
    no_reparam: bool = True
    prefix_init_by_real_act: bool = True

    # -------------------- LoRA / Loretta --------------------
    lora_alpha: int = 16
    lora_r: int = 8
    lora_dropout: float = 0.0

    tensor_rank: int = 8
    target_modules: Optional[List[str]] = None
    task_type: str = "CAUSAL_LM"
    rep_bottleneck: int = 16
    rep_alpha: int = 16

    # -------------------- 生成任务 --------------------
    sampling: bool = False
    temperature: float = 1.0
    num_beams: int = 1
    top_k: Optional[int] = None
    top_p: float = 0.95
    max_new_tokens: int = 50
    eos_token: str = "\n"

    # -------------------- 量化训练 / qft --------------------
    # precision_profile 用来一键对齐 QuZO 表格里的“Data Precision”：
    #   FP_W32A32    -> 不做 W/A 量化，等价全精度前向；此时 QZO-scale 没有 alpha，会自动退化为 LoQZO-only；
    #   FP_W8A8      -> 权重/激活都用 float codebook 的 8-bit 量化；
    #   INT_W8A8     -> 权重/激活都用 int codebook 的 8-bit 量化；
    #   INTFP_W4A8   -> 权重用 int4，激活用 float8，对齐表里的 INT/FP W4A8。
    # mode 是旧参数，未显式设置 wmode/amode 时仍然作为默认数值格式。
    precision_profile: str = ""
    mode: str = "int"
    wmode: Optional[str] = None
    amode: Optional[str] = None
    wbit: int = 8
    abit: int = 8
    percent: int = 100
    sigma: float = 0.0
    disable_quant: bool = False
    disable_input_quantization: bool = False
    search: bool = False
    w_up: int = 150
    a_up: int = 150
    w_low: int = 75
    a_low: int = 75
    layer_8bit_n: int = 0
    layer_8bit_l: str = "base"
    quantize_batch_size: int = 64
    no_outlier: bool = False
    quantize: bool = True
    qllm: bool = False
    smooth: bool = False
    qft_freeze_alpha: bool = True
    qft_alpha_only: bool = False

    # -------------------- 其他训练控制 --------------------
    save_model: bool = False
    no_eval: bool = False
    save_on_interrupt: bool = False
    tag: str = ""
    verbose: bool = False
    untie_emb: bool = False
    linear_probing: bool = False
    lp_early_stopping: bool = False
    head_tuning: bool = False
    optim: str = "sgd"

    # 原 trainer_new.py 在每个 zo_lowbit step 更新后还会额外 forward 一次。
    # 这对 LoQZO/QZO 训练没有必要，默认关闭；如需复现旧代码可手动设 True。
    post_update_forward: bool = False

    # 训练后做 ReCoRD / 多选任务评估时，把候选项合批前向。
    # ReCoRD 每个样本可能有几十个 entity 候选，逐候选 forward 会非常慢。
    eval_candidate_batch_size: int = 16

    # qft 包装后是否只训练 LinearQuantizer 中的 weight/bias。
    # True 时会冻结原始 embedding / lm_head / norm 等非量化模块，避免它们在 LoQZO 中
    # 被错误地全空间扰动，这是大模型训练变慢的主要原因之一。
    qft_train_quantized_module_only: bool = True

    # -------------------- 日志 / wandb --------------------
    wandb_project: str = "LoQZO"
    use_wandb: bool = False
    wandb_api_key: Optional[str] = None

    # -------------------- LoQZO 参数（run.py 中也保留，便于共用入口） --------------------
    loqzo_enable: bool = False
    loqzo_rank: int = 8
    loqzo_rank_min: int = 2
    loqzo_rank_max: int = 64
    loqzo_rank_budget: int = 0
    loqzo_adaptive_rank: bool = False
    loqzo_rank_update_freq: int = 200
    loqzo_rank_ema: float = 0.9
    loqzo_basis_init: str = "random_orth"  # random_orth / svd_weight
    loqzo_target_modules: Optional[str] = None
    loqzo_include_embeddings: bool = False
    loqzo_fullspace_for_1d: bool = False
    loqzo_quantize_coeff: bool = True
    loqzo_coeff_bits: int = 0

    # LoQZO 加速开关：
    # - fast_lowrank_update=True 时，低秩方向使用 addmm_ 原地加到参数上，
    #   不再先显式构造完整 delta 矩阵；
    # - requantize_weight_every=0 时，更新后不再每步把 latent weight 重新量化一次，
    #   因为 qft forward 本身已经会按 WBIT/ABIT 做前向量化。
    #   若要严格复现“每步权重也压回 int 低比特”的旧行为，可设为 1。
    loqzo_fast_lowrank_update: bool = True
    loqzo_requantize_weight_every: int = 0


def parse_args() -> OurArguments:
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args


def normalize_precision_args(args: OurArguments) -> None:
    """
    统一解释实验表中的 Data Precision 配置。

    中文说明：
    - 旧代码只有 WBIT/ABIT/QMODE，因此只能表达“几比特 + 一个统一 codebook”；
    - QuZO 表格里的 FP / INT / INT-FP 实际还涉及 codebook 类型；
    - 这里把它拆成 wmode/amode：wmode 控制权重格式，amode 控制激活格式。

    支持的 profile：
    - FP_W32A32 / W32A32 / FP32：wbit=32, abit=32, wmode=float, amode=float；
    - FP_W8A8 / FP8 / FP_W8：wbit=8, abit=8, wmode=float, amode=float；
    - INT_W8A8 / INT8 / W8A8：wbit=8, abit=8, wmode=int, amode=int；
    - INTFP_W4A8 / INT_FP_W4A8 / W4A8：wbit=4, abit=8, wmode=int, amode=float。

    注意：FP_W32A32 不会生成 quant_weight.alpha，因此不能真正做 QZO-scale；
    交替训练器会在 QZO 阶段自动退回 LoQZO 权重更新。
    """
    raw_profile = str(getattr(args, "precision_profile", "") or "").strip()
    profile = raw_profile.lower().replace("-", "_").replace("/", "_").replace(" ", "")

    profile_map = {
        "fp_w32a32": (32, 32, "float", "float", "FP-W32A32"),
        "fp32": (32, 32, "float", "float", "FP-W32A32"),
        "w32a32": (32, 32, "float", "float", "FP-W32A32"),
        "fp_w8a8": (8, 8, "float", "float", "FP-W8A8"),
        "fp8": (8, 8, "float", "float", "FP-W8A8"),
        "float_w8a8": (8, 8, "float", "float", "FP-W8A8"),
        "int_w8a8": (8, 8, "int", "int", "INT-W8A8"),
        "int8": (8, 8, "int", "int", "INT-W8A8"),
        "w8a8": (8, 8, "int", "int", "INT-W8A8"),
        "intfp_w4a8": (4, 8, "int", "float", "INT/FP-W4A8"),
        "int_fp_w4a8": (4, 8, "int", "float", "INT/FP-W4A8"),
        "int_fp": (4, 8, "int", "float", "INT/FP-W4A8"),
        "w4a8": (4, 8, "int", "float", "INT/FP-W4A8"),
    }

    if profile:
        if profile not in profile_map:
            raise ValueError(
                f"未知 precision_profile={raw_profile}。可选："
                "FP_W32A32, FP_W8A8, INT_W8A8, INTFP_W4A8。"
            )
        wbit, abit, wmode, amode, label = profile_map[profile]
        args.wbit = int(wbit)
        args.abit = int(abit)
        args.wmode = wmode
        args.amode = amode
        args.mode = wmode if wmode == amode else "mixed"
        args.precision_profile = label
    else:
        # 没有 profile 时保持旧行为：wmode/amode 为空就继承 mode。
        args.wmode = getattr(args, "wmode", None) or getattr(args, "mode", "int")
        args.amode = getattr(args, "amode", None) or getattr(args, "mode", "int")
        if not getattr(args, "precision_profile", ""):
            if int(getattr(args, "wbit", 8)) > 8 and int(getattr(args, "abit", 8)) > 8:
                args.precision_profile = "FP-W32A32"
            elif args.wmode == "float" and args.amode == "float":
                args.precision_profile = f"FP-W{args.wbit}A{args.abit}"
            elif args.wmode == "int" and args.amode == "int":
                args.precision_profile = f"INT-W{args.wbit}A{args.abit}"
            else:
                args.precision_profile = f"{args.wmode.upper()}/{args.amode.upper()}-W{args.wbit}A{args.abit}"

    logger.info(
        "量化精度配置 | profile=%s | wbit=%s | abit=%s | wmode=%s | amode=%s | legacy_mode=%s",
        args.precision_profile,
        args.wbit,
        args.abit,
        args.wmode,
        args.amode,
        args.mode,
    )


def prepare_runtime_paths(args: OurArguments) -> None:
    args.project_root = str(Path(args.project_root).expanduser().resolve())
    args.data_root = str(Path(args.data_root).expanduser().resolve())
    args.models_root = str(Path(args.models_root).expanduser().resolve())
    args.base_models_root = str(Path(args.base_models_root).expanduser().resolve())
    args.quantized_manifests_root = str(Path(args.quantized_manifests_root).expanduser().resolve())
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 这些环境变量主要用于 tasks.py / 其他工具脚本读取路径
    os.environ["LOQZO_PROJECT_ROOT"] = args.project_root
    os.environ["LOQZO_DATA_ROOT"] = args.data_root
    os.environ["LOQZO_MODELS_ROOT"] = args.models_root
    os.environ["LOQZO_BASE_MODELS_ROOT"] = args.base_models_root
    os.environ["LOQZO_QUANTIZED_MANIFESTS_ROOT"] = args.quantized_manifests_root
    os.environ["DATA_ROOT"] = args.data_root
    os.environ["TASK_DATA_ROOT"] = args.data_root

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    os.chdir(args.project_root)


def configure_cuda_visible_devices(args: OurArguments) -> None:
    """统一处理单卡 / 多卡可见设备。"""
    if args.gpu_ids:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
        logger.info("按参数指定 CUDA_VISIBLE_DEVICES=%s", args.gpu_ids)
    elif args.gpu_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        logger.info("按参数指定 CUDA_VISIBLE_DEVICES=%s", args.gpu_id)
    else:
        logger.info("沿用外部环境中的 CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "<系统默认>"))


def detect_distributed_launch(args: OurArguments) -> None:
    """如果是 torchrun 启动，自动设置 local_rank / distributed。"""
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size > 1:
            args.distributed = True


def _load_local_model_index(models_root: Path, base_models_root: Path) -> Dict[str, str]:
    """尝试读取下载脚本生成的模型索引。"""
    candidates = [
        models_root / "requested_models_index.json",
        base_models_root / "requested_models_index.json",
        models_root / "model_index.json",
    ]
    index_map: Dict[str, str] = {}
    for fp in candidates:
        if not fp.exists():
            continue
        try:
            raw = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("读取模型索引失败: %s (%s)", fp, exc)
            continue
        if isinstance(raw, dict):
            if "models" in raw and isinstance(raw["models"], list):
                for item in raw["models"]:
                    if isinstance(item, dict):
                        alias = item.get("alias") or item.get("name") or item.get("model_name")
                        local_path = item.get("local_path") or item.get("path")
                        if alias and local_path:
                            index_map[str(alias)] = str(local_path)
            else:
                for k, v in raw.items():
                    if isinstance(v, str):
                        index_map[str(k)] = v
                    elif isinstance(v, dict):
                        local_path = v.get("local_path") or v.get("path")
                        if local_path:
                            index_map[str(k)] = str(local_path)
        if index_map:
            return index_map
    return index_map


def resolve_model_source(args: OurArguments) -> str:
    """解析模型来源。优先级：model_path > 精确本地目录 > HF repo。"""
    models_root = Path(args.models_root)
    base_models_root = Path(args.base_models_root)

    if args.model_path:
        mp = Path(args.model_path).expanduser().resolve()
        if not mp.exists():
            raise FileNotFoundError(f"你指定的 --model_path 不存在: {mp}")
        args.resolved_model_short_name = mp.name
        return str(mp)

    requested_name = str(args.model_name).strip()
    if Path(requested_name).expanduser().exists():
        mp = Path(requested_name).expanduser().resolve()
        args.resolved_model_short_name = mp.name
        return str(mp)

    registry_info = MODEL_ALIAS_REGISTRY.get(requested_name, {})
    hf_name = registry_info.get("hf", requested_name)

    # 如果别名里自带 4bit/8bit 提示，也同步打开加载开关
    quant_hint = registry_info.get("quant")
    if quant_hint == "8bit" and not args.load_int4:
        args.load_int8 = True
    if quant_hint == "4bit":
        args.load_int4 = True
        args.load_int8 = False

    if args.prefer_local_model:
        index_map = _load_local_model_index(models_root, base_models_root)
        # 先查索引
        direct_keys = [requested_name, hf_name, Path(hf_name).name] + _safe_local_name_patterns(hf_name)
        for k in direct_keys:
            if k in index_map:
                p = Path(index_map[k]).expanduser().resolve()
                if p.exists() and _looks_like_hf_model_dir(p):
                    logger.info("检测到本地模型目录，优先使用本地模型: %s", p)
                    args.resolved_model_short_name = p.name
                    return str(p)

        # 再做“精确目录名匹配”，注意：不做模糊匹配，避免 opt-1.3b 误匹配成 opt-13b
        exact_names = {_normalize_name(x) for x in (_safe_local_name_patterns(hf_name) + _safe_local_name_patterns(requested_name)) if x}
        for root in [base_models_root, models_root]:
            if not root.exists():
                continue
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                child_name_norm = _normalize_name(child.name)
                rel_name_norm = _normalize_name(str(child.relative_to(root)))
                if (child_name_norm in exact_names or rel_name_norm in exact_names) and _looks_like_hf_model_dir(child):
                    logger.info("检测到本地模型目录，优先使用本地模型: %s", child)
                    args.resolved_model_short_name = child.name
                    return str(child)

    args.resolved_model_short_name = Path(hf_name).name
    return hf_name


def build_task_from_args(args: OurArguments):
    """兼容不同版本 tasks.py：如果支持 data_root，就注入；否则回退。"""
    get_task_fn = getattr(tasks, "get_task")
    sig = inspect.signature(get_task_fn)
    task_kwargs = {}
    for key in ["data_root", "data_dir", "dataset_root", "root_dir", "base_dir"]:
        if key in sig.parameters:
            task_kwargs[key] = args.data_root
            break

    if task_kwargs:
        logger.info("get_task 支持路径参数，按 %s 注入数据目录", list(task_kwargs.keys())[0])
        task = get_task_fn(args.task_name, **task_kwargs)
    else:
        logger.info("get_task 未暴露数据目录参数，先按原方式构造 task")
        task = get_task_fn(args.task_name)

    for attr in ["data_root", "data_dir", "dataset_root", "root_dir", "base_dir"]:
        if hasattr(task, attr):
            try:
                setattr(task, attr, args.data_root)
            except Exception:
                pass
    return task


def _build_hf_common_kwargs(args: OurArguments) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if args.hf_token:
        kwargs["token"] = args.hf_token
    if args.local_files_only:
        kwargs["local_files_only"] = True
    return kwargs


def _is_family(config, model_source: str, keyword: str) -> bool:
    text = " ".join(
        [
            str(model_source).lower(),
            str(getattr(config, "model_type", "")).lower(),
            str(getattr(config, "_name_or_path", "")).lower(),
        ]
    )
    return keyword in text


class HFDataset(Dataset):
    """一个很薄的 dataset 包装器，让 list[dict] 能被 HF Trainer 读取。"""

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class Framework:
    """训练 / 推理 / 评估的一体化封装。"""

    def __init__(self, args: OurArguments, task) -> None:
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        """加载模型与 tokenizer，并根据 tuning_type 挂 LoRA / Prefix / Loretta。"""
        model_source = getattr(self.args, "resolved_model_name", self.args.model_name)
        hf_common_kwargs = _build_hf_common_kwargs(self.args)

        logger.info("原始模型请求: %s", self.args.model_name)
        logger.info("实际模型加载源: %s", model_source)

        with count_time("Loading model with FP%d" % (16 if self.args.load_float16 else 32)):
            config = AutoConfig.from_pretrained(model_source, **hf_common_kwargs)

            if self.args.untie_emb:
                logger.warning("Untie embeddings and LM head")
                config.tie_word_embeddings = False

            if self.args.head_tuning and _is_family(config, model_source, "opt"):
                from ht_opt import OPTForCausalLM

                model = OPTForCausalLM.from_pretrained(model_source, config=config, **hf_common_kwargs)
            else:
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16

                device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
                if self.args.no_auto_device:
                    device_map = None
                    max_memory = None
                else:
                    if getattr(self.args, "distributed", False):
                        # DDP 下每个进程只见到自己的 local device
                        device_map = {"": 0} if device_count > 0 else None
                        max_memory = None
                    else:
                        if device_count <= 1:
                            device_map = {"": 0} if device_count == 1 else None
                            max_memory = None
                        else:
                            # 单进程多卡 / 模型并行时交给 HF auto 切层
                            device_map = "auto"
                            free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
                            max_memory = {i: f"{max(free_in_GB - 5, 4)}GB" for i in range(device_count)}

                logger.info("device_map 设置为: %s", device_map)
                if max_memory is not None:
                    logger.info("max_memory 设置为: %s", max_memory)

                if self.args.load_int4:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_source,
                        config=config,
                        quantization_config=bnb_config,
                        device_map=device_map,
                        max_memory=max_memory,
                        **hf_common_kwargs,
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_source,
                        config=config,
                        device_map=device_map,
                        max_memory=max_memory,
                        torch_dtype=torch_dtype,
                        load_in_8bit=self.args.load_int8,
                        **hf_common_kwargs,
                    )

            model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=False, **hf_common_kwargs)

        if _is_family(config, model_source, "opt"):
            tokenizer.bos_token_id = 0
        if _is_family(config, model_source, "llama"):
            tokenizer.pad_token_id = 0
        if _is_family(config, model_source, "mistral"):
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(tokenizer))

        if self.args.prefix_tuning or self.args.tuning_type == "prefix":
            from prefix import PrefixTuning

            PrefixTuning(
                model,
                num_prefix=self.args.num_prefix,
                reparam=not self.args.no_reparam,
                float16=self.args.load_float16,
                init_by_real_act=self.args.prefix_init_by_real_act,
            )

        if self.args.tuning_type == "lora":
            from peft import LoraConfig, get_peft_model

            peft_config = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)

        if self.args.tuning_type == "loretta_rep":
            from loretta import LorettaRepConfig, get_peft_model
            from transformers.models.llama.modeling_llama import LlamaRMSNorm

            loretta_config = LorettaRepConfig(
                r=self.args.rep_bottleneck,
                lora_alpha=self.args.rep_alpha,
                target_modules=self.args.target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type=self.args.task_type,
                tensor_rank=self.args.tensor_rank,
            )
            model = get_peft_model(model, loretta_config)

            for _, sub_module in model.named_modules():
                if isinstance(sub_module, LlamaRMSNorm):
                    for param_name, param in sub_module.named_parameters():
                        if any(k in param_name for k in ["lm_head", "layernorm", "post_attention_layernorm"]):
                            param.requires_grad = True

        if self.args.head_tuning:
            if _is_family(config, model_source, "opt"):
                head_name = "lm_head" if self.args.untie_emb else "embed_tokens"
            else:
                raise NotImplementedError("当前只对 OPT 家族显式支持 head_tuning")
            for n, p in model.named_parameters():
                if head_name not in n:
                    p.requires_grad = False
                else:
                    logger.info("Only tuning %s", n)

        return model, tokenizer

    def _make_eval_demos(self) -> List[Any]:
        """训练后评估用的 few-shot demos。"""
        if not self.args.use_eval_demos_after_training or self.args.eval_num_demos <= 0:
            return []
        try:
            demos = self.task.sample_subset(
                data_split="train",
                seed=self.args.eval_demo_seed,
                num=self.args.eval_num_demos,
            )
            logger.info("训练后评估将使用 %d 个 demonstration（seed=%d）", len(demos), self.args.eval_demo_seed)
            return demos
        except Exception as exc:
            logger.warning("构造评估 demonstrations 失败，退回零 demo 评估：%s", exc)
            return []

    def forward(self, input_ids, option_len=None, generation=False):
        """推理：分类/多选返回 option log-prob，生成任务返回文本。"""
        input_ids = torch.tensor([input_ids]).to(self.model.device)
        if generation:
            args = self.args
            outputs = self.model.generate(
                input_ids,
                do_sample=args.sampling,
                temperature=args.temperature,
                num_beams=args.num_beams,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)),
                num_return_sequences=1,
                eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
            )
            return self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
        return self.forward_candidates_batch([input_ids], [option_len])[0]

    def forward_candidates_batch(self, encoded_candidates, option_lens):
        """
        合批计算多个候选答案的 option-token log-prob。

        中文说明：
        原来的评估逻辑对每个 candidate 单独 forward。ReCoRD 一个样本常有几十个
        entity 候选，1000 个 eval 样本会变成数万次 forward，非常慢。这里把同一
        样本的候选项按 eval_candidate_batch_size 合批，结果与逐候选 forward 等价，
        但能显著减少 GPU kernel 启动和模型前向次数。
        """
        if len(encoded_candidates) == 0:
            return []

        batch_size = max(1, int(getattr(self.args, "eval_candidate_batch_size", 16) or 16))
        outputs = []
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        self.model.eval()
        # 和训练阶段保持一致，候选答案始终位于序列末尾；这样下面取 -option_len 才可靠。
        old_padding_side = getattr(self.tokenizer, "padding_side", "left")
        self.tokenizer.padding_side = "left"
        for start in range(0, len(encoded_candidates), batch_size):
            chunk_ids = encoded_candidates[start:start + batch_size]
            chunk_lens = option_lens[start:start + batch_size]
            features = [{"input_ids": ids} for ids in chunk_ids]
            batch = self.tokenizer.pad(
                features,
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            batch = {k: v.to(self.model.device) for k, v in batch.items()}

            with torch.inference_mode():
                logits = self.model(**batch).logits

            input_ids = batch["input_ids"]
            shift_labels = input_ids[:, 1:].contiguous()
            shift_logits = logits[:, :-1, :].contiguous()
            log_probs = F.log_softmax(shift_logits, dim=-1)

            safe_labels = shift_labels.clone()
            safe_labels[safe_labels == pad_id] = 0
            selected = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

            for row_id, opt_len in enumerate(chunk_lens):
                opt_len = int(opt_len) if opt_len is not None else 0
                if opt_len <= 0:
                    outputs.append(selected[row_id, :0].cpu().detach())
                else:
                    outputs.append(selected[row_id, -opt_len:].cpu().detach())

        self.tokenizer.padding_side = old_padding_side
        return outputs

    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        verbose = verbose or self.args.verbose
        if verbose:
            logger.info("========= Example =========")
            logger.info("Candidate: %s", eval_sample.candidates)
            logger.info("Correct candidate: %s", eval_sample.correct_candidate)

        encoded_candidates, option_lens = encode_prompt(
            self.task,
            self.task.get_template(),
            train_samples,
            eval_sample,
            self.tokenizer,
            max_length=self.args.max_length,
            generation=self.task.generation,
            max_new_tokens=self.args.max_new_tokens,
        )

        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(
                self.task,
                self.task.get_template(),
                train_samples,
                eval_sample,
                self.tokenizer,
                max_length=self.args.max_length,
                sfc=self.args.sfc,
                icl_sfc=self.args.icl_sfc,
                generation=self.task.generation,
                max_new_tokens=self.args.max_new_tokens,
            )

        outputs = []
        if self.task.generation:
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info("Output: %s", output_text)
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)

        batch_selected_log_probs = self.forward_candidates_batch(encoded_candidates, option_lens)
        if self.args.sfc or self.args.icl_sfc:
            batch_sfc_selected_log_probs = self.forward_candidates_batch(sfc_encoded_candidates, sfc_option_lens)
        else:
            batch_sfc_selected_log_probs = [None] * len(batch_selected_log_probs)

        for candidate_id, encoded_candidate in enumerate(encoded_candidates):
            selected_log_probs = batch_selected_log_probs[candidate_id]
            if verbose:
                logger.info("=== Candidate %d ===", candidate_id)
                logger.info(self.tokenizer.decode(encoded_candidate))
                logger.info("Log probabilities of the option tokens: %s", selected_log_probs)

            sfc_selected_log_probs = batch_sfc_selected_log_probs[candidate_id]
            outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs})

        if self.args.sfc or self.args.icl_sfc:
            scores = [x["log_probs"].sum().item() - x["sfc_log_probs"].sum().item() for x in outputs]
        else:
            scores = [x["log_probs"].mean().item() for x in outputs]

        if isinstance(eval_sample.correct_candidate, list):
            correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
        else:
            correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

        return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))

    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        if one_train_set_per_eval_sample:
            logger.info("共有 %d 个评估样本，并且每个样本有自己的一组 demonstrations", len(eval_samples))
        else:
            logger.info("共有 %d 个评估样本；评估 demonstrations 数量=%d", len(eval_samples), len(train_samples))

        predictions = []
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(
                self.one_step_pred(
                    train_samples[eval_id] if one_train_set_per_eval_sample else train_samples,
                    eval_sample,
                    verbose=(eval_id < 3),
                )
            )

        metric_name = getattr(self.task, "metric_name", "accuracy")
        return {metric_name: calculate_metric(predictions, metric_name)}

    def _convert_samples(self, samples: Sequence[Any]) -> List[Dict[str, Any]]:
        data = []
        for sample in samples:
            encoded_candidates, option_lens = encode_prompt(
                self.task,
                self.task.get_template(),
                [],
                sample,
                self.tokenizer,
                max_length=self.args.max_length,
                generation=self.task.generation,
                generation_with_gold=True,
                max_new_tokens=self.args.max_new_tokens,
            )
            if self.task.generation:
                correct_candidate_id = 0
            elif isinstance(sample.correct_candidate, list):
                correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
            else:
                correct_candidate_id = sample.candidates.index(sample.correct_candidate)

            if self.args.non_diff:
                encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]

            if self.args.train_as_classification:
                data.append(
                    [
                        {
                            "input_ids": encoded_candidates[i],
                            "labels": correct_candidate_id,
                            "option_len": option_lens[i],
                            "num_options": len(sample.candidates),
                        }
                        for i in range(len(encoded_candidates))
                    ]
                )
            elif self.args.only_train_option:
                item = {
                    "input_ids": encoded_candidates[correct_candidate_id],
                    "labels": encoded_candidates[correct_candidate_id],
                    "option_len": option_lens[correct_candidate_id],
                }
                if self.args.non_diff:
                    item["gold"] = sample.correct_candidate
                data.append(item)
            else:
                data.append({
                    "input_ids": encoded_candidates[correct_candidate_id],
                    "labels": encoded_candidates[correct_candidate_id],
                })
        return data

    def train(self, train_samples, eval_or_dev_samples):
        """启动训练。"""
        self.tokenizer.padding_side = "left"

        # 防御性检查：CB 等小数据集在 num_dev 过大时容易被切成空训练集。
        # 如果这里为空，继续进入 transformers.RandomSampler 只会得到更难定位的 num_samples=0。
        if train_samples is None or len(train_samples) == 0:
            raise ValueError("训练样本为空：请检查 --num_train/--num_dev，或把 DEV 设为 0/更小值。")
        if eval_or_dev_samples is None or len(eval_or_dev_samples) == 0:
            logger.warning("训练期评估样本为空；将构造空 eval_dataset，建议检查 --num_eval 或验证集 split。")
            eval_or_dev_samples = []

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(self._convert_samples(train_samples))
            eval_dataset = HFDataset(self._convert_samples(eval_or_dev_samples))

        if self.args.only_train_option and not self.args.non_diff:
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        collator_cls = NondiffCollator if self.args.non_diff else DataCollatorForTokenClassification

        if self.args.tuning_type == "qft":
            from quant_func.quant_model import enable_quantization, quantize_model
            from quant_func.quant_modules import LinearQuantizer
            from quant_func.quant_utils import set_quantizer

            set_quantizer(self.args)
            self.model = quantize_model(self.model)
            enable_quantization(self.model)

            # 中文说明：
            # 旧代码在 qft_freeze_alpha=True 时把所有非 alpha 参数都设为可训练，
            # 这会把 embed_tokens / lm_head / norm 等未被量化包装的参数也纳入 LoQZO。
            # 这些大矩阵随后会走全空间扰动，LLaMA-2 7B 上会极大拖慢训练。
            # 这里默认只训练 LinearQuantizer 内部的 weight/bias；QZO 的 alpha 虽然
            # requires_grad=False，但仍会在 trainer_alternating.py 中被手动零阶更新。
            quantized_param_ids = set()
            quantized_linear_count = 0
            if bool(getattr(self.args, "qft_train_quantized_module_only", True)):
                for module in self.model.modules():
                    if isinstance(module, LinearQuantizer):
                        quantized_linear_count += 1
                        quantized_param_ids.add(id(module.weight))
                        if getattr(module, "bias", None) is not None:
                            quantized_param_ids.add(id(module.bias))

            for name, param in self.model.named_parameters():
                is_alpha = ("alpha" in name)
                in_quantized_module = id(param) in quantized_param_ids
                if self.args.qft_alpha_only:
                    param.requires_grad = is_alpha
                elif self.args.qft_freeze_alpha:
                    if bool(getattr(self.args, "qft_train_quantized_module_only", True)):
                        param.requires_grad = in_quantized_module
                    else:
                        param.requires_grad = not is_alpha
                else:
                    if bool(getattr(self.args, "qft_train_quantized_module_only", True)):
                        param.requires_grad = in_quantized_module or is_alpha
                    else:
                        param.requires_grad = True

            if bool(getattr(self.args, "qft_train_quantized_module_only", True)):
                logger.info("qft 训练范围：仅 LinearQuantizer weight/bias；量化线性层数量=%d", quantized_linear_count)
            else:
                logger.info("qft 训练范围：旧逻辑，所有非 alpha 参数均可训练。")

        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_num = sum(p.numel() for p in self.model.parameters())
        logger.info("参数摘要 | total=%s | trainable=%s | trainable%%=%.4f", f"{total_num:,}", f"{trainable_num:,}", 100.0 * trainable_num / max(total_num, 1))

        trainer_module_name = self.args.trainer_module or "trainer_new"
        try:
            trainer_module = importlib.import_module(trainer_module_name)
        except Exception:
            trainer_module = importlib.import_module(f"Code.train.{trainer_module_name}")
        OurTrainer = getattr(trainer_module, "OurTrainer")

        trainer = OurTrainer(
            model=self.model,
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=(
                DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8)
                if self.args.train_as_classification
                else collator_cls(self.tokenizer, pad_to_multiple_of=8)
            ),
        )
        trainer.add_callback(EpochProgressCallback())

        # 尽量给 trainer 挂上这些对象，便于自定义 trainer 在内部拿到 task / 评估函数
        trainer.task = self.task
        trainer.framework = self
        trainer.evaluate_func = self.evaluate

        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())

        last_checkpoint = None
        if self.args.resume_from_checkpoint:
            last_checkpoint = self.args.resume_from_checkpoint
        elif os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
            if last_checkpoint is not None:
                logger.info("检测到 checkpoint，自动续训: %s", last_checkpoint)

        trainer.train(resume_from_checkpoint=last_checkpoint)

        if self.args.save_model:
            logger.warning("显式保存最终模型")
            trainer.save_model()

        self.model = trainer.model

        if self.args.only_train_option and not self.args.non_diff:
            if hasattr(self.model, "original_forward"):
                self.model.forward = self.model.original_forward
            elif hasattr(self.model, "_fsdp_wrapped_module") and hasattr(self.model._fsdp_wrapped_module, "original_forward"):
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward


def split_sampled_train_and_dev(sampled_train_set: Sequence[Any], requested_num_dev: Optional[int], eval_samples: Sequence[Any]):
    """安全地把从 train split 采样出来的样本切成 train/dev。

    原始代码直接使用：
      dev_samples = sampled_train_set[-num_dev:]
      train_samples = sampled_train_set[:-num_dev]

    当 CB 这种小数据集只有 250 条训练样本、但脚本默认 num_dev=500 时，
    上面的切法会得到空训练集，随后 RandomSampler 报
    ``num_samples=0``。这里的策略是：
      1) num_dev 为空或 <=0：不从训练集切 dev，直接用 validation 作为训练期评估集；
      2) num_dev >= 已采样训练样本数：保留全部样本用于训练，同样用 validation 作为训练期评估集；
      3) 只有在样本足够时，才从训练集尾部切出 dev。

    这样不会静默产生空训练集，也不会在小数据集上因为默认 DEV 太大而崩溃。
    """
    sampled_train_set = list(sampled_train_set)
    eval_samples = list(eval_samples)

    if requested_num_dev is None or int(requested_num_dev) <= 0:
        logger.info("未从 train split 中切分 dev；训练期评估使用 validation/eval split。")
        return sampled_train_set, None, eval_samples

    requested_num_dev = int(requested_num_dev)
    if len(sampled_train_set) == 0:
        raise ValueError("训练样本为空：请检查 --num_train、数据集路径和 split 是否正确。")

    if requested_num_dev >= len(sampled_train_set):
        logger.warning(
            "请求 num_dev=%d，但当前采样训练样本只有 %d 条；为避免 train_dataset 为空，"
            "本次不从 train split 切 dev，改用 validation/eval split 作为训练期评估集。",
            requested_num_dev,
            len(sampled_train_set),
        )
        return sampled_train_set, None, eval_samples

    dev_samples = sampled_train_set[-requested_num_dev:]
    train_samples = sampled_train_set[:-requested_num_dev]
    if len(train_samples) == 0:
        raise ValueError(
            f"切分后训练样本为空：len(sampled_train_set)={len(sampled_train_set)}, "
            f"num_dev={requested_num_dev}。请减小 --num_dev 或增大 --num_train。"
        )
    logger.info("训练/dev 切分完成：train=%d | dev=%d", len(train_samples), len(dev_samples))
    return train_samples, dev_samples, dev_samples


def result_file_tag(args: OurArguments) -> str:
    save_model_name = str(args.model_name).split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = f"-sampleeval{args.num_eval}" if args.num_eval is not None else ""
    sample_train_tag = f"-ntrain{args.num_train}" if args.num_train > 0 else ""
    sample_dev_tag = f"-ndev{args.num_dev}" if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}{sfc_tag}{icl_sfc_tag}{sample_eval_tag}{sample_train_tag}{sample_dev_tag}{customized_tag}"


def main(default_overrides: Optional[Dict[str, Any]] = None) -> None:
    args = parse_args()
    args.task_name = normalize_task_name(args.task_name)

    # 允许 run_loqzo.py 传默认覆盖值进来
    if default_overrides:
        for key, value in default_overrides.items():
            current = getattr(args, key, None)
            # 只有在“当前值等于 dataclass 默认值/空值”时才覆盖，避免吞掉用户显式传参
            if current in [None, False, "", 0, "regular"]:
                setattr(args, key, value)

    detect_distributed_launch(args)
    configure_cuda_visible_devices(args)
    prepare_runtime_paths(args)
    setup_logging(args.output_dir, log_filename=os.environ.get("LOQZO_LOG_FILENAME", "train.log"))
    configure_reporting_and_wandb(args)
    set_seed_all(args.seed)
    normalize_precision_args(args)

    # trainer 别名统一
    alias_map = {
        "quzo": "zo_lowbit",
        "quzo_ft": "zo_lowbit_ft",
        "mezo": "zo",
        "fo": "regular",
        "loqzo": "zo_lowbit",
        "loqzo_ft": "zo_lowbit_ft",
    }
    args.trainer = alias_map.get(args.trainer, args.trainer)

    if args.trainer in {"zo_lowbit", "zo_lowbit_ft"} and not getattr(args, "loqzo_enable", False):
        args.loqzo_enable = True

    if args.trainer_module is None:
        args.trainer_module = "trainer_loqzo"

    # run_loqzo.py 作为独立入口时，默认不开启本地模糊优先，避免把 opt-1.3b 误匹配到 opt-13b
    if getattr(args, "prefer_local_model", None) is None:
        args.prefer_local_model = False

    args.resolved_model_name = resolve_model_source(args)

    logger.info("当前可见 GPU 数量: %s", torch.cuda.device_count())
    if torch.cuda.is_available():
        logger.info("当前默认 CUDA 设备: cuda:%s (%s)", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
    logger.info(
        "运行参数 | task=%s | model=%s | trainer=%s | tuning_type=%s | bs=%s | grad_acc=%s | lr=%s | max_steps=%s | save_steps=%s | logging_steps=%s | no_eval=%s",
        args.task_name,
        args.resolved_model_name,
        args.trainer,
        args.tuning_type,
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        args.learning_rate,
        args.max_steps,
        args.save_steps,
        args.logging_steps,
        args.no_eval,
    )

    task = build_task_from_args(args)
    train_sets = task.sample_train_sets(
        num_train=args.num_train,
        num_dev=args.num_dev,
        num_eval=args.num_eval,
        num_train_sets=args.num_train_sets,
        seed=args.train_set_seed,
    )

    framework = Framework(args, task)

    if args.train_set_seed is not None or args.num_train_sets is not None:
        for train_set_id, sampled_train_set in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            if args.num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
            else:
                eval_samples = task.valid_samples

            if args.trainer != "none":
                train_samples, dev_samples, train_eval_samples = split_sampled_train_and_dev(
                    sampled_train_set,
                    args.num_dev,
                    eval_samples,
                )

                framework.train(train_samples, train_eval_samples)

                if not args.no_eval:
                    eval_demos = framework._make_eval_demos()
                    metrics = framework.evaluate(eval_demos, eval_samples)
                    if dev_samples is not None:
                        dev_metrics = framework.evaluate(eval_demos, dev_samples)
                        for m, v in dev_metrics.items():
                            metrics[f"dev_{m}"] = v
                else:
                    metrics = None
            else:
                assert args.num_dev is None
                metrics = framework.evaluate(sampled_train_set, eval_samples)

            if metrics is not None:
                logger.info("===== Train set %d 最终结果 =====", train_set_seed)
                logger.info(metrics)
                if args.result_file and args.local_rank <= 0:
                    write_metrics_to_file(metrics, args.result_file)
                elif args.write_result_json and args.local_rank <= 0:
                    result_path = os.path.join(args.output_dir, result_file_tag(args) + f"-trainset{train_set_id}.json")
                    write_metrics_to_file(metrics, result_path)
    else:
        assert args.trainer == "none"
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = task.valid_samples
        metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
        logger.info(metrics)
        if args.result_file and args.local_rank <= 0:
            write_metrics_to_file(metrics, args.result_file)
        elif args.write_result_json and args.local_rank <= 0:
            result_path = os.path.join(args.output_dir, result_file_tag(args) + "-onetrainpereval.json")
            write_metrics_to_file(metrics, result_path)


if __name__ == "__main__":
    main()


# =========================
# 运行示例（单行）
# =========================
# 1) QuZO：python Code/train/run.py --trainer zo_lowbit --trainer_module trainer_new --model_name facebook/opt-1.3b --task_name SST2 --num_train 1000 --num_dev 500 --num_eval 1000 --max_steps 10000 --save_steps 500 --logging_steps 20 --output_dir ./outputs/demo_quzo_sst2 --learning_rate 1e-5 --zo_eps 1e-3 --per_device_train_batch_size 16 --max_length 512 --use_eval_demos_after_training True --eval_num_demos 32
# 2) MeZO：python Code/train/run.py --trainer zo --trainer_module trainer_new --model_name facebook/opt-1.3b --task_name MultiRC --num_train 1000 --num_dev 500 --num_eval 1000 --max_steps 5000 --save_steps 500 --logging_steps 20 --output_dir ./outputs/demo_mezo_multirc --learning_rate 1e-4 --zo_eps 1e-3 --per_device_train_batch_size 8 --max_length 512 --use_eval_demos_after_training True --eval_num_demos 32
# 3) FO：python Code/train/run.py --trainer regular --trainer_module trainer_new --model_name facebook/opt-1.3b --task_name SST2 --num_train 1000 --num_dev 500 --num_eval 1000 --max_steps 5000 --save_steps 500 --logging_steps 20 --output_dir ./outputs/demo_fo_sst2 --learning_rate 1e-5 --per_device_train_batch_size 8 --max_length 512 --load_bfloat16 True
# 4) LoRA：python Code/train/run.py --trainer zo_lowbit --trainer_module trainer_new --tuning_type lora --model_name facebook/opt-1.3b --task_name MultiRC --num_train 1000 --num_dev 500 --num_eval 1000 --max_steps 5000 --save_steps 500 --logging_steps 20 --output_dir ./outputs/demo_quzo_lora_multirc --learning_rate 1e-4 --zo_eps 1e-3 --per_device_train_batch_size 4 --max_length 512
# 5) QFT：python Code/train/run.py --trainer regular --trainer_module trainer_new --tuning_type qft --model_name facebook/opt-1.3b --task_name MultiRC --num_train 1000 --num_dev 500 --num_eval 1000 --max_steps 2000 --save_steps 500 --logging_steps 20 --output_dir ./outputs/demo_qft_multirc --learning_rate 1e-5 --per_device_train_batch_size 1 --max_length 256 --mode int --wbit 4 --abit 8
