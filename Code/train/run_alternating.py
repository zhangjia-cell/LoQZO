# -*- coding: utf-8 -*-
# =========================
# run_alternating.py — LoQZO + QZO(scale) 交替优化入口
#
# 说明：
# 1) LoQZO 阶段：固定 scale / zero point，只更新权重；
# 2) QZO 阶段：固定权重 / zero point，只更新量化 scale(alpha)；
# 3) 必须使用 tuning_type=qft，否则模型中没有可更新的 quant_weight.alpha；
# 4) 复用 run_loqzo.py 的模型加载、数据构建、评估和日志基础设施。
# =========================

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import HfArgumentParser

# =========================
# 路径初始化
# =========================
CURRENT_FILE = Path(__file__).resolve()
TRAIN_DIR = CURRENT_FILE.parent
CODE_ROOT = TRAIN_DIR.parent
PROJECT_ROOT = CODE_ROOT.parent

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# quant_cuda 是 C++/CUDA 扩展，通常位于 Code/quant/build/lib.* 目录。
# 这里自动加入 sys.path，避免只在 shell 脚本里配置 PYTHONPATH 时才可运行。
QUANT_BUILD_ROOT = CODE_ROOT / "quant" / "build"
if QUANT_BUILD_ROOT.exists():
    for _p in sorted(QUANT_BUILD_ROOT.glob("lib.*")):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
if str(CODE_ROOT / "quant") not in sys.path:
    sys.path.insert(0, str(CODE_ROOT / "quant"))

from run_loqzo import (  # noqa: E402
    OurArguments as _BaseOurArguments,
    Framework,
    build_task_from_args,
    configure_cuda_visible_devices,
    configure_reporting_and_wandb,
    detect_distributed_launch,
    normalize_task_name,
    prepare_runtime_paths,
    resolve_model_source,
    result_file_tag,
    set_seed_all,
    setup_logging,
    split_sampled_train_and_dev,
)
from utils import write_metrics_to_file  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class OurArguments(_BaseOurArguments):
    """在 run_loqzo.OurArguments 基础上增加交替优化和 QZO-scale 参数。"""

    # -------------------- 交替优化参数 --------------------
    alt_a_steps: int = 1              # 每个 cycle 中 LoQZO 阶段步数
    alt_b_steps: int = 1              # 每个 cycle 中 QZO-scale 阶段步数
    alt_start: int = 0                # 该 step 前只跑 LoQZO warmup

    # -------------------- QZO-scale 参数 --------------------
    qzo_eps: float = 0.0              # QZO scale 扰动半径；<=0 时复用 zo_eps
    qzo_scale_lr_mult: float = 1.0    # QZO scale 学习率倍率：lr_scale = learning_rate * mult
    qzo_scale_min: float = 1e-8       # scale 下界，防止 alpha 非正
    qzo_scale_max: float = 0.0        # scale 绝对上界；<=0 时使用初始 alpha 的相对上界
    qzo_scale_max_mult: float = 10.0  # scale 相对上界倍率：alpha <= init_alpha * mult
    qzo_scale_scope: str = "weight"   # weight / activation / all
    qzo_layerwise_scale_perturb: bool = False  # True: 每个 alpha 张量共享一个随机标量
    qzo_require_qft: bool = True      # True: 自动强制 tuning_type=qft

    # -------------------- DDC / 方向导数裁剪 --------------------
    clip_zo_grad: bool = True         # 是否裁剪零阶方向导数，默认开启以避免训练后期发散
    qzo_clip_threshold: float = 100.0 # 裁剪阈值，方向导数限制在 [-threshold, threshold]


def parse_args() -> OurArguments:
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args


def _normalize_alternating_args(args: OurArguments) -> None:
    """统一修正交替优化需要的参数，避免脚本遗漏导致算法语义错误。"""
    # 1) 训练器别名归一化。
    alias_map = {
        "quzo": "zo_lowbit",
        "quzo_ft": "zo_lowbit_ft",
        "mezo": "zo",
        "fo": "regular",
        "loqzo": "zo_lowbit",
        "loqzo_ft": "zo_lowbit_ft",
        "alternating": "zo_lowbit",
    }
    args.trainer = alias_map.get(args.trainer, args.trainer)

    # 2) 交替训练必须启用 LoQZO 和专用 trainer。
    args.loqzo_enable = True
    if args.trainer_module is None or args.trainer_module == "trainer_loqzo":
        args.trainer_module = "trainer_alternating"

    # 3) QZO-scale 需要 qft 量化包装，否则模型里没有 alpha。
    if bool(getattr(args, "qzo_require_qft", True)) and args.tuning_type != "qft":
        logger.warning(
            "LoQZO+QZO 交替优化需要 tuning_type=qft 才能更新 scale；已自动从 %s 改为 qft。",
            args.tuning_type,
        )
        args.tuning_type = "qft"

    # 4) 固定 LoQZO 阶段的 scale，只让 LoQZO 看到权重参数；
    #    QZO 阶段会手动更新 alpha，所以 alpha 不需要 requires_grad=True。
    if args.tuning_type == "qft":
        if bool(getattr(args, "qft_alpha_only", False)):
            logger.warning("qft_alpha_only=True 会导致 LoQZO 阶段没有权重可更新；已自动改为 False。")
        args.qft_alpha_only = False
        args.qft_freeze_alpha = True

    # 5) 交替步数合法化。
    args.alt_a_steps = max(0, int(args.alt_a_steps))
    args.alt_b_steps = max(0, int(args.alt_b_steps))
    args.alt_start = max(0, int(args.alt_start))
    if args.alt_a_steps == 0 and args.alt_b_steps == 0:
        logger.warning("alt_a_steps 和 alt_b_steps 不能同时为 0；已回退为 LoQZO:QZO=1:1。")
        args.alt_a_steps = 1
        args.alt_b_steps = 1

    # 6) 默认不开启本地模糊优先，避免 OPT-1.3B 被误匹配到 OPT-13B。
    if getattr(args, "prefer_local_model", None) is None:
        args.prefer_local_model = False


def main(default_overrides: Optional[Dict[str, Any]] = None) -> None:
    args = parse_args()
    args.task_name = normalize_task_name(args.task_name)

    # 外部调用 main(default_overrides=...) 时可提供默认覆盖值。
    if default_overrides:
        for key, value in default_overrides.items():
            current = getattr(args, key, None)
            if current in [None, False, "", 0, "regular"]:
                setattr(args, key, value)

    detect_distributed_launch(args)
    configure_cuda_visible_devices(args)
    prepare_runtime_paths(args)
    setup_logging(
        args.output_dir,
        log_filename=os.environ.get("LOQZO_LOG_FILENAME", "train_alternating.log"),
    )
    configure_reporting_and_wandb(args)
    set_seed_all(args.seed)

    _normalize_alternating_args(args)
    args.resolved_model_name = resolve_model_source(args)

    alt_cycle = max(1, args.alt_a_steps + args.alt_b_steps)
    logger.info("========== LoQZO + QZO(scale) 交替优化配置 ==========")
    logger.info("  A-steps / LoQZO 权重低秩子空间更新 = %d", args.alt_a_steps)
    logger.info("  B-steps / QZO scale 更新 = %d", args.alt_b_steps)
    logger.info("  cycle = %d | alt_start = %d", alt_cycle, args.alt_start)
    logger.info("  trainer = %s | trainer_module = %s | tuning_type = %s", args.trainer, args.trainer_module, args.tuning_type)
    logger.info("  wbit = %s | abit = %s | mode = %s", args.wbit, args.abit, args.mode)
    logger.info("  loqzo_rank = %s | adaptive_rank = %s | basis_init = %s", args.loqzo_rank, args.loqzo_adaptive_rank, args.loqzo_basis_init)
    logger.info("  qzo_eps = %s | qzo_scale_lr_mult = %s | qzo_scope = %s", args.qzo_eps or args.zo_eps, args.qzo_scale_lr_mult, args.qzo_scale_scope)
    logger.info("  qzo_scale_min = %s | qzo_scale_max = %s | qzo_scale_max_mult = %s", args.qzo_scale_min, args.qzo_scale_max, args.qzo_scale_max_mult)
    logger.info("  clip_zo_grad = %s | qzo_clip_threshold = %s", args.clip_zo_grad, args.qzo_clip_threshold)
    logger.info("====================================================")

    logger.info("当前可见 GPU 数量: %s", torch.cuda.device_count())
    if torch.cuda.is_available():
        logger.info(
            "当前默认 CUDA 设备: cuda:%s (%s)",
            torch.cuda.current_device(),
            torch.cuda.get_device_name(torch.cuda.current_device()),
        )

    logger.info(
        "运行参数 | task=%s | model=%s | bs=%s | grad_acc=%s | lr=%s | zo_eps=%s | max_steps=%s | no_eval=%s",
        args.task_name,
        args.resolved_model_name,
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        args.learning_rate,
        args.zo_eps,
        args.max_steps,
        args.no_eval,
    )

    # ========== 构建 task 和 data ==========
    task = build_task_from_args(args)
    train_sets = task.sample_train_sets(
        num_train=args.num_train,
        num_dev=args.num_dev,
        num_eval=args.num_eval,
        num_train_sets=args.num_train_sets,
        seed=args.train_set_seed,
    )

    framework = Framework(args, task)

    # ========== 训练 + 评估循环 ==========
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

                logger.info(
                    "===== Train set %d 开始交替训练 (LoQZO x%d + QZO-scale x%d) =====",
                    train_set_seed,
                    args.alt_a_steps,
                    args.alt_b_steps,
                )
                # 注意：这里只训练一次。原版本重复调用 framework.train 两次，会导致 max_steps 实际翻倍。
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
                    result_path = os.path.join(
                        args.output_dir,
                        result_file_tag(args) + f"-trainset{train_set_id}.json",
                    )
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
            result_path = os.path.join(
                args.output_dir,
                result_file_tag(args) + "-onetrainpereval.json",
            )
            write_metrics_to_file(metrics, result_path)


if __name__ == "__main__":
    main()
