# ============================================================
# tasks_refactored_cn.py
# ------------------------------------------------------------
# 这是根据你当前项目文件树重构后的 tasks.py。
#
# 你的当前目录结构核心如下：
#   LoQZO/
#   ├── Code/
#   │   └── train/
#   │       ├── run.py
#   │       └── tasks.py
#   ├── Data/
#   │   ├── sst2/
#   │   ├── boolq/
#   │   ├── cb/
#   │   ├── copa/
#   │   ├── multirc/
#   │   ├── record/
#   │   ├── rte/
#   │   ├── squad/
#   │   ├── drop/
#   │   ├── wic/
#   │   └── wsc/
#   └── Models/
#
# 因此这份代码的主要目标是：
#   1. 优先从项目根目录下的 Data/ 中读取本地数据集；
#   2. 如果 Data/ 中不存在对应数据集，则回退到 HuggingFace 下载；
#   3. 下载后自动保存回 Data/，方便下次直接本地加载；
#   4. 保持原来 tasks.py 对 run.py / trainer 的接口基本不变；
#   5. 为后续的 QuZO / MeZO / LoRA 实验提供更稳定的数据入口。
#
# 你可以直接把这份文件替换到：
#   Code/train/tasks.py
# ============================================================

from templates import *
from utils import temp_seed

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from datasets import load_dataset, load_from_disk

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================
# 路径推断与本地数据目录管理
# ------------------------------------------------------------
# 这里的逻辑是：
#   - 优先使用显式传入的 data_root
#   - 其次使用环境变量 LOQZO_DATA_ROOT / DATA_ROOT / TASK_DATA_ROOT
#   - 最后自动从 tasks.py 所在位置向上推断项目根目录，并拼出 Data/
# ============================================================

def _guess_project_root() -> Path:
    """尽量从当前文件位置向上推断 LoQZO 项目根目录。"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "Data").exists() or (parent / "Models").exists():
            return parent
    # 如果没找到，就使用比较保守的默认值
    if len(current.parents) >= 3:
        return current.parents[2]
    return current.parent


PROJECT_ROOT = Path(os.environ.get("LOQZO_PROJECT_ROOT", _guess_project_root())).expanduser().resolve()
DATA_ROOT = Path(
    os.environ.get(
        "LOQZO_DATA_ROOT",
        os.environ.get("TASK_DATA_ROOT", os.environ.get("DATA_ROOT", PROJECT_ROOT / "Data")),
    )
).expanduser().resolve()


def resolve_data_root(data_root: Optional[Union[str, Path]] = None) -> Path:
    """统一解析数据目录，并同步回环境变量，便于其他模块读取。"""
    root = Path(data_root).expanduser().resolve() if data_root is not None else DATA_ROOT
    os.environ["LOQZO_DATA_ROOT"] = str(root)
    os.environ["TASK_DATA_ROOT"] = str(root)
    os.environ["DATA_ROOT"] = str(root)
    return root


# ============================================================
# 任务名别名映射
# ------------------------------------------------------------
# 这样可以同时支持：
#   --task_name SST2
#   --task_name sst2
#   --task_name record
#   --task_name ReCoRD
# ============================================================
TASK_NAME_ALIAS_MAP: Dict[str, str] = {
    "sst2": "SST2",
    "sst-2": "SST2",
    "copa": "Copa",
    "boolq": "BoolQ",
    "multirc": "MultiRC",
    "cb": "CB",
    "wic": "WIC",
    "wsc": "WSC",
    "wsc.fixed": "WSC",
    "record": "ReCoRD",
    "recordd": "ReCoRD",
    "rte": "RTE",
    "squad": "SQuAD",
    "drop": "DROP",
    "winogrande": "WinoGrande",
    "wikitext": "WikiText",
}


# ============================================================
# 任务分发函数
# ------------------------------------------------------------
# 支持显式传入 data_root，例如：
#   get_task("SST2", data_root="/path/to/LoQZO/Data")
# ============================================================

def get_task(task_name: str, data_root: Optional[Union[str, Path]] = None, **kwargs) -> "Dataset":
    parts = task_name.split("__")
    if len(parts) == 2:
        task_group, subtask = parts
    else:
        task_group, subtask = parts[0], None

    # 先把大小写/别名归一化
    canonical_task_group = TASK_NAME_ALIAS_MAP.get(task_group, task_group)
    if canonical_task_group != task_group and subtask is None:
        logger.info(f"任务名已归一化: {task_group} -> {canonical_task_group}")

    # 兼容 WikiText 特殊情况
    if canonical_task_group == "WikiText":
        return WikiTextDataset(subtask=subtask, data_root=data_root, **kwargs)

    class_name = f"{canonical_task_group}Dataset"
    try:
        class_ = getattr(sys.modules[__name__], class_name)
    except AttributeError as exc:
        raise ValueError(f"未找到任务类: {class_name}，请检查 --task_name={task_name}") from exc

    return class_(subtask=subtask, data_root=data_root, **kwargs)


# ============================================================
# 一些底层工具函数
# ============================================================

def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")


def _candidate_dataset_dirs(data_root: Path, aliases: List[str]) -> List[Path]:
    """为同一个数据集生成一组可能的本地目录候选路径。"""
    candidate_names: List[str] = []
    for alias in aliases:
        candidate_names.extend(
            [
                alias,
                alias.lower(),
                alias.upper(),
                _normalize_name(alias),
                alias.replace(".", ""),
                alias.replace(".", "_").lower(),
            ]
        )

    # 去重并保序
    dedup_names: List[str] = []
    seen = set()
    for name in candidate_names:
        if name not in seen:
            seen.add(name)
            dedup_names.append(name)

    candidate_dirs: List[Path] = []
    for name in dedup_names:
        candidate_dirs.append(data_root / name)
        candidate_dirs.append(data_root / "saved" / name)
    return candidate_dirs


def find_local_dataset_dir(data_root: Path, aliases: List[str]) -> Optional[Path]:
    """在 Data/ 下查找本地数据集目录。"""
    for candidate in _candidate_dataset_dirs(data_root, aliases):
        if candidate.exists():
            return candidate
    return None




# ============================================================
# HuggingFace datasets 版本兼容补丁
# ------------------------------------------------------------
# 你本地 Data/squad、Data/record、Data/drop 很可能是用较新版本
# datasets.save_to_disk() 保存的，dataset_info.json 里会出现
#   {"_type": "List"}
# 但当前服务器环境中的 datasets 版本只认识 Sequence / LargeList，
# 因而 load_from_disk() 会报：
#   ValueError: Feature type 'List' not found
#
# 这里在加载本地数据集前，把新版的 List 特征临时映射到旧版
# datasets 中已有的 Sequence。它只影响 feature 元信息解析，不改变
# arrow 数据本身，因此可以解决 SQuAD / ReCoRD / DROP 的本地加载问题。
# ============================================================
def _patch_hf_datasets_list_feature_alias() -> None:
    try:
        import datasets.features.features as hf_features

        feature_types = getattr(hf_features, "_FEATURE_TYPES", None)
        if isinstance(feature_types, dict) and "List" not in feature_types:
            # 优先映射到 Sequence，因为旧版 datasets 中 Sequence 的构造参数
            # 与新版 List 的常见保存格式（feature/length/id）最接近。
            feature_types["List"] = getattr(hf_features, "Sequence")
            logger.info("已注册 datasets feature 兼容别名: List -> Sequence")
    except Exception as exc:
        # 不因为兼容补丁失败而中断；后续 load_from_disk 若仍不兼容，
        # 会给出原始错误，便于继续定位。
        logger.warning("注册 datasets List feature 兼容别名失败: %s", exc)

def load_local_dataset_bundle(local_dir: Path):
    """
    尝试从本地目录加载 Dataset / DatasetDict。

    支持两种常见布局：
    1. 整个目录就是 HuggingFace save_to_disk() 保存的 DatasetDict
    2. 目录下按 train/validation/test 分开保存
    """
    # 先打 datasets 版本兼容补丁。这样 SQuAD / ReCoRD / DROP 这类
    # 含 list 字段的数据，即使由较新 datasets 版本保存，也能在旧版环境中读取。
    _patch_hf_datasets_list_feature_alias()

    # 情况 1：目录自身就是一个 DatasetDict 或 Dataset
    try:
        bundle = load_from_disk(str(local_dir))
        return bundle
    except Exception:
        pass

    # 情况 2：目录下分 split 单独保存
    split_dirs = {
        "train": local_dir / "train",
        "validation": local_dir / "validation",
        "valid": local_dir / "valid",
        "test": local_dir / "test",
    }
    loaded = {}
    for split_name, split_dir in split_dirs.items():
        if split_dir.exists():
            canonical_name = "validation" if split_name == "valid" else split_name
            loaded[canonical_name] = load_from_disk(str(split_dir))

    if loaded:
        return loaded

    raise FileNotFoundError(f"无法识别本地数据集目录结构: {local_dir}")


def save_dataset_bundle(bundle, save_dir: Path) -> None:
    """把下载好的数据集保存回本地 Data/ 目录。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    bundle.save_to_disk(str(save_dir))


def get_split(bundle, *candidate_names: str):
    """从 DatasetDict / dict 中读取某个 split。"""
    for name in candidate_names:
        if isinstance(bundle, dict) and name in bundle:
            return bundle[name]
        if hasattr(bundle, "keys") and name in bundle.keys():
            return bundle[name]
    raise KeyError(f"在数据集中找不到 split: {candidate_names}")


def load_bundle_with_local_fallback(
    *,
    display_name: str,
    data_root: Path,
    aliases: List[str],
    remote_loader: Callable,
    save_name: str,
):
    """
    优先从本地 Data/ 加载。
    如果本地没有，则从 HuggingFace 下载后保存到 Data/save_name。
    """
    local_dir = find_local_dataset_dir(data_root, aliases)
    if local_dir is not None:
        logger.info(f"[{display_name}] 优先使用本地数据集: {local_dir}")
        return load_local_dataset_bundle(local_dir)

    logger.warning(f"[{display_name}] 本地 Data/ 未找到，开始从 HuggingFace 下载")
    bundle = remote_loader()
    target_dir = data_root / save_name
    save_dataset_bundle(bundle, target_dir)
    logger.info(f"[{display_name}] 已保存到本地: {target_dir}")
    return bundle


# ============================================================
# Sample：统一样本结构
# ============================================================
@dataclass
class Sample:
    id: int = None
    data: dict = None
    correct_candidate: Union[str, int, List[str]] = None
    candidates: List[Union[str, int]] = None


# ============================================================
# 字段与标签兼容工具
# ------------------------------------------------------------
# 本地 Data/ 中的数据可能来自不同来源：
#   - boolq 原始数据通常使用 answer 字段；
#   - super_glue/boolq 在 HuggingFace 上通常使用 label 字段；
#   - 部分手动处理的数据会把标签写成字符串 True/False、Yes/No、0/1。
# 这里集中做一层兼容，避免因为字段名不同导致 KeyError。
# ============================================================
def _get_first_existing(example: Dict, keys: List[str], dataset_name: str):
    """按顺序返回 example 中第一个存在且非 None 的字段。"""
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    raise KeyError(f"[{dataset_name}] 样本缺少必要字段，候选字段={keys}，实际字段={list(example.keys())}")


def _to_bool_label(value, dataset_name: str = "BoolQ") -> bool:
    """把 BoolQ 的 answer/label 统一转成 bool。"""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, float):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        true_set = {"true", "yes", "y", "1", "entailment", "positive"}
        false_set = {"false", "no", "n", "0", "contradiction", "negative"}
        if normalized in true_set:
            return True
        if normalized in false_set:
            return False
    raise ValueError(f"[{dataset_name}] 无法把标签转换成 bool: {value!r}")


def _to_cb_label(value) -> int:
    """把 CB 的标签统一转成 0/1/2。

    约定与 CBTemplate 保持一致：
      0 -> entailment / Yes
      1 -> contradiction / No
      2 -> neutral / Maybe
    """
    if isinstance(value, (int, np.integer)):
        label = int(value)
        if label in (0, 1, 2):
            return label
    if isinstance(value, str):
        normalized = value.strip().lower()
        mapping = {
            "0": 0, "entailment": 0, "yes": 0, "true": 0,
            "1": 1, "contradiction": 1, "no": 1, "false": 1,
            "2": 2, "neutral": 2, "maybe": 2, "unknown": 2,
        }
        if normalized in mapping:
            return mapping[normalized]
    raise ValueError(f"[CB] 无法把标签转换成 0/1/2: {value!r}")


# ============================================================
# Dataset：所有任务类的基类
# ============================================================
class Dataset:
    mixed_set = False
    train_sep = "\n\n"
    generation = False

    def __init__(self, subtask=None, data_root: Optional[Union[str, Path]] = None, **kwargs) -> None:
        self.subtask = subtask
        self.data_root = resolve_data_root(data_root)

    def get_task_name(self):
        return self.subtask

    def load_dataset(self, *args, **kwargs):
        raise NotImplementedError

    def get_template(self, template_version=0):
        templates = {0: Template}
        return templates[template_version]

    def build_sample(self, example):
        raise NotImplementedError

    def sample_train_sets(self, num_train=32, num_dev=None, num_eval=None, num_train_sets=None, seed=None):
        # num_dev<=0 表示“不从 train split 里额外切 dev”。
        # 这样脚本里 DEV=0 时不会触发 Python 的 -0 切片空训练集问题。
        if num_dev is not None and int(num_dev) <= 0:
            num_dev = None

        if seed is not None:
            seeds = [seed]
        elif num_train_sets is not None:
            seeds = list(range(num_train_sets))
        else:
            assert num_dev is None
            len_valid_samples = len(self.samples["valid"]) if num_eval is None else num_eval
            with temp_seed(0):
                seeds = np.random.randint(0, 10000, len_valid_samples)

        train_samples = []
        for i, set_seed in enumerate(seeds):
            if self.mixed_set:
                raise NotImplementedError
            else:
                if num_dev is not None:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train + num_dev))
                    if num_train + num_dev > len(self.samples["train"]):
                        logger.warning("num_train + num_dev > available training examples")
                else:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train))
                if num_dev is not None:
                    logger.info(f"Sample train set {len(train_samples[-1])}/{len(self.samples['train'])}")
                    logger.info(f"... including dev set {num_dev} samples")
        return train_samples

    def sample_subset(self, data_split="train", seed=0, num=100, exclude=None):
        with temp_seed(seed):
            samples = self.samples[data_split]
            lens = len(samples)
            index = np.random.permutation(lens).tolist()[: num if exclude is None else num + 1]
            if exclude is not None and exclude in index:
                index.remove(exclude)
            else:
                index = index[:num]
            return [samples[i] for i in index]

    @property
    def valid_samples(self):
        return self.samples["valid"]

    # ------------------------------------------------------------
    # 供各任务复用的本地/远程加载器
    # ------------------------------------------------------------
    def _load_local_or_remote(self, display_name: str, aliases: List[str], remote_loader: Callable, save_name: str):
        return load_bundle_with_local_fallback(
            display_name=display_name,
            data_root=self.data_root,
            aliases=aliases,
            remote_loader=remote_loader,
            save_name=save_name,
        )


# ============================================================
# SST-2：二分类情感分析任务
# ============================================================
class SST2Dataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, data_root=None, **kwargs) -> None:
        super().__init__(subtask=subtask, data_root=data_root, **kwargs)
        self.load_dataset(**kwargs)

    def load_dataset(self, **kwargs):
        bundle = self._load_local_or_remote(
            display_name="SST-2",
            aliases=["sst2", "sst-2", "glue_sst2"],
            remote_loader=lambda: load_dataset("glue", "sst2"),
            save_name="sst2",
        )
        train_d = get_split(bundle, "train")
        validation_d = get_split(bundle, "validation", "valid")
        self.samples = {
            "train": [self.build_sample(example) for example in train_d],
            "valid": [self.build_sample(example) for example in validation_d],
        }

    def build_sample(self, example):
        label = int(example["label"])
        example = dict(example)
        # 额外补一个统一键，方便模板层做兼容
        if "sentence" not in example and "text" in example:
            example["sentence"] = example["text"]
        return Sample(id=example.get("idx"), data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: SST2Template}[template_version]()


# ============================================================
# COPA：因果推理二选一任务
# ============================================================
class CopaDataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self, subtask=None, data_root=None, **kwargs) -> None:
        super().__init__(subtask=subtask, data_root=data_root, **kwargs)
        self.load_dataset(**kwargs)

    def load_dataset(self, **kwargs):
        bundle = self._load_local_or_remote(
            display_name="COPA",
            aliases=["copa"],
            remote_loader=lambda: load_dataset("super_glue", "copa", trust_remote_code=True),
            save_name="copa",
        )
        train_examples = get_split(bundle, "train")
        valid_examples = get_split(bundle, "validation", "valid")
        self.samples = {
            "train": [self.build_sample(example) for example in train_examples],
            "valid": [self.build_sample(example) for example in valid_examples],
        }

    def build_sample(self, example):
        example = dict(example)
        return Sample(
            id=example.get("idx"),
            data=example,
            candidates=[example["choice1"], example["choice2"]],
            correct_candidate=example[f"choice{example['label'] + 1}"],
        )

    def get_template(self, template_version=0):
        return {0: CopaTemplate}[template_version]()


# ============================================================
# BoolQ：是/否问答任务
# ============================================================
class BoolQDataset(Dataset):
    def __init__(self, subtask=None, data_root=None, **kwargs) -> None:
        super().__init__(subtask=subtask, data_root=data_root, **kwargs)
        self.load_dataset(**kwargs)

    def load_dataset(self, **kwargs):
        bundle = self._load_local_or_remote(
            display_name="BoolQ",
            aliases=["boolq"],
            remote_loader=lambda: load_dataset("super_glue", "boolq", trust_remote_code=True),
            save_name="boolq",
        )
        train_d = get_split(bundle, "train")
        validation_d = get_split(bundle, "validation", "valid")
        self.samples = {
            "train": [self.build_sample(example) for example in train_d],
            "valid": [self.build_sample(example) for example in validation_d],
        }

    def build_sample(self, example):
        example = dict(example)

        # 兼容两种常见 BoolQ 数据格式：
        #   1) 原始 boolq: answer=True/False
        #   2) super_glue/boolq: label=1/0 或 bool
        # 这样本地 Data/boolq 无论来自哪个来源，都不会因为缺少 answer 而崩溃。
        answer_value = _get_first_existing(example, ["answer", "label", "labels", "target"], "BoolQ")
        answer_bool = _to_bool_label(answer_value, "BoolQ")

        # 模板需要 passage/question 两个字段；这里也做轻量别名兼容。
        if "passage" not in example:
            example["passage"] = _get_first_existing(example, ["text", "context", "paragraph"], "BoolQ")
        if "question" not in example:
            example["question"] = _get_first_existing(example, ["query", "sentence", "claim"], "BoolQ")

        return Sample(
            id=example.get("idx", example.get("id")),
            data=example,
            candidates=["Yes", "No"],
            correct_candidate="Yes" if answer_bool else "No",
        )

    def get_template(self, template_version=2):
        return {0: BoolQTemplate, 1: BoolQTemplateV2, 2: BoolQTemplateV3}[template_version]()


# ============================================================
# MultiRC：多句阅读理解二分类任务
# ============================================================
class MultiRCDataset(Dataset):
    def __init__(self, subtask=None, data_root=None, **kwargs) -> None:
        super().__init__(subtask=subtask, data_root=data_root, **kwargs)
        self.load_dataset(**kwargs)

    def load_dataset(self, **kwargs):
        bundle = self._load_local_or_remote(
            display_name="MultiRC",
            aliases=["multirc", "multi_rc"],
            remote_loader=lambda: load_dataset("super_glue", "multirc", trust_remote_code=True),
            save_name="multirc",
        )
        train_set = get_split(bundle, "train")
        valid_set = get_split(bundle, "validation", "valid")
        self.samples = {
            "train": [self.build_sample(example) for example in train_set],
            "valid": [self.build_sample(example) for example in valid_set],
        }

    def build_sample(self, example):
        example = dict(example)
        return Sample(data=example, candidates=[0, 1], correct_candidate=example["label"])

    def get_template(self, template_version=0):
        return {0: MultiRCTemplate}[template_version]()


# ============================================================
# CB：三分类自然语言推理任务
# ============================================================
class CBDataset(Dataset):
    def __init__(self, subtask=None, data_root=None, **kwargs) -> None:
        super().__init__(subtask=subtask, data_root=data_root, **kwargs)
        self.load_dataset(**kwargs)

    def load_dataset(self, **kwargs):
        bundle = self._load_local_or_remote(
            display_name="CB",
            aliases=["cb"],
            remote_loader=lambda: load_dataset("super_glue", "cb", trust_remote_code=True),
            save_name="cb",
        )
        train_set = get_split(bundle, "train")
        valid_set = get_split(bundle, "validation", "valid")
        self.samples = {
            "train": [self.build_sample(example) for example in train_set],
            "valid": [self.build_sample(example) for example in valid_set],
        }

    def build_sample(self, example):
        example = dict(example)

        # CB 标准字段是 premise/hypothesis/label；部分本地数据可能沿用
        # sentence1/sentence2 或 label_text，这里统一补齐，保证模板层可用。
        if "premise" not in example:
            example["premise"] = _get_first_existing(example, ["sentence1", "text1", "context"], "CB")
        if "hypothesis" not in example:
            example["hypothesis"] = _get_first_existing(example, ["sentence2", "text2", "claim"], "CB")
        label_value = _get_first_existing(example, ["label", "labels", "target", "gold_label", "label_text"], "CB")

        return Sample(
            id=example.get("idx", example.get("id")),
            data=example,
            candidates=[0, 1, 2],
            correct_candidate=_to_cb_label(label_value),
        )

    def get_template(self, template_version=0):
        return {0: CBTemplate}[template_version]()


# ============================================================
# WIC：词义消歧二分类任务
# ============================================================
class WICDataset(Dataset):
    def __init__(self, subtask=None, data_root=None, **kwargs) -> None:
        super().__init__(subtask=subtask, data_root=data_root, **kwargs)
        self.load_dataset(**kwargs)

    def load_dataset(self, **kwargs):
        bundle = self._load_local_or_remote(
            display_name="WIC",
            aliases=["wic"],
            remote_loader=lambda: load_dataset("super_glue", "wic", trust_remote_code=True),
            save_name="wic",
        )
        train_set = get_split(bundle, "train")
        valid_set = get_split(bundle, "validation", "valid")
        self.samples = {
            "train": [self.build_sample(example) for example in train_set],
            "valid": [self.build_sample(example) for example in valid_set],
        }

    def build_sample(self, example):
        example = dict(example)
        return Sample(data=example, candidates=[0, 1], correct_candidate=example["label"])

    def get_template(self, template_version=0):
        return {0: WICTemplate}[template_version]()


# ============================================================
# WSC：代词指代消解二分类任务
# ============================================================
class WSCDataset(Dataset):
    def __init__(self, subtask=None, data_root=None, **kwargs) -> None:
        super().__init__(subtask=subtask, data_root=data_root, **kwargs)
        self.load_dataset(**kwargs)

    def load_dataset(self, **kwargs):
        bundle = self._load_local_or_remote(
            display_name="WSC",
            aliases=["wsc", "wsc.fixed", "wsc_fixed"],
            remote_loader=lambda: load_dataset("super_glue", "wsc.fixed", trust_remote_code=True),
            save_name="wsc",
        )
        train_set = get_split(bundle, "train")
        valid_set = get_split(bundle, "validation", "valid")
        self.samples = {
            "train": [self.build_sample(example) for example in train_set],
            "valid": [self.build_sample(example) for example in valid_set],
        }

    def build_sample(self, example):
        example = dict(example)
        return Sample(data=example, candidates=[0, 1], correct_candidate=example["label"])

    def get_template(self, template_version=0):
        return {0: WSCTemplate}[template_version]()


# ============================================================
# ReCoRD：阅读理解 / 实体填空任务
# ============================================================
class ReCoRDDataset(Dataset):
    def __init__(self, subtask=None, data_root=None, **kwargs) -> None:
        super().__init__(subtask=subtask, data_root=data_root, **kwargs)
        self.load_dataset(**kwargs)

    def load_dataset(self, **kwargs):
        bundle = self._load_local_or_remote(
            display_name="ReCoRD",
            aliases=["record", "recordd", "record_dataset", "record_task"],
            remote_loader=lambda: load_dataset("super_glue", "record", trust_remote_code=True),
            save_name="record",
        )
        train_set = get_split(bundle, "train")
        valid_set = get_split(bundle, "validation", "valid")
        self.samples = {
            "train": [self.build_sample(example) for example in train_set],
            "valid": [self.build_sample(example) for example in valid_set],
        }

    def build_sample(self, example):
        example = dict(example)
        return Sample(data=example, candidates=example["entities"], correct_candidate=example["answers"])

    def get_template(self, template_version=0):
        return {0: ReCoRDTemplateGPT3}[template_version]()


# ============================================================
# RTE：文本蕴含二分类任务
# ------------------------------------------------------------
# 这里做了一层字段兼容：
#   - 如果本地数据来自 super_glue/rte，则通常是 premise / hypothesis
#   - 如果本地数据来自 glue/rte，则通常是 sentence1 / sentence2
# 我们统一补齐 premise/hypothesis，尽量减少模板层出错概率。
# ============================================================
class RTEDataset(Dataset):
    def __init__(self, subtask=None, data_root=None, **kwargs) -> None:
        super().__init__(subtask=subtask, data_root=data_root, **kwargs)
        self.load_dataset(**kwargs)

    def load_dataset(self, **kwargs):
        bundle = self._load_local_or_remote(
            display_name="RTE",
            aliases=["rte"],
            remote_loader=lambda: load_dataset("super_glue", "rte", trust_remote_code=True),
            save_name="rte",
        )
        train_set = get_split(bundle, "train")
        valid_set = get_split(bundle, "validation", "valid")
        self.samples = {
            "train": [self.build_sample(example) for example in train_set],
            "valid": [self.build_sample(example) for example in valid_set],
        }

    def build_sample(self, example):
        example = dict(example)
        if "premise" not in example and "sentence1" in example:
            example["premise"] = example["sentence1"]
        if "hypothesis" not in example and "sentence2" in example:
            example["hypothesis"] = example["sentence2"]
        return Sample(data=example, candidates=[0, 1], correct_candidate=example["label"])

    def get_template(self, template_version=0):
        return {0: RTETemplate}[template_version]()


# ============================================================
# SQuAD：生成式问答任务
# ============================================================
class SQuADDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, data_root=None, **kwargs) -> None:
        super().__init__(subtask=subtask, data_root=data_root, **kwargs)
        self.load_dataset(**kwargs)

    def load_dataset(self, **kwargs):
        bundle = self._load_local_or_remote(
            display_name="SQuAD",
            aliases=["squad", "squad_v1", "squad1"],
            remote_loader=lambda: load_dataset("squad"),
            save_name="squad",
        )
        train_examples = get_split(bundle, "train")
        valid_examples = get_split(bundle, "validation", "valid")
        self.samples = {
            "train": [self.build_sample(example, idx) for idx, example in enumerate(train_examples)],
            "valid": [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)],
        }

    def build_sample(self, example, idx):
        example = dict(example)
        answers = example["answers"]["text"]
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "title": example.get("title", ""),
                "context": example["context"],
                "question": example["question"],
                "answers": answers,
            },
            candidates=None,
            correct_candidate=answers,
        )

    def get_template(self, template_version=0):
        return {0: SQuADv2Template}[template_version]()


# ============================================================
# DROP：生成式阅读理解任务
# ============================================================
class DROPDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, data_root=None, **kwargs) -> None:
        super().__init__(subtask=subtask, data_root=data_root, **kwargs)
        self.load_dataset(**kwargs)

    def load_dataset(self, **kwargs):
        bundle = self._load_local_or_remote(
            display_name="DROP",
            aliases=["drop"],
            remote_loader=lambda: load_dataset("drop"),
            save_name="drop",
        )
        train_examples = get_split(bundle, "train")
        valid_examples = get_split(bundle, "validation", "valid")
        self.samples = {
            "train": [self.build_sample(example, idx) for idx, example in enumerate(train_examples)],
            "valid": [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)],
        }

    def build_sample(self, example, idx):
        example = dict(example)
        answers = example["answers_spans"]["spans"]
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "context": example["passage"],
                "question": example["question"],
                "answers": answers,
            },
            candidates=None,
            correct_candidate=answers,
        )

    def get_template(self, template_version=0):
        return {0: DROPTemplate}[template_version]()


# ============================================================
# WinoGrande：常识推理 / 指代消解任务
# ============================================================
class WinoGrandeDataset(Dataset):
    def __init__(self, subtask=None, data_root=None, **kwargs) -> None:
        super().__init__(subtask=subtask, data_root=data_root, **kwargs)
        self.load_dataset(**kwargs)

    def load_dataset(self, **kwargs):
        bundle = self._load_local_or_remote(
            display_name="WinoGrande",
            aliases=["winogrande"],
            remote_loader=lambda: load_dataset("winogrande", "winogrande_m"),
            save_name="winogrande",
        )
        train_set = get_split(bundle, "train")
        valid_set = get_split(bundle, "validation", "valid")
        self.samples = {
            "train": [self.build_sample(example) for example in train_set],
            "valid": [self.build_sample(example) for example in valid_set],
        }

    def build_sample(self, example):
        example = dict(example)
        sentence = example["sentence"]
        context, target = sentence.split("_")
        return Sample(
            data=example,
            candidates=[example["option1"] + target, example["option2"] + target],
            correct_candidate=example[f'option{example["answer"]}'] + target,
        )

    def get_template(self, template_version=0):
        if template_version == 0:
            return WinoGrandeTemplate()
        raise NotImplementedError(f"Template version {template_version} not implemented for WinoGrande")


# ============================================================
# WikiText：语言建模 / 困惑度评估任务
# ============================================================
class WikiTextDataset(Dataset):
    metric_name = "perplexity"
    generation = True
    train_sep = "\n\n"

    def __init__(self, subtask=None, data_root=None, **kwargs) -> None:
        super().__init__(subtask=subtask, data_root=data_root, **kwargs)
        self.load_dataset(**kwargs)

    def load_dataset(self, **kwargs):
        bundle = self._load_local_or_remote(
            display_name="WikiText",
            aliases=["wikitext", "wikitext_2_v1", "wikitext-2-v1"],
            remote_loader=lambda: load_dataset("wikitext", "wikitext-2-v1"),
            save_name="wikitext",
        )
        train_examples = get_split(bundle, "train")
        valid_examples = get_split(bundle, "validation", "valid")
        test_examples = get_split(bundle, "test")

        def filter_text(examples):
            return [ex for ex in examples if ex["text"].strip() and len(ex["text"].split()) >= 40]

        filtered_train = filter_text(train_examples)
        filtered_valid = filter_text(valid_examples)
        filtered_test = filter_text(test_examples)

        self.samples = {
            "train": [self.build_sample(example, idx) for idx, example in enumerate(filtered_train)],
            "valid": [self.build_sample(example, idx) for idx, example in enumerate(filtered_valid)],
            "test": [self.build_sample(example, idx) for idx, example in enumerate(filtered_test)],
        }

    def build_sample(self, example, idx):
        text = example["text"].strip()
        split_point = len(text) // 2
        context = text[:split_point]
        continuation = text[split_point:]
        return Sample(
            id=idx,
            data={
                "context": context,
                "continuation": continuation,
                "full_text": text,
            },
            candidates=[continuation],
            correct_candidate=continuation,
        )

    def get_template(self, template_version=0):
        return {0: WikiTextTemplate}[template_version]()


# ============================================================
# 文件尾部说明
# ------------------------------------------------------------
# 说明 1：这份 tasks.py 已经优先从项目根目录的 Data/ 下读取数据。
# 说明 2：如果 run.py 调用 get_task(args.task_name, data_root=args.data_root)，会直接生效。
# 说明 3：如果 run.py 还是旧版，只调用 get_task(args.task_name)，这份代码也会自动尝试从 LoQZO/Data 读取。
#
# 运行命令示例（全部改成单行，不使用反斜杠续行）：
# python Code/train/run.py --model_name OPT-1.3B --task_name SST2 --output_dir outputs/opt13b_sst2
# python Code/train/run.py --model_name Llama2-7B --task_name BoolQ --output_dir outputs/llama2_7b_boolq
# python Code/train/run.py --model_name Llama3-8B-8bit --task_name MultiRC --output_dir outputs/llama3_8b_int8_multirc
# python Code/train/run.py --model_name Mistral-7B-4bit --task_name Copa --output_dir outputs/mistral_7b_int4_copa
# ============================================================
