#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 QuZO 相关模型与数据集的脚本。

这个版本已经根据你的项目目录结构做了调整：
- 默认把模型下载到项目根目录下的 Models/ 文件夹
- 默认把数据集下载到项目根目录下的 Data/ 文件夹
- 默认兼容你截图中的目录结构：
    项目根目录/
      ├── Code/
      │   └── download/
      │       └── download_quzo_assets.py
      ├── Data/
      └── Models/

说明：
1. 你列出的 “8bit / 4bit Llama2-7B / 13B”、“8bit / 4bit Llama3-8B”、
   “8bit / 4bit Mistral-7B” 在 Hugging Face / Transformers 的常见用法里，
   通常不是单独再下载一份新 checkpoint，而是基于同一个 base model，
   在加载时通过 BitsAndBytesConfig 以 8bit / 4bit 方式量化加载。
   因此本脚本会：
   - 下载对应的 base model
   - 在 Models/quantized_manifests/ 下生成量化加载说明文件（json）

2. Meta 的 Llama 2 / Llama 3 仓库是 gated（受限访问）模型。
   你需要：
   - 先在 Hugging Face 网页上接受对应 license / access request
   - 再使用你自己的 Hugging Face token 下载

3. 公开模型（例如大多数 OPT、RoBERTa-Large）和公开数据集，通常不强制要求登录；
   但为了避免权限或速率限制问题，建议你统一使用 token。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


# ============================================================
# 数据结构定义
# ============================================================

@dataclass(frozen=True)
class ModelSpec:
    """模型配置。

    alias:
        你在命令行里使用的模型别名。
    repo_id:
        Hugging Face 仓库名。
    quantization:
        量化模式，可选：None / "8bit" / "4bit"。
        注意：这里的 8bit / 4bit 是“加载方式说明”，不是单独的官方 checkpoint。
    """

    alias: str
    repo_id: str
    quantization: Optional[str] = None


@dataclass(frozen=True)
class DatasetSpec:
    """数据集配置。"""

    alias: str
    path: str
    name: Optional[str] = None


# ============================================================
# 你当前需要的模型列表
# ============================================================
# 说明：
# - OPT-1.3B / 2.7B / 6.7B / 13B / 30B：直接下载 base checkpoint
# - RoBERTa-Large：直接下载 base checkpoint（公开模型）
# - Llama2-7B / 13B：直接下载 base checkpoint（需要 token + 访问权限）
# - Llama2-7B / 13B 的 8bit / 4bit：下载同一个 base model，并生成量化加载 manifest
# - Llama3-8B：下载 base checkpoint（需要 token + 访问权限）
# - 8bit / 4bit Llama3-8B：下载同一个 base model，并生成量化加载 manifest
# - 8bit / 4bit Mistral-7B：下载 Mistral base model，并生成量化加载 manifest

REQUESTED_MODELS: List[ModelSpec] = [
    ModelSpec("OPT-1.3B", "facebook/opt-1.3b"),
    ModelSpec("OPT-2.7B", "facebook/opt-2.7b"),
    ModelSpec("OPT-6.7B", "facebook/opt-6.7b"),
    ModelSpec("OPT-13B", "facebook/opt-13b"),
    ModelSpec("OPT-30B", "facebook/opt-30b"),

    ModelSpec("RoBERTa-Large", "FacebookAI/roberta-large"),

    ModelSpec("Llama2-7B", "meta-llama/Llama-2-7b-hf"),
    ModelSpec("Llama2-7B-8bit", "meta-llama/Llama-2-7b-hf", quantization="8bit"),
    ModelSpec("Llama2-7B-4bit", "meta-llama/Llama-2-7b-hf", quantization="4bit"),

    ModelSpec("Llama2-13B", "meta-llama/Llama-2-13b-hf"),
    ModelSpec("Llama2-13B-8bit", "meta-llama/Llama-2-13b-hf", quantization="8bit"),
    ModelSpec("Llama2-13B-4bit", "meta-llama/Llama-2-13b-hf", quantization="4bit"),

    ModelSpec("Llama3-8B", "meta-llama/Meta-Llama-3-8B"),
    ModelSpec("Llama3-8B-8bit", "meta-llama/Meta-Llama-3-8B", quantization="8bit"),
    ModelSpec("Llama3-8B-4bit", "meta-llama/Meta-Llama-3-8B", quantization="4bit"),

    ModelSpec("Mistral-7B-8bit", "mistralai/Mistral-7B-v0.3", quantization="8bit"),
    ModelSpec("Mistral-7B-4bit", "mistralai/Mistral-7B-v0.3", quantization="4bit"),
]


# ============================================================
# QuZO 论文相关常用数据集
# ============================================================
# 你现在主要是先准备下载脚本，所以数据集我也一并保留。
# 默认 profile 为 quzo-full，覆盖论文正文和附录中常见数据集。

DATASET_PROFILES: Dict[str, List[DatasetSpec]] = {
    "quzo-core": [
        DatasetSpec("sst2", "glue", "sst2"),
        DatasetSpec("rte", "glue", "rte"),
        DatasetSpec("boolq", "super_glue", "boolq"),
        DatasetSpec("cb", "super_glue", "cb"),
        DatasetSpec("wsc.fixed", "super_glue", "wsc.fixed"),
        DatasetSpec("multirc", "super_glue", "multirc"),
        DatasetSpec("copa", "super_glue", "copa"),
        DatasetSpec("record", "super_glue", "record"),
        DatasetSpec("squad", "squad", None),
        DatasetSpec("drop", "ucinlp/drop", None),
    ],
    "quzo-full": [
        DatasetSpec("sst2", "glue", "sst2"),
        DatasetSpec("rte", "glue", "rte"),
        DatasetSpec("mnli", "glue", "mnli"),
        DatasetSpec("snli", "snli", None),
        DatasetSpec("sst5", "SetFit/sst5", None),
        DatasetSpec("trec", "CogComp/trec", None),
        DatasetSpec("boolq", "super_glue", "boolq"),
        DatasetSpec("cb", "super_glue", "cb"),
        DatasetSpec("wic", "super_glue", "wic"),
        DatasetSpec("wsc.fixed", "super_glue", "wsc.fixed"),
        DatasetSpec("multirc", "super_glue", "multirc"),
        DatasetSpec("copa", "super_glue", "copa"),
        DatasetSpec("record", "super_glue", "record"),
        DatasetSpec("squad", "squad", None),
        DatasetSpec("drop", "ucinlp/drop", None),
    ],
}


# ============================================================
# 下载过滤规则
# ============================================================
# 只保留常见模型加载需要的文件，避免把 onnx / tflite 等无关格式一起拉下来。

MODEL_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.py",
    "tokenizer*",
    "special_tokens_map.json",
    "generation_config.json",
    "config.json",
    "merges.txt",
    "vocab.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "pytorch_model*.bin",
    "pytorch_model*.bin.index.json",
]

MODEL_IGNORE_PATTERNS = [
    "original/*",  # Meta Llama 仓库里原始格式文件通常体积更大，这里默认不下载
    "*.onnx",
    "*.tflite",
    "*.msgpack",
    "*.h5",
]


# ============================================================
# 工具函数
# ============================================================

def eprint(*args: object) -> None:
    """打印到 stderr。"""
    print(*args, file=sys.stderr)


def slugify(text: str) -> str:
    """把 repo_id / alias 转成适合作为文件夹名的形式。"""
    text = text.strip().replace("/", "--")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    return text.strip("-")


def ensure_dir(path: Path) -> None:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: object) -> None:
    """把对象写入 json 文件。"""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def infer_project_root(cli_project_root: Optional[Path]) -> Path:
    """推断项目根目录。

    推断逻辑：
    1. 如果命令行显式传了 --project-root，就直接用它。
    2. 否则从当前脚本位置开始逐级向上寻找：
       - 同时包含 Models/ 和 Data/ 的目录
       - 或者同时包含 Code/ 和 Models/ 的目录
    3. 若都没找到，则退回到当前工作目录。
    """

    if cli_project_root is not None:
        return cli_project_root.resolve()

    script_path = Path(__file__).resolve()
    candidates = [script_path.parent] + list(script_path.parents)

    for candidate in candidates:
        has_models = (candidate / "Models").exists()
        has_data = (candidate / "Data").exists()
        has_code = (candidate / "Code").exists()
        if (has_models and has_data) or (has_models and has_code):
            return candidate

    return Path.cwd().resolve()


def parse_alias_filter(values: Optional[List[str]]) -> Optional[Set[str]]:
    """解析命令行里传入的模型 / 数据集别名过滤器。"""
    if not values:
        return None

    merged: List[str] = []
    for item in values:
        merged.extend(part.strip() for part in item.split(",") if part.strip())
    return set(merged)


def load_hf_token(cli_token: Optional[str]) -> Optional[str]:
    """按优先级读取 Hugging Face token。

    优先级：
    1. --hf-token
    2. 环境变量 HF_TOKEN
    3. 环境变量 HUGGINGFACE_HUB_TOKEN
    4. 本机已经通过 `hf auth login` 保存过的 token
    """

    if cli_token:
        return cli_token.strip()

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        return env_token.strip()

    try:
        from huggingface_hub import get_token

        cached = get_token()
        if cached:
            return cached.strip()
    except Exception:
        pass

    return None


def maybe_login_to_hf(token: Optional[str], do_login: bool) -> None:
    """可选：把 token 登录到本机 huggingface_hub。

    注意：
    - 这一步不是必须的。
    - 如果你已经 `hf auth login` 过，或者已经通过环境变量传 token，通常就够了。
    """

    if not do_login:
        return

    if not token:
        raise SystemExit("[HF] 你开启了 --hf-login，但当前没有检测到 token。")

    try:
        from huggingface_hub import login
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖 huggingface_hub，请先安装：\n"
            "pip install -U huggingface_hub"
        ) from exc

    print("[HF] 正在把 token 登录到本机 huggingface_hub ...")
    login(token=token, add_to_git_credential=False)
    print("[HF] 登录完成。")


def repo_requires_auth(repo_id: str) -> bool:
    """判断一个仓库是否大概率需要认证。"""
    return repo_id.startswith("meta-llama/")


def quantization_manifest(spec: ModelSpec, base_dir: Path) -> Dict[str, object]:
    """生成 8bit / 4bit 加载说明文件。

    这里不会生成一套新的量化 checkpoint，只是生成一个 manifest，
    告诉你后续如何基于 base model 以 bitsandbytes 的方式加载。
    """

    if spec.quantization == "8bit":
        config = {
            "load_in_8bit": True,
            "device_map": "auto",
            "dtype": "auto",
        }
    elif spec.quantization == "4bit":
        config = {
            "load_in_4bit": True,
            "device_map": "auto",
            "dtype": "auto",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
        }
    else:
        raise ValueError(f"不支持的量化模式: {spec.quantization}")

    loader_example = (
        "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n"
        "import torch\n\n"
        f"base_model = r'{base_dir.resolve()}'\n"
        f"quant_mode = '{spec.quantization}'\n\n"
        "if quant_mode == '8bit':\n"
        "    quantization_config = BitsAndBytesConfig(load_in_8bit=True)\n"
        "else:\n"
        "    quantization_config = BitsAndBytesConfig(\n"
        "        load_in_4bit=True,\n"
        "        bnb_4bit_quant_type='nf4',\n"
        "        bnb_4bit_compute_dtype=torch.bfloat16,\n"
        "    )\n\n"
        "tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)\n"
        "model = AutoModelForCausalLM.from_pretrained(\n"
        "    base_model,\n"
        "    device_map='auto',\n"
        "    torch_dtype='auto',\n"
        "    quantization_config=quantization_config,\n"
        ")\n"
    )

    return {
        "alias": spec.alias,
        "repo_id": spec.repo_id,
        "base_model_dir": str(base_dir.resolve()),
        "quantization": spec.quantization,
        "transformers_bitsandbytes_config": config,
        "loader_example": loader_example,
        "note": "这是运行时量化加载说明文件，不是单独下载的官方量化 checkpoint。",
    }


# ============================================================
# 模型下载逻辑
# ============================================================

def download_models(
    models_dir: Path,
    token: Optional[str],
    selected_aliases: Optional[Set[str]],
    max_workers: int,
) -> None:
    """下载模型到 Models/ 文件夹。"""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖 huggingface_hub，请先安装：\n"
            "pip install -U huggingface_hub"
        ) from exc

    base_root = models_dir / "base_models"
    manifests_root = models_dir / "quantized_manifests"
    ensure_dir(base_root)
    ensure_dir(manifests_root)

    requested = [m for m in REQUESTED_MODELS if selected_aliases is None or m.alias in selected_aliases]
    if not requested:
        print("[models] 没有模型匹配 --only-models 过滤条件，跳过模型下载。")
        return

    unknown_aliases = set()
    if selected_aliases is not None:
        known_aliases = {m.alias for m in REQUESTED_MODELS}
        unknown_aliases = selected_aliases - known_aliases

    if unknown_aliases:
        eprint(f"[models] 警告：以下模型别名不存在，将被忽略: {sorted(unknown_aliases)}")

    unique_repo_ids = sorted({m.repo_id for m in requested})
    repo_to_dir: Dict[str, Path] = {}

    print(f"[models] 需要准备的 base model 仓库数量: {len(unique_repo_ids)}")

    for repo_id in unique_repo_ids:
        local_dir = base_root / slugify(repo_id)
        repo_to_dir[repo_id] = local_dir
        done_flag = local_dir / ".download_complete.json"

        # 如果已经下载成功过，就直接跳过。
        if done_flag.exists():
            print(f"[models] 跳过已存在模型: {repo_id} -> {local_dir}")
            continue

        if repo_requires_auth(repo_id) and not token:
            eprint(
                f"[models] 警告：{repo_id} 看起来是需要认证的 gated 仓库，"
                "但当前未检测到 Hugging Face token。"
            )

        print(f"[models] 开始下载: {repo_id} -> {local_dir}")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=str(local_dir),
                token=token,
                allow_patterns=MODEL_ALLOW_PATTERNS,
                ignore_patterns=MODEL_IGNORE_PATTERNS,
                max_workers=max_workers,
                local_files_only=False,
            )
        except Exception as exc:  # noqa: BLE001
            eprint(f"[models] 下载失败: {repo_id}")
            eprint(f"[models] 错误信息: {exc}")
            eprint(
                "[models] 提示：如果这是 gated 仓库，请先在 Hugging Face 网页上"
                "接受 license / access request，并确认 token 具有访问权限。"
            )
            raise

        dump_json(
            done_flag,
            {
                "repo_id": repo_id,
                "local_dir": str(local_dir.resolve()),
                "status": "ok",
            },
        )

    # 生成一个总索引文件，方便你后续训练脚本读取。
    requested_index = []
    for spec in requested:
        base_dir = repo_to_dir[spec.repo_id]
        entry: Dict[str, object] = {
            "alias": spec.alias,
            "repo_id": spec.repo_id,
            "base_model_dir": str(base_dir.resolve()),
        }

        if spec.quantization is None:
            entry["type"] = "base"
        else:
            entry["type"] = "quantized-manifest"
            entry["quantization"] = spec.quantization
            manifest_path = manifests_root / f"{slugify(spec.alias)}.json"
            dump_json(manifest_path, quantization_manifest(spec, base_dir))
            entry["manifest_path"] = str(manifest_path.resolve())

        requested_index.append(entry)

    dump_json(models_dir / "requested_models_index.json", requested_index)
    print(f"[models] 已写出索引文件: {models_dir / 'requested_models_index.json'}")


# ============================================================
# 数据集下载逻辑
# ============================================================

def download_datasets(
    data_dir: Path,
    dataset_profile: str,
    selected_aliases: Optional[Set[str]],
) -> None:
    """下载并保存数据集到 Data/ 文件夹。"""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖 datasets，请先安装：\n"
            "pip install -U datasets"
        ) from exc

    # 这里把 cache 和 save_to_disk 分开。
    cache_root = data_dir / "_hf_cache"
    saved_root = data_dir / "hf_saved"
    ensure_dir(cache_root)
    ensure_dir(saved_root)

    specs = DATASET_PROFILES[dataset_profile]
    specs = [d for d in specs if selected_aliases is None or d.alias in selected_aliases]

    if not specs:
        print("[datasets] 没有数据集匹配过滤条件，跳过数据集下载。")
        return

    unknown_aliases = set()
    if selected_aliases is not None:
        known_aliases = {d.alias for d in DATASET_PROFILES[dataset_profile]}
        unknown_aliases = selected_aliases - known_aliases

    if unknown_aliases:
        eprint(f"[datasets] 警告：以下数据集别名不存在，将被忽略: {sorted(unknown_aliases)}")

    index = []
    for spec in specs:
        save_dir = saved_root / slugify(spec.alias)
        done_flag = save_dir / ".download_complete.json"

        if done_flag.exists():
            print(f"[datasets] 跳过已存在数据集: {spec.alias} -> {save_dir}")
            index.append(
                {
                    "alias": spec.alias,
                    "path": spec.path,
                    "name": spec.name,
                    "saved_dir": str(save_dir.resolve()),
                    "status": "ok",
                }
            )
            continue

        print(f"[datasets] 开始下载: {spec.path} | name={spec.name!r} -> {save_dir}")
        try:
            ds = load_dataset(spec.path, spec.name, cache_dir=str(cache_root))
            ensure_dir(save_dir.parent)
            ds.save_to_disk(str(save_dir))
        except Exception as exc:  # noqa: BLE001
            eprint(f"[datasets] 下载失败: alias={spec.alias}, path={spec.path}, name={spec.name}")
            eprint(f"[datasets] 错误信息: {exc}")
            raise

        dump_json(
            done_flag,
            {
                "alias": spec.alias,
                "path": spec.path,
                "name": spec.name,
                "saved_dir": str(save_dir.resolve()),
                "status": "ok",
            },
        )

        index.append(
            {
                "alias": spec.alias,
                "path": spec.path,
                "name": spec.name,
                "saved_dir": str(save_dir.resolve()),
                "status": "ok",
            }
        )

    dump_json(data_dir / "datasets_index.json", index)
    print(f"[datasets] 已写出索引文件: {data_dir / 'datasets_index.json'}")


# ============================================================
# 命令行参数
# ============================================================

def build_argparser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="下载 QuZO 相关模型和数据集（默认模型放到 Models/，数据集放到 Data/）。"
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="项目根目录。若不传，脚本会自动根据当前文件位置推断。",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="模型保存目录。若不传，默认使用 <project_root>/Models",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="数据集保存目录。若不传，默认使用 <project_root>/Data",
    )
    parser.add_argument(
        "--dataset-profile",
        choices=sorted(DATASET_PROFILES.keys()),
        default="quzo-full",
        help="要下载的数据集组合。默认 quzo-full。",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="跳过模型下载。",
    )
    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="跳过数据集下载。",
    )
    parser.add_argument(
        "--only-models",
        nargs="*",
        default=None,
        help=(
            "只下载指定模型别名。"
            "可以写空格分隔，也可以写逗号分隔。"
            "例如：--only-models OPT-1.3B OPT-6.7B RoBERTa-Large Llama2-7B-8bit Llama3-8B-4bit"
        ),
    )
    parser.add_argument(
        "--only-datasets",
        nargs="*",
        default=None,
        help=(
            "只下载指定数据集别名。"
            "例如：--only-datasets sst2 rte squad drop"
        ),
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token。若不传，则依次尝试 HF_TOKEN / HUGGINGFACE_HUB_TOKEN / 本地已登录 token。",
    )
    parser.add_argument(
        "--hf-login",
        action="store_true",
        help="把当前 token 登录到本机 huggingface_hub。不是必须项。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="模型文件并行下载线程数，默认 8。",
    )

    return parser


# ============================================================
# 主函数
# ============================================================

def main() -> None:
    """脚本入口。"""

    parser = build_argparser()
    args = parser.parse_args()

    project_root = infer_project_root(args.project_root)
    models_dir = args.models_dir.resolve() if args.models_dir else (project_root / "Models")
    data_dir = args.data_dir.resolve() if args.data_dir else (project_root / "Data")

    ensure_dir(project_root)
    ensure_dir(models_dir)
    ensure_dir(data_dir)

    token = load_hf_token(args.hf_token)
    maybe_login_to_hf(token, args.hf_login)

    only_models = parse_alias_filter(args.only_models)
    only_datasets = parse_alias_filter(args.only_datasets)

    summary = {
        "project_root": str(project_root.resolve()),
        "models_dir": str(models_dir.resolve()),
        "data_dir": str(data_dir.resolve()),
        "dataset_profile": args.dataset_profile,
        "skip_models": args.skip_models,
        "skip_datasets": args.skip_datasets,
        "only_models": sorted(only_models) if only_models else None,
        "only_datasets": sorted(only_datasets) if only_datasets else None,
        "has_hf_token": bool(token),
    }
    dump_json(project_root / "download_plan.json", summary)

    print("=" * 70)
    print("下载计划")
    print(f"项目根目录 : {project_root.resolve()}")
    print(f"模型目录   : {models_dir.resolve()}")
    print(f"数据目录   : {data_dir.resolve()}")
    print(f"数据集配置 : {args.dataset_profile}")
    print(f"检测到 token: {'是' if token else '否'}")
    print(f"计划文件   : {(project_root / 'download_plan.json').resolve()}")
    print("=" * 70)

    if not args.skip_models:
        download_models(models_dir, token, only_models, args.max_workers)

    if not args.skip_datasets:
        download_datasets(data_dir, args.dataset_profile, only_datasets)

    print("[done] 所有请求的下载任务已完成。")


if __name__ == "__main__":
    main()


# ============================================================
# 用法说明（文件末尾注释）
# ============================================================
# 1）安装依赖：
#    pip install -U huggingface_hub datasets
#
#    如果你后续要真正加载 8bit / 4bit 模型，还需要：
#    pip install -U transformers accelerate bitsandbytes
#
# 2）最推荐的两种认证方式：
#
#    方式 A：先在命令行登录一次（推荐）
#    hf auth login
#
#    然后直接运行：
#    python download_quzo_assets.py
#
#    方式 B：运行时显式传入 token
#    HF_TOKEN=你的token python download_quzo_assets.py
#    或者
#    python download_quzo_assets.py --hf-token 你的token
#
# 3）如果你的脚本就放在你截图里的 Code/download/ 目录下，
#    并且项目根目录下已经有 Models/ 和 Data/，
#    那么直接运行时会自动保存到：
#    - 模型：<项目根目录>/Models/
#    - 数据：<项目根目录>/Data/
#
# 4）只下载模型：
#    HF_TOKEN=你的token python download_quzo_assets.py --skip-datasets
#
# 5）只下载数据集：
#    python download_quzo_assets.py --skip-models
#
# 6）只下载部分模型：
#    HF_TOKEN=你的token python download_quzo_assets.py \
#        --only-models OPT-1.3B OPT-6.7B RoBERTa-Large OPT-13B Llama2-7B Llama3-8B-4bit
#
# 7）只下载 OPT-6.7B：
#    python download_quzo_assets.py \
#        --skip-datasets \
#        --only-models OPT-6.7B
#
# 8）只下载 Llama2 量化模型入口：
#    HF_TOKEN=你的token python download_quzo_assets.py \
#        --skip-datasets \
#        --only-models Llama2-7B-8bit Llama2-7B-4bit Llama2-13B-8bit Llama2-13B-4bit
#
# 9）只下载部分数据集：
#    python download_quzo_assets.py --only-datasets sst2 rte squad drop
#
# 10）如果你不想依赖自动推断项目根目录，也可以手动指定：
#     HF_TOKEN=你的token python download_quzo_assets.py \
#         --project-root /你的项目根目录
#
# 11）如果你只想下载 RoBERTa-Large：
#     python download_quzo_assets.py \
#         --skip-datasets \
#         --only-models RoBERTa-Large
#
# 12）关于是否必须登录 Hugging Face：
#     - 公开模型 / 公开数据集：通常不强制要求登录。
#     - Meta Llama 2 / Llama 3：必须先在网页上获得访问权限，并使用 token。
#     - 你既然已经有 Hugging Face 账号和 token，建议直接用 token，最稳妥。
#
# 13）目录说明：
#     - base model 会下载到：Models/base_models/
#     - 8bit / 4bit manifest 会保存到：Models/quantized_manifests/
#     - 数据集 save_to_disk 结果会保存到：Data/hf_saved/
#     - Hugging Face 数据缓存会保存到：Data/_hf_cache/