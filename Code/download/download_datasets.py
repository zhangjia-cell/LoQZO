#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download NLP datasets and save each dataset into its own subfolder.

Example:
    Data/
    ├── sst2/
    ├── rte/
    ├── cb/
    └── ...
"""

from __future__ import annotations
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset


# 数据集映射：名字 -> (HF数据集名, config)
DATASET_REGISTRY: Dict[str, Tuple[str, str | None]] = {
    "sst2": ("glue", "sst2"),
    "rte": ("super_glue", "rte"),
    "cb": ("super_glue", "cb"),
    "boolq": ("super_glue", "boolq"),
    "wsc": ("super_glue", "wsc"),
    "wic": ("super_glue", "wic"),
    "multirc": ("super_glue", "multirc"),
    "copa": ("super_glue", "copa"),
    "record": ("super_glue", "record"),
    "squad": ("squad", None),
    "drop": ("drop", None),

    # 新增数据集
    # WinoGrande 默认使用 winogrande_xl
    "winogrande": ("allenai/winogrande", "winogrande_xl"),

    # WikiText 默认使用 wikitext-2-raw-v1
    # 如果你想换成更大的版本，可改成:
    # "wikitext-103-raw-v1"
    "wikitext": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
}


DEFAULT_DATASETS: List[str] = list(DATASET_REGISTRY.keys())


# 别名支持：输入会先归一化，再映射到标准名字
NAME_ALIASES: Dict[str, str] = {
    "sst2": "sst2",
    "rte": "rte",
    "cb": "cb",
    "boolq": "boolq",
    "wsc": "wsc",
    "wic": "wic",
    "multirc": "multirc",
    "copa": "copa",
    "record": "record",
    "squad": "squad",
    "drop": "drop",

    # WinoGrande aliases
    "winogrande": "winogrande",
    "wino": "winogrande",

    # WikiText aliases
    "wikitext": "wikitext",
    "wiki": "wikitext",
    "wikitext2": "wikitext",
}


def normalize_name(name: str) -> str:
    """统一输入格式"""
    name = name.strip().lower().replace("-", "").replace("_", "")
    name = NAME_ALIASES.get(name, name)

    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {name}")
    return name


def download_one(dataset_key: str, output_dir: Path) -> dict:
    """下载单个数据集，并保存到独立子目录"""
    hf_name, hf_config = DATASET_REGISTRY[dataset_key]

    # ✅ 每个数据集一个独立子文件夹
    save_path = output_dir / dataset_key

    print("\n" + "=" * 60)
    print(f"[INFO] Downloading: {dataset_key}")
    print(f"[INFO] HF source: {hf_name}, config: {hf_config}")
    print(f"[INFO] Save path: {save_path}")

    # 下载数据
    if hf_config is None:
        ds = load_dataset(hf_name)
    else:
        ds = load_dataset(hf_name, hf_config)

    # 创建目录
    save_path.mkdir(parents=True, exist_ok=True)

    # 保存
    print("[INFO] Saving to disk...")
    ds.save_to_disk(str(save_path))

    # 统计
    split_sizes = {k: len(v) for k, v in ds.items()}

    print(f"[DONE] Saved: {save_path}")
    print(f"[INFO] Splits: {split_sizes}")

    # 保存 meta 信息
    meta = {
        "dataset": dataset_key,
        "hf_name": hf_name,
        "hf_config": hf_config,
        "path": str(save_path.resolve()),
        "splits": split_sizes,
    }

    with open(save_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="目标保存目录，例如: /home/.../Data",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="指定下载哪些数据集，例如: sst2 rte cb winogrande wikitext",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [normalize_name(x) for x in args.datasets]

    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Datasets: {datasets}")

    success = []
    failed = []

    for name in datasets:
        try:
            meta = download_one(name, output_dir)
            success.append(meta)
        except Exception as e:
            print(f"[ERROR] Failed: {name} -> {e}")
            failed.append({"dataset": name, "error": str(e)})

    # 总结
    summary_path = output_dir / "download_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {"success": success, "failed": failed},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print(f"[FINISH] success={len(success)}, failed={len(failed)}")
    print(f"[INFO] Summary: {summary_path}")


if __name__ == "__main__":
    main()


# ============================
# 运行方式（直接复制用）
# ============================

# 下载全部数据集
# python -u download_datasets.py --output_dir /home/zhangjia/Code/LoQZO/LoQZO/Data

# 只下载部分数据集
# python -u download_datasets.py --output_dir /home/zhangjia/Code/LoQZO/LoQZO/Data --datasets sst2 rte cb

# 下载新增的两个数据集
# python -u download_datasets.py --output_dir /home/zhangjia/Code/LoQZO/LoQZO/Data --datasets winogrande wikitext