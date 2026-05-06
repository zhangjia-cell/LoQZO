# -*- coding: utf-8 -*-
"""
quant_model.py

qft / QZO 所需的模型量化包装工具。

主要修复点：
1) 原文件从 dir(model) 递归替换子模块，容易遍历到非注册属性，
   也会把 nn.ModuleList 替换成 nn.Sequential，影响 Hugging Face 模型结构；
2) run_loqzo.py 期望从本文件导入 enable_quantization，但原文件没有导出；
3) QZO 交替训练需要稳定找到 *.quant_weight.alpha，因此这里用 named_children
   原地替换 nn.Linear，保留原模型层级与参数命名。
"""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from .quant_modules import TensorQuantizer, LinearQuantizer
from .quant_utils import quant_args

logger = logging.getLogger(__name__)


# 默认不量化 lm_head。decoder-only 分类/生成实验通常不训练输出头，
# 跳过它可以减少显存和数值不稳定。
DEFAULT_SKIP_MODULE_NAMES = {"lm_head"}


def _dist_rank_or_zero() -> int:
    """兼容未初始化分布式时的 rank 查询。"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _iter_quant_build_notes(model: nn.Module) -> Iterable[str]:
    """调试用：返回已包装量化层的名字。"""
    for name, module in model.named_modules():
        if isinstance(module, LinearQuantizer):
            yield name


def quantize_model(model: nn.Module, skip_module_names: Tuple[str, ...] = tuple(DEFAULT_SKIP_MODULE_NAMES)) -> nn.Module:
    """
    将模型中的 nn.Linear 原地替换为 LinearQuantizer。

    参数：
    - model：Hugging Face 模型或其子模块；
    - skip_module_names：按子模块名跳过，例如 lm_head。

    返回：
    - 原地替换后的 model。保留 ModuleList / ModuleDict / 自定义模型类结构，
      因此比把 ModuleList 改成 Sequential 更安全。
    """
    skip_set = set(skip_module_names or ())

    for child_name, child in list(model.named_children()):
        if child_name in skip_set:
            continue

        # 只替换原生 nn.Linear；已经是 LinearQuantizer 时不重复包装。
        if isinstance(child, nn.Linear):
            quant_mod = LinearQuantizer(**quant_args)
            quant_mod.set_param(child)
            quant_mod.to(device=child.weight.device)
            setattr(model, child_name, quant_mod)
        else:
            quantize_model(child, skip_module_names=skip_module_names)

    if _dist_rank_or_zero() == 0:
        # 只在最外层调用时通常会打印多次，因此这里控制为 debug，避免污染训练日志。
        logger.debug("已完成 qft 量化包装，示例量化层：%s", list(_iter_quant_build_notes(model))[:5])
    return model


def enable_quantization(model: nn.Module) -> None:
    """打开所有 TensorQuantizer 的量化开关。"""
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            module.enable_quantization(name)


def disable_quantization(model: nn.Module) -> None:
    """关闭所有 TensorQuantizer 的量化开关，用于调试/消融。"""
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            module.disable_quantization(name)


def disable_input_quantization(model: nn.Module) -> None:
    """只关闭激活量化，保留权重量化。"""
    for module in model.modules():
        if isinstance(module, TensorQuantizer) and module.is_input:
            module.disable_input_quantization()


def set_first_last_layer(model):
    """保留旧接口：当前版本不主动修改首尾层，仅收集模块方便后续扩展。"""
    module_list_weight = []
    module_list_input = []
    for m in model.modules():
        if isinstance(m, TensorQuantizer) and m.is_input is False:
            module_list_weight += [m]
        if isinstance(m, TensorQuantizer) and m.is_input is True:
            module_list_input += [m]
    return module_list_weight, module_list_input


def set_8_bit_layer_l(model, layer_list):
    """按层号把指定层的 weight/input quantizer 临时设置成 8-bit。"""
    if layer_list == "None":
        return
    layer_list = list(map(lambda x: int(x), str(layer_list).split(',')))
    module_list = []

    for m in model.modules():
        if isinstance(m, TensorQuantizer):
            module_list += [m]
            m.has_inited_quant_para.data = torch.zeros_like(m.has_inited_quant_para)
            if hasattr(m, "_quant_initialized_py"):
                m._quant_initialized_py = False

    if _dist_rank_or_zero() == 0:
        print("------------- 8-bit Re-SET -------------")
        print(len(layer_list))
    assert len(layer_list) > 0

    for i in range(int(len(module_list) / 2)):
        if i in layer_list:
            if _dist_rank_or_zero() == 0:
                print(module_list[i * 2].name, i)
                print(module_list[i * 2 + 1].name, i)
            module_list[i * 2].bit.data = torch.tensor(8, device=module_list[i * 2].bit.device)
            module_list[i * 2 + 1].bit.data = torch.tensor(8, device=module_list[i * 2 + 1].bit.device)

    if _dist_rank_or_zero() == 0:
        print("------------- 8-bit Re-SET -------------")


def set_8_bit_layer_n(model, l_num):
    """按量化误差选择若干层设置成 8-bit，保留旧实验接口。"""
    module_list = []
    mse_list = []

    for m in model.modules():
        if isinstance(m, TensorQuantizer):
            module_list += [m]
            mse_list += [m.mse.item()]
            m.has_inited_quant_para.data = torch.zeros_like(m.has_inited_quant_para)
            if hasattr(m, "_quant_initialized_py"):
                m._quant_initialized_py = False

    print("------------- 8-bit Re-SET -------------")
    print(l_num)
    assert l_num > 0
    l_num *= 2

    first_num = 0 * 2
    for i in range(0, first_num):
        print(module_list[i].name)
        module_list[i].bit.data = torch.tensor(8, device=module_list[i].bit.device)

    # 与旧代码保持一致：默认保留最后 2 层为 8-bit。
    last_num = 2 * 2
    for i in range(len(mse_list) - last_num, len(mse_list)):
        print(module_list[i].name)
        module_list[i].bit.data = torch.tensor(8, device=module_list[i].bit.device)

    print("------------- First and Last end -------------")
    module_list = module_list[first_num: len(mse_list) - last_num]
    mse_list = mse_list[first_num: len(mse_list) - last_num]

    mse_list_pair = []
    for i in range(0, int(len(mse_list) / 2)):
        mse_list_pair += [mse_list[i * 2] + mse_list[i * 2 + 1]]

    mses = np.array(mse_list_pair)
    mse_idx = np.argsort(-mses)
    l_num -= first_num
    l_num -= last_num
    l_num = int(l_num / 2)

    if l_num > 0:
        for i in mse_idx[0:l_num]:
            print(module_list[i * 2].name, mses[i], i)
            print(module_list[i * 2 + 1].name, mses[i], i)
            module_list[i * 2].bit.data = torch.tensor(8, device=module_list[i * 2].bit.device)
            module_list[i * 2 + 1].bit.data = torch.tensor(8, device=module_list[i * 2 + 1].bit.device)

    print("------------- 8-bit Re-SET -------------")


def load_ant_state_dict(model, checkpoint):
    """从 checkpoint 中恢复 ANT/QFT 量化 codebook。"""
    for name, module in model.named_modules():
        if name + ".quant_grid" in checkpoint.keys():
            module.quant_grid.data = checkpoint[name + ".quant_grid"]
