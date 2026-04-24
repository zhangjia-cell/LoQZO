# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

# ============================= 中文总览 =============================
# 这个文件本质上是一个“自定义 HuggingFace Trainer”。
# 它在原生 Trainer 的基础上，额外支持：
# 1) MeZO / Zeroth-Order 训练（trainer == 'zo'）
# 2) 低比特 Zeroth-Order 训练（trainer == 'zo_lowbit'）
# 3) 低比特 full-tuning 风格的 ZO 训练（trainer == 'zo_lowbit_ft'）
# 4) 若干量化辅助函数：扰动量化、梯度量化、反量化
# 5) 分布式 / FSDP / DeepSpeed 场景下更稳妥的保存逻辑
#
# 阅读建议：
#   先看 class OurTrainer 里的 _inner_training_loop，理解训练主流程；
#   再看 zo_forward / lowbit_zo_step / lowbit_zo_update；
#   最后再回来看前面的量化辅助函数。
# ==================================================================
import contextlib
import functools
import glob
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from metrics import f1
import numpy as np

from tqdm.auto import tqdm
from transformers import Trainer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV

#new transformers

# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    get_reporting_integration_callbacks,
    hp_params,
)

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from huggingface_hub import Repository

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    find_labels,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tensorrt_fx_available,
    is_torch_tpu_available,
    is_torchdynamo_available,
    logging,
)
from transformers.utils.generic import ContextManagers



DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    # from .utils.notebook import NotebookProgressCallback
    from .utils import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# if is_apex_available():
#     from apex import amp

if is_datasets_available():
    import datasets

# if is_torch_tpu_available(check_device=False):
#     import torch_xla.core.xla_model as xm
#     import torch_xla.debug.metrics as met
#     import torch_xla.distributed.parallel_loader as pl

# if is_fairscale_available():
#     dep_version_check("fairscale")
#     import fairscale
#     from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
#     from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
#     from fairscale.nn.wrap import auto_wrap
#     from fairscale.optim import OSS
#     from fairscale.optim.grad_scaler import ShardedGradScaler


# if is_sagemaker_mp_enabled():
#     import smdistributed.modelparallel.torch as smp
#     from smdistributed.modelparallel import __version__ as SMP_VERSION

#     IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

#     from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
# else:
#     IS_SAGEMAKER_MP_POST_1_10 = False


if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing

# ============================= checkpoint 文件名常量 =============================
# 这些常量对应 HuggingFace Trainer 保存 checkpoint 时使用的标准文件名。
# 自定义 trainer 在恢复训练、保存优化器、保存调度器时都会依赖它们。
# ===============================================================================
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


# ============================= 量化辅助函数（权重 / 梯度 / 扰动） =============================
# 这一组函数不是训练主循环，而是“被主循环调用的工具函数”。
# 其中最重要的是：
#   - zo_quant / zo_dequant：给 ZO 扰动或参数更新量做低比特量化与反量化
#   - zo_quant_data：给参数本身做低比特量化
#   - quantize_gradients：给一阶训练的梯度做量化，便于和 ZO 方法对比
# =========================================================================================

# 按通道 absmax 量化权重：
# 对每个输出通道分别统计最大绝对值，再把该通道缩放到整数网格上。
# 这种做法在权重分布差异较大的线性层里通常比全张量统一缩放更稳。
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    zero = 0
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_()

    return w, 1 / scales, zero

import torch


# 对梯度做简单的线性量化：
# 这里会先把梯度归一化到 [0, 1]，再映射到离散量化网格，最后反量化回浮点数。
# 注意：这更像“模拟低比特梯度”的实现，而不是把梯度永久存成整数类型。
def quantize_gradients(parameters, bits=8):
    """
    Quantize gradients to a specified number of bits.
    """
    scale = 2**bits - 1
    for param in parameters:
        if param.grad is not None:
            grad = param.grad.data
            min_val, max_val = grad.min(), grad.max()
            grad = (grad - min_val) / (max_val - min_val)  # Normalize to 0-1
            grad = torch.round(grad * scale) / scale  # Quantize and dequantize
            param.grad.data = grad * (max_val - min_val) + min_val  # Rescale


def _safe_bernoulli_probability(prob: torch.Tensor) -> torch.Tensor:
    """
    将随机舍入的小数概率清洗到 [0, 1]。

    说明：
    torch.bernoulli 要求输入概率严格位于 [0, 1]。当训练后期参数/scale 出现
    Inf、NaN 或极端大值时，rest = x_scaled - floor(x_scaled) 可能变成 NaN，
    CUDA 会触发 `Assertion 0 <= p && p <= 1` 并中断进程。
    这里做防御性处理，不改变正常情况下的随机舍入结果。
    """
    return torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


def _safe_positive_scale(scale, device=None, dtype=None, eps: float = 1e-12):
    """
    保证量化 scale / scaling_factor 是有限正数。

    说明：
    参数全零、全 NaN 或训练发散时，scale 可能为 0/NaN/Inf。
    如果继续除以这个值，会把量化结果污染成 NaN，最终导致随机舍入崩溃。
    """
    if not torch.is_tensor(scale):
        scale = torch.tensor(scale, device=device, dtype=dtype if dtype is not None else torch.float32)
    if device is not None and scale.device != torch.device(device):
        scale = scale.to(device=device)
    if dtype is not None and scale.dtype != dtype:
        scale = scale.to(dtype=dtype)
    fallback = torch.full_like(scale, eps)
    scale = torch.where(torch.isfinite(scale) & (scale.abs() > eps), scale, fallback)
    return scale


# 随机舍入量化：
# 给定一个浮点张量，先按 scale 缩放到整数网格，再根据小数部分做 Bernoulli 随机上取整。
# 返回值里同时包含：
#   1) 量化后的整数张量
#   2) 反量化后的浮点张量
#   3) 对应的 scale
# 这类函数通常用于模拟论文里 stochastic rounding 的效果。
def stochastic_quantize(tensor, bit_width=8):
    """
    Stochastically quantize a tensor into a given bit-width.
    
    Args:
        tensor (torch.Tensor): Input tensor to quantize.
        bit_width (int): Number of bits for quantization.
    
    Returns:
        quantized_tensor (torch.Tensor): Stochastically quantized tensor.
        dequantized_tensor (torch.Tensor): Dequantized tensor.
    """
    # Compute scaling factor
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    max_abs_value = torch.max(torch.abs(tensor))
    scale = max_abs_value / (2**(bit_width - 1) - 1)  # Signed quantization
    scale = _safe_positive_scale(scale, device=tensor.device, dtype=tensor.dtype, eps=1e-6)

    # Normalize tensor to quantization range
    normalized_tensor = tensor / scale

    # Compute stochastic rounding
    lower = torch.floor(normalized_tensor)  # Floor value
    fractional = _safe_bernoulli_probability(normalized_tensor - lower)  # Fractional part
    stochastic = torch.bernoulli(fractional)  # Bernoulli(p=fractional)

    # Quantize with stochastic rounding
    quantized_tensor = lower + stochastic

    # Clamp to valid range
    quantized_tensor = torch.clamp(quantized_tensor, -(2**(bit_width - 1)), 2**(bit_width - 1) - 1)

    # Dequantize back to FP32
    dequantized_tensor = quantized_tensor * scale

    return quantized_tensor, dequantized_tensor,scale


# 给“参数值 / 更新值”做量化：
# 这里支持：
#   - 对称量化 / 非对称量化
#   - 随机舍入 / 普通舍入
#   - block exponent（把 scale 约束成 2 的整数次幂，便于硬件实现）
# 返回的是：量化整数、缩放因子、零点。
def zo_quant_data(x, nbits=16,blk_exp=True, sym=True, stochastic=True, seed=None):
    # Set the random seed if provided
    # seed = np.random.randint(1000000000)
    
    if seed is not None:
        torch.manual_seed(seed)

    # 防止上游发散产生 NaN/Inf 后继续污染随机舍入概率。
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    # n_levels = 2**nbits
    n_levels = 2**(nbits-1)
    
    if sym:  # symmetric quantization
        x1 = torch.max(torch.abs(x))
        # x1 = x.abs().max(dim=-1, keepdim=True)[0]
        x0 = 0
    else:  # asymmetric quantization
        x0, x1 = torch.min(x), torch.max(x)
        # x0, x1 = x.abs().min(dim=-1, keepdim=True)[0], x.abs().max(dim=-1, keepdim=True)[0]
    
    # Calculate the scale factor a
    scaling_factor = n_levels / (x1 - x0)
    scaling_factor = _safe_positive_scale(scaling_factor, device=x.device, dtype=x.dtype)

    if blk_exp:  # block exponent
        scaling_factor = 2**torch.floor(torch.log2(scaling_factor))
        scaling_factor = _safe_positive_scale(scaling_factor, device=x.device, dtype=x.dtype)
    # if scaling_factor == 0:
    #         scaling_factor = 1e-6
    # Calculate the zero point b
    zero_point = torch.floor(-scaling_factor * (x0 + x1) / 2)

    if stochastic:
        # Apply stochastic rounding
        x_floor = torch.floor(x * scaling_factor)
        rest = _safe_bernoulli_probability(x * scaling_factor - x_floor)
        x_int = x_floor + torch.bernoulli(rest)  # Use random rounding based on the fractional part
    else:
        # Apply standard rounding
        x_int = torch.floor(x * scaling_factor)

    # Clamp the quantized values to the valid range - signed
    # x_quant = torch.clamp(x_int + b, 0, n_levels - 1)
    x_quant = torch.clamp(x_int + zero_point, -n_levels, n_levels - 1)
    
    # Dequantize the result
    # qx = (x_quant - zero_point) / scaling_factor
    
    return x_quant,scaling_factor,zero_point  # Return the quantized and dequantized tensor


# def zo_quant_data(x, nbits=16,blk_exp=True, sym=True, stochastic=True, seed=None):
#     # Set the random seed if provided
#     # seed = np.random.randint(1000000000)
    
#     if seed is not None:
#         torch.manual_seed(seed)
    
#     # n_levels = 2**nbits
#     n_levels = 2**(nbits-1)
    
#     if sym:  # symmetric quantization
#         x1 = torch.max(torch.abs(x))
#         x0 = -x1
#     else:  # asymmetric quantization
#         x0, x1 = torch.min(x), torch.max(x)
    
#     # Calculate the scale factor a
#     scaling_factor = n_levels / (x1 - x0)

#     if blk_exp:  # block exponent
#         scaling_factor = 2**torch.floor(torch.log2(scaling_factor))

#     # Calculate the zero point b
#     zero_point = torch.floor(-scaling_factor * (x0 + x1) / 2)

#     if stochastic:
#         # Apply stochastic rounding
#         x_floor = torch.floor(x * scaling_factor)
#         rest = x * scaling_factor - x_floor
#         x_int = x_floor + torch.bernoulli(rest)  # Use random rounding based on the fractional part
#     else:
#         # Apply standard rounding
#         x_int = torch.round(x * scaling_factor)


#     x_quant = torch.clamp(x_int + zero_point, -n_levels, n_levels - 1)
    
    
#     return x_quant,scaling_factor,zero_point  # Return the quantized and dequantized tensor


# 给 ZO 扰动向量做量化：
# 这是这份文件里最关键的量化函数之一。
# 它单独处理了 1-bit、2-bit 和 >=3-bit 三种情况。
# 直觉上：
#   - nbits 越低，扰动越粗糙，但硬件/显存越省；
#   - stochastic=True 时，更贴近论文里“随机量化扰动”的设定。
def zo_quant(x, nbits=4, sym=True, stochastic=True, seed=None):
    # Set the random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # 防止扰动量化阶段因为 NaN/Inf 产生非法 Bernoulli 概率。
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    if nbits == 1:
        # 1-bit quantization (binary quantization)
        if sym:
            x_quant = torch.sign(x)  # Symmetric: -1 for negative, +1 for positive values
        else:
            x_quant = (x > 0).float()  # Unsigned: 0 or 1 based on threshold at 0
        scaler = 1  # Scale factor is not needed as values are already constrained
        zero_point = 0 if sym else 0
        return x_quant, scaler, zero_point

    elif nbits == 2:
        # 2-bit quantization
        if sym:
            # For signed quantization, levels could be -3, -1, +1, +3 (centered around 0)
            x1 = x.abs().max()  # max absolute value for symmetric quantization
            scaler = _safe_positive_scale(3 / x1, device=x.device, dtype=x.dtype)
            x_scaled = x * scaler
            x_quant = torch.round(x_scaled).clamp(-3, 3)  # Clamp to the 4 levels
            zero_point = 0
        else:
            # For unsigned quantization, levels could be 0, 1, 2, 3
            x_min, x_max = x.min(), x.max()
            scaler = _safe_positive_scale(3 / (x_max - x_min), device=x.device, dtype=x.dtype)
            zero_point = torch.round(-scaler * x_min)
            x_scaled = x * scaler + zero_point
            x_quant = torch.round(x_scaled).clamp(0, 3)
        return x_quant, scaler, zero_point

    else:
        # For 3-bit and above, use general quantization
        n_levels = 2 ** (nbits - 1) if sym else 2 ** nbits - 1
        
        if sym:
            x1 = x.abs().max()  # Max absolute value
            scaler = _safe_positive_scale(n_levels / x1, device=x.device, dtype=x.dtype)
            zero_point = 0  # Center around 0 for symmetric quantization
        else:
            x_min, x_max = x.min(), x.max()
            scaler = _safe_positive_scale(n_levels / (x_max - x_min), device=x.device, dtype=x.dtype)
            zero_point = torch.round(-scaler * x_min)

        # Apply scaling and optional stochastic rounding
        x_scaled = x * scaler
        if stochastic:
            x_floor = torch.floor(x_scaled)
            rest = _safe_bernoulli_probability(x_scaled - x_floor)
            x_quant = x_floor + torch.bernoulli(rest)  # Stochastic rounding
        else:
            x_quant = torch.round(x_scaled)

        # Clamp the quantized values to valid range
        x_quant = torch.clamp(x_quant + zero_point, -n_levels if sym else 0, n_levels if sym else n_levels)
        # x_dequant = (x_quant - zero_point) / scaler
        
        return x_quant, scaler, zero_point

# def zo_quant(x, nbits=2, scaler=1, sym=True, stochastic=False, seed=None):
#     # Set the random seed if provided
#     # seed = np.random.randint(1000000000)
    
#     if seed is not None:
#         torch.manual_seed(seed)
    
#     # n_levels = 2**nbits
#     n_levels = 2**(nbits-1) - 1
    
#     # if sym:  # symmetric quantization
#     #     x1 = torch.max(torch.abs(x))
#     #     x0 = -x1
#     # else:  # asymmetric quantization
#     #     x0, x1 = torch.min(x), torch.max(x)
#     if sym:  # symmetric quantization
#         # x1 = torch.max(torch.abs(x))
#         x1 = x.abs().max(dim=-1, keepdim=True)[0]
#         x0 = 0
#     else:  # asymmetric quantization
#         # x0, x1 = torch.min(x), torch.max(x)
#         x0, x1 = x.abs().min(dim=-1, keepdim=True)[0], x.abs().max(dim=-1, keepdim=True)[0]
#     # Calculate the scale factor a
#     scaler = n_levels / (x1 - x0)


#     # Calculate the zero point b
#     zero_point = 0 if sym else torch.round(-scaler * (x0 + x1) / 2)

#     if stochastic:
#         # Apply stochastic rounding
#         x_floor = torch.floor(x * scaler)
#         rest = x * scaler - x_floor
#         x_int = x_floor + torch.bernoulli(rest)  # Use random rounding based on the fractional part
#     else:
#         # Apply standard rounding
#         x_int = torch.round(x * scaler)

#     # Clamp the quantized values to the valid range - signed
#     # x_quant = torch.clamp(x_int + b, 0, n_levels - 1)
#     x_quant = torch.clamp(x_int + zero_point, -n_levels, n_levels )
    
#     # Dequantize the result
#     # x_quant = (x_quant - zero_point) / scaler
    
#     return x_quant,scaler,zero_point  # Return the quantized and dequantized tensor

# Define the quantizer function with seed support

# 反量化：把离散整数表示恢复为浮点近似值。
# 在这份代码里，很多时候不会直接拿整数去更新参数，
# 而是“量化 -> 再反量化 -> 用浮点近似值参与更新”。
def zo_dequant(x_quant, scaling_factor, zero_point):

    
    # Dequantize the result
    scaling_factor = _safe_positive_scale(
        scaling_factor,
        device=x_quant.device if torch.is_tensor(x_quant) else None,
        dtype=x_quant.dtype if torch.is_tensor(x_quant) and x_quant.is_floating_point() else None,
    )
    qx = (x_quant - zero_point) / scaling_factor
    qx = torch.nan_to_num(qx, nan=0.0, posinf=0.0, neginf=0.0)
    # qx = (qx - x).detach() + x
    
    return qx  # Return the quantized and dequantized tensor


# 计算一组梯度估计值的方差。
# 这更像实验分析工具：用来观察不同 step / 不同 query 下的梯度波动有多大。
def compute_gradient_variance(gradient_list):
    """
    Compute variance of gradients from a list of gradient estimates.
    Args:
        gradient_list: List of gradients (projected_grad) across steps.
    Returns:
        Variance (tensor) across the gradient estimates.
    """
    gradients = torch.tensor(gradient_list)
    mean_grad = gradients.mean()
    variance = ((gradients - mean_grad) ** 2).mean()
    return variance


# ============================= 自定义 Trainer 主体 =============================
# 这个类继承自 HuggingFace Trainer，但重写了训练主循环。
# 也就是说：
#   - 数据集如何取 batch：仍然基本沿用 HF Trainer
#   - 训练一步到底怎么做：由这里重新定义
#
# 这份代码最重要的阅读顺序是：
#   1) _inner_training_loop        训练总控
#   2) zo_forward                 无反传的前向 loss 计算
#   3) lowbit_zo_step / zo_step   零阶梯度估计
#   4) lowbit_zo_update           真正更新参数
#   5) save_model                 复杂分布式场景下的保存
# ============================================================================
class OurTrainer(Trainer):

    from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    # ----------------------------- 分布式辅助函数 -----------------------------
    def _dist_is_initialized(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def _dist_world_size(self) -> int:
        return dist.get_world_size() if self._dist_is_initialized() else 1

    def _dist_rank(self) -> int:
        return dist.get_rank() if self._dist_is_initialized() else 0

    def _dist_device(self) -> torch.device:
        if torch.cuda.is_available():
            local_rank = getattr(self.args, "local_rank", -1)
            if local_rank not in (-1, None):
                return torch.device(f"cuda:{local_rank}")
            return torch.device("cuda")
        return torch.device("cpu")

    def _broadcast_seed(self, seed: int | None = None) -> int:
        if self._dist_world_size() == 1:
            return int(seed if seed is not None else np.random.randint(1000000000))

        if self._dist_rank() == 0:
            value = int(seed if seed is not None else np.random.randint(1000000000))
        else:
            value = 0
        seed_tensor = torch.tensor([value], device=self._dist_device(), dtype=torch.long)
        dist.broadcast(seed_tensor, src=0)
        return int(seed_tensor.item())

    def _all_reduce_mean_scalar(self, value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().float().reshape(1).to(self._dist_device())
        else:
            tensor = torch.tensor([float(value)], device=self._dist_device(), dtype=torch.float32)

        if self._dist_world_size() > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self._dist_world_size()
        return tensor[0]

    # ----------------------------- 训练主循环入口 -----------------------------
    # 这是整个 trainer 的核心：
    #   - 它先准备 dataloader、optimizer、scheduler、checkpoint 恢复状态；
    #   - 然后在 epoch/step 双层循环里决定当前是走：
    #         1) 普通一阶训练
    #         2) zo（标准 MeZO）
    #         3) zo_lowbit（低比特零阶）
    #         4) zo_lowbit_ft（另一种低比特 full-tuning 风格）
    # 如果你只想快速抓主线，这个函数必须先读。
    # ------------------------------------------------------------------------
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # MeZO added: Linear probing

        # ========== 线性探测（Linear Probing）分支 ==========
        # 这个分支不会进入正常的梯度更新流程。
        # 它先从模型最后一层前抽特征，再用 sklearn 的逻辑回归拟合一个分类头。
        # 拟合完成后，直接把逻辑回归的权重写回模型的 lm_head。
        # 这是一个很便宜的 baseline，用来快速测试表示质量。
        # ====================================================
        if self.args.linear_probing:

            def _get_token_prediction_layer(model):
                if model.config.model_type == "opt":
                    return model.lm_head
                else:
                    raise NotImplementedError(model.config.model_type)

            def _extract_features(model, *args, **kwargs):
                """some magic for getting features pre last layer"""
                features = {}
                def __hook(model_, input_, output_):
                    features["features"] = input_[0].detach()

                _get_token_prediction_layer(model).register_forward_hook(__hook)
                model.forward(*args, **kwargs)
                return features["features"]

            logger.info("Linear probing")
            logger.info("Starting to get features for training dataset")
            targets = []
            features = []
            with torch.inference_mode():
                for step, inputs in enumerate(tqdm(train_dataloader)):
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)
                        
                    feature = _extract_features(self.model, **inputs)
                    target = inputs["labels"]

                    # Shift the target (bc it's autoregressive LM) and add the corresponding part
                    assert not self.args.train_as_classification and self.args.only_train_option
                    feature, target = feature[:, :-1], target[:, 1:]
                    for _i, _len in enumerate(inputs["option_len"]):
                        features.append(feature[_i, -_len:])
                        targets.append(target[_i, -_len:])

            logger.info("Finished getting features for training dataset")

            features = torch.cat(features, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            # Whether to use bias
            if self.model.config.model_type in ["opt", "gpt2"]:
                use_bias = False
            else:
                raise NotImplementedError
            # Set early stopping
            tol = 0.01 if self.args.lp_early_stopping else 1e-4 # 1e-4 is scipy default
            max_iter = 1000 if self.args.lp_early_stopping else 5000

            logger.info("Fitting logistic regression...")
            reg = LogisticRegressionCV(max_iter=max_iter, fit_intercept=use_bias, multi_class="multinomial", random_state=0, tol=tol, n_jobs=-1).fit(features, targets)
            logger.info("Done")

            logger.info("Assigning weights to model")
            decoder = _get_token_prediction_layer(self.model)
            coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if use_bias:
                bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if coef_torch.shape[0] == 1: # The regressor only detects two classes
                assert len(reg.classes_) == 2
                coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
                if use_bias:
                    bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

            for _i, token_id in enumerate(reg.classes_):
                decoder.weight.data[token_id] = coef_torch[_i]
                if use_bias:
                    decoder.bias.data[token_id] = bias_torch[_i]

            return None

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        # 下面开始准备训练调度相关的全局量：
        #   - 每个 epoch 有多少步
        #   - 总共有多少步
        #   - 总 batch size 是多少
        # 这些量会影响 scheduler、日志打印、恢复训练和 speed metrics。
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )

        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )



        # 真正进入 epoch 循环。
        # 从这里开始，训练会在每个 epoch 内继续遍历 step。
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            # if is_torch_tpu_available():
            #     parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device) # type :ignore
            #     epoch_iterator = parallel_loader
            # else:
            #     epoch_iterator = train_dataloader

            epoch_iterator = train_dataloader
            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            # 每个 step 都会进入这里。
            # 对于 ZO 方法，这里不是 backward + optimizer.step，
            # 而是“先估计投影梯度，再显式更新参数”。
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                
                # Layer-wise Perturb
                # if step %100 == 0:
                #     layer_sensitivity = {}
                #     original_loss = self.zo_forward(model, inputs)  # This is the baseline loss
                #     for name, param in model.named_parameters():
                #         if param.requires_grad:
                #             original_data = param.data.clone()
                #             param.data.add_(torch.randn_like(param) * 1e-3)  # Apply a small perturbation
                #             perturbed_loss = self.zo_forward(model, inputs)  # Compute loss with perturbed layer
                #             layer_sensitivity[name] = perturbed_loss - original_loss
                #             param.data = original_data
                #     ranked_layers = sorted(layer_sensitivity.items(), key=lambda x: abs(x[1]), reverse=True)

                #     k = 30
                #     top_layers = [layer[0] for layer in ranked_layers[:k]]
                #     for name, param in model.named_parameters():
                #         if param.requires_grad and name not in top_layers:
                #             param.requires_grad = False
                #     print("step:",step, "perturb",k,"layer")

                # MeZO added: estimate gradient

                # ========== 选择当前 step 采用哪种训练方式 ==========
                # regular       : 普通一阶训练（会走 backward）
                # zo            : 标准 MeZO / ZO-SGD
                # zo_lowbit     : 低比特扰动 + 低比特更新
                # zo_lowbit_ft  : 另一条低比特 full-tuning 更新分支
                # =================================================
                if args.trainer == "zo":
                    tr_loss_step = self.zo_step(model, inputs)
                elif args.trainer == "zo_lowbit":
                    tr_loss_step = self.lowbit_zo_step(model, inputs)
                elif args.trainer == "zo_lowbit_ft":
                    tr_loss_step = self.lowbit_zo_ftstep(model, inputs)
                else:
                    if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        # with model.no_sync():
                            tr_loss_step = self.training_step(model, inputs)
                    else:
                        tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # MeZO added: update model with the estimated gradient

                    # ========== 到了真正“更新参数”的时刻 ==========
                    # 前面 tr_loss_step 只是做 loss 评估或估计投影梯度，
                    # 这里才决定参数是否真的被修改。
                    # =================================================
                    if args.trainer == "zo":
                        self.zo_update(model)
                    elif args.trainer == "zo_lowbit":
                        self.lowbit_zo_update(model)
                        self.zo_forward(model, inputs) # LoRA
                    elif args.trainer == "zo_lowbit_ft":
                            self.lowbit_zo_ftupdate(model)
                    else:
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                            # deepspeed does its own clipping

                            if self.do_grad_scaling:
                                # Reduce gradients first for XLA
                                if is_torch_tpu_available():
                                    gradients = xm._fetch_gradients(self.optimizer) # type: ignore
                                    xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size()) # type: ignore
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.optimizer)

                            if is_sagemaker_mp_enabled() and args.fp16:
                                self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                model.clip_grad_norm_(args.max_grad_norm)
                            # else:
                            #     # Revert to normal clipping otherwise, handling Apex or full precision
                            #     nn.utils.clip_grad_norm_(
                            #         amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                            #         args.max_grad_norm,
                            #     )
                            # Quantize gradients
                        # if hasattr(model, "parameters"):
                        #     print("fo gradient quantization")
                        #     quantize_gradients(model.parameters(), bits=8)  # Adjust bits as necessary
                        if hasattr(model, "parameters") and self.args.trainer == "regular" and getattr(self.args, "fo_quant_grad", False):
                            # 获取原始FP32梯度
                            # 这里在 regular 分支下额外做了一件事：
                            # 先保存原始梯度，再把梯度量化，随后统计量化误差。
                            # 这明显是作者为了分析 STE / 梯度量化误差而加的实验代码。
                            original_grads = {}
                            for name, param in model.named_parameters():
                                if param.requires_grad and param.grad is not None:
                                    original_grads[name] = param.grad.clone()
                            
                            # 执行梯度量化（STE）
                            print("Computing STE gradient error with bits=", getattr(self.args, "fo_quant_bits", self.args.wbit))
                            quantize_gradients(model.parameters(), bits=getattr(self.args, "fo_quant_bits", self.args.wbit))  # 使用显式配置的 FO 梯度量化比特数
                            
                            # 计算量化后的梯度误差
                            total_mse = 0.0
                            total_angle_error = 0.0
                            layer_errors = {}
                            
                            for name, param in model.named_parameters():
                                # if param.requires_grad and param.grad is not None and name in original_grads:
                                    # 原始梯度和量化后的梯度
                                    orig_grad = original_grads[name]
                                    quant_grad = param.grad
                                    
                                    # 1. 计算MSE误差
                                    mse = torch.mean((orig_grad - quant_grad) ** 2).item()
                                    
                                    # 2. 计算角度误差（梯度方向变化）
                                    orig_norm = torch.norm(orig_grad)
                                    quant_norm = torch.norm(quant_grad)
                                    
                                    if orig_norm > 0 and quant_norm > 0:
                                        cos_sim = torch.sum(orig_grad * quant_grad) / (orig_norm * quant_norm)
                                        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
                                        angle_error = torch.acos(cos_sim).item() * (180.0 / 3.14159)  # 角度制
                                    else:
                                        angle_error = 0.0
                                    
                                    # 3. 按层类型分组
                                    layer_type = name.split('.')[1] if len(name.split('.')) > 1 else 'other'
                                    if layer_type not in layer_errors:
                                        layer_errors[layer_type] = {'mse': [], 'angle': []}
                                    
                                    layer_errors[layer_type]['mse'].append(mse)
                                    layer_errors[layer_type]['angle'].append(angle_error)
                                    
                                    # 累计总误差
                                    total_mse += mse
                                    total_angle_error += angle_error
                                    
                                    # 记录每层误差
                                    self.log({
                                        f"ste_error/{name}_mse": mse,
                                        f"ste_error/{name}_angle": angle_error,
                                        f"ste_error/{name}_orig_norm": orig_norm.item(),
                                        f"ste_error/{name}_quant_norm": quant_norm.item()
                                    })
                            
                            # 计算平均误差
                            param_count = len(original_grads)
                            avg_mse = total_mse / param_count if param_count > 0 else 0
                            avg_angle = total_angle_error / param_count if param_count > 0 else 0
                            
                            # 记录总体误差
                            self.log({
                                "ste_total/avg_mse": avg_mse,
                                "ste_total/avg_angle_error": avg_angle,
                                "ste_total/bits": getattr(self.args, "fo_quant_bits", self.args.wbit)
                            })
                        # Optimizer step
                        optimizer_was_run = True
                        if self.deepspeed:
                            pass  # called outside the loop
                        # elif is_torch_tpu_available():
                        #     if self.do_grad_scaling:
                        #         self.scaler.step(self.optimizer)
                        #         self.scaler.update()
                        #     else:
                        #         xm.optimizer_step(self.optimizer)
                        elif self.do_grad_scaling:
                                scale_before = self.scaler.get_scale()
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                                scale_after = self.scaler.get_scale()
                                optimizer_was_run = scale_before <= scale_after
                        else:
                            self.optimizer.step()

                        if optimizer_was_run and not self.deepspeed:
                            self.lr_scheduler.step()
                        model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    if self.state.global_step % 1000 == 0 :
                            self.args.learning_rate /=5
                            print(self.state.global_step)
                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            # if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            #     if is_torch_tpu_available():
            #         # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            #         xm.master_print(met.metrics_report())
            #     else:
            #         logger.warning(
            #             "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
            #             "configured. Check your training configuration if this is unexpected."
            #         )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            # if is_torch_tpu_available():
            #     xm.rendezvous("load_best_model_at_end")
            # elif args.local_rank != -1:
            #     dist.barrier()
            # elif is_sagemaker_mp_enabled():
            #     smp.barrier()

            if args.local_rank != -1:
                dist.barrier()  # Make sure only the first process that found the best model checkpoint will load it.
            # elif is_sagemaker_mp_enabled():
            #     smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step
        # 训练结束后，除了常规 loss / speed metrics，
        # 这里还统计了 CUDA 峰值显存，便于对比 QuZO 和 FO 的内存开销。
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss


        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)


    ############## MeZO ##############




    # ----------------------------- 低比特 masked 扰动 -----------------------------
    # 这个函数会：
    #   1) 按随机种子生成全精度高斯扰动
    #   2) 把扰动量化成低比特
    #   3) 反量化回浮点近似值
    #   4) 乘上 zo_eps 并原地加到参数上
    # 它和普通 zo_perturb_parameters 的区别在于：这里强调“先量化扰动再加到参数”。
    # ---------------------------------------------------------------------------
    def zo_lowbitperturb_maskparameters(self, random_seed=None, scaling_factor=1):
        random_seed = self.zo_random_seed
        max_memory_fp = 0
        max_memory_quantized = 0
        max_memory_sparse = 0


        for name, param in self.named_parameters_to_optim:
 

            torch.manual_seed(random_seed)

            fp_perturb = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

            # fp_memory = fp_perturb.numel() * fp_perturb.element_size()
            # max_memory_fp = max(max_memory_fp, fp_memory)


            if self.args.quantized_perturb_ours == True:
                quantized_perturb, s, z = zo_quant(fp_perturb,nbits=self.args.perturb_bits,seed=random_seed+4,stochastic=True)
                # mask = torch.rand_like(quantized_perturb) >= (90 / 100)
                # quantized_perturb = quantized_perturb *mask 
                dequantized_perturb = zo_dequant(quantized_perturb,s ,z) 

            else:
                quantized_perturb, s, z = zo_quant(fp_perturb,nbits=self.args.perturb_bits,stochastic=True)
                # mask = torch.rand_like(quantized_perturb) >= (90 / 100)
                # quantized_perturb = quantized_perturb *mask 
  
                dequantized_perturb = zo_dequant(quantized_perturb,s ,z) 

            param.data = param.data + scaling_factor * dequantized_perturb * self.args.zo_eps

            random_seed += 2




    # ----------------------------- 低比特扰动（主版本） -----------------------------
    # 这是 lowbit_zo_step / lowbit_zo_update 依赖的核心函数之一。
    # 与标准 zo_perturb_parameters 不同，它不会直接把全精度高斯噪声加到参数上，
    # 而是先把噪声量化到 self.args.perturb_bits，再反量化成低比特近似值。
    # 这样更贴近 QuZO / 低比特 ZO 的实验设定。
    # -----------------------------------------------------------------------------
    def zo_lowbitperturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        random_seed = self.zo_random_seed
        
        for name, param in self.named_parameters_to_optim:
            torch.manual_seed(random_seed)

            if torch.isnan(param.data).any():
                print("NaN detected in param! Stopping execution.")
                print(name)
                sys.exit(1)
            fp_perturb = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            
            if self.args.quantized_perturb_ours == True:
                quantized_perturb, s, z = zo_quant(fp_perturb,nbits=self.args.perturb_bits,seed=random_seed+4,stochastic=True)
                # mask = torch.rand_like(quantized_perturb) >= (getattr(self.args, 'mask_ratio', 50) / 100)
                # quantized_perturb = quantized_perturb * mask
                dequantized_perturb = zo_dequant(quantized_perturb,s ,z) 
            else:
                quantized_perturb, s, z = zo_quant(fp_perturb,nbits=self.args.perturb_bits,stochastic=False)
                # mask = torch.rand_like(quantized_perturb) >= (getattr(self.args, 'mask_ratio', 50) / 100)
                # quantized_perturb = quantized_perturb * mask
                dequantized_perturb = zo_dequant(quantized_perturb,s ,z) 
            
            param.data = param.data + scaling_factor * dequantized_perturb * self.args.zo_eps

            random_seed += 2


         
            
            





    # ----------------------------- 标准 MeZO 扰动 -----------------------------
    # 这是“非低比特版本”的扰动函数：
    #   - 直接采样全精度高斯 z
    #   - 按 theta <- theta + scaling_factor * z * eps 原地改参数
    # 它对应标准 MeZO / ZO-SGD 的对称差分估计。
    # ----------------------------------------------------------------------
    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        total_pertub32_memory = 0
        for name, param in self.named_parameters_to_optim:
            # z = torch.bernoulli(0.5*torch.ones(size=param.data.size(), device=param.data.device, dtype=torch.int8))
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            num_elements = z.numel()  # 元素总数
            dtype_size = z.element_size()  # 每个元素的字节数
            memory_in_bytes = num_elements * dtype_size
            total_pertub32_memory += memory_in_bytes
            # self.log({f"mezo/{name}_perturbation_mean": z.mean().item()})
            # self.log({f"mezo/{name}_perturbation_std": z.std().item()})
            param.data = param.data + scaling_factor * z * self.args.zo_eps
        # self.perturb_memory_gb = total_pertub32_memory / (1024 ** 3)
        # self.log({f"mezo/scale_{scaling_factor}_perturbation_mem(GB)": self.perturb_memory_gb})
        



    # ----------------------------- 无反传前向 -----------------------------
    # 零阶方法的关键就是：只做 forward，不保留 backward graph。
    # 这里会：
    #   - model.eval() 关掉 dropout
    #   - 用 torch.inference_mode() 只算 loss
    #   - 如果是 non_diff 目标，则转到 zo_forward_nondiff
    # -------------------------------------------------------------------
    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()



    # ----------------------------- 不可导目标的前向 -----------------------------
    # 当前只支持 SQuAD。
    # 这里不是直接算交叉熵，而是让模型 generate 出答案，再用 F1 当指标。
    # 最后返回 -mean(F1) 作为优化目标。
    # 这说明该框架不仅能处理可导 loss，也能处理任务级别的不可导指标。
    # ------------------------------------------------------------------------
    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]
        
        return -torch.tensor(np.mean(f1s), dtype=torch.float32)




    # ----------------------------- 低比特 FT 风格的 ZO 估计 -----------------------------
    # 流程与标准 MeZO 类似：
    #   +eps 扰动 -> loss1
    #   -eps 扰动 -> loss2
    #   projected_grad = (loss1 - loss2) / (2 * eps)
    # 但这里用的是“低比特 masked 扰动”。
    # -------------------------------------------------------------------------------
    def lowbit_zo_ftstep(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize 
        self.named_parameters_to_optim = []


        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))


        # 分布式场景下由 rank0 采样随机种子并广播，确保所有进程使用同一扰动方向
        self.zo_random_seed = self._broadcast_seed()

        # First function evaluation
        self.zo_lowbitperturb_maskparameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        self.zo_lowbitperturb_maskparameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)

        # 所有 rank 上先各自得到标量估计，再做均值同步，保证后续参数更新保持一致
        self.projected_grad = self._all_reduce_mean_scalar((loss1 - loss2) / (2 * self.args.zo_eps))
        # self.projected_grad, scale, zero = zo_quant(self.projected_grad ,nbits=8,stochastic=False,sym=True)
        # self.projected_grad = zo_dequant(self.projected_grad,scale, zero).item()
        # Compute max absolute value
        # max_abs_value = torch.max(torch.abs(self.projected_grad))

        # # Calculate scaling factor
        # s = max_abs_value / 127
        #         # Avoid division by zero
        # if s == 0:
        #     s = 1e-6
        # quantized_grad = torch.round(self.projected_grad / s)
        # quantized_grad = torch.clamp(quantized_grad, -127, 127)
        # self.projected_grad = quantized_grad * s
        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        # Reset model back to its parameters at start of step
        self.zo_lowbitperturb_maskparameters(scaling_factor=1)
        
        return loss1


    # ----------------------------- 低比特 ZO 主估计器 -----------------------------
    # 这是这份文件最值得精读的函数之一。
    # 它支持多次 query（self.args.num_pertub），每次都会：
    #   1) 采样随机种子
    #   2) 用低比特扰动做 +eps 前向
    #   3) 再做 -eps 前向
    #   4) 用对称差分得到一个标量 projected_grad
    #   5) 再把这个标量本身量化到近似 int8 范围（[-127, 127]）
    #   6) 保存随机种子和 projected_grad，供 update 阶段重建同一方向
    # --------------------------------------------------------------------------
    def lowbit_zo_step(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize 
        self.named_parameters_to_optim = []


        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        iterations = self.args.num_pertub
        projected_grad_list = []
        zo_random_seed_list = []
        for i in range(iterations):
            # 分布式场景下使用同一个随机种子，保证各 rank 扰动方向完全一致
            self.zo_random_seed = self._broadcast_seed()
            zo_random_seed_list.append(self.zo_random_seed)

            # First function evaluation
            self.zo_lowbitperturb_parameters(scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)

            # Second function evaluation
            self.zo_lowbitperturb_parameters(scaling_factor=-2)
            loss2 = self.zo_forward(model, inputs)

            projected_grad = self._all_reduce_mean_scalar((loss1 - loss2) / (2 * self.args.zo_eps))
            max_abs_value = torch.max(torch.abs(projected_grad))

            # Calculate scaling factor
            s = max_abs_value / 127
            if s == 0:
                s = 1e-3  # 使用更大的保护值
            quantized_grad = torch.round(projected_grad / s)
            quantized_grad = torch.clamp(quantized_grad, -127, 127)
            self.projected_grad = quantized_grad * s
            projected_grad_list.append(self.projected_grad)

            # Reset model back to its parameters at start of step
            self.zo_lowbitperturb_parameters(scaling_factor=1)
            
        self.zo_random_seed = zo_random_seed_list
        self.projected_grad = projected_grad_list
        return loss1


    # ----------------------------- 标准 MeZO 梯度估计 -----------------------------
    # 这里实现的是最经典的对称差分：
    #   loss(theta + eps*z) - loss(theta - eps*z)
    #   -----------------------------------------  * z
    #                    2 * eps
    # 需要注意：这个函数只估计投影梯度，并不直接改参数。
    # --------------------------------------------------------------------------
    def zo_step(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize 
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        # 分布式场景下由 rank0 采样随机种子并广播，确保所有进程使用同一扰动方向
        self.zo_random_seed = self._broadcast_seed()

        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)

        # 多卡数据并行时，每个 rank 基于不同数据切片得到一个 projected_grad；
        # 这里做 all-reduce 求均值，保证所有 rank 后续应用同一个更新量。
        self.projected_grad = self._all_reduce_mean_scalar((loss1 - loss2) / (2 * self.args.zo_eps)).item()


        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        # # Append gradient estimate for variance tracking
        # if not hasattr(self, "gradient_estimates"):
        #     self.gradient_estimates = []  # Initialize gradient list

        # self.gradient_estimates.append(self.projected_grad)

        # # Compute and log gradient variance every 10 steps
        # if len(self.gradient_estimates) % 10 == 0:
        #     grad_variance = compute_gradient_variance(self.gradient_estimates)
        #     self.log({"gradient_variance/mezo": grad_variance.item()})

        # Reset model back to its parameters at start of step
        self.zo_perturb_parameters(scaling_factor=1)
        
        return loss1



    # ----------------------------- 标准 MeZO 参数更新 -----------------------------
    # 理论上，这里应该：
    #   1) 用相同随机种子重采样出同一个 z
    #   2) 执行 theta <- theta - lr * projected_grad * z
    # 但你要特别注意：这份代码里真正的 param.data 更新语句大多被注释掉了。
    # 当前有效执行的几乎只剩下 self.lr_scheduler.step()。
    # 所以如果你直接跑 trainer == 'zo'，要先确认这是不是作者有意为之。
    # --------------------------------------------------------------------------
    def zo_update(self, model):
        """
        Update the parameters with the estimated gradients.

        这是标准 MeZO / FP32-ZO 的真正更新步骤：
          theta <- theta - lr * projected_grad * z
        之前这里的 param.data 更新语句被注释掉了，导致 trainer=="zo" 基本没有真正训练。
        这里恢复成可执行版本，便于把它作为 MeZO baseline 使用。
        """
        args = self.args

        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - self._get_learning_rate() * (
                    self.projected_grad * z + args.weight_decay * param.data
                )
            else:
                param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)
        self.lr_scheduler.step()


    # ----------------------------- 低比特 ZO 参数更新（主版本） -----------------------------
    # 这是 lowbit_zo_step 的配套更新函数，也是这份文件最关键的部分之一。
    # 它会：
    #   1) 取出每次 query 保存下来的随机种子和 projected_grad
    #   2) 重新生成同一个高斯扰动
    #   3) 再次量化 / 反量化得到低比特扰动方向 z
    #   4) 执行 param <- param - lr * projected_grad * z
    #   5) 再把更新后的参数重新量化到低比特（wbit）
    #
    # 这一步才是真正让模型参数发生变化的地方。
    # -------------------------------------------------------------------------------
    def lowbit_zo_update(self, model):
        """
        Update the parameters with the estimated gradients.
        """
        args = self.args
        iterations = self.args.num_pertub

        for i in range(iterations):
            # 使用对应迭代的随机种子和梯度
            zo_random_seed_update = self.zo_random_seed[i]
            projected_grad = self.projected_grad[i]

            for name, param in self.named_parameters_to_optim:
                torch.manual_seed(zo_random_seed_update) 
                # Resample z
                fp_perturb = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

                if self.args.quantized_perturb_ours == True:
                    quantized_perturb, s1, z1 = zo_quant(fp_perturb, nbits=self.args.perturb_bits, seed=zo_random_seed_update+4, stochastic=True)
                    # mask = torch.rand_like(quantized_perturb) >= (getattr(self.args, 'mask_ratio', 50) / 100)
                    # quantized_perturb = quantized_perturb * mask
                    z = zo_dequant(quantized_perturb, s1, z1) 
                else:
                    quantized_perturb, s1, z1 = zo_quant(fp_perturb, nbits=self.args.perturb_bits, stochastic=False)
                    # mask = torch.rand_like(quantized_perturb) >= (getattr(self.args, 'mask_ratio', 50) / 100)
                    # quantized_perturb = quantized_perturb * mask
                    z = zo_dequant(quantized_perturb, s1, z1) 

                # if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                #     param.data = param.data - self._get_learning_rate() * (projected_grad * z + args.weight_decay * param.data)
                # else:
                #     param.data = param.data - self._get_learning_rate() * (projected_grad * z)
                # zo_random_seed_update += 2
                if self.args.quantized_perturb_ours == True:
                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                        param.data = param.data - self._get_learning_rate() * (projected_grad * z + args.weight_decay * param.data)
                        quantized_data, scaling, zero = zo_quant_data(param.data,nbits=self.args.wbit,stochastic=True, sym=True)
                        param.data = zo_dequant(quantized_data,scaling ,zero)    
                    else:
                            param.data = param.data - self._get_learning_rate() * projected_grad * z
                            quantized_data, scaling, zero = zo_quant_data(param.data,nbits=self.args.wbit,stochastic=True, sym=True)
                            param.data = zo_dequant(quantized_data,scaling ,zero)    
                else:       
                        if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                            max_bit = max(self.args.wbit,self.args.perturb_bits)
                            zo_grad1 = self._get_learning_rate() * (projected_grad * z + args.weight_decay * param.data)
                            param.data = param.data - zo_grad1
                            quantized_data, scaling, zero = zo_quant_data(param.data,nbits=max_bit,stochastic=False, sym=True)
                            param.data = zo_dequant(quantized_data,scaling ,zero)  

                        else:
                            param.data = param.data - self._get_learning_rate() * projected_grad * z
                            quantized_data, scaling, zero = zo_quant_data(param.data,nbits=self.args.wbit,stochastic=False, sym=True)
                            param.data = zo_dequant(quantized_data,scaling ,zero)     
                zo_random_seed_update += 2
        self.lr_scheduler.step()

    

    # ----------------------------- 低比特 FT 风格更新 -----------------------------
    # 这是与 lowbit_zo_ftstep 对应的更新函数。
    # 和 lowbit_zo_update 相比，它更直接，后半段没有显式再做一轮参数量化，
    # 更像“低比特扰动 + 浮点参数更新”的折中实现。
    # ---------------------------------------------------------------------------
    def lowbit_zo_ftupdate(self, model):
        """
        Update the parameters with the estimated gradients.
        """
        args = self.args

        # Reset the random seed for sampling zs
        # torch.manual_seed(self.zo_random_seed)     
        zo_random_seed_update = self.zo_random_seed

        for name, param in self.named_parameters_to_optim:
            torch.manual_seed(zo_random_seed_update) 
            # Resample z
            fp_perturb = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

            if self.args.quantized_perturb_ours == True:
                quantized_perturb, s1, z1 = zo_quant(fp_perturb, nbits=self.args.perturb_bits, seed=zo_random_seed_update+4, stochastic=True)
                # mask = torch.rand_like(quantized_perturb) >= (90 / 100)
                # quantized_perturb = quantized_perturb * mask
                z = zo_dequant(quantized_perturb, s1, z1) 
            else:
                quantized_perturb, s1, z1 = zo_quant(fp_perturb, nbits=self.args.perturb_bits, stochastic=True)
                # mask = torch.rand_like(quantized_perturb) >= (90 / 100)
                # quantized_perturb = quantized_perturb * mask
                z = zo_dequant(quantized_perturb, s1, z1) 



            # if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
            #     max_bit = max(self.args.wbit,self.args.perturb_bits)
            #     zo_grad1 = self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)

            #     quantized_grad, dequant_grad,s1 = stochastic_quantize(zo_grad1,max_bit)
            #     param.data = param.data - dequant_grad
            #     # quantized_grad, dequant_grad,s1 = stochastic_quantize(zo_grad1,max_bit)
            #     # param.data = param.data - zo_grad1
   
            #     # quantized_data, scaling, zero = zo_quant_data(param.data,nbits=max_bit,stochastic=True, sym=True)
            #     # param.data = zo_dequant(quantized_data,scaling ,zero)  

            # else:
            #     param.data = param.data - self._get_learning_rate() * self.projected_grad * z
            #     quantized_data, scaling, zero = zo_quant_data(param.data,nbits=self.args.wbit,stochastic=True, sym=True)
            #     param.data = zo_dequant(quantized_data,scaling ,zero)    


            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
 

                param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
            else:
                param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)
            zo_random_seed_update += 2
        self.lr_scheduler.step()
    ############## Misc overload functions ##############



    # ----------------------------- Trainer 输入字段适配 -----------------------------
    # HuggingFace Trainer 默认只保留 forward 需要的字段。
    # 这里额外把 gold 也留下来，目的是支持 SQuAD 这类 non_diff 目标，
    # 因为 zo_forward_nondiff 计算 F1 时需要 gold answer。
    # -----------------------------------------------------------------------------
    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    

    # ----------------------------- 自定义保存逻辑 -----------------------------
    # 这里重载 save_model，主要是为了解决 FSDP / DeepSpeed / TPU 等场景下
    # 直接 state_dict() 可能导致 OOM 或保存不完整的问题。
    # 所以这部分更多是“工程稳健性”代码，而不是算法核心。
    # ------------------------------------------------------------------------
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM) 
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10: # type: ignore
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
