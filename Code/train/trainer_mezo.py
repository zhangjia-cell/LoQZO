# ==============================
# 中文说明：
# 这个文件是在 HuggingFace Trainer 基础上二次改写的自定义 Trainer。
# 它主要做了四件事：
# 1) 保留原生 Trainer 的大部分训练/保存/分布式逻辑；
# 2) 增加了一些量化辅助函数（如 stochastic_quantize、quantize_gradients）；
# 3) 在 _inner_training_loop 中加入了 zeroth-order / MeZO 分支；
# 4) 对 save_model 做了重载，修复 FSDP 保存时可能的 OOM 问题。
#
# 你可以把它理解成：
# “一个把 HuggingFace Trainer 改造成支持 MeZO / 量化实验的训练器”。
# ==============================

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

from packaging import version
from transformers.integrations import is_sagemaker_mp_enabled
from transformers.integrations import is_fairscale_available
from transformers.utils.versions import dep_version_check

from transformers.integrations import is_sagemaker_mp_enabled
# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    # default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)
from transformers.integrations import (  # isort: split
       # default_hp_search_backend,
       get_reporting_integration_callbacks,
       hp_params,
       is_fairscale_available,  # This line is causing the error
       is_optuna_available,
       is_ray_tune_available,
       is_sigopt_available,
       is_wandb_available,
       run_hp_search_optuna,
       run_hp_search_ray,
       run_hp_search_sigopt,
       run_hp_search_wandb,
   )
import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD, Adam
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
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_less_than_1_11
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
    # default_hp_space,
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


# _is_native_cpu_amp_available = is_torch_greater_or_equal_than_1_10

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
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

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler


# if is_sagemaker_mp_enabled():
#     import smdistributed.modelparallel.torch as smp
#     from smdistributed.modelparallel import __version__ as SMP_VERSION

#     IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

#     from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
# else:
#     IS_SAGEMAKER_MP_POST_1_10 = False

from transformers.integrations import is_sagemaker_mp_enabled  

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp  # type: ignore
    from smdistributed.modelparallel import __version__ as SMP_VERSION  # type: ignore

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
    
    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat  # type: ignore
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

# ------------------------------
# 工具函数：估算一个 batch 的输入张量大概占多少显存/内存。
# 这里只统计 inputs 字典中直接出现的 Tensor，不会递归统计更复杂的嵌套结构。
# ------------------------------
def calculate_inputs_memory(inputs):
    total_memory = 0
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            total_memory += value.numel() * value.element_size()
        # 如果有列表或其他结构包含 tensor，也可以做进一步处理
    return total_memory

# ------------------------------
# 工具函数：对任意张量做“随机舍入量化（stochastic rounding）”。
# 注意：这里返回的是“反量化后的浮点张量”，也就是：
#   先量化到低比特整数表示 -> 再乘 scale 恢复到浮点数。
# 所以它更像是在“模拟低比特噪声/低比特效果”，
# 而不是把张量真的永久存成 int8/int4 类型。
# ------------------------------
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
    max_abs_value = torch.max(torch.abs(tensor))
    scale = max_abs_value / (2**(bit_width - 1) - 1)  # Signed quantization

    # Avoid division by zero
    if scale == 0:
        scale = 1e-6

    # Normalize tensor to quantization range
    normalized_tensor = tensor / scale

    # Compute stochastic rounding
    lower = torch.floor(normalized_tensor)  # Floor value
    fractional = normalized_tensor - lower  # Fractional part
    stochastic = torch.bernoulli(fractional)  # Bernoulli(p=fractional)

    # Quantize with stochastic rounding
    quantized_tensor = lower + stochastic

    # Clamp to valid range
    quantized_tensor = torch.clamp(quantized_tensor, -(2**(bit_width - 1)), 2**(bit_width - 1) - 1)

    # Dequantize back to FP32
    dequantized_tensor = quantized_tensor * scale

    return dequantized_tensor

# ------------------------------
# 工具函数：对一组参数的梯度 param.grad.data 做随机量化。
# 这是“逐个参数遍历 + 原地修改梯度”的写法。
# 注意：函数名叫 quant_grad，但实际输入 x 应该是“参数迭代器/参数列表”。
# ------------------------------
def quant_grad(x, nbits=16):
    # Set the random seed if provided
    # seed = np.random.randint(1000000000)
    for param in x:
        if param.grad is not None:
            x = param.grad.data
    # Compute scaling factor
            max_abs_value = torch.max(torch.abs(x))
            scale = max_abs_value / (2**(nbits - 1) - 1)  # Signed quantization

    # Avoid division by zero
            if scale == 0:
                scale = 1e-6

    # Normalize tensor to quantization range
            normalized_tensor = x / scale

    # Compute stochastic rounding
            lower = torch.floor(normalized_tensor)  # Floor value
            fractional = normalized_tensor - lower  # Fractional part
            stochastic = torch.bernoulli(fractional)  # Bernoulli(p=fractional)

    # Quantize with stochastic rounding
            quantized_tensor = lower + stochastic

    # Clamp to valid range
            quantized_tensor = torch.clamp(quantized_tensor, -(2**(nbits - 1)), 2**(nbits - 1) - 1)

    # Dequantize back to FP32
            param.grad.data  = quantized_tensor * scale

# ------------------------------
# 工具函数：对参数本身 param.data 做随机量化。
# 与 quant_grad 类似，只不过改的是权重值而不是梯度。
# ------------------------------
def quant_data(x, nbits=16):
    # Set the random seed if provided
    # seed = np.random.randint(1000000000)
    for param in x:
        if param.data is not None:
            x = param.data
    # Compute scaling factor
            max_abs_value = torch.max(torch.abs(x))
            scale = max_abs_value / (2**(nbits - 1) - 1)  # Signed quantization

    # Avoid division by zero
            if scale == 0:
                scale = 1e-6

    # Normalize tensor to quantization range
            normalized_tensor = x / scale

    # Compute stochastic rounding
            lower = torch.floor(normalized_tensor)  # Floor value
            fractional = normalized_tensor - lower  # Fractional part
            stochastic = torch.bernoulli(fractional)  # Bernoulli(p=fractional)

    # Quantize with stochastic rounding
            quantized_tensor = lower + stochastic

    # Clamp to valid range
            quantized_tensor = torch.clamp(quantized_tensor, -(2**(nbits - 1)), 2**(nbits - 1) - 1)

    # Dequantize back to FP32
            param.data  = quantized_tensor * scale
    
    # Dequantize the result
    # qx = (x_quant - zero_point) / scaling_factor
    

# def quantize_gradients(parameters, bits=4):
#     """
#     Quantize gradients to a specified number of bits.
#     """
#     # scale = 2**bits - 1
#     for param in parameters:
#         if param.grad is not None:
#             grad = param.grad.data
#             min_val, max_val = grad.min(), grad.max()
#             scale = max_val / (2**(bits - 1) - 1)  # Signed quantization
#             if scale == 0:
#                 scale = 1e-6  
#             grad = grad / scale  
#             grad = torch.clamp(grad, -(2**(bits - 1)), 2**(bits - 1) - 1)
#             grad = grad * scale
#             param.grad.data = grad  # Rescale
# ------------------------------
# 另一版梯度量化函数：这里不是随机舍入，而是更简单的 clamp + rescale。
# 可以把它理解成一个更朴素的低比特梯度近似版本。
# ------------------------------
def quantize_gradients(parameters, bits=8):
    """
    Quantize gradients to a specified number of bits.
    """
    # scale = 2**bits - 1
    for param in parameters:
        if param.grad is not None:
            # print("quantize_gradients")
            grad = param.grad.data
            min_val, max_val = grad.min(), grad.max()
            # grad = (grad - min_val) / (max_val - min_val)  # Normalize to 0-1
            scale = max_val / (2**(bits - 1) - 1)
            # if scale == 0:
            #     scale = 1e-24
            grad = torch.clamp(grad / scale, -(2**(bits - 1)), 2**(bits - 1) - 1)
            # grad = torch.round(grad / scale) * scale  # Quantize and dequantize
            param.grad.data = grad* scale   # Rescale

# ------------------------------
# 另一版参数量化函数：对 param.data 做 clamp + rescale。
# ------------------------------
def quantize_data(parameters, bits=8):
    """
    Quantize gradients to a specified number of bits.
    """
    # scale = 2**bits - 1
    for param in parameters:
        if param.data is not None:
            grad = param.data
            min_val, max_val = grad.min(), grad.max()
            # grad = (grad - min_val) / (max_val - min_val)  # Normalize to 0-1
            scale = max_val / (2**(bits - 1) - 1)
            if scale == 0:
                scale = 1e-24
            grad = torch.clamp(grad / scale, -(2**(bits - 1)), 2**(bits - 1) - 1)
            # grad = torch.round(grad / scale) * scale  # Quantize and dequantize
            param.data = grad* scale   # Rescale

# =====================================================================
# 核心类：OurTrainer
# 继承自 HuggingFace 的 Trainer，并重写了内部训练循环 _inner_training_loop。
#
# 这个类最关键的地方有三块：
# 1) 训练主循环 _inner_training_loop：决定每一步到底走 FO 还是 ZO；
# 2) zo_step / zo_update：MeZO 风格的零阶梯度估计与参数更新；
# 3) save_model：兼容 FSDP / DeepSpeed / TPU 等保存逻辑。
#
# 非常重要：
# 从这份代码来看，这里实现的是“MeZO/ZO-SGD 风格”的对称差分零阶更新；
# 它还没有体现 QuZO 论文里 Algorithm 1 的“双独立量化扰动 u_{i,1}, u_{i,2}”。
# 也就是说，这份代码更接近 MeZO 基础版 Trainer，而不是论文里最完整的 QuZO 核心。
# =====================================================================
class OurTrainer(Trainer):

    from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    # -----------------------------------------------------------------
    # 训练主循环：这是整个 Trainer 最重要的函数。
    # 它覆盖了 HuggingFace 原始训练循环，以便插入：
    # - linear probing
    # - zeroth-order / MeZO 分支
    # - 自定义优化器与保存逻辑
    # -----------------------------------------------------------------
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

        # ========================
        # 线性探测（linear probing）分支
        # 思路：
        # 1) 不更新整个模型；
        # 2) 先把训练集样本过一遍，抽取 lm_head 之前的特征；
        # 3) 再用 sklearn 的 LogisticRegressionCV 在 CPU 上拟合分类器；
        # 4) 最后把拟合出的系数直接写回 decoder / lm_head。
        # 这是一个非常省训练成本的 baseline。
        # ========================
        # MeZO added: Linear probing
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

        # ========================
        # 下面开始是“正常训练模式”的初始化部分。
        # 这里会计算：
        # - 每个 epoch 多少 step
        # - 总共训练多少步
        # - 总样本数 / 总 batch size
        # 这些量后面会用于日志、lr scheduler、恢复训练等。
        # ========================
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
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

        # 调试分支：如果开启 underflow/overflow 检查，就注册数值监控。
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

        # 判断是否需要“延迟创建优化器”。
        # 在某些分布式/分片训练场景（如 sharded_ddp / fsdp / sagemaker mp）下，
        # 需要先 wrap 模型，再创建优化器。
        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        # DeepSpeed 分支：如果用了 deepspeed，就由 deepspeed 负责初始化模型、优化器和调度器。
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

        # TrainerState：HuggingFace Trainer 用它记录 global_step、epoch、best checkpoint 等状态。
        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # 如果开启 gradient checkpointing，就在模型上启用，以节省激活显存。
        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # _wrap_model 会根据当前环境把模型包装成 DDP / FSDP / deepspeed 等外层形式。
        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # 如果从 checkpoint 恢复，这里会把优化器和学习率调度器状态一并恢复。
        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        # 这里额外支持 args.optimizer == 'sgd' 的分支，强行把 optimizer 换成 SGD。
        if args.optimizer == "sgd":
                self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)
        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        # 训练开始前的日志输出：打印样本数、epoch 数、batch size、总 step、可训练参数量等。
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
        # 下面这块处理“从 checkpoint 恢复训练”的逻辑：
        # 包括恢复 epoch、global_step，以及跳过已经训练过的数据 batch。
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
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

            # ========================
            # 进入 epoch 循环
            # 每个 epoch 会：
            # 1) 设置 distributed sampler 的 epoch；
            # 2) 准备当前 epoch 的 dataloader/iterator；
            # 3) 进入 step 循环执行训练。
            # ========================
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            # if is_torch_tpu_available():
            #     parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
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
                # ------------------------
                # 进入 step 级别训练循环
                # 这是每一个 batch 真正执行训练逻辑的地方。
                # ------------------------
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

                # ========================
                # 关键分支：选择 FO 还是 ZO
                # - 如果 args.trainer == 'zo'：走零阶优化，调用 zo_step()
                # - 否则：走标准 HuggingFace training_step()，即一阶反向传播
                # ========================
                # MeZO added: estimate gradient
                if args.trainer == "zo":
                    tr_loss_step = self.zo_step(model, inputs)
                else:
                    # before_grad_memory = sum(param.grad.element_size() * param.grad.nelement() for param in model.parameters() if param.grad is not None)
                    # with torch.profiler.profile(
                    #     activities=[
                    #         torch.profiler.ProfilerActivity.CPU,
                    #         torch.profiler.ProfilerActivity.CUDA,
                    #     ],
                    #     profile_memory=True,
                    #     with_stack=True
                    # ) as prof:
                    #     before_grad_memory = sum(param.grad.element_size() * param.grad.nelement() for param in model.parameters() if param.grad is not None)
                    #     before_grad_memory_gb = before_grad_memory / (1024 ** 3)  # Convert bytes to GB
                        # print(self.optimizer)

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
                        # sequence_length = inputs["input_ids"].shape[1]
                        # print("sequence length",sequence_length)
                        # total_inputs_memory = calculate_inputs_memory(inputs)
                        # print("Total inputs memory consumption: {:.2f} MB".format(total_inputs_memory / (1024 ** 2)))
                        # after_grad_memory = sum(param.grad.element_size() * param.grad.nelement() for param in model.parameters() if param.grad is not None)
                        # after_grad_memory_gb = after_grad_memory / (1024 ** 3)  # Convert bytes to GB
                        # print(f"Gradient Memory Before Backward Pass: {before_grad_memory} Bytes({before_grad_memory_gb:.3f} GB)")
                        # print(f"Gradient Memory After Backward Pass: {after_grad_memory} Bytes({after_grad_memory_gb:.3f} GB)")
                # 处理 loss 为 nan / inf 的情况，避免训练日志直接炸掉。
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
                # prof.export_chrome_trace("memory_trace.json")
                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                    # 当达到 gradient accumulation 的更新边界时，才真正做一次参数更新。
                    # 在 ZO 模式下，这里调用 zo_update(model)；
                    # 在 FO 模式下，这里执行 grad clipping + optimizer.step() + lr_scheduler.step()。
                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # ZO 分支：前面的 zo_step 只负责“估计 projected gradient”，
                    # 真正的参数更新在这里完成。
                    # MeZO added: update model with the estimated gradient
                    if args.trainer == "zo":
                        self.zo_update(model)
                    else:
                        # FO 分支：标准梯度裁剪逻辑。
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                            # deepspeed does its own clipping

                            if self.do_grad_scaling:
                                # Reduce gradients first for XLA
                                # if is_torch_tpu_available():
                                #     gradients = xm._fetch_gradients(self.optimizer)
                                #     xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
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


                   
                                
                       
                        # FO 分支：真正的 optimizer.step() 在这里发生。
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

                    # 完成一次参数更新后，刷新 TrainerState，并触发 log/save/eval 回调。
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

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

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report()) # type: ignore
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        # 训练结束后的收尾工作：
        # - 清理缓存状态
        # - 如有需要，加载 best checkpoint
        # - 汇总 train_loss / FLOPs / speed 等指标
        # - 清理旧 checkpoint
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end") # type: ignore
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

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


    # =================================================================
    # 以下是 MeZO / Zeroth-Order 相关函数
    # 这是这份文件最值得你精读的地方。
    #
    # 调用关系如下：
    #   _inner_training_loop -> zo_step() -> zo_perturb_parameters() / zo_forward()
    #                                         再回到 _inner_training_loop -> zo_update()
    #
    # 整体逻辑是：
    # 1) 采样随机方向 z；
    # 2) 分别计算 f(theta + eps*z) 和 f(theta - eps*z)；
    # 3) 用对称差分估计一个标量 projected_grad；
    # 4) 再用同一个随机方向 z 去更新参数。
    #
    # 这正是 MeZO 的核心思路。
    # =================================================================
    ############## MeZO ##############


    # 用随机向量 z 原地扰动参数。
    # 由于后面 update 时还要“复现同一个 z”，
    # 所以这里必须依赖同一个随机种子。
    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.args.zo_eps


    # 在“不构建反向图”的前提下做一次 forward，返回 loss。
    # 这就是 ZO/MeZO 节省显存的关键之一：只做 forward，不保留 backward graph。
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


    # 非可导目标分支：这里专门处理像 SQuAD F1 这种不能直接反传的指标。
    # 它通过 generate() 生成答案，再用 metrics.f1 计算分数，最后返回负 F1 作为“loss”。
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


    # zo_step 的职责不是直接更新参数，而是“估计梯度方向对应的标量系数”。
    # 公式对应标准对称差分：
    #   projected_grad = [f(theta + eps z) - f(theta - eps z)] / (2 eps)
    #
    # 然后把模型参数恢复回原位置，等待 zo_update() 真正执行更新。
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

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        # Reset model back to its parameters at start of step
        self.zo_perturb_parameters(scaling_factor=1)
        
        return loss1


    # 使用 zo_step 里同一个随机种子重新采样 z，
    # 从而保证这里的 z 与前面估计 projected_grad 时的 z 完全一致。
    #
    # 更新公式本质上是：
    #   theta <- theta - lr * (projected_grad * z)
    # 如果参数不是 bias/layernorm，还会附加 weight decay。
    def zo_update(self, model):
        """
        Update the parameters with the estimated gradients.
        """
        args = self.args

        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)     

        for name, param in self.named_parameters_to_optim:
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
            else:
                param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)

        self.lr_scheduler.step()


    ############## Misc overload functions ##############


    # 非可导目标训练时，需要把 gold 文本也保留到 dataloader / batch 里。
    # 所以这里重写 signature columns，把 'gold' 加进去。
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

    
    # 重写 save_model：主要是为了解决 FSDP / DeepSpeed 下直接保存可能 OOM 的问题。
    # 核心思路是：
    # - TPU / SageMaker / FSDP / DeepSpeed 各走各的保存分支；
    # - FSDP 时优先收集 full state dict 到 CPU；
    # - zero3 下删除无效权重文件，必要时走 deepspeed.save_checkpoint()。
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
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
            ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
            or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
            or self.fsdp is not None
        ):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            # Fix the FSDP loading bug
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = self.model.state_dict()
            # state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
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