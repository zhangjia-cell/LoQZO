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
import wandb
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

# import smdistributed.modelparallel.torch as smp
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

from transformers.deepspeed import (
    deepspeed_init,
    deepspeed_load_checkpoint,
    is_deepspeed_available,
    is_deepspeed_zero3_enabled
)
# from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.deepspeed import is_deepspeed_available
from transformers.deepspeed import is_deepspeed_zero3_enabled
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
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
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
    is_accelerate_available,
)
from transformers.utils.generic import ContextManagers
from accelerate.state import DistributedType

from transformers.integrations import is_wandb_available

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from Code.train.utils import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# if is_apex_available():
#    from apex import amp

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

# 已禁用 SageMaker 相关代码（本地运行无需使用）
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

def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    zero = 0
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_()

    return w, 1 / scales, zero

import torch

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

    # # Compute stochastic rounding
    # lower = torch.floor(normalized_tensor)  # Floor value
    # fractional = normalized_tensor - lower  # Fractional part
    # stochastic = torch.bernoulli(fractional)  # Bernoulli(p=fractional)

    # # Quantize with stochastic rounding
    # quantized_tensor = lower + stochastic
    quantized_tensor = torch.round(normalized_tensor)

    # Clamp to valid range
    quantized_tensor = torch.clamp(quantized_tensor, -(2**(bit_width - 1)), 2**(bit_width - 1) - 1)

    # Dequantize back to FP32
    dequantized_tensor = quantized_tensor * scale

    return dequantized_tensor
    
def quantize_gradients(parameters, bits=8):
    """
    Quantize gradients to a specified number of bits using symmetric linear quantization.
    """
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1

    for param in parameters:
        if param.grad is not None:
            grad = param.grad.data
            max_abs = torch.max(torch.abs(grad))
            scale = max_abs / qmax if max_abs > 0 else 1e-6  # avoid div-by-zero

            # Normalize, quantize, clamp
            normalized = grad / scale
            quantized = torch.round(normalized)
            quantized = torch.clamp(quantized, qmin, qmax)

            # Dequantize
            param.grad.data = quantized * scale


def zo_quant_data(x, nbits=16,blk_exp=True, sym=True, stochastic=True, seed=None):
    # Set the random seed if provided
    # seed = np.random.randint(1000000000)
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # n_levels = 2**nbits
    n_levels = 2**(nbits-1)
    
    if sym:  # symmetric quantization
        x1 = torch.max(torch.abs(x))
        x0 = -x1
    else:  # asymmetric quantization
        x0, x1 = torch.min(x), torch.max(x)
    
    # Calculate the scale factor a
    scaling_factor = n_levels / (x1 - x0)

    if blk_exp:  # block exponent
        scaling_factor = 2**torch.floor(torch.log2(scaling_factor))

    # Calculate the zero point b
    zero_point = torch.floor(-scaling_factor * (x0 + x1) / 2)

    if stochastic:
        # Apply stochastic rounding
        x_floor = torch.floor(x * scaling_factor)
        rest = x * scaling_factor - x_floor
        x_int = x_floor + torch.bernoulli(rest)  # Use random rounding based on the fractional part
    else:
        # Apply standard rounding
        x_int = torch.round(x * scaling_factor)


    x_quant = torch.clamp(x_int + zero_point, -n_levels, n_levels - 1)
    
    
    return x_quant,scaling_factor,zero_point  # Return the quantized and dequantized tensor

def zo_quant(x, nbits=4, sym=False, stochastic=True, seed=None):
    # Set the random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    
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
            scaler = 3 / x1
            x_scaled = x * scaler
            x_quant = torch.round(x_scaled).clamp(-3, 3)  # Clamp to the 4 levels
            zero_point = 0
        else:
            # For unsigned quantization, levels could be 0, 1, 2, 3
            x_min, x_max = x.min(), x.max()
            scaler = 3 / (x_max - x_min)
            zero_point = torch.round(-scaler * x_min)
            x_scaled = x * scaler + zero_point
            x_quant = torch.round(x_scaled).clamp(0, 3)
        return x_quant, scaler, zero_point

    else:
        # For 3-bit and above, use general quantization
        n_levels = 2 ** (nbits - 1) if sym else 2 ** nbits - 1
        
        if sym:
            x1 = x.abs().max()  # Max absolute value
            scaler = n_levels / x1
            zero_point = 0  # Center around 0 for symmetric quantization
        else:
            x_min, x_max = x.min(), x.max()
            scaler = n_levels / (x_max - x_min)
            zero_point = torch.round(-scaler * x_min)

        # Apply scaling and optional stochastic rounding
        x_scaled = x * scaler
        if stochastic:
            x_floor = torch.floor(x_scaled)
            rest = x_scaled - x_floor
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
def zo_dequant(x_quant, scaling_factor, zero_point):

    
    # Dequantize the result
    qx = (x_quant - zero_point) / scaling_factor
    # qx = (qx - x).detach() + x
    
    return qx  # Return the quantized and dequantized tensor

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

# Add this context manager near the top of the file, after imports but before the class definitions
@contextlib.contextmanager
def maybe_no_sync(model):
    """
    Context manager to handle models with or without no_sync method (used in distributed training)
    """
    if hasattr(model, "no_sync") and callable(model.no_sync):
        with model.no_sync():
            yield
    else:
        # Fallback for models without no_sync
        yield

class OurTrainer(Trainer):

    from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

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

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

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


        total_steps = 0
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
            for step, inputs in enumerate(epoch_iterator):
                total_steps += 1
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
                if args.trainer == "zo":
                    tr_loss_step = self.zo_step(model, inputs)
                elif args.trainer == "zo_lowbit":
                    tr_loss_step = self.lowbit_zo_step(model, inputs)
                else:
                    if (
                        ((step + 1) % args.gradient_accumulation_steps != 0)
                        and args.local_rank != -1
                        and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with maybe_no_sync(model):
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
                    if args.trainer == "zo":
                        self.zo_update(model)
                        grad_norm=None
                    elif args.trainer == "zo_lowbit":
                        self.lowbit_zo_update(model)
                        grad_norm=None
                    else:
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    # amp.master_params(self.optimizer),
                                    self.optimizer.param_groups,
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                    is_accelerate_available()
                                    and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm
                        # print("Computing STE gradient error with bits=", 4)
                        quantize_gradients(model.parameters(), bits=4)  # 使用模型的wbit参数
                            
                        # Optimizer step
                        self.optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    # if self.state.global_step % 1000 == 0 :
                    #         self.args.learning_rate /=5
                    #         print(self.state.global_step)
                    # # self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                    # wandb.log({"training_loss": tr_loss.item() / self.state.global_step}, step=self.state.global_step)
                    # self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

                # if self.args.eval_steps is not None and (total_steps + 1) % self.args.eval_steps == 0:
                #     print(f"=========================> Evaluating at step {total_steps + 1}... <=========================")
                #     # val_metrics = self.evaluate_func([], self.dev_samples)
                #     test_metrics = self.evaluate_func([], self.eval_samples)
                #     if "accuracy" in test_metrics:
                #         self.log({"test_acc": test_metrics["accuracy"]})
                #         wandb.log({"test_acc": test_metrics["accuracy"]})
                #     else:
                #         keys = list(test_metrics.keys())
                #         log_dict = {}
                #         for k in keys:
                #             log_dict['test_' + k] = test_metrics[k]
                #         self.log(log_dict)
                #         wandb.log(log_dict)

                #     max_memory_allocated = 0
                #     for device_id in range(torch.cuda.device_count()):
                #         # this is not accurate since max memory does not happen simultaneously across all devices
                #         max_memory_allocated += torch.cuda.max_memory_allocated(device_id)
                #     self.log({"peak_mem": max_memory_allocated / 1024 ** 3,
                #               "step_consumption": train_step_duration * 1000})
                #     wandb.log({"peak_mem": max_memory_allocated / 1024 ** 3,
                #                "step_consumption": train_step_duration * 1000})
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            grad_norm = None
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
            # self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

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
                dist.barrier()

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
        # wandb.log(metrics)

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
 

    def zo_lowbitperturb_maskparameters(self, random_seed=None, scaling_factor=1):
        random_seed = self.zo_random_seed
        max_memory_fp = 0
        max_memory_quantized = 0
        max_memory_sparse = 0


        for name, param in self.named_parameters_to_optim:
 

            torch.manual_seed(random_seed)

            fp_perturb = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

            fp_memory = fp_perturb.numel() * fp_perturb.element_size()
            max_memory_fp = max(max_memory_fp, fp_memory)


            if self.args.quantized_perturb_ours == True:
                quantized_perturb, s, z = zo_quant(fp_perturb,nbits=self.args.perturb_bits,seed=random_seed+4,stochastic=True)
                # Memory for quantized perturbation

                # nbits = self.args.perturb_bits
                # if nbits == 8:
                #     quantized_memory_bytes = quantized_perturb.numel() * (nbits // 8)
                # if nbits == 4:
                #     quantized_memory_bytes = quantized_perturb.numel() /2
                # max_memory_quantized = max(max_memory_quantized, quantized_memory_bytes)
                mask = torch.rand_like(quantized_perturb) >= (90 / 100)
                quantized_perturb = quantized_perturb *mask 
                # # Sparse memory (mask + quantized non-zero elements)
                # non_zero_elements = mask.sum().item()
                # sparse_memory = (mask.numel() * mask.element_size()) + (non_zero_elements * (nbits // 8))
                # max_memory_sparse = max(max_memory_sparse, sparse_memory)
                dequantized_perturb = zo_dequant(quantized_perturb,s ,z) 

            else:
                quantized_perturb, s, z = zo_quant(fp_perturb,nbits=self.args.perturb_bits,stochastic=True)
                # Memory for quantized perturbation
                # nbits = self.args.perturb_bits
                # if nbits == 8:
                #     quantized_memory_bytes = quantized_perturb.numel() * (nbits // 8)
                # if nbits == 4:
                #     quantized_memory_bytes = quantized_perturb.numel() /2
                # max_memory_quantized = max(max_memory_quantized, quantized_memory_bytes)
                mask = torch.rand_like(quantized_perturb) >= (90 / 100)
                quantized_perturb = quantized_perturb *mask 
                # # Sparse memory (mask + quantized non-zero elements)
                # non_zero_elements = mask.sum().item()
                # sparse_memory = (mask.numel() * mask.element_size()) + (non_zero_elements * (nbits // 8))
                # max_memory_sparse = max(max_memory_sparse, sparse_memory)
                dequantized_perturb = zo_dequant(quantized_perturb,s ,z) 

            param.data = param.data + scaling_factor * dequantized_perturb * self.args.zo_eps

            random_seed += 2
        # 转换为MB并记录最大内存消耗
        # max_memory_fp_mb = max_memory_fp / (1024 ** 2)
        # max_memory_quantized_mb = max_memory_quantized / (1024 ** 2)
        # max_memory_sparse_mb = max_memory_sparse / (1024 ** 2)

        # self.log({
        #     "max_memory/fp_perturbation_mb": max_memory_fp_mb,
        #     "max_memory/quantized_perturbation_mb": max_memory_quantized_mb,
        #     "max_memory/sparse_perturbation_mb": max_memory_sparse_mb,
        # })



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
                mask = torch.rand_like(quantized_perturb) >= (getattr(self.args, 'mask_ratio', 50) / 100)
                quantized_perturb = quantized_perturb * mask
                dequantized_perturb = zo_dequant(quantized_perturb,s ,z) 
            else:
                quantized_perturb, s, z = zo_quant(fp_perturb,nbits=self.args.perturb_bits,stochastic=True)
                mask = torch.rand_like(quantized_perturb) >= (getattr(self.args, 'mask_ratio', 50) / 100)
                quantized_perturb = quantized_perturb * mask
                dequantized_perturb = zo_dequant(quantized_perturb,s ,z) 
            
            param.data = param.data + scaling_factor * dequantized_perturb * self.args.zo_eps

            random_seed += 2
        # 转换为MB并记录最大内存消耗
        # max_memory_fp_mb = max_memory_fp / (1024 ** 1)
        # max_memory_quantized_mb = max_memory_quantized / (1024 ** 1)
        # max_memory_sparse_mb = max_memory_sparse / (1024 ** 1)

        # self.log({
        #     "max_memory/fp_perturbation_kb": max_memory_fp_mb,
        #     "max_memory/quantized_perturbation_kb": max_memory_quantized_mb,
        #     "max_memory/sparse_perturbation_kb": max_memory_sparse_mb,
        # })


         
            
            




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


        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        self.zo_lowbitperturb_maskparameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        self.zo_lowbitperturb_maskparameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps))
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
            # Sample the random seed for sampling z
            self.zo_random_seed = np.random.randint(1000000000)
            zo_random_seed_list.append(self.zo_random_seed)

            # First function evaluation
            self.zo_lowbitperturb_parameters(scaling_factor=1)
            loss1 = self.zo_forward(model, inputs)

            # Second function evaluation
            self.zo_lowbitperturb_parameters(scaling_factor=-2)
            loss2 = self.zo_forward(model, inputs)

            self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps))
            # Compute max absolute value
            max_abs_value = torch.max(torch.abs(self.projected_grad))

            # Calculate scaling factor
            s = max_abs_value / 127
            if s == 0:
                s = 1e-3  # 使用更大的保护值
            quantized_grad = torch.round(self.projected_grad / s)
            quantized_grad = torch.clamp(quantized_grad, -127, 127)
            self.projected_grad = quantized_grad * s
            projected_grad_list.append(self.projected_grad)
            # self.projected_grad, scale, zero = zo_quant(self.projected_grad ,nbits=4,stochastic=False,sym=True)
            # self.projected_grad = zo_dequant(self.projected_grad,scale, zero).item()


            # No gradient accumulation support
            assert self.args.gradient_accumulation_steps == 1

            # # Append gradient estimate for variance tracking
            # if not hasattr(self, "gradient_estimates"):
            #     self.gradient_estimates = []  # Initialize gradient list

            # self.gradient_estimates.append(self.projected_grad)

            # # Compute and log gradient variance every 10 steps
            # if len(self.gradient_estimates) % 10 == 0:
            #     grad_variance = compute_gradient_variance(self.gradient_estimates)
            #     self.log({"gradient_variance/lowbit": grad_variance.item()})

            # Reset model back to its parameters at start of step
            self.zo_lowbitperturb_parameters(scaling_factor=1)
            
        self.zo_random_seed = zo_random_seed_list
        self.projected_grad = projected_grad_list
        return loss1

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


    def zo_update(self, model):
        """
        Update the parameters with the estimated gradients.
        """
        args = self.args

        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)     

        for name, param in self.named_parameters_to_optim:
            original_data = param.data.clone()
            # Resample z
            # z = torch.bernoulli(0.5*torch.ones(size=param.data.size(), device=param.data.device, dtype=torch.int8))

            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            # if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
            #     param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
            # else:
            #     param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)
            # if "layers.0.self_attn" in name or "layers.15.self_attn" in name or "layers.30.self_attn" in name:

            #         kl_div = torch.sum(
            #             torch.abs(original_data) / torch.sum(torch.abs(original_data))
            #             * torch.log(
            #             (torch.abs(original_data) + 1e-10)
            #             / (torch.abs(param.data) + 1e-10)
            #             )
            #         ).item()
            #         self.log({f"quzo/kl_quant_error_{name}": kl_div})
            #         mse = torch.mean((original_data-param.data ) ** 2).item()
            #         self.log({f"quzo/mse_quant_error_{name}": mse})
        self.lr_scheduler.step()

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
                    mask = torch.rand_like(quantized_perturb) >= (getattr(self.args, 'mask_ratio', 50) / 100)
                    quantized_perturb = quantized_perturb * mask
                    z = zo_dequant(quantized_perturb, s1, z1) 
                else:
                    quantized_perturb, s1, z1 = zo_quant(fp_perturb, nbits=self.args.perturb_bits, stochastic=False)
                    mask = torch.rand_like(quantized_perturb) >= (getattr(self.args, 'mask_ratio', 50) / 100)
                    quantized_perturb = quantized_perturb * mask
                    z = zo_dequant(quantized_perturb, s1, z1) 

                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data = param.data - self._get_learning_rate() * (projected_grad * z + args.weight_decay * param.data)
                else:
                    param.data = param.data - self._get_learning_rate() * (projected_grad * z)
                zo_random_seed_update += 2
        
        self.lr_scheduler.step()

    
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
                mask = torch.rand_like(quantized_perturb) >= (90 / 100)
                quantized_perturb = quantized_perturb * mask
                z = zo_dequant(quantized_perturb, s1, z1) 
            else:
                quantized_perturb, s1, z1 = zo_quant(fp_perturb, nbits=self.args.perturb_bits, stochastic=True)
                mask = torch.rand_like(quantized_perturb) >= (90 / 100)
                quantized_perturb = quantized_perturb * mask
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


            # if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
 

            #     param.data = param.data - self._get_learning_rate() * (projected_grad * z + args.weight_decay * param.data)
            # else:
            #     param.data = param.data - self._get_learning_rate() * (projected_grad * z)
            # zo_random_seed_update += 2
        self.lr_scheduler.step()
     
    ############## Misc overload functions ##############


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

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        """
        结合记录损失到日志、保存和评估的逻辑。
        同时记录到wandb。
        """
        if self.control.should_log:
            # 所有进程都等待 0 号进程计算并广播资源指标
            metrics = None
            if self.is_world_process_zero():
                # 只有启用了 memory tracking 才计算并记录资源指标
                self._memory_tracker.start()

            logs = {}
            # 先确保将张量移动到CPU
            tr_loss_cpu = tr_loss.detach().cpu()
            tr_loss_scalar = self._nested_gather(tr_loss_cpu).mean().item()
            # 重置 tr_loss 为 0
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            if grad_norm is not None:
                logs["grad_norm"] = grad_norm

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

            # 记录到wandb
            if is_wandb_available() and self.is_world_process_zero():
                try:
                    wandb.log(
                        {
                            "train/loss": logs["loss"],
                            "train/learning_rate": logs["learning_rate"],
                            "train/epoch": self.state.epoch,
                            "train/step": self.state.global_step,
                        },
                        step=self.state.global_step
                    )
                    if grad_norm is not None:
                        wandb.log({"train/grad_norm": grad_norm}, step=self.state.global_step)
                
                    # 记录当前步骤每秒处理的样本数
                    if hasattr(self, 'train_step_duration'):
                        samples_per_second = self.args.train_batch_size / self.train_step_duration if hasattr(self, 'train_step_duration') else 0
                        wandb.log({"train/samples_per_second": samples_per_second}, step=self.state.global_step)
                except Exception as e:
                    logger.warning(f"Error logging to wandb: {e}")

        metrics = None
        if self.control.should_evaluate:
            try:
                # 尝试评估并捕获可能的错误
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
                self._report_to_hp_search(trial, self.state.global_step, metrics)
                
                # 评估结果也记录到wandb
                if metrics and is_wandb_available() and self.is_world_process_zero():
                    try:
                        for key, value in metrics.items():
                            wandb.log({f"eval/{key}": value}, step=self.state.global_step)
                    except Exception as e:
                        logger.warning(f"Error logging evaluation metrics to wandb: {e}")
            except Exception as e:
                logger.warning(f"Error during evaluation: {e}")
                metrics = None

        if self.control.should_save:
            try:
                self._save_checkpoint(model, trial, metrics=metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            except Exception as e:
                logger.warning(f"保存检查点时出错: {e}")

    def evaluate_func(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluate function to calculate metrics during training.
        If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        """
        from tqdm import tqdm
        import numpy as np
        
        # Use the Framework.evaluate logic
        logger.info(f"Evaluating {len(eval_samples)} samples")
        
        # For classification tasks
        if hasattr(self, 'task') and hasattr(self.task, 'metric_name'):
            metric_name = self.task.metric_name
        else:
            metric_name = "accuracy"
        
        # If we're using a classification approach
        if hasattr(self.args, 'train_as_classification') and self.args.train_as_classification:
            from metrics import calculate_metric, Prediction
            
            # Prepare the model for evaluation
            self.model.eval()
            
            # Process each sample
            predictions = []
            for eval_sample in tqdm(eval_samples):
                # Get the input for this sample
                inputs = self._prepare_inputs(eval_sample)
                
                # Get model predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                
                # Get the scores and make prediction
                scores = logits.detach().cpu().numpy()
                if hasattr(eval_sample, 'correct_candidate'):
                    if isinstance(eval_sample.correct_candidate, list):
                        # For datasets with multiple correct answers
                        correct_candidate_id = [
                            eval_sample.candidates.index(c)
                            for c in eval_sample.correct_candidate
                        ]
                    else:
                        correct_candidate_id = eval_sample.candidates.index(
                            eval_sample.correct_candidate
                        )
                    
                    predictions.append(Prediction(
                        correct_candidate=correct_candidate_id,
                        predicted_candidate=int(np.argmax(scores)),
                    ))
            
            # Calculate metrics
            metrics = {metric_name: calculate_metric(predictions, metric_name)}
            return metrics
        
        # Default evaluation for non-classification tasks
        # Use HuggingFace's evaluate method
        metrics = self.evaluate(ignore_keys_for_eval=None)
        return metrics