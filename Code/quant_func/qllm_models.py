'''
文件说明：

'''
import torch
import torch.nn as nn
import numpy as np
import copy
from .qllm_modules import QLLMQuantizer

from .quant_utils import quant_args
import torch.distributed as dist
from transformers import pytorch_utils


def qllmquantize_model(model):
    """
    Recursively quantize a pretrained single-precision model to int8 quantized model
    model: pretrained single-precision model
    """
    # quantize layers
    if type(model) == nn.Linear:
        quant_mod = QLLMQuantizer(**quant_args)
        quant_mod.set_param(model)
        return quant_mod
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            mods.append(qllmquantize_model(m))
        return nn.Sequential(*mods)
    elif type(model) == nn.ModuleList:
        mods = []
        for n, m in model.named_children():
            mods.append(qllmquantize_model(m))
        return nn.Sequential(*mods)
    elif isinstance(model, nn.Sequential):
        mods = []
        for n, m in model.named_children():
            mods.append(qllmquantize_model(m))
        return nn.Sequential(*mods)
    else:
        # recursively use the quantized module to replace the single-precision module
        q_model = copy.deepcopy(model)
        # q_model = model
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and attr != 'base_model' and attr!= 'lm_head':
                setattr(q_model, attr, qllmquantize_model(mod))
        return q_model


def load_ant_state_dict(model, checkpoint):
    for name, module in model.named_modules():
        if name + ".quant_grid" in checkpoint.keys():
            module.quant_grid.data = checkpoint[name + ".quant_grid"]
