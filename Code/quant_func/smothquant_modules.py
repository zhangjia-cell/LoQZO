import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import quant_cuda
import torch
from torch import nn
from functools import partial


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w = w / scales
    w.round_()
    w = w * scales
    # w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w = w / scales
    w.round_()
    w = w * scales
    # w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t = t / scales
    t.round_()
    t = t * scales
    # t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t = t / scales
    t.round_()
    t = t * scales
    # t.div_(scales).round_().mul_(scales)
    return t
class SmoothLinearQuantizer(nn.Module):
    """
    Class to quantize given linear layer
    """
    def __init__(self, mode=None, wbit=None, abit=None, args=None):
        super(SmoothLinearQuantizer, self).__init__()

        if wbit <= 8:
         self.quant_weight = partial(quantize_weight_per_channel_absmax, n_bits=wbit)
        else:
         self.quant_weight = None
        if abit <= 8:
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=abit)
        else: 
         self.act_quant_name = None
         
        self.output_quant_name = self.act_quant_name
        self.output_quant = self.act_quant

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.weight = nn.Parameter(linear.weight.data.clone())
        try:
            self.bias = nn.Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, input): 
        # with torch.autocast(dtype=torch.float32):
        if self.quant_weight != None:
          weight = self.quant_weight(self.weight) 
        else:
            weight = self.weight
        # print(weight.dtype)
        if self.act_quant_name != None:
                  q_x = self.act_quant(input)
        else:
            q_x = input
        y = torch.functional.F.linear(q_x, weight, self.bias)
        q_y = self.output_quant(y)
        return q_y
