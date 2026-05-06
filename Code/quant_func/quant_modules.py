import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import quant_cuda

class QuantBase():
    def _quantization(x, quant_grid):
        shape = x.shape
        quant_array = x.reshape(-1).contiguous()

        # 中文说明：quant_cuda 当前 CUDA 扩展只通过 AT_DISPATCH_FLOATING_TYPES
        # 支持 float32/float64，不支持 torch.float16 / torch.bfloat16。
        # 为了让 LLaMA-2 这类模型可以用 BF16/FP16 加载来提速，这里在量化核内部临时转成
        # float32 做 nearest-codebook 查询，再转回原 dtype。这样 Data Precision 仍由 WBIT/ABIT/WMODE/AMODE 控制，
        # LOAD_BFLOAT16/LOAD_FLOAT16 只影响底层矩阵乘 dtype。
        orig_dtype = quant_array.dtype
        if orig_dtype not in (torch.float32, torch.float64):
            quant_array_for_cuda = quant_array.float().contiguous()
            quant_grid_for_cuda = quant_grid.float().contiguous()
            quant_array_for_cuda, _ = quant_cuda.quant(quant_array_for_cuda, quant_grid_for_cuda)
            quant_array = quant_array_for_cuda.to(dtype=orig_dtype)
        else:
            quant_grid = quant_grid.type_as(quant_array).contiguous()
            quant_array, _ = quant_cuda.quant(quant_array, quant_grid)

        quant_array = quant_array.view(shape)
        return quant_array

    @staticmethod
    def forward(real_val, quant_grid):
        with torch.no_grad():
            dequantized_val = QuantBase._quantization(real_val, quant_grid)
            return dequantized_val


class Quantizer(nn.Module):
    def __init__(self, mode="base", bit=8, is_signed=True, is_enable=False, is_input=False, args=None, operator=None):
        super(Quantizer, self).__init__()
        self.mode = mode
        self.is_input = is_input
        self.is_signed = is_signed
        self.is_enable = is_enable
        self.is_enable_activation = is_enable
        self.is_enable_weight = is_enable
        self.args = args
        self.operator = operator
        
        self.w_up = self.args.w_up
        self.a_up = self.args.a_up
        self.w_low = self.args.w_low
        self.a_low = self.args.a_low

        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        # Python 侧缓存 bit 宽，避免每次 forward 对 GPU buffer 调 .item()。
        self.bit_width_py = int(bit)
        self.register_buffer('bit', torch.tensor(bit,dtype=torch.int))
        self.register_buffer('has_inited_quant_para', torch.tensor(0.0,dtype=torch.int))
        self.register_buffer('quant_grid', torch.ones(2**bit))
        self.register_buffer('outliers', torch.ones(2**bit))
        self.percent = self.args.percent / 100
        self.is_perchannel = True
        if is_input:
            # Input shouldn't be per-channel quantizaton！
            self.is_perchannel = False
        self.search = args.search
        self.mse = torch.tensor(0.0)

        ## debug
        self.name = None

        # 中文说明：has_inited_quant_para 是 GPU buffer。旧代码每次 forward 用它做 Python if，
        # 会触发 GPU->CPU 同步；大模型每层每次 forward 都同步会明显拖慢训练。
        # 这里增加 Python bool 作为快速路径，buffer 仍保留给旧接口 / checkpoint 使用。
        self._quant_initialized_py = False

    def disable_input_quantization(self):
        self.is_enable_activation = False
        
    def enable_quantization(self, name):
        self.name = name
        self.is_enable = True

    def disable_quantization(self, name):
        self.name = name
        self.is_enable = False

    def update_signed(self, tensor):
        if tensor.min() < 0:
            self.is_signed = True

    def convert_tensor(self, values):
        if 2 ** self.bit.item() > len(values):
            values.append(0.)
        assert(2 ** self.bit.item() == len(values))
        values = torch.tensor(values, device=self.quant_grid.device)
        values, _ = torch.sort(values)
        values = values.mul(10.0 / torch.max(values))
        # print(values.shape, values.data, end="--")
        return values
    
    @torch.no_grad()
    def int_value(self):
        bit_width = self.bit.item()
        B = torch.tensor(bit_width, device=self.quant_grid.device) 
        # B = bit_width
        if self.is_signed:
            B = bit_width - 1

        values = []
        values.append(0.)
        for i in range(1, 2 ** B):
            values.append(i)
            if self.is_signed:
                values.append(-i)
                
        values = torch.tensor(values, device=self.quant_grid.device) 
        values, _ = torch.sort(values)
    
        # add a bias to normalize the codebook (the threshold between outliers and normal values is 32)
        values *= 32 / (2 ** B)
        # values = torch.tensor(values, device=self.quant_grid.device,dtype=torch.int8) 

        return values
    
    @torch.no_grad()
    def float_value(self):
        B = self.bit.item()
        if self.is_signed:
            B = B - 1
        exp_bit = 3
        man_bit = B - 3
        if B == 2:
            exp_bit = 2
            man_bit = 0
        values = []
        min_to_zero = True
        for i in range(2 ** exp_bit):
            for j in range(2 ** man_bit):
                if min_to_zero:
                    values.append(0.)
                    min_to_zero = False
                else:
                    values.append(2 ** i * (1 + j * 2 ** (-man_bit)))
                    if self.is_signed:
                        values.append(- 2 ** i * (1 + j * 2 ** (-man_bit)))

        return self.convert_tensor(values)
    
    @torch.no_grad()
    def flint_value(self,  exp_base = 0):
        B = self.bit.item()
        if self.is_signed:
            B = B - 1
        value_bit = B
        assert(value_bit >= 2)

        exp_num =     value_bit * 2 - 1
        neg_exp_num = value_bit - 1
        pos_exp_num = value_bit - 1
        
        exp_max = pos_exp_num + exp_base
        exp_min = -neg_exp_num

        ## zero
        values = [0.]

        ## exponent negtive
        for i in range(0, neg_exp_num + 1):
            exp_bit = i + 2
            exp_value = -(exp_bit - 1)
            mant_bit = value_bit - exp_bit
            for j in range(int(2 ** mant_bit)):
                v = 2 ** exp_value * (1 + 2 ** (-mant_bit) * j)
                values.append(v)
                if self.is_signed:
                    values.append(-v)

        ## exponent zero
        exp_bit = 2
        exp_value = 0
        mant_bit = value_bit - exp_bit
        for j in range(int(2 ** mant_bit)):
            v = 2 ** (exp_value + exp_base) * (1 + 2 ** (-mant_bit) * j)
            values.append(v)
            if self.is_signed:
                values.append(-v)
                
        ## exponent positive     
        for i in range(1, pos_exp_num):
            exp_bit = i + 2
            exp_value = i
            mant_bit = value_bit - exp_bit
            for j in range(int(2 ** mant_bit)):
                v = 2 ** (exp_value + exp_base) * (1 + 2 ** (-mant_bit) * j)
                values.append(v)
                if self.is_signed:
                    values.append(-v)
                    
        ## max value
        values.append(2 ** exp_max)
        if self.is_signed:
            values.append(-2 ** exp_max)
            
        values = torch.tensor(values, device=self.quant_grid.device)
        values, _ = torch.sort(values)
        # add a bias to normalize the codebook (the threshold between outliers and normal values is 32)
        values *= 32 / (2 ** exp_max)

        return values
    
    # abfloat 
    @torch.no_grad()
    def outlier_value(self, exp_bit = 2, exp_base = 5):
        B = self.bit.item()
        if self.is_signed:
            B = B - 1
            
        value_bit = B
        mant_bit = value_bit - exp_bit
        values = []
        
        for i in range(exp_base, exp_base + 2 ** exp_bit):
            for j in range(int(2 ** mant_bit)):
                if i == exp_base and j == 0:
                    continue

                v = 2 ** i * (1 + 2 ** (-mant_bit) * j)
                values.append(v)
                if self.is_signed:
                    values.append(-v)
                    
        values = torch.tensor(values, device=self.quant_grid.device)
        values, _ = torch.sort(values)
                    
        return values

    @torch.no_grad()
    def mse_loss(self, quant_tensor, source_tensor, p=2.0, is_perchannel=True):
        if is_perchannel:
            mean_tensor =  (quant_tensor-source_tensor).abs().pow(p).view(quant_tensor.shape[0], -1).mean(-1).unsqueeze(1)
            return mean_tensor
        else:
            return (quant_tensor-source_tensor).abs().pow(p).mean()
        
    @torch.no_grad()
    def search_mse(self, tensor):
        if self.is_perchannel and (not self.is_input):
            if not self.args.no_outlier:
                mean = tensor.view(tensor.shape[0], -1).mean(dim=-1)
                std = tensor.view(tensor.shape[0], -1).std(dim=-1)
                x_max = torch.maximum((mean + 3 * std).abs(), (mean - 3 * std).abs())
            else:
                x_max, _ = tensor.view(tensor.shape[0], -1).abs().max(1)
            x_max = x_max.unsqueeze(1)            
            best_score = torch.ones_like(x_max) * 1e10
            alpha = x_max.clone()
            base_alpha = x_max.clone()
            lb = int(self.w_low)
            ub = int(self.w_up)
            for i in range(lb, ub, 2):
                new_alpha = base_alpha * (i * 0.01)
                self.alpha.data = new_alpha
                quant_tensor = self._forward(tensor)

                score = self.mse_loss(quant_tensor, tensor)
                alpha[score < best_score] = new_alpha[score < best_score]
                best_score[score < best_score] = score[score < best_score]
        else:        
            if not self.args.no_outlier:
                mean = tensor.mean()
                std = tensor.std()
                x_max = torch.maximum((mean + 3 * std).abs(), (mean - 3 * std).abs())
            else:
                x_max = tensor.abs().max()
            best_score = 1e10
            alpha = x_max.clone()
            base_alpha = alpha.clone()            
            lb = int(self.a_low)
            ub = int(self.a_up)
            for i in range(lb, ub, 2):
                new_alpha = base_alpha * (i * 0.01)
                self.alpha.data = new_alpha
                quant_tensor = self._forward(tensor)
                score = self.mse_loss(quant_tensor, tensor, p = 2, is_perchannel=False)
                if score < best_score:
                    best_score = score
                    alpha = new_alpha

        return torch.tensor(best_score).sum(), alpha, (alpha / x_max).mean().item()

    @torch.no_grad()
    def search_adaptive_numeric_type(self, data):
        modes = []
        mse_list = []
        mode = self.mode
        if "-int" in mode:
            self.mode = 'int'
            self.quant_grid.data = self.int_value()
            best_score_int, _, _ = self.search_mse(data)
            modes.append('int')
            mse_list.append(best_score_int.item())

        if "-flint" in mode:
            self.mode = 'flint'
            self.quant_grid.data = self.flint_value()
            best_score_flint, _, _ = self.search_mse(data)
            modes.append('flint')
            mse_list.append(best_score_flint.item())

        if "-float" in mode:
            self.mode = 'float'
            self.quant_grid.data = self.float_value()
            best_score_float, _, _ = self.search_mse(data)
            modes.append('float')
            mse_list.append(best_score_float.item())
            # if dist.get_rank() == 0:
            #     print("ANT search, FLOAT score: %f" %best_score_float)

        mse_list = np.array(mse_list)
        mse_idx = np.argsort(mse_list)
        self.mode = modes[mse_idx[0]]

    @torch.no_grad()
    def _init_quant_para(self, data, data_b):
        with torch.no_grad():
            # 初始化完成后直接返回，避免每个 forward 读取 GPU 标量 buffer。
            if self._quant_initialized_py:
                return

            # 兼容从 checkpoint 恢复、或旧接口手动设置 has_inited_quant_para 的情况。
            try:
                if bool(self.has_inited_quant_para.detach().cpu().item() != 0):
                    self._quant_initialized_py = True
                    return
            except Exception:
                pass

            if not self._quant_initialized_py:
                self.update_signed(data)                
                self.outliers.data = self.outlier_value()

                if self.is_perchannel:
                    x_max = data.view(data.shape[0], -1).abs().max(1).values
                    self.alpha.data = x_max.unsqueeze(1)
                else:
                    self.alpha.data = data.abs().max()

                # if self.bit > 6:
                #     self.mode = 'int'
                #     self.quant_grid.data = self.int_value()
                # else:
                if "ant-" in self.mode:
                        self.search_adaptive_numeric_type(data)

                if self.mode == "flint":
                    self.quant_grid.data = self.flint_value()
                elif self.mode == "int":
                    self.quant_grid.data = self.int_value()
                elif self.mode == "float":
                    self.quant_grid.data = self.float_value()
                else:
                    raise RuntimeError("Unsupported mode: " + self.mode)
                
                _, self.alpha.data, alpha_ratio = self.search_mse(data)

                quant_data = self._forward(data)
                self.mse = self.mse_loss(quant_data, data, 2, is_perchannel=self.is_perchannel).mean()
                print(self.mode, end="\t")
                print("%d-bit \t %s," %(self.bit.item(), self.name))
                
                self.has_inited_quant_para.data = torch.ones_like(self.has_inited_quant_para)
                self._quant_initialized_py = True
         
    @torch.no_grad()
    def _forward_int_uniform(self, data):
        """int codebook + no_outlier=True 时的等价快速量化路径。

        原路径会把每个元素送入 quant_cuda，与完整 codebook 做 nearest-neighbor 查找。
        当 mode=int 且 no_outlier=True 时，codebook 是等间距整数网格，最近邻量化可以直接用
        round + clamp 表达，避免每个 forward 对整层权重做通用 codebook 搜索。
        该路径只在 args.fast_int_quant=True 且 args.no_outlier=True 时启用，因此不会影响默认 baseline。
        """
        orig_dtype = data.dtype
        bit_width = int(getattr(self, "bit_width_py", 8))
        b = bit_width - 1 if self.is_signed else bit_width
        step = 32.0 / float(2 ** b)
        qmax = step * float((2 ** b) - 1)
        qmin = -qmax if self.is_signed else 0.0

        # 与旧实现一致：scale = alpha / max(quant_grid)，量化发生在 data / scale 的归一化空间。
        scale = self.alpha / qmax
        if self.is_perchannel:
            x = (data.view(data.shape[0], -1) / scale).view(data.shape)
        else:
            x = data / scale

        q = torch.round(x / step) * step
        q = torch.clamp(q, min=qmin, max=qmax)
        tensor = (q - x).detach() + x

        if self.is_perchannel:
            tensor = (tensor.view(tensor.shape[0], -1) * scale).view(data.shape)
        else:
            tensor = tensor * scale
        if tensor.dtype != orig_dtype:
            tensor = tensor.to(dtype=orig_dtype)
        return tensor

    @torch.no_grad()   
    def _forward(self, data, display=False):
        # mode=int 且 no_outlier=True 时，使用 round/clamp 快速路径；
        # 否则保留通用 quant_cuda codebook 路径，保证旧实验语义不变。
        if (
            self.mode == "int"
            and bool(getattr(self.args, "fast_int_quant", True))
            and bool(getattr(self.args, "no_outlier", False))
        ):
            return self._forward_int_uniform(data)

        orig_dtype = data.dtype
        scale = self.alpha / torch.max(self.quant_grid)
        
        if self.is_perchannel: 
            data = (data.view(data.shape[0], -1) / scale).view(data.shape)
        else:
            data = data / scale
            
        if not self.args.no_outlier:
            quant_grid = torch.cat((self.quant_grid, self.outliers), dim = 0)
        else:
            quant_grid = self.quant_grid
            
        quant_data = QuantBase.forward(data, quant_grid)
        shape = data.shape
        
        # Outlier Victim Pair Encoding
        if not self.args.no_outlier:
            quant_data = quant_data.view(-1)                
            mask = quant_data.abs() > 32
            victim_odd = torch.roll(mask, 1, -1)
            victim_odd[::2] = 0
            victim_even = torch.roll(mask & (~victim_odd), -1, -1)
            victim_even[1::2] = 0
            victim = victim_even | victim_odd
            quant_data = quant_data * (~victim)

        quant_data = quant_data.view(shape)
        tensor = (quant_data - data).detach() + data
        
        if self.is_perchannel:
            tensor = (tensor.view(tensor.shape[0], -1) * scale).view(data.shape)
        else:
            tensor = tensor * scale

        # 若模型底层以 BF16/FP16 加载，量化内部可用 FP32 做 codebook 查询，
        # 但返回给 F.linear/Conv 的仍保持原 dtype，以便使用 Tensor Core 加速。
        if tensor.dtype != orig_dtype:
            tensor = tensor.to(dtype=orig_dtype)
        return tensor

    @torch.no_grad()
    def tensor_forward(self, tensor, input_tensor = None):
        if self.mode == "base":
            return tensor
        if not self.is_enable:
            return tensor
        if self.is_input:
            if not self.is_enable_activation:
                return tensor
        else:
            if not self.is_enable_weight:
                return tensor

        self._init_quant_para(tensor, input_tensor)

        q_tensor = self._forward(tensor)

        return q_tensor    

class TensorQuantizer(Quantizer):
    def __init__(self, **kwargs):
        super(TensorQuantizer, self).__init__(**kwargs)

    def forward(self, tensor, input_tensor = None):
        return self.tensor_forward(tensor, input_tensor)

class Conv1dQuantizer(nn.Module):
    """
    Class to quantize given convolutional layer
    """
    def __init__(self, mode=None, wbit=None, abit=None, args=None, wmode=None, amode=None):
        super(Conv1dQuantizer, self).__init__()
        assert mode is not None,'Quantizer is not initilized!'
        # wmode / amode 用来分别控制权重和激活的数值格式；
        # 不显式传入时沿用旧参数 mode，保证旧脚本兼容。
        wmode = wmode or mode
        amode = amode or mode
        self.quant_weight = TensorQuantizer(mode=wmode, bit=wbit, is_signed=True, is_enable=True, args=args, operator=self._conv_forward)
        self.quant_input  = TensorQuantizer(mode=amode, bit=abit, is_signed=False, is_enable=True, args=args, operator=self._conv_forward, is_input=True)

    def set_param(self, conv):

        self.nf = conv.nf
        self.weight = nn.Parameter(conv.weight.data.clone())
        try:
            self.bias = nn.Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None
            
    def _conv_forward(self, x, weight):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), weight)
        x = x.view(size_out)
        return x

    def forward(self, input):
        weight = self.quant_weight(self.weight, input)
        input = self.quant_input(input, self.weight)
        return self._conv_forward(input, weight)


class Conv2dQuantizer(nn.Module):
    """
    Class to quantize given convolutional layer
    """
    def __init__(self, mode=None, wbit=None, abit=None, args=None, wmode=None, amode=None):
        super(Conv2dQuantizer, self).__init__()
        assert mode is not None,'Quantizer is not initilized!'
        # wmode / amode 用来分别控制权重和激活的数值格式；
        # 不显式传入时沿用旧参数 mode，保证旧脚本兼容。
        wmode = wmode or mode
        amode = amode or mode
        self.quant_weight = TensorQuantizer(mode=wmode, bit=wbit, is_signed=True, is_enable=True, args=args, operator=self._conv_forward)
        self.quant_input  = TensorQuantizer(mode=amode, bit=abit, is_signed=False, is_enable=True, args=args, operator=self._conv_forward, is_input=True)

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        
        self.quant_weight.alpha.data = torch.ones([self.out_channels,1])

        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = nn.Parameter(conv.weight.data.clone())
        try:
            self.bias = nn.Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def _conv_forward(self, input, weight):
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        weight = self.quant_weight(self.weight, input)
        input = self.quant_input(input, self.weight)
        return self._conv_forward(input, weight)



class LinearQuantizer(nn.Module):
    """
    线性层量化包装器。

    修复点：
    1) 当 wbit > 8 时 quant_weight=None，旧代码仍然访问 quant_weight.alpha 会报错；
    2) alpha 初始化时显式放到原 linear.weight 的 device/dtype，避免模型已经在 GPU 上
       再插入量化模块时出现 CPU/GPU device mismatch；
    3) QZO 交替优化会直接更新 quant_weight.alpha，因此这里保持 alpha 为 nn.Parameter。
    """
    def __init__(self, mode=None, wbit=None, abit=None, args=None, wmode=None, amode=None):
        super(LinearQuantizer, self).__init__()
        assert mode is not None, 'Quantizer is not initilized!'

        # 中文说明：
        #   mode  是旧代码里的统一数值格式；
        #   wmode 是权重量化格式；amode 是激活量化格式。
        # 这样就能复现实验表中的 FP W8A8、INT W8A8、INT/FP W4A8：
        #   FP W8A8     -> wmode=float, amode=float, wbit=8, abit=8
        #   INT W8A8    -> wmode=int,   amode=int,   wbit=8, abit=8
        #   INT/FP W4A8 -> wmode=int,   amode=float, wbit=4, abit=8
        # 不传 wmode/amode 时自动沿用 mode，保证原有 WBIT/ABIT/QMODE 脚本不受影响。
        wmode = wmode or mode
        amode = amode or mode
        self.wmode = wmode
        self.amode = amode

        if wbit <= 8:
            self.quant_weight = TensorQuantizer(mode=wmode, bit=wbit, is_signed=True, is_enable=True, args=args, operator=F.linear)
        else:
            # wbit > 8 视为不做权重量化，例如 FP W32A32。
            self.quant_weight = None
        if abit <= 8:
            self.quant_input = TensorQuantizer(mode=amode, bit=abit, is_signed=False, is_enable=True, args=args, operator=F.linear, is_input=True)
        else:
            # abit > 8 视为不做激活量化，例如 FP W32A32。
            self.quant_input = None

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        weight_data = linear.weight.data.detach().clone()
        self.weight = nn.Parameter(weight_data)

        if self.quant_weight is not None:
            # 权重量化采用 per-output-channel scale，shape=[out_features, 1]
            # alpha/scale 由 QZO 手动更新，保持 FP32 可以避免 BF16/FP16 下 lr 很小时更新被舍入掉。
            self.quant_weight.alpha.data = torch.ones(
                [self.out_features, 1],
                device=weight_data.device,
                dtype=torch.float32,
            )

        try:
            self.bias = nn.Parameter(linear.bias.data.detach().clone())
        except AttributeError:
            self.bias = None

    def forward(self, input):
        if self.quant_weight is not None:
            weight = self.quant_weight(self.weight, input)
        else:
            # wbit > 8 时不做权重量化，直接使用浮点权重。
            weight = self.weight

        if self.quant_input is not None:
            input = self.quant_input(input, self.weight)
        return F.linear(input, weight, self.bias)
