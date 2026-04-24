import torch
import torch.nn as nn
import torch.nn.functional as F

class STEFunction(torch.autograd.Function):
    """
    Custom autograd function for Straight-Through Estimator (STE).
    Forward: quantized values
    Backward: gradients as if no quantization was applied
    """
    @staticmethod
    def forward(ctx, input, bits=8):
        """
        In the forward pass, we compute the quantized values
        """
        # Save the input for backward pass
        ctx.save_for_backward(input)
        
        # Compute quantization parameters
        qmin = -(2 ** (bits - 1))
        qmax = (2 ** (bits - 1)) - 1
        
        # Compute scaling factor
        max_abs = torch.max(torch.abs(input))
        scale = max_abs / qmax if max_abs > 0 else 1e-6
        
        # Quantize
        input_scaled = input / scale
        input_quantized = torch.round(input_scaled)
        input_quantized = torch.clamp(input_quantized, qmin, qmax)
        
        # Dequantize to return floating point values
        output = input_quantized * scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, we straight-through the gradient
        """
        input, = ctx.saved_tensors
        # STE: Pass gradient through as if no quantization was applied
        return grad_output, None  # Return None for the bits parameter

def apply_ste_quantization(tensor, bits=8):
    """
    Apply STE quantization to a tensor
    """
    return STEFunction.apply(tensor, bits)

class QuantizationHook:
    """
    Hook to apply STE quantization to gradients during backward pass
    """
    def __init__(self, bits=8):
        self.bits = bits
    
    def __call__(self, module, grad_input):
        """
        Hook called during backward pass
        """
        # Apply STE quantization to each gradient
        return tuple(
            apply_ste_quantization(g, self.bits) if g is not None else None 
            for g in grad_input
        )

def add_ste_hooks(model, bits=8):
    """
    Add STE quantization hooks to all linear layers in the model
    """
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = module.register_full_backward_pre_hook(
                QuantizationHook(bits)
            )
            hooks.append(hook)
    return hooks

class STEQuantizedLinear(nn.Linear):
    """
    Linear layer with STE quantization for weights and activations
    """
    def __init__(self, in_features, out_features, bias=True, w_bits=8, a_bits=8):
        super(STEQuantizedLinear, self).__init__(in_features, out_features, bias)
        self.w_bits = w_bits
        self.a_bits = a_bits
    
    def forward(self, input):
        # Quantize activations (inputs) with STE
        if self.a_bits < 32:
            input_q = apply_ste_quantization(input, self.a_bits)
        else:
            input_q = input
            
        # Quantize weights with STE
        if self.w_bits < 32:
            weight_q = apply_ste_quantization(self.weight, self.w_bits)
        else:
            weight_q = self.weight
            
        # Normal linear operation with quantized weights and activations
        output = F.linear(input_q, weight_q, self.bias)
        return output

def convert_model_to_ste_quantized(model, w_bits=8, a_bits=8):
    """
    Convert a model's linear layers to STE quantized linear layers
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, STEQuantizedLinear(
                module.in_features, 
                module.out_features, 
                module.bias is not None,
                w_bits=w_bits,
                a_bits=a_bits
            ))
            # Copy the weights and bias
            getattr(model, name).weight.data.copy_(module.weight.data)
            if module.bias is not None:
                getattr(model, name).bias.data.copy_(module.bias.data)
        else:
            convert_model_to_ste_quantized(module, w_bits, a_bits)
    
    return model 