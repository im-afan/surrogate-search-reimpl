import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from distrloss_layer import Distrloss_layer

class Quantize(torch.autograd.Function): #todo: scaling instead of just using sign function
    @staticmethod
    def forward(ctx, input, k):
        ctx.save_for_backward(input, k)
        output = torch.sign(input)
        return output.float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, k = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = 0.5 * k * (1 - torch.pow(torch.tanh(input * k), 2)) * grad_input
        return grad, None

quantize = Quantize.apply

class QuantConv2d(nn.Module):
    def __init__(self, m: nn.Module):
        self.__dict__ = m.__dict__.copy()
        self.k = torch.ones(1)

    def forward(self, x: torch.Tensor):
        x_b = quantize(x, self.k.to(x.device))

        bias_b = None
        if(self.bias is not None):
            bias_b = quantize(self.bias, self.k.to(x.device))
        weight_b = quantize(self.weight, self.k.to(x.device))

        return F.conv2d(
            x_b, weight_b, bias_b, self.stride, self.padding, self.dilation, self.groups
        )

class QuantLinear(nn.Module):
    def __init__(self, m: nn.Module):
        self.__dict__ = m.__dict__.copy()
        self.k = torch.ones(1)

    def forward(self, x: torch.Tensor):
        x_b = quantize(x, self.k.to(x.device))

        bias_b = None
        if(self.bias is not None):
            bias_b = quantize(self.bias, self.k.to(x.device))
        weight_b = quantize(self.weight, self.k.to(x.device))

        return F.linear(
            x_b, weight_b, bias_b
        )

