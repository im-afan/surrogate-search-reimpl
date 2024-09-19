import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from distrloss_layer import Distrloss_layer
steps = 4
dt = 5
simwin = dt * steps
a = 0.25
aa = 0.5    
Vth = 0.5   
tau = 0.20  

class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        ctx.save_for_backward(input, k)
        # if input = u > Vth then output = 1
        output = torch.gt(input, Vth) 
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, k = ctx.saved_tensors 
        grad_input = grad_output.clone()
        # hu is an approximate func of df/du
        #hu = abs(input-Vth) < aa
        #hu = hu.float() / (2 * aa)
        grad = 0.5 * k * (1 - torch.pow(torch.tanh((input - Vth) * k), 2)) * grad_input
        return grad, None

quantize = Quantize.apply()

class QuantLayer(nn.Module):
    def __init__(self, module: nn.Module):
        self.module = module

    def forward(self, x):
        x = quantize(x)
        