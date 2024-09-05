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

class SpikeAct(torch.autograd.Function):
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

class SpikeAct_kt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        output = torch.gt(input, Vth) 
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        #grad_o = grad_output.clone()
        #hu = torch.abs(input - Vth) < aa
        #hu = hu.float() / (2 * aa)
        grad_input = 0.5 * t * (1 - torch.pow(torch.tanh((input - Vth) * t), 2))
        #return grad_o * hu, None, None
        return grad_input * grad_output, None, None

spikeAct = SpikeAct.apply
spikeAct_kt = SpikeAct_kt.apply
distrloss_layer = Distrloss_layer()


def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n, k):
    u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
    o_t1_n1 = spikeAct(u_t1_n1, k)
    return u_t1_n1, o_t1_n1


def state_update_loss(u_t_n1, o_t_n1, W_mul_o_t1_n):
    u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
    o_t1_n1 = spikeAct(u_t1_n1)
    return u_t1_n1, o_t1_n1, u_t1_n1

def state_update_loss_kt(u_t_n1, o_t_n1, W_mul_o_t1_n,k,t, debug=False):
    u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
    o_t1_n1 = spikeAct_kt(u_t1_n1,k,t)
    if(debug):
        print((u_t1_n1[0] - u_t1_n1[1]).abs().mean(), u_t1_n1[0].abs().mean(), (o_t1_n1[0] - o_t1_n1[1]).abs().sum())
    return u_t1_n1, o_t1_n1, u_t1_n1


class tdLayer(nn.Module):

    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = layer
        self.bn = bn

    def forward(self, x):
        x_ = torch.zeros(self.layer(x[..., 0]).shape + (steps,), device=x.device)
        for step in range(steps):
            x_[..., step] = self.layer(x[..., step])

        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class tdLayer_loss(nn.Module):

    def __init__(self, layer, bn=None):
        super(tdLayer_loss, self).__init__()
        self.layer = layer
        self.bn = bn

    def forward(self, x):
        x_ = torch.zeros(self.layer(x[..., 0]).shape + (steps,), device=x.device)
        for step in range(steps):
            x_[..., step] = self.layer(x[..., step])

        if self.bn is not None:
            x_ = self.bn(x_)
        return x_ , x[1]

        
class LIFSpike(nn.Module):

    def __init__(self):
        super(LIFSpike, self).__init__()
        self.k = torch.tensor([1]).float()

    def forward(self, x):
        u = torch.zeros(x.shape[:-1] , device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(steps):
            u, out[..., step] = state_update(u, out[..., max(step-1, 0)], x[..., step], self.k.to(x.device))
        return out

class LIFSpike_loss(nn.Module):

    def __init__(self):
        super(LIFSpike_loss, self).__init__()

    def forward(self, x, test_cal = None):
        u = torch.zeros(x.shape[:-1] , device=x.device)
        u_pre = torch.zeros(x.shape , device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(steps):
            u, out[..., step], u_pre[..., step] = state_update_loss(u, out[..., max(step-1, 0)], x[..., step])
        loss = distrloss_layer(u_pre)
        # if test_cal:
            # test_cal.add_rate(u_pre)
        return out, loss

class LIFSpike_loss_kt(nn.Module):
    #对带有时间维度的张量进行一次LIF神经元的发放模拟，可以视为一个激活函数，用法类似ReLU。
    #    Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the #data.
    #"""
    def __init__(self):
        super(LIFSpike_loss_kt, self).__init__()
        self.k = torch.tensor([5]).float()
        self.t = torch.tensor([10]).float()
        
    def forward(self, x, debug=False):
        device_x = x.device
        u = torch.zeros(x.shape[:-1] , device=x.device)
        u_pre = torch.zeros(x.shape , device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(steps):
            u, out[..., step], u_pre[..., step] = state_update_loss_kt(u, out[..., max(step-1, 0)], x[..., step], self.k.to(device_x), self.t.to(device_x), debug=debug)

        if(debug):
            print((u_pre[0] - u_pre[1]).abs().sum(), (out[0] - out[1]).abs().sum(), (x[0] - x[1]).abs().sum(), out.abs().sum())
        loss = distrloss_layer(u_pre)
        return out, loss
    
    
class tdBatchNorm(nn.BatchNorm3d): # TODO MAKE TDBN FOR LINEAR AS WELL # TODO MAKE TDBN FOR LINEAR AS WELL # TODO MAKE TDBN FOR LINEAR AS WELL # TODO MAKE TDBN FOR LINEAR AS WELL
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True):
        super(tdBatchNorm, self).__init__(
             num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)
        #global Vth
        #Vth = Vth.to(input.device)
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps)
