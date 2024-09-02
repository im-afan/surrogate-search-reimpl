import math
import torch
import torch.nn as nn
from layers import *
import torch.nn.functional as F

from distrloss_layer import Distrloss_layer

# ------------------- #
#   ResNet Example    #
# ------------------- #


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = tdBatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, alpha=1)
        self.downsample = downsample
        self.stride = stride

        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.conv2_s = tdLayer(self.conv2, self.bn2)

        self.spike = LIFSpike_loss_kt()

    def forward(self, x):
     
        identity = x[0]
        all_layer_loss = x[1]

        #print('identity', identity.shape)
        #print('all_layer_loss', len(all_layer_loss))
        out = self.conv1_s(x[0])
        out, layer_loss = self.spike(out)

        out = self.conv2_s(out)

        if self.downsample is not None:
            identity = self.downsample(x[0])

        out += identity
        out, layer_loss = self.spike(out)
        all_layer_loss.append(layer_loss)
        #print('all_layer_loss-add2', len(all_layer_loss))

        return [out, all_layer_loss]


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = tdBatchNorm
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((4, 4)))# enlarge the output size after pooling
        
        #self.fc1 = nn.Linear(512 * 16 * block.expansion, 256)
        self.fc1 = nn.Linear(512 * 16, 256)
        self.fc1_s = tdLayer(self.fc1, nn.BatchNorm1d(256))
        self.fc2 = nn.Linear(256, 10)
        self.fc2_s = tdLayer(self.fc2)
        self.spike = LIFSpike_loss_kt()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, tdBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                m.bias.data.zero_()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tdLayer(
                conv3x3(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, alpha=1)
            )  # downsample via conv3脳3 instead of 1脳1

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        g_loss = []

        # See note [TorchScript super()]
        x = self.conv1_s(x)
        x, layer_loss = self.spike(x)
        g_loss.append(layer_loss)

        #from IPython import embed
        #embed(header='First time')
        x = self.layer1([x, g_loss])
        x = self.layer2(x)
        x = self.layer3(x)

        tmp = x[0]
        g_loss = x[1]
       
        x = self.avgpool(tmp)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        #print(f"diff: {(x[0] - x[1]).abs().sum()}, shape: {x.shape}")
        x, layer_loss = self.spike(x)
        #print(f"diff: {(x[0] - x[1]).abs().sum()}, shape: {x.shape} (1)")
        g_loss.append(layer_loss)
        x = self.fc2_s(x) 
        out = torch.sum(x, dim=2) / steps
     
        num = 0
        for ele in g_loss:
            num = ele + num 
        loss = num / len(g_loss)

        return out, loss

    def forward(self, x):
        return self._forward_impl(x)
    
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet19(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet19', BasicBlock, [3, 3, 2], pretrained, progress,
                   **kwargs)

def set_kt(module, k, t):
    for name, child_module in module.named_parameters():
        if(isinstance(child_module, LIFSpike_loss_kt)):
            child_module.k = k
            child_module.t = t
        set_kt(child_module, k, t)