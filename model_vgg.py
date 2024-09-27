from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn

from layers import *
import math

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 10, init_weights: bool = True, dropout: float = 0
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            QuantLinear(nn.Linear(512*7*7, 4096)),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            QuantLinear(nn.Linear(4096, 4096)),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            QuantLinear(nn.Linear(4096, num_classes)),
        )
        #self.classifier = tdLayer(nn.Linear(512*7*7, num_classes))
        """self.surrogate_pred = nn.Sequential(
            nn.Linear(512*7*7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10),
        ) #decide whether to use categoriacl or normal"""
        self.surrogate = {}
        if init_weights:
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 1.0 / float(n))
                    m.bias.data.zero_()

            for name, m in self.named_modules():
                if isinstance(m, (QuantConv2d, QuantLinear)):
                    print(name)
                    name_ = name.replace(".", "-")
                    self.surrogate[name_] = nn.Parameter(torch.tensor([0, -2]).to(dtype=torch.float32), requires_grad=True)
        self.surrogate = nn.ParameterDict(self.surrogate)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x, self.surrogate


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:

    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = QuantConv2d(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg: str, batch_norm: bool, progress: bool, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model

def vgg11_bn(*, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg("A", True, progress, **kwargs)

def vgg13_bn(*, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg("B", True, progress, **kwargs)

def vgg16_bn(*, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg("D", True, progress, **kwargs)

def vgg19_bn(*, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg("E", True, progress, **kwargs)
