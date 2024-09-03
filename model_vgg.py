from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn

from layers import *

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 10, init_weights: bool = True, dropout: float = 0
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((7, 7)))
        self.classifier = nn.Sequential(
            tdLayer(nn.Linear(512*7*7, 4096), nn.BatchNorm1d(4096)),
            LIFSpike(),
            nn.Dropout(p=dropout),
            tdLayer(nn.Linear(4096, 4096), nn.BatchNorm1d(4096)),
            LIFSpike(),
            nn.Dropout(p=dropout),
            tdLayer(nn.Linear(4096, num_classes)),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.classifier(x)
        out = torch.sum(x, dim=2) / steps

        return out, x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [tdLayer(nn.MaxPool2d(kernel_size=2, stride=2))]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [tdLayer(conv2d, tdBatchNorm(v)), LIFSpike()]
            else:
                layers += [tdLayer(conv2d), LIFSpike()]
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
