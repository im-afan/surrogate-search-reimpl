from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn

from layers import *
import math

class TestModel(nn.Module):
    def __init__(
        self,num_classes: int = 10, init_weights: bool = True, dropout: float = 0
    ) -> None:
        super().__init__()
        """self.classifier = nn.Sequential(
            tdLayer(nn.Linear(32*32*3, 100), nn.BatchNorm1d(100)),
            LIFSpike(),
            nn.Dropout(p=dropout),
            tdLayer(nn.Linear(100, num_classes)),
        )"""
        #self.classifier = nn.Sequential(tdLayer(nn.Linear(32*32*3, 512)), LIFSpike(), tdLayer(nn.Linear(512, num_classes)))
        self.classifier = nn.Sequential(tdLayer(nn.Linear(32*32*3, 512)), LIFSpike(), tdLayer(nn.Linear(512, num_classes)))
        #self.classifier = nn.Sequential(tdLayer(nn.Linear(32*32*3, num_classes)), tdLayer(nn.Linear(num_classes, num_classes)))
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
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, tdBatchNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 1.0 / float(n))
                    m.bias.data.zero_()
                elif isinstance(m, LIFSpike):
                    print(name)
                    name_ = name.replace(".", "-")
                    self.surrogate[name_] = nn.Parameter(torch.tensor([0, -2]).to(dtype=torch.float32), requires_grad=True)
        self.surrogate = nn.ParameterDict(self.surrogate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.classifier(x)
        out = torch.sum(x, dim=2) / steps

        return out, x, self.surrogate

def test_model():
    return TestModel()