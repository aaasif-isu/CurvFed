"""Complete ResNet-18 architecture for standard Federated Learning on CIFAR-10 and MNIST."""

import math
import torch
from torch import nn
import torch.nn.functional as F


class Baseblock(nn.Module):
    expansion = 1

    def __init__(self, input_planes: int, planes: int, stride: int = 1, dim_change=None):
        super().__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, stride=stride, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride=1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dim_change = dim_change

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))

        if self.dim_change is not None:
            res = self.dim_change(res)

        output = output + res
        return F.relu(output)


class ResNet18(nn.Module):
    """Complete ResNet-18 model for image classification."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        block=Baseblock,
        num_layers=None,
    ):
        super().__init__()
        if num_layers is None:
            num_layers = [2, 2, 2]

        self.input_planes = 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        self.layer4 = self._layer(block, 128, num_layers[0], stride=2)
        self.layer5 = self._layer(block, 256, num_layers[1], stride=2)
        self.layer6 = self._layer(block, 512, num_layers[2], stride=2)
        self.averagePool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _layer(self, block, planes: int, num_layers: int, stride: int = 2) -> nn.Sequential:
        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = nn.Sequential(
                nn.Conv2d(
                    self.input_planes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [
            block(self.input_planes, planes, stride=stride, dim_change=dim_change)
        ]
        self.input_planes = planes * block.expansion
        for _ in range(1, num_layers):
            layers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res1 = F.relu(self.layer1(x))
        out1 = self.layer2(res1) + res1
        res2 = F.relu(out1)

        out2 = self.layer3(res2) + res2
        x3 = F.relu(out2)

        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)

        x7 = self.averagePool(x6)
        x8 = torch.flatten(x7, 1)
        return self.fc(x8)


def build_resnet18(config) -> ResNet18:
    return ResNet18(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
    )

