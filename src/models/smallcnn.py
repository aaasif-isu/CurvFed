"""Complete SmallCNN model for CPU smoke tests and fast Federated Learning validation.

Architecture:
- 2 convolutional layers (`conv1`, `conv2`), each followed by ReLU and 2x2 MaxPool.
- 1 convolutional layer (`conv3`) + ReLU + AdaptiveAvgPool2d((4, 4)) + 1 linear classifier (`fc`).
"""

import torch
from torch import nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """Complete 3-conv + 1-FC CNN for MNIST and CIFAR-10."""

    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = self.avg_pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_smallcnn(config) -> SmallCNN:
    return SmallCNN(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
    )

