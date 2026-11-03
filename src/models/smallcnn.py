"""Small CNN split model for CPU smoke tests and fast validation.

Architecture:
- Client-side (`SmallCNN_client_side`): 2 convolutional layers (`conv1`, `conv2`),
  each followed by ReLU and 2x2 MaxPool.
- Server-side (`SmallCNN_server_side`): 1 convolutional layer (`conv3`) + ReLU +
  AdaptiveAvgPool2d((4, 4)) + 1 fully-connected classifier layer (`fc`).
"""

import torch
from torch import nn
import torch.nn.functional as F


class SmallCNN_client_side(nn.Module):
    """Client-side sub-network with 2 convolutional layers."""

    def __init__(self, input_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x


class SmallCNN_server_side(nn.Module):
    """Server-side sub-network with 1 convolutional layer + 1 fully connected layer."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_smallcnn(config):
    client = SmallCNN_client_side(input_channels=config.input_channels)
    server = SmallCNN_server_side(num_classes=config.num_classes)
    return client, server
