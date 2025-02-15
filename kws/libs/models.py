from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, num_classes: int = 12, input_channels: int = 1):
        super(DepthwiseSeparableConv, self).__init__()
        # standard convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=10, stride=2, padding=2)

        # depthwise convolutional layer
        self.depthwise1 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, groups=64
        )
        self.pointwise1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=1)

        self.depthwise2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, groups=64
        )
        self.pointwise2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=1)

        self.depthwise3 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, groups=64
        )
        self.pointwise3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=1)

        self.depthwise4 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, groups=64
        )
        self.pointwise4 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=1)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.pointwise1(self.depthwise1(x)))
        x = F.relu(self.pointwise2(self.depthwise2(x)))
        x = F.relu(self.pointwise3(self.depthwise3(x)))
        x = F.relu(self.pointwise4(self.depthwise4(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
