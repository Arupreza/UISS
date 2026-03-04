# lcnn_lib.py
# Simple library file: define LCNN + helper to load weights.

import torch
import torch.nn as nn


class LCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)

        self.conv_pw1 = self._pointwise_conv(8, 16, stride=1)
        self.conv_pw2 = self._pointwise_conv(16, 32, stride=1)
        self.conv_pw3 = self._pointwise_conv(32, 64, stride=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _pointwise_conv(self, inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv_pw1(x)
        x = self.conv_pw2(x)
        x = self.conv_pw3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)