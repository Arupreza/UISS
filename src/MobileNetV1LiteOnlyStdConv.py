import torch
import torch.nn as nn


class StdConvBlock(nn.Module):
    """
    Standard 3x3 convolution block.
    """
    def __init__(self, inp, oup, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                inp, oup,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LCNN_RemovedDW_PW(nn.Module):
    """
    L-CNN (Removed DW + PW)

    Table description:
    - Mixing: StdConv 3x3
    - Channel progression: 8 -> 16 -> 32 -> 64
    - 4 stages (s=2)
    - GAP + Linear
    """
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()

        # Stage 1: 1 -> 8
        self.stage1 = StdConvBlock(in_channels, 8, stride=2)

        # Stage 2: 8 -> 16
        self.stage2 = StdConvBlock(8, 16, stride=2)

        # Stage 3: 16 -> 32
        self.stage3 = StdConvBlock(16, 32, stride=2)

        # Stage 4: 32 -> 64
        self.stage4 = StdConvBlock(32, 64, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stage1(x)   # -> 8
        x = self.stage2(x)   # -> 16
        x = self.stage3(x)   # -> 32
        x = self.stage4(x)   # -> 64
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x