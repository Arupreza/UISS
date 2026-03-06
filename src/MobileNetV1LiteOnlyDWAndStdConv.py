import torch
import torch.nn as nn


class StdConvDWBlock(nn.Module):
    """
    Standard 3x3 convolution + depthwise 3x3 convolution block.
    Pointwise convolution is removed.
    """
    def __init__(self, inp, oup, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            # Standard 3x3 conv: handles channel change
            nn.Conv2d(
                inp, oup,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),

            # Depthwise 3x3 conv: no channel mixing
            nn.Conv2d(
                oup, oup,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=oup,
                bias=False
            ),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LCNN_RemovedPW(nn.Module):
    """
    L-CNN (Removed PW)

    Table description:
    - Mixing: StdConv (3x3) + DW
    - Channel progression: 8 -> 16 -> 32 -> 64
    - 4 stages (s=2)
    - GAP + Linear
    """
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()

        self.stage1 = StdConvDWBlock(in_channels, 8, stride=2)   # 1 -> 8
        self.stage2 = StdConvDWBlock(8, 16, stride=2)            # 8 -> 16
        self.stage3 = StdConvDWBlock(16, 32, stride=2)           # 16 -> 32
        self.stage4 = StdConvDWBlock(32, 64, stride=2)           # 32 -> 64

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x