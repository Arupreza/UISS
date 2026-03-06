import torch
import torch.nn as nn


class StdConvPWBlock(nn.Module):
    """
    Standard 3x3 convolution + pointwise 1x1 convolution block.
    """
    def __init__(self, inp, oup, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            # Standard 3x3 convolution
            nn.Conv2d(
                inp, oup,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),

            # Pointwise 1x1 convolution
            nn.Conv2d(
                oup, oup,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LCNN_AggressiveWidth(nn.Module):
    """
    L-CNN (aggressive width)

    Table description:
    - Mixing: StdConv (3x3) + PW
    - Channel progression: 8 -> 32 -> 128 -> 128
    - 4 stages (s=2)
    - GAP + Linear
    """
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()

        self.stage1 = StdConvPWBlock(in_channels, 8, stride=2)   # 1 -> 8
        self.stage2 = StdConvPWBlock(8, 32, stride=2)            # 8 -> 32
        self.stage3 = StdConvPWBlock(32, 128, stride=2)          # 32 -> 128
        self.stage4 = StdConvPWBlock(128, 128, stride=2)         # 128 -> 128

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x