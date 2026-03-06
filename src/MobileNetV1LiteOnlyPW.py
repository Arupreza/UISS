import torch
import torch.nn as nn


class PointwiseConv(nn.Module):
    """
    Pointwise-only 1x1 convolution block.
    """
    def __init__(self, inp, oup, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                inp, oup,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LCNN_RemovedStdConv_DW(nn.Module):
    """
    L-CNN (Removed StdConv + DW)

    Table description:
    - Mixing: PW
    - Channel progression: 8 -> 16 -> 32 -> 64
    - 4 stages (s=2)
    - GAP + Linear

    Since StdConv and DW are removed, only pointwise (1x1) convolutions remain.
    """
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()

        # Stage 1: 1 -> 8
        self.stage1 = PointwiseConv(in_channels, 8, stride=2)

        # Stage 2: 8 -> 16
        self.stage2 = PointwiseConv(8, 16, stride=2)

        # Stage 3: 16 -> 32
        self.stage3 = PointwiseConv(16, 32, stride=2)

        # Stage 4: 32 -> 64
        self.stage4 = PointwiseConv(32, 64, stride=2)

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