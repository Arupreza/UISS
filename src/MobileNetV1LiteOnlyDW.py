import torch
import torch.nn as nn


class DepthwiseOnlyConv(nn.Module):
    """
    Depthwise-only 3x3 convolution block.
    To allow channel progression (e.g., 8 -> 16), this uses a depthwise-style
    grouped convolution where groups=in_channels.

    Note:
    - Pure depthwise conv strictly keeps in_channels == out_channels.
    - For table-style ablation, this implementation keeps the 'DW-only' idea
      while still allowing channel expansion.
    """
    def __init__(self, inp, oup, stride=1):
        super().__init__()

        # For grouped conv, out_channels must be divisible by groups.
        if oup % inp != 0:
            raise ValueError(f"out_channels ({oup}) must be divisible by in_channels ({inp}) for groups=inp.")

        self.block = nn.Sequential(
            nn.Conv2d(
                inp, oup,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=inp,   # depthwise-style grouped convolution
                bias=False
            ),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LCNN_RemovedStdConv_PW(nn.Module):
    """
    MobileNetV1-lite (DW-only ablation)

    Table description:
    - Mixing: PW
    - Channel progression: 8 -> 16 -> 32 -> 64
    - 4 stages (s=2)
    - GAP + Linear
    """
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()

        # Stage 1: standard stem to 8 channels
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, 8,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        # Stages 2-4: depthwise-only grouped convs
        self.stage2 = DepthwiseOnlyConv(8, 16, stride=2)
        self.stage3 = DepthwiseOnlyConv(16, 32, stride=2)
        self.stage4 = DepthwiseOnlyConv(32, 64, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)      # -> 8
        x = self.stage2(x)    # -> 16
        x = self.stage3(x)    # -> 32
        x = self.stage4(x)    # -> 64
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x