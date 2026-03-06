import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    MobileNetV1-style block:
    depthwise 3x3 + pointwise 1x1
    """
    def __init__(self, inp, oup, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            # Depthwise
            nn.Conv2d(
                inp, inp,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=inp,
                bias=False
            ),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            # Pointwise
            nn.Conv2d(
                inp, oup,
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


class MobileNetV1Lite(nn.Module):
    """
    MobileNetV1-lite (baseline, context)

    Target table description:
    - Mixing: DW + PW
    - Channel progression: 8 -> 16 -> 32 -> 64
    - 4 stages (s=2)
    - GAP + Linear
    """
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()

        # Stem: first stage to 8 channels
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

        # 4 stages total in the compact interpretation:
        # stem(->8), then DW+PW stages to 16, 32, 64
        self.stage2 = DepthwiseSeparableConv(8, 16, stride=2)
        self.stage3 = DepthwiseSeparableConv(16, 32, stride=2)
        self.stage4 = DepthwiseSeparableConv(32, 64, stride=2)

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