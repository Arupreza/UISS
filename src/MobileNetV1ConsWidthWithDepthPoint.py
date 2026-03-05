import torch
import torch.nn as nn

class DW_PW_ConstWidth(nn.Module):
    """
    MobileNetV1 block (constant width):
    DW 3x3 (groups=C) + PW 1x1 (C->C), no channel expansion.
    """
    def __init__(self, C=32, stride=1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0, bias=False),  # 32 -> 32
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x


class MobileNetV1_ConstantWidth32(nn.Module):
    """
    MobileNetV1 (Constant Width): 32 channels everywhere (no expansion).
    Depthwise + Pointwise exist in every block.
    """
    def __init__(self, num_classes=10, in_channels=1, C=32, stem_stride=2):
        super().__init__()

        # Stem to reach constant width C
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, kernel_size=3, stride=stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # Stride schedule (MobileNetV1-like downsampling), but constant channels
        # Edit these if input is very small.
        strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]

        self.features = nn.Sequential(*[DW_PW_ConstWidth(C=C, stride=s) for s in strides])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x