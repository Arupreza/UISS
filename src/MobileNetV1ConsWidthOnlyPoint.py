import torch
import torch.nn as nn


class PW_Only_ConstWidth(nn.Module):
    """
    Pointwise-only block (constant width):
    PW 1x1 (C->C), NO depthwise.
    """
    def __init__(self, C=32, stride=1):
        super().__init__()
        self.pw = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1, stride=stride, padding=0, bias=False),  # 32 -> 32
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.pw(x)


class MobileNetV1_ConstantWidth32_PWOnly(nn.Module):
    """
    Same as your constant-width model, but blocks are pointwise-only.
    Stem/strides/GAP/classifier remain the same.
    """
    def __init__(self, num_classes=10, in_channels=1, C=32, stem_stride=2):
        super().__init__()

        # Stem stays the same
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, kernel_size=3, stride=stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # Same stride schedule
        strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]

        # Replace DW+PW blocks with PW-only blocks
        self.features = nn.Sequential(*[PW_Only_ConstWidth(C=C, stride=s) for s in strides])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x