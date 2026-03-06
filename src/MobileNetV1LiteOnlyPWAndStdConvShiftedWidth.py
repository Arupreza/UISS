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


class LCNN_Shifted(nn.Module):
    """
    L-CNN (shifted)

    Table description:
    - Mixing: StdConv (3x3) + PW
    - Channel progression: 16 -> 16 -> 32 -> 64
    - 4 stages (s=2)
    - GAP + Linear
    """
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()

        self.stage1 = StdConvPWBlock(in_channels, 16, stride=2)  # 1 -> 16
        self.stage2 = StdConvPWBlock(16, 16, stride=2)           # 16 -> 16
        self.stage3 = StdConvPWBlock(16, 32, stride=2)           # 16 -> 32
        self.stage4 = StdConvPWBlock(32, 64, stride=2)           # 32 -> 64

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
