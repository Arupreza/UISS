import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    MobileNetV1 block: depthwise 3x3 conv + pointwise 1x1 conv
    """
    def __init__(self, inp, oup, stride=1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1,
                    groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNetV1(nn.Module):
    """
    "Pure" MobileNetV1-like architecture (depthwise separable convs),
    adapted for 1-channel input and small classification.
    """
    def __init__(self, num_classes=10, width_mult=1.0, in_channels=1):
        super().__init__()

        def c(ch):
            return max(1, int(ch * width_mult))

        # Standard MobileNetV1 stem is 3x3 conv stride=2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(32)),
            nn.ReLU(inplace=True),
        )

        # Classic MobileNetV1 block configuration:
        # (out_channels, stride)
        cfg = [
            (64,  1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (1024, 2),
            (1024, 1),
        ]

        layers = []
        in_ch = c(32)
        for out_ch, s in cfg:
            layers.append(DepthwiseSeparableConv(in_ch, c(out_ch), stride=s))
            in_ch = c(out_ch)

        self.features = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x