import torch
import torch.nn as nn

class DepthwiseOnlyConv(nn.Module):
    """
    Depthwise-only conv block: per-channel 3x3 conv (no pointwise mixing).
    Constraint: out_channels == in_channels.
    """
    def __init__(self, channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1,
                    groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseOnlyNet(nn.Module):
    """
    Depthwise-only network: channel count stays constant throughout.
    """
    def __init__(self, num_classes=10, in_channels=1, base_channels=32):
        super().__init__()

        # You still need a normal conv once to go from 1 channel -> base_channels,
        # otherwise a depthwise conv with 1 channel is extremely limited.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Depthwise-only blocks (channel stays base_channels)
        # You can choose where to downsample (stride=2).
        self.features = nn.Sequential(
            DepthwiseOnlyConv(base_channels, stride=1),
            DepthwiseOnlyConv(base_channels, stride=2),
            DepthwiseOnlyConv(base_channels, stride=1),
            DepthwiseOnlyConv(base_channels, stride=2),
            DepthwiseOnlyConv(base_channels, stride=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x