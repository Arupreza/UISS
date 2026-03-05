import torch
import torch.nn as nn

NUM_CLASSES = 11
# ------------------------------------------------------------
# 7) Model (Exact STParNet)
# ------------------------------------------------------------
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        )

class LiPar_STParNet(nn.Module):
    """
    Spatial DWParNet on x_img (B,3,9,9)
    Temporal LSTM on x_seq (B,27,9)
    Final logits = (spatial_logits + temporal_logits) / 2
    """
    def __init__(self, input_size=9, hidden_size=32, num_layers=2, num_classes=NUM_CLASSES):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Temporal branch
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, num_classes)

        # Spatial branches
        self.branch1 = nn.Sequential(
            ConvBNReLU(3, 64, kernel_size=1, stride=1, groups=1),
            ConvBNReLU(64, 64, kernel_size=3, stride=8, groups=64),
        )

        self.branch2 = nn.Sequential(
            ConvBNReLU(3, 128, kernel_size=1, stride=1, groups=1),
            ConvBNReLU(128, 128, kernel_size=3, stride=4, groups=128),
            ConvBN(128, 256, kernel_size=1, stride=1, groups=1),
            ConvBNReLU(256, 256, kernel_size=3, stride=2, groups=256),
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=1, stride=1, groups=1),
            ConvBNReLU(32, 32, kernel_size=3, stride=2, groups=32),
            ConvBN(32, 96, kernel_size=1, stride=1, groups=1),
            ConvBNReLU(96, 96, kernel_size=3, stride=2, groups=96),
            ConvBN(96, 192, kernel_size=1, stride=1, groups=1),
            ConvBNReLU(192, 192, kernel_size=3, stride=2, groups=192),
        )

        self.conv = ConvBN(512, 64, kernel_size=3, stride=1, groups=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x_img, x_seq):
        B = x_img.size(0)
        device = x_img.device

        # Temporal logits
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
        y_out, _ = self.lstm(x_seq, (h0, c0))
        temporal_logits = self.fc1(y_out[:, -1, :])

        # Spatial logits
        b1 = self.branch1(x_img)
        b2 = self.branch2(x_img)
        b3 = self.branch3(x_img)
        x = torch.cat([b1, b2, b3], dim=1)
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        spatial_logits = self.fc2(x)

        return (spatial_logits + temporal_logits) * 0.5