# models/baseline_models.py
# Baseline segmentation models for BUSI breast ultrasound segmentation.
# Supports: U-Net, Attention U-Net, Lightweight DeepLabV3+.
# Output format: logits tensor with shape [B, 1, H, W].
#
# Put this file into your project folder:
# Breast-Cancer/models/baseline_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Conv-BN-ReLU twice."""
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.0):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downscaling with maxpool then DoubleConv."""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upscaling then DoubleConv."""
    def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad/resize if sizes differ due to odd input sizes.
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_x != 0 or diff_y != 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2]
            )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetBaseline(nn.Module):
    """Standard U-Net baseline."""
    def __init__(self, in_channels=3, num_classes=1, base_channels=32, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8, dropout=0.1)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor, dropout=0.1)

        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear, dropout=0.1)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


class AttentionGate(nn.Module):
    """Attention gate used in Attention U-Net."""
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        psi = self.relu(self.w_g(gate) + self.w_x(skip))
        psi = self.psi(psi)
        return skip * psi


class AttentionUp(nn.Module):
    """Upsampling block with attention-gated skip connection."""
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True, dropout=0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True) if bilinear else \
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.att = AttentionGate(gate_channels=in_channels, skip_channels=skip_channels, inter_channels=max(out_channels // 2, 1))
        self.conv = DoubleConv(in_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip_att = self.att(x, skip)
        x = torch.cat([skip_att, x], dim=1)
        return self.conv(x)


class AttentionUNetBaseline(nn.Module):
    """Attention U-Net baseline."""
    def __init__(self, in_channels=3, num_classes=1, base_channels=32, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8, dropout=0.1)
        self.down4 = Down(base_channels * 8, base_channels * 16, dropout=0.1)

        self.up1 = AttentionUp(base_channels * 16, base_channels * 8, base_channels * 8, bilinear, dropout=0.1)
        self.up2 = AttentionUp(base_channels * 8, base_channels * 4, base_channels * 4, bilinear)
        self.up3 = AttentionUp(base_channels * 4, base_channels * 2, base_channels * 2, bilinear)
        self.up4 = AttentionUp(base_channels * 2, base_channels, base_channels, bilinear)
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling block for DeepLabV3+."""
    def __init__(self, in_channels, out_channels=128, rates=(1, 6, 12, 18)):
        super().__init__()
        branches = []
        for rate in rates:
            if rate == 1:
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ))
            else:
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ))
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(rates), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        x = torch.cat(feats, dim=1)
        return self.project(x)


class DeepLabV3PlusBaseline(nn.Module):
    """
    Lightweight DeepLabV3+ style baseline.

    This is a runnable DeepLabV3+ style implementation with:
    encoder + ASPP + low-level feature decoder.
    It does not download external pretrained weights, so it is easier to run on CPU/offline.
    """
    def __init__(self, in_channels=3, num_classes=1, base_channels=32):
        super().__init__()
        self.stem = DoubleConv(in_channels, base_channels)
        self.enc1 = Down(base_channels, base_channels * 2)
        self.enc2 = Down(base_channels * 2, base_channels * 4)
        self.enc3 = Down(base_channels * 4, base_channels * 8, dropout=0.1)

        self.aspp = ASPP(base_channels * 8, out_channels=base_channels * 4)
        self.low_project = nn.Sequential(
            nn.Conv2d(base_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            DoubleConv(base_channels * 4 + 48, base_channels * 2),
            nn.Dropout2d(0.1),
            nn.Conv2d(base_channels * 2, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        low = self.stem(x)         # H, W
        x = self.enc1(low)         # H/2
        x = self.enc2(x)           # H/4
        x = self.enc3(x)           # H/8

        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_project(low)
        x = torch.cat([x, low], dim=1)
        logits = self.decoder(x)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


def get_baseline_model(model_name, in_channels=3, num_classes=1, base_channels=32):
    """
    Build a baseline model by name.

    Available names:
    - unet
    - attention_unet
    - deeplabv3plus
    """
    name = model_name.lower().replace("-", "_")
    if name in ["unet", "u_net", "u-net"]:
        return UNetBaseline(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels)
    if name in ["attention_unet", "attention_u_net", "attention-u-net", "attunet"]:
        return AttentionUNetBaseline(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels)
    if name in ["deeplabv3plus", "deeplabv3_plus", "deeplabv3+", "deeplab"]:
        return DeepLabV3PlusBaseline(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels)
    raise ValueError(
        f"Unknown model_name={model_name}. "
        "Use one of: unet, attention_unet, deeplabv3plus."
    )


if __name__ == "__main__":
    # Quick shape test
    for name in ["unet", "attention_unet", "deeplabv3plus"]:
        model = get_baseline_model(name)
        x = torch.randn(2, 3, 256, 256)
        y = model(x)
        print(name, y.shape)
