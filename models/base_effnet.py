import torch.nn as nn
from core.backbones import EfficientNetBackbone

class BaseEffNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = EfficientNetBackbone('efficientnet-b0')
        # 简易解码头：把特征图拉回原尺寸
        self.head = nn.Sequential(
            nn.Conv2d(1280, 256, 3, padding=1), # Eff-B0 最后一层通道是1280
            nn.ReLU(),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True), # 粗略上采样
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out