# models/base_effnet.py
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class BaseEffNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        # 简易解码：把最后一层特征上采样回原图
        self.head = nn.Conv2d(1280, 1, 1)

    def forward(self, x):
        features = self.backbone.extract_features(x)
        return self.head(features)