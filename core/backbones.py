import torch.nn as nn
try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    print("Please pip install efficientnet-pytorch")

class EfficientNetBackbone(nn.Module):
    def __init__(self, name='efficientnet-b0', pretrained=True):
        super().__init__()
        if pretrained:
            self.net = EfficientNet.from_pretrained(name)
        else:
            self.net = EfficientNet.from_name(name)

    def forward(self, x):
        # 提取多层特征 (用于U-Net跳连)
        # EfficientNet 自带 extract_features
        features = self.net.extract_features(x)
        return features