import torch
import torch.nn as nn
import timm

class EfficientNetEncoder(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super(EfficientNetEncoder, self).__init__()
        
        self.encoder = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            features_only=True,

            out_indices=(0, 1, 2, 3, 4) 
        )
        
        # EfficientNet-B0 默认的输出通道数，把这些数字传给 Decoder: 16, 24, 40, 112, 320
        self.out_channels = self.encoder.feature_info.channels()

    def forward(self, x):
        # features 里面装着 5 个大小不断减半的特征图
        features = self.encoder(x)
        return features

#  测试代码 
if __name__ == '__main__':
    model = EfficientNetEncoder()
    dummy_input = torch.randn(1, 3, 256, 256)
    feats = model(dummy_input)
    for i, f in enumerate(feats):
        print(f"特征层 {i} 的形状: {f.shape}")