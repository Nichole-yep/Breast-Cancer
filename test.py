import torch
from core.blocks import PPM  

# 假设 EfficientNet-B0 最后一层的输出特征图是: [Batch=2, Channel=320, H=8, W=8]
dummy_features = torch.randn(2, 320, 8, 8)

# 初始化 PPM：输入是 EfficientNet 的 320 通道，我们希望输出 512 通道给 U-Net
ppm_block = PPM(in_channels=320, out_channels=512)

# 前向传播
output = ppm_block(dummy_features)

print("输入的形状:", dummy_features.shape)
print("经过 PPM 后的输出形状:", output.shape)