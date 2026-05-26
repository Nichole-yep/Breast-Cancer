import torch
import torch.nn as nn
import torchvision.models as models
from CBAM import CBAM 

# 辅助模块：定义一个基本的“上采样+卷积”模块
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # 1. 上采样层：负责把图片放大一倍，同时减少通道数
        self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        
        # 2. 两个卷积层：负责在拼接后融合特征
        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels + mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_features):
        # x 是上一层传上来的小特征图，skip_features 是从 encoder 传过来的大特征图
        x = self.up(x) # 放大 x
        # 将放大的 x 和传过来的 skip_features 在通道维度（dim=1）拼接起来
        x = torch.cat([x, skip_features], dim=1) 
        x = self.conv(x) # 卷积融合
        return x

# 主模型：核心 LEResUNet
class LEResUNet(nn.Module):
    def __init__(self):
        super(LEResUNet, self).__init__()
        
        resnet = models.resnet18(pretrained=True)
        
        # 砍下需要的层作为 Encoder
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 输出尺寸 [B, 64, 128, 128]
        self.encoder1 = nn.Sequential(resnet.maxpool, resnet.layer1)         # 输出尺寸 [B, 64, 64, 64]
        self.encoder2 = resnet.layer2                                        # 输出尺寸 [B, 128, 32, 32]
        self.encoder3 = resnet.layer3                                        # 输出尺寸 [B, 256, 16, 16]
        self.encoder4 = resnet.layer4                                        # 输出尺寸 [B, 512, 8, 8] (最底层)

        # 2. 插入CBAM 过滤器
        # 参数是对应特征图的通道数
        self.cbam3 = CBAM(channel=256) 
        self.cbam2 = CBAM(channel=128)
        self.cbam1 = CBAM(channel=64)
        self.cbam0 = CBAM(channel=64)

        # DecoderBlock(输入通道, 上采样后的通道, 融合后的最终通道)
        self.decoder4 = DecoderBlock(512, 256, 256) # 负责处理 e4 和过滤后的 e3
        self.decoder3 = DecoderBlock(256, 128, 128) # 负责处理上一层的输出和过滤后的 e2
        self.decoder2 = DecoderBlock(128, 64, 64)   # 负责处理上一层的输出和过滤后的 e1
        self.decoder1 = DecoderBlock(64, 64, 64)    # 负责处理上一层的输出和过滤后的 e0
        
        # 3. 再做一次上采样把 128x128 变成原图 256x256，并输出单通道预测概率图
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1) # 1 个通道代表肿瘤预测

    def forward(self, x):
        e0 = self.encoder0(x)  # [B, 64, 128, 128]
        e1 = self.encoder1(e0) # [B, 64, 64, 64]
        e2 = self.encoder2(e1) # [B, 128, 32, 32]
        e3 = self.encoder3(e2) # [B, 256, 16, 16]
        e4 = self.encoder4(e3) # [B, 512, 8, 8] 

        # 过滤器 (CBAM) 
        # 给这些要跳跃过去的特征图“净化”一下
        e3_filtered = self.cbam3(e3)
        e2_filtered = self.cbam2(e2)
        e1_filtered = self.cbam1(e1)
        e0_filtered = self.cbam0(e0)

        # d4 是拿最底层的 e4 和 净化后的 e3 拼装
        d4 = self.decoder4(e4, e3_filtered) # 输出 [B, 256, 16, 16]
        
        # d3 是拿 d4 和 净化后的 e2 拼装
        d3 = self.decoder3(d4, e2_filtered) # 输出 [B, 128, 32, 32]
        
        # d2 是拿 d3 和 净化后的 e1 拼装
        d2 = self.decoder2(d3, e1_filtered) # 输出 [B, 64, 64, 64]
        
        # d1 是拿 d2 和 净化后的 e0 拼装
        d1 = self.decoder1(d2, e0_filtered) # 输出 [B, 64, 128, 128]

        # 最后出图 
        out = self.final_up(d1)         # 放回 [B, 32, 256, 256]
        out = self.final_conv(out)      # 压扁成单通道 [B, 1, 256, 256]
        
        # 使用 Sigmoid 让输出的所有数值都在 0~1 之间（代表概率）
        return torch.sigmoid(out)