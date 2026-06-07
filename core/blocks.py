import torch
from torch import nn
from torch.nn import functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out) 
    
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    attn = CBAM(64)
    y = attn(x)
    print(y.shape)

class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6)):
        super(PPM, self).__init__()

        # 每个分支的通道数降维为输入的 1/4
        reduction_dim = in_channels // len(sizes)

        self.stages = []

        # 4 个池化分支
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, reduction_dim, size) for size in sizes]
        )

        # 拼接后的总通道数 = 原通道 + 4个分支的通道数
        concat_channels = in_channels + (reduction_dim * len(sizes))

        # 最后的融合卷积 (Bottleneck)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_stage(self, in_channels, out_channels, size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(size, size)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 记住原始输入的宽高
        h, w = x.size(2), x.size(3)
        
        # 第一部分：保留原始特征图
        out = [x]
        
        # 第二部分：依次经过 4 个池化分支，并放大回原始宽高
        for stage in self.stages:
            pooled = stage(x)
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            out.append(upsampled)
            
        # 第三部分：在通道维度上合并它们 (dim=1)
        out = torch.cat(out, dim=1)
        
        # 第四部分：融合并输出目标通道数
        return self.bottleneck(out)