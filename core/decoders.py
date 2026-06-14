import torch
import torch.nn as nn
import torch.nn.functional as F

class UpBlock(nn.Module):
    """
    基础的上采样积木块：放大底层图 -> 拼接跳跃连接图 -> 两次卷积融合
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpBlock, self).__init__()
        # 融合后的总通道数 = 底层放大后的通道 + 旁边跳跃过来的通道
        concat_channels = in_channels + skip_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),  # 新增：随机丢弃 30% 的特征图
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3)   # 新增：再次丢弃
        )

    def forward(self, x, skip):
        # 1. 把底层特征放大到和 skip 一样大
        x_up = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        # 2. 在通道维度(dim=1)拼接
        x_concat = torch.cat([x_up, skip], dim=1)
        # 3. 卷积融合
        return self.conv(x_concat)

class DeepSupervisionDecoder(nn.Module):
    def __init__(self, encoder_channels, num_classes=1):
        """
        encoder_channels: 就是 EfficientNet 提取出来的通道数列表 [16, 24, 40, 112, 320]
        """
        super(DeepSupervisionDecoder, self).__init__()
        
        # 通道数按 256, 128, 64, 32 递减
        self.up1 = UpBlock(in_channels=encoder_channels[4], skip_channels=encoder_channels[3], out_channels=256)
        self.up2 = UpBlock(in_channels=256, skip_channels=encoder_channels[2], out_channels=128)
        self.up3 = UpBlock(in_channels=128, skip_channels=encoder_channels[1], out_channels=64)
        self.up4 = UpBlock(in_channels=64, skip_channels=encoder_channels[0], out_channels=32)
        
        # 深监督输出头 
        self.ds_head1 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.ds_head2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.ds_head3 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.ds_head4 = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, features):
        """
        features: 列表 [feat0, feat1, feat2, feat3, feat4]
        feat4 是最深层的图 (320通道)，feat0 是最浅层的图 (16通道)
        """
        # 这里的 features[4] 应该是 PPM 的输出
        x_bottom = features[4] 
        
        # 第1层解码 -> 输出草稿1
        d1 = self.up1(x_bottom, features[3])
        out1 = self.ds_head1(d1)
        
        # 第2层解码 -> 输出草稿2
        d2 = self.up2(d1, features[2])
        out2 = self.ds_head2(d2)
        
        # 第3层解码 -> 输出草稿3
        d3 = self.up3(d2, features[1])
        out3 = self.ds_head3(d3)
        
        # 第4层解码 -> 输出最终成品
        d4 = self.up4(d3, features[0])
        out4 = self.ds_head4(d4)
        
        # 把 4 个预测图打包成列表返回
        return [out1, out2, out3, out4]
