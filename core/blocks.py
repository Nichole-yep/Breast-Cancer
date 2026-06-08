import torch
from torch import nn
from torch.nn import functional as F

class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6)):
        super().__init__()
        
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, size) for size in sizes]
        )
        
        total_channels = in_channels + out_channels * len(sizes)
        self.bottleneck = nn.Conv2d(total_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, in_channels, out_channels, size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(size, size)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        
        # 对每个stage进行插值并收集
        priors = [
            F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=False)
            for stage in self.stages
        ]
        
        # 拼接原特征 + 多尺度特征
        out = torch.cat(priors + [x], dim=1)
        
        # 1×1 卷积融合
        out = self.bottleneck(out)
        
        return self.relu(out)