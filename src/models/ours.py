
# AUTO PATH FIX FOR FINAL GITHUB STRUCTURE
from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
for _p in [_PROJECT_ROOT, _PROJECT_ROOT / "src"]:
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
# END AUTO PATH FIX
import torch
import torch.nn as nn

from src.core.backbones import EfficientNetEncoder
from src.core.blocks import PPM, CBAM
from src.core.decoders import DeepSupervisionDecoder

class OurBreastCancerNet(nn.Module):
    """
    终极架构：EfficientNet-B0 + PPM多尺度桥接 + 深监督U-Net解码器
    专门针对乳腺超声肿瘤：尺度多变、边界模糊设计。
    """
    def __init__(self, model_name='efficientnet_b0', pretrained=True, num_classes=1):
        super(OurBreastCancerNet, self).__init__()
        
        # 特征提取器 (Encoder)
        # 实例化 Backbone，提取 5 个尺度的特征图
        self.encoder = EfficientNetEncoder(model_name=model_name, pretrained=pretrained)
        
        # 获取 EfficientNet 每一层的输出通道数 (对于 b0，默认是 [16, 24, 40, 112, 320])
        enc_channels = self.encoder.out_channels
        
        # 多尺度桥接模块 (Bridge / PPM)
        # PPM 接在网络的最深层，即 enc_channels[4] (320通道)
        # 为了让解码器好处理， PPM 的输出通道数和输入保持一致 (依然是 320)
        self.ppm = PPM(in_channels=enc_channels[4], out_channels=enc_channels[4])

        # 在PPM后添加CBAM注意力模块
        self.cbam_ppm = CBAM(enc_channels[4])
        
        # 深监督解码器 (Decoder)
        # 将通道数列表传给解码器，它会自动对齐通道，并准备好输出 4 个预测图
        self.decoder = DeepSupervisionDecoder(encoder_channels=enc_channels, num_classes=num_classes)

    def forward(self, x):
        """
        前向传播
        """
        # 1. 图片进入 EfficientNet，变成 5 种不同分辨率的特征图
        # features = [feat0(最大), feat1, feat2, feat3, feat4(最小最深)]
        features = self.encoder(x)
        
        # 2. 拿出最深处、最抽象的特征图 (feat4)，送进 PPM 进行“多尺度视野拓展”
        # PPM 会捕捉超大肿瘤和微小肿瘤的全局信息
        feat4_ppm = self.ppm(features[4])

        # 3. 在PPM后应用CBAM注意力
        feat4_ppm = self.cbam_ppm(feat4_ppm)
        
        # 4. 把增强后的 feat4_ppm 塞回列表中，替换掉原来的 feat4
        features[4] = feat4_ppm
        
        # 5. 把这 5 个特征图给深监督解码器
        # 解码器会一层层放大，并输出 4 张草稿/成品图
        preds_list = self.decoder(features)
        
        # 返回这 4 张图 (形状分别是原图的 1/8, 1/4, 1/2 和 1/1 大小)
        # 它们将对接 loss.py 里的 DBDS 动态边界深监督
        return preds_list

if __name__ == '__main__':
    # 模拟一张预处理好的乳腺超声图 (Batch=2, Channel=3, Height=256, Width=256)
    dummy_image = torch.randn(2, 3, 256, 256)
    
    print("正在加载 OurBreastCancerNet...")
    model = OurBreastCancerNet(pretrained=False) # 测试时先不下载预训练权重，跑得快
    
    # 把图片喂给网络
    print("正在进行前向传播推理...")
    outputs = model(dummy_image)
    
    # 打印输出结果，验证我们的深监督是否生效
    print("\n✅ 网络输出成功！包含以下 4 个深监督预测图：")
    for i, out in enumerate(outputs):
        print(f"-> 预测图 {i+1} (草稿/成品) 的形状: {out.shape}")