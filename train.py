# train.py
import torch
from models.base_effnet import BaseEffNet

model = BaseEffNet()
x = torch.randn(2, 3, 256, 256)  # 轻量尺寸
y = model(x)
loss = y.sum()
loss.backward()
print("✅ 模型能跑！")
