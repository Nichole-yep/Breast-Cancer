import torch
import torch.nn as nn
import yaml
from models.base_effnet import BaseEffNet

# 1. 加载配置
with open("configs/default.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# 2. 造模型
device = "cpu"
model = BaseEffNet(cfg['MODEL']['NUM_CLASSES']).to(device)

# 3. 造假数据 (模拟 BUSI 图片)
dummy_img = torch.randn(2, 3, cfg['MODEL']['IMG_SIZE'], cfg['MODEL']['IMG_SIZE']).to(device)
dummy_mask = torch.randn(2, 1, cfg['MODEL']['IMG_SIZE'], cfg['MODEL']['IMG_SIZE']).to(device)

# 4. 前向传播
pred = model(dummy_img)
print("Output shape:", pred.shape) # 期望: [2, 1, 256, 256]

# 5. 计算 Loss (简单版)
criterion = nn.BCEWithLogitsLoss()
loss = criterion(pred, dummy_mask)
print("Loss:", loss.item())

# 6. 反向传播
loss.backward()
print("✅ Backward pass successful!")