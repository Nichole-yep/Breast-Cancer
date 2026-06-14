import os
import torch
import torch.optim as optim
from tqdm import tqdm 
import numpy as np
import torch.nn.functional as F 

# 导入我们的模型和 Loss
from models.ours import OurBreastCancerNet
from loss import DBDSLoss, dice_loss
# 导入预处理模块
from preprocess.dataset import get_loaders
# 导入评估
from evaluate.eval import SegmentationMetrics

def train_and_validate():
    # 1. 超参数设置
    EPOCHS = 100
    BATCH_SIZE = 8       
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 正在使用计算设备: {DEVICE}")

    # 2. 调用 DataLoader
    print(" 正在加载真实数据集...")
    train_loader, val_loader, test_loader = get_loaders(
        train_csv='preprocess/train.csv', 
        val_csv='preprocess/val.csv', 
        test_csv='preprocess/test.csv',
        batch_size=BATCH_SIZE,
        use_lee=True,
        use_clahe=True
    )
    print(f" 数据加载成功！训练集batch数: {len(train_loader)}, 验证集batch数: {len(val_loader)}")

    # 3. 初始化网络与工具
    model = OurBreastCancerNet(pretrained=True).to(DEVICE)
    criterion = DBDSLoss(max_epochs=EPOCHS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    os.makedirs("results/weights", exist_ok=True)
    best_val_dice = 0.0 

    # Sanity Check (预检查) 
    print("🚀 正在进行训练前预检查 (Sanity Check)...")
    model.eval()
    with torch.no_grad():
        # 从验证集随便抽一个 batch
        try:
            chk_images, chk_masks, _ = next(iter(val_loader))
            chk_images = chk_images.to(DEVICE)
            
            chk_preds = model(chk_images)[-1]
            chk_target_h, chk_target_w = chk_masks.shape[-2], chk_masks.shape[-1]
            
            chk_preds = F.interpolate(chk_preds, size=(chk_target_h, chk_target_w), mode='bilinear', align_corners=False)
            chk_preds_bin = (torch.sigmoid(chk_preds) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
            
            if chk_masks.dim() == 4:
                chk_masks = chk_masks.squeeze(1)
            chk_masks_np = chk_masks.cpu().numpy().astype(np.uint8)
            
            chk_metrics = SegmentationMetrics(num_classes=2)
            # 只测第一张图看会不会崩
            chk_metrics.update_with_boundary(chk_preds_bin[0], chk_masks_np[0])
            print("✅ 预检查通过！维度对齐正常，模型准备好开始正式训练！")
        except Exception as e:
            print("❌ 预检查失败！请先解决以下报错：")
            raise e
    # =================================================================
    # 4. 训练与验证大循环
    for epoch in range(EPOCHS):
        model.train() # 开启训练模式
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train")
        for images, masks, edges in train_bar:
            images, masks, edges = images.to(DEVICE), masks.to(DEVICE), edges.to(DEVICE)
            
            # 1. 梯度清零
            optimizer.zero_grad()
            
            # 2. 前向传播：模型吐出 4 张深监督预测图
            # preds_list = [草稿1, 草稿2, 草稿3, 最终成品]
            preds_list = model(images)
            
            # 3. 计算误差：把 epoch 传进去，激活 DBDS 的动态机制！
            loss = criterion(preds_list, masks, edge_mask=edges, current_epoch=epoch)
            
            # 4. 反向传播与参数更新
            loss.backward()
            optimizer.step()
            
            # 累加 Loss 用于打印
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # 计算这个 epoch 的平均 Loss
        avg_train_loss = train_loss / len(train_loader)
        

    # 验证阶段 
    model.eval()
    
    val_metrics = SegmentationMetrics(num_classes=2)
    
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Val  ")
        # 接收三个变量（图片，金标准，真实边界）
        for images, masks, edges in val_bar:
            images = images.to(DEVICE)
            
             # 前向传播拿到预测结果
            preds_list = model(images)
            final_pred_logits = preds_list[-1]
            
            # 把 128x128 的网络输出，强行放大回金标准的 256x256
            final_pred_logits = F.interpolate(final_pred_logits, size=masks.shape[2:], mode='bilinear', align_corners=False)
            
            final_pred_probs = torch.sigmoid(final_pred_logits)
            
            # 加上 .squeeze(1)，把 (B, 1, H, W) 压成 (B, H, W)
            final_pred_binary = (final_pred_probs > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
            
            # 同样把真实标签也 squeeze(1) 压扁
            masks_np = masks.squeeze(1).cpu().numpy().astype(np.uint8)
                
    scores = val_metrics.get_scores()
    val_dice = scores['dice']
    val_iou = scores['iou']
    val_hd95 = scores['hd95_mean']
    
    print(f" Epoch [{epoch+1}] 成绩单 | Train Loss: {avg_train_loss:.4f}")
    print(f" 验证集表现 -> Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | HD95: {val_hd95:.2f} 像素")

    # -------- 保存最佳模型 (根据验证集 Dice) --------
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        save_path = "results/weights/best_our_model.pth"
        torch.save(model.state_dict(), save_path)
        print(f" 最佳模型 (最佳 Dice: {best_val_dice:.4f}) 模型已保存！\n")
    else:
        print("\n")
        

if __name__ == '__main__':
    train_and_validate()