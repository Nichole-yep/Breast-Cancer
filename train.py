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

    # 3. 初始化网络与工具
    model = OurBreastCancerNet(pretrained=True).to(DEVICE)
    criterion = DBDSLoss(max_epochs=EPOCHS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    os.makedirs("results/weights", exist_ok=True)
    best_val_dice = 0.0 

    print("\n 正在进行训练前预检查 (Sanity Check)...")
    model.eval()
    with torch.no_grad():
        chk_images, chk_masks, _ = next(iter(val_loader))
        chk_images = chk_images.to(DEVICE)
        chk_preds = model(chk_images)[-1]
        
        chk_target_h, chk_target_w = chk_masks.shape[-2], chk_masks.shape[-1]
        chk_preds = F.interpolate(chk_preds, size=(chk_target_h, chk_target_w), mode='bilinear', align_corners=False)
        chk_preds_bin = (torch.sigmoid(chk_preds) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
        
        if chk_masks.dim() == 4:
            chk_masks = chk_masks.squeeze(1)
        # 强制将标签二值化 (防255变成0)
        chk_masks_np = (chk_masks.cpu().numpy() > 0).astype(np.uint8) 
        
        chk_metrics = SegmentationMetrics(num_classes=2)
        chk_metrics.update_with_boundary(chk_preds_bin[0], chk_masks_np[0])
        print(" 预检查通过！Dice计算逻辑没报错，开始正式训练！\n")

    # 4. 训练与验证大循环
    for epoch in range(EPOCHS):
        # 训练阶段 
        model.train() 
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train")
        
        for images, masks, edges in train_bar:
            images, masks, edges = images.to(DEVICE), masks.to(DEVICE), edges.to(DEVICE)
            optimizer.zero_grad()
            preds_list = model(images)
            loss = criterion(preds_list, masks, edge_mask=edges, current_epoch=epoch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段 
        model.eval()
        val_metrics = SegmentationMetrics(num_classes=2)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Val  ")
            for images, masks, edges in val_bar:
                images = images.to(DEVICE)
                
                # 处理真实标签
                target_h, target_w = masks.shape[-2], masks.shape[-1] 
                if masks.dim() == 4:
                    masks = masks.squeeze(1)
                masks_np = (masks.cpu().numpy() > 0).astype(np.uint8)
                
                # 前向传播与尺寸放大
                preds_list = model(images)
                final_pred_logits = F.interpolate(preds_list[-1], size=(target_h, target_w), mode='bilinear', align_corners=False)
                final_pred_probs = torch.sigmoid(final_pred_logits)
                if final_pred_probs.dim() == 4:
                    final_pred_probs = final_pred_probs.squeeze(1)
                final_pred_binary = (final_pred_probs > 0.5).cpu().numpy().astype(np.uint8)
                
                # 逐样本更新
                for i in range(images.size(0)): 
                    val_metrics.update_with_boundary(final_pred_binary[i], masks_np[i])
                    
        # 获取本 Epoch 的分数
        scores = val_metrics.get_scores()
        val_dice = scores['dice']
        val_iou = scores['iou']
        val_hd95 = scores['hd95_mean']
        
        print(f" Epoch [{epoch+1}] 成绩单 | Train Loss: {avg_train_loss:.4f}")
        print(f" 验证集表现 -> Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | HD95: {val_hd95:.2f} 像素")

        # 保存最佳模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_path = "results/weights/best_our_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f" 最佳模型 (最佳 Dice: {best_val_dice:.4f}) 已保存！\n")
        else:
            print("\n")
        
if __name__ == '__main__':
    train_and_validate()