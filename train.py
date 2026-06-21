import os
import torch
import torch.optim as optim
from tqdm import tqdm 
import numpy as np
import torch.nn.functional as F 
import matplotlib.pyplot as plt  # 新增：导入matplotlib

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
    criterion = DBDSLoss(max_epochs=EPOCHS, pos_weight=15.0).to(DEVICE)  # 添加 pos_weight
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # 新增：
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    os.makedirs("results/logs", exist_ok=True)
    csv_log_path = "results/logs/our_model_training_log.csv"

    # 写入 CSV 表头
    import csv
    with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_dice",
            "val_iou",
            "val_accuracy",
            "val_sensitivity",
            "val_specificity",
            "val_hd95_mean"
        ])

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

    # 新增：用于存储训练过程中的指标
    train_losses = []
    val_dices = []
    val_ious = []
    val_hd95s = []

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
        train_losses.append(avg_train_loss)  # 新增：记录训练损失
        
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
        
        # 新增：记录验证指标
        val_dices.append(val_dice)
        val_ious.append(val_iou)
        val_hd95s.append(val_hd95)
        
        print(f" Epoch [{epoch+1}] 成绩单 | Train Loss: {avg_train_loss:.4f}")
        print(f" 验证集表现 -> Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | HD95: {val_hd95:.2f} 像素")

    # ========================= 新增：写 CSV 日志 =========================
    with open(csv_log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1,
            f"{avg_train_loss:.6f}",
            f"{val_dice:.6f}",
            f"{val_iou:.6f}",
            f"{scores['accuracy']:.6f}",
            f"{scores['sensitivity']:.6f}",
            f"{scores['specificity']:.6f}",
            f"{val_hd95:.6f}"
        ])
    # ================================================================


        # 保存最佳模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_path = "results/weights/best_our_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f" 最佳模型 (最佳 Dice: {best_val_dice:.4f}) 已保存！\n")
        else:
            print("\n")
        scheduler.step()
    
    # 新增：绘制训练曲线
    print("\n 正在生成训练曲线图...")
    os.makedirs("results/plots", exist_ok=True)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 子图1：训练损失
    axes[0, 0].plot(range(1, EPOCHS + 1), train_losses, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss Curve')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2：Dice系数
    axes[0, 1].plot(range(1, EPOCHS + 1), val_dices, 'g-', linewidth=2, label='Validation Dice')
    axes[0, 1].axhline(y=best_val_dice, color='r', linestyle='--', alpha=0.7, label=f'Best Dice: {best_val_dice:.4f}')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Coefficient')
    axes[0, 1].set_title('Dice Coefficient Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3：IoU
    axes[1, 0].plot(range(1, EPOCHS + 1), val_ious, 'orange', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_title('IoU Curve')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 子图4：HD95
    axes[1, 1].plot(range(1, EPOCHS + 1), val_hd95s, 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('HD95 (pixels)')
    axes[1, 1].set_title('HD95 Curve')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📄 训练日志已保存至: {csv_log_path}")
    
    print(f" 训练曲线图已保存至: results/plots/training_curves.png")
    print(f" 最佳验证Dice: {best_val_dice:.4f}")

    # 5. 在测试集上评估最佳模型
    test_best_model(model, test_loader, DEVICE)
    
    return model, best_val_dice

def test_best_model(model, test_loader, DEVICE):
    """在测试集上评估最佳模型"""
    print("\n" + "="*60)
    print(" 开始在 Test 集上评估最佳模型 ")
    print("="*60)
    
    # 加载最佳模型权重
    weights_path = "results/weights/best_our_model.pth"
    print(f"正在加载最佳模型权重: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    
    metrics = SegmentationMetrics(num_classes=2)
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing Best Model")
        for images, masks, edges in test_bar:
            images = images.to(DEVICE)
            
            # 处理真实标签
            target_h, target_w = masks.shape[-2], masks.shape[-1]
            if masks.dim() == 4:
                masks = masks.squeeze(1)
            masks_np = (masks.cpu().numpy() > 0).astype(np.uint8)
            
            # 推理
            preds_list = model(images)
            
            # 先放大，再二值化
            final_pred_logits = F.interpolate(preds_list[-1], size=(target_h, target_w), mode='bilinear', align_corners=False)
            final_pred_binary = (torch.sigmoid(final_pred_logits) > 0.65).squeeze(1).cpu().numpy().astype(np.uint8)
            
            # 更新评估指标
            for i in range(images.size(0)): 
                metrics.update_with_boundary(final_pred_binary[i], masks_np[i])
                
    # 打印最终成绩单
    scores = metrics.get_scores()
    print("\n" + "="*50)
    print(" 测试集 (Test Set) 最终成绩单 ")
    print("="*50)
    print(f"Dice 系数:     {scores['dice']:.4f}")
    print(f"IoU (Jaccard): {scores['iou']:.4f}")
    print(f"准确率:        {scores['accuracy']:.4f}")
    print(f"灵敏度:        {scores['sensitivity']:.4f}")
    print(f"特异度:        {scores['specificity']:.4f}")
    print(f"HD95:          {scores['hd95_mean']:.2f} 像素")
    print("="*50)
    
    # 保存测试结果到文件
    save_test_results(scores)
    
    return scores

def save_test_results(scores):
    """保存测试结果到文件"""
    os.makedirs("results", exist_ok=True)
    
    with open("results/test_results.txt", "w") as f:
        f.write("="*50 + "\n")
        f.write(" 测试集 (Test Set) 最终成绩单 \n")
        f.write("="*50 + "\n")
        f.write(f"Dice 系数:     {scores['dice']:.4f}\n")
        f.write(f"IoU (Jaccard): {scores['iou']:.4f}\n")
        f.write(f"准确率:        {scores['accuracy']:.4f}\n")
        f.write(f"灵敏度:        {scores['sensitivity']:.4f}\n")
        f.write(f"特异度:        {scores['specificity']:.4f}\n")
        f.write(f"HD95:          {scores['hd95_mean']:.2f} 像素\n")
        f.write("="*50 + "\n")
    
    print(f" 测试结果已保存至: results/test_results.txt")
    
if __name__ == '__main__':
    train_and_validate()
