
# AUTO PATH FIX FOR FINAL GITHUB STRUCTURE
from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
for _p in [_PROJECT_ROOT, _PROJECT_ROOT / "src"]:
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
# END AUTO PATH FIX
import os
import time  # [新增: 用于计算推理FPS]
import csv   # [新增: 用于保存测试集独立样本指标]
import torch
import torch.optim as optim
from tqdm import tqdm 
import numpy as np
import torch.nn.functional as F 
import matplotlib.pyplot as plt  
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score # [新增: ROC/PR曲线]

# 导入我们的模型和 Loss
from src.models.ours import OurBreastCancerNet
from scripts.loss import DBDSLoss, dice_loss
# 导入预处理模块
from src.data.dataset import get_loaders
# 导入评估
from utils.eval import SegmentationMetrics

# [新增: 导入 thop 用于算 FLOPs 和参数量]
from thop import profile

def train_and_validate():
    # 1. 超参数设置
    EPOCHS = 100
    BATCH_SIZE = 8       
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 正在使用计算设备: {DEVICE}")

    os.makedirs("outputs/results/val_preds", exist_ok=True)
    os.makedirs("outputs/results/plots", exist_ok=True) # 确保绘图文件夹存在

    # 2. 调用 DataLoader
    print(" 正在加载真实数据集...")
    train_loader, val_loader, test_loader = get_loaders(
        train_csv='src/data/train.csv', 
        val_csv='src/data/val.csv', 
        test_csv='src/data/test.csv',
        batch_size=BATCH_SIZE,
        use_lee=True,
        use_clahe=True
    )

    # 3. 初始化网络与工具
    model = OurBreastCancerNet(pretrained=True).to(DEVICE)
    criterion = DBDSLoss(max_epochs=EPOCHS, pos_weight=15.0).to(DEVICE)  
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    os.makedirs("outputs/results/logs", exist_ok=True)
    csv_log_path = "outputs/results/logs/our_model_training_log.csv"

    # 写入 CSV 表头
    with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_dice", "val_iou", "val_hd95_mean"])

    os.makedirs("outputs/results/weights", exist_ok=True)
    best_val_dice = 0.0 

    # 用于存储训练过程中的指标
    train_losses, val_dices, val_ious, val_hd95s = [], [], [], []
    
    # ========== [新增: 追踪动态深监督权重的列表] ==========
    w0_list, w1_list, w2_list, w3_list = [], [], [], []
    # =======================================================

    # 4. 训练与验证大循环
    for epoch in range(EPOCHS):
        
        # ========== [新增: 记录当前 Epoch 的动态权重大小] ==========
        progress = epoch / EPOCHS
        w0 = max(0.0, 0.1 - 0.2 * progress)
        w1 = max(0.0, 0.2 - 0.4 * progress)
        w2 = max(0.0, 0.3 - 0.4 * progress)
        w3 = 1.0 - (w0 + w1 + w2)
        w0_list.append(w0); w1_list.append(w1); w2_list.append(w2); w3_list.append(w3)
        # ============================================================
        
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
        train_losses.append(avg_train_loss)  
        
        # 验证阶段 
        model.eval()
        val_metrics = SegmentationMetrics(num_classes=2)

        epoch_preds, epoch_masks = [], []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Val  ")
            for images, masks, edges in val_bar:
                images = images.to(DEVICE)
                
                target_h, target_w = masks.shape[-2], masks.shape[-1] 
                if masks.dim() == 4:
                    masks = masks.squeeze(1)
                masks_np = (masks.cpu().numpy() > 0).astype(np.uint8)
                
                preds_list = model(images)
                final_pred_logits = F.interpolate(preds_list[-1], size=(target_h, target_w), mode='bilinear', align_corners=False)
                final_pred_probs = torch.sigmoid(final_pred_logits)
                if final_pred_probs.dim() == 4:
                    final_pred_probs = final_pred_probs.squeeze(1)
                final_pred_binary = (final_pred_probs > 0.5).cpu().numpy().astype(np.uint8)
                
                for i in range(images.size(0)): 
                    val_metrics.update_with_boundary(final_pred_binary[i], masks_np[i])
                    epoch_preds.append(final_pred_binary[i])
                    epoch_masks.append(masks_np[i])
                    
        scores = val_metrics.get_scores()
        val_dice, val_iou, val_hd95 = scores['dice'], scores['iou'], scores['hd95_mean']
        
        val_dices.append(val_dice)
        val_ious.append(val_iou)
        val_hd95s.append(val_hd95)
        
        print(f" Epoch [{epoch+1}] 成绩单 | Train Loss: {avg_train_loss:.4f} | Dice: {val_dice:.4f} | HD95: {val_hd95:.2f}")

        # 保存最佳模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "outputs/results/weights/best_our_model.pth")
            print(f" 最佳模型 (最佳 Dice: {best_val_dice:.4f}) 已保存！\n")
        else:
            print("\n")
        scheduler.step()
        
        with open(csv_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{avg_train_loss:.6f}", f"{val_dice:.6f}", f"{val_iou:.6f}", f"{val_hd95:.6f}"])
    
    # ========== [新增: 绘制包含动态权重的进阶训练曲线] ==========
    print("\n 正在生成训练与动态权重演化图...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 训练损失
    axes[0, 0].plot(range(1, EPOCHS + 1), train_losses, 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. HD95 下降图
    axes[0, 1].plot(range(1, EPOCHS + 1), val_hd95s, 'purple', linewidth=2)
    axes[0, 1].set_title('Validation HD95 (pixels)')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 动态权重与 Dice 叠加图 (核心图！)
    ax_weight = axes[1, 0]
    ax_dice = ax_weight.twinx() # 创建双Y轴
    
    ax_weight.plot(range(1, EPOCHS + 1), w0_list, 'r--', alpha=0.6, label='w0 (Draft 1)')
    ax_weight.plot(range(1, EPOCHS + 1), w1_list, 'g--', alpha=0.6, label='w1 (Draft 2)')
    ax_weight.plot(range(1, EPOCHS + 1), w2_list, 'y--', alpha=0.6, label='w2 (Draft 3)')
    ax_weight.plot(range(1, EPOCHS + 1), w3_list, 'b-', linewidth=2, label='w3 (Final Output)')
    ax_weight.set_xlabel('Epochs')
    ax_weight.set_ylabel('Weight Value', color='b')
    ax_weight.legend(loc='upper left')
    
    ax_dice.plot(range(1, EPOCHS + 1), val_dices, 'k-', linewidth=2.5, label='Validation Dice')
    ax_dice.set_ylabel('Dice Coefficient', color='k')
    ax_dice.legend(loc='lower right')
    axes[1, 0].set_title('Dynamic Weights vs. Model Performance')

    # 4. IoU 曲线
    axes[1, 1].plot(range(1, EPOCHS + 1), val_ious, 'orange', linewidth=2)
    axes[1, 1].set_title('Validation IoU')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/results/plots/training_and_dynamic_weights_curves.png', dpi=300)
    plt.close()
    # ============================================================

    # 5. 在测试集上评估最佳模型
    test_best_model(model, test_loader, DEVICE)
    
    return model, best_val_dice

def test_best_model(model, test_loader, DEVICE):
    print("\n" + "="*60)
    print(" 开始在 Test 集上进行全面评估 (含复杂度、ROC与统计学数据) ")
    print("="*60)
    
    model.load_state_dict(torch.load("outputs/results/weights/best_our_model.pth", map_location=DEVICE))
    model.eval()
    
    # ========== [新增: 计算 Params, FLOPs] ==========
    try:
        # 假设输入图像大小为 256x256，你可以根据实际情况修改
        dummy_input = torch.randn(1, 3, 256, 256).to(DEVICE)
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        flops_g = (macs * 2) / 1e9  # 1 MAC = 2 FLOPs
        params_m = params / 1e6
        print(f" [模型复杂度] 参数量 (Params): {params_m:.2f} M | 计算量 (FLOPs): {flops_g:.2f} G")
    except Exception as e:
        print(" FLOPs 计算跳过:", e)
    # ==================================================

    metrics = SegmentationMetrics(num_classes=2)
    
    # ========== [新增: 收集像素级概率用于画 ROC/PR] ==========
    all_y_true = []
    all_y_scores = []
    # ==========================================================

    # ========== [新增: 收集逐样本数据用于统计学检验] ==========
    per_sample_csv_path = "outputs/results/metrics/test_per_sample_metrics.csv"
    with open(per_sample_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "dice", "iou", "hd95"])
    # ==========================================================

    total_inference_time = 0.0
    total_samples = 0
    sample_count = 0
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing")
        for images, masks, edges in test_bar:
            images = images.to(DEVICE)
            
            target_h, target_w = masks.shape[-2], masks.shape[-1]
            if masks.dim() == 4:
                masks = masks.squeeze(1)
            masks_np = (masks.cpu().numpy() > 0).astype(np.uint8)
            
            # ========== [新增: 测算推理时间] ==========
            start_time = time.time()
            preds_list = model(images)
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            total_samples += images.size(0)
            # ==========================================

            final_pred_logits = F.interpolate(preds_list[-1], size=(target_h, target_w), mode='bilinear', align_corners=False)
            final_pred_probs = torch.sigmoid(final_pred_logits).squeeze(1).cpu().numpy() # 提取连续概率 (0~1)
            final_pred_binary = (final_pred_probs > 0.65).astype(np.uint8) # 论文阈值
            
            # 更新整体指标 & 保存连续概率用于画曲线
            for i in range(images.size(0)): 
                metrics.update_with_boundary(final_pred_binary[i], masks_np[i])
                
                # 展平像素并加入列表 (如果是大图，可考虑降采样保存防爆内存，通常256没问题)
                all_y_true.extend(masks_np[i].flatten())
                all_y_scores.extend(final_pred_probs[i].flatten())

                # ========== [新增: 计算单样本指标用于配对T检验] ==========
                single_metric = SegmentationMetrics(num_classes=2)
                single_metric.update_with_boundary(final_pred_binary[i], masks_np[i])
                single_scores = single_metric.get_scores()
                with open(per_sample_csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([sample_count, single_scores['dice'], single_scores['iou'], single_scores['hd95_mean']])
                sample_count += 1
                # ==========================================================
                
    # 打印最终成绩单
    scores = metrics.get_scores()
    fps = total_samples / total_inference_time
    print("\n" + "="*50)
    print(" 测试集 (Test Set) 最终成绩单 ")
    print("="*50)
    print(f"Dice 系数:     {scores['dice']:.4f}")
    print(f"IoU (Jaccard): {scores['iou']:.4f}")
    print(f"准确率:        {scores['accuracy']:.4f}")
    print(f"灵敏度/召回:   {scores['sensitivity']:.4f}")
    print(f"特异度:        {scores['specificity']:.4f}")
    print(f"HD95:          {scores['hd95_mean']:.2f} 像素")
    print(f"推理速度 (FPS): {fps:.2f} 帧/秒")
    print("="*50)
    print(f" 每张图片的独立成绩已保存至: {per_sample_csv_path} (用于统计学显著性检验)")
    
    # ========== [新增: 绘制并保存 ROC 与 PR 曲线] ==========
    print(" 正在计算并绘制 ROC 和 PR 曲线 (可能需要几十秒)...")
    all_y_true = np.array(all_y_true)
    all_y_scores = np.array(all_y_scores)
    
    # 绘制图像框架
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'Ours ROC curve (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[0].set_ylabel('True Positive Rate (Sensitivity)')
    axes[0].set_title('Receiver Operating Characteristic (ROC)')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # 2. PR Curve
    precision, recall, _ = precision_recall_curve(all_y_true, all_y_scores)
    pr_auc = average_precision_score(all_y_true, all_y_scores)
    axes[1].plot(recall, precision, color='green', lw=2, label=f'Ours PR curve (AP = {pr_auc:.4f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall (Sensitivity)')
    axes[1].set_ylabel('Precision (PPV)')
    axes[1].set_title('Precision-Recall (PR) Curve')
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/results/plots/test_roc_pr_curves.png', dpi=300)
    plt.close()
    print(" 曲线图已保存至: outputs/results/plots/test_roc_pr_curves.png")
    # ==========================================================

    save_test_results(scores, fps)
    return scores

def save_test_results(scores, fps):
    os.makedirs("results", exist_ok=True)
    with open("outputs/results/test_results.txt", "w") as f:
        f.write("="*50 + "\n")
        f.write(" 测试集 (Test Set) 最终成绩单 \n")
        f.write("="*50 + "\n")
        f.write(f"Dice 系数:     {scores['dice']:.4f}\n")
        f.write(f"IoU (Jaccard): {scores['iou']:.4f}\n")
        f.write(f"准确率:        {scores['accuracy']:.4f}\n")
        f.write(f"灵敏度:        {scores['sensitivity']:.4f}\n")
        f.write(f"特异度:        {scores['specificity']:.4f}\n")
        f.write(f"HD95:          {scores['hd95_mean']:.2f} 像素\n")
        f.write(f"FPS:           {fps:.2f} \n")
        f.write("="*50 + "\n")
    print(f" 测试文本日志已保存至: outputs/results/test_results.txt")
    
if __name__ == '__main__':
    train_and_validate()