# ========================= 中文注释说明 =========================
# 本文件负责训练 baseline 对照模型。
# 目的：让 U-Net、Attention U-Net、DeepLabV3+ 使用同一个数据集划分、同一个预处理、同一个 loss 设置。
# 数据入口：直接调用 preprocess/dataset.py 里的 get_loaders。
# 评价入口：训练过程中调用 evaluate/eval.py 里的 SegmentationMetrics，在验证集上计算 Dice/IoU/HD95 等指标。
# 保存规则：每个 epoch 验证一次，如果 Val Dice 变高，就保存 best_xxx.pth。
# 注意：这里的 test_csv 只是传给 get_loaders 以保持接口一致，训练阶段不使用 test set 选择模型。
# ===============================================================

# train_baseline_models.py
# Train baseline segmentation models using the group's own BUSI preprocess/dataset.
#
# Put this file into your project root:
# Breast-Cancer/train_baseline_models.py
#
# Example commands:
# python train_baseline_models.py --model unet --device cpu --epochs 20 --batch_size 2
# python train_baseline_models.py --model attention_unet --device cpu --epochs 20 --batch_size 2
# python train_baseline_models.py --model deeplabv3plus --device cpu --epochs 20 --batch_size 2

import os
import argparse
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time           # [新增: 用于计算推理FPS]
import numpy as np    # [新增: 数组处理]
import matplotlib.pyplot as plt  # [新增: 用于画训练曲线]
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score # [新增: ROC/PR曲线]

from preprocess.dataset import get_loaders
from evaluate.eval import SegmentationMetrics
from models.baseline_models import get_baseline_model
from baseline_loss import BCETverskyLoss
from thop import profile # [新增: 导入 thop 用于算 FLOPs 和参数量]

def extract_logits(model_output):
    """
    Extract the final logits from different model output formats.

    中文说明：
    有的模型直接返回 tensor，有的模型返回 list/tuple/dict。
    为了让训练代码兼容不同模型，这里统一取最终 logits。

    Make the training code compatible with both:
    - baseline models returning a tensor [B,1,H,W]
    - deep-supervision models returning list/tuple, where last item is final logits
    """
    if isinstance(model_output, (list, tuple)):
        return model_output[-1]
    if isinstance(model_output, dict):
        # Common torchvision-style output uses key "out"
        if "out" in model_output:
            return model_output["out"]
        # Otherwise use the last value
        return list(model_output.values())[-1]
    return model_output


def dice_loss_from_logits(logits, targets, eps=1e-7):
    """
    Binary Dice loss.

    中文说明：
    Dice 衡量预测区域和真实 mask 的重叠程度。
    Dice 越大越好，所以 Dice Loss = 1 - Dice。
    logits:  [B,1,H,W]
    targets: [B,1,H,W], values 0/1
    """
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dims)
    union = torch.sum(probs, dims) + torch.sum(targets, dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


#def combined_bce_dice_loss(logits, targets):
    """
    Fair simple baseline loss: 0.5 * BCE + 0.5 * Dice.

    中文说明：
    BCE 负责逐像素二分类，Dice 用来缓解前景/背景不平衡。
    baseline 不加入 Edge Loss，避免把小组最终模型的创新点加入对照模型。
    Edge loss is not used for baselines because U-Net/Attention U-Net/DeepLabV3+ are comparison models.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets.float())
    d_loss = dice_loss_from_logits(logits, targets)
    return 0.5 * bce + 0.5 * d_loss


def evaluate_on_loader(model, dataloader, device):
    """
    Validate/test using the same SegmentationMetrics from evaluate/eval.py.

    中文说明：
    每个 epoch 训练结束后，用验证集计算 Dice、IoU、HD95 等指标，
    并根据验证集 Dice 保存 best model。
    """
    model.eval()
    metrics = SegmentationMetrics(num_classes=2)

    with torch.no_grad():
        for images, masks, _edges in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            masks = masks.to(device).float()  # [B,1,H,W]

            outputs = model(images)
            logits = extract_logits(outputs)

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            preds = (torch.sigmoid(logits) > 0.5).long()  # [B,1,H,W]
            preds_np = preds.squeeze(1).cpu().numpy().astype("uint8")
            masks_np = (masks.squeeze(1).cpu().numpy() > 0).astype("uint8")

            for i in range(preds_np.shape[0]):
                metrics.update_with_boundary(preds_np[i], masks_np[i])

    return metrics.get_scores()

# ========== [新增: 专门用于最后测试并保存所有图表和CSV的函数] ==========
def test_best_model(model, test_loader, device, args):
    print("\n" + "="*60)
    print(f" 开始在 Test 集上评估最佳 Baseline 模型: {args.model} ")
    print("="*60)
    
    best_path = os.path.join(args.save_dir, f"best_{args.model}.pth")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    
    # --- 1. 计算 Params 和 FLOPs ---
    try:
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        flops_g = (macs * 2) / 1e9
        params_m = params / 1e6
        print(f" [模型复杂度] 参数量 (Params): {params_m:.2f} M | 计算量 (FLOPs): {flops_g:.2f} G")
    except Exception as e:
        print(" FLOPs 计算跳过:", e)

    metrics = SegmentationMetrics(num_classes=2)
    all_y_true, all_y_scores = [], []
    
    # --- 2. 准备保存每个样本指标的 CSV ---
    per_sample_csv_path = os.path.join(args.log_dir, f"test_{args.model}_per_sample_metrics.csv")
    with open(per_sample_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "dice", "iou", "hd95"])

    total_inference_time = 0.0
    total_samples = 0
    sample_count = 0

    # --- 3. 遍历测试集 ---
    with torch.no_grad():
        for images, masks, _edges in tqdm(test_loader, desc="Testing Baseline"):
            images = images.to(device)
            masks = masks.to(device).float()
            
            # 推理并记录时间
            start_time = time.time()
            outputs = model(images)
            logits = extract_logits(outputs)
            end_time = time.time()
            
            total_inference_time += (end_time - start_time)
            total_samples += images.size(0)

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            
            preds_np = preds.squeeze(1).cpu().numpy().astype("uint8")
            masks_np = (masks.squeeze(1).cpu().numpy() > 0).astype("uint8")
            probs_np = probs.squeeze(1).cpu().numpy() # 获取连续概率

            for i in range(preds_np.shape[0]):
                # 全局评价指标
                metrics.update_with_boundary(preds_np[i], masks_np[i])
                
                # 收集用于 ROC/PR 的数据
                all_y_true.extend(masks_np[i].flatten())
                all_y_scores.extend(probs_np[i].flatten())

                # 独立计算每个样本的指标并存入 CSV
                single_metric = SegmentationMetrics(num_classes=2)
                single_metric.update_with_boundary(preds_np[i], masks_np[i])
                single_scores = single_metric.get_scores()
                
                with open(per_sample_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([sample_count, single_scores['dice'], single_scores['iou'], single_scores['hd95_mean']])
                sample_count += 1

    # --- 4. 打印整体成绩单 ---
    scores = metrics.get_scores()
    fps = total_samples / total_inference_time
    print("\n" + "="*50)
    print(f" {args.model} 测试集 (Test Set) 最终成绩单 ")
    print("="*50)
    print(f"Dice 系数:     {scores['dice']:.4f}")
    print(f"IoU (Jaccard): {scores['iou']:.4f}")
    print(f"准确率:        {scores['accuracy']:.4f}")
    print(f"灵敏度/召回:   {scores['sensitivity']:.4f}")
    print(f"特异度:        {scores['specificity']:.4f}")
    print(f"HD95:          {scores['hd95_mean']:.2f} 像素")
    print(f"推理速度 (FPS): {fps:.2f} 帧/秒")
    print("="*50)
    print(f" 每张图片的独立成绩已保存至: {per_sample_csv_path}")

    # --- 5. 画 ROC 和 PR 曲线 ---
    print(" 正在绘制 ROC 和 PR 曲线...")
    plots_dir = os.path.join(args.save_dir, "../plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    all_y_true = np.array(all_y_true)
    all_y_scores = np.array(all_y_scores)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC
    fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'{args.model} (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'{args.model} ROC Curve')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # PR
    precision, recall, _ = precision_recall_curve(all_y_true, all_y_scores)
    pr_auc = average_precision_score(all_y_true, all_y_scores)
    axes[1].plot(recall, precision, color='green', lw=2, label=f'{args.model} (AP = {pr_auc:.4f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'{args.model} PR Curve')
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'test_roc_pr_{args.model}.png'), dpi=300)
    plt.close()
# =======================================================================

def train_one_model(args):
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # [新增: 确保图表文件夹存在]
    plots_dir = os.path.join(args.save_dir, "../plots")
    os.makedirs(plots_dir, exist_ok=True)


    # Device
    # 如果用户选择 cuda 但电脑没有 GPU，就自动退回 CPU，避免程序报错。
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Data
    # 这里直接调用小组已有 preprocess/dataset.py，保证 baseline 和最终模型使用同样的数据预处理。
    print("Loading BUSI DataLoaders from preprocess/dataset.py ...")
    train_loader, val_loader, test_loader = get_loaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        use_lee=(not args.no_lee),
        use_clahe=(not args.no_clahe)
    )

    # Model
    # 根据命令行 --model 选择 unet / attention_unet / deeplabv3plus。
    model = get_baseline_model(
        model_name=args.model,
        in_channels=3,
        num_classes=1,
        base_channels=args.base_channels
    ).to(device)

    # Optimizer and learning-rate scheduler.
    # AdamW 用于更新参数；CosineAnnealingLR 让学习率随 epoch 平滑下降。
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    criterion = BCETverskyLoss(
        pos_weight=15.0,
        alpha=0.2,
        beta=0.8
    )

    best_dice = -1.0
    best_path = os.path.join(args.save_dir, f"best_{args.model}.pth")
    last_path = os.path.join(args.save_dir, f"last_{args.model}.pth")
    log_path = os.path.join(args.log_dir, f"{args.model}_training_log.csv")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_dice", "val_iou", "val_accuracy", "val_sensitivity", "val_specificity", "val_hd95_mean"])

    print(f"\nStart training baseline model: {args.model}")
    print(f"Best model will be saved to: {best_path}\n")

    # ========== [新增: 记录画图数据] ==========
    epoch_train_losses, epoch_val_dices, epoch_val_ious, epoch_val_hd95s = [], [], [], []
    # ==========================================

    for epoch in range(1, args.epochs + 1):
        # Training phase.
        # model.train() 会启用训练模式，例如 Dropout/BatchNorm 的训练行为。
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}] Train")
        for images, masks, _edges in train_bar:
            images = images.to(device)
            masks = masks.to(device).float()  # [B,1,H,W]

            # Standard PyTorch training steps:
            # clear gradients -> forward -> loss -> backward -> update weights.
            optimizer.zero_grad()
            outputs = model(images)
            logits = extract_logits(outputs)

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)


            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / max(len(train_loader), 1)

        # Validation phase.
        # 用验证集 Dice 选择 best model，不能用 test set 选模型。
        val_scores = evaluate_on_loader(model, val_loader, device)
        val_dice = val_scores["dice"]
        val_iou = val_scores["iou"]
        val_hd95 = val_scores["hd95_mean"]

        # ========== [新增: 保存画图数据] ==========
        epoch_train_losses.append(avg_train_loss)
        epoch_val_dices.append(val_dice)
        epoch_val_ious.append(val_iou)
        epoch_val_hd95s.append(val_hd95)
        # ==========================================

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Dice: {val_scores['dice']:.4f} | "
            f"Val IoU: {val_scores['iou']:.4f} | "
            f"Val HD95: {val_scores['hd95_mean']:.2f}"
        )

        # Save log.
        # 每个 epoch 保存一行日志，后续可以画训练曲线或写实验报告。
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{avg_train_loss:.6f}",
                f"{val_scores['dice']:.6f}",
                f"{val_scores['iou']:.6f}",
                f"{val_scores['accuracy']:.6f}",
                f"{val_scores['sensitivity']:.6f}",
                f"{val_scores['specificity']:.6f}",
                f"{val_scores['hd95_mean']:.6f}",
            ])

        # Save best.
        # 只有当前 Val Dice 更高，才覆盖 best_xxx.pth。
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_path)
            print(f"Saved new best {args.model}: Dice={best_dice:.4f}")

        scheduler.step()

    torch.save(model.state_dict(), last_path)
    print("\nTraining finished.")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Best weight: {best_path}")
    print(f"Last weight: {last_path}")
    print(f"Training log: {log_path}")


# ========== [新增: 绘制 Baseline 的训练曲线并保存] ==========
    print("\n 正在生成训练曲线图...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(range(1, args.epochs + 1), epoch_train_losses, 'b-', linewidth=2)
    axes[0, 0].set_title(f'{args.model} Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(range(1, args.epochs + 1), epoch_val_dices, 'g-', linewidth=2)
    axes[0, 1].set_title(f'{args.model} Validation Dice')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(range(1, args.epochs + 1), epoch_val_ious, 'orange', linewidth=2)
    axes[1, 0].set_title(f'{args.model} Validation IoU')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(range(1, args.epochs + 1), epoch_val_hd95s, 'purple', linewidth=2)
    axes[1, 1].set_title(f'{args.model} Validation HD95')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'training_curves_{args.model}.png'), dpi=300)
    plt.close()
    # ============================================================

    print("\nTraining finished.")
    
    # ========== [新增: 训练结束后，调用测试函数计算所有的指标和保存CSV] ==========
    test_best_model(model, test_loader, device, args)
    # =======================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline models with the group's BUSI preprocess and evaluation metrics.")

    parser.add_argument("--model", type=str, required=True,
                        choices=["unet", "attention_unet", "deeplabv3plus"],
                        help="Baseline model name.")
    parser.add_argument("--train_csv", type=str, default="preprocess/train.csv")
    parser.add_argument("--val_csv", type=str, default="preprocess/val.csv")
    parser.add_argument("--test_csv", type=str, default="preprocess/test.csv")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="For CPU, use 1 or 2. For GPU, try 4 or 8.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=32,
                        help="Model width. Use 16 if CPU memory is limited; use 32 for normal baseline.")

    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--save_dir", type=str, default="results/weights")
    parser.add_argument("--log_dir", type=str, default="results/logs")

    parser.add_argument("--no_lee", action="store_true", help="Disable Lee filtering in BUSIDataset if supported.")
    parser.add_argument("--no_clahe", action="store_true", help="Disable CLAHE in BUSIDataset if supported.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_one_model(args)