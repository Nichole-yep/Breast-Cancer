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

from preprocess.dataset import get_loaders
from evaluate.eval import SegmentationMetrics
from models.baseline_models import get_baseline_model


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


def combined_bce_dice_loss(logits, targets):
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


def train_one_model(args):
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

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
    train_loader, val_loader, _test_loader = get_loaders(
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

    best_dice = -1.0
    best_path = os.path.join(args.save_dir, f"best_{args.model}.pth")
    last_path = os.path.join(args.save_dir, f"last_{args.model}.pth")
    log_path = os.path.join(args.log_dir, f"{args.model}_training_log.csv")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_dice", "val_iou", "val_accuracy", "val_sensitivity", "val_specificity", "val_hd95_mean"])

    print(f"\nStart training baseline model: {args.model}")
    print(f"Best model will be saved to: {best_path}\n")

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

            loss = combined_bce_dice_loss(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / max(len(train_loader), 1)

        # Validation phase.
        # 用验证集 Dice 选择 best model，不能用 test set 选模型。
        val_scores = evaluate_on_loader(model, val_loader, device)
        val_dice = val_scores["dice"]

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
