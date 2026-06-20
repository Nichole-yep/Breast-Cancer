"""
消融实验脚本(不修改核心模型，使用包装类控制开关)
实验 A (Baseline): PPM=Identity, CBAM=Identity, 仅取最后一个输出
实验 B (+PPM&CBAM): 使用PPM和CBAM，仅取最后一个输出
实验 C (+深监督): 使用PPM和CBAM，返回所有4个输出 (pos_weight=1.0)

核心模型: OurBreastCancerNet (固定包含PPM、CBAM、深监督解码器)
包装类: AblationWrapper 根据配置替换模块并控制返回
运行：
冒烟测试：python ablation.py --mode test
完整实验：python ablation.py --mode full
"""


import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from PIL import Image

# =========================== 请修改以下配置 ===========================
# 1. 导入您的核心模型
from models.ours import OurBreastCancerNet as CoreModel

# 2. 数据集路径
TRAIN_CSV = 'preprocess/train.csv'
VAL_CSV = 'preprocess/val.csv'
# =====================================================================

# -------------------- 包装类  --------------------
class AblationWrapper(nn.Module):
    """
    包装核心模型，根据消融配置禁用 PPM / CBAM / 深监督
    """
    def __init__(self, core_model, use_ppm, use_cbam, use_deep_supervision):
        super().__init__()
        self.core = core_model

        # 条件替换为恒等映射
        if not use_ppm:
            self.core.ppm = nn.Identity()
        if not use_cbam:
            self.core.cbam_ppm = nn.Identity()

        self.use_deep_supervision = use_deep_supervision

    def forward(self, x):
        outputs = self.core(x)  # 核心模型始终返回列表 (4个预测)
        if self.use_deep_supervision:
            return outputs  # 返回所有
        else:
            return outputs[-1]  # 仅最高分辨率


# -------------------- 1. 配置类 --------------------
class ExperimentConfig:
    def __init__(self, exp_id):
        self.num_classes = 1
        self.img_size = 256
        self.epochs = 30
        self.batch_size = 8
        self.lr = 0.001
        self.seed = 42
        self.num_workers = 4
        self.pin_memory = True
        self.use_amp = True
        self.grad_accum_steps = 1

        # 消融开关
        self.use_ppm = False
        self.use_cbam = False
        self.use_deep_supervision = False
        self.pos_weight = 1.0

        self._set_experiment(exp_id)
        self.exp_name = exp_id
        self._fix_seed()

    def _set_experiment(self, exp_id):
        if exp_id == 'A':
            self.use_ppm = False
            self.use_cbam = False
            self.use_deep_supervision = False
            self.name = "Baseline (No PPM, No CBAM, No DeepSup)"
        elif exp_id == 'B':
            self.use_ppm = True
            self.use_cbam = True
            self.use_deep_supervision = False
            self.name = "+PPM & CBAM"
        elif exp_id == 'C':
            self.use_ppm = True
            self.use_cbam = True
            self.use_deep_supervision = True
            self.pos_weight = 1.0
            self.name = "+Deep Supervision (pos_weight=1.0)"
        else:
            raise ValueError("Invalid exp_id. Choose from A, B, C.")

    def _fix_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -------------------- 2. 模型构建 (包装核心模型) --------------------
def build_model(config):
    core = CoreModel(
        model_name='efficientnet_b0',
        pretrained=True,           # 正式实验必须为 True
        num_classes=config.num_classes
    )
    wrapper = AblationWrapper(
        core,
        use_ppm=config.use_ppm,
        use_cbam=config.use_cbam,
        use_deep_supervision=config.use_deep_supervision
    )
    return wrapper


# -------------------- 3. CSV数据集 (已修正列名匹配) --------------------
class CSVMedicalDataset(Dataset):
    def __init__(self, csv_path, img_size=256):
        self.img_size = img_size
        self.samples = []

        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError(f"CSV 文件 {csv_path} 为空")

            header_clean = [col.strip() for col in header]
            print(f" 检测到列名: {header_clean}")

            # ---- 修正：添加 'img_path' 和 'mask_path' 到匹配列表 ----
            img_idx = mask_idx = None
            for idx, name in enumerate(header_clean):
                if name.lower() in ['image_path', 'img_path', 'img', 'image', 'file_path', 'filename']:
                    img_idx = idx
                if name.lower() in ['mask_path', 'mask', 'label', 'seg', 'gt']:
                    mask_idx = idx
            # ---------------------------------------------------------

            if img_idx is None:
                img_idx = 0
                print(f"⚠ 未匹配到图像列名，默认使用第 {img_idx} 列 ('{header_clean[img_idx]}')")
            if mask_idx is None:
                mask_idx = 1 if len(header_clean) > 1 else 0
                print(f"⚠ 未匹配到掩码列名，默认使用第 {mask_idx} 列 ('{header_clean[mask_idx]}')")

            print(f" 使用图像列索引: {img_idx}, 掩码列索引: {mask_idx}")

            for row in reader:
                if len(row) <= max(img_idx, mask_idx):
                    continue
                img_path = row[img_idx].strip()
                mask_path = row[mask_idx].strip()
                if img_path and mask_path:
                    self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"未从 {csv_path} 加载到任何有效样本。")
        print(f"✅ 成功加载 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # 读取图像
        if img_path.endswith('.npy'):
            image = np.load(img_path).astype(np.float32)
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=0)
            else:
                image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float()
        else:
            image = Image.open(img_path).convert('RGB')
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # 读取掩码
        if mask_path.endswith('.npy'):
            mask = np.load(mask_path).astype(np.float32)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            mask = torch.from_numpy(mask).float()
        else:
            mask = Image.open(mask_path).convert('L')
            mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0

        # Resize 到统一尺寸
        if image.shape[1] != self.img_size or image.shape[2] != self.img_size:
            image = F.interpolate(image.unsqueeze(0), size=(self.img_size, self.img_size),
                                  mode='bilinear').squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0), size=(self.img_size, self.img_size),
                                 mode='nearest').squeeze(0)

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        mask = (mask > 0.5).float()
        return image, mask


# -------------------- 4. 数据加载器 --------------------
def get_dataloaders(config, train_csv, val_csv):
    train_dataset = CSVMedicalDataset(train_csv, img_size=config.img_size)
    val_dataset = CSVMedicalDataset(val_csv, img_size=config.img_size)
    print(f" 训练集: {len(train_dataset)} 张 | 验证集: {len(val_dataset)} 张")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    return train_loader, val_loader


# -------------------- 5. 损失函数 --------------------
def get_criterion(config):
    def dice_loss(pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        return 1 - (2. * intersection + smooth) / (union + smooth)

    def bce_loss(pred, target):
        pos_weight = torch.tensor([config.pos_weight]).to(pred.device)
        return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)

    def single_loss(pred, target):
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=True)
        return bce_loss(pred, target) + dice_loss(pred, target).mean()

    if config.use_deep_supervision:
        def deep_supervision_loss(preds, target):
            total = 0.0
            for pred in preds:
                total += single_loss(pred, target)
            return total  # 不加权，直接求和
        return deep_supervision_loss
    else:
        return single_loss


# -------------------- 6. 训练与评估 --------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler, config):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    for idx, (imgs, masks) in enumerate(tqdm(loader, desc='Training')):
        imgs, masks = imgs.to(device), masks.to(device)

        with torch.autocast(device_type=device_type, enabled=config.use_amp):
            preds = model(imgs)
            loss = criterion(preds, masks)
            if config.grad_accum_steps > 1:
                loss = loss / config.grad_accum_steps

        scaler.scale(loss).backward()
        if (idx + 1) % config.grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * config.grad_accum_steps

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    hd_metric = HausdorffDistanceMetric(percentile=95, include_background=True, reduction="mean")
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Evaluating'):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            if isinstance(preds, list):
                preds = preds[-1]
            preds = torch.sigmoid(preds) > 0.5
            preds = preds.float()
            dice_metric(preds, masks)
            hd_metric(preds, masks)
    dice = dice_metric.aggregate().item()
    hd95 = hd_metric.aggregate().item()
    dice_metric.reset()
    hd_metric.reset()
    return dice, hd95


# -------------------- 7. 运行单个实验 --------------------
def run_experiment(exp_id, train_csv, val_csv):
    config = ExperimentConfig(exp_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 50}\n运行 {config.name} (ID: {exp_id})\n{'=' * 50}")
    print(f"配置: {config.__dict__}")

    train_loader, val_loader = get_dataloaders(config, train_csv, val_csv)

    model = build_model(config).to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)
    criterion = get_criterion(config)
    scaler = GradScaler(enabled=config.use_amp)

    best_dice = 0.0
    best_hd95 = float('inf')
    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, config)
        dice, hd95 = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Val Dice: {dice:.4f} | Val HD95: {hd95:.4f}")
        if dice > best_dice:
            best_dice = dice
            best_hd95 = hd95
            torch.save(model.state_dict(), f"exp_{exp_id}_best.pth")

    print(f"\n✅ {exp_id} 最佳: Dice={best_dice:.4f}, HD95={best_hd95:.4f}")
    return best_dice, best_hd95


# -------------------- 8. 主入口 --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="消融实验 A, B, C (包装核心模型)")
    parser.add_argument('--mode', type=str, default='full', choices=['test', 'full'],
                        help='test: 只跑A和C(2个epoch验证代码); full: 跑全部三个实验')
    parser.add_argument('--train_csv', type=str, default=TRAIN_CSV,
                        help='训练集CSV文件路径 (默认使用顶部定义的TRAIN_CSV)')
    parser.add_argument('--val_csv', type=str, default=VAL_CSV,
                        help='验证集CSV文件路径 (默认使用顶部定义的VAL_CSV)')
    args = parser.parse_args()

    if args.mode == 'test':
        experiments = ['A', 'C']
        print("\n 冒烟测试模式：只跑 A 和 C (各2个epoch快速验证)")
        ExperimentConfig.epochs = 2
    else:
        experiments = ['A', 'B', 'C']
        print("\n 完整消融模式：运行 A, B, C 三个实验 (30 epochs)")
        ExperimentConfig.epochs = 30

    results = {}
    for exp in experiments:
        dice, hd95 = run_experiment(exp, args.train_csv, args.val_csv)
        results[exp] = {'Dice': dice, 'HD95': hd95}

    print("\n" + "=" * 60)
    print(" 消融实验结果汇总")
    print("=" * 60)
    print(f"{'实验':<10} {'Dice ↑':<15} {'HD95 ↓':<15}")
    for exp, metrics in results.items():
        print(f"{exp:<10} {metrics['Dice']:.4f}         {metrics['HD95']:.4f}")
