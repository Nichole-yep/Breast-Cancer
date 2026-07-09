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

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR  # 新增
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from PIL import Image
from src.data.dataset import resolve_data_path

# =========================== 请修改以下配置 ===========================
# 1. 导入您的核心模型
from src.models.ours import OurBreastCancerNet as CoreModel

# 2. 数据集路径
TRAIN_CSV = 'src/data/train.csv'
VAL_CSV = 'src/data/val.csv'
TEST_CSV = 'src/data/test.csv'  # 测试集路径

# 3. 导入损失函数和评估指标
from scripts.loss import tversky_loss
from utils.eval import SegmentationMetrics
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
        # 模型参数：单通道输出（二值分割）
        self.num_classes = 1
        self.img_size = 256
        self.epochs = 100          # 与 train.py 一致
        self.batch_size = 8
        self.lr = 1e-4             # 与 train.py 一致
        self.weight_decay = 1e-4   # 与 train.py 一致
        self.seed = 42
        self.num_workers = 4
        self.pin_memory = True
        self.use_amp = True
        self.grad_accum_steps = 1

        # 消融开关
        self.use_ppm = False
        self.use_cbam = False
        self.use_deep_supervision = False
        self.pos_weight = 1.0  # 仅保留占位，tversky 不使用

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
        pretrained=True,  # 正式实验必须为 True
        num_classes=config.num_classes  # 此时为 1
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
                print(f" 未匹配到图像列名，默认使用第 {img_idx} 列 ('{header_clean[img_idx]}')")
            if mask_idx is None:
                mask_idx = 1 if len(header_clean) > 1 else 0
                print(f" 未匹配到掩码列名，默认使用第 {mask_idx} 列 ('{header_clean[mask_idx]}')")

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
            image = np.load(resolve_data_path(img_path)).astype(np.float32)
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=0)
            else:
                image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float()
        else:
            image = Image.open(resolve_data_path(img_path)).convert('RGB')
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # 读取掩码
        if mask_path.endswith('.npy'):
            mask = np.load(resolve_data_path(mask_path)).astype(np.float32)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            mask = torch.from_numpy(mask).float()
        else:
            mask = Image.open(resolve_data_path(mask_path)).convert('L')
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
def get_dataloaders(config, train_csv, val_csv, test_csv=None):
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

    test_loader = None
    if test_csv is not None and os.path.exists(test_csv):
        test_dataset = CSVMedicalDataset(test_csv, img_size=config.img_size)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # 测试时 batch_size=1 更精确计算 HD95
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        print(f" 测试集: {len(test_dataset)} 张")
    return train_loader, val_loader, test_loader


# -------------------- 5. 损失函数（使用 Tversky Loss） --------------------
def get_criterion(config):
    """
    返回损失函数：单输出使用 tversky_loss，深监督模式对各输出求和。
    注意：tversky_loss 期望 pred 和 target 形状一致，内部会做 sigmoid。
    """

    # 定义单输出损失
    def single_loss(pred, target):
        # 上采样预测到 target 尺寸
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=True)
        return tversky_loss(pred, target, alpha=0.2, beta=0.8)

    if config.use_deep_supervision:
        def deep_supervision_loss(preds, target):
            total = 0.0
            for pred in preds:
                total += single_loss(pred, target)
            return total  # 不加权求和

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


# ===================== 评估函数（使用 SegmentationMetrics）=====================
def evaluate(model, loader, device):
    """
    使用 SegmentationMetrics 计算 Dice、IoU 和 HD95
    """
    model.eval()
    # 实例化评估指标对象（二分类：背景+前景，故 num_classes=2）
    metrics = SegmentationMetrics(num_classes=2, pixel_spacing=(1.0, 1.0))

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Evaluating'):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            # 如果是深监督，取最后一个输出
            if isinstance(preds, list):
                preds = preds[-1]

            #上采样预测到与掩码相同的空间尺寸
            if preds.shape[-2:] != masks.shape[-2:]:
                preds = F.interpolate(preds, size=masks.shape[-2:], mode='bilinear', align_corners=True)
           
            # 转为二值预测 (B, 1, H, W) -> (B, H, W)
            preds = (torch.sigmoid(preds) > 0.5).squeeze(1).long()
            # 真值 (B, 1, H, W) -> (B, H, W)
            masks = masks.squeeze(1).long()

            # 逐样本更新指标（确保输入为 numpy 数组，类型为 uint8）
            for i in range(preds.shape[0]):
                pred_np = preds[i].cpu().numpy().astype(np.uint8)
                mask_np = masks[i].cpu().numpy().astype(np.uint8)
                metrics.update_with_boundary(pred_np, mask_np)

    scores = metrics.get_scores()
    dice = scores['dice']
    iou = scores['iou']
    hd95 = scores['hd95_mean']
    return dice, iou, hd95
# =====================================================================


# -------------------- 7. 运行单个实验（返回配置） --------------------
def run_experiment(exp_id, train_csv, val_csv, test_csv=None):
    config = ExperimentConfig(exp_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 50}\n运行 {config.name} (ID: {exp_id})\n{'=' * 50}")
    print(f"配置: {config.__dict__}")

    train_loader, val_loader, _ = get_dataloaders(config, train_csv, val_csv, test_csv)

    model = build_model(config).to(device)

    # ---- 修改：使用 AdamW 优化器（与 train.py 一致） ----
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # ---- 新增：余弦退火学习率调度器 ----
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    criterion = get_criterion(config)
    scaler = GradScaler(enabled=config.use_amp)

    best_dice = 0.0
    best_iou = 0.0
    best_hd95 = float('inf')
    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, config)
        dice, iou, hd95 = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Val Dice: {dice:.4f} | IoU: {iou:.4f} | HD95: {hd95:.4f}")
        if dice > best_dice:
            best_dice = dice
            best_iou = iou
            best_hd95 = hd95
            torch.save(model.state_dict(), f"exp_{exp_id}_best.pth")

        # ---- 每 epoch 后更新学习率 ----
        scheduler.step()

    print(f"\n✅ {exp_id} 最佳: Dice={best_dice:.4f}, IoU={best_iou:.4f}, HD95={best_hd95:.4f}")
    return config


# -------------------- 8. 测试函数 --------------------
def test_model(exp_id, config, test_loader, device):
    """
    加载 exp_id 的最佳模型权重并在测试集上评估
    """
    model = build_model(config).to(device)
    weight_path = f"exp_{exp_id}_best.pth"
    if not os.path.exists(weight_path):
        print(f"权重文件 {weight_path} 不存在，跳过测试")
        return None

    model.load_state_dict(torch.load(weight_path, map_location=device))
    dice, iou, hd95 = evaluate(model, test_loader, device)
    print(f"  {config.name} 测试结果: Dice={dice:.4f}, IoU={iou:.4f}, HD95={hd95:.4f}")
    return {'Dice': dice, 'IoU': iou, 'HD95': hd95}


# -------------------- 9. 主入口 --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="消融实验 A, B, C (包装核心模型)")
    parser.add_argument('--mode', type=str, default='full', choices=['test', 'full'],
                        help='test: 只跑A和C(2个epoch快速验证); full: 跑全部三个实验')
    parser.add_argument('--train_csv', type=str, default=TRAIN_CSV,
                        help='训练集CSV文件路径')
    parser.add_argument('--val_csv', type=str, default=VAL_CSV,
                        help='验证集CSV文件路径')
    parser.add_argument('--test_csv', type=str, default=TEST_CSV,
                        help='测试集CSV文件路径')
    args = parser.parse_args()

    test_exists = os.path.exists(args.test_csv)

    if args.mode == 'test':
        experiments = ['A', 'C']
        print("\n 冒烟测试模式：只跑 A 和 C (各2个epoch快速验证)")
        # 冒烟测试时轮数太少，调度器作用不大，但仍保持一致
        ExperimentConfig.epochs = 2
    else:
        experiments = ['B']
        print("\n 完整消融模式：运行  B 实验 (100 epochs)")
        ExperimentConfig.epochs = 100

    # 存储每个实验的配置
    exp_configs = {}

    # 训练实验
    for exp in experiments:
        config = run_experiment(exp, args.train_csv, args.val_csv, args.test_csv)
        exp_configs[exp] = config

    # ========== 测试集评估 ==========
    if test_exists:
        print("\n" + "=" * 60)
        print(" 开始测试集评估（加载最佳模型权重）")
        print("=" * 60)

        # 构建测试集加载器
        first_exp = experiments[0]
        config_first = exp_configs[first_exp]
        _, _, test_loader = get_dataloaders(config_first, args.train_csv, args.val_csv, args.test_csv)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        test_results = {}
        for exp_id in experiments:
            config = exp_configs[exp_id]
            result = test_model(exp_id, config, test_loader, device)
            if result is not None:
                test_results[exp_id] = result

        if test_results:
            print("\n" + "=" * 60)
            print(" 测试集结果汇总")
            print("=" * 60)
            print(f"{'实验':<10} {'Dice ↑':<12} {'IoU ↑':<12} {'HD95 ↓':<12}")
            for exp, metrics in test_results.items():
                print(f"{exp:<10} {metrics['Dice']:.4f}     {metrics['IoU']:.4f}     {metrics['HD95']:.4f}")
        else:
            print("\n 无有效测试结果（可能是权重文件缺失）")
    else:
        print("\n 测试集文件不存在，跳过测试集评估")