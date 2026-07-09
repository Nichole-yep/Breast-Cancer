
# AUTO PATH FIX FOR FINAL GITHUB STRUCTURE
from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
for _p in [_PROJECT_ROOT, _PROJECT_ROOT / "src"]:
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
# END AUTO PATH FIX
# test_exp_A_complete.py
import os
import argparse
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.dataset import resolve_data_path, imread_unicode

# ========================== 从你的 evaluate_segmentation.py 复制必要的组件 ==========================
class PreprocessedBUSIDataset:
    """加载预处理后的 BUSI 测试集，支持 resize 和归一化"""
    def __init__(self, csv_file, input_size=None, normalize=None):
        import pandas as pd
        self.df = pd.read_csv(csv_file)
        self.input_size = input_size
        self.normalize = normalize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['img_path']
        mask_path = self.df.iloc[idx]['mask_path']

        # 读取图像
        img = imread_unicode(resolve_data_path(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 读取掩码（二值化）
        mask = imread_unicode(resolve_data_path(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法读取掩码: {mask_path}")
        mask = (mask > 0).astype(np.uint8)

        orig_h, orig_w = img.shape[:2]

        # 应用 resize
        if self.input_size is not None:
            img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
            mask = cv2.resize(mask, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # 转换为 tensor 并归一化
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.normalize is not None:
            mean, std = self.normalize
            img = (img - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
        
        mask = torch.from_numpy(mask).long()
        return img, mask, orig_h, orig_w


class SegmentationMetrics:
    def __init__(self, num_classes=2, pixel_spacing=(1.0, 1.0)):
        self.num_classes = num_classes
        self.pixel_spacing = pixel_spacing
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.hd95_list = []
        self.valid_hd95_count = 0

    def update_with_boundary(self, pred_mask, gt_mask):
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(f"预测掩码 {pred_mask.shape} 与真实掩码 {gt_mask.shape} 尺寸不匹配")
        
        # 更新混淆矩阵
        hist = self._fast_hist(gt_mask.flatten(), pred_mask.flatten())
        self.confusion_matrix += hist

        # 提取边界并计算 HD95
        pred_boundary = self._extract_boundary(pred_mask)
        gt_boundary = self._extract_boundary(gt_mask)

        if len(pred_boundary) > 0 and len(gt_boundary) > 0:
            hd95 = self._compute_hd95(pred_boundary, gt_boundary, self.pixel_spacing)
            self.hd95_list.append(hd95)
            self.valid_hd95_count += 1

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask] + label_pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def _extract_boundary(self, mask):
        from scipy import ndimage
        if mask.ndim == 3:
            mask = mask.squeeze()
        struct = ndimage.generate_binary_structure(2, 1)
        eroded = ndimage.binary_erosion(mask, struct)
        boundary = mask ^ eroded
        return np.argwhere(boundary)

    def _compute_hd95(self, pred_points, gt_points, spacing):
        pred_scaled = pred_points * spacing
        gt_scaled = gt_points * spacing
        
        if len(gt_scaled) == 0 or len(pred_scaled) == 0:
            return float('inf')
        
        d_pred_gt = [np.min(np.linalg.norm(gt_scaled - p, axis=1)) for p in pred_scaled]
        d_gt_pred = [np.min(np.linalg.norm(pred_scaled - g, axis=1)) for g in gt_scaled]
        
        hd95 = max(np.percentile(d_pred_gt, 95), np.percentile(d_gt_pred, 95))
        return hd95

    def get_scores(self):
        hist = self.confusion_matrix
        tp = np.diag(hist)
        fp = hist.sum(axis=0) - tp
        fn = hist.sum(axis=1) - tp
        tn = hist.sum() - (tp + fp + fn)
        eps = 1e-8

        # 前景指标 (class 1)
        accuracy = (tp[1] + tn[1]) / (tp[1] + tn[1] + fp[1] + fn[1] + eps)
        precision = tp[1] / (tp[1] + fp[1] + eps)
        dice = 2 * tp[1] / (2 * tp[1] + fp[1] + fn[1] + eps)
        iou = tp[1] / (tp[1] + fp[1] + fn[1] + eps)
        sensitivity = tp[1] / (tp[1] + fn[1] + eps)
        specificity = tn[1] / (tn[1] + fp[1] + eps)
        bg_iou = tn[0] / (tn[0] + fp[0] + fn[0] + eps) if self.num_classes == 2 else 0
        miou = (iou + bg_iou) / 2

        hd95_mean = np.mean(self.hd95_list) if self.hd95_list else 0.0
        hd95_std = np.std(self.hd95_list) if self.hd95_list else 0.0

        return {
            'accuracy': accuracy, 'precision': precision, 'dice': dice,
            'iou': iou, 'miou': miou, 'sensitivity': sensitivity,
            'specificity': specificity, 'hd95_mean': hd95_mean, 'hd95_std': hd95_std,
            'valid_hd95_count': self.valid_hd95_count, 'TP': tp[1], 'FP': fp[1],
            'FN': fn[1], 'TN': tn[1]
        }


# ========================== 从消融实验代码导入模型构建 ==========================
from scripts.Ablation import ExperimentConfig, build_model
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description='评估 BUSI 分割模型 (完整版)')
    parser.add_argument('--weights', type=str, default='outputs/results/weights/exp_B_best.pth', help='模型权重文件')
    parser.add_argument('--csv_file', type=str, default='src/data/test.csv', help='测试集 CSV 文件')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--input_size', type=int, nargs=2, default=[256, 256], help='模型输入尺寸')
    parser.add_argument('--normalize', action='store_true', help='是否使用 ImageNet 归一化')
    parser.add_argument('--pixel_spacing', type=float, nargs=2, default=[1.0, 1.0], help='像素间距')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 归一化参数
    normalize = None
    if args.normalize:
        normalize = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        print("使用 ImageNet 归一化")

    # 1. 构建模型（使用消融实验的配置）
    config = ExperimentConfig('B')
    model = build_model(config).to(device)
    
    # 加载权重
    if not os.path.exists(args.weights):
        print(f"错误: 权重文件 {args.weights} 不存在!")
        return
    
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f"成功加载权重: {args.weights}")

    # 2. 准备数据集
    dataset = PreprocessedBUSIDataset(
        csv_file=args.csv_file,
        input_size=tuple(args.input_size),
        normalize=normalize
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"测试集样本数量: {len(dataset)}")

    # 3. 初始化指标对象
    metrics = SegmentationMetrics(num_classes=2, pixel_spacing=tuple(args.pixel_spacing))

    # 4. 评估循环
    with torch.no_grad():
        for images, masks, orig_h_list, orig_w_list in tqdm(dataloader, desc="评估中"):
            images = images.to(device)
            
            # 模型输出
            outputs = model(images)
            
            # 取最后一个输出（最高分辨率）
            if isinstance(outputs, list):
                final_logits = outputs[-1]
            else:
                final_logits = outputs
            
            # 二值化预测
            preds = (torch.sigmoid(final_logits) > 0.5).squeeze(1).long()
            
            # 上采样到与 masks 相同尺寸
            _, H, W = masks.shape
            preds = F.interpolate(
                preds.unsqueeze(1).float(),
                size=(H, W),
                mode='nearest'
            ).squeeze(1).long()
            
            # 逐样本更新指标
            for i in range(preds.shape[0]):
                pred_np = preds[i].cpu().numpy()
                mask_np = masks[i].cpu().numpy()
                metrics.update_with_boundary(pred_np, mask_np)

    # 5. 输出结果
    scores = metrics.get_scores()
    print("\n" + "=" * 60)
    print("BUSI 测试集评估结果 (实验 A)")
    print("=" * 60)
    print(f"Dice 系数:           {scores['dice']:.4f}")
    print(f"IoU (Jaccard):       {scores['iou']:.4f}")
    print(f"平均 IoU (背景+前景): {scores['miou']:.4f}")
    print(f"准确率 (Accuracy):   {scores['accuracy']:.4f}")
    print(f"精确度 (Precision):  {scores['precision']:.4f}")
    print(f"灵敏度 (Sensitivity):{scores['sensitivity']:.4f}")
    print(f"特异度 (Specificity):{scores['specificity']:.4f}")
    print(f"HD95 (均值 ± 标准差): {scores['hd95_mean']:.2f} ± {scores['hd95_std']:.2f} 像素 (有效样本数: {scores['valid_hd95_count']})")
    print(f"混淆矩阵 (前景类): TP={scores['TP']}, FP={scores['FP']}, FN={scores['FN']}, TN={scores['TN']}")
    print("=" * 60)


if __name__ == '__main__':
    main()