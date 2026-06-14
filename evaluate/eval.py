# evaluate_segmentation.py
import os
import argparse
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from scipy import ndimage
import torch.nn as nn

# ========================== 1. 数据集定义 ==========================
class PreprocessedBUSIDataset(Dataset):
    """加载预处理后的 BUSI 测试集，支持 resize 和归一化"""

    def __init__(self, csv_file, input_size=None, normalize=None, transform=None):

        self.df = pd.read_csv(csv_file)
        self.input_size = input_size  # (h, w)
        self.normalize = normalize

        # 构建基础 transforms
        t_list = []
        t_list.append(transforms.ToPILImage())
        if input_size is not None:
            t_list.append(transforms.Resize(input_size))
        t_list.append(transforms.ToTensor())  # 将 [0,255] 转为 [0,1]
        if normalize is not None:
            t_list.append(transforms.Normalize(mean=normalize[0], std=normalize[1]))
        self.transform = transforms.Compose(t_list)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['images']
        mask_path = self.df.iloc[idx]['masks']

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 读取掩码（二值化）
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法读取掩码: {mask_path}")
        mask = (mask > 0).astype(np.uint8)

        orig_h, orig_w = img.shape[:2]

        # 应用 transform (resize, to tensor, normalize)
        img_tensor = self.transform(img)  # (C, H', W')
        # 对掩码也需要做相同的 resize，但保持为 numpy 数组，后续转为 tensor
        if self.input_size is not None:
            mask = cv2.resize(mask, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask).long()

        return img_tensor, mask_tensor, orig_h, orig_w


# ========================== 2. 评估指标计算类 ==========================
class SegmentationMetrics:
    def __init__(self, num_classes=2, pixel_spacing=(1.0, 1.0)):
        self.num_classes = num_classes
        self.pixel_spacing = pixel_spacing  # 像素间距 (mm/pixel)，用于 HD95
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.hd95_list = []  # 存储每张图像的 HD95 值
        self.valid_hd95_count = 0  # 有效 HD95 的图像数（边界非空）

    def update(self, y_true, y_pred):
        """更新混淆矩阵（y_true, y_pred 为展平的一维数组或同形状数组）"""
        hist = self._fast_hist(y_true.flatten(), y_pred.flatten())
        self.confusion_matrix += hist

    def update_with_boundary(self, pred_mask, gt_mask):
        """同时更新混淆矩阵和 HD95"""
        # 确保尺寸一致
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(f"预测掩码 {pred_mask.shape} 与真实掩码 {gt_mask.shape} 尺寸不匹配")
        self.update(gt_mask, pred_mask)

        # 提取边界并计算 HD95
        pred_boundary = self._extract_boundary(pred_mask)
        gt_boundary = self._extract_boundary(gt_mask)

        if len(pred_boundary) > 0 and len(gt_boundary) > 0:
            hd95 = self._compute_hd95(pred_boundary, gt_boundary, self.pixel_spacing)
            self.hd95_list.append(hd95)
            self.valid_hd95_count += 1
        else:
            pass

    def _fast_hist(self, label_true, label_pred):
        """计算两个展平数组的混淆矩阵"""
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask] + label_pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def _extract_boundary(self, mask):
        """提取二值掩码的边界像素（坐标列表）"""
        if mask.ndim == 3:
            mask = mask.squeeze()
        struct = ndimage.generate_binary_structure(2, 1)
        eroded = ndimage.binary_erosion(mask, struct)
        boundary = mask ^ eroded
        return np.argwhere(boundary)  # (N, 2)

    def _compute_hd95(self, pred_points, gt_points, spacing):
        """计算 95% Hausdorff 距离（单位：mm 或像素，取决于 spacing）"""
        # 应用像素间距缩放
        pred_scaled = pred_points * spacing
        gt_scaled = gt_points * spacing

        # 计算 pred -> gt 的所有最小距离
        if len(gt_scaled) == 0 or len(pred_scaled) == 0:
            return float('inf')
        # 使用广播计算欧氏距离，避免循环
        # 对于少量点可以使用循环，若点集较大可优化
        # 这里点集通常较小（边界像素），循环可接受
        d_pred_gt = [np.min(np.linalg.norm(gt_scaled - p, axis=1)) for p in pred_scaled]
        d_gt_pred = [np.min(np.linalg.norm(pred_scaled - g, axis=1)) for g in gt_scaled]

        hd95 = max(np.percentile(d_pred_gt, 95), np.percentile(d_gt_pred, 95))
        return hd95

    def get_scores(self):
        """计算并返回所有评估指标（针对前景类）"""
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
        sensitivity = tp[1] / (tp[1] + fn[1] + eps)  # 召回率
        specificity = tn[1] / (tn[1] + fp[1] + eps)

        # 背景 IoU (class 0)
        bg_iou = tn[0] / (tn[0] + fp[0] + fn[0] + eps) if self.num_classes == 2 else 0
        miou = (iou + bg_iou) / 2

        # HD95 统计
        if self.hd95_list:
            hd95_mean = np.mean(self.hd95_list)
            hd95_std = np.std(self.hd95_list)
        else:
            hd95_mean = 0.0
            hd95_std = 0.0

        scores = {
            'accuracy': accuracy,
            'precision': precision,
            'dice': dice,
            'iou': iou,
            'miou': miou,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'hd95_mean': hd95_mean,
            'hd95_std': hd95_std,
            'valid_hd95_count': self.valid_hd95_count,
            'TP': tp[1],
            'FP': fp[1],
            'FN': fn[1],
            'TN': tn[1],
        }
        return scores


# ========================== 3. 模型定义（用户需根据实际模型修改） ==========================
def get_model(weights_path, device, num_classes=1): # 注意：这里是 1
    from models.ours import OurBreastCancerNet 
    # 注意 num_classes=1 并且不需要 pretrained 权重（因为我们加载本地权重）
    model = OurBreastCancerNet(pretrained=False, num_classes=1).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ========================== 4. 将预测上采样回原始尺寸 ==========================
def resize_pred_to_original(pred_tensor, orig_h, orig_w):
    #将模型输出的预测图 (1, H', W') 或 (H', W') 上采样回原始图像尺寸 (orig_h, orig_w)
    if isinstance(pred_tensor, torch.Tensor):
        if pred_tensor.dim() == 2:
            pred_tensor = pred_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif pred_tensor.dim() == 3:
            pred_tensor = pred_tensor.unsqueeze(0)  # (1,H,W) -> (1,1,H,W)
        # 上采样
        pred_resized = F.interpolate(pred_tensor.float(), size=(orig_h, orig_w), mode='nearest')
        pred_resized = pred_resized.squeeze().cpu().numpy().astype(np.uint8)
    else:
        # 若已经是 numpy
        pred_resized = cv2.resize(pred_tensor, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return pred_resized


# ========================== 5. 主评估流程 ==========================
def main():
    parser = argparse.ArgumentParser(description='评估 BUSI 分割模型')
    parser.add_argument('--weights', type=str, required=True, help='模型权重文件 (.pt/.pth)')
    parser.add_argument('--csv_file', type=str, required=True, help='测试集 CSV 文件 (含 images 和 masks 列)')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--device', type=str, default='cuda', help='运行设备 (cuda/cpu)')
    parser.add_argument('--input_size', type=int, nargs=2, default=[256, 256],
                        help='模型输入尺寸 (height width)，例如 --input_size 256 256')
    parser.add_argument('--normalize', action='store_true', help='是否使用 ImageNet 归一化')
    parser.add_argument('--pixel_spacing', type=float, nargs=2, default=[1.0, 1.0],
                        help='像素间距 (mm/pixel) 用于 HD95，例如 --pixel_spacing 0.1 0.1')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 归一化参数
    normalize = None
    if args.normalize:
        # ImageNet 统计
        normalize = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        print("使用 ImageNet 归一化")

    # 1. 加载模型
    model = get_model(args.weights, device, num_classes=2)
    model.to(device)
    model.eval()

    # 2. 准备数据集
    dataset = PreprocessedBUSIDataset(
        csv_file=args.csv_file,
        input_size=tuple(args.input_size) if args.input_size else None,
        normalize=normalize
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"测试集样本数量: {len(dataset)}")

    # 3. 初始化指标对象
    metrics = SegmentationMetrics(num_classes=2, pixel_spacing=tuple(args.pixel_spacing))

    # 4. 评估循环
    with torch.no_grad():
        for images, masks, orig_h_list, orig_w_list in tqdm(dataloader, desc="评估中"):
            images = images.to(device)
            masks = masks.to(device)  # (B, H', W')

            # 1. 我们的模型输出的是 4 张深监督图的列表！
            outputs_list = model(images)  
            
            # 2. 我们只取最后一张最清晰的原尺寸图
            final_logits = outputs_list[-1] 
            
            # 3. 把实数(Logits)变成概率(Sigmoid)，再通过 >0.5 变成 0和1 的二值图
            # squeeze(1) 的作用是把形状从 [B, 1, H, W] 变成 [B, H, W]
            preds = (torch.sigmoid(final_logits) > 0.5).squeeze(1).long()

            # 逐样本处理，上采样回原始尺寸
            for i in range(preds.shape[0]):
                pred_small = preds[i]  # (H', W')
                gt_small = masks[i]  # (H', W')
                orig_h = orig_h_list[i].item()
                orig_w = orig_w_list[i].item()
                # 上采样预测图到原始尺寸
                pred_original = resize_pred_to_original(pred_small, orig_h, orig_w)
                gt_original = gt_small.cpu().numpy()
                metrics.update_with_boundary(pred_original, gt_original)

    # 5. 输出结果
    scores = metrics.get_scores()
    print("\n" + "=" * 60)
    print("BUSI 测试集评估结果")
    print("=" * 60)
    print(f"Dice 系数:           {scores['dice']:.4f}")
    print(f"IoU (Jaccard):       {scores['iou']:.4f}")
    print(f"平均 IoU (背景+前景): {scores['miou']:.4f}")
    print(f"准确率 (Accuracy):   {scores['accuracy']:.4f}")
    print(f"精确度 (Precision):  {scores['precision']:.4f}")
    print(f"灵敏度 (Sensitivity):{scores['sensitivity']:.4f}")
    print(f"特异度 (Specificity):{scores['specificity']:.4f}")
    print(
        f"HD95 (均值 ± 标准差): {scores['hd95_mean']:.2f} ± {scores['hd95_std']:.2f} 像素 (有效样本数: {scores['valid_hd95_count']})")
    print(f"混淆矩阵 (前景类): TP={scores['TP']}, FP={scores['FP']}, FN={scores['FN']}, TN={scores['TN']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
