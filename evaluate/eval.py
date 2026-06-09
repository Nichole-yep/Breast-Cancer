计算指标：Dice、IoU、平均IoU、准确率、灵敏度、特异度、HD95
import os
import argparse
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from scipy import ndimage

# --------------------------- 1. 数据集定义---------------------------
class PreprocessedBUSIDataset(Dataset):
    """加载预处理后的 BUSI 测试集"""
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['images']
        mask_path = self.df.iloc[idx]['masks']
        # 读取图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 读取标签
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)   # 确保二值化
        h, w = img.shape[:2]   
        return img, mask, h, w

def get_simple_transform():
    """无需缩放和归一化的最小预处理（仅将图像转为Tensor）"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()   # 只转换，不归一化
    ])

# --------------------------- 2. 评估指标计算类---------------------------
class SegmentationMetrics:
    def __init__(self, num_classes=2, pixel_spacing=(1.0, 1.0)):
        self.num_classes = num_classes
        self.pixel_spacing = pixel_spacing
        self.reset()

    def reset(self):
        #重置混淆矩阵和HD95列表
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.hd95_list = []

    def update(self, y_true, y_pred):
        #更新混淆矩阵（单张图像的标签与预测）
        hist = self._fast_hist(y_true.flatten(), y_pred.flatten())
        self.confusion_matrix += hist

    def update_with_boundary(self, pred_mask, gt_mask):
        #同时更新混淆矩阵并计算当前图像的HD95
        self.update(gt_mask, pred_mask)
        pred_boundary = self._extract_boundary(pred_mask)
        gt_boundary = self._extract_boundary(gt_mask)
        if len(pred_boundary) > 0 and len(gt_boundary) > 0:
            hd95 = self._compute_hd95(pred_boundary, gt_boundary, self.pixel_spacing)
            self.hd95_list.append(hd95)

    def _fast_hist(self, label_true, label_pred):
        #计算两个展平数组的混淆矩阵
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask] + label_pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def _extract_boundary(self, mask):
        #提取二值掩码的边界像素
        if mask.ndim == 3:
            mask = mask.squeeze()
        struct = ndimage.generate_binary_structure(2, 1)
        eroded = ndimage.binary_erosion(mask, struct)
        boundary = mask ^ eroded
        return np.argwhere(boundary)

    def _compute_hd95(self, pred_points, gt_points, spacing):
        #计算95%豪斯多夫距离（HD95）
        pred_scaled = pred_points * spacing
        gt_scaled = gt_points * spacing
        d_pred_gt = []
        for p in pred_scaled:
            if len(gt_scaled) == 0:
                continue
            min_dist = np.min(np.linalg.norm(gt_scaled - p, axis=1))
            d_pred_gt.append(min_dist)
        d_gt_pred = []
        for g in gt_scaled:
            if len(pred_scaled) == 0:
                continue
            min_dist = np.min(np.linalg.norm(pred_scaled - g, axis=1))
            d_gt_pred.append(min_dist)
        if len(d_pred_gt) == 0 or len(d_gt_pred) == 0:
            return float('inf')
        # 取双向距离的95百分位数的最大值
        hd95 = max(np.percentile(d_pred_gt, 95), np.percentile(d_gt_pred, 95))
        return hd95

    def get_scores(self):
        #计算并返回所有评估指标
        hist = self.confusion_matrix
        tp = np.diag(hist)
        fp = hist.sum(axis=0) - tp
        fn = hist.sum(axis=1) - tp
        tn = hist.sum() - (tp + fp + fn)
        eps = 1e-8  # 防止除零
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
            'TP': tp[1],
            'FP': fp[1],
            'FN': fn[1],
            'TN': tn[1],
        }
        return scores

# --------------------------- 3. 主评估流程---------------------------
def main():
    parser = argparse.ArgumentParser(description='使用预处理后的CSV评估 BUSI 分割模型')
    parser.add_argument('--weights', type=str, required=True, help='模型权重文件 (.pt 或 .pth)')
    parser.add_argument('--csv_file', type=str, required=True, help='测试集CSV文件（含 images 和 masks 列）')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--device', type=str, default='cuda', help='运行设备')
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载模型
    try:
        model = torch.load(args.weights, map_location=device)
    except:
        # 若权重为 state_dict，需先定义模型类（请替换为你的模型定义）
        from your_model_file import YourModel
        model = YourModel(num_classes=2)
        model.load_state_dict(torch.load(args.weights, map_location=device))
        model.to(device)
    model.eval()

    # 2. 准备数据集（使用CSV）
    dataset = PreprocessedBUSIDataset(csv_file=args.csv_file, transform=get_simple_transform())
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    print(f"测试集样本数量: {len(dataset)}")

    # 3. 评估指标对象（像素间距默认1，单位为像素）
    metrics = SegmentationMetrics(num_classes=2, pixel_spacing=(1.0, 1.0))

    # 4. 评估循环
    with torch.no_grad():
        for images, masks, orig_h, orig_w in tqdm(dataloader, desc="评估中"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(preds_np.shape[0]):
                metrics.update_with_boundary(preds_np[i], masks_np[i])

    # 5. 输出结果
    scores = metrics.get_scores()
    print("\n" + "="*60)
    print("BUSI 测试集评估结果（使用预处理数据）")
    print("="*60)
    print(f"Dice 系数:           {scores['dice']:.4f}")
    print(f"IoU (Jaccard):       {scores['iou']:.4f}")
    print(f"平均 IoU (背景+前景): {scores['miou']:.4f}")
    print(f"准确率 (Accuracy):   {scores['accuracy']:.4f}")
    print(f"精确度 (Precision):  {scores['precision']:.4f}")
    print(f"灵敏度 (Sensitivity):{scores['sensitivity']:.4f}")
    print(f"特异度 (Specificity):{scores['specificity']:.4f}")
    print(f"HD95 (均值 ± 标准差): {scores['hd95_mean']:.2f} ± {scores['hd95_std']:.2f} 像素")
    print(f"混淆矩阵 (前景类): TP={scores['TP']}, FP={scores['FP']}, FN={scores['FN']}, TN={scores['TN']}")
    print("="*60)

    # 可选：保存结果
    # pd.DataFrame([scores]).to_csv('evaluation_results.csv', index=False)


if __name__ == '__main__':
    main()
