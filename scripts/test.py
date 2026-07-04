# test_best_model.py
import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import csv

# 导入与训练代码相同的组件
from models.ours import OurBreastCancerNet
from preprocess.dataset import get_loaders
from evaluate.eval import SegmentationMetrics

def test_best_model():
    # 1. 设置设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 正在使用计算设备: {DEVICE}")

    # 2. 加载测试集（使用与训练相同的预处理）
    print(" 正在加载测试集...")
    _, _, test_loader = get_loaders(
        train_csv='preprocess/train.csv', 
        val_csv='preprocess/val.csv', 
        test_csv='preprocess/test.csv',
        batch_size=1,  # 测试时通常使用 batch_size=1
        use_lee=True,
        use_clahe=True
    )
    print(f"测试集样本数: {len(test_loader.dataset)}")

    # 3. 初始化模型并加载最佳权重
    print(" 正在加载最佳模型权重...")
    model = OurBreastCancerNet(pretrained=False).to(DEVICE)  # pretrained=False 因为我们加载本地权重
    weights_path = "results/weights/best_our_model.pth"
    
    if not os.path.exists(weights_path):
        print(f"错误: 权重文件 {weights_path} 不存在!")
        return None
    
    # 加载权重（处理可能的 DataParallel 前缀）
    state_dict = torch.load(weights_path, map_location=DEVICE)
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"成功加载权重: {weights_path}")

    # 4. 初始化评估指标
    metrics = SegmentationMetrics(num_classes=2)

    # 5. 测试循环
    print("\n 开始在测试集上评估最佳模型...")
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
            
            # 取最后一个输出（最高分辨率），并上采样到原始尺寸
            final_pred_logits = preds_list[-1]
            if final_pred_logits.shape[-2:] != (target_h, target_w):
                final_pred_logits = F.interpolate(
                    final_pred_logits, 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # 使用0.5阈值二值化（与训练时验证一致）
            final_pred_probs = torch.sigmoid(final_pred_logits)
            if final_pred_probs.dim() == 4:
                final_pred_probs = final_pred_probs.squeeze(1)
            final_pred_binary = (final_pred_probs > 0.65).cpu().numpy().astype(np.uint8)
            
            # 更新评估指标
            for i in range(images.size(0)): 
                metrics.update_with_boundary(final_pred_binary[i], masks_np[i])
                
            # 更新进度条显示当前指标
            current_scores = metrics.get_scores()
            test_bar.set_postfix({
                'Dice': f"{current_scores['dice']:.4f}",
                'IoU': f"{current_scores['iou']:.4f}"
            })

    # 6. 输出最终结果
    scores = metrics.get_scores()
    print("\n" + "="*60)
    print(" 测试集 (Test Set) 最终成绩单 ")
    print("="*60)
    print(f"Dice 系数:           {scores['dice']:.4f}")
    print(f"IoU (Jaccard):       {scores['iou']:.4f}")
    print(f"平均 IoU (背景+前景): {scores['miou']:.4f}")
    print(f"准确率 (Accuracy):   {scores['accuracy']:.4f}")
    print(f"精确度 (Precision):  {scores['precision']:.4f}")
    print(f"灵敏度 (Sensitivity):{scores['sensitivity']:.4f}")
    print(f"特异度 (Specificity):{scores['specificity']:.4f}")
    print(f"HD95 (均值):         {scores['hd95_mean']:.2f} 像素 (有效样本数: {scores['valid_hd95_count']})")
    print(f"混淆矩阵 (前景类): TP={scores['TP']}, FP={scores['FP']}, FN={scores['FN']}, TN={scores['TN']}")
    print("="*60)

    # 7. 保存测试结果到文件
    save_test_results(scores)
    
    return scores

def save_test_results(scores):
    """保存测试结果到文件"""
    os.makedirs("results", exist_ok=True)
    
    # 保存为文本文件
    with open("results/test_results.txt", "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write(" 测试集 (Test Set) 最终成绩单 \n")
        f.write("="*60 + "\n")
        f.write(f"Dice 系数:           {scores['dice']:.4f}\n")
        f.write(f"IoU (Jaccard):       {scores['iou']:.4f}\n")
        f.write(f"平均 IoU (背景+前景): {scores['miou']:.4f}\n")
        f.write(f"准确率 (Accuracy):   {scores['accuracy']:.4f}\n")
        f.write(f"精确度 (Precision):  {scores['precision']:.4f}\n")
        f.write(f"灵敏度 (Sensitivity):{scores['sensitivity']:.4f}\n")
        f.write(f"特异度 (Specificity):{scores['specificity']:.4f}\n")
        f.write(f"HD95 (均值):         {scores['hd95_mean']:.2f} 像素\n")
        f.write(f"有效 HD95 样本数:    {scores['valid_hd95_count']}\n")
        f.write(f"混淆矩阵 (前景类): TP={scores['TP']}, FP={scores['FP']}, FN={scores['FN']}, TN={scores['TN']}\n")
        f.write("="*60 + "\n")
    
    # 保存为 CSV 文件（方便后续分析）
    csv_path = "results/test_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Dice", f"{scores['dice']:.6f}"])
        writer.writerow(["IoU", f"{scores['iou']:.6f}"])
        writer.writerow(["mIoU", f"{scores['miou']:.6f}"])
        writer.writerow(["Accuracy", f"{scores['accuracy']:.6f}"])
        writer.writerow(["Precision", f"{scores['precision']:.6f}"])
        writer.writerow(["Sensitivity", f"{scores['sensitivity']:.6f}"])
        writer.writerow(["Specificity", f"{scores['specificity']:.6f}"])
        writer.writerow(["HD95_Mean", f"{scores['hd95_mean']:.6f}"])
        writer.writerow(["Valid_HD95_Count", scores['valid_hd95_count']])
        writer.writerow(["TP", scores['TP']])
        writer.writerow(["FP", scores['FP']])
        writer.writerow(["FN", scores['FN']])
        writer.writerow(["TN", scores['TN']])
    
    print(f" 测试结果已保存至: results/test_results.txt")
    print(f" CSV 格式结果已保存至: {csv_path}")

if __name__ == '__main__':
    test_best_model()