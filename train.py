
import os
import torch
import torch.optim as optim
from tqdm import tqdm 
import numpy as np

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
    print(f" 数据加载成功！训练集batch数: {len(train_loader)}, 验证集batch数: {len(val_loader)}")

    # 3. 初始化网络与工具
    model = OurBreastCancerNet(pretrained=True).to(DEVICE)
    criterion = DBDSLoss(max_epochs=EPOCHS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    os.makedirs("results/weights", exist_ok=True)
    best_val_dice = 0.0 

    # 4. 训练与验证大循环
    for epoch in range(EPOCHS):
        model.train() # 开启训练模式
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train")
        for images, masks, edges in train_bar:
            images, masks, edges = images.to(DEVICE), masks.to(DEVICE), edges.to(DEVICE)
            
            # 1. 梯度清零
            optimizer.zero_grad()
            
            # 2. 前向传播：模型吐出 4 张深监督预测图
            # preds_list = [草稿1, 草稿2, 草稿3, 最终成品]
            preds_list = model(images)
            
            # 3. 计算误差：把 epoch 传进去，激活 DBDS 的动态机制！
            loss = criterion(preds_list, masks, edge_mask=edges, current_epoch=epoch)
            
            # 4. 反向传播与参数更新
            loss.backward()
            optimizer.step()
            
            # 累加 Loss 用于打印
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # 计算这个 epoch 的平均 Loss
        avg_train_loss = train_loss / len(train_loader)
        

    # 验证阶段 
    model.eval()
    
    val_metrics = SegmentationMetrics(num_classes=2)
    
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Val  ")
        # 接收三个变量（图片，金标准，真实边界）
        for images, masks, edges in val_bar:
            images = images.to(DEVICE)
            
            # 模型前向传播，拿到 4 个深监督图
            preds_list = model(images)
            final_pred_logits = preds_list[-1] 
            
            # 1. Sigmoid 把实数变成 0~1 概率
            final_pred_probs = torch.sigmoid(final_pred_logits)
            # 2. >0.5 把概率变成 0和1 的布尔值，再转成 np.uint8 整数类型
            final_pred_binary = (final_pred_probs > 0.5).cpu().numpy().astype(np.uint8)
            
            # 把 GPU 上的真实标签也搬到 CPU 并转成 numpy 数组
            masks_np = masks.cpu().numpy().astype(np.uint8)
            
            # 逐张图片更新混淆矩阵和 HD95 距离
            for i in range(images.size(0)):
                val_metrics.update_with_boundary(final_pred_binary[i], masks_np[i])
                
    scores = val_metrics.get_scores()
    val_dice = scores['dice']
    val_iou = scores['iou']
    val_hd95 = scores['hd95_mean']
    
    print(f" Epoch [{epoch+1}] 成绩单 | Train Loss: {avg_train_loss:.4f}")
    print(f" 验证集表现 -> Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | HD95: {val_hd95:.2f} 像素")

    # 保存最佳模型 (根据验证集 Dice) 
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        save_path = "results/weights/best_our_model.pth"
        torch.save(model.state_dict(), save_path)
        print(f" 最佳模型 (最佳 Dice: {best_val_dice:.4f}) 模型已保存！\n")
    else:
        print("\n")
        

if __name__ == '__main__':
    train_and_validate()
        else:
            print("\n")

if __name__ == '__main__':
    train_and_validate()
