
import os
import torch
import torch.optim as optim
from tqdm import tqdm 

# 导入我们的模型和 Loss
from models.ours import OurBreastCancerNet
from loss import DBDSLoss, dice_loss
# 导入预处理模块
from preprocess.dataset import get_loaders

def train_and_validate():
    # 1. 超参数设置
    EPOCHS = 100
    BATCH_SIZE = 4       
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
        model.eval() # 关闭 Dropout 和 BatchNorm 的参数更新，开启验证
        val_dice_score = 0.0
        
        with torch.no_grad(): 
            val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Val  ")
            for images, masks, edges in val_bar:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                
                # 前向传播拿到预测结果
                preds_list = model(images)
                final_pred = preds_list[-1]
                
                # 计算验证集上的 Dice 精度 
                batch_dice = 1.0 - dice_loss(final_pred, masks).item()
                val_dice_score += batch_dice
                
        avg_val_dice = val_dice_score / len(val_loader)
        
        # 打印这一轮的成绩单
        print(f" 总结 | 训练 Loss: {avg_train_loss:.4f} | 验证集 Dice 精度: {avg_val_dice:.4f}")

        # 保存最佳模型 
        # 核心逻辑：只在验证集 Dice 变高时，才保存模型，防止过拟合
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            save_path = "results/weights/best_our_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f" 验证集表现提升！(最佳 Dice: {best_val_dice:.4f}) 模型已保存\n")
        else:
            print("\n")

if __name__ == '__main__':
    train_and_validate()