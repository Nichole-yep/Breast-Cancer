import torch
from loss import LEResUNetLoss 
from preprocess.dataset import get_dataloader  
from methods.model import LEResUNet       

def main():
    # 1. 准备机器和材料
    train_loader = get_dataloader(...) # 拿到装满张量图片的数据集
    model = LEResUNet()                # 建好模型
    
    # 2. 准备 损失函数 和 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = LEResUNetLoss(weight_dice=0.5, weight_bce=0.5, weight_edge=0.1)
    
    # 3. 开启训练循环 
    for epoch in range(50):    #50遍
        model.train() 
        
        # 从 DataLoader 里一批一批地抓取图片和标签
        for images, masks, edge_masks in train_loader:
            
            predictions = model(images) 
            
            # 算误差
            total_loss = criterion(predictions, masks, edge_masks)
            
            # 反向传播
            optimizer.zero_grad() 
            total_loss.backward() 
            optimizer.step()      
            
        print(f"Epoch {epoch} 完成，误差: {total_loss.item()}")

if __name__ == "__main__":
    main()