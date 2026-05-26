import torch
import torch.nn as nn

# 1. Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth # 加一个极小的数，防止分母为 0 导致程序崩溃

    def forward(self, pred, target):
        # pred 是模型预测的概率图，target 是真实的黑白标签
        # 把二维图片展平变成一维长条，方便算数学公式
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        # 核心公式：2 * (预测与真实的交集) / (预测的面积 + 真实的面积)
        intersection = (pred * target).sum()
        dice_score = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        # 因为 Loss 是越小越好，而 Dice 分数是越大（1.0）越好，所以要用 1 减去它
        return 1.0 - dice_score

# 2. 复合损失函数 (Total Edge-Aware Loss)
class LEResUNetLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5, weight_edge=0.1):
        super(LEResUNetLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.weight_edge = weight_edge
        
        self.dice_fn = DiceLoss()
        self.bce_fn = nn.BCELoss() 

    def forward(self, pred, true_mask, edge_mask):
        # 1. 算主体的 Dice 误差 
        loss_dice = self.dice_fn(pred, true_mask)
        
        # 2. 算主体的 BCE 误差 
        loss_bce = self.bce_fn(pred, true_mask)
        
        # 3. 算边缘的特供误差 
        loss_edge = self.bce_fn(pred, edge_mask)
        
        total_loss = (self.weight_dice * loss_dice) + \
                     (self.weight_bce * loss_bce) + \
                     (self.weight_edge * loss_edge)
                     
        return total_loss