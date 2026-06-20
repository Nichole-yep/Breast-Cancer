import torch
import torch.nn as nn
import torch.nn.functional as F


# tversky_loss
def tversky_loss(pred, target, alpha=0.2, beta=0.8, smooth=1e-5):
    """
    alpha: 对假阴性(FN)的惩罚权重
    beta: 对假阳性(FP)的惩罚权重。beta 设大一点 (如0.8)，强迫模型少预测孤立噪点！
    """
    pred = torch.sigmoid(pred)
    
    # 展平以便计算
    pred = pred.view(-1)
    target = target.view(-1)
    
    # True Positives
    TP = (pred * target).sum()
    # False Positives (预测了1但真实是0) -> 导致高 HD95 的罪魁祸首！
    FP = (pred * (1.0 - target)).sum()
    # False Negatives (预测了0但真实是1)
    FN = ((1.0 - pred) * target).sum()
    
    # 计算 Tversky 系数
    tversky = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)
    
    return 1.0 - tversky


# 1. 计算 Dice Loss 
def dice_loss(pred, target, smooth=1e-5):
    # 将预测结果通过 sigmoid 映射到 0~1 的概率
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

# 3. DBDS 动态边界深监督损失函数
class DBDSLoss(nn.Module):
    def __init__(self, max_epochs=100, pos_weight=15.0):
        super(DBDSLoss, self).__init__()
        self.max_epochs = max_epochs
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        # 新增pos_weight: 前景像素的基础惩罚倍数，用于对抗类别不平衡
        self.pos_weight = pos_weight
        

    def forward(self, preds_list, target, edge_mask, current_epoch):
        """
        preds_list: 解码器输出的 4 个预测图的列表 [草稿1, 草稿2, 草稿3, 最终成品]
        target: 真实金标准 Mask [B, 1, H, W]
        current_epoch: 当前训练到第几个 epoch
        """
        total_loss = 0.0
        
        # 动态机制 (Dynamic): 随着 epoch 增加，边界损失的权重从 0 慢慢增大到 1
        dynamic_weight = min(1.0, current_epoch / (self.max_epochs * 0.5))
        # 新增：动态深监督权重
        # 随着训练进行，让低分辨率草稿的权重逐渐衰减到 0，最终输出的权重逐渐提升到 1.0
        progress = current_epoch / self.max_epochs
        w0 = max(0.0, 0.1 - 0.2 * progress) # 草稿1：逐渐归零
        w1 = max(0.0, 0.2 - 0.4 * progress) # 草稿2：逐渐归零
        w2 = max(0.0, 0.3 - 0.4 * progress) # 草稿3：逐渐归零
        w3 = 1.0 - (w0 + w1 + w2)           # 成品：吸收所有权重，后期接近 1.0
        current_ds_weights = [w0, w1, w2, w3]
        
        # 深监督机制 (Deep Supervision): 循环遍历 4 个尺度的输出，分别算 Loss
        for i, pred in enumerate(preds_list):
            
            # 如果当前是草稿图（尺寸比 target 小），就先放大到和 target 一样大
            if pred.shape[2:] != target.shape[2:]:
                pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)

            # 同样将 edge_mask 对齐尺寸
            if edge_mask.shape[2:] != pred.shape[2:]:
                curr_edge = F.interpolate(edge_mask.float(), size=pred.shape[2:], mode='nearest')
            else:
                curr_edge = edge_mask.float()

            # 1. 算每个像素的基础 BCE Loss
            pixel_bce = self.bce(pred, target)
            
            # 2. 生成权重地图 (Weight Map)
            # 基础分为 1。如果是边缘像素 (curr_edge==1)，额外加上 dynamic_weight * 5.0 的惩罚倍数
            #漏掉前景时付出更大的BCE损失代价
            base_weight_map = 1.0 + target * (self.pos_weight - 1.0)

            #将基础权重于动态边缘权重叠加
            weight_map = base_weight_map + (curr_edge * dynamic_weight * 5.0)
            
            # 3. 把权重乘到 BCE 上，然后求平均
            weighted_bce = (pixel_bce * weight_map).mean()

            # 4. 融合 Dice Loss
            # layer_loss = weighted_bce + dice_loss(pred, target)
            layer_loss = weighted_bce + tversky_loss(pred, target, alpha=0.2, beta=0.8)

            total_loss +=current_ds_weights[i] * layer_loss

        return total_loss