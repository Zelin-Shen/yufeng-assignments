import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    优化版复合损失函数：Dice + Focal Loss，专注医疗图像分割的稳定性与精度提升
    """
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.8, focal_weight=0.2, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.alpha = alpha          # Focal Loss前景权重系数
        self.gamma = gamma          # Focal Loss难样本聚焦参数
        self.dice_weight = dice_weight  # Dice损失权重
        self.focal_weight = focal_weight  # Focal损失权重
        self.smooth = smooth        # 平滑项防除零

    def forward(self, y_pred, y_true):
        # 输入校验与形状处理
        assert y_pred.size() == y_true.size()
        batch_size = y_pred.shape[0]
        y_pred = torch.sigmoid(y_pred)  # 确保输出为概率值
        y_pred = y_pred.contiguous().view(batch_size, -1)  # 展平为[B, N]
        y_true = y_true.contiguous().view(batch_size, -1)

        # 1. Dice损失计算（核心区域匹配）
        intersection = (y_pred * y_true).sum(dim=1)
        union = y_pred.sum(dim=1) + y_true.sum(dim=1)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1. - dice_score.mean()

        # 2. Focal损失计算（自适应处理不平衡与难样本）
        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        pt = torch.exp(-bce_loss)  # 预测概率的置信度
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss.mean()

        # 3. 加权复合损失
        total_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss
        return total_loss