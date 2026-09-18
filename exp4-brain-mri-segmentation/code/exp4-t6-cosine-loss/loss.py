import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0
        self.epsilon = 1e-8
        self.focal_gamma = 2.0
        self.false_neg_weight = 0.8
        self.small_target_thr = 0.01

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        device = y_pred.device

        y_pred = torch.sigmoid(y_pred)

        total_pixels = y_true.numel()
        foreground_pixels = y_true.sum()
        background_pixels = total_pixels - foreground_pixels

        fg_weight = total_pixels / (foreground_pixels + self.epsilon)
        bg_weight = total_pixels / (background_pixels + self.epsilon)
        pixel_weights = (y_true * fg_weight) + ((1 - y_true) * bg_weight)

        is_small_target = (foreground_pixels / total_pixels) < self.small_target_thr
        if is_small_target:
            pixel_weights = pixel_weights * 1.5

        weighted_intersection = (y_pred * y_true * pixel_weights).sum()
        weighted_union = (y_pred * pixel_weights).sum() + \
                         (y_true * pixel_weights).sum() + \
                         (self.false_neg_weight * (1 - y_pred) * y_true * pixel_weights).sum()

        pred_grad = torch.abs(y_pred[:-1] - y_pred[1:])
        true_grad = torch.abs(y_true[:-1] - y_true[1:])
        pred_boundary = (pred_grad > 0.1).float()
        true_boundary = (true_grad > 0.1).float()
        boundary_inter = (pred_boundary * true_boundary).sum()
        boundary_union = pred_boundary.sum() + true_boundary.sum()
        boundary_iou = (boundary_inter + self.smooth) / (boundary_union + self.smooth + self.epsilon)

        confidence_penalty = torch.mean(
            torch.where(
                (y_pred > 0.3) & (y_pred < 0.7),
                (y_pred - torch.round(y_pred)) ** 2,
                torch.zeros_like(y_pred)
            )
        )

        base_dice = (2 * weighted_intersection + self.smooth) / (weighted_union + self.smooth + self.epsilon)
        dice_loss = 1.0 - base_dice
        total_loss = dice_loss + 0.15 * (1 - boundary_iou) + 0.08 * confidence_penalty

        return total_loss