import torch
import torch.nn as nn
import torch.nn.functional as F

class customLoss(nn.Module):
    def __init__(self, loss_type='bce', weight=None, reduction='mean', pos_weight=None, alpha=0.8, gamma=2, smooth=1e-6):
        """
        Custom loss function that supports BCE, Dice, Focal, and BCE + Dice losses.
        """
        super(CustomLoss, self).__init__()
        self.loss_type = loss_type
        self.bce_loss = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction, pos_weight=pos_weight)
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

    def focal_loss(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()

    def forward(self, input, target):
        if self.loss_type == 'bce':
            return self.bce_loss(input, target)
        elif self.loss_type == 'dice':
            return self.dice_loss(input, target)
        elif self.loss_type == 'focal':
            return self.focal_loss(input, target)
        elif self.loss_type == 'combined':
            bce_loss = self.bce_loss(input, target)
            dice_loss = self.dice_loss(input, target)
            return bce_loss + dice_loss
        elif self.loss_type == 'l1':
            return F.l1_loss(input, target)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}. Choose from 'bce', 'dice', 'focal', 'combined'.")
