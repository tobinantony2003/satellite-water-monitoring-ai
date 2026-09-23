import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes soft Dice loss.
        logits: (B, 1, H, W) raw model outputs before sigmoid
        targets: (B, 1, H, W) binary ground truth {0, 1}
        """
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions
        probs_flat = probs.contiguous().view(probs.size(0), -1)
        targets_flat = targets.contiguous().view(targets.size(0), -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        return (1.0 - dice).mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, pos_weight: float = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        else:
            self.bce = nn.BCEWithLogitsLoss()
            
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
