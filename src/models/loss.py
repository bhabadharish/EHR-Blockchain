import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CostSensitiveFocalLoss(nn.Module):
    """
    Cost-Sensitive Focal Loss for high-security intrusion detection.
    Penalizes False Negatives (critical missed cyber attacks) with higher weight
    while dynamically downweighting easy well-classified negative examples.
    """
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        fn_penalty: float = 2.5,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.fn_penalty = fn_penalty
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (batch, num_classes)
        # targets: (batch)
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.class_weights)
        p = torch.exp(-ce_loss)
        
        # Standard focal factor
        focal_weight = (1.0 - p) ** self.gamma
        
        # Cost-sensitive asymmetric false-negative penalty on attack class (label 1)
        cost_multiplier = torch.where(targets == 1, self.fn_penalty, 1.0)
        
        loss = cost_multiplier * focal_weight * ce_loss
        return loss.mean()
