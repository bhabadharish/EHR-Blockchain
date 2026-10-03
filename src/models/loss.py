"""
src/models/loss.py
==================
LOSS FUNCTIONS FOR IMBALANCED CYBERSECURITY & HEALTHCARE THREAT DETECTION

Supports:
1. Standard Cross-Entropy
2. Weighted Cross-Entropy (handling ~93% attack vs 7% normal imbalance)
3. Cost-Sensitive Focal Loss (dynamically focusing on hard samples with asymmetric false negative penalty)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CostSensitiveFocalLoss(nn.Module):
    """
    Cost-Sensitive Focal Loss with optional class weights and focal modulation.
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(
        self,
        alpha: float = 0.50,
        gamma: float = 2.0,
        fn_penalty: float = 1.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.fn_penalty = fn_penalty
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (batch_size, num_classes)
        # targets: (batch_size,)
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=-1)
        
        # Gather probabilities of true classes
        targets = targets.view(-1, 1)
        target_probs = probs.gather(1, targets).squeeze(-1) # p_t
        target_probs = torch.clamp(target_probs, min=1e-7, max=1.0 - 1e-7)

        # Cross entropy term: -log(p_t)
        log_pt = torch.log(target_probs)
        
        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - target_probs, self.gamma)

        # Class weights if provided
        if self.class_weights is not None:
            w = self.class_weights.gather(0, targets.squeeze(-1))
        else:
            w = torch.ones_like(target_probs)

        # Asymmetric False Negative penalty (true attack classified with low confidence)
        if self.fn_penalty > 1.0:
            fn_multiplier = torch.where(targets.squeeze(-1) == 1, self.fn_penalty, 1.0)
        else:
            fn_multiplier = 1.0

        loss = -1.0 * w * fn_multiplier * focal_weight * log_pt
        return loss.mean()
