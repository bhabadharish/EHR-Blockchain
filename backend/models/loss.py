"""
backend/models/loss.py
Custom loss functions and class imbalance mitigation strategies (Phase 17).
Implements Multi-class Focal Loss and class weighting calculations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

def compute_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Computes balanced class weights: w_c = N / (C * N_c).
    """
    classes, counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    weights = np.ones(num_classes, dtype=np.float32)
    
    for c, count in zip(classes, counts):
        if count > 0:
            weights[c] = total_samples / (num_classes * count)
            
    # Clip extreme weights to prevent gradient explosion
    weights = np.clip(weights, 0.1, 50.0)
    return torch.tensor(weights, dtype=torch.float32)

class MultiClassFocalLoss(nn.Module):
    """
    Multi-Class Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy examples and focuses training on hard/rare attack classes.
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (batch_size, num_classes)
        # targets: (batch_size,)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha.to(inputs.device) if self.alpha is not None else None)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
