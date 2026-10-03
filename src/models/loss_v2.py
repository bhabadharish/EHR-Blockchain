"""
src/models/loss_v2.py
=====================
ASYMMETRIC CLASS-BALANCED FOCAL MARGIN LOSS (V2)

Specifically designed to solve the minority-class precision degradation problem
in cyber-threat detection by:
1. Inverse-frequency / Effective-number class weighting.
2. Focal modulation (gamma) focusing gradient updates on difficult boundary samples.
3. Additive margin constraint enforcing a separation distance between Normal and Attack logits.
4. Auxiliary Risk calibration loss aligning predicted risk scores with threat severity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class AsymmetricClassBalancedMarginLoss(nn.Module):
    """
    Multi-objective loss combining Class-Balanced Focal Loss with an additive decision margin.
    """
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        margin: float = 0.15,
        label_smoothing: float = 0.02,
        aux_risk_weight: float = 0.20
    ):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.margin = margin
        self.label_smoothing = label_smoothing
        self.aux_risk_weight = aux_risk_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        risk_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        logits: (batch_size, 2)
        targets: (batch_size,) integer class labels (0=Normal, 1=Attack)
        risk_scores: (batch_size, 1) continuous risk output
        """
        targets = targets.long()
        num_classes = logits.size(-1)

        # 1. Softmax probabilities
        probs = F.softmax(logits, dim=-1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # prob of true class

        # 2. Focal Modulation Factor: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - p_t, self.gamma)

        # 3. Class Weights
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            w_t = weights.gather(0, targets)
        else:
            w_t = 1.0

        # 4. Smoothed Cross-Entropy with focal weight
        log_probs = F.log_softmax(logits, dim=-1)
        # One-hot target with smoothing
        with torch.no_grad():
            smooth_targets = torch.full_like(probs, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        focal_ce_loss = (w_t * focal_weight * ce_loss).mean()

        # 5. Additive Decision Margin Constraint
        # Enforces: logit(true) - logit(false) >= margin
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        other_class = 1 - targets
        true_logits = logits[batch_indices, targets]
        false_logits = logits[batch_indices, other_class]
        margin_violations = F.relu(self.margin - (true_logits - false_logits))
        margin_loss = margin_violations.mean()

        # 6. Auxiliary Risk Score Alignment Loss
        if risk_scores is not None:
            risk_targets = targets.float().unsqueeze(1)
            risk_loss = F.binary_cross_entropy(risk_scores, risk_targets)
        else:
            risk_loss = torch.tensor(0.0, device=logits.device)

        total_loss = focal_ce_loss + (0.5 * margin_loss) + (self.aux_risk_weight * risk_loss)

        return {
            "total_loss": total_loss,
            "focal_ce_loss": focal_ce_loss,
            "margin_loss": margin_loss,
            "risk_loss": risk_loss
        }
