"""Disagreement-Aware Routing Engine for HAB-IDS (Phase 17).

Routes incoming inference flows dynamically based on base-model disagreement:
- Low-disagreement samples: routed to fast, direct calibrated meta-learner.
- High-disagreement samples: routed to specialized secondary classifier trained on difficult boundary cases.
Disagreement threshold is determined strictly on validation data.
"""

import os
import joblib
from typing import Dict, Any, Tuple, Optional
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score


class DisagreementRouter:
    """Adaptive routing between consensus meta-learner and secondary specialist."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.threshold_: float = 0.25  # default; learned on validation
        self.specialist_: Optional[Any] = None
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        train_disagreements: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        val_disagreements: np.ndarray,
        val_meta_probs: np.ndarray,
        val_threshold: float = 0.5
    ) -> "DisagreementRouter":
        """Fit secondary specialist on high-disagreement train samples and select optimal threshold on validation."""
        # 1. Identify high-disagreement training samples (top 20%)
        initial_cutoff = float(np.percentile(train_disagreements, 80))
        high_disagree_mask = train_disagreements >= initial_cutoff

        # Train specialized LightGBM on high-uncertainty subset
        self.specialist_ = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.03,
            num_leaves=31,
            random_state=self.seed,
            class_weight="balanced",
            verbose=-1
        )
        if np.sum(high_disagree_mask) > 100:
            self.specialist_.fit(X_train[high_disagree_mask], y_train[high_disagree_mask])
        else:
            self.specialist_.fit(X_train, y_train)

        # 2. Sweep candidate disagreement thresholds on VALIDATION ONLY
        best_f1 = f1_score(y_val, (val_meta_probs >= val_threshold).astype(int), average="macro")
        best_thresh = initial_cutoff
        val_spec_probs = self.specialist_.predict_proba(X_val)[:, 1]

        for pct in [70, 75, 80, 85, 90, 95]:
            cand_thresh = float(np.percentile(val_disagreements, pct))
            cand_high = val_disagreements >= cand_thresh

            hybrid_probs = val_meta_probs.copy()
            hybrid_probs[cand_high] = val_spec_probs[cand_high]
            hybrid_preds = (hybrid_probs >= val_threshold).astype(int)

            score = f1_score(y_val, hybrid_preds, average="macro")
            if score > best_f1:
                best_f1 = score
                best_thresh = cand_thresh

        self.threshold_ = best_thresh
        self.is_fitted = True
        return self

    def route_predict_proba(
        self,
        X: np.ndarray,
        meta_probs: np.ndarray,
        disagreements: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict probabilities with routing logic. Returns (final_probs, is_high_disagreement_mask)."""
        if not self.is_fitted:
            return meta_probs, np.zeros_like(meta_probs, dtype=bool)

        high_mask = disagreements >= self.threshold_
        final_probs = meta_probs.copy()

        if np.any(high_mask) and self.specialist_ is not None:
            spec_probs = self.specialist_.predict_proba(X[high_mask])[:, 1]
            final_probs[high_mask] = spec_probs

        return final_probs, high_mask

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "DisagreementRouter":
        return joblib.load(filepath)
