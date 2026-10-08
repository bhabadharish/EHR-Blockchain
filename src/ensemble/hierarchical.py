"""Hierarchical Detection Architecture for HAB-IDS (Phase 18).

Two-stage hierarchical intrusion detection:
- Stage 1: High-precision, low-FNR binary detection (Normal vs Attack).
- Stage 2: Attack family classification conditioned on Stage-1 attack detection.
Rare attack categories are evaluated with macro-F1 and balanced weighting.
"""

import os
import joblib
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report


class HierarchicalHABIDS:
    """Hierarchical two-stage security classification architecture."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.stage2_classifier = lgb.LGBMClassifier(
            n_estimators=250,
            num_leaves=35,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1
        )
        self.family_labels_: List[str] = []
        self.is_fitted: bool = False

    def fit_stage2(
        self,
        X_attack_train: np.ndarray,
        y_attack_families_train: np.ndarray,
        family_names: List[str]
    ) -> "HierarchicalHABIDS":
        """Fit Stage-2 multiclass classifier exclusively on attack records."""
        self.family_labels_ = family_names
        self.stage2_classifier.fit(X_attack_train, y_attack_families_train)
        self.is_fitted = True
        return self

    def predict_hierarchical(
        self,
        X: np.ndarray,
        stage1_binary_preds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Classify end-to-end:

        - Samples predicted 0 (Benign) are assigned family 0 ("Benign").
        - Samples predicted 1 (Attack) are evaluated by Stage-2 multiclass model.
        Returns (binary_predictions, family_predictions).
        """
        n_samples = len(X)
        family_preds = np.zeros(n_samples, dtype=int)

        attack_mask = stage1_binary_preds == 1
        if np.any(attack_mask) and self.is_fitted:
            # Predict attack family for detected attacks (offset by +1 since 0 is Benign)
            stage2_out = self.stage2_classifier.predict(X[attack_mask])
            family_preds[attack_mask] = stage2_out + 1

        return stage1_binary_preds, family_preds

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "HierarchicalHABIDS":
        return joblib.load(filepath)
