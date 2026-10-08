"""Adaptive Disagreement-Aware Meta-Learner for HAB-IDS (Phase 16).

Extracts multi-model consensus and disagreement features:
- Base probabilities: P_xgb, P_lgb, P_cat
- Summary statistics: max(P), min(P), mean(P), std(P)
- Disagreement features:
    * D_range = max(P) - min(P)
    * D_std = standard deviation of model predictions
    * D_pairwise = |P_xgb - P_lgb|, |P_xgb - P_cat|, |P_lgb - P_cat|
- Information metrics: Shannon prediction entropy, confidence gap
Trains a regularized meta-learner strictly on out-of-fold training predictions.
"""

import os
import joblib
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb


def compute_meta_features(
    p_xgb: np.ndarray,
    p_lgb: np.ndarray,
    p_cat: np.ndarray
) -> np.ndarray:
    """Compute consensus, disagreement, and entropy features from base booster probabilities."""
    # Ensure 1D arrays of positive class probability
    p1 = np.asarray(p_xgb, dtype=np.float32).ravel()
    p2 = np.asarray(p_lgb, dtype=np.float32).ravel()
    p3 = np.asarray(p_cat, dtype=np.float32).ravel()

    # Stack: shape (N, 3)
    p_stack = np.column_stack([p1, p2, p3])

    p_max = np.max(p_stack, axis=1)
    p_min = np.min(p_stack, axis=1)
    p_mean = np.mean(p_stack, axis=1)
    p_std = np.std(p_stack, axis=1)

    # Disagreement features
    d_range = p_max - p_min
    d_pairwise_12 = np.abs(p1 - p2)
    d_pairwise_13 = np.abs(p1 - p3)
    d_pairwise_23 = np.abs(p2 - p3)

    # Prediction entropy across binary outcomes using mean probability
    eps = 1e-7
    p_safe = np.clip(p_mean, eps, 1.0 - eps)
    entropy = -(p_safe * np.log2(p_safe) + (1.0 - p_safe) * np.log2(1.0 - p_safe))

    # Confidence gap: distance of the consensus from the uncertainty point 0.5
    confidence_gap = np.abs(p_mean - 0.5)

    # Full meta-feature matrix: shape (N, 12)
    meta_features = np.column_stack([
        p1, p2, p3,
        p_max, p_min, p_mean, p_std,
        d_range,
        d_pairwise_12, d_pairwise_13, d_pairwise_23,
        entropy,
        confidence_gap
    ])
    return meta_features.astype(np.float32)


class AdaptiveMetaLearner:
    """Consensus and disagreement meta-learner fitted strictly on out-of-fold predictions."""

    def __init__(self, model_type: str = "logistic_regression", C: float = 1.0, seed: int = 42):
        self.model_type = model_type
        self.seed = seed
        if model_type == "logistic_regression":
            self.model = LogisticRegression(C=C, max_iter=1000, random_state=seed, solver="lbfgs")
        else:
            self.model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05, random_state=seed, verbose=-1
            )
        self.is_fitted = False

    def fit(self, meta_features: np.ndarray, y_train: np.ndarray) -> "AdaptiveMetaLearner":
        self.model.fit(meta_features, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, meta_features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Meta-learner is not fitted.")
        return self.model.predict_proba(meta_features)[:, 1]

    def predict(self, meta_features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(meta_features)
        return (probs >= threshold).astype(int)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "AdaptiveMetaLearner":
        return joblib.load(filepath)
