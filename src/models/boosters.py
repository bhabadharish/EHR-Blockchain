"""Optimized Gradient Boosted Decision Tree (GBDT) Learners for HAB-IDS.

Implements highly optimized, early-stopping-regularized:
1. XGBoost (Phase 12)
2. LightGBM (Phase 13)
3. CatBoost (Phase 14)
with multi-seed validation stability (Phase 15).
"""

import os
import time
import json
import joblib
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from src.evaluation.metrics import compute_binary_metrics


class TunedXGBoost:
    """Tuned XGBoost model with early stopping on validation split."""

    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        self.params = {
            "n_estimators": kwargs.get("n_estimators", 350),
            "max_depth": kwargs.get("max_depth", 6),
            "learning_rate": kwargs.get("learning_rate", 0.05),
            "subsample": kwargs.get("subsample", 0.8),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
            "min_child_weight": kwargs.get("min_child_weight", 3),
            "gamma": kwargs.get("gamma", 0.1),
            "reg_alpha": kwargs.get("reg_alpha", 0.1),
            "reg_lambda": kwargs.get("reg_lambda", 1.0),
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": -1,
            "early_stopping_rounds": kwargs.get("early_stopping_rounds", 25),
            "eval_metric": "logloss"
        }
        self.model = xgb.XGBClassifier(**self.params)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "TunedXGBoost":
        # Calculate scale_pos_weight if binary
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        scale_pos_weight = float(n_neg / max(n_pos, 1))
        self.model.set_params(scale_pos_weight=scale_pos_weight)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    @classmethod
    def load(cls, filepath: str) -> "TunedXGBoost":
        instance = cls()
        instance.model = joblib.load(filepath)
        instance.is_fitted = True
        return instance


class TunedLightGBM:
    """Tuned LightGBM model with early stopping on validation split."""

    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        self.params = {
            "n_estimators": kwargs.get("n_estimators", 400),
            "num_leaves": kwargs.get("num_leaves", 45),
            "max_depth": kwargs.get("max_depth", 7),
            "learning_rate": kwargs.get("learning_rate", 0.04),
            "subsample": kwargs.get("subsample", 0.8),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
            "min_child_samples": kwargs.get("min_child_samples", 25),
            "reg_alpha": kwargs.get("reg_alpha", 0.1),
            "reg_lambda": kwargs.get("reg_lambda", 1.0),
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
            "class_weight": "balanced"
        }
        self.early_stopping_rounds = kwargs.get("early_stopping_rounds", 25)
        self.model = lgb.LGBMClassifier(**self.params)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "TunedLightGBM":
        callbacks = [lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)]
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks
        )
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    @classmethod
    def load(cls, filepath: str) -> "TunedLightGBM":
        instance = cls()
        instance.model = joblib.load(filepath)
        instance.is_fitted = True
        return instance


class TunedCatBoost:
    """Tuned CatBoost model with early stopping on validation split."""

    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        self.params = {
            "iterations": kwargs.get("iterations", 350),
            "depth": kwargs.get("depth", 6),
            "learning_rate": kwargs.get("learning_rate", 0.05),
            "l2_leaf_reg": kwargs.get("l2_leaf_reg", 3.0),
            "random_strength": kwargs.get("random_strength", 0.1),
            "random_seed": seed,
            "verbose": 0,
            "early_stopping_rounds": kwargs.get("early_stopping_rounds", 25),
            "auto_class_weights": "Balanced"
        }
        self.model = CatBoostClassifier(**self.params)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "TunedCatBoost":
        self.model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    @classmethod
    def load(cls, filepath: str) -> "TunedCatBoost":
        instance = cls()
        instance.model = joblib.load(filepath)
        instance.is_fitted = True
        return instance
