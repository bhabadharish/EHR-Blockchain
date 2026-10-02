"""
backend/models/baselines.py
Classical machine learning baselines (Logistic Regression, Random Forest, LightGBM/XGBoost)
following Phase 15 protocol for comparative benchmarking.
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

def get_baseline_model(model_name: str, seed: int = 42):
    """Factory function returning configured baseline classifiers."""
    m = model_name.upper()
    if m in ("LR", "LOGISTIC_REGRESSION", "LOGISTICREGRESSION"):
        return LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=seed,
            solver="lbfgs"
        )
    elif m in ("RF", "RANDOM_FOREST", "RANDOMFOREST"):
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            random_state=seed,
            n_jobs=1
        )
    elif m in ("LGBM", "LIGHTGBM"):
        return lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            n_jobs=1,
            verbose=-1
        )
    elif m in ("XGB", "XGBOOST"):
        return xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            random_state=seed,
            n_jobs=2,
            eval_metric="mlogloss"
        )
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")
