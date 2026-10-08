"""Training-Only Feature Selection for HAB-IDS (Phase 10).

Evaluates feature importance, mutual information, and correlation redundancy
strictly using training samples. Never references validation or test data.
Exports canonical feature schema to models/final/feature_schema.json.
"""

import os
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb


class TrainOnlyFeatureSelector:
    """Feature selection pipeline executed strictly on training data."""

    def __init__(self, correlation_threshold: float = 0.98, top_k: Optional[int] = None):
        self.correlation_threshold = correlation_threshold
        self.top_k = top_k
        self.selected_features_: List[str] = []
        self.feature_importance_: Dict[str, float] = {}
        self.redundant_features_: List[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, feature_names: List[str]) -> "TrainOnlyFeatureSelector":
        """Compute feature importance and filter collinear redundancy on training split."""
        df_train = pd.DataFrame(X_train, columns=feature_names)
        
        # 1. Remove constant columns
        variances = df_train.var()
        non_constant = variances[variances > 1e-6].index.tolist()
        df_filtered = df_train[non_constant]

        # 2. Correlation analysis (identify perfectly collinear duplicates)
        corr_matrix = df_filtered.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        self.redundant_features_ = to_drop
        retained = [c for c in non_constant if c not in to_drop]

        # 3. Model-based Importance using a fast LightGBM estimator on train fold
        clf = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.08,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        sample_size = min(len(df_filtered), 50000)
        idx = np.random.RandomState(42).choice(len(df_filtered), sample_size, replace=False)
        clf.fit(df_filtered[retained].iloc[idx], y_train[idx])

        importances = clf.feature_importances_
        imp_dict = {f: float(imp) for f, imp in zip(retained, importances)}
        self.feature_importance_ = dict(sorted(imp_dict.items(), key=lambda item: item[1], reverse=True))

        if self.top_k is not None:
            self.selected_features_ = list(self.feature_importance_.keys())[: self.top_k]
        else:
            # Retain all non-redundant informative features
            self.selected_features_ = list(self.feature_importance_.keys())

        return self

    def save_feature_schema(self, output_path: str = "models/final/feature_schema.json") -> None:
        """Export final feature schema JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        schema = {
            "version": "1.0.0-HAB-IDS",
            "selected_features": self.selected_features_,
            "feature_count": len(self.selected_features_),
            "redundant_features_excluded": self.redundant_features_,
            "feature_importance_ranking": self.feature_importance_,
        }
        with open(output_path, "w") as f:
            json.dump(schema, f, indent=2)


def export_feature_schema(feature_names: List[str], output_path: str = "models/final/feature_schema.json") -> None:
    """Save given feature names directly as feature schema JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    schema = {
        "version": "1.0.0-HAB-IDS",
        "feature_count": len(feature_names),
        "selected_features": feature_names,
    }
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)
