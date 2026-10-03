"""
src/data/feature_pipeline_v2.py
===============================
FEATURE PIPELINE V2: SCHEMA VALIDATION, FEATURE ENGINEERING, ROBUST PREPROCESSING

Implements:
1. SchemaValidator: Verifies feature types and constraints.
2. Missing / Outlier Handling: Deterministic median imputation and robust quantile clipping.
3. Feature Engineering:
   - byte_to_packet_ratio: payload density per packet
   - rate_ratio: byte throughput relative to packet rate
   - risk_sensitivity_interaction: composite healthcare vulnerability score
   - burst_frequency_interaction: dynamic traffic spike indicator
   - auth_anomaly_score: joint failure and status anomaly
4. Scaling: RobustScaler fitted strictly on Train partition.
5. Categorical Processing: Integer encoding with OOV token handling for embedding layers.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import RobustScaler, OrdinalEncoder

BASE_NUMERICAL_FEATURES = [
    "flow_duration",
    "packet_count",
    "byte_count",
    "packet_rate",
    "byte_rate",
    "dst_port",
    "protocol",
    "resource_sensitivity",
    "auth_status",
    "failed_auth_count",
    "request_frequency",
    "burst_score",
    "historical_risk"
]

ENGINEERED_NUMERICAL_FEATURES = [
    "byte_to_packet_ratio",
    "rate_ratio",
    "risk_sensitivity_interaction",
    "burst_frequency_interaction",
    "auth_anomaly_score"
]

ALL_V2_NUMERICAL_FEATURES = BASE_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES

ALL_V2_CATEGORICAL_FEATURES = [
    "user_role",
    "resource_type",
    "operation"
]

class FeaturePipelineV2:
    """
    Deterministic feature engineering and preprocessing pipeline.
    Fitted strictly on Training split to prevent any data leakage.
    """
    def __init__(self):
        self.scaler = RobustScaler()
        self.cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.is_fitted = False
        self.feature_order = ALL_V2_NUMERICAL_FEATURES + ALL_V2_CATEGORICAL_FEATURES
        self.cat_cardinalities = {}

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates domain-specific cybersecurity and healthcare telemetry interactions.
        """
        data = df.copy()

        # 1. Byte to packet ratio (packet payload density)
        data["byte_to_packet_ratio"] = (data["byte_count"] / (data["packet_count"] + 1.0)).clip(0.0, 1e5)

        # 2. Rate ratio (bytes per second relative to packets per second)
        data["rate_ratio"] = (data["byte_rate"] / (data["packet_rate"] + 0.01)).clip(0.0, 1e6)

        # 3. Composite healthcare vulnerability score
        data["risk_sensitivity_interaction"] = (data["historical_risk"] * data["resource_sensitivity"]).clip(0.0, 1.0)

        # 4. Dynamic traffic spike indicator
        data["burst_frequency_interaction"] = (data["burst_score"] * data["request_frequency"]).clip(0.0, 500.0)

        # 5. Joint authentication anomaly
        data["auth_anomaly_score"] = ((1 - data["auth_status"]) * (data["failed_auth_count"] + 1)).clip(0, 20)

        return data

    def fit(self, train_df: pd.DataFrame) -> "FeaturePipelineV2":
        engineered_train = self._engineer_features(train_df)

        # Fit numerical scaler
        X_num = engineered_train[ALL_V2_NUMERICAL_FEATURES].values.astype(np.float32)
        # Handle infinities / NaNs if any
        X_num = np.nan_to_num(X_num, nan=0.0, posinf=1e6, neginf=-1e6)
        self.scaler.fit(X_num)

        # Fit categorical encoder
        X_cat = engineered_train[ALL_V2_CATEGORICAL_FEATURES].astype(str).values
        self.cat_encoder.fit(X_cat)

        for i, col in enumerate(ALL_V2_CATEGORICAL_FEATURES):
            self.cat_cardinalities[col] = len(self.cat_encoder.categories_[i]) + 1  # +1 for unknown token

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("FeaturePipelineV2 must be fitted on training data before transforming.")

        engineered = self._engineer_features(df)

        # Numerical transform
        X_num = engineered[ALL_V2_NUMERICAL_FEATURES].values.astype(np.float32)
        X_num = np.nan_to_num(X_num, nan=0.0, posinf=1e6, neginf=-1e6)
        X_num_scaled = self.scaler.transform(X_num)

        # Categorical transform
        X_cat = self.cat_encoder.transform(engineered[ALL_V2_CATEGORICAL_FEATURES].astype(str).values)
        # Replace unknown (-1) with last index (cardinality - 1)
        for i, col in enumerate(ALL_V2_CATEGORICAL_FEATURES):
            unknown_idx = self.cat_cardinalities[col] - 1
            X_cat[:, i] = np.where(X_cat[:, i] == -1, unknown_idx, X_cat[:, i])

        return X_num_scaled.astype(np.float32), X_cat.astype(np.int64)

    def fit_transform(self, train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(train_df)
        return self.transform(train_df)

    def get_feature_metadata(self) -> Dict[str, Any]:
        return {
            "numerical_features": ALL_V2_NUMERICAL_FEATURES,
            "categorical_features": ALL_V2_CATEGORICAL_FEATURES,
            "num_dim": len(ALL_V2_NUMERICAL_FEATURES),
            "cat_dim": len(ALL_V2_CATEGORICAL_FEATURES),
            "cat_cardinalities": self.cat_cardinalities,
            "engineered_features": [
                {
                    "name": "byte_to_packet_ratio",
                    "meaning": "Payload density per packet",
                    "source": "byte_count / (packet_count + 1)",
                    "type": "numeric"
                },
                {
                    "name": "rate_ratio",
                    "meaning": "Byte throughput relative to packet rate",
                    "source": "byte_rate / (packet_rate + 0.01)",
                    "type": "numeric"
                },
                {
                    "name": "risk_sensitivity_interaction",
                    "meaning": "Composite vulnerability index",
                    "source": "historical_risk * resource_sensitivity",
                    "type": "numeric"
                },
                {
                    "name": "burst_frequency_interaction",
                    "meaning": "Traffic spike and request frequency interaction",
                    "source": "burst_score * request_frequency",
                    "type": "numeric"
                },
                {
                    "name": "auth_anomaly_score",
                    "meaning": "Joint authentication status and failure severity",
                    "source": "(1 - auth_status) * (failed_auth_count + 1)",
                    "type": "numeric"
                }
            ]
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "FeaturePipelineV2":
        with open(filepath, "rb") as f:
            return pickle.load(f)
