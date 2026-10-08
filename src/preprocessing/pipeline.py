"""Strictly Training-Fitted Preprocessing Pipeline for HAB-IDS (Phase 8).

Implements leakage-free tabular preprocessing:
- Missing, infinite, and malformed token sanitization based solely on training statistics
- Coerces mixed-type columns (e.g. hex '0x00', float '0.0', int '0') to uniform numeric representation,
  eliminating PCAP formatting leakage
- Discards zero-variance constant features identified strictly on train fold
- Training-fitted categorical encoding (with unknown category handling)
- Zero global scaling on test data
"""

import os
import json
import joblib
from typing import Dict, List, Optional, Any, Set
import numpy as np
import pandas as pd


def robust_numeric_parse(val: Any) -> float:
    """Parse mixed strings, hex strings, and numeric tokens safely to float."""
    if val is None or pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null", ""):
        return np.nan
    if s.startswith(("0x", "0X")):
        try:
            return float(int(s, 16))
        except (ValueError, TypeError):
            return np.nan
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


class LeakageFreePreprocessor:
    """Preprocessor fitted strictly on training data with zero lookahead or test leakage."""

    def __init__(self, categorical_cols: Optional[List[str]] = None):
        self.explicit_categorical_cols = set(categorical_cols or [])
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.constant_cols_: Set[str] = set()
        self.retained_features_: List[str] = []
        self.train_medians_: Dict[str, float] = {}
        self.category_maps_: Dict[str, Dict[str, int]] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "LeakageFreePreprocessor":
        """Fit preprocessor strictly on training data."""
        self.numeric_cols_ = []
        self.categorical_cols_ = []
        self.train_medians_ = {}
        self.category_maps_ = {}
        self.constant_cols_ = set()

        # Step 1: Distinguish true categoricals from mixed-type numeric columns
        for col in df.columns:
            if col in self.explicit_categorical_cols:
                self.categorical_cols_.append(col)
                continue

            # Check if column is already numeric or parseable as numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_cols_.append(col)
            else:
                # Test a sample to see if it is primarily numeric or text
                sample = df[col].dropna().head(100)
                parsed_sample = [robust_numeric_parse(v) for v in sample]
                non_nan_count = sum(1 for v in parsed_sample if not np.isnan(v))
                if non_nan_count > len(parsed_sample) * 0.7:
                    self.numeric_cols_.append(col)
                else:
                    self.categorical_cols_.append(col)

        # Step 2: Fit numeric features (medians and check variance)
        for col in self.numeric_cols_:
            # Clean and parse
            if pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].astype(float).replace([np.inf, -np.inf], np.nan)
            else:
                series = df[col].apply(robust_numeric_parse).replace([np.inf, -np.inf], np.nan)

            median_val = float(series.median(skipna=True))
            if np.isnan(median_val):
                median_val = 0.0
            self.train_medians_[col] = median_val

            # Check for constant zero-variance columns strictly on training split
            series_filled = series.fillna(median_val)
            if series_filled.nunique() <= 1 or series_filled.var() < 1e-7:
                self.constant_cols_.add(col)

        # Step 3: Fit categorical features
        for col in self.categorical_cols_:
            series = df[col].astype(str).fillna("missing")
            unique_vals = sorted(series.unique().tolist())
            if len(unique_vals) <= 1:
                self.constant_cols_.add(col)
            else:
                self.category_maps_[col] = {val: idx for idx, val in enumerate(unique_vals)}

        # Retained non-constant features in deterministic order
        all_cols = list(df.columns)
        self.retained_features_ = [c for c in all_cols if c not in self.constant_cols_]
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform dataset using only frozen training parameters."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted on training data before transforming.")

        data_dict = {}

        # Numeric transformations
        for col in self.numeric_cols_:
            if col in self.constant_cols_:
                continue
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    s = df[col].astype(float).replace([np.inf, -np.inf], np.nan)
                else:
                    s = df[col].apply(robust_numeric_parse).replace([np.inf, -np.inf], np.nan)
                data_dict[col] = s.fillna(self.train_medians_[col]).to_numpy(dtype=np.float32)
            else:
                data_dict[col] = np.full(len(df), self.train_medians_[col], dtype=np.float32)

        # Categorical transformations
        for col in self.categorical_cols_:
            if col in self.constant_cols_:
                continue
            cat_map = self.category_maps_[col]
            if col in df.columns:
                s = df[col].astype(str)
                mapped = s.map(cat_map).fillna(-1.0).to_numpy(dtype=np.float32)
                data_dict[col] = mapped
            else:
                data_dict[col] = np.full(len(df), -1.0, dtype=np.float32)

        df_out = pd.DataFrame(data_dict, index=df.index)[self.retained_features_]
        return df_out.to_numpy(dtype=np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "LeakageFreePreprocessor":
        return joblib.load(filepath)
