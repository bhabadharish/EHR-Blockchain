import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder

class UnifiedSecurityPreprocessor:
    """
    Leakage-safe Preprocessor fitted strictly on Train partition.
    Handles numerical scaling and categorical encoding across network and FHIR context.
    """
    def __init__(self, scaler_type: str = "robust"):
        self.scaler_type = scaler_type
        self.scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, numerical_cols: List[str], categorical_cols: List[str]):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        
        # Fit numerical
        X_num = df[self.numerical_cols].values
        self.scaler.fit(X_num)
        
        # Fit categorical
        X_cat = df[self.categorical_cols].astype(str).values
        self.cat_encoder.fit(X_cat)
        
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted! Call fit() on Train data first.")
        
        X_num = self.scaler.transform(df[self.numerical_cols].values)
        X_cat = self.cat_encoder.transform(df[self.categorical_cols].astype(str).values)
        
        # Replace unknown categories (-1) with 0
        X_cat = np.where(X_cat == -1, 0, X_cat)
        
        return X_num.astype(np.float32), X_cat.astype(np.int64)

    def fit_transform(self, df: pd.DataFrame, numerical_cols: List[str], categorical_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(df, numerical_cols, categorical_cols)
        return self.transform(df)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "UnifiedSecurityPreprocessor":
        with open(filepath, "rb") as f:
            return pickle.load(f)
