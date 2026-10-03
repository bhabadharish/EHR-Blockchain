import os
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import StratifiedShuffleSplit

def create_leakage_safe_splits(
    df: pd.DataFrame,
    stratify_col: str = "binary_label",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates stratified 70% Train, 15% Validation, and 15% Test splits.
    Guarantees zero overlap across partitions.
    """
    assert np.isclose(train_size + val_size + test_size, 1.0)
    
    # 1. Split Train vs (Val + Test)
    split1 = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, temp_idx = next(split1.split(df, df[stratify_col]))
    
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)
    
    # 2. Split Val vs Test from temp (50/50 of the remaining 30% = 15% / 15%)
    val_ratio_in_temp = val_size / (val_size + test_size)
    split2 = StratifiedShuffleSplit(n_splits=1, train_size=val_ratio_in_temp, random_state=seed)
    val_idx, test_idx = next(split2.split(temp_df, temp_df[stratify_col]))
    
    val_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df
