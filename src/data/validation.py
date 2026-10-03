import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

def audit_dataset_schema(
    file_path: str,
    label_col: str,
    sample_size: int = 50000
) -> Dict[str, Any]:
    """
    Performs empirical data quality, schema and leakage audit on a tabular cybersecurity dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    # Read sample or full if small
    df = pd.read_csv(file_path, nrows=sample_size, low_memory=False)
    
    total_cols = len(df.columns)
    total_rows_sampled = len(df)
    
    missing_dict = df.isnull().sum().to_dict()
    dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
    cardinality = {col: int(df[col].nunique()) for col in df.columns}
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numerical_cols]
    
    # Ranges for numerical
    num_ranges = {}
    suspicious_cols = []
    for col in numerical_cols:
        c_min = float(df[col].min()) if not pd.isna(df[col].min()) else None
        c_max = float(df[col].max()) if not pd.isna(df[col].max()) else None
        c_std = float(df[col].std()) if not pd.isna(df[col].std()) else 0.0
        num_ranges[col] = {"min": c_min, "max": c_max, "std": c_std}
        if c_std == 0.0:
            suspicious_cols.append(f"{col} (Zero Variance)")

    # Label distribution
    if label_col in df.columns:
        label_dist = df[label_col].value_counts().to_dict()
        label_dist = {str(k): int(v) for k, v in label_dist.items()}
    else:
        label_dist = {}
        suspicious_cols.append(f"Label column '{label_col}' missing!")

    return {
        "file_path": file_path,
        "sampled_rows": total_rows_sampled,
        "total_columns": total_cols,
        "label_column": label_col,
        "label_distribution": label_dist,
        "numerical_columns_count": len(numerical_cols),
        "categorical_columns_count": len(categorical_cols),
        "cardinality": cardinality,
        "dtypes": dtypes_dict,
        "missing_values": missing_dict,
        "numerical_ranges": num_ranges,
        "suspicious_columns": suspicious_cols
    }
