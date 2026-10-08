import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
scripts/create_canonical_splits.py
==================================
CANONICAL DATA SPLITTING, ID LOCKING, FEATURE PREPROCESSING & STATISTICS

Tasks:
1. Lock 70/15/15 partitions and save sample index IDs with SHA-256 hashes.
2. Fit UnifiedSecurityPreprocessor STRICTLY on Train data only.
3. Persist preprocessor.pkl, feature_schema.json, feature_order.json.
4. Generate feature_statistics.csv strictly from Train partition.
5. Generate separate train, validation, and test class distribution files.
6. Export preprocessed arrays into experiments/data/ and data/processed/.
"""

import os
import sys
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.harmonization import UNIFIED_NUMERICAL_FEATURES, UNIFIED_CATEGORICAL_FEATURES
from src.data.cleaning import UnifiedSecurityPreprocessor

def compute_file_hash(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("=" * 70)
    print("STAGE 2: CANONICAL SPLIT LOCKING & DETERMINISTIC PREPROCESSING")
    print("=" * 70)

    os.makedirs("experiments/splits", exist_ok=True)
    os.makedirs("splits", exist_ok=True)
    os.makedirs("experiments/data", exist_ok=True)
    os.makedirs("experiments/preprocessors", exist_ok=True)
    os.makedirs("models/preprocessors", exist_ok=True)
    os.makedirs("experiments/metrics", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Load Parquet Partitions
    train_path = "data/splits/train.parquet"
    val_path = "data/splits/validation.parquet"
    test_path = "data/splits/test.parquet"

    print("Loading canonical parquet splits...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    print(f"  Train samples: {len(train_df)} | Val samples: {len(val_df)} | Test samples: {len(test_df)}")

    # 2. Persist Split IDs with SHA-256
    print("\n--- Generating and Locking Split IDs ---")
    train_ids = pd.DataFrame({"sample_id": [f"TRAIN_{i:06d}" for i in range(len(train_df))]})
    val_ids = pd.DataFrame({"sample_id": [f"VAL_{i:06d}" for i in range(len(val_df))]})
    test_ids = pd.DataFrame({"sample_id": [f"TEST_{i:06d}" for i in range(len(test_df))]})

    for path in ["splits/train_ids.csv", "experiments/splits/train_ids.csv"]:
        train_ids.to_csv(path, index=False)
    for path in ["splits/validation_ids.csv", "experiments/splits/validation_ids.csv"]:
        val_ids.to_csv(path, index=False)
    for path in ["splits/test_ids.csv", "experiments/splits/test_ids.csv"]:
        test_ids.to_csv(path, index=False)

    split_hashes = {
        "train_parquet_hash": compute_file_hash(train_path),
        "val_parquet_hash": compute_file_hash(val_path),
        "test_parquet_hash": compute_file_hash(test_path),
        "train_ids_hash": compute_file_hash("splits/train_ids.csv"),
        "val_ids_hash": compute_file_hash("splits/validation_ids.csv"),
        "test_ids_hash": compute_file_hash("splits/test_ids.csv"),
        "row_counts": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
            "total": len(train_df) + len(val_df) + len(test_df)
        }
    }
    with open("experiments/splits/split_lock.json", "w") as f:
        json.dump(split_hashes, f, indent=2)
    print(f"  Split lock saved to experiments/splits/split_lock.json")

    # 3. Fit Preprocessor STRICTLY ON TRAIN ONLY
    print("\n--- Fitting UnifiedSecurityPreprocessor Strictly on Train ---")
    preprocessor = UnifiedSecurityPreprocessor(scaler_type="robust")
    preprocessor.fit(train_df, UNIFIED_NUMERICAL_FEATURES, UNIFIED_CATEGORICAL_FEATURES)

    # Save Preprocessor Artifacts
    for p_path in ["models/preprocessors/preprocessor.pkl", "experiments/preprocessors/preprocessor.pkl"]:
        preprocessor.save(p_path)

    feature_schema = {
        "numerical_features": UNIFIED_NUMERICAL_FEATURES,
        "categorical_features": UNIFIED_CATEGORICAL_FEATURES,
        "num_numerical": len(UNIFIED_NUMERICAL_FEATURES),
        "num_categorical": len(UNIFIED_CATEGORICAL_FEATURES),
        "total_features": len(UNIFIED_NUMERICAL_FEATURES) + len(UNIFIED_CATEGORICAL_FEATURES),
        "scaling": "RobustScaler (IQR based)",
        "categorical_encoding": "OrdinalEncoder (with 0-imputed unknown values)",
        "fit_partition": "TRAIN_ONLY"
    }
    for s_path in ["models/preprocessors/feature_schema.json", "experiments/preprocessors/feature_schema.json"]:
        with open(s_path, "w") as f:
            json.dump(feature_schema, f, indent=2)

    feature_order = {
        "feature_order": UNIFIED_NUMERICAL_FEATURES + UNIFIED_CATEGORICAL_FEATURES
    }
    for o_path in ["models/preprocessors/feature_order.json", "experiments/preprocessors/feature_order.json"]:
        with open(o_path, "w") as f:
            json.dump(feature_order, f, indent=2)

    # 4. Transform Partitions and Save Numpy Arrays
    print("\n--- Transforming Partitions into Standardized Arrays ---")
    X_train_num, X_train_cat = preprocessor.transform(train_df)
    y_train = train_df["binary_label"].values.astype(np.int64)

    X_val_num, X_val_cat = preprocessor.transform(val_df)
    y_val = val_df["binary_label"].values.astype(np.int64)

    X_test_num, X_test_cat = preprocessor.transform(test_df)
    y_test = test_df["binary_label"].values.astype(np.int64)

    for arr_path in ["data/processed/processed_arrays.npz", "experiments/data/processed_arrays.npz"]:
        np.savez_compressed(
            arr_path,
            X_train_num=X_train_num, X_train_cat=X_train_cat, y_train=y_train,
            X_val_num=X_val_num, X_val_cat=X_val_cat, y_val=y_val,
            X_test_num=X_test_num, X_test_cat=X_test_cat, y_test=y_test
        )
    print("  Processed arrays saved to data/processed/ and experiments/data/")

    # 5. Feature Statistics (STRICTLY FROM TRAIN PARTITION)
    print("\n--- Computing Feature Statistics Strictly on Training Data ---")
    num_stats = train_df[UNIFIED_NUMERICAL_FEATURES].describe().T
    num_stats = num_stats.reset_index().rename(columns={"index": "feature"})
    num_stats["type"] = "numerical"

    cat_stats = []
    for cat_col in UNIFIED_CATEGORICAL_FEATURES:
        vc = train_df[cat_col].value_counts()
        cat_stats.append({
            "feature": cat_col,
            "type": "categorical",
            "count": len(train_df[cat_col]),
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "25%": np.nan,
            "50%": np.nan,
            "75%": np.nan,
            "max": np.nan,
            "unique_values": len(vc),
            "top_category": vc.index[0] if len(vc) > 0 else "None"
        })
    df_cat_stats = pd.DataFrame(cat_stats)
    combined_stats = pd.concat([num_stats, df_cat_stats], ignore_index=True)

    for f_stat_path in ["results/feature_statistics.csv", "experiments/metrics/feature_statistics.csv"]:
        combined_stats.to_csv(f_stat_path, index=False)
    print(f"  Feature statistics (train only) saved to results/feature_statistics.csv")

    # 6. Class Distributions (Separate for train, validation, test)
    print("\n--- Generating Class Distributions ---")
    def get_class_dist_df(df: pd.DataFrame, partition_name: str) -> pd.DataFrame:
        counts = df["binary_label"].value_counts()
        total = len(df)
        return pd.DataFrame([
            {"partition": partition_name, "class": 0, "class_name": "Normal", "count": int(counts.get(0, 0)), "percentage": round(float(counts.get(0, 0) / total * 100), 3)},
            {"partition": partition_name, "class": 1, "class_name": "Attack", "count": int(counts.get(1, 0)), "percentage": round(float(counts.get(1, 0) / total * 100), 3)},
            {"partition": partition_name, "class": -1, "class_name": "Total", "count": total, "percentage": 100.0}
        ])

    train_dist = get_class_dist_df(train_df, "Train")
    val_dist = get_class_dist_df(val_df, "Validation")
    test_dist = get_class_dist_df(test_df, "Test")

    for path, df_dist in [
        ("results/train_class_distribution.csv", train_dist),
        ("results/validation_class_distribution.csv", val_dist),
        ("results/test_class_distribution.csv", test_dist),
        ("experiments/metrics/train_class_distribution.csv", train_dist),
        ("experiments/metrics/validation_class_distribution.csv", val_dist),
        ("experiments/metrics/test_class_distribution.csv", test_dist)
    ]:
        df_dist.to_csv(path, index=False)

    # Also unified class distribution summary for dashboard
    unified_dist = pd.concat([train_dist.iloc[:2], val_dist.iloc[:2], test_dist.iloc[:2]], ignore_index=True)
    unified_dist.to_csv("results/class_distribution.csv", index=False)
    unified_dist.to_csv("experiments/metrics/class_distribution.csv", index=False)

    print("  Class distributions saved for train, val, and test splits.")
    print("\n" + "=" * 70)
    print("CANONICAL SPLIT & PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
