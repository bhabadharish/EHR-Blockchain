import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
scripts/fit_and_export_v2_features.py
=====================================
FITS FEATURE PIPELINE V2 ON TRAIN ONLY AND EXPORTS STANDARDIZED ARRAYS

Artifacts generated:
- experiments/CAHTDNET_V2_001/preprocessing/preprocessor.pkl
- experiments/CAHTDNET_V2_001/preprocessing/feature_schema.json
- experiments/CAHTDNET_V2_001/preprocessing/feature_order.json
- experiments/CAHTDNET_V2_001/features/feature_metadata.json
- experiments/CAHTDNET_V2_001/features/feature_statistics.csv
- experiments/CAHTDNET_V2_001/data/processed_arrays.npz
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.feature_pipeline_v2 import FeaturePipelineV2, ALL_V2_NUMERICAL_FEATURES, ALL_V2_CATEGORICAL_FEATURES

def main():
    print("=" * 70)
    print("CA-HTDNet V2: FITTING FEATURE PIPELINE STRICTLY ON TRAIN SPLIT")
    print("=" * 70)

    base_dir = "experiments/CAHTDNET_V2_001"
    os.makedirs(f"{base_dir}/preprocessing", exist_ok=True)
    os.makedirs(f"{base_dir}/features", exist_ok=True)
    os.makedirs(f"{base_dir}/data", exist_ok=True)
    os.makedirs("models/preprocessors", exist_ok=True)

    # 1. Load splits
    print("Loading Parquet splits...")
    train_df = pd.read_parquet(f"{base_dir}/splits/train.parquet")
    val_df = pd.read_parquet(f"{base_dir}/splits/validation.parquet")
    test_df = pd.read_parquet(f"{base_dir}/splits/test.parquet")

    # 2. Fit FeaturePipelineV2 strictly on Train
    print("Fitting FeaturePipelineV2 on Train split...")
    pipeline = FeaturePipelineV2()
    pipeline.fit(train_df)

    # Save Preprocessor
    preprocessor_path = f"{base_dir}/preprocessing/preprocessor.pkl"
    pipeline.save(preprocessor_path)
    pipeline.save("models/preprocessors/preprocessor.pkl")
    print(f"  Fitted preprocessor saved to {preprocessor_path}")

    # Metadata & Schema
    meta = pipeline.get_feature_metadata()
    with open(f"{base_dir}/features/feature_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    schema = {
        "experiment_id": "CAHTDNET_V2_001",
        "feature_version": "2.0.0",
        "numerical_features": meta["numerical_features"],
        "categorical_features": meta["categorical_features"],
        "num_dim": meta["num_dim"],
        "cat_dim": meta["cat_dim"],
        "cat_cardinalities": meta["cat_cardinalities"]
    }
    with open(f"{base_dir}/preprocessing/feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    with open(f"{base_dir}/preprocessing/feature_order.json", "w") as f:
        json.dump(meta["numerical_features"] + meta["categorical_features"], f, indent=2)

    # 3. Compute Feature Statistics Strictly on Train
    print("Generating feature statistics strictly from Train split...")
    stats_rows = []
    engineered_train = pipeline._engineer_features(train_df)
    for col in ALL_V2_NUMERICAL_FEATURES:
        series = engineered_train[col]
        stats_rows.append({
            "feature": col,
            "type": "numeric",
            "mean": float(series.mean()),
            "std": float(series.std()),
            "median": float(series.median()),
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75)),
            "min": float(series.min()),
            "max": float(series.max()),
            "missing_pct": float(series.isna().mean() * 100)
        })
    for col in ALL_V2_CATEGORICAL_FEATURES:
        series = train_df[col]
        stats_rows.append({
            "feature": col,
            "type": "categorical",
            "unique_count": int(series.nunique()),
            "top_category": str(series.mode()[0]),
            "missing_pct": float(series.isna().mean() * 100)
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(f"{base_dir}/features/feature_statistics.csv", index=False)
    print(f"  Feature statistics saved to {base_dir}/features/feature_statistics.csv")

    # 4. Transform all splits into numpy arrays
    print("Transforming Train, Validation, and Locked Test splits...")
    X_train_num, X_train_cat = pipeline.transform(train_df)
    y_train = train_df["binary_label"].values.astype(np.int64)

    X_val_num, X_val_cat = pipeline.transform(val_df)
    y_val = val_df["binary_label"].values.astype(np.int64)

    X_test_num, X_test_cat = pipeline.transform(test_df)
    y_test = test_df["binary_label"].values.astype(np.int64)

    np.savez_compressed(
        f"{base_dir}/data/processed_arrays.npz",
        X_train_num=X_train_num, X_train_cat=X_train_cat, y_train=y_train,
        X_val_num=X_val_num, X_val_cat=X_val_cat, y_val=y_val,
        X_test_num=X_test_num, X_test_cat=X_test_cat, y_test=y_test
    )
    print(f"  Processed arrays saved to {base_dir}/data/processed_arrays.npz")
    print(f"  Train: X_num {X_train_num.shape}, X_cat {X_train_cat.shape}, y {y_train.shape}")
    print(f"  Val:   X_num {X_val_num.shape}, X_cat {X_val_cat.shape}, y {y_val.shape}")
    print(f"  Test:  X_num {X_test_num.shape}, X_cat {X_test_cat.shape}, y {y_test.shape}")

if __name__ == "__main__":
    main()
