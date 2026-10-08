import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""Pipeline Script: Leakage-Controlled Dataset Splitting (Phase 7).

Generates canonical, locked 70% Train / 15% Validation / 15% Test splits
with stratified class balance. Saves exact split files and manifest to data/splits/.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_and_save_dataset(
    df: pd.DataFrame,
    stratify_col: str,
    prefix: str,
    output_dir: str = "data/splits",
    random_seed: int = 42
) -> dict:
    """Split a dataframe into 70/15/15 stratified splits and save as parquets."""
    os.makedirs(output_dir, exist_ok=True)

    # First split: 70% train, 30% temp (val + test)
    df_train, df_temp = train_test_split(
        df,
        test_size=0.30,
        random_state=random_seed,
        stratify=df[stratify_col] if stratify_col in df.columns else None
    )

    # Second split: 15% val, 15% test (50% of temp)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        random_state=random_seed,
        stratify=df_temp[stratify_col] if stratify_col in df_temp.columns else None
    )

    train_path = os.path.join(output_dir, f"{prefix}_train.parquet")
    val_path = os.path.join(output_dir, f"{prefix}_val.parquet")
    test_path = os.path.join(output_dir, f"{prefix}_test.parquet")

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)

    return {
        "train_count": len(df_train),
        "val_count": len(df_val),
        "test_count": len(df_test),
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "train_distribution": {str(k): int(v) for k, v in df_train[stratify_col].value_counts().items()},
        "val_distribution": {str(k): int(v) for k, v in df_val[stratify_col].value_counts().items()},
        "test_distribution": {str(k): int(v) for k, v in df_test[stratify_col].value_counts().items()},
    }


def main():
    print("==================================================")
    print("SPLITTING: Generating Canonical Stratified Splits")
    print("==================================================")
    split_manifest = {
        "random_seed": 42,
        "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
        "datasets": {}
    }

    # 1. Edge-IIoT
    edge_pq = "data/processed/edge_iiot.parquet"
    if os.path.exists(edge_pq):
        print("Splitting Edge-IIoTset (157,800 rows)...")
        df_edge = pd.read_parquet(edge_pq)
        info_edge = split_and_save_dataset(df_edge, stratify_col="Attack_type", prefix="edge")
        split_manifest["datasets"]["edge_iiot"] = info_edge

    # 2. Synthetic FHIR
    fhir_pq = "data/processed/fhir_security.parquet"
    if os.path.exists(fhir_pq):
        print("Splitting Synthetic FHIR Security Events (50,000 rows)...")
        df_fhir = pd.read_parquet(fhir_pq)
        info_fhir = split_and_save_dataset(df_fhir, stratify_col="attack_category", prefix="fhir")
        split_manifest["datasets"]["fhir"] = info_fhir

    # 3. CICIoT2023 (Already partitioned train/val/test in data/processed/)
    ciciot_train_pq = "data/processed/ciciot_train.parquet"
    ciciot_val_pq = "data/processed/ciciot_val.parquet"
    ciciot_test_pq = "data/processed/ciciot_test.parquet"
    if os.path.exists(ciciot_train_pq):
        print("Locking CICIoT2023 canonical splits...")
        df_tr = pd.read_parquet(ciciot_train_pq)
        df_va = pd.read_parquet(ciciot_val_pq)
        df_te = pd.read_parquet(ciciot_test_pq)

        df_tr.to_parquet("data/splits/ciciot_train.parquet", index=False)
        df_va.to_parquet("data/splits/ciciot_val.parquet", index=False)
        df_te.to_parquet("data/splits/ciciot_test.parquet", index=False)

        split_manifest["datasets"]["ciciot2023"] = {
            "train_count": len(df_tr),
            "val_count": len(df_va),
            "test_count": len(df_te),
            "train_path": "data/splits/ciciot_train.parquet",
            "val_path": "data/splits/ciciot_val.parquet",
            "test_path": "data/splits/ciciot_test.parquet",
        }

    manifest_path = "data/splits/split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(split_manifest, f, indent=2)

    print(f"Split manifest successfully written to: {manifest_path}")


if __name__ == "__main__":
    main()
