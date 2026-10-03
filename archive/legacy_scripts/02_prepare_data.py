import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.harmonization import (
    map_ciciot2023_to_unified,
    map_edge_iiot_to_unified,
    map_synthetic_fhir_to_unified,
    UNIFIED_NUMERICAL_FEATURES,
    UNIFIED_CATEGORICAL_FEATURES,
    ALL_UNIFIED_FEATURES
)
from src.data.cleaning import UnifiedSecurityPreprocessor
from src.data.splitting import create_leakage_safe_splits

def main():
    print("=" * 70)
    print("STAGE 2: DATA HARMONIZATION, PREPROCESSING & LEAKAGE-SAFE SPLITTING")
    print("=" * 70)

    os.makedirs("data/splits", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models/preprocessors", exist_ok=True)
    os.makedirs("data/metadata", exist_ok=True)

    # 1. Load CICIoT2023 sample (representative 120,000 samples for balanced multi-source training on Apple M2)
    print("Loading and harmonizing CICIoT2023...")
    cic_raw = pd.read_csv("data/raw/cic_iot/train.csv", nrows=120000, low_memory=False)
    cic_harmonized = map_ciciot2023_to_unified(cic_raw)
    print(f"  CICIoT2023 harmonized: {len(cic_harmonized)} rows, {cic_harmonized['binary_label'].value_counts().to_dict()}")

    # 2. Load Edge-IIoTset sample (representative 40,000 samples)
    print("Loading and harmonizing Edge-IIoTset...")
    edge_raw = pd.read_csv("data/raw/edge_iiot/ML-EdgeIIoT-dataset.csv", nrows=40000, low_memory=False)
    edge_harmonized = map_edge_iiot_to_unified(edge_raw)
    print(f"  Edge-IIoTset harmonized: {len(edge_harmonized)} rows, {edge_harmonized['binary_label'].value_counts().to_dict()}")

    # 3. Load Synthetic FHIR (20,000 samples from medium set)
    print("Loading and harmonizing Synthetic FHIR Security dataset...")
    fhir_path = "data/raw/synthetic_fhir/fhir_security_med.parquet"
    if not os.path.exists(fhir_path):
        from src.data.synthetic_fhir_generator import generate_fhir_splits
        generate_fhir_splits()
    fhir_raw = pd.read_parquet(fhir_path)
    fhir_harmonized = map_synthetic_fhir_to_unified(fhir_raw)
    print(f"  Synthetic FHIR harmonized: {len(fhir_harmonized)} rows, {fhir_harmonized['binary_label'].value_counts().to_dict()}")

    # 4. Save independent cross-dataset benchmark evaluation sets
    print("\nSaving independent cross-dataset evaluation sets...")
    cic_harmonized.to_parquet("data/splits/eval_ciciot_pure.parquet", index=False)
    edge_harmonized.to_parquet("data/splits/eval_edge_iiot_pure.parquet", index=False)
    fhir_harmonized.to_parquet("data/splits/eval_synthetic_fhir_pure.parquet", index=False)

    # 5. Create Unified Multimodal Research Dataset
    print("\nCreating Unified Multimodal Research Dataset (Combined)...")
    combined_df = pd.concat([cic_harmonized, edge_harmonized, fhir_harmonized], ignore_index=True)
    # Shuffle
    combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Total Unified samples: {len(combined_df)} rows.")

    # 6. Leakage-safe 70 / 15 / 15 Split
    print("\nGenerating 70% Train, 15% Validation, 15% Test stratified splits...")
    train_df, val_df, test_df = create_leakage_safe_splits(combined_df, stratify_col="binary_label", seed=42)
    print(f"  Train: {len(train_df)} rows | Val: {len(val_df)} rows | Test: {len(test_df)} rows")

    train_df.to_parquet("data/splits/train.parquet", index=False)
    val_df.to_parquet("data/splits/validation.parquet", index=False)
    test_df.to_parquet("data/splits/test.parquet", index=False)

    # 7. Fit Preprocessor STRICTLY ON TRAIN ONLY
    print("\nFitting UnifiedSecurityPreprocessor strictly on Train partition...")
    preprocessor = UnifiedSecurityPreprocessor(scaler_type="robust")
    preprocessor.fit(train_df, UNIFIED_NUMERICAL_FEATURES, UNIFIED_CATEGORICAL_FEATURES)
    preprocessor_path = "models/preprocessors/preprocessor.pkl"
    preprocessor.save(preprocessor_path)
    print(f"  Fitted preprocessor saved to {preprocessor_path}")

    # 8. Transform and cache processed arrays
    print("Transforming partitions into memory-efficient numpy arrays...")
    X_train_num, X_train_cat = preprocessor.transform(train_df)
    y_train = train_df["binary_label"].values.astype(np.int64)

    X_val_num, X_val_cat = preprocessor.transform(val_df)
    y_val = val_df["binary_label"].values.astype(np.int64)

    X_test_num, X_test_cat = preprocessor.transform(test_df)
    y_test = test_df["binary_label"].values.astype(np.int64)

    np.savez_compressed(
        "data/processed/processed_arrays.npz",
        X_train_num=X_train_num, X_train_cat=X_train_cat, y_train=y_train,
        X_val_num=X_val_num, X_val_cat=X_val_cat, y_val=y_val,
        X_test_num=X_test_num, X_test_cat=X_test_cat, y_test=y_test
    )

    manifest = {
        "unified_features": {
            "numerical": UNIFIED_NUMERICAL_FEATURES,
            "categorical": UNIFIED_CATEGORICAL_FEATURES,
            "total_count": len(ALL_UNIFIED_FEATURES)
        },
        "splits": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "sources": {
                "ciciot2023": len(cic_harmonized),
                "edge_iiot": len(edge_harmonized),
                "synthetic_fhir": len(fhir_harmonized)
            }
        },
        "preprocessor": {
            "scaler": "RobustScaler",
            "cat_encoder": "OrdinalEncoder",
            "fit_partition": "TRAIN_ONLY"
        }
    }
    with open("data/metadata/split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nData preparation complete! Manifest saved to data/metadata/split_manifest.json.")

if __name__ == "__main__":
    main()
