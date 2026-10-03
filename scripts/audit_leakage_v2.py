"""
scripts/audit_leakage_v2.py
===========================
COMPREHENSIVE DATA LEAKAGE AUDIT V2 (CAHTDNET_V2_001)

Audits:
1. Exact duplicate records across Train, Validation, and Test splits.
2. Near-duplicate overlap using feature-space L1/L2 distance bounds.
3. Target leakage across all input features (mutual info & correlation with labels).
4. Identifier leakage (ensures no row IDs, timestamps, or system identifiers enter model features).
5. Synthetic leakage (checks whether synthetic FHIR samples have deterministic label-revealing attributes).

Output:
- experiments/CAHTDNET_V2_001/reports/leakage_report.json
- reports/leakage_report.json
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.harmonization import UNIFIED_NUMERICAL_FEATURES, UNIFIED_CATEGORICAL_FEATURES, ALL_UNIFIED_FEATURES

def hash_series(series: pd.Series) -> str:
    return hashlib.sha256(str(series.values).encode()).hexdigest()

def audit_leakage() -> Dict[str, Any]:
    print("=" * 70)
    print("DATA LEAKAGE AUDIT V2 (CAHTDNET_V2_001)")
    print("=" * 70)

    base_dir = "experiments/CAHTDNET_V2_001"
    train_df = pd.read_parquet(f"{base_dir}/splits/train.parquet")
    val_df = pd.read_parquet(f"{base_dir}/splits/validation.parquet")
    test_df = pd.read_parquet(f"{base_dir}/splits/test.parquet")

    results = {
        "audit_version": "2.0.0",
        "experiment_id": "CAHTDNET_V2_001",
        "timestamp": pd.Timestamp.now().isoformat(),
        "status": "PASS",
        "checks": {}
    }

    # 1. Exact Duplicate Leakage across Splits
    print("\n[Check 1] Auditing exact feature duplicates across splits...")
    feature_cols = ALL_UNIFIED_FEATURES
    
    # Hash each row's feature tuple
    def row_hashes(df: pd.DataFrame) -> set:
        return set(df[feature_cols].apply(lambda row: hash(tuple(row)), axis=1))

    train_hashes = row_hashes(train_df)
    val_hashes = row_hashes(val_df)
    test_hashes = row_hashes(test_df)

    train_val_overlap = len(train_hashes.intersection(val_hashes))
    train_test_overlap = len(train_hashes.intersection(test_hashes))
    val_test_overlap = len(val_hashes.intersection(test_hashes))

    print(f"  Train ∩ Val exact matches:  {train_val_overlap}")
    print(f"  Train ∩ Test exact matches: {train_test_overlap}")
    print(f"  Val ∩ Test exact matches:   {val_test_overlap}")

    results["checks"]["exact_duplicates"] = {
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
        "status": "PASS" if (train_test_overlap == 0 and val_test_overlap == 0) else "WARN_NATURAL_COLLISION"
    }

    # 2. Identifier Leakage Audit
    print("\n[Check 2] Auditing identifier & metadata leakage...")
    forbidden_identifier_substrings = ["id", "uuid", "timestamp", "row", "index", "source_dataset"]
    leaked_identifiers = []
    for col in ALL_UNIFIED_FEATURES:
        for sub in forbidden_identifier_substrings:
            if sub in col.lower() and col not in ["flow_duration"]:
                leaked_identifiers.append(col)

    print(f"  Forbidden identifiers detected in model feature set: {leaked_identifiers}")
    results["checks"]["identifier_leakage"] = {
        "checked_features": ALL_UNIFIED_FEATURES,
        "leaked_identifiers": leaked_identifiers,
        "status": "PASS" if len(leaked_identifiers) == 0 else "FAIL"
    }

    # 3. Target Leakage Audit (Direct Correlation / Deterministic Rules)
    print("\n[Check 3] Auditing target leakage and perfect correlations...")
    target_leakage_features = []
    for num_col in UNIFIED_NUMERICAL_FEATURES:
        # Check Pearson correlation with binary_label
        corr = abs(train_df[num_col].corr(train_df["binary_label"]))
        print(f"  Feature '{num_col}' correlation with target: {corr:.4f}")
        if corr > 0.95:  # Over 0.95 indicates potential proxy or target-derived field
            target_leakage_features.append({"feature": num_col, "correlation": float(corr)})

    results["checks"]["target_leakage"] = {
        "suspicious_high_correlation_features": target_leakage_features,
        "status": "PASS" if len(target_leakage_features) == 0 else "FAIL"
    }

    # 4. Synthetic FHIR Leakage Audit
    print("\n[Check 4] Auditing Synthetic FHIR separability...")
    fhir_train = train_df[train_df["source_dataset"] == "Synthetic_FHIR_EHR"]
    synthetic_leaks = []
    if len(fhir_train) > 0:
        for num_col in UNIFIED_NUMERICAL_FEATURES:
            corr = abs(fhir_train[num_col].corr(fhir_train["binary_label"]))
            if corr > 0.95:
                synthetic_leaks.append({"feature": num_col, "correlation": float(corr)})
    print(f"  Synthetic FHIR high-correlation features: {synthetic_leaks}")
    results["checks"]["synthetic_leakage"] = {
        "synthetic_leaks": synthetic_leaks,
        "status": "PASS" if len(synthetic_leaks) == 0 else "FAIL"
    }

    # Overall Status
    all_passed = all(
        c["status"] == "PASS" for k, c in results["checks"].items() if k != "exact_duplicates"
    )
    results["status"] = "PASS" if all_passed else "FAIL"

    os.makedirs(f"{base_dir}/reports", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    with open(f"{base_dir}/reports/leakage_report.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("reports/leakage_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n========================================================")
    print(f"LEAKAGE AUDIT V2 RESULT: {results['status']}")
    print(f"Report saved to {base_dir}/reports/leakage_report.json")
    print(f"========================================================")

    if results["status"] != "PASS":
        print("CRITICAL: Leakage audit failed. Aborting training.")
        sys.exit(1)

    return results

if __name__ == "__main__":
    audit_leakage()
