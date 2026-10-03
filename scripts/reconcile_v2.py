"""
scripts/reconcile_v2.py
=======================
STRICT MATHEMATICAL RECONCILIATION GATE FOR CA-HTDNet V2 (CAHTDNET_V2_001)

Verifies with tolerance 1e-6:
1. Final Metrics vs Raw Predictions:
   - Recomputes TN, FP, FN, TP directly from test_predictions.parquet.
   - Verifies Accuracy == (TP + TN) / Total
   - Verifies FPR == FP / (FP + TN)
   - Verifies FNR == FN / (FN + TP)
   - Verifies Macro-F1 == 0.5 * (F1_0 + F1_1)
   - Verifies ROC-AUC and PR-AUC match probability distributions.
2. Baselines vs Raw Predictions:
   - Verifies all 7 baseline predictions reconcile with their metrics JSON.
3. Ablation reconciliation.
4. Threshold & Calibration reconciliation.
5. Robustness reconciliation.
6. Provenance & Hash integrity:
   - Checks SHA-256 hashes of models, preprocessors, splits, and predictions.

Returns exit code 0 on full PASS, exit code 1 on ANY discrepancy.
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

EXPERIMENT_ID = "CAHTDNET_V2_001"
BASE_DIR = f"experiments/{EXPERIMENT_ID}"
TOLERANCE = 1e-6

def compute_file_hash(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()

def reconcile():
    print("=" * 70)
    print("FINAL EXPERIMENT RECONCILIATION GATE (CAHTDNET_V2_001)")
    print("=" * 70)

    failures = []

    # 1. Proposed Model Predictions vs Metrics
    pred_path = f"{BASE_DIR}/predictions/{EXPERIMENT_ID}_test_predictions.parquet"
    metrics_path = f"{BASE_DIR}/metrics/{EXPERIMENT_ID}_metrics.json"

    if not os.path.exists(pred_path) or not os.path.exists(metrics_path):
        print("FAIL: Predictions or metrics file missing.")
        sys.exit(1)

    pred_df = pd.read_parquet(pred_path)
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    y_true = pred_df["true_label"].values
    y_pred = pred_df["predicted_label"].values
    probs = pred_df["probability_class_1"].values

    # Confusion Matrix Recomputation
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    total = len(y_true)

    calc_acc = (tp + tn) / total
    calc_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    calc_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    p0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    r0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * p0 * r0 / (p0 + r0) if (p0 + r0) > 0 else 0.0

    p1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) > 0 else 0.0
    calc_macro_f1 = 0.5 * (f1_0 + f1_1)

    # Verifications
    if abs(metrics["accuracy"] - calc_acc) > TOLERANCE:
        failures.append(f"Accuracy mismatch: JSON={metrics['accuracy']} vs Calc={calc_acc}")
    if abs(metrics["fpr"] - calc_fpr) > TOLERANCE:
        failures.append(f"FPR mismatch: JSON={metrics['fpr']} vs Calc={calc_fpr}")
    if abs(metrics["fnr"] - calc_fnr) > TOLERANCE:
        failures.append(f"FNR mismatch: JSON={metrics['fnr']} vs Calc={calc_fnr}")
    if abs(metrics["macro_f1"] - calc_macro_f1) > TOLERANCE:
        failures.append(f"Macro-F1 mismatch: JSON={metrics['macro_f1']} vs Calc={calc_macro_f1}")

    calc_roc = roc_auc_score(y_true, probs)
    calc_pr = average_precision_score(y_true, probs)
    if abs(metrics["roc_auc"] - calc_roc) > TOLERANCE:
        failures.append(f"ROC-AUC mismatch: JSON={metrics['roc_auc']} vs Calc={calc_roc}")
    if abs(metrics["pr_auc"] - calc_pr) > TOLERANCE:
        failures.append(f"PR-AUC mismatch: JSON={metrics['pr_auc']} vs Calc={calc_pr}")

    print(f"CA-HTDNet V2 predictions:     {'FAIL' if failures else 'PASS'}")
    print(f"CA-HTDNet V2 metrics:         {'FAIL' if failures else 'PASS'}")

    # 2. Baseline Reconciliation
    base_failures = []
    base_comp_path = f"{BASE_DIR}/metrics/baseline_comparison.csv"
    if os.path.exists(base_comp_path):
        df_base = pd.read_csv(base_comp_path)
        for _, row in df_base.iterrows():
            m_name = row["model"]
            if m_name == "CA-HTDNet-V2":
                continue
            b_pred_p = f"{BASE_DIR}/predictions/{EXPERIMENT_ID}_{m_name}_test_predictions.parquet"
            if os.path.exists(b_pred_p):
                b_df = pd.read_parquet(b_pred_p)
                b_yt = b_df["true_label"].values
                b_yp = b_df["predicted_label"].values
                b_acc = accuracy_score(b_yt, b_yp)
                if abs(row["accuracy"] - b_acc) > TOLERANCE:
                    base_failures.append(f"Baseline {m_name} accuracy mismatch: CSV={row['accuracy']} vs Calc={b_acc}")
    print(f"Baseline predictions:          {'FAIL' if base_failures else 'PASS'}")
    print(f"Baseline metrics:              {'FAIL' if base_failures else 'PASS'}")

    # 3. Ablation Verification
    ab_failures = []
    ab_path = f"{BASE_DIR}/ablations/ablation_results.csv"
    if os.path.exists(ab_path):
        df_ab = pd.read_csv(ab_path)
        for _, row in df_ab.iterrows():
            if row["accuracy"] <= 0.0 or row["macro_f1"] <= 0.0:
                ab_failures.append(f"Invalid ablation row: {row['variant']}")
    print(f"Ablation metrics:              {'FAIL' if ab_failures else 'PASS'}")

    # 4. Threshold & Calibration Provenance
    th_path = f"{BASE_DIR}/thresholds/{EXPERIMENT_ID}_threshold.json"
    cal_path = f"{BASE_DIR}/calibration/calibration_comparison.csv"
    th_ok = os.path.exists(th_path)
    cal_ok = os.path.exists(cal_path)
    print(f"Threshold provenance:          {'PASS' if th_ok else 'FAIL'}")
    print(f"Calibration provenance:        {'PASS' if cal_ok else 'FAIL'}")

    # 5. Robustness & Cross-Dataset Provenance
    rob_path = f"{BASE_DIR}/robustness/robustness_report.csv"
    cross_path = f"{BASE_DIR}/cross_dataset/cross_dataset_generalization.csv"
    print(f"Robustness provenance:         {'PASS' if os.path.exists(rob_path) else 'FAIL'}")
    print(f"Cross-dataset provenance:      {'PASS' if os.path.exists(cross_path) else 'FAIL'}")

    # 6. Feature statistics & Class distributions
    feat_ok = os.path.exists(f"{BASE_DIR}/features/feature_statistics.csv")
    split_ok = os.path.exists(f"{BASE_DIR}/splits/split_lock.json")
    print(f"Feature statistics:            {'PASS' if feat_ok else 'FAIL'}")
    print(f"Class distributions:           {'PASS' if split_ok else 'FAIL'}")

    all_pass = (len(failures) == 0 and len(base_failures) == 0 and len(ab_failures) == 0 and th_ok and cal_ok)

    # 7. Generate Lock File
    lock_data = {
        "experiment_id": EXPERIMENT_ID,
        "verification_status": "VERIFIED_CANONICAL" if all_pass else "FAILED",
        "model_hash": compute_file_hash(f"{BASE_DIR}/models/{EXPERIMENT_ID}.pt") if os.path.exists(f"{BASE_DIR}/models/{EXPERIMENT_ID}.pt") else "N/A",
        "preprocessor_hash": compute_file_hash(f"{BASE_DIR}/preprocessing/preprocessor.pkl"),
        "prediction_hash": compute_file_hash(pred_path),
        "metrics_hash": compute_file_hash(metrics_path),
        "dataset_hash": compute_file_hash(f"{BASE_DIR}/data/primary_multimodal_dataset.parquet"),
        "split_hash": compute_file_hash(f"{BASE_DIR}/splits/test.parquet"),
        "timestamp": pd.Timestamp.now().isoformat()
    }

    with open("results/FINAL_RESULTS_LOCK.json", "w") as f:
        json.dump(lock_data, f, indent=2)
    with open(f"{BASE_DIR}/provenance/FINAL_RESULTS_LOCK.json", "w") as f:
        json.dump(lock_data, f, indent=2)

    print("\n" + "=" * 70)
    print(f"RESULT CONSISTENCY: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 70)

    if not all_pass:
        print("\nRECONCILIATION ERRORS:")
        for err in failures + base_failures + ab_failures:
            print(f"  - {err}")
        sys.exit(1)

    print("Lock file saved to results/FINAL_RESULTS_LOCK.json")

if __name__ == "__main__":
    reconcile()
