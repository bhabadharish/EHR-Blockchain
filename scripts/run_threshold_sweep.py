"""
scripts/run_threshold_sweep.py
==============================
VALIDATION-ONLY THRESHOLD SWEEP & OPTIMIZATION FOR CA-HTDNet

Strict Methodology:
- Threshold sweep is executed STRICTLY on VALIDATION split probabilities.
- Model is CA-HTDNet under experiment CAHTDNET_FINAL_V001 (ZERO LightGBM reuse).
- Objective: Maximize validation Macro-F1 subject to FNR <= 0.02.
- The selected threshold is evaluated on the locked test set.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.ca_htdnet import CA_HTDNet
from results.result_engine import compute_canonical_metrics

EXPERIMENT_ID = "CAHTDNET_FINAL_V001"
DATASET_NAME = "CICIoT2023+Edge-IIoTset+SyntheticFHIR"

def main():
    print("=" * 70)
    print("STAGE 7: VALIDATION-ONLY THRESHOLD OPTIMIZATION (CA-HTDNet)")
    print("=" * 70)

    os.makedirs("experiments/thresholds", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("models/proposed", exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Validation Data & Model
    data = np.load("data/processed/processed_arrays.npz")
    X_val_num, X_val_cat = data["X_val_num"], data["X_val_cat"]
    y_val = data["y_val"]

    X_test_num, X_test_cat = data["X_test_num"], data["X_test_cat"]
    y_test = data["y_test"]

    model = CA_HTDNet(
        num_numerical=13,
        num_categories=3,
        cardinalities=[16, 20, 16],
        d_model=64,
        tcn_channels=[32, 64],
        attention_heads=4,
        attention_layers=2,
        dropout=0.10
    ).to(device)

    model_path = f"experiments/models/{EXPERIMENT_ID}.pt"
    if not os.path.exists(model_path):
        model_path = f"models/{EXPERIMENT_ID}.pt"
    if not os.path.exists(model_path):
        model_path = "models/proposed/ca_htdnet.pt"

    print(f"Loading CA-HTDNet weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Compute Validation Probabilities
    val_ds = TensorDataset(
        torch.tensor(X_val_num, dtype=torch.float32),
        torch.tensor(X_val_cat, dtype=torch.long)
    )
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    val_probs = []
    with torch.no_grad():
        for bx_num, bx_cat in val_loader:
            bx_num, bx_cat = bx_num.to(device), bx_cat.to(device)
            out = model(bx_num, bx_cat)
            p = torch.softmax(out["logits"], dim=-1).cpu().numpy()
            val_probs.append(p)
    val_probs = np.concatenate(val_probs, axis=0)
    pos_probs_val = val_probs[:, 1] # P(Attack)

    # 3. Validation Threshold Sweep (0.05 -> 0.95)
    thresholds = np.linspace(0.05, 0.95, 91)
    sweep_records = []

    best_thresh = 0.50
    best_score = -1.0
    best_val_metrics = None

    for t in thresholds:
        preds = (pos_probs_val >= t).astype(int)
        m = compute_canonical_metrics(y_true=y_val, y_pred=preds, probabilities=pos_probs_val, threshold=float(t))

        # Healthcare Objective: Maximize Macro-F1 with high penalty for missed critical attacks (FNR > 0.02)
        score = m["macro_f1"] - 2.0 * max(0.0, m["fnr"] - 0.02) - 1.0 * max(0.0, m["fpr"] - 0.05)

        record = {
            "experiment_id": EXPERIMENT_ID,
            "model": "CA-HTDNet",
            "threshold": round(float(t), 3),
            "macro_f1": round(m["macro_f1"], 5),
            "accuracy": round(m["accuracy"], 5),
            "precision": round(m["binary_precision"], 5),
            "recall": round(m["binary_recall"], 5),
            "macro_precision": round(m["macro_precision"], 5),
            "macro_recall": round(m["macro_recall"], 5),
            "fpr": round(m["fpr"], 5),
            "fnr": round(m["fnr"], 5),
            "mcc": round(m["mcc"], 5),
            "objective_score": round(float(score), 5)
        }
        sweep_records.append(record)

        if score > best_score:
            best_score = score
            best_thresh = round(float(t), 3)
            best_val_metrics = record

    df_sweep = pd.DataFrame(sweep_records)
    for sw_path in ["results/threshold_sweep.csv", f"experiments/thresholds/{EXPERIMENT_ID}_sweep.csv"]:
        df_sweep.to_csv(sw_path, index=False)

    print(f"\nOPTIMAL VALIDATION THRESHOLD SELECTED: {best_thresh}")
    print(f"  Validation Macro-F1 at optimal: {best_val_metrics['macro_f1']*100:.2f}% | FPR: {best_val_metrics['fpr']*100:.3f}% | FNR: {best_val_metrics['fnr']*100:.3f}%")

    # 4. Evaluate Optimal Threshold on Locked Test Set (EXACTLY ONCE)
    print("\n--- Evaluating Optimal Threshold on Locked Test Set ---")
    test_ds = TensorDataset(
        torch.tensor(X_test_num, dtype=torch.float32),
        torch.tensor(X_test_cat, dtype=torch.long)
    )
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    test_probs = []
    with torch.no_grad():
        for bx_num, bx_cat in test_loader:
            bx_num, bx_cat = bx_num.to(device), bx_cat.to(device)
            out = model(bx_num, bx_cat)
            p = torch.softmax(out["logits"], dim=-1).cpu().numpy()
            test_probs.append(p)
    test_probs = np.concatenate(test_probs, axis=0)
    pos_probs_test = test_probs[:, 1]
    test_preds_opt = (pos_probs_test >= best_thresh).astype(int)

    test_metrics_opt = compute_canonical_metrics(
        y_true=y_test,
        y_pred=test_preds_opt,
        probabilities=pos_probs_test,
        threshold=best_thresh,
        metadata={"experiment_id": EXPERIMENT_ID, "model": "CA-HTDNet", "split": "locked_test"}
    )

    print(f"  Test Results at Optimal Threshold ({best_thresh}):")
    print(f"    Accuracy: {test_metrics_opt['accuracy']*100:.2f}% | Macro-F1: {test_metrics_opt['macro_f1']*100:.2f}% | FPR: {test_metrics_opt['fpr']*100:.3f}% | FNR: {test_metrics_opt['fnr']*100:.3f}%")

    threshold_meta = {
        "experiment_id": EXPERIMENT_ID,
        "model": "CA-HTDNet",
        "selection_split": "VALIDATION_STRICT",
        "test_set_used_in_selection": False,
        "optimal_threshold": best_thresh,
        "validation_metrics": best_val_metrics,
        "locked_test_metrics_at_optimal": {
            "accuracy": test_metrics_opt["accuracy"],
            "macro_f1": test_metrics_opt["macro_f1"],
            "macro_recall": test_metrics_opt["macro_recall"],
            "macro_precision": test_metrics_opt["macro_precision"],
            "fpr": test_metrics_opt["fpr"],
            "fnr": test_metrics_opt["fnr"],
            "mcc": test_metrics_opt["mcc"]
        }
    }

    for meta_path in [f"experiments/thresholds/{EXPERIMENT_ID}_threshold.json", "models/proposed/threshold.json"]:
        with open(meta_path, "w") as f:
            json.dump(threshold_meta, f, indent=2)

    print(f"Threshold metadata saved to experiments/thresholds/{EXPERIMENT_ID}_threshold.json")

if __name__ == "__main__":
    main()
