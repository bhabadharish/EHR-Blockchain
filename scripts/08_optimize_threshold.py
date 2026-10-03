import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef

def main():
    print("=" * 70)
    print("STAGE 8: MULTI-OBJECTIVE VALIDATION THRESHOLD OPTIMIZATION")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)
    os.makedirs("models/proposed", exist_ok=True)

    data = np.load("data/processed/processed_arrays.npz")
    y_val = data["y_val"]

    # Load validation probabilities
    hints = np.load("data/processed/gbdt_hints.npz")
    y_prob = hints["gbdt_val"][:, 1] # attack probability

    thresholds = np.linspace(0.05, 0.95, 91)
    sweep_results = []

    best_thresh = 0.50
    best_obj_score = -1.0
    best_metrics = {}

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        
        macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
        prec = float(precision_score(y_val, preds, zero_division=0))
        rec = float(recall_score(y_val, preds, zero_division=0))
        mcc = float(matthews_corrcoef(y_val, preds))

        cm = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        # Multi-objective objective: Maximize Macro-F1 with penalties if FPR > 0.05 or FNR > 0.05
        # High-security healthcare objective: Heavy penalty for Missed Attacks (FNR)
        obj_score = macro_f1 - 2.0 * max(0.0, fnr - 0.02) - 1.0 * max(0.0, fpr - 0.05)

        res_dict = {
            "threshold": float(round(t, 3)),
            "macro_f1": float(round(macro_f1, 4)),
            "precision": float(round(prec, 4)),
            "recall": float(round(rec, 4)),
            "fpr": float(round(fpr, 4)),
            "fnr": float(round(fnr, 4)),
            "mcc": float(round(mcc, 4)),
            "objective_score": float(round(obj_score, 4))
        }
        sweep_results.append(res_dict)

        if obj_score > best_obj_score:
            best_obj_score = obj_score
            best_thresh = float(round(t, 3))
            best_metrics = res_dict

    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_csv("results/threshold_sweep.csv", index=False)

    threshold_metadata = {
        "selection_partition": "VALIDATION_STRICT",
        "test_set_used": False,
        "objective": "Maximize Macro-F1 subject to FNR <= 0.02, FPR <= 0.05",
        "optimal_threshold": best_thresh,
        "validation_metrics_at_optimal": best_metrics
    }

    with open("models/proposed/threshold.json", "w") as f:
        json.dump(threshold_metadata, f, indent=2)

    print(f"\nOPTIMAL VALIDATION THRESHOLD DETERMINED: {best_thresh}")
    print(f"  Macro-F1: {best_metrics['macro_f1']*100:.2f}% | Recall: {best_metrics['recall']*100:.2f}% | FPR: {best_metrics['fpr']*100:.3f}% | FNR: {best_metrics['fnr']*100:.3f}%")
    print(f"Threshold metadata frozen and saved to models/proposed/threshold.json.")

if __name__ == "__main__":
    main()
