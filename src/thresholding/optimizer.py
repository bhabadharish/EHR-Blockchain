"""Security Decision Threshold Optimization and Cost-Sensitive Evaluation (Phase 20 & 21).

Searches validation probabilities for an optimal operational threshold:
Primary Objective: Maximize Macro-F1 subject to:
    FPR <= 1.0% (0.01)
    FNR <= 1.0% (0.01)
Evaluates asymmetric healthcare security costs (FN:FP ratios: 1:1, 2:1, 5:1, 10:1).
Selected threshold is saved to models/final/decision_threshold.json.
Never optimizes against test data.
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix


def sweep_thresholds(
    val_probs: np.ndarray,
    y_val: np.ndarray,
    step: float = 0.005
) -> pd.DataFrame:
    """Sweep decision thresholds across [0.01, 0.99] on validation split."""
    y_val = np.asarray(y_val, dtype=int)
    thresholds = np.arange(0.01, 0.99 + step, step)
    records = []

    for th in thresholds:
        preds = (val_probs >= th).astype(int)
        cm = confusion_matrix(y_val, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fpr = float(fp / max(fp + tn, 1))
        fnr = float(fn / max(fn + tp, 1))
        macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
        mcc = float(matthews_corrcoef(y_val, preds))
        acc = float((tp + tn) / max(len(y_val), 1))

        # Cost evaluations
        cost_1_1 = 1.0 * fn + 1.0 * fp
        cost_2_1 = 2.0 * fn + 1.0 * fp
        cost_5_1 = 5.0 * fn + 1.0 * fp
        cost_10_1 = 10.0 * fn + 1.0 * fp

        records.append({
            "threshold": round(float(th), 4),
            "macro_f1": round(macro_f1, 5),
            "mcc": round(mcc, 5),
            "accuracy": round(acc, 5),
            "fpr": round(fpr, 5),
            "fnr": round(fnr, 5),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "cost_1_1": round(cost_1_1, 1),
            "cost_2_1": round(cost_2_1, 1),
            "cost_5_1": round(cost_5_1, 1),
            "cost_10_1": round(cost_10_1, 1),
        })

    return pd.DataFrame(records)


def optimize_security_threshold(
    val_probs: np.ndarray,
    y_val: np.ndarray,
    target_fpr_max: float = 0.01,
    target_fnr_max: float = 0.01,
    output_path: str = "models/final/decision_threshold.json"
) -> Dict[str, Any]:
    """Find threshold maximizing Macro-F1 on validation subject to FPR/FNR constraints."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sweep = sweep_thresholds(val_probs, y_val)

    # 1. Primary Filter: Both FPR <= target and FNR <= target
    candidates = df_sweep[(df_sweep["fpr"] <= target_fpr_max) & (df_sweep["fnr"] <= target_fnr_max)]
    constraints_satisfied = True

    if not candidates.empty:
        # Pick threshold maximizing Macro-F1
        best_row = candidates.sort_values(by=["macro_f1", "mcc"], ascending=False).iloc[0]
    else:
        # If strict mutual constraint cannot be met simultaneously, find point minimizing total error rate
        constraints_satisfied = False
        df_sweep["constraint_penalty"] = np.maximum(0, df_sweep["fpr"] - target_fpr_max) + np.maximum(0, df_sweep["fnr"] - target_fnr_max)
        best_row = df_sweep.sort_values(by=["constraint_penalty", "macro_f1"], ascending=[True, False]).iloc[0]

    chosen_threshold = float(best_row["threshold"])

    result = {
        "optimal_threshold": chosen_threshold,
        "constraints_satisfied": constraints_satisfied,
        "target_fpr_max": target_fpr_max,
        "target_fnr_max": target_fnr_max,
        "validation_metrics_at_threshold": {
            "macro_f1": float(best_row["macro_f1"]),
            "mcc": float(best_row["mcc"]),
            "accuracy": float(best_row["accuracy"]),
            "fpr": float(best_row["fpr"]),
            "fnr": float(best_row["fnr"]),
            "confusion_matrix": {
                "TN": int(best_row["tn"]),
                "FP": int(best_row["fp"]),
                "FN": int(best_row["fn"]),
                "TP": int(best_row["tp"]),
            },
            "costs": {
                "cost_1_1": float(best_row["cost_1_1"]),
                "cost_2_1": float(best_row["cost_2_1"]),
                "cost_5_1": float(best_row["cost_5_1"]),
                "cost_10_1": float(best_row["cost_10_1"]),
            }
        },
        "cost_analysis_rationale": "In healthcare environments, a False Negative (undetected intrusion, compromised EHR/patient monitoring) carries severe patient safety and data breach liability, requiring asymmetric weighting (2:1 to 10:1)."
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
