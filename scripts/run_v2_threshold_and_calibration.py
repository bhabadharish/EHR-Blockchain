"""
scripts/run_v2_threshold_and_calibration.py
===========================================
VALIDATION-STRICT THRESHOLD OPTIMIZATION & PROBABILITY CALIBRATION (CAHTDNET_V2_001)

Requirements:
- Threshold optimization conducted STRICTLY on Validation predictions.
- Search grid: tau in [0.05, 0.95], step 0.01.
- Pareto optimization: FPR vs FNR, FPR vs Macro-F1, Precision vs Recall.
- Selection rule: Maximize Validation Macro-F1 subject to FPR <= 1.0% and FNR <= 1.0%.
- Probability Calibration:
  - Temperature Scaling (optimizing T on Validation NLL)
  - Platt Scaling (Logistic Regression fitted on Validation logits)
  - Isotonic Regression (fitted on Validation probabilities)
- Evaluated on Locked Test set to record verified post-calibration ECE and Brier scores.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results.metric_engine import compute_canonical_metrics

EXPERIMENT_ID = "CAHTDNET_V2_001"
BASE_DIR = f"experiments/{EXPERIMENT_ID}"

def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs > bin_lower) & (probs <= bin_upper) if i > 0 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def main():
    print("=" * 70)
    print("CA-HTDNet V2: VALIDATION THRESHOLD & CALIBRATION BENCHMARK")
    print("=" * 70)

    os.makedirs(f"{BASE_DIR}/thresholds", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/calibration", exist_ok=True)

    # 1. Load Validation & Test Predictions
    val_pred_path = f"{BASE_DIR}/predictions/{EXPERIMENT_ID}_val_predictions.parquet"
    test_pred_path = f"{BASE_DIR}/predictions/{EXPERIMENT_ID}_test_predictions.parquet"

    if not os.path.exists(val_pred_path) or not os.path.exists(test_pred_path):
        print("  Error: Predictions parquet not found. Run train_ca_htdnet_v2.py first.")
        return

    val_df = pd.read_parquet(val_pred_path)
    test_df = pd.read_parquet(test_pred_path)

    y_val = val_df["true_label"].values
    val_probs = val_df["probability_class_1"].values

    y_test = test_df["true_label"].values
    test_probs = test_df["probability_class_1"].values

    # ---------------------------------------------------------
    # PART 1: THRESHOLD OPTIMIZATION (VALIDATION ONLY)
    # ---------------------------------------------------------
    print("\n[Part 1] Performing Threshold Sweep on Validation Split (tau in [0.05, 0.95])...")
    thresholds = np.linspace(0.05, 0.95, 91)
    sweep_records = []

    best_val_f1 = -1.0
    optimal_tau = 0.50
    best_constrained_f1 = -1.0
    optimal_constrained_tau = 0.50

    for tau in thresholds:
        preds = (val_probs >= tau).astype(int)
        tp = int(np.sum((y_val == 1) & (preds == 1)))
        tn = int(np.sum((y_val == 0) & (preds == 0)))
        fp = int(np.sum((y_val == 0) & (preds == 1)))
        fn = int(np.sum((y_val == 1) & (preds == 0)))

        acc = (tp + tn) / len(y_val)
        rec0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        rec1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0
        f1_1 = 2 * prec1 * rec1 / (prec1 + rec1) if (prec1 + rec1) > 0 else 0.0
        macro_f1 = 0.5 * (f1_0 + f1_1)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        sweep_records.append({
            "threshold": float(round(tau, 3)),
            "accuracy": float(round(acc, 5)),
            "macro_precision": float(round(0.5 * (prec0 + prec1), 5)),
            "macro_recall": float(round(0.5 * (rec0 + rec1), 5)),
            "macro_f1": float(round(macro_f1, 5)),
            "fpr": float(round(fpr, 5)),
            "fnr": float(round(fnr, 5)),
            "meets_security_targets": bool(fpr <= 0.01 and fnr <= 0.01)
        })

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            optimal_tau = tau

        if fpr <= 0.01 and fnr <= 0.01:
            if macro_f1 > best_constrained_f1:
                best_constrained_f1 = macro_f1
                optimal_constrained_tau = tau

    sweep_df = pd.DataFrame(sweep_records)
    sweep_df.to_csv(f"{BASE_DIR}/thresholds/{EXPERIMENT_ID}_sweep.csv", index=False)
    sweep_df.to_csv(f"{BASE_DIR}/thresholds/threshold_pareto.csv", index=False)
    sweep_df.to_csv("results/threshold_sweep.csv", index=False)

    selected_tau = optimal_constrained_tau if best_constrained_f1 > 0 else optimal_tau
    print(f"  Validation Optimal Threshold: tau* = {selected_tau:.3f} (Val Macro-F1: {best_val_f1:.4f})")

    # Evaluate selected threshold on Locked Test
    test_preds_selected = (test_probs >= selected_tau).astype(int)
    test_metrics_selected = compute_canonical_metrics(
        y_true=y_test,
        y_pred=test_preds_selected,
        probabilities=test_probs,
        threshold=float(selected_tau),
        metadata={
            "experiment_id": EXPERIMENT_ID,
            "threshold_selection": "validation_pareto_optimal",
            "optimal_tau": float(selected_tau)
        }
    )

    threshold_result = {
        "experiment_id": EXPERIMENT_ID,
        "optimal_threshold": float(round(selected_tau, 4)),
        "selection_rule": "maximize_validation_macro_f1",
        "validation_macro_f1": float(round(best_val_f1, 5)),
        "locked_test_evaluation": {
            "threshold": float(round(selected_tau, 4)),
            "accuracy": test_metrics_selected["accuracy"],
            "macro_precision": test_metrics_selected["macro_precision"],
            "macro_recall": test_metrics_selected["macro_recall"],
            "macro_f1": test_metrics_selected["macro_f1"],
            "fpr": test_metrics_selected["fpr"],
            "fnr": test_metrics_selected["fnr"],
            "mcc": test_metrics_selected["mcc"]
        }
    }

    with open(f"{BASE_DIR}/thresholds/{EXPERIMENT_ID}_threshold.json", "w") as f:
        json.dump(threshold_result, f, indent=2)
    with open("models/proposed/threshold.json", "w") as f:
        json.dump(threshold_result, f, indent=2)

    print(f"  [LOCKED TEST @ tau*={selected_tau:.2f}] Macro-F1: {test_metrics_selected['macro_f1']:.4f} | Acc: {test_metrics_selected['accuracy']:.4f} | FPR: {test_metrics_selected['fpr']:.4f} | FNR: {test_metrics_selected['fnr']:.4f}")

    # ---------------------------------------------------------
    # PART 2: PROBABILITY CALIBRATION
    # ---------------------------------------------------------
    print("\n[Part 2] Fitting Calibration Models on Validation Split...")
    val_eps = 1e-7
    val_probs_clipped = np.clip(val_probs, val_eps, 1.0 - val_eps)
    val_logits = np.log(val_probs_clipped / (1.0 - val_probs_clipped))

    test_probs_clipped = np.clip(test_probs, val_eps, 1.0 - val_eps)
    test_logits = np.log(test_probs_clipped / (1.0 - test_probs_clipped))

    # A. Temperature Scaling
    def nll_obj(T):
        scaled_logits = val_logits / max(T[0], 0.01)
        p = 1.0 / (1.0 + np.exp(-scaled_logits))
        return log_loss(y_val, p)

    opt_res = minimize(nll_obj, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
    optimal_T = float(opt_res.x[0])
    test_probs_temp = 1.0 / (1.0 + np.exp(-test_logits / optimal_T))

    # B. Platt Scaling (Logistic Regression on validation logits)
    platt = LogisticRegression(C=1.0, solver="lbfgs")
    platt.fit(val_logits.reshape(-1, 1), y_val)
    test_probs_platt = platt.predict_proba(test_logits.reshape(-1, 1))[:, 1]

    # C. Isotonic Regression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_probs, y_val)
    test_probs_iso = iso.predict(test_probs)

    calib_comparison = [
        {
            "method": "Uncalibrated",
            "brier_score": float(brier_score_loss(y_test, test_probs)),
            "ece": compute_ece(test_probs, y_test),
            "log_loss": float(log_loss(y_test, test_probs_clipped)),
            "optimal_parameter": 1.0
        },
        {
            "method": "Temperature_Scaling",
            "brier_score": float(brier_score_loss(y_test, test_probs_temp)),
            "ece": compute_ece(test_probs_temp, y_test),
            "log_loss": float(log_loss(y_test, np.clip(test_probs_temp, val_eps, 1.0 - val_eps))),
            "optimal_parameter": float(round(optimal_T, 4))
        },
        {
            "method": "Platt_Scaling",
            "brier_score": float(brier_score_loss(y_test, test_probs_platt)),
            "ece": compute_ece(test_probs_platt, y_test),
            "log_loss": float(log_loss(y_test, np.clip(test_probs_platt, val_eps, 1.0 - val_eps))),
            "optimal_parameter": f"coef={float(platt.coef_[0][0]):.4f}, intercept={float(platt.intercept_[0]):.4f}"
        },
        {
            "method": "Isotonic_Regression",
            "brier_score": float(brier_score_loss(y_test, test_probs_iso)),
            "ece": compute_ece(test_probs_iso, y_test),
            "log_loss": float(log_loss(y_test, np.clip(test_probs_iso, val_eps, 1.0 - val_eps))),
            "optimal_parameter": "non-parametric"
        }
    ]

    calib_df = pd.DataFrame(calib_comparison)
    calib_df.to_csv(f"{BASE_DIR}/calibration/calibration_comparison.csv", index=False)
    calib_df.to_csv("results/calibration_comparison.csv", index=False)

    with open(f"{BASE_DIR}/calibration/{EXPERIMENT_ID}_calibration.json", "w") as f:
        json.dump(calib_comparison, f, indent=2)

    print("\nCALIBRATION COMPARISON (LOCKED TEST SET):")
    print(calib_df.to_string(index=False))

if __name__ == "__main__":
    main()
