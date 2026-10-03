import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) across confidence bins.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)

def main():
    print("=" * 70)
    print("STAGE 7: PROBABILITY CALIBRATION (TEMPERATURE SCALING & PLATT SCALING)")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)
    os.makedirs("models/proposed", exist_ok=True)

    # Load validation metrics and predictions
    data = np.load("data/processed/processed_arrays.npz")
    y_val = data["y_val"]

    # Load GBDT hints
    hints = np.load("data/processed/gbdt_hints.npz")
    raw_probs = hints["gbdt_val"][:, 1] # LightGBM attack probability

    # Uncalibrated baseline metrics
    uncal_brier = float(brier_score_loss(y_val, raw_probs))
    uncal_ece = compute_expected_calibration_error(y_val, raw_probs)

    # 1. Temperature Scaling
    # Optimize scalar T on validation cross-entropy
    T_candidates = np.linspace(0.5, 3.0, 50)
    best_t = 1.0
    best_t_brier = 1.0
    
    for T in T_candidates:
        scaled_p = 1.0 / (1.0 + np.exp(-np.log(np.maximum(raw_probs, 1e-7) / np.maximum(1 - raw_probs, 1e-7)) / T))
        b_score = brier_score_loss(y_val, scaled_p)
        if b_score < best_t_brier:
            best_t_brier = b_score
            best_t = float(T)

    temp_probs = 1.0 / (1.0 + np.exp(-np.log(np.maximum(raw_probs, 1e-7) / np.maximum(1 - raw_probs, 1e-7)) / best_t))
    temp_ece = compute_expected_calibration_error(y_val, temp_probs)

    # 2. Platt Scaling (Logistic Regression on validation logits)
    val_logits = np.log(np.maximum(raw_probs, 1e-7) / np.maximum(1 - raw_probs, 1e-7)).reshape(-1, 1)
    platt_model = LogisticRegression(C=1.0, solver="lbfgs")
    platt_model.fit(val_logits, y_val)
    platt_probs = platt_model.predict_proba(val_logits)[:, 1]
    platt_brier = float(brier_score_loss(y_val, platt_probs))
    platt_ece = compute_expected_calibration_error(y_val, platt_probs)

    calibration_report = {
        "Uncalibrated": {
            "Brier_Score": uncal_brier,
            "ECE": uncal_ece
        },
        "Temperature_Scaling": {
            "Optimal_Temperature": best_t,
            "Brier_Score": float(best_t_brier),
            "ECE": temp_ece
        },
        "Platt_Scaling": {
            "Brier_Score": platt_brier,
            "ECE": platt_ece
        },
        "Selected_Method": "Temperature_Scaling" if temp_ece < platt_ece else "Platt_Scaling"
    }

    with open("results/calibration_report.json", "w") as f:
        json.dump(calibration_report, f, indent=2)

    df_cal = pd.DataFrame([
        {"Method": "Uncalibrated", "Brier Score": f"{uncal_brier:.5f}", "ECE": f"{uncal_ece:.5f}"},
        {"Method": f"Temperature Scaling (T={best_t:.2f})", "Brier Score": f"{best_t_brier:.5f}", "ECE": f"{temp_ece:.5f}"},
        {"Method": "Platt Scaling (Sigmoid)", "Brier Score": f"{platt_brier:.5f}", "ECE": f"{platt_ece:.5f}"}
    ])
    df_cal.to_csv("results/calibration_comparison.csv", index=False)

    print("\nCALIBRATION BENCHMARKS (MEASURED ON VALIDATION SPLIT):")
    print(df_cal.to_string(index=False))
    print(f"\nSaved calibration metadata to results/calibration_report.json.")

if __name__ == "__main__":
    main()
