"""Probability Calibration Engine for HAB-IDS (Phase 19).

Calibrates raw ensemble probabilities using validation data:
- Platt Scaling (Sigmoid / Logistic regression)
- Isotonic Regression (Non-parametric piecewise monotonic mapping)
Evaluates Expected Calibration Error (ECE) and Brier Score strictly on validation set.
Never accesses test labels.
"""

import os
import joblib
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE) across uniform confidence bins."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(round(ece, 5))


class ProbabilityCalibrator:
    """Wrapper for Platt Scaling and Isotonic Regression calibration."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrator_ = None
        self.is_fitted = False

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> "ProbabilityCalibrator":
        """Fit calibration curve on validation predictions only."""
        val_probs = np.clip(val_probs.reshape(-1, 1), 1e-6, 1.0 - 1e-6)

        if self.method == "isotonic":
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
            self.calibrator_.fit(val_probs.ravel(), y_val)
        elif self.method == "platt":
            # Platt scaling via univariate logistic regression
            self.calibrator_ = LogisticRegression(C=1.0, solver="lbfgs")
            self.calibrator_.fit(val_probs, y_val)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Transform input probabilities into well-calibrated posterior probabilities."""
        if not self.is_fitted:
            return probs

        p_clip = np.clip(probs, 1e-6, 1.0 - 1e-6)
        if self.method == "isotonic":
            return np.clip(self.calibrator_.predict(p_clip), 0.0, 1.0)
        elif self.method == "platt":
            return np.clip(self.calibrator_.predict_proba(p_clip.reshape(-1, 1))[:, 1], 0.0, 1.0)
        return probs

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ProbabilityCalibrator":
        return joblib.load(filepath)


def select_best_calibrator(
    val_probs: np.ndarray,
    y_val: np.ndarray,
    output_dir: str = "models/calibration"
) -> Tuple[ProbabilityCalibrator, Dict[str, Any]]:
    """Compare uncalibrated, Platt, and Isotonic on validation Brier Score & ECE, returning the superior calibrator."""
    os.makedirs(output_dir, exist_ok=True)

    uncal_brier = float(brier_score_loss(y_val, val_probs))
    uncal_ece = calculate_ece(y_val, val_probs)

    # Platt
    platt = ProbabilityCalibrator(method="platt").fit(val_probs, y_val)
    platt_probs = platt.calibrate(val_probs)
    platt_brier = float(brier_score_loss(y_val, platt_probs))
    platt_ece = calculate_ece(y_val, platt_probs)

    # Isotonic
    isotonic = ProbabilityCalibrator(method="isotonic").fit(val_probs, y_val)
    iso_probs = isotonic.calibrate(val_probs)
    iso_brier = float(brier_score_loss(y_val, iso_probs))
    iso_ece = calculate_ece(y_val, iso_probs)

    comparison = {
        "Uncalibrated": {"Brier": round(uncal_brier, 5), "ECE": uncal_ece},
        "Platt": {"Brier": round(platt_brier, 5), "ECE": platt_ece},
        "Isotonic": {"Brier": round(iso_brier, 5), "ECE": iso_ece},
    }

    # Select lowest Brier score on validation
    if iso_brier <= platt_brier:
        best_name = "isotonic"
        best_cal = isotonic
    else:
        best_name = "platt"
        best_cal = platt

    comparison["Selected_Method"] = best_name
    best_cal.save(os.path.join(output_dir, "best_calibrator.pkl"))

    return best_cal, comparison
