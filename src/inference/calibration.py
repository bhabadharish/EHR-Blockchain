"""
src/inference/calibration.py
============================
Probability calibration layer for real-time threat inference.
"""

import os
import json
import numpy as np
from typing import Dict, Any

class ProbabilityCalibrator:
    def __init__(self, config_path: str = "experiments/CAHTDNET_V2_001/calibration/CAHTDNET_V2_001_calibration.json"):
        self.temperature = 1.0
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                data = json.load(f)
                for entry in data:
                    if entry.get("method") == "Temperature_Scaling":
                        self.temperature = float(entry.get("optimal_parameter", 1.0))

    def calibrate(self, uncalibrated_prob: float) -> float:
        eps = 1e-7
        p = np.clip(uncalibrated_prob, eps, 1.0 - eps)
        logit = np.log(p / (1.0 - p))
        scaled_logit = logit / max(self.temperature, 0.01)
        calibrated_p = 1.0 / (1.0 + np.exp(-scaled_logit))
        return float(np.clip(calibrated_p, 0.0, 1.0))
