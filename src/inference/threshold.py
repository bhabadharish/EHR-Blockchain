"""
src/inference/threshold.py
==========================
Threshold logic layer for real-time threat decision making.
"""

import os
import json
from typing import Dict, Any

class ThresholdManager:
    def __init__(self, config_path: str = "models/proposed/threshold.json"):
        if not os.path.exists(config_path):
            config_path = "experiments/CAHTDNET_V2_001/thresholds/CAHTDNET_V2_001_threshold.json"
        
        self.optimal_threshold = 0.50
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                data = json.load(f)
                self.optimal_threshold = data.get("optimal_threshold", 0.50)

    def apply_threshold(self, probability: float, custom_threshold: float = None) -> Dict[str, Any]:
        tau = custom_threshold if custom_threshold is not None else self.optimal_threshold
        is_attack = int(probability >= tau)
        return {
            "threshold_used": tau,
            "probability": probability,
            "predicted_class": is_attack,
            "decision": "DENY" if is_attack == 1 else "ALLOW",
            "security_status": "THREAT_DETECTED" if is_attack == 1 else "NORMAL_ACCESS"
        }
