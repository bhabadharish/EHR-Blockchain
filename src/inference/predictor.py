"""
src/inference/predictor.py
==========================
End-to-end real-time inference predictor for CA-HTDNet V2.
Orchestrates:
Validation -> Preprocessing -> Neural Inference -> Calibration -> Threshold Decision -> Risk Severity
"""

import time
import torch
import numpy as np
from typing import Dict, Any, List, Optional

from src.inference.validator import validate_event_schema
from src.inference.preprocessing import InferencePreprocessor
from src.inference.model_loader import ModelLoader
from src.inference.threshold import ThresholdManager
from src.inference.calibration import ProbabilityCalibrator

class RealTimeThreatPredictor:
    def __init__(self):
        self.preprocessor = InferencePreprocessor()
        self.model, self.device = ModelLoader.load_ca_htdnet_v2()
        self.threshold_mgr = ThresholdManager()
        self.calibrator = ProbabilityCalibrator()

    def predict_event(self, event: Dict[str, Any], custom_threshold: Optional[float] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # 1. Validation
        is_valid, msg = validate_event_schema(event)
        if not is_valid:
            return {"status": "ERROR", "error": msg, "latency_ms": 0.0}

        t_val = time.perf_counter()

        # 2. Preprocessing
        x_num, x_cat = self.preprocessor.transform_single(event)
        t_prep = time.perf_counter()

        # 3. Model Inference
        with torch.no_grad():
            t_num = torch.tensor(x_num, device=self.device)
            t_cat = torch.tensor(x_cat, device=self.device)
            out = self.model(t_num, t_cat)
            raw_prob = out["probabilities"][0, 1].item()
            risk_score = out["risk_score"][0, 0].item()
        t_inf = time.perf_counter()

        # 4. Calibration
        calibrated_prob = self.calibrator.calibrate(raw_prob)
        t_cal = time.perf_counter()

        # 5. Threshold Decision
        decision_info = self.threshold_mgr.apply_threshold(calibrated_prob, custom_threshold)
        t_dec = time.perf_counter()

        total_latency_ms = (t_dec - t0) * 1000.0

        return {
            "status": "SUCCESS",
            "prediction": "ATTACK" if decision_info["predicted_class"] == 1 else "NORMAL",
            "raw_probability": float(round(raw_prob, 5)),
            "calibrated_probability": float(round(calibrated_prob, 5)),
            "risk_score": float(round(risk_score, 5)),
            "decision": decision_info["decision"],
            "threshold_used": decision_info["threshold_used"],
            "security_status": decision_info["security_status"],
            "profile": {
                "validation_ms": float(round((t_val - t0) * 1000, 3)),
                "preprocessing_ms": float(round((t_prep - t_val) * 1000, 3)),
                "inference_ms": float(round((t_inf - t_prep) * 1000, 3)),
                "calibration_ms": float(round((t_cal - t_inf) * 1000, 3)),
                "decision_ms": float(round((t_dec - t_cal) * 1000, 3)),
                "total_ms": float(round(total_latency_ms, 3))
            }
        }
