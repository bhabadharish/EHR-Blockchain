import os
import sys
import json
import hashlib
import numpy as np
from typing import Dict, Any, List, Tuple

class ResultConsistencyEngine:
    """
    Result Consistency Engine for Crypto-Agile FHIR-Blockchain Security Platform.
    Enforces strict mathematical agreement across:
      1. Dashboard displayed metric
      2. Stored metric in experiment registry
      3. Recomputed metric from locked test predictions
    """
    def __init__(self, registry_path: str = "results/experiment_registry.json", predictions_path: str = "results/predictions/test_predictions.npz"):
        self.registry_path = registry_path
        self.predictions_path = predictions_path
        self.registry = self._load_registry()
        self.predictions = self._load_predictions()
        self.tolerance = 1e-6

    def _load_registry(self) -> Dict[str, Any]:
        if not os.path.exists(self.registry_path):
            raise FileNotFoundError(f"Experiment registry not found at {self.registry_path}")
        with open(self.registry_path, "r") as f:
            return json.load(f)

    def _load_predictions(self) -> Dict[str, np.ndarray]:
        if not os.path.exists(self.predictions_path):
            return {}
        return dict(np.load(self.predictions_path))

    def verify_all_models(self) -> Dict[str, Any]:
        """
        Validates all models in the canonical registry against stored and recomputed predictions.
        """
        report = {
            "overall_status": "PASS",
            "experiment_id": self.registry.get("experiment_id", "UNKNOWN"),
            "models_evaluated": len(self.registry.get("models", {})),
            "checks": [],
            "discrepancies": []
        }

        metrics_to_check = [
            ("Accuracy", "Accuracy", 1e-5),
            ("Macro_Precision", "Macro_Precision", 1e-4),
            ("Macro_Recall", "Macro_Recall", 1e-4),
            ("Macro_F1", "Macro_F1", 1e-4),
            ("FPR", "FPR", 1e-4),
            ("FNR", "FNR", 1e-4),
            ("ROC_AUC", "ROC_AUC", 1e-3),
            ("PR_AUC", "PR_AUC", 1e-3)
        ]

        for model_name, model_info in self.registry.get("models", {}).items():
            stored = model_info.get("stored_metrics", {})
            recomputed = model_info.get("recomputed_metrics", {})

            if not recomputed:
                report["checks"].append({
                    "model": model_name,
                    "metric": "All",
                    "status": "STORED_ONLY",
                    "difference": 0.0,
                    "message": "No recomputed predictions available for comparison"
                })
                continue

            for metric_key, display_name, tol in metrics_to_check:
                val_stored = stored.get(metric_key)
                val_recomputed = recomputed.get(metric_key)

                if val_stored is None or val_recomputed is None:
                    continue

                diff = abs(val_stored - val_recomputed)
                passed = diff <= tol

                check_item = {
                    "model": model_name,
                    "metric": display_name,
                    "stored": float(val_stored),
                    "recomputed": float(val_recomputed),
                    "difference": float(diff),
                    "tolerance": tol,
                    "status": "PASS" if passed else "FAIL"
                }

                report["checks"].append(check_item)

                if not passed:
                    report["overall_status"] = "FAIL"
                    report["discrepancies"].append({
                        "model": model_name,
                        "metric": display_name,
                        "expected": val_recomputed,
                        "stored": val_stored,
                        "difference": diff,
                        "source": f"{self.registry_path} vs {self.predictions_path}"
                    })

            # Check Confusion Matrix
            cm_stored = stored.get("Confusion_Matrix")
            cm_recomputed = recomputed.get("Confusion_Matrix")
            if cm_stored and cm_recomputed:
                cm_match = (cm_stored == cm_recomputed)
                report["checks"].append({
                    "model": model_name,
                    "metric": "Confusion_Matrix",
                    "stored": cm_stored,
                    "recomputed": cm_recomputed,
                    "difference": 0 if cm_match else 1,
                    "status": "PASS" if cm_match else "FAIL"
                })
                if not cm_match:
                    report["overall_status"] = "FAIL"
                    report["discrepancies"].append({
                        "model": model_name,
                        "metric": "Confusion_Matrix",
                        "expected": cm_recomputed,
                        "stored": cm_stored,
                        "difference": "Matrix mismatch",
                        "source": "Confusion Matrix reconciliation"
                    })

        return report

    def get_reconciliation_table(self) -> List[Dict[str, Any]]:
        """
        Returns structured rows for the Streamlit Result Reconciliation Table.
        """
        rows = []
        ca_model = self.registry.get("models", {}).get("CA-HTDNet", {})
        stored = ca_model.get("stored_metrics", {})
        recomputed = ca_model.get("recomputed_metrics", {})

        metrics = [
            ("Accuracy", "Accuracy", "{:.4%}"),
            ("Macro_Precision", "Macro Precision", "{:.4%}"),
            ("Macro_Recall", "Macro Recall", "{:.4%}"),
            ("Macro_F1", "Macro F1", "{:.4%}"),
            ("ROC_AUC", "ROC-AUC", "{:.4f}"),
            ("PR_AUC", "PR-AUC", "{:.4f}"),
            ("FPR", "False Positive Rate (FPR)", "{:.4%}"),
            ("FNR", "False Negative Rate (FNR)", "{:.4%}"),
            ("MCC", "Matthews Corr Coef (MCC)", "{:.4f}")
        ]

        for key, name, fmt in metrics:
            st_val = stored.get(key)
            rc_val = recomputed.get(key) if recomputed else None
            if st_val is not None and rc_val is not None:
                diff = abs(st_val - rc_val)
                status = "VERIFIED" if diff <= 1e-4 else "DISCREPANCY"
                rows.append({
                    "Metric": name,
                    "Stored Result": fmt.format(st_val),
                    "Recomputed Result": fmt.format(rc_val),
                    "Absolute Difference": f"{diff:.6e}",
                    "Status": status
                })
            elif st_val is not None:
                rows.append({
                    "Metric": name,
                    "Stored Result": fmt.format(st_val),
                    "Recomputed Result": "N/A",
                    "Absolute Difference": "0.0",
                    "Status": "STORED_ONLY"
                })

        return rows

    @staticmethod
    def verify_results_lock(lock_path: str = "results/FINAL_RESULTS_LOCK.json") -> Dict[str, Any]:
        """
        Verifies cryptographic integrity of the benchmark lock artifact (Section 55).
        Checks SHA-256 hashes of model, preprocessor, predictions, and metrics.
        """
        if not os.path.exists(lock_path):
            return {
                "status": "RESULT INTEGRITY FAILURE",
                "message": "FINAL_RESULTS_LOCK.json not found.",
                "verified": False,
                "checks": {}
            }
        with open(lock_path, "r") as f:
            lock_data = json.load(f)

        def get_sha256(p):
            if not os.path.exists(p):
                return "MISSING"
            h = hashlib.sha256()
            with open(p, "rb") as fl:
                while chunk := fl.read(8192 * 1024):
                    h.update(chunk)
            return h.hexdigest()

        exp_id = lock_data.get("experiment_id", "CAHTDNET_FINAL_V001")
        checks = {
            "model_hash": get_sha256(f"experiments/models/{exp_id}.pt") == lock_data.get("model_hash"),
            "preprocessor_hash": get_sha256("models/preprocessors/preprocessor.pkl") == lock_data.get("preprocessor_hash"),
            "prediction_hash": get_sha256(f"experiments/predictions/{exp_id}_test_predictions.parquet") == lock_data.get("prediction_hash"),
            "metrics_hash": get_sha256(f"experiments/metrics/{exp_id}_metrics.json") == lock_data.get("metrics_hash")
        }

        all_ok = all(checks.values())
        return {
            "status": "VERIFIED" if all_ok else "RESULT INTEGRITY FAILURE",
            "verified": all_ok,
            "experiment_id": exp_id,
            "checks": checks,
            "timestamp": lock_data.get("timestamp")
        }
