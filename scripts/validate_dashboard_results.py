import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard.validation.result_consistency import ResultConsistencyEngine
from dashboard.inference.model_loader import FrozenModelLoader

def sha256_file(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("====================================")
    print("DASHBOARD RESULT VALIDATION")
    print("====================================")

    results = {}

    # 1. Model Artifacts
    model_path = "models/proposed/ca_htdnet.pt"
    results["Model artifacts"] = "PASS" if os.path.exists(model_path) and os.path.getsize(model_path) > 100000 else "FAIL"

    # 2. Preprocessors
    prep_path = "models/preprocessors/preprocessor.pkl"
    results["Preprocessors"] = "PASS" if os.path.exists(prep_path) and os.path.getsize(prep_path) > 500 else "FAIL"

    # 3. Dataset Metadata
    meta_path = "data/metadata/split_manifest.json"
    results["Dataset metadata"] = "PASS" if os.path.exists(meta_path) else "FAIL"

    # 4. Class Mapping
    map_path = "data/metadata/label_mapping.json"
    if os.path.exists(map_path):
        with open(map_path) as f:
            lm = json.load(f)
            has_classes = ("binary_classes" in lm) and ("detailed_classes" in lm)
            results["Class mapping"] = "PASS" if has_classes else "FAIL"
    else:
        results["Class mapping"] = "FAIL"

    # 5. Metrics & Confusion Matrix & Experiment Registry
    try:
        consistency_engine = ResultConsistencyEngine()
        val_report = consistency_engine.verify_all_models()
        results["Metrics"] = "PASS" if val_report["overall_status"] == "PASS" else "FAIL"
        
        # Check CM specifically
        cm_pass = True
        for c in val_report.get("checks", []):
            if c.get("metric") == "Confusion_Matrix" and c.get("status") != "PASS":
                cm_pass = False
        results["Confusion matrix"] = "PASS" if cm_pass else "FAIL"
        results["Experiment registry"] = "PASS" if os.path.exists("results/experiment_registry.json") else "FAIL"
    except Exception as e:
        print(f"Consistency check error: {e}")
        results["Metrics"] = "FAIL"
        results["Confusion matrix"] = "FAIL"
        results["Experiment registry"] = "FAIL"

    # 6. ROC Data & PR Data
    curves_path = "results/curves/roc_pr_curves.json"
    if os.path.exists(curves_path):
        with open(curves_path) as f:
            curves = json.load(f)
            ca_curves = curves.get("CA-HTDNet", {})
            has_roc = "roc" in ca_curves and len(ca_curves["roc"].get("fpr", [])) > 10
            has_pr = "pr" in ca_curves and len(ca_curves["pr"].get("recall", [])) > 10
            results["ROC data"] = "PASS" if has_roc else "FAIL"
            results["PR data"] = "PASS" if has_pr else "FAIL"
    else:
        results["ROC data"] = "FAIL"
        results["PR data"] = "FAIL"

    # 7. Threshold
    thresh_path = "models/proposed/threshold.json"
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            tm = json.load(f)
            t_val = tm.get("optimal_threshold", 0.0)
            results["Threshold"] = "PASS" if (0.0 < t_val < 1.0) else "FAIL"
    else:
        results["Threshold"] = "FAIL"

    # 8. Cross-Dataset Data
    cross_path = "results/cross_dataset_generalization.csv"
    results["Cross-dataset data"] = "PASS" if os.path.exists(cross_path) and os.path.getsize(cross_path) > 100 else "FAIL"

    # Print Formatted Report
    for key, status in results.items():
        print(f"{key + ':':22s} {status}")

    overall = "PASS" if all(v == "PASS" for v in results.values()) else "FAIL"
    print("\nOverall:")
    print(overall)

    # Save JSON and Markdown Reports
    val_report_dict = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "overall_status": overall,
        "results": results,
        "registry_experiment_id": "CAHTDNet_final_locked_v001"
    }
    with open("dashboard_validation_report.json", "w") as f:
        json.dump(val_report_dict, f, indent=2)

    with open("dashboard_validation_report.md", "w") as f:
        f.write("# Dashboard Result Validation Report\n\n")
        f.write(f"**Overall Status:** {overall}  \n\n")
        f.write("| Component | Status |\n|---|---|\n")
        for k, v in results.items():
            f.write(f"| {k} | **{v}** |\n")
        f.write("\n")

    return 0 if overall == "PASS" else 1

if __name__ == "__main__":
    sys.exit(main())
