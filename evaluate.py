"""
evaluate.py
===========
Canonical top-level evaluation entry point on the Locked Test partition.
Evaluates CA-HTDNet V2, records raw predictions, and produces canonical metrics.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.ca_htdnet_v2 import CAHTDNetV2
from results.metric_engine import compute_canonical_metrics

EXPERIMENT_ID = "CAHTDNET_V2_001"
BASE_DIR = f"experiments/{EXPERIMENT_ID}"

def main():
    print("=" * 70)
    print("CANONICAL MODEL EVALUATION ON LOCKED TEST SET (CAHTDNET_V2_001)")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    data = np.load(f"{BASE_DIR}/data/processed_arrays.npz")
    X_test_num = data["X_test_num"]
    X_test_cat = data["X_test_cat"]
    y_test = data["y_test"]

    test_ids = pd.read_csv(f"{BASE_DIR}/splits/test_ids.csv")

    model_path = f"{BASE_DIR}/models/{EXPERIMENT_ID}.pt"
    if not os.path.exists(model_path):
        model_path = "models/proposed/ca_htdnet_v2.pt"

    model = CAHTDNetV2(
        num_numerical=X_test_num.shape[1],
        cat_cardinalities=[10, 15, 10],
        d_model=128
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_ds = TensorDataset(
        torch.tensor(X_test_num, dtype=torch.float32),
        torch.tensor(X_test_cat, dtype=torch.long)
    )
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    t0 = time.time()
    probs_list = []
    risk_list = []
    with torch.no_grad():
        for bx_num, bx_cat in test_loader:
            bx_num, bx_cat = bx_num.to(device), bx_cat.to(device)
            out = model(bx_num, bx_cat)
            probs_list.append(out["probabilities"].cpu().numpy())
            risk_list.append(out["risk_score"].cpu().numpy())
    t_inf = time.time() - t0

    probs = np.vstack(probs_list)
    risk = np.vstack(risk_list).ravel()
    preds = (probs[:, 1] >= 0.50).astype(int)

    metrics = compute_canonical_metrics(
        y_true=y_test,
        y_pred=preds,
        probabilities=probs,
        threshold=0.50,
        metadata={
            "experiment_id": EXPERIMENT_ID,
            "model": "CA-HTDNet-V2",
            "split": "locked_test",
            "inference_time_s": float(round(t_inf, 4)),
            "latency_ms_per_sample": float(round((t_inf / len(y_test)) * 1000, 4))
        }
    )

    # Save raw predictions
    pred_df = pd.DataFrame({
        "sample_id": test_ids["sample_id"],
        "true_label": y_test,
        "predicted_label": preds,
        "probability_class_0": probs[:, 0],
        "probability_class_1": probs[:, 1],
        "confidence": np.max(probs, axis=1),
        "risk_score": risk,
        "threshold": 0.50,
        "model": "CA-HTDNet-V2",
        "split": "locked_test",
        "experiment_id": EXPERIMENT_ID
    })

    pred_df.to_parquet(f"{BASE_DIR}/predictions/{EXPERIMENT_ID}_test_predictions.parquet", index=False)
    pred_df.to_csv(f"{BASE_DIR}/predictions/{EXPERIMENT_ID}_test_predictions.csv", index=False)

    with open(f"{BASE_DIR}/metrics/{EXPERIMENT_ID}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nEVALUATION RESULTS (LOCKED TEST SET):")
    print(f"  Accuracy:        {metrics['accuracy']*100:.2f}%")
    print(f"  Macro-Precision: {metrics['macro_precision']*100:.2f}%")
    print(f"  Macro-Recall:    {metrics['macro_recall']*100:.2f}%")
    print(f"  Macro-F1:        {metrics['macro_f1']*100:.2f}%")
    print(f"  ROC-AUC:         {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:          {metrics['pr_auc']:.4f}")
    print(f"  FPR:             {metrics['fpr']*100:.2f}%")
    print(f"  FNR:             {metrics['fnr']*100:.2f}%")
    print(f"  MCC:             {metrics['mcc']:.4f}")

if __name__ == "__main__":
    main()
