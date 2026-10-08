import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
scripts/evaluate_final_model.py
===============================
FINAL CANONICAL EVALUATION OF CA-HTDNet ON THE LOCKED TEST SET

Strict Methodology:
- Single execution on LOCKED TEST SET.
- Outputs raw predictions file with sample IDs, labels, and probabilities.
- Generates official metrics using results/result_engine.py.
- Exports confusion matrix, ROC curve data, and PR curve data.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, precision_recall_curve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.ca_htdnet import CA_HTDNet
from results.result_engine import compute_canonical_metrics

EXPERIMENT_ID = "CAHTDNET_FINAL_V001"
DATASET_NAME = "CICIoT2023+Edge-IIoTset+SyntheticFHIR"
SPLIT_NAME = "locked_test"

def main():
    print("=" * 70)
    print("STAGE 5: FINAL LOCKED TEST EVALUATION OF CA-HTDNet")
    print("=" * 70)

    os.makedirs("experiments/predictions", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("experiments/metrics", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("experiments/curves", exist_ok=True)
    os.makedirs("curves", exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load canonical test arrays and IDs
    data = np.load("data/processed/processed_arrays.npz")
    X_test_num = data["X_test_num"]
    X_test_cat = data["X_test_cat"]
    y_test = data["y_test"]

    test_ids_df = pd.read_csv("splits/test_ids.csv")
    sample_ids = test_ids_df["sample_id"].values

    print(f"Loaded Locked Test Set: {len(y_test)} samples")

    # 2. Instantiate and load model
    model = CA_HTDNet(
        num_numerical=13,
        num_categories=3,
        cardinalities=[16, 20, 16],
        d_model=64,
        tcn_channels=[32, 64],
        attention_heads=4,
        attention_layers=2,
        dropout=0.10
    ).to(device)

    model_path = f"experiments/models/{EXPERIMENT_ID}.pt"
    if not os.path.exists(model_path):
        model_path = f"models/{EXPERIMENT_ID}.pt"
    if not os.path.exists(model_path):
        model_path = "models/proposed/ca_htdnet.pt"

    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    model_size_kb = os.path.getsize(model_path) / 1024.0
    param_count = sum(p.numel() for p in model.parameters())

    # 3. Test Inference
    test_ds = TensorDataset(
        torch.tensor(X_test_num, dtype=torch.float32),
        torch.tensor(X_test_cat, dtype=torch.long)
    )
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    all_probs = []
    all_risks = []

    t0_inf = time.time()
    with torch.no_grad():
        for bx_num, bx_cat in test_loader:
            bx_num, bx_cat = bx_num.to(device), bx_cat.to(device)
            out = model(bx_num, bx_cat)
            p = torch.softmax(out["logits"], dim=-1).cpu().numpy()
            r = out["risk_score"].cpu().numpy()
            all_probs.append(p)
            all_risks.append(r)
    inf_time = time.time() - t0_inf

    probs = np.concatenate(all_probs, axis=0)
    risks = np.concatenate(all_risks, axis=0)
    preds = np.argmax(probs, axis=1)

    # 4. Save Raw Predictions DataFrame
    pred_df = pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": y_test,
        "predicted_label": preds,
        "probability_class_0": probs[:, 0],
        "probability_class_1": probs[:, 1],
        "confidence": np.max(probs, axis=1),
        "threat_risk_score": risks
    })

    pred_pq_1 = f"experiments/predictions/{EXPERIMENT_ID}_test_predictions.parquet"
    pred_csv_1 = f"experiments/predictions/{EXPERIMENT_ID}_test_predictions.csv"
    pred_pq_2 = f"predictions/{EXPERIMENT_ID}_test_predictions.parquet"
    pred_csv_2 = f"predictions/{EXPERIMENT_ID}_test_predictions.csv"

    for path in [pred_pq_1, pred_pq_2]:
        pred_df.to_parquet(path, index=False)
    for path in [pred_csv_1, pred_csv_2]:
        pred_df.to_csv(path, index=False)

    print(f"Raw test predictions saved to {pred_pq_1} and {pred_csv_1}")

    # 5. Compute Canonical Metrics via Result Engine
    meta = {
        "experiment_id": EXPERIMENT_ID,
        "model": "CA-HTDNet",
        "model_version": "1.0.0",
        "dataset": DATASET_NAME,
        "split": SPLIT_NAME,
        "inference_time_s": float(round(inf_time, 4)),
        "latency_ms_per_sample": float(round((inf_time * 1000.0) / len(y_test), 4)),
        "model_size_kb": float(round(model_size_kb, 2)),
        "parameter_count": param_count
    }

    metrics = compute_canonical_metrics(
        y_true=y_test,
        y_pred=preds,
        probabilities=probs,
        metadata=meta
    )

    metrics_path_1 = f"experiments/metrics/{EXPERIMENT_ID}_metrics.json"
    metrics_path_2 = f"results/{EXPERIMENT_ID}_metrics.json"
    metrics_path_3 = f"results/final_results.json"

    with open(metrics_path_1, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(metrics_path_2, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(metrics_path_3, "w") as f:
        json.dump(metrics, f, indent=2)

    # 6. Save Confusion Matrix and Curve Data
    cm_dict = metrics["confusion_matrix"]
    cm_path_1 = f"experiments/curves/{EXPERIMENT_ID}_confusion_matrix.json"
    cm_path_2 = f"curves/{EXPERIMENT_ID}_confusion_matrix.json"
    for p in [cm_path_1, cm_path_2]:
        with open(p, "w") as f:
            json.dump(cm_dict, f, indent=2)

    # ROC Curve points
    fpr_arr, tpr_arr, roc_thresh = roc_curve(y_test, probs[:, 1])
    roc_data = {
        "fpr": fpr_arr.tolist(),
        "tpr": tpr_arr.tolist(),
        "roc_auc": metrics["roc_auc"]
    }
    with open(f"experiments/curves/{EXPERIMENT_ID}_roc_curve.json", "w") as f:
        json.dump(roc_data, f)
    with open(f"curves/{EXPERIMENT_ID}_roc_curve.json", "w") as f:
        json.dump(roc_data, f)

    # PR Curve points
    prec_arr, rec_arr, pr_thresh = precision_recall_curve(y_test, probs[:, 1])
    pr_data = {
        "precision": prec_arr.tolist(),
        "recall": rec_arr.tolist(),
        "pr_auc": metrics["pr_auc"]
    }
    with open(f"experiments/curves/{EXPERIMENT_ID}_pr_curve.json", "w") as f:
        json.dump(pr_data, f)
    with open(f"curves/{EXPERIMENT_ID}_pr_curve.json", "w") as f:
        json.dump(pr_data, f)

    # Print final verified report
    print("\n" + "=" * 70)
    print("PROPOSED MODEL CA-HTDNet LOCKED TEST RESULTS:")
    print("=" * 70)
    print(f"Accuracy:         {metrics['accuracy']*100:.2f}%")
    print(f"Macro Precision:  {metrics['macro_precision']*100:.2f}%")
    print(f"Macro Recall:     {metrics['macro_recall']*100:.2f}%")
    print(f"Macro F1:         {metrics['macro_f1']*100:.2f}%")
    print(f"Weighted F1:      {metrics['weighted_f1']*100:.2f}%")
    print(f"ROC-AUC:          {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:           {metrics['pr_auc']:.4f}")
    print(f"FPR:              {metrics['fpr']*100:.3f}%")
    print(f"FNR:              {metrics['fnr']*100:.3f}%")
    print(f"MCC:              {metrics['mcc']:.4f}")
    print(f"Inference Latency:{meta['latency_ms_per_sample']:.4f} ms/sample")
    print("=" * 70)

if __name__ == "__main__":
    main()
