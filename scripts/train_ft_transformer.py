import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
scripts/train_ft_transformer.py
===============================
FT-TRANSFORMER BASELINE TRAINING (ISOLATED PROCESS)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results.result_engine import compute_canonical_metrics
from src.models.ft_transformer import FTTransformerExpert

EXPERIMENT_ID = "CAHTDNET_FINAL_V001"
DATASET_NAME = "CICIoT2023+Edge-IIoTset+SyntheticFHIR"
SPLIT_NAME = "locked_test"

def main():
    print("--- Training Baseline: FT-Transformer (PyTorch Isolated Process) ---")
    data = np.load("data/processed/processed_arrays.npz")
    X_train_num, X_train_cat = data["X_train_num"], data["X_train_cat"]
    y_train = data["y_train"]

    X_test_num, X_test_cat = data["X_test_num"], data["X_test_cat"]
    y_test = data["y_test"]

    test_ids_df = pd.read_csv("splits/test_ids.csv")
    sample_ids = test_ids_df["sample_id"].values

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  FT-Transformer Device: {device}")

    ft_model = FTTransformerExpert(
        num_numerical=13,
        num_categories=3,
        cardinalities=[16, 20, 16],
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.10,
        num_classes=2
    ).to(device)

    counts = np.bincount(y_train)
    weights = len(y_train) / (len(counts) * counts.astype(float))
    cw_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    ft_criterion = nn.CrossEntropyLoss(weight=cw_tensor)
    ft_optimizer = torch.optim.AdamW(ft_model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_ds = TensorDataset(
        torch.tensor(X_train_num, dtype=torch.float32),
        torch.tensor(X_train_cat, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    t0_ft = time.time()
    ft_model.train()
    for ep in range(5):
        for bx_num, bx_cat, by in train_loader:
            bx_num, bx_cat, by = bx_num.to(device), bx_cat.to(device), by.to(device)
            ft_optimizer.zero_grad()
            logits, _ = ft_model(bx_num, bx_cat)
            loss = ft_criterion(logits, by)
            loss.backward()
            ft_optimizer.step()
    ft_train_time = time.time() - t0_ft

    # Test evaluation
    ft_model.eval()
    test_ds = TensorDataset(
        torch.tensor(X_test_num, dtype=torch.float32),
        torch.tensor(X_test_cat, dtype=torch.long)
    )
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    ft_probs = []
    t_inf_ft = time.time()
    with torch.no_grad():
        for bx_num, bx_cat in test_loader:
            bx_num, bx_cat = bx_num.to(device), bx_cat.to(device)
            logits, _ = ft_model(bx_num, bx_cat)
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            ft_probs.append(p)
    ft_inf_time = time.time() - t_inf_ft
    y_prob_ft = np.concatenate(ft_probs, axis=0)
    y_pred_ft = np.argmax(y_prob_ft, axis=1)

    os.makedirs("experiments/models/baselines", exist_ok=True)
    os.makedirs("models/baselines", exist_ok=True)
    torch.save(ft_model.state_dict(), f"experiments/models/baselines/FT-Transformer.pt")
    torch.save(ft_model.state_dict(), f"models/baselines/FT-Transformer.pt")
    ft_size_kb = os.path.getsize(f"models/baselines/FT-Transformer.pt") / 1024.0

    pred_df_ft = pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": y_test,
        "predicted_label": y_pred_ft,
        "probability_class_0": y_prob_ft[:, 0],
        "probability_class_1": y_prob_ft[:, 1],
        "confidence": np.max(y_prob_ft, axis=1)
    })
    pred_df_ft.to_parquet(f"experiments/predictions/{EXPERIMENT_ID}_FT-Transformer_test_predictions.parquet", index=False)
    pred_df_ft.to_csv(f"experiments/predictions/{EXPERIMENT_ID}_FT-Transformer_test_predictions.csv", index=False)

    meta_ft = {
        "experiment_id": EXPERIMENT_ID,
        "model": "FT-Transformer",
        "model_type": "baseline",
        "dataset": DATASET_NAME,
        "split": SPLIT_NAME,
        "train_time_s": float(round(ft_train_time, 3)),
        "inf_time_s": float(round(ft_inf_time, 4)),
        "latency_ms_per_sample": float(round((ft_inf_time * 1000.0) / len(y_test), 4)),
        "model_size_kb": float(round(ft_size_kb, 2))
    }
    metrics_ft = compute_canonical_metrics(
        y_true=y_test,
        y_pred=y_pred_ft,
        probabilities=y_prob_ft,
        metadata=meta_ft
    )
    with open(f"experiments/metrics/{EXPERIMENT_ID}_FT-Transformer_metrics.json", "w") as f:
        json.dump(metrics_ft, f, indent=2)

    print(f"  FT-Transformer | Acc: {metrics_ft['accuracy']*100:.2f}% | Macro-F1: {metrics_ft['macro_f1']*100:.2f}% | Macro-Recall: {metrics_ft['macro_recall']*100:.2f}% | FPR: {metrics_ft['fpr']*100:.3f}% | FNR: {metrics_ft['fnr']*100:.3f}%")

if __name__ == "__main__":
    main()
