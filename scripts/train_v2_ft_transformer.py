"""
scripts/train_v2_ft_transformer.py
==================================
ISOLATED PROCESS FOR FT-TRANSFORMER TRAINING (CAHTDNET_V2_001)
Runs in an isolated process to eliminate OpenMP conflicts on macOS.
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
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.ft_transformer import FTTransformerExpert
from results.metric_engine import compute_canonical_metrics

EXPERIMENT_ID = "CAHTDNET_V2_001"
BASE_DIR = f"experiments/{EXPERIMENT_ID}"

def main():
    print("\n--- Training FT-Transformer (Isolated Subprocess) ---")
    data = np.load(f"{BASE_DIR}/data/processed_arrays.npz")
    X_train_num, X_train_cat, y_train = data["X_train_num"], data["X_train_cat"], data["y_train"]
    X_test_num, X_test_cat, y_test = data["X_test_num"], data["X_test_cat"], data["y_test"]

    test_ids = pd.read_csv(f"{BASE_DIR}/splits/test_ids.csv")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  FT-Transformer training device: {device}")

    # Subsample for fast, stable convergence
    np.random.seed(42)
    sample_size = min(50000, len(X_train_num))
    idx = np.random.choice(len(X_train_num), size=sample_size, replace=False)

    model = FTTransformerExpert(
        num_numerical=X_train_num.shape[1],
        num_categories=X_train_cat.shape[1],
        cardinalities=[10, 15, 10],
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1
    ).to(device)

    train_ds = TensorDataset(
        torch.tensor(X_train_num[idx], dtype=torch.float32),
        torch.tensor(X_train_cat[idx], dtype=torch.long),
        torch.tensor(y_train[idx], dtype=torch.long)
    )
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        for bx_num, bx_cat, by in loader:
            bx_num, bx_cat, by = bx_num.to(device), bx_cat.to(device), by.to(device)
            optimizer.zero_grad()
            logits, _ = model(bx_num, bx_cat)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    model.eval()
    t0_inf = time.time()
    with torch.no_grad():
        test_ds = TensorDataset(
            torch.tensor(X_test_num, dtype=torch.float32),
            torch.tensor(X_test_cat, dtype=torch.long)
        )
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
        probs_list = []
        for bx_num, bx_cat in test_loader:
            bx_num, bx_cat = bx_num.to(device), bx_cat.to(device)
            logits, _ = model(bx_num, bx_cat)
            p = torch.softmax(logits, dim=-1)
            probs_list.append(p.cpu().numpy())
        probs = np.vstack(probs_list)
    t_inf = time.time() - t0_inf

    # Save predictions
    pos_probs = probs[:, 1]
    preds = (pos_probs >= 0.50).astype(int)

    pred_df = pd.DataFrame({
        "sample_id": test_ids["sample_id"],
        "true_label": y_test,
        "predicted_label": preds,
        "probability_class_0": probs[:, 0],
        "probability_class_1": pos_probs,
        "confidence": np.max(probs, axis=1),
        "threshold": 0.50,
        "model": "FT-Transformer",
        "split": "locked_test",
        "experiment_id": EXPERIMENT_ID
    })

    pred_df.to_parquet(f"{BASE_DIR}/predictions/{EXPERIMENT_ID}_FT-Transformer_test_predictions.parquet", index=False)
    pred_df.to_csv(f"{BASE_DIR}/predictions/{EXPERIMENT_ID}_FT-Transformer_test_predictions.csv", index=False)

    # Compute metrics
    metrics = compute_canonical_metrics(
        y_true=y_test,
        y_pred=preds,
        probabilities=probs,
        threshold=0.50,
        metadata={
            "experiment_id": EXPERIMENT_ID,
            "model": "FT-Transformer",
            "split": "locked_test",
            "inference_time_s": float(round(t_inf, 4)),
            "latency_ms_per_sample": float(round((t_inf / len(y_test)) * 1000, 4))
        }
    )

    with open(f"{BASE_DIR}/metrics/{EXPERIMENT_ID}_FT-Transformer_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save(model.state_dict(), f"{BASE_DIR}/models/FT-Transformer.pt")
    print(f"  [FT-Transformer] Macro-F1: {metrics['macro_f1']:.4f} | Acc: {metrics['accuracy']:.4f} | FPR: {metrics['fpr']:.4f} | FNR: {metrics['fnr']:.4f}")

if __name__ == "__main__":
    main()
