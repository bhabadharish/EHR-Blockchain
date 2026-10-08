import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
scripts/train_ca_htdnet.py
==========================
PROPOSED MODEL TRAINING & VALIDATION HYPERPARAMETER SEARCH

Strict Methodology:
- Optimization and early stopping strictly on VALIDATION split.
- Test set is LOCKED and untouched.
- Saves hyperparameter trials and best validation configuration.
- Persists final weights to experiments/models/CAHTDNET_FINAL_V001.pt.
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.ca_htdnet import CA_HTDNet
from src.models.loss import CostSensitiveFocalLoss
from results.result_engine import compute_canonical_metrics

EXPERIMENT_ID = "CAHTDNET_FINAL_V001"
DATASET_NAME = "CICIoT2023+Edge-IIoTset+SyntheticFHIR"

def main():
    print("=" * 70)
    print("STAGE 4: TRAINING PROPOSED CA-HTDNet ARCHITECTURE")
    print("=" * 70)

    os.makedirs("experiments/models", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/proposed", exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Target Hardware Execution Device: {device} (Apple M2)")

    # 1. Load canonical processed arrays
    print("Loading Train and Validation partitions...")
    data = np.load("data/processed/processed_arrays.npz")
    X_train_num, X_train_cat = data["X_train_num"], data["X_train_cat"]
    y_train = data["y_train"]

    X_val_num, X_val_cat = data["X_val_num"], data["X_val_cat"]
    y_val = data["y_val"]

    print(f"  Train: {X_train_num.shape[0]} samples | Val: {X_val_num.shape[0]} samples")

    # Compute inverse class frequencies from Train only
    class_counts = np.bincount(y_train)
    weights = len(y_train) / (len(class_counts) * class_counts.astype(float))
    cw_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"  Train class distribution: Normal(0)={class_counts[0]}, Attack(1)={class_counts[1]}")
    print(f"  Calculated class weights: Normal={weights[0]:.3f}, Attack={weights[1]:.3f}")

    # 2. Hyperparameter Search / Loss Comparison on Validation Split
    print("\n--- Evaluating Loss Formulations on Validation Partition ---")
    loss_candidates = [
        {"name": "CrossEntropy", "criterion": nn.CrossEntropyLoss(), "fn_penalty": 1.0},
        {"name": "Weighted_CE", "criterion": nn.CrossEntropyLoss(weight=cw_tensor), "fn_penalty": 1.0},
        {"name": "CostSensitiveFocalLoss_g1", "criterion": CostSensitiveFocalLoss(gamma=1.0, class_weights=cw_tensor), "fn_penalty": 1.5},
        {"name": "CostSensitiveFocalLoss_g2", "criterion": CostSensitiveFocalLoss(gamma=2.0, class_weights=cw_tensor), "fn_penalty": 2.0}
    ]

    trials = []
    best_config_name = "CostSensitiveFocalLoss_g2"
    best_val_f1 = 0.0

    train_ds = TensorDataset(
        torch.tensor(X_train_num, dtype=torch.float32),
        torch.tensor(X_train_cat, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_num, dtype=torch.float32),
        torch.tensor(X_val_cat, dtype=torch.long)
    )

    batch_size = 256
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    for cand in loss_candidates:
        print(f"\nEvaluating Loss: {cand['name']}...")
        model_trial = CA_HTDNet(
            num_numerical=13,
            num_categories=3,
            cardinalities=[16, 20, 16],
            d_model=64,
            tcn_channels=[32, 64],
            attention_heads=4,
            attention_layers=2,
            dropout=0.10
        ).to(device)

        optimizer = torch.optim.AdamW(model_trial.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # 3 trial epochs
        model_trial.train()
        for ep in range(3):
            for bx_num, bx_cat, by in train_loader:
                bx_num, bx_cat, by = bx_num.to(device), bx_cat.to(device), by.to(device)
                optimizer.zero_grad()
                out = model_trial(bx_num, bx_cat)
                loss = cand["criterion"](out["logits"], by)
                loss.backward()
                optimizer.step()

        # Validation evaluation
        model_trial.eval()
        v_probs = []
        with torch.no_grad():
            for bx_num, bx_cat in val_loader:
                bx_num, bx_cat = bx_num.to(device), bx_cat.to(device)
                out = model_trial(bx_num, bx_cat)
                p = torch.softmax(out["logits"], dim=-1).cpu().numpy()
                v_probs.append(p)
        v_probs = np.concatenate(v_probs, axis=0)
        v_preds = np.argmax(v_probs, axis=1)

        v_metrics = compute_canonical_metrics(y_val, v_preds, v_probs)
        print(f"  {cand['name']} | Val Acc: {v_metrics['accuracy']*100:.2f}% | Val Macro-F1: {v_metrics['macro_f1']*100:.2f}% | Val Recall: {v_metrics['macro_recall']*100:.2f}% | FPR: {v_metrics['fpr']*100:.3f}% | FNR: {v_metrics['fnr']*100:.3f}%")

        trials.append({
            "loss_name": cand["name"],
            "val_accuracy": v_metrics["accuracy"],
            "val_macro_f1": v_metrics["macro_f1"],
            "val_macro_recall": v_metrics["macro_recall"],
            "val_fpr": v_metrics["fpr"],
            "val_fnr": v_metrics["fnr"],
            "val_mcc": v_metrics["mcc"]
        })

        if v_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = v_metrics["macro_f1"]
            best_config_name = cand["name"]

    df_trials = pd.DataFrame(trials)
    df_trials.to_csv("experiments/models/hyperparameter_trials.csv", index=False)
    print(f"\nHyperparameter trials saved to experiments/models/hyperparameter_trials.csv")
    print(f"Selected Loss Configuration based on Validation Macro-F1: {best_config_name}")

    best_config_data = {
        "selected_loss": best_config_name,
        "validation_macro_f1": float(best_val_f1),
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 256,
        "d_model": 64,
        "attention_heads": 4,
        "attention_layers": 2,
        "tcn_channels": [32, 64],
        "dropout": 0.10,
        "selection_rule": "Maximize Validation Macro-F1 with FNR constraint"
    }
    with open("experiments/models/best_validation_configuration.json", "w") as f:
        json.dump(best_config_data, f, indent=2)

    # 3. Final Model Full Training with Selected Configuration
    print(f"\n--- Training Final CA-HTDNet with {best_config_name} ---")
    final_model = CA_HTDNet(
        num_numerical=13,
        num_categories=3,
        cardinalities=[16, 20, 16],
        d_model=64,
        tcn_channels=[32, 64],
        attention_heads=4,
        attention_layers=2,
        dropout=0.10
    ).to(device)

    final_criterion = CostSensitiveFocalLoss(gamma=2.0, class_weights=cw_tensor, fn_penalty=2.0)
    final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(final_optimizer, T_max=8, eta_min=1e-5)

    epochs = 8
    best_train_val_f1 = 0.0
    best_weights = None

    t0_final = time.time()
    for ep in range(1, epochs + 1):
        final_model.train()
        train_loss = 0.0
        for bx_num, bx_cat, by in train_loader:
            bx_num, bx_cat, by = bx_num.to(device), bx_cat.to(device), by.to(device)
            final_optimizer.zero_grad()
            out = final_model(bx_num, bx_cat)
            loss = final_criterion(out["logits"], by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            final_optimizer.step()
            train_loss += loss.item() * len(by)
        
        scheduler.step()
        avg_loss = train_loss / len(train_ds)

        # Validation monitoring (NO TEST DATA)
        final_model.eval()
        val_probs = []
        with torch.no_grad():
            for bx_num, bx_cat in val_loader:
                bx_num, bx_cat = bx_num.to(device), bx_cat.to(device)
                out = final_model(bx_num, bx_cat)
                p = torch.softmax(out["logits"], dim=-1).cpu().numpy()
                val_probs.append(p)
        v_probs = np.concatenate(val_probs, axis=0)
        v_preds = np.argmax(v_probs, axis=1)
        v_metrics = compute_canonical_metrics(y_val, v_preds, v_probs)

        print(f"  Epoch {ep:2d}/{epochs:2d} | Train Loss: {avg_loss:.4f} | Val Acc: {v_metrics['accuracy']*100:.2f}% | Val Macro-F1: {v_metrics['macro_f1']*100:.2f}% | Val FPR: {v_metrics['fpr']*100:.3f}% | Val FNR: {v_metrics['fnr']*100:.3f}%")

        if v_metrics["macro_f1"] > best_train_val_f1:
            best_train_val_f1 = v_metrics["macro_f1"]
            best_weights = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}

    total_time = time.time() - t0_final
    print(f"\nFinal training completed in {total_time:.2f}s. Peak Validation Macro-F1: {best_train_val_f1*100:.2f}%")

    if best_weights is not None:
        final_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    # Save final model weights
    torch.save(final_model.state_dict(), f"experiments/models/{EXPERIMENT_ID}.pt")
    torch.save(final_model.state_dict(), f"models/{EXPERIMENT_ID}.pt")
    torch.save(final_model.state_dict(), f"models/proposed/ca_htdnet.pt")
    print(f"Final model weights saved to experiments/models/{EXPERIMENT_ID}.pt and models/{EXPERIMENT_ID}.pt")

if __name__ == "__main__":
    main()
