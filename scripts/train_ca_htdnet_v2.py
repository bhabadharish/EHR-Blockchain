"""
scripts/train_ca_htdnet_v2.py
=============================
TRAINING AND RIGOROUS VALIDATION OF CA-HTDNet V2 (CAHTDNET_V2_001)

Methodology:
1. Model training conducted strictly on Train partition (119,000 samples).
2. Checkpoint selection and early stopping monitored strictly on Validation partition (25,500 samples).
3. Test set remains locked until the canonical model configuration is fixed.
4. Final evaluation executed exactly once on Locked Test partition (25,500 samples).
5. All metrics computed using the single source of truth: results/metric_engine.py.
6. Cross-seed stability benchmark across seeds [42, 123, 2024, 3407, 777].
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.ca_htdnet_v2 import CAHTDNetV2
from src.models.loss_v2 import AsymmetricClassBalancedMarginLoss
from results.metric_engine import compute_canonical_metrics

EXPERIMENT_ID = "CAHTDNET_V2_001"
BASE_DIR = f"experiments/{EXPERIMENT_ID}"

def train_single_model(
    seed: int = 42,
    max_epochs: int = 25,
    batch_size: int = 256,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    focal_gamma: float = 2.0,
    margin: float = 0.15,
    patience: int = 8,
    save_final: bool = True
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"TRAINING CA-HTDNet V2 | SEED: {seed} | DEVICE: {device}")
    print(f"========================================================")

    # 1. Load Data
    data = np.load(f"{BASE_DIR}/data/processed_arrays.npz")
    X_train_num, X_train_cat, y_train = data["X_train_num"], data["X_train_cat"], data["y_train"]
    X_val_num, X_val_cat, y_val = data["X_val_num"], data["X_val_cat"], data["y_val"]
    X_test_num, X_test_cat, y_test = data["X_test_num"], data["X_test_cat"], data["y_test"]

    # 2. Compute Class Weights on Train Only
    class_counts = np.bincount(y_train)
    total_train = len(y_train)
    class_weights = torch.tensor([
        total_train / (2.0 * class_counts[0]),
        total_train / (2.0 * class_counts[1])
    ], dtype=torch.float32).to(device)
    print(f"  Train class counts: Normal={class_counts[0]}, Attack={class_counts[1]}")
    print(f"  Computed class weights: Normal={class_weights[0]:.4f}, Attack={class_weights[1]:.4f}")

    # 3. DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train_num, dtype=torch.float32),
        torch.tensor(X_train_cat, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_num, dtype=torch.float32),
        torch.tensor(X_val_cat, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    # 4. Instantiate Model & Loss
    model = CAHTDNetV2(
        num_numerical=X_train_num.shape[1],
        cat_cardinalities=[10, 15, 10],
        d_model=128,
        embedding_dim=16,
        n_heads=4,
        num_transformer_layers=2,
        tcn_kernel_size=3,
        representation_dim=64,
        dropout=0.15
    ).to(device)

    criterion = AsymmetricClassBalancedMarginLoss(
        class_weights=class_weights,
        gamma=focal_gamma,
        margin=margin,
        label_smoothing=0.02,
        aux_risk_weight=0.20
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    # 5. Training Loop
    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_state_dict = None
    best_val_metrics = None
    epochs_no_improve = 0

    checkpoint_path = f"{BASE_DIR}/checkpoints/temp_best_seed_{seed}.pt"

    t0_train = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_accum = 0.0

        for bx_num, bx_cat, by in train_loader:
            bx_num, bx_cat, by = bx_num.to(device), bx_cat.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx_num, bx_cat)
            loss_dict = criterion(out["logits"], by, out["risk_score"])
            loss = loss_dict["total_loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss_accum += loss.item() * len(by)

        scheduler.step()
        train_loss = train_loss_accum / len(y_train)

        # Validation Evaluation
        model.eval()
        val_probs_list = []
        with torch.no_grad():
            for bx_num, bx_cat, _ in val_loader:
                bx_num, bx_cat = bx_num.to(device), bx_cat.to(device)
                out = model(bx_num, bx_cat)
                val_probs_list.append(out["probabilities"].cpu().numpy())
        val_probs = np.vstack(val_probs_list)
        val_preds = (val_probs[:, 1] >= 0.50).astype(int)

        val_metrics = compute_canonical_metrics(
            y_true=y_val,
            y_pred=val_preds,
            probabilities=val_probs,
            threshold=0.50
        )

        val_macro_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]

        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f} | Val Macro-F1: {val_macro_f1:.4f} | Val Acc: {val_acc:.4f} | Val FPR: {val_metrics['fpr']:.4f} | Val FNR: {val_metrics['fnr']:.4f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = val_metrics
            epochs_no_improve = 0
            torch.save(best_state_dict, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with Val Macro-F1: {best_val_macro_f1:.4f}")
                break

    train_time = time.time() - t0_train
    print(f"  Training completed in {train_time:.1f}s. Restoring best weights from epoch {best_epoch}...")
    model.load_state_dict(best_state_dict)
    model.to(device)

    # 6. Locked Test Set Evaluation
    print("\n--- Evaluating on Locked Test Partition (Single Canonical Evaluation) ---")
    test_ds = TensorDataset(
        torch.tensor(X_test_num, dtype=torch.float32),
        torch.tensor(X_test_cat, dtype=torch.long)
    )
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    model.eval()
    t0_inf = time.time()
    test_probs_list = []
    test_risk_list = []
    with torch.no_grad():
        for bx_num, bx_cat in test_loader:
            bx_num, bx_cat = bx_num.to(device), bx_cat.to(device)
            out = model(bx_num, bx_cat)
            test_probs_list.append(out["probabilities"].cpu().numpy())
            test_risk_list.append(out["risk_score"].cpu().numpy())
    t_inf = time.time() - t0_inf

    test_probs = np.vstack(test_probs_list)
    test_risk = np.vstack(test_risk_list).ravel()
    test_preds = (test_probs[:, 1] >= 0.50).astype(int)

    test_ids = pd.read_csv(f"{BASE_DIR}/splits/test_ids.csv")

    test_metrics = compute_canonical_metrics(
        y_true=y_test,
        y_pred=test_preds,
        probabilities=test_probs,
        threshold=0.50,
        metadata={
            "experiment_id": EXPERIMENT_ID,
            "model": "CA-HTDNet-V2",
            "model_version": "2.0.0",
            "seed": seed,
            "best_epoch": best_epoch,
            "training_time_s": float(round(train_time, 2)),
            "inference_time_s": float(round(t_inf, 4)),
            "latency_ms_per_sample": float(round((t_inf / len(y_test)) * 1000, 4)),
            "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_size_kb": float(round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1024, 2))
        }
    )

    print(f"  [LOCKED TEST] Accuracy:        {test_metrics['accuracy']:.4f}")
    print(f"  [LOCKED TEST] Macro Precision: {test_metrics['macro_precision']:.4f}")
    print(f"  [LOCKED TEST] Macro Recall:    {test_metrics['macro_recall']:.4f}")
    print(f"  [LOCKED TEST] Macro F1:        {test_metrics['macro_f1']:.4f}")
    print(f"  [LOCKED TEST] ROC-AUC:         {test_metrics['roc_auc']:.4f}")
    print(f"  [LOCKED TEST] PR-AUC:          {test_metrics['pr_auc']:.4f}")
    print(f"  [LOCKED TEST] FPR:             {test_metrics['fpr']:.4f}")
    print(f"  [LOCKED TEST] FNR:             {test_metrics['fnr']:.4f}")
    print(f"  [LOCKED TEST] MCC:             {test_metrics['mcc']:.4f}")

    if save_final:
        # Save canonical model
        model_pt_path = f"{BASE_DIR}/models/{EXPERIMENT_ID}.pt"
        torch.save(best_state_dict, model_pt_path)
        torch.save(best_state_dict, "models/proposed/ca_htdnet_v2.pt")
        print(f"  Saved canonical model to {model_pt_path}")

        # Save raw test predictions
        pred_df = pd.DataFrame({
            "sample_id": test_ids["sample_id"],
            "true_label": y_test,
            "predicted_label": test_preds,
            "probability_class_0": test_probs[:, 0],
            "probability_class_1": test_probs[:, 1],
            "confidence": np.max(test_probs, axis=1),
            "risk_score": test_risk,
            "threshold": 0.50,
            "model": "CA-HTDNet-V2",
            "split": "locked_test",
            "experiment_id": EXPERIMENT_ID
        })
        pred_df.to_parquet(f"{BASE_DIR}/predictions/{EXPERIMENT_ID}_test_predictions.parquet", index=False)
        pred_df.to_csv(f"{BASE_DIR}/predictions/{EXPERIMENT_ID}_test_predictions.csv", index=False)
        pred_df.to_parquet("predictions/CAHTDNET_V2_001_test_predictions.parquet", index=False)

        # Save metrics
        with open(f"{BASE_DIR}/metrics/{EXPERIMENT_ID}_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        with open("results/CAHTDNET_V2_001_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

        # Save validation predictions for threshold & calibration optimization
        val_ids = pd.read_csv(f"{BASE_DIR}/splits/validation_ids.csv")
        val_pred_df = pd.DataFrame({
            "sample_id": val_ids["sample_id"],
            "true_label": y_val,
            "probability_class_1": val_probs[:, 1]
        })
        val_pred_df.to_parquet(f"{BASE_DIR}/predictions/{EXPERIMENT_ID}_val_predictions.parquet", index=False)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "val_macro_f1": best_val_macro_f1,
        "test_metrics": test_metrics
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,123,2024,3407,777", help="Comma-separated random seeds")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--skip_multi_seed", action="store_true", help="Run only canonical seed 42")
    args = parser.parse_args()

    os.makedirs(f"{BASE_DIR}/models", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/checkpoints", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/predictions", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/metrics", exist_ok=True)
    os.makedirs("models/proposed", exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    canonical_seed = seeds[0]

    # 1. Train Canonical Model (Seed 42)
    print("================================================================")
    print("PHASE 1: TRAINING CANONICAL CA-HTDNet V2 MODEL (SEED 42)")
    print("================================================================")
    canonical_res = train_single_model(
        seed=canonical_seed,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        focal_gamma=args.focal_gamma,
        margin=args.margin,
        save_final=True
    )

    # 2. Multi-Seed Stability Benchmark (Section 22)
    if not args.skip_multi_seed and len(seeds) > 1:
        print("\n================================================================")
        print("PHASE 2: REPEATED-SEED STABILITY BENCHMARK (SEEDS: {})".format(seeds))
        print("================================================================")
        seed_results = [canonical_res]
        for s in seeds[1:]:
            res = train_single_model(
                seed=s,
                max_epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                focal_gamma=args.focal_gamma,
                margin=args.margin,
                save_final=False
            )
            seed_results.append(res)

        # Compile stability stats
        macro_f1s = [r["test_metrics"]["macro_f1"] for r in seed_results]
        precisions = [r["test_metrics"]["macro_precision"] for r in seed_results]
        recalls = [r["test_metrics"]["macro_recall"] for r in seed_results]
        accuracies = [r["test_metrics"]["accuracy"] for r in seed_results]
        fprs = [r["test_metrics"]["fpr"] for r in seed_results]
        fnrs = [r["test_metrics"]["fnr"] for r in seed_results]

        stability_report = {
            "experiment_id": EXPERIMENT_ID,
            "seeds_evaluated": seeds,
            "metrics": {
                "macro_f1": {
                    "mean": float(np.mean(macro_f1s)),
                    "std": float(np.std(macro_f1s)),
                    "min": float(np.min(macro_f1s)),
                    "max": float(np.max(macro_f1s)),
                    "values": macro_f1s
                },
                "macro_precision": {
                    "mean": float(np.mean(precisions)),
                    "std": float(np.std(precisions)),
                    "min": float(np.min(precisions)),
                    "max": float(np.max(precisions))
                },
                "macro_recall": {
                    "mean": float(np.mean(recalls)),
                    "std": float(np.std(recalls)),
                    "min": float(np.min(recalls)),
                    "max": float(np.max(recalls))
                },
                "accuracy": {
                    "mean": float(np.mean(accuracies)),
                    "std": float(np.std(accuracies)),
                    "min": float(np.min(accuracies)),
                    "max": float(np.max(accuracies))
                },
                "fpr": {
                    "mean": float(np.mean(fprs)),
                    "std": float(np.std(fprs)),
                    "min": float(np.min(fprs)),
                    "max": float(np.max(fprs))
                },
                "fnr": {
                    "mean": float(np.mean(fnrs)),
                    "std": float(np.std(fnrs)),
                    "min": float(np.min(fnrs)),
                    "max": float(np.max(fnrs))
                }
            }
        }

        with open(f"{BASE_DIR}/metrics/stability_benchmark.json", "w") as f:
            json.dump(stability_report, f, indent=2)

        print("\n" + "=" * 70)
        print("REPEATED-SEED STABILITY SUMMARY:")
        print("=" * 70)
        print(f"  Macro-F1:        {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f} (Min: {np.min(macro_f1s):.4f}, Max: {np.max(macro_f1s):.4f})")
        print(f"  Macro-Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
        print(f"  Macro-Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"  Accuracy:        {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"  FPR:             {np.mean(fprs):.4f} ± {np.std(fprs):.4f}")
        print(f"  FNR:             {np.mean(fnrs):.4f} ± {np.std(fnrs):.4f}")

if __name__ == "__main__":
    main()
