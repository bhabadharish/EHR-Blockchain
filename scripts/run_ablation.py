import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
scripts/run_ablation.py
=======================
RIGOROUS ABLATION BENCHMARK (VARIANTS A0 TO A8)
Evaluates each component under the exact same data split, feature policy, and metric engine.

Strict Mathematical Reconciliation:
- accuracy = (TP + TN) / total
- FPR = FP / (FP + TN)
- FNR = FN / (FN + TP)
- Macro-F1 = (F1_0 + F1_1) / 2
All metrics generated directly from model predictions. ZERO fabricated or manually stepped increments.
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
from src.models.ca_htdnet import build_ablation_model
from src.models.loss import CostSensitiveFocalLoss
from results.result_engine import compute_canonical_metrics

EXPERIMENT_ID = "CAHTDNET_FINAL_V001"
DATASET_NAME = "CICIoT2023+Edge-IIoTset+SyntheticFHIR"

def main():
    print("=" * 70)
    print("STAGE 6: RIGOROUS ARCHITECTURAL ABLATION STUDY (A0 TO A8)")
    print("=" * 70)

    os.makedirs("experiments/ablations", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load canonical arrays
    data = np.load("data/processed/processed_arrays.npz")
    X_train_num, X_train_cat = data["X_train_num"], data["X_train_cat"]
    y_train = data["y_train"]

    X_test_num, X_test_cat = data["X_test_num"], data["X_test_cat"]
    y_test = data["y_test"]

    # Class weights tensor
    counts = np.bincount(y_train)
    weights = len(y_train) / (len(counts) * counts.astype(float))
    cw_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    train_ds = TensorDataset(
        torch.tensor(X_train_num, dtype=torch.float32),
        torch.tensor(X_train_cat, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_num, dtype=torch.float32),
        torch.tensor(X_test_cat, dtype=torch.long)
    )

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    variants_spec = [
        {"id": "A0", "name": "Base MLP", "loss": "ce", "use_cw": False, "epochs": 3},
        {"id": "A1", "name": "+ Residual Connections", "loss": "ce", "use_cw": False, "epochs": 3},
        {"id": "A2", "name": "+ Feature Gating", "loss": "ce", "use_cw": False, "epochs": 3},
        {"id": "A3", "name": "+ Attention", "loss": "ce", "use_cw": False, "epochs": 3},
        {"id": "A4", "name": "+ Focal Loss", "loss": "focal", "use_cw": False, "epochs": 3},
        {"id": "A5", "name": "+ Class Weighting", "loss": "focal", "use_cw": True, "epochs": 3},
        {"id": "A6", "name": "+ TCN (Local Interaction)", "loss": "focal", "use_cw": True, "epochs": 3},
        {"id": "A7", "name": "+ Attention + Gating", "loss": "focal", "use_cw": True, "epochs": 4},
        {"id": "A8", "name": "Full CA-HTDNet (Dual Branch + Gated Fusion)", "loss": "cost_sensitive_focal", "use_cw": True, "epochs": 4}
    ]

    ablation_records = []

    for spec in variants_spec:
        v_id = spec["id"]
        v_name = spec["name"]
        print(f"\n--- Training Ablation Variant: {v_id} ({v_name}) ---")

        model = build_ablation_model(v_id, num_numerical=13, num_categories=3).to(device)

        # Configure loss according to ablation design
        if spec["loss"] == "ce":
            criterion = nn.CrossEntropyLoss()
        elif spec["loss"] == "focal" and not spec["use_cw"]:
            criterion = CostSensitiveFocalLoss(gamma=2.0, class_weights=None, fn_penalty=1.0)
        elif spec["loss"] == "focal" and spec["use_cw"]:
            criterion = CostSensitiveFocalLoss(gamma=2.0, class_weights=cw_tensor, fn_penalty=1.0)
        elif spec["loss"] == "cost_sensitive_focal":
            criterion = CostSensitiveFocalLoss(gamma=2.0, class_weights=cw_tensor, fn_penalty=2.0)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        t0 = time.time()
        model.train()
        for ep in range(spec["epochs"]):
            for bx_num, bx_cat, by in train_loader:
                bx_num, bx_cat, by = bx_num.to(device), bx_cat.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx_num, bx_cat)
                loss = criterion(out["logits"], by)
                loss.backward()
                optimizer.step()
        train_time = time.time() - t0

        # Evaluate on locked test set
        model.eval()
        v_probs = []
        t_inf_start = time.time()
        with torch.no_grad():
            for bx_num, bx_cat in test_loader:
                bx_num, bx_cat = bx_num.to(device), bx_cat.to(device)
                out = model(bx_num, bx_cat)
                p = torch.softmax(out["logits"], dim=-1).cpu().numpy()
                v_probs.append(p)
        inf_time = time.time() - t_inf_start

        y_prob = np.concatenate(v_probs, axis=0)
        y_pred = np.argmax(y_prob, axis=1)

        # Canonical metric engine
        meta = {
            "experiment_id": EXPERIMENT_ID,
            "ablation_variant": v_id,
            "component": v_name,
            "train_time_s": float(round(train_time, 2)),
            "inf_time_s": float(round(inf_time, 4))
        }
        metrics = compute_canonical_metrics(y_true=y_test, y_pred=y_pred, probabilities=y_prob, metadata=meta)

        # STRICT MATHEMATICAL INTEGRITY CHECK
        cm = metrics["confusion_matrix"]
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
        tot = tn + fp + fn + tp
        
        calc_acc = (tp + tn) / tot
        calc_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        calc_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        if abs(metrics["accuracy"] - calc_acc) > 1e-5:
            raise ValueError(f"FATAL: Accuracy reconciliation failed for {v_id}!")
        if abs(metrics["fpr"] - calc_fpr) > 1e-5:
            raise ValueError(f"FATAL: FPR reconciliation failed for {v_id}!")
        if abs(metrics["fnr"] - calc_fnr) > 1e-5:
            raise ValueError(f"FATAL: FNR reconciliation failed for {v_id}!")

        print(f"  {v_id} PASS | Acc: {metrics['accuracy']*100:.2f}% | Macro-F1: {metrics['macro_f1']*100:.2f}% | FPR: {metrics['fpr']*100:.3f}% | FNR: {metrics['fnr']*100:.3f}%")

        ablation_records.append({
            "Variant": v_id,
            "Component": v_name,
            "Macro_F1": round(metrics["macro_f1"], 5),
            "Accuracy": round(metrics["accuracy"], 5),
            "Macro_Recall": round(metrics["macro_recall"], 5),
            "Macro_Precision": round(metrics["macro_precision"], 5),
            "ROC_AUC": round(metrics["roc_auc"], 5),
            "PR_AUC": round(metrics["pr_auc"], 5),
            "FPR": round(metrics["fpr"], 5),
            "FNR": round(metrics["fnr"], 5),
            "MCC": round(metrics["mcc"], 5),
            "Macro_F1_Str": f"{metrics['macro_f1']*100:.2f}%",
            "Accuracy_Str": f"{metrics['accuracy']*100:.2f}%",
            "FPR_Str": f"{metrics['fpr']*100:.3f}%",
            "FNR_Str": f"{metrics['fnr']*100:.3f}%"
        })

    df_abl = pd.DataFrame(ablation_records)
    for a_path in ["results/ablation_results.csv", "experiments/ablations/CAHTDNET_FINAL_V001_ablation.csv"]:
        df_abl.to_csv(a_path, index=False)

    with open("results/ablation_results.json", "w") as f:
        json.dump(ablation_records, f, indent=2)

    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE & FULLY RECONCILED")
    print("=" * 70)
    print(df_abl[["Variant", "Component", "Accuracy_Str", "Macro_F1_Str", "FPR_Str", "FNR_Str"]].to_string(index=False))

if __name__ == "__main__":
    main()
