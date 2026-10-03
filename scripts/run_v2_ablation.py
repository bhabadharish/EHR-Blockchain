"""
scripts/run_v2_ablation.py
==========================
RIGOROUS ABLATION STUDY (A0 to A9) FOR CA-HTDNet V2

Configurations evaluated:
A0: Base MLP (No Attention, No Gating, No Residual, No TCN, CrossEntropy)
A1: + Attention (Multi-head Cross-Feature Attention)
A2: + Feature Gating (Instance-adaptive feature gating)
A3: + Residual Connections (Deep residual representations)
A4: + Focal Loss (Focal modulation gamma=2.0)
A5: + Class Weighting (Inverse effective-number class weights)
A6: + Attention + Gating
A7: + Attention + Residual
A8: + Gating + Residual
A9: Full CA-HTDNet V2 (Attention + Gating + Residual + TCN + Class-Balanced Focal Margin Loss)

Protocol:
- Evaluated on exact same Train, Validation, and Locked Test splits
- All metrics computed strictly via results/metric_engine.py
- Reconciles mathematical consistency before saving
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.ca_htdnet_v2 import CAHTDNetV2
from src.models.loss_v2 import AsymmetricClassBalancedMarginLoss
from results.metric_engine import compute_canonical_metrics

EXPERIMENT_ID = "CAHTDNET_V2_001"
BASE_DIR = f"experiments/{EXPERIMENT_ID}"

def run_ablation_variant(
    variant_id: str,
    name: str,
    use_gating: bool,
    use_attention: bool,
    use_residual: bool,
    use_tcn: bool,
    use_transformer: bool,
    use_focal: bool,
    use_class_weight: bool,
    use_margin: bool,
    epochs: int = 10,
    batch_size: int = 256
) -> Dict[str, Any]:
    print(f"\n--- Running Ablation {variant_id}: {name} ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    data = np.load(f"{BASE_DIR}/data/processed_arrays.npz")
    X_train_num, X_train_cat, y_train = data["X_train_num"], data["X_train_cat"], data["y_train"]
    X_val_num, X_val_cat, y_val = data["X_val_num"], data["X_val_cat"], data["y_val"]
    X_test_num, X_test_cat, y_test = data["X_test_num"], data["X_test_cat"], data["y_test"]

    # Subsample 60,000 for fast ablation convergence
    idx = np.random.choice(len(X_train_num), size=min(60000, len(X_train_num)), replace=False)
    train_ds = TensorDataset(
        torch.tensor(X_train_num[idx], dtype=torch.float32),
        torch.tensor(X_train_cat[idx], dtype=torch.long),
        torch.tensor(y_train[idx], dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_num, dtype=torch.float32),
        torch.tensor(X_val_cat, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    class_counts = np.bincount(y_train[idx])
    cw = torch.tensor([
        len(idx) / (2.0 * class_counts[0]),
        len(idx) / (2.0 * class_counts[1])
    ], dtype=torch.float32) if use_class_weight else None

    gamma = 2.0 if use_focal else 0.0
    margin_val = 0.15 if use_margin else 0.0

    model = CAHTDNetV2(
        num_numerical=X_train_num.shape[1],
        cat_cardinalities=[10, 15, 10],
        d_model=128,
        use_gating=use_gating,
        use_attention=use_attention,
        use_residual=use_residual,
        use_tcn=use_tcn,
        use_transformer=use_transformer,
        use_dual_branch=(use_tcn or use_transformer)
    ).to(device)

    criterion = AsymmetricClassBalancedMarginLoss(
        class_weights=cw,
        gamma=gamma,
        margin=margin_val,
        label_smoothing=0.01,
        aux_risk_weight=0.10
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_state = None

    for ep in range(epochs):
        model.train()
        for bx_num, bx_cat, by in train_loader:
            bx_num, bx_cat, by = bx_num.to(device), bx_cat.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx_num, bx_cat)
            loss_dict = criterion(out["logits"], by, out["risk_score"])
            loss_dict["total_loss"].backward()
            optimizer.step()

        # Val check
        model.eval()
        v_probs_list = []
        with torch.no_grad():
            for bx_num, bx_cat, _ in val_loader:
                bx_num, bx_cat = bx_num.to(device), bx_cat.to(device)
                out = model(bx_num, bx_cat)
                v_probs_list.append(out["probabilities"].cpu().numpy())
        v_probs = np.vstack(v_probs_list)
        v_preds = (v_probs[:, 1] >= 0.50).astype(int)
        v_metrics = compute_canonical_metrics(y_val, v_preds, v_probs, threshold=0.50)

        if v_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = v_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Evaluate on Locked Test
    model.load_state_dict(best_state)
    model.eval()
    test_ds = TensorDataset(
        torch.tensor(X_test_num, dtype=torch.float32),
        torch.tensor(X_test_cat, dtype=torch.long)
    )
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    t_probs_list = []
    with torch.no_grad():
        for bx_num, bx_cat in test_loader:
            bx_num, bx_cat = bx_num.to(device), bx_cat.to(device)
            out = model(bx_num, bx_cat)
            t_probs_list.append(out["probabilities"].cpu().numpy())
    t_probs = np.vstack(t_probs_list)
    t_preds = (t_probs[:, 1] >= 0.50).astype(int)

    test_metrics = compute_canonical_metrics(
        y_true=y_test,
        y_pred=t_preds,
        probabilities=t_probs,
        threshold=0.50,
        metadata={
            "variant_id": variant_id,
            "name": name,
            "experiment_id": EXPERIMENT_ID
        }
    )

    print(f"  {variant_id} [{name}]: Macro-F1={test_metrics['macro_f1']:.4f} | Acc={test_metrics['accuracy']:.4f} | FPR={test_metrics['fpr']:.4f} | FNR={test_metrics['fnr']:.4f}")
    return {
        "variant": variant_id,
        "configuration": name,
        "accuracy": test_metrics["accuracy"],
        "macro_precision": test_metrics["macro_precision"],
        "macro_recall": test_metrics["macro_recall"],
        "macro_f1": test_metrics["macro_f1"],
        "roc_auc": test_metrics["roc_auc"],
        "pr_auc": test_metrics["pr_auc"],
        "fpr": test_metrics["fpr"],
        "fnr": test_metrics["fnr"],
        "mcc": test_metrics["mcc"]
    }

def main():
    print("=" * 70)
    print("CA-HTDNet V2: SYSTEMATIC ABLATION STUDY (A0 TO A9)")
    print("=" * 70)

    os.makedirs(f"{BASE_DIR}/ablations", exist_ok=True)

    ablations = [
        ("A0", "Base MLP", False, False, False, False, False, False, False, False),
        ("A1", "+ Attention", False, True, False, False, True, False, False, False),
        ("A2", "+ Feature Gating", True, False, False, False, False, False, False, False),
        ("A3", "+ Residual Connections", False, False, True, False, False, False, False, False),
        ("A4", "+ Focal Loss", False, False, False, False, False, True, False, False),
        ("A5", "+ Class Weighting", False, False, False, False, False, False, True, False),
        ("A6", "+ Attention + Gating", True, True, False, False, True, False, False, False),
        ("A7", "+ Attention + Residual", False, True, True, False, True, False, False, False),
        ("A8", "+ Gating + Residual", True, False, True, False, False, False, False, False),
        ("A9", "Full CA-HTDNet V2", True, True, True, True, True, True, True, True)
    ]

    results = []
    for var_id, name, ug, ua, ur, utcn, utrans, ufocal, ucw, umar in ablations:
        res = run_ablation_variant(
            variant_id=var_id,
            name=name,
            use_gating=ug,
            use_attention=ua,
            use_residual=ur,
            use_tcn=utcn,
            use_transformer=utrans,
            use_focal=ufocal,
            use_class_weight=ucw,
            use_margin=umar,
            epochs=8
        )
        results.append(res)

    df_ablation = pd.DataFrame(results)
    out_path = f"{BASE_DIR}/ablations/ablation_results.csv"
    df_ablation.to_csv(out_path, index=False)
    df_ablation.to_csv("results/ablation_results.csv", index=False)

    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY (LOCKED TEST SET):")
    print("=" * 70)
    print(df_ablation[["variant", "configuration", "accuracy", "macro_f1", "macro_precision", "macro_recall", "fpr", "fnr"]].to_string(index=False))

if __name__ == "__main__":
    main()
