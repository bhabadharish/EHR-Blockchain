"""
scripts/run_robustness.py
=========================
EMPIRICAL ADVERSARIAL & TELEMETRY NOISE ROBUSTNESS BENCHMARK

Strict Methodology:
- Evaluates proposed CA-HTDNet (ZERO LightGBM reuse).
- Baseline is Clean CA-HTDNet test performance.
- Continuous features perturbed with Gaussian noise & bounded shifts.
- Categorical features perturbed with category corruptions (never Gaussian noise).
- Missingness injection (dropout masking).
- Levels: Clean, Low, Medium, High perturbations.
- All metrics computed strictly via results/result_engine.py.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.ca_htdnet import CA_HTDNet
from results.result_engine import compute_canonical_metrics

EXPERIMENT_ID = "CAHTDNET_FINAL_V001"

def evaluate_tensor_data(model, X_num, X_cat, y_true, device):
    ds = TensorDataset(torch.tensor(X_num, dtype=torch.float32), torch.tensor(X_cat, dtype=torch.long))
    loader = DataLoader(ds, batch_size=1024, shuffle=False)
    probs_list = []
    with torch.no_grad():
        for bx_num, bx_cat in loader:
            bx_num, bx_cat = bx_num.to(device), bx_cat.to(device)
            out = model(bx_num, bx_cat)
            p = torch.softmax(out["logits"], dim=-1).cpu().numpy()
            probs_list.append(p)
    probs = np.concatenate(probs_list, axis=0)
    preds = np.argmax(probs, axis=1)
    return compute_canonical_metrics(y_true=y_true, y_pred=preds, probabilities=probs)

def main():
    print("=" * 70)
    print("STAGE 9: ADVERSARIAL TELEMETRY ROBUSTNESS STRESS TESTING (CA-HTDNet)")
    print("=" * 70)

    os.makedirs("experiments/robustness", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Locked Test Data
    data = np.load("data/processed/processed_arrays.npz")
    X_test_num = data["X_test_num"].copy()
    X_test_cat = data["X_test_cat"].copy()
    y_test = data["y_test"].copy()

    # 2. Load CA-HTDNet
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

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Clean Baseline Evaluation
    print("\n--- Evaluating Baseline: Clean Test Data ---")
    clean_metrics = evaluate_tensor_data(model, X_test_num, X_test_cat, y_test, device)
    clean_acc = clean_metrics["accuracy"]
    clean_f1 = clean_metrics["macro_f1"]
    print(f"  Clean CA-HTDNet | Acc: {clean_acc*100:.2f}% | Macro-F1: {clean_f1*100:.2f}% | FPR: {clean_metrics['fpr']*100:.3f}% | FNR: {clean_metrics['fnr']*100:.3f}%")

    stress_tests = [
        {
            "Test_Condition": "Baseline (Clean Locked Test)",
            "Perturbation_Level": "None",
            "Accuracy": round(clean_acc, 5),
            "Macro_F1": round(clean_f1, 5),
            "Macro_Recall": round(clean_metrics["macro_recall"], 5),
            "FPR": round(clean_metrics["fpr"], 5),
            "FNR": round(clean_metrics["fnr"], 5),
            "F1_Degradation": "0.00%",
            "Accuracy_Str": f"{clean_acc*100:.2f}%",
            "Macro_F1_Str": f"{clean_f1*100:.2f}%",
            "FPR_Str": f"{clean_metrics['fpr']*100:.3f}%",
            "FNR_Str": f"{clean_metrics['fnr']*100:.3f}%"
        }
    ]

    # Experiment Suite 1: Continuous Gaussian Noise (Low=0.05, Med=0.15, High=0.30)
    print("\n--- Running Continuous Gaussian Jitter Stress Tests ---")
    np.random.seed(42)
    noise_levels = [("Low", 0.05), ("Medium", 0.15), ("High", 0.30)]
    for level, sigma in noise_levels:
        noisy_num = X_test_num + np.random.normal(0, sigma, X_test_num.shape).astype(np.float32)
        m = evaluate_tensor_data(model, noisy_num, X_test_cat, y_test, device)
        deg = (clean_f1 - m["macro_f1"]) * 100.0
        print(f"  Gaussian Noise (sigma={sigma:.2f}, {level}) | Acc: {m['accuracy']*100:.2f}% | Macro-F1: {m['macro_f1']*100:.2f}% | Drop: -{deg:.2f}%")
        stress_tests.append({
            "Test_Condition": f"Continuous Feature Jitter (sigma={sigma:.2f})",
            "Perturbation_Level": level,
            "Accuracy": round(m["accuracy"], 5),
            "Macro_F1": round(m["macro_f1"], 5),
            "Macro_Recall": round(m["macro_recall"], 5),
            "FPR": round(m["fpr"], 5),
            "FNR": round(m["fnr"], 5),
            "F1_Degradation": f"-{deg:.2f}%",
            "Accuracy_Str": f"{m['accuracy']*100:.2f}%",
            "Macro_F1_Str": f"{m['macro_f1']*100:.2f}%",
            "FPR_Str": f"{m['fpr']*100:.3f}%",
            "FNR_Str": f"{m['fnr']*100:.3f}%"
        })

    # Experiment Suite 2: Telemetry Missingness / Feature Masking (Low=5%, Med=15%, High=30%)
    print("\n--- Running Telemetry Dropout / Masking Stress Tests ---")
    missing_levels = [("Low", 0.05), ("Medium", 0.15), ("High", 0.30)]
    for level, rate in missing_levels:
        mask = np.random.binomial(1, 1.0 - rate, X_test_num.shape).astype(np.float32)
        masked_num = X_test_num * mask
        m = evaluate_tensor_data(model, masked_num, X_test_cat, y_test, device)
        deg = (clean_f1 - m["macro_f1"]) * 100.0
        print(f"  Feature Masking ({int(rate*100)}% Dropout, {level}) | Acc: {m['accuracy']*100:.2f}% | Macro-F1: {m['macro_f1']*100:.2f}% | Drop: -{deg:.2f}%")
        stress_tests.append({
            "Test_Condition": f"Telemetry Masking ({int(rate*100)}% packet loss)",
            "Perturbation_Level": level,
            "Accuracy": round(m["accuracy"], 5),
            "Macro_F1": round(m["macro_f1"], 5),
            "Macro_Recall": round(m["macro_recall"], 5),
            "FPR": round(m["fpr"], 5),
            "FNR": round(m["fnr"], 5),
            "F1_Degradation": f"-{deg:.2f}%",
            "Accuracy_Str": f"{m['accuracy']*100:.2f}%",
            "Macro_F1_Str": f"{m['macro_f1']*100:.2f}%",
            "FPR_Str": f"{m['fpr']*100:.3f}%",
            "FNR_Str": f"{m['fnr']*100:.3f}%"
        })

    # Experiment Suite 3: Categorical Context Corruption (Low=5%, Med=15%, High=30%)
    print("\n--- Running Categorical Context Corruption Stress Tests ---")
    cat_levels = [("Low", 0.05), ("Medium", 0.15), ("High", 0.30)]
    for level, prob in cat_levels:
        corrupted_cat = X_test_cat.copy()
        for c in range(3):
            corrupt_mask = np.random.rand(len(corrupted_cat)) < prob
            corrupted_cat[corrupt_mask, c] = np.random.randint(0, 5, size=corrupt_mask.sum())
        m = evaluate_tensor_data(model, X_test_num, corrupted_cat, y_test, device)
        deg = (clean_f1 - m["macro_f1"]) * 100.0
        print(f"  Categorical Corruption ({int(prob*100)}% flipped, {level}) | Acc: {m['accuracy']*100:.2f}% | Macro-F1: {m['macro_f1']*100:.2f}% | Drop: -{deg:.2f}%")
        stress_tests.append({
            "Test_Condition": f"Categorical Role/Operation Corruption ({int(prob*100)}%)",
            "Perturbation_Level": level,
            "Accuracy": round(m["accuracy"], 5),
            "Macro_F1": round(m["macro_f1"], 5),
            "Macro_Recall": round(m["macro_recall"], 5),
            "FPR": round(m["fpr"], 5),
            "FNR": round(m["fnr"], 5),
            "F1_Degradation": f"-{deg:.2f}%",
            "Accuracy_Str": f"{m['accuracy']*100:.2f}%",
            "Macro_F1_Str": f"{m['macro_f1']*100:.2f}%",
            "FPR_Str": f"{m['fpr']*100:.3f}%",
            "FNR_Str": f"{m['fnr']*100:.3f}%"
        })

    df_rob = pd.DataFrame(stress_tests)
    for r_path in ["results/robustness_report.csv", f"experiments/robustness/{EXPERIMENT_ID}_robustness.csv"]:
        df_rob.to_csv(r_path, index=False)

    with open("results/robustness_report.json", "w") as f:
        json.dump(stress_tests, f, indent=2)

    print("\n" + "=" * 70)
    print("ROBUSTNESS STRESS TESTING COMPLETE (EVALUATING CA-HTDNet)")
    print("=" * 70)
    print(df_rob[["Test_Condition", "Perturbation_Level", "Accuracy_Str", "Macro_F1_Str", "F1_Degradation", "FPR_Str", "FNR_Str"]].to_string(index=False))

if __name__ == "__main__":
    main()
