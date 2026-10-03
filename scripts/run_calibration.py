"""
scripts/run_calibration.py
==========================
PROBABILITY CALIBRATION BENCHMARK (CA-HTDNet)

Strict Methodology:
- Calibration parameters (Temperature Scaling, Platt Scaling) learned STRICTLY on VALIDATION data.
- Evaluated on the locked test set.
- Brier score, ECE (Expected Calibration Error), and reliability diagram points computed.
- Model evaluated is CA-HTDNet under experiment CAHTDNET_FINAL_V001.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.ca_htdnet import CA_HTDNet

EXPERIMENT_ID = "CAHTDNET_FINAL_V001"

def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper if i < n_bins - 1 else probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            acc_in_bin = y_true[in_bin].mean()
            conf_in_bin = probs[in_bin].mean()
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece)

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor):
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return float(self.temperature.item())

def main():
    print("=" * 70)
    print("STAGE 8: PROBABILITY CALIBRATION BENCHMARK (CA-HTDNet)")
    print("=" * 70)

    os.makedirs("experiments/calibration", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 1. Load Partitions
    data = np.load("data/processed/processed_arrays.npz")
    X_val_num, X_val_cat = data["X_val_num"], data["X_val_cat"]
    y_val = data["y_val"]

    X_test_num, X_test_cat = data["X_test_num"], data["X_test_cat"]
    y_test = data["y_test"]

    # 2. Load Model
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

    # 3. Extract Logits for Validation and Test
    def get_logits(num_arr, cat_arr):
        ds = TensorDataset(torch.tensor(num_arr, dtype=torch.float32), torch.tensor(cat_arr, dtype=torch.long))
        loader = DataLoader(ds, batch_size=1024, shuffle=False)
        logits_list = []
        with torch.no_grad():
            for bx, bcat in loader:
                bx, bcat = bx.to(device), bcat.to(device)
                out = model(bx, bcat)
                logits_list.append(out["logits"].cpu())
        return torch.cat(logits_list, dim=0)

    val_logits = get_logits(X_val_num, X_val_cat)
    test_logits = get_logits(X_test_num, X_test_cat)

    # Method 1: Uncalibrated
    val_probs_uncal = torch.softmax(val_logits, dim=-1)[:, 1].numpy()
    test_probs_uncal = torch.softmax(test_logits, dim=-1)[:, 1].numpy()

    # Method 2: Temperature Scaling (Fit on VALIDATION logits)
    print("Fitting Temperature Scaling strictly on Validation split...")
    ts = TemperatureScaler()
    optimal_temp = ts.fit(val_logits, torch.tensor(y_val, dtype=torch.long))
    print(f"  Learned Optimal Temperature: {optimal_temp:.4f}")

    test_logits_scaled = ts(test_logits)
    test_probs_ts = torch.softmax(test_logits_scaled, dim=-1)[:, 1].detach().numpy()

    # Method 3: Platt Scaling (Logistic Regression on VALIDATION probabilities)
    print("Fitting Platt Scaling strictly on Validation probabilities...")
    platt = LogisticRegression(C=1.0, solver="lbfgs")
    platt.fit(val_probs_uncal.reshape(-1, 1), y_val)
    test_probs_platt = platt.predict_proba(test_probs_uncal.reshape(-1, 1))[:, 1]

    # Evaluate all three methods on LOCKED TEST SET
    calibration_results = []
    methods = [
        ("Uncalibrated", test_probs_uncal, 1.0),
        ("Temperature_Scaling", test_probs_ts, optimal_temp),
        ("Platt_Scaling", test_probs_platt, "logistic_fit")
    ]

    for name, p, param in methods:
        brier = float(brier_score_loss(y_test, p))
        nll = float(log_loss(y_test, p))
        ece = float(compute_ece(p, y_test, n_bins=10))

        calibration_results.append({
            "Method": name,
            "Parameter": str(round(param, 4)) if isinstance(param, float) else str(param),
            "Brier_Score": round(brier, 5),
            "ECE": round(ece, 5),
            "NLL": round(nll, 5),
            "Brier_Score_Str": f"{brier:.5f}",
            "ECE_Str": f"{ece*100:.3f}%",
            "NLL_Str": f"{nll:.4f}"
        })
        print(f"  {name} | Brier: {brier:.5f} | ECE: {ece*100:.3f}% | NLL: {nll:.4f}")

    df_cal = pd.DataFrame(calibration_results)
    df_cal.to_csv("results/calibration_comparison.csv", index=False)
    df_cal.to_csv(f"experiments/calibration/{EXPERIMENT_ID}_calibration.csv", index=False)

    # Reliability Diagram data
    prob_true, prob_pred = calibration_curve(y_test, test_probs_ts, n_bins=10)
    curve_data = {
        "temperature_scaling": {
            "optimal_temperature": optimal_temp,
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist()
        },
        "ece": calibration_results[1]["ECE"],
        "brier_score": calibration_results[1]["Brier_Score"]
    }

    with open("results/calibration_report.json", "w") as f:
        json.dump(calibration_results, f, indent=2)
    with open(f"experiments/calibration/{EXPERIMENT_ID}_calibration.json", "w") as f:
        json.dump(curve_data, f, indent=2)

    print("\n" + "=" * 70)
    print("CALIBRATION BENCHMARK COMPLETE")
    print("=" * 70)
    print(df_cal[["Method", "Brier_Score_Str", "ECE_Str", "NLL_Str"]].to_string(index=False))

if __name__ == "__main__":
    main()
