import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def main():
    print("=" * 70)
    print("STAGE 11: ROBUSTNESS & ADVERSARIAL STRESS TESTING")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)

    data = np.load("data/processed/processed_arrays.npz")
    X_val = np.hstack([data["X_val_num"], data["X_val_cat"]])
    y_val = data["y_val"]

    with open("models/baselines/LightGBM.pkl", "rb") as f:
        model = pickle.load(f)

    clean_preds = model.predict(X_val)
    clean_acc = accuracy_score(y_val, clean_preds)
    clean_f1 = f1_score(y_val, clean_preds, average="macro")

    stress_tests = [
        {"Test": "Baseline (Clean)", "Accuracy": clean_acc, "Macro_F1": clean_f1, "Drop_F1": "0.0%"}
    ]

    # 1. Gaussian Feature Noise (sigma = 0.1, 0.25, 0.5)
    for sigma in [0.10, 0.25, 0.50]:
        X_noisy = X_val + np.random.normal(0, sigma, X_val.shape)
        preds = model.predict(X_noisy)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="macro")
        stress_tests.append({
            "Test": f"Feature Noise (std={sigma:.2f})",
            "Accuracy": acc,
            "Macro_F1": f1,
            "Drop_F1": f"-{(clean_f1 - f1)*100:.2f}%"
        })

    # 2. Missing Feature Imputation (10%, 25%, 50% dropped)
    for drop_rate in [0.10, 0.25, 0.50]:
        mask = np.random.binomial(1, 1.0 - drop_rate, X_val.shape)
        X_dropped = X_val * mask
        preds = model.predict(X_dropped)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="macro")
        stress_tests.append({
            "Test": f"Missing Features ({int(drop_rate*100)}% dropout)",
            "Accuracy": acc,
            "Macro_F1": f1,
            "Drop_F1": f"-{(clean_f1 - f1)*100:.2f}%"
        })

    df_rob = pd.DataFrame(stress_tests)
    df_rob["Accuracy_Str"] = df_rob["Accuracy"].apply(lambda x: f"{x*100:.2f}%")
    df_rob["Macro_F1_Str"] = df_rob["Macro_F1"].apply(lambda x: f"{x*100:.2f}%")

    df_rob.to_csv("results/robustness_report.csv", index=False)
    with open("results/robustness_report.json", "w") as f:
        json.dump(stress_tests, f, indent=2)

    print("\nROBUSTNESS & PERTURBATION STRESS TEST RESULTS:")
    print(df_rob[["Test", "Accuracy_Str", "Macro_F1_Str", "Drop_F1"]].to_string(index=False))

if __name__ == "__main__":
    main()
