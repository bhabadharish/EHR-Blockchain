import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""Pipeline Script: Tuned XGBoost Training and Multi-Seed Stability (Phase 12 & 15)."""

import os
import sys
import json
import pandas as pd
import numpy as np

from src.data.schema import get_edge_iiot_features
from src.preprocessing.pipeline import LeakageFreePreprocessor
from src.models.boosters import TunedXGBoost
from src.evaluation.metrics import compute_binary_metrics


def main():
    print("==================================================")
    print("PHASE 12 & 15: Training Tuned XGBoost (3 Seeds)")
    print("==================================================")
    train_pq = "data/splits/edge_train.parquet"
    val_pq = "data/splits/edge_val.parquet"
    test_pq = "data/splits/edge_test.parquet"

    df_train = pd.read_parquet(train_pq)
    df_val = pd.read_parquet(val_pq)
    df_test = pd.read_parquet(test_pq)

    features = get_edge_iiot_features(include_labels=False)
    y_train = df_train["Attack_label"].to_numpy().astype(int)
    y_val = df_val["Attack_label"].to_numpy().astype(int)
    y_test = df_test["Attack_label"].to_numpy().astype(int)

    # Preprocessing strictly on train
    preprocessor = LeakageFreePreprocessor()
    X_train = preprocessor.fit_transform(df_train[features])
    X_val = preprocessor.transform(df_val[features])
    X_test = preprocessor.transform(df_test[features])

    seeds = [42, 123, 999]
    seed_metrics = []

    primary_model = None

    for seed in seeds:
        print(f"Training XGBoost with random seed {seed}...")
        xgb_booster = TunedXGBoost(seed=seed, n_estimators=300, max_depth=6, learning_rate=0.05)
        xgb_booster.fit(X_train, y_train, X_val, y_val)

        probs = xgb_booster.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        metrics = compute_binary_metrics(y_test, preds, probs)
        metrics["Seed"] = seed
        seed_metrics.append(metrics)
        print(f"Seed {seed} -> Acc: {metrics['Accuracy']:.4f} | Macro-F1: {metrics['Macro_F1']:.4f} | FPR: {metrics['FPR']:.4f} | FNR: {metrics['FNR']:.4f}")

        if seed == 42:
            primary_model = xgb_booster
            xgb_booster.save("models/base/xgboost/xgboost_primary.pkl")

    # Aggregate multi-seed statistics
    f1_scores = [m["Macro_F1"] for m in seed_metrics]
    acc_scores = [m["Accuracy"] for m in seed_metrics]

    stability_report = {
        "model": "XGBoost",
        "seeds": seeds,
        "macro_f1_mean": round(float(np.mean(f1_scores)), 5),
        "macro_f1_std": round(float(np.std(f1_scores)), 5),
        "macro_f1_min": round(float(np.min(f1_scores)), 5),
        "macro_f1_max": round(float(np.max(f1_scores)), 5),
        "accuracy_mean": round(float(np.mean(acc_scores)), 5),
        "accuracy_std": round(float(np.std(acc_scores)), 5),
        "per_seed_metrics": seed_metrics
    }

    out_json = "models/base/xgboost/stability_report.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(stability_report, f, indent=2)

    print(f"\nXGBoost Multi-Seed Mean Macro-F1: {stability_report['macro_f1_mean']} ± {stability_report['macro_f1_std']}")
    print(f"Saved primary XGBoost checkpoint to models/base/xgboost/xgboost_primary.pkl")


if __name__ == "__main__":
    main()
