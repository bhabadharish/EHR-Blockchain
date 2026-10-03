"""
scripts/train_baselines.py
==========================
CANONICAL BASELINE MODEL TRAINING & EVALUATION BENCHMARK

Strict Methodology:
- Trained on CANONICAL TRAIN partition only.
- Evaluated on LOCKED TEST partition.
- All metrics computed strictly via results/result_engine.py.
- Raw predictions saved with sample IDs and probabilities.
- Zero manual metric editing. Zero test-set tuning.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import time
import pickle
import subprocess
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results.result_engine import compute_canonical_metrics

EXPERIMENT_ID = "CAHTDNET_FINAL_V001"
DATASET_NAME = "CICIoT2023+Edge-IIoTset+SyntheticFHIR"
SPLIT_NAME = "locked_test"

def main():
    print("=" * 70)
    print("STAGE 3: CANONICAL BASELINE TRAINING & LOCKED TEST BENCHMARK")
    print("=" * 70)

    os.makedirs("experiments/models/baselines", exist_ok=True)
    os.makedirs("models/baselines", exist_ok=True)
    os.makedirs("experiments/predictions", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("experiments/metrics", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Load canonical processed arrays
    print("Loading canonical processed arrays...")
    data = np.load("data/processed/processed_arrays.npz")
    X_train_num, X_train_cat = data["X_train_num"], data["X_train_cat"]
    y_train = data["y_train"]

    X_test_num, X_test_cat = data["X_test_num"], data["X_test_cat"]
    y_test = data["y_test"]

    X_train = np.hstack([X_train_num, X_train_cat])
    X_test = np.hstack([X_test_num, X_test_cat])

    test_ids_df = pd.read_csv("splits/test_ids.csv")
    sample_ids = test_ids_df["sample_id"].values

    print(f"  Train: {X_train.shape} samples | Locked Test: {X_test.shape} samples")

    # Define standard baselines with single-thread to avoid macOS OpenMP library conflicts
    baselines = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision_Tree": DecisionTreeClassifier(max_depth=16, min_samples_split=10, class_weight="balanced", random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=16, class_weight="balanced", n_jobs=1, random_state=42),
        "Extra_Trees": ExtraTreesClassifier(n_estimators=100, max_depth=16, class_weight="balanced", n_jobs=1, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.10, scale_pos_weight=0.5, n_jobs=1, random_state=42, eval_metric="logloss"),
        "LightGBM": lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.10, class_weight="balanced", num_leaves=31, n_jobs=1, random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=150, depth=6, learning_rate=0.10, auto_class_weights="Balanced", thread_count=1, verbose=0, random_seed=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=30, batch_size=256, early_stopping=True, random_state=42)
    }

    comparison_rows = []

    for name, model in baselines.items():
        print(f"\n--- Training Baseline: {name} ---")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t_inf_start = time.time()
        y_prob = model.predict_proba(X_test)
        inf_time = time.time() - t_inf_start
        y_pred = np.argmax(y_prob, axis=1)

        # Save model
        for m_path in [f"experiments/models/baselines/{name}.pkl", f"models/baselines/{name}.pkl"]:
            with open(m_path, "wb") as f:
                pickle.dump(model, f)
        model_size_kb = os.path.getsize(f"models/baselines/{name}.pkl") / 1024.0

        # Save raw predictions
        pred_df = pd.DataFrame({
            "sample_id": sample_ids,
            "true_label": y_test,
            "predicted_label": y_pred,
            "probability_class_0": y_prob[:, 0],
            "probability_class_1": y_prob[:, 1],
            "confidence": np.max(y_prob, axis=1)
        })
        pred_path_pq = f"experiments/predictions/{EXPERIMENT_ID}_{name}_test_predictions.parquet"
        pred_path_csv = f"experiments/predictions/{EXPERIMENT_ID}_{name}_test_predictions.csv"
        pred_df.to_parquet(pred_path_pq, index=False)
        pred_df.to_csv(pred_path_csv, index=False)

        # Single source of truth evaluation
        meta = {
            "experiment_id": EXPERIMENT_ID,
            "model": name,
            "model_type": "baseline",
            "dataset": DATASET_NAME,
            "split": SPLIT_NAME,
            "train_time_s": float(round(train_time, 3)),
            "inf_time_s": float(round(inf_time, 4)),
            "latency_ms_per_sample": float(round((inf_time * 1000.0) / len(y_test), 4)),
            "model_size_kb": float(round(model_size_kb, 2))
        }
        metrics = compute_canonical_metrics(
            y_true=y_test,
            y_pred=y_pred,
            probabilities=y_prob,
            metadata=meta
        )

        with open(f"experiments/metrics/{EXPERIMENT_ID}_{name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  Acc: {metrics['accuracy']*100:.2f}% | Macro-F1: {metrics['macro_f1']*100:.2f}% | Macro-Recall: {metrics['macro_recall']*100:.2f}% | FPR: {metrics['fpr']*100:.3f}% | FNR: {metrics['fnr']*100:.3f}%")

        comparison_rows.append({
            "experiment_id": EXPERIMENT_ID,
            "model": name,
            "dataset": DATASET_NAME,
            "split": SPLIT_NAME,
            "accuracy": round(metrics["accuracy"], 5),
            "macro_precision": round(metrics["macro_precision"], 5),
            "macro_recall": round(metrics["macro_recall"], 5),
            "macro_f1": round(metrics["macro_f1"], 5),
            "roc_auc": round(metrics["roc_auc"], 5),
            "pr_auc": round(metrics["pr_auc"], 5),
            "fpr": round(metrics["fpr"], 5),
            "fnr": round(metrics["fnr"], 5),
            "mcc": round(metrics["mcc"], 5)
        })

    # Train FT-Transformer in isolated subprocess to eliminate thread conflict
    print("\n--- Spawning Isolated Process for FT-Transformer ---")
    sub_res = subprocess.run([sys.executable, "scripts/train_ft_transformer.py"], check=True)
    
    # Load FT-Transformer metrics
    with open(f"experiments/metrics/{EXPERIMENT_ID}_FT-Transformer_metrics.json", "r") as f:
        metrics_ft = json.load(f)

    comparison_rows.append({
        "experiment_id": EXPERIMENT_ID,
        "model": "FT-Transformer",
        "dataset": DATASET_NAME,
        "split": SPLIT_NAME,
        "accuracy": round(metrics_ft["accuracy"], 5),
        "macro_precision": round(metrics_ft["macro_precision"], 5),
        "macro_recall": round(metrics_ft["macro_recall"], 5),
        "macro_f1": round(metrics_ft["macro_f1"], 5),
        "roc_auc": round(metrics_ft["roc_auc"], 5),
        "pr_auc": round(metrics_ft["pr_auc"], 5),
        "fpr": round(metrics_ft["fpr"], 5),
        "fnr": round(metrics_ft["fnr"], 5),
        "mcc": round(metrics_ft["mcc"], 5)
    })

    # Save baseline comparison tables
    df_comp = pd.DataFrame(comparison_rows)
    for c_path in ["results/baseline_comparison.csv", "experiments/metrics/baseline_comparison.csv"]:
        df_comp.to_csv(c_path, index=False)

    print("\n" + "=" * 70)
    print("BASELINE TEST BENCHMARK COMPLETE (ALL METRICS GENERATED VIA SINGLE ENGINE)")
    print("=" * 70)
    print(df_comp[["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc", "fpr", "fnr", "mcc"]].to_string(index=False))

if __name__ == "__main__":
    main()
