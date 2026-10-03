import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.metrics import evaluate_classification_performance

def main():
    print("=" * 70)
    print("STAGE 4: TRAINING & BENCHMARKING 9 BASELINE MODELS")
    print("=" * 70)

    os.makedirs("models/baselines", exist_ok=True)
    os.makedirs("experiments/runs/baselines", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Load Preprocessed Arrays
    data = np.load("data/processed/processed_arrays.npz")
    X_train = np.hstack([data["X_train_num"], data["X_train_cat"]])
    y_train = data["y_train"]

    X_val = np.hstack([data["X_val_num"], data["X_val_cat"]])
    y_val = data["y_val"]

    print(f"Loaded Train: {X_train.shape} | Val: {X_val.shape}")

    # Baseline configurations
    models_dict = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision_Tree": DecisionTreeClassifier(max_depth=15, min_samples_split=10, class_weight="balanced", random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=16, class_weight="balanced", n_jobs=-1, random_state=42),
        "Extra_Trees": ExtraTreesClassifier(n_estimators=100, max_depth=16, class_weight="balanced", n_jobs=-1, random_state=42),
        "SVM": CalibratedClassifierCV(SGDClassifier(loss="hinge", alpha=1e-4, max_iter=1000, random_state=42)),
        "XGBoost": xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42, eval_metric="logloss"),
        "LightGBM": lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, num_leaves=31, n_jobs=-1, random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=150, depth=6, learning_rate=0.1, verbose=0, random_seed=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=25, batch_size=256, early_stopping=True, random_state=42)
    }

    baseline_metrics = {}
    summary_rows = []

    for name, model in models_dict.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t_inf = time.time()
        y_prob = model.predict_proba(X_val)
        inf_time = time.time() - t_inf
        y_pred = np.argmax(y_prob, axis=1)

        # Estimate size
        model_save_path = f"models/baselines/{name}.pkl"
        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)
        model_size_kb = os.path.getsize(model_save_path) / 1024.0

        # Parameter count approximation
        param_count = 0
        if hasattr(model, "coef_"):
            param_count = model.coef_.size
        elif hasattr(model, "feature_importances_"):
            param_count = len(model.feature_importances_)

        metrics = evaluate_classification_performance(
            y_true=y_val,
            y_pred=y_pred,
            y_prob=y_prob,
            training_time=train_time,
            inference_time=inf_time,
            model_size_kb=model_size_kb,
            param_count=param_count
        )

        baseline_metrics[name] = metrics

        print(f"  Acc: {metrics['Accuracy']*100:.2f}% | Macro-F1: {metrics['Macro_F1']*100:.2f}% | Macro-Recall: {metrics['Macro_Recall']*100:.2f}% | Train Time: {train_time:.2f}s | Latency: {metrics['Inference_Time_s']*1000/len(y_val):.4f} ms/sample")

        summary_rows.append({
            "Model": name,
            "Accuracy": f"{metrics['Accuracy']*100:.2f}%",
            "Balanced_Accuracy": f"{metrics['Balanced_Accuracy']*100:.2f}%",
            "Macro_Precision": f"{metrics['Macro_Precision']*100:.2f}%",
            "Macro_Recall": f"{metrics['Macro_Recall']*100:.2f}%",
            "Macro_F1": f"{metrics['Macro_F1']*100:.2f}%",
            "Weighted_F1": f"{metrics['Weighted_F1']*100:.2f}%",
            "ROC_AUC": f"{metrics['ROC_AUC']:.4f}",
            "PR_AUC": f"{metrics['PR_AUC']:.4f}",
            "FPR": f"{metrics['FPR']*100:.3f}%",
            "FNR": f"{metrics['FNR']*100:.3f}%",
            "Training_Time_s": f"{train_time:.2f}",
            "Inference_Latency_ms": f"{metrics['Inference_Time_s']*1000/len(y_val):.4f}"
        })

    # Save outputs
    with open("experiments/runs/baselines/baseline_metrics.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("results/baseline_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("BASELINE PERFORMANCE COMPARISON (MEASURED EMPIRICALLY ON VALIDATION SPLIT):")
    print("=" * 70)
    print(df_summary[["Model", "Accuracy", "Macro_F1", "Macro_Recall", "ROC_AUC", "FPR", "FNR", "Training_Time_s"]].to_string(index=False))

if __name__ == "__main__":
    main()
