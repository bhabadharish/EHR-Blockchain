import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
scripts/train_v2_baselines.py
=============================
TRAINS AND EVALUATES TABULAR BASELINES ON CAHTDNET_V2_001 (OPENMP SAFE)

Models evaluated:
1. Random Forest
2. Extra Trees
3. XGBoost
4. LightGBM
5. CatBoost
6. Multi-Layer Perceptron (MLP)
7. FT-Transformer (via isolated subprocess)
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results.metric_engine import compute_canonical_metrics

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

EXPERIMENT_ID = "CAHTDNET_V2_001"
BASE_DIR = f"experiments/{EXPERIMENT_ID}"

def save_baseline_predictions_and_metrics(
    model_name: str,
    model_obj: Any,
    y_test: np.ndarray,
    probs: np.ndarray,
    test_ids: pd.DataFrame,
    inference_time_s: float
) -> Dict[str, Any]:
    pos_probs = probs[:, 1]
    preds = (pos_probs >= 0.50).astype(int)

    # 1. Save raw predictions
    pred_df = pd.DataFrame({
        "sample_id": test_ids["sample_id"],
        "true_label": y_test,
        "predicted_label": preds,
        "probability_class_0": probs[:, 0],
        "probability_class_1": pos_probs,
        "confidence": np.max(probs, axis=1),
        "threshold": 0.50,
        "model": model_name,
        "split": "locked_test",
        "experiment_id": EXPERIMENT_ID
    })

    pred_pq = f"{BASE_DIR}/predictions/{EXPERIMENT_ID}_{model_name}_test_predictions.parquet"
    pred_csv = f"{BASE_DIR}/predictions/{EXPERIMENT_ID}_{model_name}_test_predictions.csv"
    pred_df.to_parquet(pred_pq, index=False)
    pred_df.to_csv(pred_csv, index=False)

    # 2. Compute canonical metrics
    metrics = compute_canonical_metrics(
        y_true=y_test,
        y_pred=preds,
        probabilities=probs,
        threshold=0.50,
        metadata={
            "experiment_id": EXPERIMENT_ID,
            "model": model_name,
            "split": "locked_test",
            "inference_time_s": float(round(inference_time_s, 4)),
            "latency_ms_per_sample": float(round((inference_time_s / len(y_test)) * 1000, 4))
        }
    )

    metrics_path = f"{BASE_DIR}/metrics/{EXPERIMENT_ID}_{model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model artifact
    model_path = f"{BASE_DIR}/models/{model_name}.pkl"
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model_obj, f)
    except Exception as e:
        print(f"  Note: Pickle save skipped for {model_name}: {e}")

    print(f"  [{model_name}] Macro-F1: {metrics['macro_f1']:.4f} | Acc: {metrics['accuracy']:.4f} | FPR: {metrics['fpr']:.4f} | FNR: {metrics['fnr']:.4f}")
    return metrics

def main():
    print("=" * 70)
    print("TRAINING TABULAR BASELINES ON CAHTDNET_V2_001")
    print("=" * 70)

    os.makedirs(f"{BASE_DIR}/models", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/predictions", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/metrics", exist_ok=True)

    data = np.load(f"{BASE_DIR}/data/processed_arrays.npz")
    X_train_num, X_train_cat, y_train = data["X_train_num"], data["X_train_cat"], data["y_train"]
    X_test_num, X_test_cat, y_test = data["X_test_num"], data["X_test_cat"], data["y_test"]

    test_ids = pd.read_csv(f"{BASE_DIR}/splits/test_ids.csv")

    X_train_tab = np.hstack([X_train_num, X_train_cat.astype(np.float32)])
    X_test_tab = np.hstack([X_test_num, X_test_cat.astype(np.float32)])

    all_metrics = []

    # 1. Random Forest
    print("\n--- Training Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=1)
    rf.fit(X_train_tab, y_train)
    t0 = time.time()
    rf_probs = rf.predict_proba(X_test_tab)
    t_inf = time.time() - t0
    all_metrics.append(save_baseline_predictions_and_metrics("Random_Forest", rf, y_test, rf_probs, test_ids, t_inf))

    # 2. Extra Trees
    print("\n--- Training Extra Trees ---")
    et = ExtraTreesClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=1)
    et.fit(X_train_tab, y_train)
    t0 = time.time()
    et_probs = et.predict_proba(X_test_tab)
    t_inf = time.time() - t0
    all_metrics.append(save_baseline_predictions_and_metrics("Extra_Trees", et, y_test, et_probs, test_ids, t_inf))

    # 3. XGBoost
    print("\n--- Training XGBoost ---")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.08, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=1, eval_metric="logloss"
    )
    xgb_clf.fit(X_train_tab, y_train)
    t0 = time.time()
    xgb_probs = xgb_clf.predict_proba(X_test_tab)
    t_inf = time.time() - t0
    all_metrics.append(save_baseline_predictions_and_metrics("XGBoost", xgb_clf, y_test, xgb_probs, test_ids, t_inf))

    # 4. LightGBM
    print("\n--- Training LightGBM ---")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=150, num_leaves=31, learning_rate=0.08, random_state=42, n_jobs=1, verbose=-1
    )
    lgb_clf.fit(X_train_tab, y_train)
    t0 = time.time()
    lgb_probs = lgb_clf.predict_proba(X_test_tab)
    t_inf = time.time() - t0
    all_metrics.append(save_baseline_predictions_and_metrics("LightGBM", lgb_clf, y_test, lgb_probs, test_ids, t_inf))

    # 5. CatBoost
    print("\n--- Training CatBoost ---")
    cb_clf = CatBoostClassifier(iterations=150, depth=6, learning_rate=0.08, random_seed=42, verbose=0, thread_count=1)
    cb_clf.fit(X_train_tab, y_train)
    t0 = time.time()
    cb_probs = cb_clf.predict_proba(X_test_tab)
    t_inf = time.time() - t0
    all_metrics.append(save_baseline_predictions_and_metrics("CatBoost", cb_clf, y_test, cb_probs, test_ids, t_inf))

    # 6. MLP
    print("\n--- Training Multi-Layer Perceptron (MLP) ---")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=25, random_state=42, batch_size=256)
    mlp.fit(X_train_tab, y_train)
    t0 = time.time()
    mlp_probs = mlp.predict_proba(X_test_tab)
    t_inf = time.time() - t0
    all_metrics.append(save_baseline_predictions_and_metrics("MLP", mlp, y_test, mlp_probs, test_ids, t_inf))

    # 7. FT-Transformer via Subprocess
    print("\n--- Executing FT-Transformer Training via Isolated Subprocess ---")
    subprocess.run([sys.executable, "scripts/train_v2_ft_transformer.py"], check=True)
    with open(f"{BASE_DIR}/metrics/{EXPERIMENT_ID}_FT-Transformer_metrics.json", "r") as f:
        ft_m = json.load(f)
    all_metrics.append(ft_m)

    # Compile Consolidated Baseline Comparison Table
    summary_rows = []
    for m in all_metrics:
        summary_rows.append({
            "experiment_id": EXPERIMENT_ID,
            "model": m["metadata"]["model"],
            "dataset": "CICIoT2023+Edge-IIoTset+SyntheticFHIR_V2",
            "split": "locked_test",
            "accuracy": m["accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "fpr": m["fpr"],
            "fnr": m["fnr"],
            "mcc": m["mcc"],
            "latency_ms": m["metadata"]["latency_ms_per_sample"]
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(f"{BASE_DIR}/metrics/baseline_comparison.csv", index=False)
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON SUMMARY (V2 LOCKED TEST):")
    print("=" * 70)
    print(df_summary[["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "fpr", "fnr"]].to_string(index=False))

if __name__ == "__main__":
    main()
