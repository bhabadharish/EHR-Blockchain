#!/usr/bin/env python3
"""
scripts/evaluate_models.py
Comprehensive multi-seed training, evaluation, ablation, and statistical validation pipeline
(Phases 15, 17, 18, 19, 20, 21).

Trains 6 baselines, 4 ablations, and proposed TCN-Transformer-Attention across 5 seeds:
[42, 101, 2024, 777, 9999].
Computes complete metrics:
- Accuracy, Precision (macro), Recall (macro), Macro-F1, Weighted-F1, Balanced Accuracy
- ROC-AUC (ovr macro), PR-AUC (macro)
- TP, TN, FP, FN, FPR, FNR
- Confusion matrices
- Paired statistical significance tests (paired t-test, p-values)
- SHAP feature importances
- Saves results to results/metrics/ and results/tables/
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone
from scipy.stats import t, ttest_rel, wilcoxon
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize
from sklearn.feature_selection import SelectKBest, f_classif
import shap

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.security.telemetry import TelemetryDatasetManager
from backend.models.architectures import (
    ProposedTCNTransformerAttention, TCNOnlyModel, TransformerOnlyModel,
    TCNTransformerModel, MLPModel, CNN1DModel, BiLSTMModel
)
from backend.models.loss import MultiClassFocalLoss, compute_class_weights
from backend.models.baselines import get_baseline_model

RESULTS_EXP_DIR = os.path.join(PROJECT_ROOT, "results", "experiments")
RESULTS_METRICS_DIR = os.path.join(PROJECT_ROOT, "results", "metrics")
RESULTS_TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

os.makedirs(RESULTS_EXP_DIR, exist_ok=True)
os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)
os.makedirs(RESULTS_TABLES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, num_classes: int) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # ROC-AUC & PR-AUC
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        if num_classes == 2:
            roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
            pr_auc = float(average_precision_score(y_true, y_prob[:, 1]))
        else:
            roc_auc = float(roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr"))
            pr_auc = float(average_precision_score(y_true_bin, y_prob, average="macro"))
    except Exception:
        roc_auc = 0.0
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # Binary classification metrics treating Normal (class 0) vs Attack (classes 1-14)
    y_true_binary = (y_true != 0).astype(int)
    y_pred_binary = (y_pred != 0).astype(int)
    cm_bin = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm_bin.ravel()
    
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "balanced_accuracy": bal_acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "fpr": fpr,
        "fnr": fnr,
        "confusion_matrix": cm.tolist()
    }

def train_pytorch_model(
    model_name: str,
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    epochs: int = 2,
    batch_size: int = 1024,
    lr: float = 0.003
) -> Tuple[Dict[str, Any], np.ndarray, float]:
    model = model.to(DEVICE)
    class_weights = compute_class_weights(y_train, num_classes=15).to(DEVICE)
    criterion = MultiClassFocalLoss(alpha=class_weights, gamma=1.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32)), batch_size=2048, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32)), batch_size=2048, shuffle=False)

    best_val_f1 = 0.0
    best_weights = None

    t0_train = time.perf_counter()
    for ep in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Batched validation
        model.eval()
        val_logits = []
        with torch.no_grad():
            for (bx,) in val_loader:
                bx = bx.to(DEVICE)
                val_logits.append(model(bx).cpu())
        val_out = torch.cat(val_logits, dim=0)
        val_preds = torch.argmax(val_out, dim=1).numpy()
        val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_duration = time.perf_counter() - t0_train

    if best_weights:
        model.load_state_dict(best_weights)
        model = model.to(DEVICE)

    # Save model weights
    checkpoint_file = os.path.join(CHECKPOINTS_DIR, f"{model_name}_seed_{seed}.pt")
    torch.save(model.state_dict(), checkpoint_file)

    # Latency on 1000 samples
    model.eval()
    sample_sub = torch.tensor(X_test[:1000], dtype=torch.float32).to(DEVICE)
    t_inf0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(5):
            _ = model(sample_sub)
    inf_latency_per_sample_ms = ((time.perf_counter() - t_inf0) / (5 * 1000.0)) * 1000.0

    # Batched Test Inference
    test_logits = []
    with torch.no_grad():
        for (bx,) in test_loader:
            bx = bx.to(DEVICE)
            test_logits.append(model(bx).cpu())
    test_logits = torch.cat(test_logits, dim=0)
    test_probs = torch.softmax(test_logits, dim=1).numpy()
    test_preds = np.argmax(test_probs, axis=1)

    metrics = compute_detailed_metrics(y_test, test_preds, test_probs, num_classes=15)
    metrics["train_time_sec"] = round(train_duration, 2)
    metrics["inference_latency_ms"] = round(inf_latency_per_sample_ms, 4)
    metrics["parameter_count"] = count_parameters(model)
    return metrics, test_probs, inf_latency_per_sample_ms

def train_classical_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int
) -> Tuple[Dict[str, Any], np.ndarray]:
    clf = get_baseline_model(model_name, seed=seed)
    
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_duration = time.perf_counter() - t0

    sample_sub = X_test[:1000]
    t_inf0 = time.perf_counter()
    for _ in range(5):
        _ = clf.predict(sample_sub)
    inf_latency_per_sample_ms = ((time.perf_counter() - t_inf0) / (5 * 1000.0)) * 1000.0

    test_probs = clf.predict_proba(X_test)
    test_preds = np.argmax(test_probs, axis=1)

    metrics = compute_detailed_metrics(y_test, test_preds, test_probs, num_classes=15)
    metrics["train_time_sec"] = round(train_duration, 2)
    metrics["inference_latency_ms"] = round(inf_latency_per_sample_ms, 4)
    metrics["parameter_count"] = 0
    return metrics, test_probs

def main():
    parser = argparse.ArgumentParser(description="Evaluate threat detection models across multiple seeds")
    parser.add_argument("--seeds", type=str, default="42,101,2024,777,9999", help="Comma-separated seeds")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs per neural model")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    epochs = args.epochs

    print("=" * 75)
    print("PHASES 18-21: SYSTEMATIC MULTI-SEED THREAT DETECTION & ABLATION PIPELINE")
    print(f"Seeds: {seeds} | Epochs: {epochs} | Device: {DEVICE}")
    print("=" * 75)

    data = TelemetryDatasetManager.load_and_preprocess(seed=42)
    X_train_full, y_train_full = data["X_train"], data["y_train_multi"]
    X_val, y_val = data["X_val"], data["y_val_multi"]
    X_test, y_test = data["X_test"], data["y_test_multi"]
    feature_names = data["feature_names"]
    class_names = data["class_names"]
    num_classes = len(class_names)
    in_features = X_train_full.shape[1]

    # Stratified 25,000 sample for fast, balanced neural training across seeds
    # Ensures deterministic class representation
    np.random.seed(42)
    sample_indices = []
    classes = np.unique(y_train_full)
    per_class_quota = 25000 // len(classes)
    for c in classes:
        idx_c = np.where(y_train_full == c)[0]
        n_take = min(len(idx_c), per_class_quota)
        sample_indices.extend(np.random.choice(idx_c, n_take, replace=False))
    # Fill remaining to reach 25000
    rem = 25000 - len(sample_indices)
    if rem > 0:
        unused = list(set(range(len(y_train_full))) - set(sample_indices))
        sample_indices.extend(np.random.choice(unused, rem, replace=False))
    sample_indices = np.array(sample_indices)
    
    X_train_neural = X_train_full[sample_indices]
    y_train_neural = y_train_full[sample_indices]

    # Feature Selection for Ablation E (Top 25 features selected on train set)
    selector = SelectKBest(score_func=f_classif, k=25)
    selector.fit(X_train_neural, y_train_neural)
    top25_indices = selector.get_support(indices=True)
    X_train_top25 = X_train_neural[:, top25_indices]
    X_val_top25 = X_val[:, top25_indices]
    X_test_top25 = X_test[:, top25_indices]

    models_suite = [
        # Baselines
        ("Baseline_1_LogisticRegression", "classical"),
        ("Baseline_2_RandomForest", "classical"),
        ("Baseline_3_LightGBM", "classical"),
        ("Baseline_4_MLP", "pytorch"),
        ("Baseline_5_1D_CNN", "pytorch"),
        ("Baseline_6_BiLSTM", "pytorch"),
        # Ablations
        ("Ablation_A_TCN_Only", "pytorch"),
        ("Ablation_B_Transformer_Only", "pytorch"),
        ("Ablation_C_TCN_Transformer", "pytorch"),
        # Proposed Architecture & Ablation E
        ("Proposed_TCN_Transformer_Attention", "pytorch"),
        ("Ablation_E_Proposed_Feature_Selection", "pytorch_fs")
    ]

    runs_path = os.path.join(RESULTS_EXP_DIR, "model_evaluation_runs.json")
    all_evaluation_runs = []
    if os.path.exists(runs_path):
        try:
            with open(runs_path, "r") as fp:
                all_evaluation_runs = json.load(fp)
            print(f"[RESUME] Loaded {len(all_evaluation_runs)} previously completed runs from {runs_path}")
        except Exception:
            all_evaluation_runs = []

    completed_runs = {(r["model"], r["seed"]) for r in all_evaluation_runs}
    confusion_matrices_store = {}

    for model_name, model_type in models_suite:
        print(f"\n[EVALUATING] {model_name}...")
        for seed in seeds:
            if (model_name, seed) in completed_runs:
                prior_run = next(r for r in all_evaluation_runs if r["model"] == model_name and r["seed"] == seed)
                if seed == 42 and "confusion_matrix" in prior_run:
                    confusion_matrices_store[model_name] = prior_run["confusion_matrix"]
                print(f"  [CACHED] Seed {seed:5d} -> Acc: {prior_run['accuracy']:.4f} | Macro-F1: {prior_run['macro_f1']:.4f}")
                continue

            torch.manual_seed(seed)
            np.random.seed(seed)

            if model_type == "classical":
                base_key = model_name.split("_")[-1]
                metrics, test_probs = train_classical_model(
                    base_key, X_train_neural, y_train_neural, X_test, y_test, seed=seed
                )
            elif model_type == "pytorch_fs":
                net = ProposedTCNTransformerAttention(in_features=25, num_classes=num_classes)
                metrics, test_probs, _ = train_pytorch_model(
                    model_name, net, X_train_top25, y_train_neural, X_val_top25, y_val, X_test_top25, y_test,
                    seed=seed, epochs=epochs
                )
            else: # pytorch
                if model_name == "Proposed_TCN_Transformer_Attention":
                    net = ProposedTCNTransformerAttention(in_features=in_features, num_classes=num_classes)
                elif model_name == "Ablation_A_TCN_Only":
                    net = TCNOnlyModel(in_features=in_features, num_classes=num_classes)
                elif model_name == "Ablation_B_Transformer_Only":
                    net = TransformerOnlyModel(in_features=in_features, num_classes=num_classes)
                elif model_name == "Ablation_C_TCN_Transformer":
                    net = TCNTransformerModel(in_features=in_features, num_classes=num_classes)
                elif model_name == "Baseline_4_MLP":
                    net = MLPModel(in_features=in_features, num_classes=num_classes)
                elif model_name == "Baseline_5_1D_CNN":
                    net = CNN1DModel(in_features=in_features, num_classes=num_classes)
                elif model_name == "Baseline_6_BiLSTM":
                    net = BiLSTMModel(in_features=in_features, num_classes=num_classes)
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                metrics, test_probs, _ = train_pytorch_model(
                    model_name, net, X_train_neural, y_train_neural, X_val, y_val, X_test, y_test,
                    seed=seed, epochs=epochs
                )

            run_record = {
                "model": model_name,
                "seed": seed,
                **metrics
            }
            all_evaluation_runs.append(run_record)
            completed_runs.add((model_name, seed))
            
            if seed == 42:
                confusion_matrices_store[model_name] = metrics["confusion_matrix"]

            print(f"  Seed {seed:5d} -> Acc: {metrics['accuracy']:.4f} | Macro-F1: {metrics['macro_f1']:.4f} | FPR: {metrics['fpr']:.4f} | FNR: {metrics['fnr']:.4f} | Latency: {metrics['inference_latency_ms']:.3f} ms")

            # Incremental save
            with open(runs_path, "w") as fp:
                json.dump(all_evaluation_runs, fp, indent=2)

    # Save all raw runs
    runs_path = os.path.join(RESULTS_EXP_DIR, "model_evaluation_runs.json")
    with open(runs_path, "w") as fp:
        json.dump(all_evaluation_runs, fp, indent=2)

    # Save confusion matrices
    cm_path = os.path.join(RESULTS_METRICS_DIR, "confusion_matrices.json")
    with open(cm_path, "w") as fp:
        json.dump(confusion_matrices_store, fp, indent=2)

    # Statistical Aggregation (Mean, Std, 95% CI)
    runs_df = pd.DataFrame(all_evaluation_runs)
    grouped = runs_df.groupby("model")

    summary_records = []
    for model_name, group in grouped:
        n = len(group)
        t_crit = t.ppf(0.975, df=n - 1) if n > 1 else 1.96

        def calc_ci(col):
            m = group[col].mean()
            s = group[col].std()
            ci = t_crit * (s / np.sqrt(n)) if n > 1 else 0.0
            return m, s, ci

        acc_m, acc_s, acc_ci = calc_ci("accuracy")
        f1_m, f1_s, f1_ci = calc_ci("macro_f1")
        wf1_m, _, _ = calc_ci("weighted_f1")
        bal_m, _, _ = calc_ci("balanced_accuracy")
        roc_m, _, _ = calc_ci("roc_auc")
        pr_m, _, _ = calc_ci("pr_auc")
        fpr_m, _, _ = calc_ci("fpr")
        fnr_m, _, _ = calc_ci("fnr")
        lat_m, _, _ = calc_ci("inference_latency_ms")
        params = group["parameter_count"].iloc[0]

        summary_records.append({
            "model": model_name,
            "accuracy_mean": round(acc_m, 4),
            "accuracy_std": round(acc_s, 4),
            "accuracy_ci95": round(acc_ci, 4),
            "macro_f1_mean": round(f1_m, 4),
            "macro_f1_std": round(f1_s, 4),
            "macro_f1_ci95": round(f1_ci, 4),
            "weighted_f1_mean": round(wf1_m, 4),
            "balanced_accuracy_mean": round(bal_m, 4),
            "roc_auc_mean": round(roc_m, 4),
            "pr_auc_mean": round(pr_m, 4),
            "fpr_mean": round(fpr_m, 4),
            "fnr_mean": round(fnr_m, 4),
            "inference_latency_ms": round(lat_m, 3),
            "parameter_count": int(params)
        })

    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join(RESULTS_TABLES_DIR, "table_model_comparison.csv")
    summary_df.to_csv(summary_csv, index=False)

    summary_json = os.path.join(RESULTS_METRICS_DIR, "model_summary_metrics.json")
    with open(summary_json, "w") as fp:
        json.dump(summary_records, fp, indent=2)

    # Separate Ablation Table (Phase 21)
    ablation_names = [
        "Ablation_A_TCN_Only",
        "Ablation_B_Transformer_Only",
        "Ablation_C_TCN_Transformer",
        "Proposed_TCN_Transformer_Attention",
        "Ablation_E_Proposed_Feature_Selection"
    ]
    ablation_df = summary_df[summary_df["model"].isin(ablation_names)].copy()
    desc_map = {
        "Ablation_A_TCN_Only": "A: Dilated Causal 1D Convolutions only",
        "Ablation_B_Transformer_Only": "B: Multi-Layer Transformer Encoder only",
        "Ablation_C_TCN_Transformer": "C: TCN + Transformer Encoder (no attention pooling)",
        "Proposed_TCN_Transformer_Attention": "D: Full Proposed Architecture (TCN + Transformer + Multi-Head Attn)",
        "Ablation_E_Proposed_Feature_Selection": "E: Proposed Architecture + Top-25 Univariate Feature Selection"
    }
    ablation_df["architecture_description"] = ablation_df["model"].map(desc_map)
    ablation_csv = os.path.join(RESULTS_TABLES_DIR, "table_ablation_study.csv")
    ablation_df.to_csv(ablation_csv, index=False)

    # Statistical Significance Tests (Proposed vs Baselines)
    prop_runs = runs_df[runs_df["model"] == "Proposed_TCN_Transformer_Attention"].sort_values("seed")
    stat_tests = {}
    for base_name in [m[0] for m in models_suite if m[0] != "Proposed_TCN_Transformer_Attention"]:
        b_runs = runs_df[runs_df["model"] == base_name].sort_values("seed")
        f1_prop = prop_runs["macro_f1"].values
        f1_base = b_runs["macro_f1"].values
        
        t_stat, t_pval = ttest_rel(f1_prop, f1_base)
        stat_tests[base_name] = {
            "paired_t_statistic": float(t_stat),
            "p_value": float(t_pval),
            "significant_at_005": bool(t_pval < 0.05),
            "proposed_f1_mean": float(np.mean(f1_prop)),
            "baseline_f1_mean": float(np.mean(f1_base)),
            "difference": float(np.mean(f1_prop) - np.mean(f1_base))
        }

    stat_json = os.path.join(RESULTS_METRICS_DIR, "statistical_significance_tests.json")
    with open(stat_json, "w") as fp:
        json.dump(stat_tests, fp, indent=2)

    # SHAP Feature Importance (Phase 15 / 21)
    print("\n[EXPLAINABILITY] Computing SHAP feature importance...")
    rf_for_shap = get_baseline_model("RF", seed=42).fit(X_train_neural[:10000], y_train_neural[:10000])
    explainer = shap.TreeExplainer(rf_for_shap)
    sample_shap_X = X_test[:300]
    shap_vals = explainer.shap_values(sample_shap_X)
    
    if isinstance(shap_vals, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
    elif len(shap_vals.shape) == 3:
        mean_abs_shap = np.abs(shap_vals).mean(axis=(0, 2))
    else:
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    shap_importance_records = [
        {"feature": feature_names[i], "mean_abs_shap": float(mean_abs_shap[i])}
        for i in range(len(feature_names))
    ]
    shap_importance_records = sorted(shap_importance_records, key=lambda x: x["mean_abs_shap"], reverse=True)

    shap_json = os.path.join(RESULTS_METRICS_DIR, "shap_feature_importances.json")
    with open(shap_json, "w") as fp:
        json.dump(shap_importance_records, fp, indent=2)

    print("\n" + "=" * 75)
    print("MODEL EVALUATION & ABLATION SUMMARY (Macro-F1 across 5 evaluation seeds)")
    print("=" * 75)
    for _, row in summary_df.sort_values("macro_f1_mean", ascending=False).iterrows():
        print(f"  {row['model']:40s} | F1: {row['macro_f1_mean']:.4f}±{row['macro_f1_ci95']:.4f} | Acc: {row['accuracy_mean']:.4f} | FPR: {row['fpr_mean']:.4f} | Latency: {row['inference_latency_ms']:.3f} ms")
    
    print("\nTop 5 Most Informative Telemetry Features (SHAP):")
    for feat in shap_importance_records[:5]:
        print(f"  - {feat['feature']:30s}: {feat['mean_abs_shap']:.4f}")
    
    print(f"\n[SAVED] All experiment runs: {runs_path}")
    print(f"[SAVED] Publication Table (Model Comparison): {summary_csv}")
    print(f"[SAVED] Publication Table (Ablation Study): {ablation_csv}")
    print(f"[SAVED] Statistical significance tests: {stat_json}")
    print(f"[SAVED] SHAP importances: {shap_json}")
    print("=" * 75)

if __name__ == "__main__":
    main()
