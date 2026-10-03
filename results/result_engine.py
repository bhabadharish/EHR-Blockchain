"""
results/result_engine.py
========================
CANONICAL METRIC ENGINE & SINGLE SOURCE OF TRUTH FOR ALL BENCHMARKS.

Strict mathematical compliance:
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- FPR = FP / (FP + TN)
- FNR = FN / (FN + TP)
- Macro-F1 = (F1_class0 + F1_class1) / 2
- Macro-Precision = (Prec_class0 + Prec_class1) / 2
- Macro-Recall = (Rec_class0 + Rec_class1) / 2
- ROC-AUC = roc_auc_score(y_true, pos_probs)
- PR-AUC = average_precision_score(y_true, pos_probs)
- MCC = matthews_corrcoef(y_true, y_pred)

No other script in the repository may independently compute official benchmark metrics.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    brier_score_loss,
    classification_report
)

ENGINE_VERSION = "1.0.0-canonical"

def compute_canonical_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Optional[Union[np.ndarray, list]] = None,
    probabilities: Optional[Union[np.ndarray, list]] = None,
    threshold: float = 0.50,
    class_mapping: Optional[Dict[int, str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Computes all official metrics from raw true labels and predictions/probabilities.
    Enforces strict mathematical identity checks before returning.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    n_samples = len(y_true)
    if n_samples == 0:
        raise ValueError("y_true cannot be empty.")

    # 1. Resolve probabilities & predictions
    pos_probs = None
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim == 2:
            if probabilities.shape[1] == 2:
                pos_probs = probabilities[:, 1]
            else:
                pos_probs = probabilities[:, -1]
        elif probabilities.ndim == 1:
            pos_probs = probabilities
        else:
            raise ValueError(f"Unsupported probabilities shape: {probabilities.shape}")

    if y_pred is None:
        if pos_probs is None:
            raise ValueError("Either y_pred or probabilities must be provided.")
        y_pred = (pos_probs >= threshold).astype(np.int64)
    else:
        y_pred = np.asarray(y_pred, dtype=np.int64).ravel()

    if len(y_pred) != n_samples:
        raise ValueError(f"Length mismatch: y_true has {n_samples}, y_pred has {len(y_pred)}")

    # 2. Confusion Matrix & Components
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    total_cm = tn + fp + fn + tp
    if total_cm != n_samples:
        raise ValueError(f"Confusion matrix sum ({total_cm}) does not equal total samples ({n_samples})")

    # 3. Mathematical Identities & Ratios
    # FPR = FP / (FP + TN)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    # FNR = FN / (FN + TP)
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    # Accuracy = (TP + TN) / total
    acc_identity = float((tp + tn) / n_samples)
    acc = float(accuracy_score(y_true, y_pred))

    if abs(acc - acc_identity) > tolerance:
        raise AssertionError(
            f"Mathematical inconsistency: accuracy_score ({acc}) != (TP+TN)/total ({acc_identity})"
        )

    # 4. Precision, Recall, F1 Scores
    # Per-class scores
    per_class_p = [float(x) for x in precision_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)]
    per_class_r = [float(x) for x in recall_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)]
    per_class_f1 = [float(x) for x in f1_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)]

    macro_p = float((per_class_p[0] + per_class_p[1]) / 2.0)
    macro_r = float((per_class_r[0] + per_class_r[1]) / 2.0)
    macro_f1 = float((per_class_f1[0] + per_class_f1[1]) / 2.0)

    # Verification of macro averages
    sklearn_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if abs(macro_f1 - sklearn_macro_f1) > tolerance:
        raise AssertionError(f"Macro-F1 discrepancy: calculated {macro_f1} vs sklearn {sklearn_macro_f1}")

    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_p = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    # Binary positive class metrics
    binary_precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    binary_recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    binary_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # Correlation / Agreement
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    # 5. Curve Metrics (ROC-AUC, PR-AUC, Brier)
    if pos_probs is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, pos_probs))
        except Exception:
            roc_auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_true, pos_probs))
        except Exception:
            pr_auc = float(np.mean(y_true))
        brier = float(brier_score_loss(y_true, pos_probs))
    else:
        roc_auc = 0.5
        pr_auc = float(np.mean(y_true))
        brier = 0.0

    target_names = [class_mapping.get(0, "Normal"), class_mapping.get(1, "Attack")] if class_mapping else ["Normal", "Attack"]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)

    results = {
        "engine_version": ENGINE_VERSION,
        "n_samples": n_samples,
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "raw_matrix": [[tn, fp], [fn, tp]]
        },
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "binary_f1": binary_f1,
        "fpr": fpr,
        "fnr": fnr,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "per_class": {
            "class_0": {
                "name": target_names[0],
                "precision": per_class_p[0],
                "recall": per_class_r[0],
                "f1": per_class_f1[0],
                "support": tn + fp
            },
            "class_1": {
                "name": target_names[1],
                "precision": per_class_p[1],
                "recall": per_class_r[1],
                "f1": per_class_f1[1],
                "support": fn + tp
            }
        },
        "classification_report": report
    }

    if metadata:
        results["metadata"] = metadata

    return results

def compute_from_prediction_df(
    df: pd.DataFrame,
    threshold: float = 0.50,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Computes canonical metrics directly from a standardized predictions dataframe.
    Expected columns:
      - true_label
      - predicted_label (or generated from probability_class_1 >= threshold)
      - probability_class_1 (optional but recommended)
    """
    if "true_label" not in df.columns:
        raise ValueError("DataFrame must contain 'true_label' column.")

    y_true = df["true_label"].values
    y_pred = df["predicted_label"].values if "predicted_label" in df.columns else None
    probs = df["probability_class_1"].values if "probability_class_1" in df.columns else None

    return compute_canonical_metrics(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probs,
        threshold=threshold,
        metadata=metadata
    )

if __name__ == "__main__":
    # Self-test validation
    print("Testing results/result_engine.py...")
    y_t = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    p_1 = np.array([0.1, 0.2, 0.8, 0.05, 0.9, 0.85, 0.4, 0.95, 0.88, 0.92])
    metrics = compute_canonical_metrics(y_t, probabilities=p_1, threshold=0.50)
    print("Engine self-test passed!")
    print(f"Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}, FPR: {metrics['fpr']:.4f}, FNR: {metrics['fnr']:.4f}")
