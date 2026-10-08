"""Evaluation Metrics Engine for HAB-IDS.

Calculates comprehensive cybersecurity classification metrics:
Accuracy, Macro-Precision, Macro-Recall, Macro-F1, Weighted-F1, MCC, ROC-AUC, PR-AUC, FPR, FNR,
per-class statistics, and confusion matrix totals.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Compute complete binary classification metrics including FPR, FNR, ROC-AUC, PR-AUC, MCC."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Rates
    fpr = float(fp / max(fp + tn, 1))
    fnr = float(fn / max(fn + tp, 1))
    tpr = float(tp / max(tp + fn, 1))
    tnr = float(tn / max(tn + fp, 1))

    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    roc_auc = 0.5
    pr_auc = 0.0
    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.5
            pr_auc = 0.0

    return {
        "Accuracy": round(acc, 5),
        "Macro_Precision": round(macro_prec, 5),
        "Macro_Recall": round(macro_rec, 5),
        "Macro_F1": round(macro_f1, 5),
        "Weighted_F1": round(weighted_f1, 5),
        "MCC": round(mcc, 5),
        "ROC_AUC": round(roc_auc, 5),
        "PR_AUC": round(pr_auc, 5),
        "FPR": round(fpr, 5),
        "FNR": round(fnr, 5),
        "TPR": round(tpr, 5),
        "TNR": round(tnr, 5),
        "Confusion_Matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        },
        "Total_Samples": int(len(y_true)),
    }


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compute multiclass evaluation metrics across attack families."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    roc_auc = 0.0
    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)

    return {
        "Accuracy": round(acc, 5),
        "Macro_Precision": round(macro_prec, 5),
        "Macro_Recall": round(macro_rec, 5),
        "Macro_F1": round(macro_f1, 5),
        "Weighted_F1": round(weighted_f1, 5),
        "MCC": round(mcc, 5),
        "ROC_AUC": round(roc_auc, 5),
        "Confusion_Matrix": cm,
        "Classification_Report": report,
        "Total_Samples": int(len(y_true)),
    }
