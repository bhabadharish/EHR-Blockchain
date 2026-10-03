import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import lightgbm as lgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.harmonization import UNIFIED_NUMERICAL_FEATURES, UNIFIED_CATEGORICAL_FEATURES
from src.data.cleaning import UnifiedSecurityPreprocessor

def evaluate_preds(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0, 0], 0, 0, 0)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    return acc, macro_f1, macro_rec, macro_prec, fpr, fnr

def main():
    print("=" * 70)
    print("STAGE 9: CROSS-DATASET GENERALIZATION EVALUATION")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)
    os.makedirs("experiments/reports", exist_ok=True)

    # 1. Load pure individual dataset partitions
    df_cic = pd.read_parquet("data/splits/eval_ciciot_pure.parquet")
    df_edge = pd.read_parquet("data/splits/eval_edge_iiot_pure.parquet")
    df_fhir = pd.read_parquet("data/splits/eval_synthetic_fhir_pure.parquet")
    df_combined_test = pd.read_parquet("data/splits/test.parquet")

    # Fit preprocessor on pure CICIoT
    prep_cic = UnifiedSecurityPreprocessor()
    prep_cic.fit(df_cic, UNIFIED_NUMERICAL_FEATURES, UNIFIED_CATEGORICAL_FEATURES)
    X_cic_num, X_cic_cat = prep_cic.transform(df_cic)
    X_cic = np.hstack([X_cic_num, X_cic_cat])
    y_cic = df_cic["binary_label"].values

    # Fit preprocessor on pure Edge-IIoT
    prep_edge = UnifiedSecurityPreprocessor()
    prep_edge.fit(df_edge, UNIFIED_NUMERICAL_FEATURES, UNIFIED_CATEGORICAL_FEATURES)
    X_edge_num, X_edge_cat = prep_edge.transform(df_edge)
    X_edge = np.hstack([X_edge_num, X_edge_cat])
    y_edge = df_edge["binary_label"].values

    # Cross-dataset transform: Edge through CIC preprocessor and vice-versa
    X_edge_via_cic = np.hstack(prep_cic.transform(df_edge))
    X_cic_via_edge = np.hstack(prep_edge.transform(df_cic))
    X_fhir_via_cic = np.hstack(prep_cic.transform(df_fhir))

    # Preprocessor for combined
    with open("models/preprocessors/preprocessor.pkl", "rb") as f:
        prep_comb = pickle.load(f)
    X_test_comb = np.hstack(prep_comb.transform(df_combined_test))
    y_test_comb = df_combined_test["binary_label"].values

    # Experiment A: Train CICIoT2023 -> Test Edge-IIoTset
    print("\n--- Running Experiment A: Train CICIoT2023 -> Test Edge-IIoTset ---")
    model_cic = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
    model_cic.fit(X_cic[:50000], y_cic[:50000])
    p_edge = model_cic.predict(X_edge_via_cic[:20000])
    acc_a, f1_a, rec_a, prec_a, fpr_a, fnr_a = evaluate_preds(y_edge[:20000], p_edge)
    print(f"  Exp A Result | Acc: {acc_a*100:.2f}% | Macro-F1: {f1_a*100:.2f}% | Recall: {rec_a*100:.2f}%")

    # Experiment B: Train Edge-IIoTset -> Test CICIoT2023
    print("\n--- Running Experiment B: Train Edge-IIoTset -> Test CICIoT2023 ---")
    model_edge = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
    model_edge.fit(X_edge[:30000], y_edge[:30000])
    p_cic = model_edge.predict(X_cic_via_edge[:20000])
    acc_b, f1_b, rec_b, prec_b, fpr_b, fnr_b = evaluate_preds(y_cic[:20000], p_cic)
    print(f"  Exp B Result | Acc: {acc_b*100:.2f}% | Macro-F1: {f1_b*100:.2f}% | Recall: {rec_b*100:.2f}%")

    # Experiment C: Combined Training -> Held-out Combined Test
    print("\n--- Running Experiment C: Combined Training -> Held-out Combined Test ---")
    with open("models/baselines/LightGBM.pkl", "rb") as f:
        model_comb = pickle.load(f)
    p_comb = model_comb.predict(X_test_comb)
    acc_c, f1_c, rec_c, prec_c, fpr_c, fnr_c = evaluate_preds(y_test_comb, p_comb)
    print(f"  Exp C Result | Acc: {acc_c*100:.2f}% | Macro-F1: {f1_c*100:.2f}% | Recall: {rec_c*100:.2f}%")

    # Experiment D: Network Datasets -> Synthetic Healthcare Security Events (Zero-Day Healthcare Attack Detection)
    print("\n--- Running Experiment D: Network Model -> Synthetic Healthcare FHIR Telemetry ---")
    p_fhir = model_cic.predict(X_fhir_via_cic[:10000])
    acc_d, f1_d, rec_d, prec_d, fpr_d, fnr_d = evaluate_preds(df_fhir["binary_label"].values[:10000], p_fhir)
    print(f"  Exp D Result | Acc: {acc_d*100:.2f}% | Macro-F1: {f1_d*100:.2f}% | Recall: {rec_d*100:.2f}%")

    cross_dataset_results = [
        {
            "Experiment": "Exp A: CICIoT2023 -> Edge-IIoTset",
            "Train_Dataset": "CICIoT2023",
            "Test_Dataset": "Edge-IIoTset",
            "Accuracy": f"{acc_a*100:.2f}%",
            "Macro_F1": f"{f1_a*100:.2f}%",
            "Macro_Recall": f"{rec_a*100:.2f}%",
            "FPR": f"{fpr_a*100:.3f}%",
            "FNR": f"{fnr_a*100:.3f}%",
            "Domain_Shift_Impact": f"-{(acc_c - acc_a)*100:.2f}%"
        },
        {
            "Experiment": "Exp B: Edge-IIoTset -> CICIoT2023",
            "Train_Dataset": "Edge-IIoTset",
            "Test_Dataset": "CICIoT2023",
            "Accuracy": f"{acc_b*100:.2f}%",
            "Macro_F1": f"{f1_b*100:.2f}%",
            "Macro_Recall": f"{rec_b*100:.2f}%",
            "FPR": f"{fpr_b*100:.3f}%",
            "FNR": f"{fnr_b*100:.3f}%",
            "Domain_Shift_Impact": f"-{(acc_c - acc_b)*100:.2f}%"
        },
        {
            "Experiment": "Exp C: Multimodal Harmonized (Held-out)",
            "Train_Dataset": "Combined (CIC + Edge + FHIR)",
            "Test_Dataset": "Held-Out Combined Test",
            "Accuracy": f"{acc_c*100:.2f}%",
            "Macro_F1": f"{f1_c*100:.2f}%",
            "Macro_Recall": f"{rec_c*100:.2f}%",
            "FPR": f"{fpr_c*100:.3f}%",
            "FNR": f"{fnr_c*100:.3f}%",
            "Domain_Shift_Impact": "Baseline (0.0%)"
        },
        {
            "Experiment": "Exp D: Network Model -> FHIR Healthcare",
            "Train_Dataset": "CICIoT2023",
            "Test_Dataset": "Synthetic FHIR Telemetry",
            "Accuracy": f"{acc_d*100:.2f}%",
            "Macro_F1": f"{f1_d*100:.2f}%",
            "Macro_Recall": f"{rec_d*100:.2f}%",
            "FPR": f"{fpr_d*100:.3f}%",
            "FNR": f"{fnr_d*100:.3f}%",
            "Domain_Shift_Impact": f"-{(acc_c - acc_d)*100:.2f}%"
        }
    ]

    df_res = pd.DataFrame(cross_dataset_results)
    df_res.to_csv("results/cross_dataset_generalization.csv", index=False)
    with open("results/cross_dataset_generalization.json", "w") as f:
        json.dump(cross_dataset_results, f, indent=2)

    print("\nCROSS-DATASET GENERALIZATION COMPARISON TABLE:")
    print(df_res[["Experiment", "Train_Dataset", "Test_Dataset", "Accuracy", "Macro_F1", "Domain_Shift_Impact"]].to_string(index=False))

if __name__ == "__main__":
    main()
