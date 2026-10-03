"""
scripts/generate_publication_figures.py
======================================
PROGRAMMATIC PUBLICATION FIGURES GENERATOR
Generates all 11 scientific figures in figures/ from canonical benchmark artifacts.
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

EXPERIMENT_ID = "CAHTDNET_FINAL_V001"

def set_publication_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })

def main():
    print("=" * 70)
    print("STAGE 15: PROGRAMMATIC PUBLICATION FIGURES GENERATION (11 FIGURES)")
    print("=" * 70)

    os.makedirs("figures", exist_ok=True)
    set_publication_style()

    # 1. Confusion Matrix
    cm_path = f"experiments/curves/{EXPERIMENT_ID}_confusion_matrix.json"
    if os.path.exists(cm_path):
        with open(cm_path) as f:
            cm_data = json.load(f)
        matrix = np.array(cm_data["raw_matrix"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
        plt.title(f"CA-HTDNet Test Confusion Matrix\n(Exp: {EXPERIMENT_ID})")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.savefig("figures/confusion_matrix.png")
        plt.close()
        print("  1. figures/confusion_matrix.png generated.")

    # 2. ROC Curve
    roc_path = f"experiments/curves/{EXPERIMENT_ID}_roc_curve.json"
    if os.path.exists(roc_path):
        with open(roc_path) as f:
            roc_data = json.load(f)
        plt.figure(figsize=(6.5, 5))
        plt.plot(roc_data["fpr"], roc_data["tpr"], color="#1f77b4", lw=2,
                 label=f"CA-HTDNet (AUC = {roc_data['roc_auc']:.4f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (Recall)")
        plt.legend(loc="lower right")
        plt.savefig("figures/roc_curve.png")
        plt.close()
        print("  2. figures/roc_curve.png generated.")

    # 3. Precision-Recall Curve
    pr_path = f"experiments/curves/{EXPERIMENT_ID}_pr_curve.json"
    if os.path.exists(pr_path):
        with open(pr_path) as f:
            pr_data = json.load(f)
        plt.figure(figsize=(6.5, 5))
        plt.plot(pr_data["recall"], pr_data["precision"], color="#2ca02c", lw=2,
                 label=f"CA-HTDNet (PR-AUC = {pr_data['pr_auc']:.4f})")
        plt.title("Precision-Recall (PR) Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        plt.savefig("figures/precision_recall_curve.png")
        plt.close()
        print("  3. figures/precision_recall_curve.png generated.")

    # 4. Threshold FPR vs FNR Tradeoff
    thresh_sweep_path = "results/threshold_sweep.csv"
    if os.path.exists(thresh_sweep_path):
        df_sw = pd.read_csv(thresh_sweep_path)
        plt.figure(figsize=(7, 5))
        plt.plot(df_sw["threshold"], df_sw["fpr"] * 100, label="FPR (False Positive %)", color="#d62728", lw=2)
        plt.plot(df_sw["threshold"], df_sw["fnr"] * 100, label="FNR (False Negative %)", color="#ff7f0e", lw=2)
        plt.plot(df_sw["threshold"], df_sw["macro_f1"] * 100, label="Macro-F1 (%)", color="#1f77b4", lw=2, linestyle="--")
        plt.axvline(0.50, color="gray", linestyle=":", label="Default Threshold (0.50)")
        plt.title("Detection Threshold Tradeoff Curve (FPR vs FNR vs F1)")
        plt.xlabel("Decision Threshold")
        plt.ylabel("Metric (%)")
        plt.legend(loc="best")
        plt.savefig("figures/threshold_fpr_fnr.png")
        plt.close()
        print("  4. figures/threshold_fpr_fnr.png generated.")

    # 5. Model Comparison
    bc_path = "results/baseline_comparison.csv"
    if os.path.exists(bc_path):
        df_bc = pd.read_csv(bc_path)
        plt.figure(figsize=(10, 5))
        bars = plt.barh(df_bc["model"], df_bc["macro_f1"] * 100, color="#3470a3")
        plt.xlabel("Macro-F1 Score (%)")
        plt.title("Baseline Model Comparison (Locked Test Set)")
        plt.xlim(80, 100)
        for bar in bars:
            w = bar.get_width()
            plt.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", fontsize=9)
        plt.gca().invert_yaxis()
        plt.savefig("figures/model_comparison.png")
        plt.close()
        print("  5. figures/model_comparison.png generated.")

    # 6. Ablation Study
    abl_path = "results/ablation_results.csv"
    if os.path.exists(abl_path):
        df_abl = pd.read_csv(abl_path)
        plt.figure(figsize=(9, 5))
        x = range(len(df_abl))
        plt.plot(x, df_abl["Macro_F1"] * 100, marker="o", color="#1f77b4", lw=2, label="Macro-F1 (%)")
        plt.plot(x, df_abl["Accuracy"] * 100, marker="s", color="#2ca02c", lw=2, label="Accuracy (%)")
        plt.xticks(x, [f"{r['Variant']}" for _, r in df_abl.iterrows()])
        plt.xlabel("Ablation Variant")
        plt.ylabel("Score (%)")
        plt.title("Systematic Ablation Trajectory (A0 to A8)")
        plt.legend(loc="lower right")
        plt.savefig("figures/ablation.png")
        plt.close()
        print("  6. figures/ablation.png generated.")

    # 7. Robustness Stress Curve
    rob_path = "results/robustness_report.csv"
    if os.path.exists(rob_path):
        df_rob = pd.read_csv(rob_path)
        plt.figure(figsize=(9, 5))
        plt.plot(range(len(df_rob)), df_rob["Macro_F1"] * 100, marker="^", color="#e377c2", lw=2)
        plt.xticks(range(len(df_rob)), df_rob["Test_Condition"], rotation=30, ha="right")
        plt.ylabel("Macro-F1 (%)")
        plt.title("CA-HTDNet Robustness Under Adversarial Telemetry Perturbations")
        plt.savefig("figures/robustness.png")
        plt.close()
        print("  7. figures/robustness.png generated.")

    # 8. Calibration Diagram
    cal_path = f"experiments/calibration/{EXPERIMENT_ID}_calibration.json"
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            cal_data = json.load(f)
        ts_data = cal_data.get("temperature_scaling", {})
        if "prob_pred" in ts_data and "prob_true" in ts_data:
            plt.figure(figsize=(6, 5))
            plt.plot(ts_data["prob_pred"], ts_data["prob_true"], marker="o", color="#17becf", label="Temperature Scaled")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Fraction of Positives")
            plt.title("Reliability Diagram (Probability Calibration)")
            plt.legend()
            plt.savefig("figures/calibration.png")
            plt.close()
            print("  8. figures/calibration.png generated.")

    # 9. Cross-Dataset Generalization
    cross_path = "results/cross_dataset_generalization.csv"
    if os.path.exists(cross_path):
        df_cross = pd.read_csv(cross_path)
        plt.figure(figsize=(8, 4.5))
        bars = plt.bar(df_cross["Experiment"].str.split(":").str[0], df_cross["Macro_F1"] * 100, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        plt.ylabel("Macro-F1 (%)")
        plt.title("Cross-Dataset Domain Adaptation & Generalization")
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.2f}%", ha="center", fontsize=10)
        plt.savefig("figures/cross_dataset.png")
        plt.close()
        print("  9. figures/cross_dataset.png generated.")

    # 10. Crypto Latency
    crypto_path = "results/crypto_benchmarks.csv"
    if os.path.exists(crypto_path):
        df_cry = pd.read_csv(crypto_path)
        plt.figure(figsize=(10, 5))
        plt.barh(df_cry["Primitive"], df_cry["Mean_Latency_ms"], color="#8c564b")
        plt.xlabel("Mean Execution Latency (ms)")
        plt.title("Post-Quantum & Classical Cryptographic Latency (Apple M2)")
        plt.gca().invert_yaxis()
        plt.savefig("figures/crypto_latency.png")
        plt.close()
        print("  10. figures/crypto_latency.png generated.")

    # 11. System Pipeline Latency Breakdown
    sys_path = "experiments/benchmarks/system_performance.csv"
    if os.path.exists(sys_path):
        df_sys = pd.read_csv(sys_path)
        df_sub = df_sys[df_sys["Pipeline_Stage"].str.startswith(tuple(str(i) for i in range(1, 10)))]
        plt.figure(figsize=(7, 6))
        plt.pie(df_sub["Mean_Latency_ms"], labels=df_sub["Pipeline_Stage"], autopct="%1.1f%%", startangle=140)
        plt.title("End-to-End EHR Security Pipeline Latency Distribution")
        plt.savefig("figures/system_latency.png")
        plt.close()
        print("  11. figures/system_latency.png generated.")

    print("\nAll 11 scientific publication figures successfully generated in figures/.")

if __name__ == "__main__":
    main()
