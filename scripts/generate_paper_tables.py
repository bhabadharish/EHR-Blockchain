import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
scripts/generate_paper_tables.py
================================
PUBLICATION PAPER TABLES EXPORTER (TABLES 1 TO 8)
Generates publication-ready CSV tables strictly from canonical result files.
"""

import os
import sys
import json
import shutil
import pandas as pd

def main():
    print("=" * 70)
    print("STAGE 14: EXPORTING PUBLICATION TABLES (TABLES 1 TO 8)")
    print("=" * 70)

    os.makedirs("paper_results", exist_ok=True)

    # Table 1: Dataset Statistics
    if os.path.exists("results/class_distribution.csv") and os.path.exists("results/feature_statistics.csv"):
        df_dist = pd.read_csv("results/class_distribution.csv")
        df_dist.to_csv("paper_results/Table_1_Dataset_Statistics.csv", index=False)
        print("  Table 1 (Dataset Statistics) exported.")

    # Table 2: Baseline Comparison
    if os.path.exists("results/baseline_comparison.csv"):
        df_bc = pd.read_csv("results/baseline_comparison.csv")
        df_bc.to_csv("paper_results/Table_2_Baseline_Comparison.csv", index=False)
        print("  Table 2 (Baseline Comparison) exported.")

    # Table 3: CAHTDNet Ablation
    if os.path.exists("results/ablation_results.csv"):
        df_abl = pd.read_csv("results/ablation_results.csv")
        df_abl.to_csv("paper_results/Table_3_CAHTDNet_Ablation.csv", index=False)
        print("  Table 3 (CA-HTDNet Ablation) exported.")

    # Table 4: Cross-Dataset Generalization
    if os.path.exists("results/cross_dataset_generalization.csv"):
        df_cross = pd.read_csv("results/cross_dataset_generalization.csv")
        df_cross.to_csv("paper_results/Table_4_CrossDataset.csv", index=False)
        print("  Table 4 (Cross-Dataset Generalization) exported.")

    # Table 5: Robustness Under Telemetry Perturbations
    if os.path.exists("results/robustness_report.csv"):
        df_rob = pd.read_csv("results/robustness_report.csv")
        df_rob.to_csv("paper_results/Table_5_Robustness.csv", index=False)
        print("  Table 5 (Robustness) exported.")

    # Table 6: Calibration Comparison
    if os.path.exists("results/calibration_comparison.csv"):
        df_cal = pd.read_csv("results/calibration_comparison.csv")
        df_cal.to_csv("paper_results/Table_6_Calibration.csv", index=False)
        print("  Table 6 (Calibration) exported.")

    # Table 7: Cryptographic Latency Benchmarks
    if os.path.exists("results/crypto_benchmarks.csv"):
        df_cry = pd.read_csv("results/crypto_benchmarks.csv")
        df_cry.to_csv("paper_results/Table_7_Crypto_Benchmarks.csv", index=False)
        print("  Table 7 (Crypto Benchmarks) exported.")

    # Table 8: System Performance & Blockchain
    if os.path.exists("experiments/benchmarks/system_performance.csv"):
        df_sys = pd.read_csv("experiments/benchmarks/system_performance.csv")
        df_sys.to_csv("paper_results/Table_8_System_Performance.csv", index=False)
        print("  Table 8 (System Performance) exported.")

    print("\nAll publication tables successfully generated in paper_results/.")

if __name__ == "__main__":
    main()
