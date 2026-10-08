#!/usr/bin/env python3
"""
scripts/generate_latex_tables.py
================================
Generates publication-ready IEEE/ACM Transactions LaTeX tables
from empirical benchmark CSV files.
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(PROJECT_ROOT, "results/tables")

def generate_blockchain_tex():
    df = pd.read_csv(os.path.join(TABLES_DIR, "blockchain_benchmark.csv"))
    tex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Permissioned Blockchain Performance Across 10 Standardized Healthcare EHR Workloads}",
        "\\label{tab:blockchain_benchmark}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{llccccccc}",
        "\\hline",
        "\\textbf{Code} & \\textbf{EHR Workload Operation} & \\textbf{Type} & \\textbf{Success (\\%)} & \\textbf{Throughput (TPS)} & \\textbf{P50 (ms)} & \\textbf{P95 (ms)} & \\textbf{P99 (ms)} & \\textbf{Mean (ms)} \\\\",
        "\\hline"
    ]
    for _, r in df.iterrows():
        tex.append(f"{r['Workload_Code']} & {r['Workload_Name'].replace('_', ' ')} & {r['Operation_Type']} & {r['Success_Rate_Pct']:.1f} & {r['Throughput_TPS']:,.1f} & {r['P50_Latency_ms']:.4f} & {r['P95_Latency_ms']:.4f} & {r['P99_Latency_ms']:.4f} & {r['Mean_Latency_ms']:.4f} \\\\")
    tex.extend([
        "\\hline",
        "\\end{tabular}",
        "}",
        "\\end{table*}"
    ])
    out_path = os.path.join(TABLES_DIR, "blockchain_benchmark.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Saved: {out_path}")

def generate_security_tex():
    df = pd.read_csv(os.path.join(TABLES_DIR, "threat_matrix.csv"))
    tex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Healthcare Cyberattack Threat Mitigation and Empirical Defense Verification Matrix}",
        "\\label{tab:security_analysis}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{p{3.2cm}p{3.8cm}p{4.2cm}p{3.5cm}p{2.8cm}}",
        "\\hline",
        "\\textbf{Threat Scenario} & \\textbf{Conventional Baseline} & \\textbf{Proposed Quantum-Secure Control} & \\textbf{Empirical Test} & \\textbf{Detection Result} \\\\",
        "\\hline"
    ]
    for _, r in df.iterrows():
        tex.append(f"{r['Threat']} & {r['Baseline_Control']} & {r['Proposed_Control']} & {r['Test_Performed']} & {r['Detection']} \\\\")
    tex.extend([
        "\\hline",
        "\\end{tabular}",
        "}",
        "\\end{table*}"
    ])
    out_path = os.path.join(TABLES_DIR, "security_analysis.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Saved: {out_path}")

def generate_model_tex():
    df = pd.read_csv(os.path.join(TABLES_DIR, "model_metrics.csv"))
    tex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Comparison of Intrusion Detection Baseline Classifiers vs. Proposed HAB-IDS Ensemble}",
        "\\label{tab:model_comparison}",
        "\\resizebox{\\columnwidth}{!}{",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "\\textbf{Model Architecture} & \\textbf{Accuracy} & \\textbf{Macro-F1} & \\textbf{MCC} & \\textbf{ROC-AUC} & \\textbf{FPR (\\%)} & \\textbf{FNR (\\%)} \\\\",
        "\\hline"
    ]
    for _, r in df.iterrows():
        m_name = r['Model'].replace('_', ' ')
        tex.append(f"{m_name} & {r['Accuracy']:.5f} & {r['Macro_F1']:.5f} & {r['MCC']:.5f} & {r['ROC_AUC']:.5f} & {r['FPR']*100:.3f}\\% & {r['FNR']*100:.3f}\\% \\\\")
    tex.extend([
        "\\hline",
        "\\end{tabular}",
        "}",
        "\\end{table}"
    ])
    out_path = os.path.join(TABLES_DIR, "model_comparison.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Saved: {out_path}")

def generate_ablation_tex():
    df = pd.read_csv(os.path.join(TABLES_DIR, "ablation_results.csv"))
    tex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Component Ablation Study: Architectural Progression to Full HAB-IDS}",
        "\\label{tab:ablation_table}",
        "\\resizebox{\\columnwidth}{!}{",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "\\textbf{Pipeline Component Progression} & \\textbf{Accuracy} & \\textbf{Macro-F1} & \\textbf{MCC} & \\textbf{FPR (\\%)} & \\textbf{Size (MB)} \\\\",
        "\\hline"
    ]
    for _, r in df.iterrows():
        tex.append(f"{r['Stage']} & {r['Accuracy']:.5f} & {r['Macro_F1']:.5f} & {r['MCC']:.5f} & {r['FPR']*100:.3f}\\% & {r['Model_Size_MB']:.2f} \\\\")
    tex.extend([
        "\\hline",
        "\\end{tabular}",
        "}",
        "\\end{table}"
    ])
    out_path = os.path.join(TABLES_DIR, "ablation_table.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    generate_blockchain_tex()
    generate_security_tex()
    generate_model_tex()
    generate_ablation_tex()
