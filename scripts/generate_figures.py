#!/usr/bin/env python3
"""
scripts/generate_figures.py
Phase 30: Generates publication-ready figures (Figures 1 to 16) in high-resolution (300 DPI)
using Matplotlib and Seaborn, strictly based on empirical benchmark results from results/.

Figure 1:  System Architecture Overview
Figure 2:  FHIR Data Pipeline (Ingestion -> Validation -> Minimization -> JCS)
Figure 3:  Cryptographic Workflow (ML-KEM-768 + AES-GCM + ML-DSA + SHA-3-256)
Figure 4:  Blockchain Workflow & Consent Lifecycle
Figure 5:  Security Threat Model & Defense Mapping
Figure 6:  Proposed TCN-Transformer-Attention Model Architecture
Figure 7:  Confusion Matrix (Multi-Class Cyber Threat Detection)
Figure 8:  ROC Curves across Threat Detection Models
Figure 9:  Precision-Recall Curves
Figure 10: PQC Latency Comparison (Classical vs Hybrid vs PQC)
Figure 11: Encryption & Decryption Overhead across Payload Sizes
Figure 12: Blockchain Latency and Throughput
Figure 13: Large-Scale System Scalability Curves (10k to 100k records)
Figure 14: Neural Architecture Ablation Study (Macro-F1 vs Latency)
Figure 15: SHAP Feature Importance (Top Telemetry Predictors)
Figure 16: End-to-End Hospital A -> Hospital B Exchange Latency Breakdown
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
RESULTS_METRICS_DIR = os.path.join(PROJECT_ROOT, "results", "metrics")
RESULTS_TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
RESULTS_EXP_DIR = os.path.join(PROJECT_ROOT, "results", "experiments")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Publication formatting parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#E0E0E0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5

def plot_fig1_architecture():
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.axis('off')

    # Color palette
    c_fhir = '#E3F2FD'
    c_pqc = '#E8F5E9'
    c_bc = '#FFF3E0'
    c_ai = '#F3E5F5'
    c_border = '#37474F'

    # Tier 1: Ingestion & FHIR Gateway
    box1 = patches.FancyBboxPatch((0.05, 0.65), 0.25, 0.28, boxstyle="round,pad=0.03", fc=c_fhir, ec=c_border, lw=1.5)
    ax.add_patch(box1)
    ax.text(0.175, 0.90, "FHIR Gateway & Ingestion", fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(0.175, 0.82, "• HL7 FHIR R4 Validation\n• Minimization (Policies)\n• RFC 8785 JCS Canonicalization\n• FastAPI REST Services", fontsize=9, ha='center', va='center')

    # Tier 2: Crypto-Agility Engine
    box2 = patches.FancyBboxPatch((0.38, 0.65), 0.25, 0.28, boxstyle="round,pad=0.03", fc=c_pqc, ec=c_border, lw=1.5)
    ax.add_patch(box2)
    ax.text(0.505, 0.90, "Crypto-Agility Engine", fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(0.505, 0.82, "• ML-KEM-768/1024 (FIPS 203)\n• ML-DSA-65/87 (FIPS 204)\n• AES-256-GCM (NIST SP 800-38D)\n• SHA-3-256 Integrity Digest", fontsize=9, ha='center', va='center')

    # Tier 3: Off-Chain Storage & Ledger
    box3 = patches.FancyBboxPatch((0.70, 0.65), 0.25, 0.28, boxstyle="round,pad=0.03", fc=c_bc, ec=c_border, lw=1.5)
    ax.add_patch(box3)
    ax.text(0.825, 0.90, "Storage & Blockchain", fontsize=11, fontweight='bold', ha='center', va='center')
    ax.text(0.825, 0.82, "• Off-Chain Encrypted Vault\n• Zero-PHI Hyperledger Ledger\n• Dynamic Consent State\n• Immutable Audit Trails", fontsize=9, ha='center', va='center')

    # Tier 4: Intelligent Threat Detection
    box4 = patches.FancyBboxPatch((0.20, 0.15), 0.60, 0.35, boxstyle="round,pad=0.03", fc=c_ai, ec=c_border, lw=1.5)
    ax.add_patch(box4)
    ax.text(0.50, 0.44, "Intelligent Cyber Threat Detection & Explainability Layer", fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(0.50, 0.30, "• Multi-Dilation Dilated Causal TCN Blocks (Receptive Field)\n• Multi-Layer Contextual Transformer Encoders (Temporal Dependencies)\n• Multi-Head Attentive Feature Aggregation\n• SHAP Game-Theoretic Feature Attribution & Explainability", fontsize=10, ha='center', va='center')

    # Flow arrows
    ax.annotate('', xy=(0.38, 0.79), xytext=(0.30, 0.79), arrowprops=dict(facecolor='#37474F', shrink=0.05, width=1.5, headwidth=8))
    ax.annotate('', xy=(0.70, 0.79), xytext=(0.63, 0.79), arrowprops=dict(facecolor='#37474F', shrink=0.05, width=1.5, headwidth=8))
    ax.annotate('', xy=(0.50, 0.50), xytext=(0.50, 0.65), arrowprops=dict(facecolor='#37474F', shrink=0.05, width=1.5, headwidth=8))

    ax.set_title("Figure 1: Architectural Blueprint of the Crypto-Agile FHIR-Blockchain System", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1_system_architecture.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[GENERATED] {path}")

def plot_fig2_fhir_pipeline():
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.axis('off')
    steps = ["Clinical EHR\n(Synthea/MIMIC-IV)", "Pydantic FHIR R4\nStrict Validation", "Contextual Data\nMinimization", "RFC 8785 JCS\nCanonicalization", "Cryptographic\nEnvelope"]
    colors = ['#E1F5FE', '#B3E5FC', '#81D4FA', '#4FC3F7', '#29B6F6']
    
    for i, (s, c) in enumerate(zip(steps, colors)):
        x = 0.05 + i * 0.19
        box = patches.FancyBboxPatch((x, 0.3), 0.15, 0.45, boxstyle="round,pad=0.02", fc=c, ec='#0277BD', lw=1.2)
        ax.add_patch(box)
        ax.text(x + 0.075, 0.525, s, fontsize=9, fontweight='bold', ha='center', va='center')
        if i < len(steps) - 1:
            ax.annotate('', xy=(x + 0.19, 0.525), xytext=(x + 0.15, 0.525),
                        arrowprops=dict(facecolor='#0277BD', shrink=0.05, width=1.5, headwidth=6))

    ax.set_title("Figure 2: FHIR Ingestion, Privacy Minimization, and Canonical Serialization Pipeline", fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_fhir_pipeline.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[GENERATED] {path}")

def plot_fig3_crypto_workflow():
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.axis('off')
    stages = [
        ("Canonical FHIR", "RFC 8785 byte string"),
        ("SHA-3-256 Digest", "FIPS 202 hash"),
        ("ML-KEM-768", "FIPS 203 key encap"),
        ("AES-256-GCM", "NIST SP 800-38D enc"),
        ("ML-DSA-65", "FIPS 204 digital sign"),
        ("Encrypted Package", "Tamper-evident JSON")
    ]
    colors = ['#E8EAF6', '#C5CAE9', '#9FA8DA', '#7986CB', '#5C6BC0', '#3F51B5']
    text_colors = ['#1A237E', '#1A237E', '#1A237E', '#FFFFFF', '#FFFFFF', '#FFFFFF']

    for i, ((name, sub), c, tc) in enumerate(zip(stages, colors, text_colors)):
        x = 0.03 + i * 0.16
        box = patches.FancyBboxPatch((x, 0.25), 0.13, 0.5, boxstyle="round,pad=0.02", fc=c, ec='#1A237E', lw=1.2)
        ax.add_patch(box)
        ax.text(x + 0.065, 0.55, name, fontsize=9, fontweight='bold', ha='center', va='center', color=tc)
        ax.text(x + 0.065, 0.40, sub, fontsize=7.5, ha='center', va='center', color=tc)
        if i < len(stages) - 1:
            ax.annotate('', xy=(x + 0.16, 0.5), xytext=(x + 0.13, 0.5),
                        arrowprops=dict(facecolor='#1A237E', shrink=0.05, width=1.2, headwidth=5))

    ax.set_title("Figure 3: Post-Quantum Cryptographic Packaging & Integrity Attestation Workflow", fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_cryptographic_workflow.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[GENERATED] {path}")

def plot_fig4_blockchain_workflow():
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.axis('off')
    stages = [
        ("1. Consent Creation", "Patient authorizes orgs\n& purpose on ledger"),
        ("2. Reference Commit", "Store off-chain locator\n& SHA-3 digest only"),
        ("3. Query & ABAC", "Smart contract verifies\nactive consent state"),
        ("4. Revocation", "Patient revokes consent;\ninstant registry update"),
        ("5. Immutable Audit", "Append-only blocks log\nevery access attempt")
    ]
    for i, (title, desc) in enumerate(stages):
        x = 0.04 + i * 0.19
        box = patches.FancyBboxPatch((x, 0.2), 0.15, 0.55, boxstyle="round,pad=0.02", fc='#FFF8E1', ec='#FF8F00', lw=1.2)
        ax.add_patch(box)
        ax.text(x + 0.075, 0.60, title, fontsize=9, fontweight='bold', ha='center', va='center', color='#E65100')
        ax.text(x + 0.075, 0.40, desc, fontsize=7.5, ha='center', va='center', color='#333333')
        if i < len(stages) - 1:
            ax.annotate('', xy=(x + 0.19, 0.48), xytext=(x + 0.15, 0.48),
                        arrowprops=dict(facecolor='#FF8F00', shrink=0.05, width=1.2, headwidth=5))

    ax.set_title("Figure 4: Permissioned Blockchain Smart Contract Lifecycle (Zero PHI On-Chain)", fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_blockchain_workflow.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[GENERATED] {path}")

def plot_fig5_threat_model():
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    ax.axis('off')
    attackers = [
        ("A1: Quantum Eavesdropper", "HNDL passive sniffing", "ML-KEM lattice key exchange", '#E8F5E9', '#2E7D32'),
        ("A2: Unauthorized Worker", "Snooping unassigned records", "ABAC & smart contract consent", '#E3F2FD', '#1565C0'),
        ("A3: Malicious Org", "Revocation bypass attempt", "Real-time revocation ledger query", '#FFF3E0', '#E65100'),
        ("A4: Storage Attacker", "Off-chain blob exfiltration", "Full AES-256-GCM encryption at rest", '#FCE4EC', '#C2185B'),
        ("A5: Replay / Tamperer", "Replay packet / bit flip", "300s freshness window & GCM tag", '#F3E5F5', '#7B1FA2')
    ]
    for i, (atk, vec, defn, bg, fg) in enumerate(attackers):
        y = 0.8 - i * 0.16
        box_atk = patches.FancyBboxPatch((0.05, y), 0.28, 0.12, boxstyle="round,pad=0.01", fc=bg, ec=fg, lw=1)
        ax.add_patch(box_atk)
        ax.text(0.19, y + 0.06, atk, fontsize=8.5, fontweight='bold', ha='center', va='center', color=fg)

        ax.annotate('', xy=(0.42, y + 0.06), xytext=(0.33, y + 0.06), arrowprops=dict(facecolor='#666666', width=1, headwidth=4))
        ax.text(0.375, y + 0.09, "Vector", fontsize=7, ha='center', va='center', color='#666666')

        box_vec = patches.FancyBboxPatch((0.42, y), 0.25, 0.12, boxstyle="round,pad=0.01", fc='#EEEEEE', ec='#9E9E9E', lw=1)
        ax.add_patch(box_vec)
        ax.text(0.545, y + 0.06, vec, fontsize=8, ha='center', va='center', color='#333333')

        ax.annotate('', xy=(0.74, y + 0.06), xytext=(0.67, y + 0.06), arrowprops=dict(facecolor='#666666', width=1, headwidth=4))
        ax.text(0.705, y + 0.09, "Mitigation", fontsize=7, ha='center', va='center', color='#666666')

        box_def = patches.FancyBboxPatch((0.74, y), 0.22, 0.12, boxstyle="round,pad=0.01", fc=bg, ec=fg, lw=1)
        ax.add_patch(box_def)
        ax.text(0.85, y + 0.06, defn, fontsize=8, fontweight='bold', ha='center', va='center', color=fg)

    ax.set_title("Figure 5: Threat Model: Attackers, Attack Vectors, and Layered Mitigations", fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig5_threat_model.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[GENERATED] {path}")

def plot_fig6_model_architecture():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axis('off')
    layers = [
        ("Input Telemetry", "(batch, 50 features)"),
        ("1D Conv / TCN Block 1", "dilation=1, c=64, GELU"),
        ("1D Conv / TCN Block 2", "dilation=2, c=64, GELU"),
        ("1D Conv / TCN Block 3", "dilation=4, d_model=64"),
        ("Transformer Encoder", "2 layers, 4 heads, ff=128"),
        ("Multi-Head Attention", "attentive context pooling"),
        ("Global Avg+Max Pool", "d_model * 2 = 128"),
        ("Classification Head", "Dense -> 15 Classes")
    ]
    for i, (name, dim) in enumerate(layers):
        y = 0.88 - i * 0.11
        box = patches.FancyBboxPatch((0.25, y), 0.5, 0.08, boxstyle="round,pad=0.01", fc='#EDE7F6', ec='#512DA8', lw=1.2)
        ax.add_patch(box)
        ax.text(0.40, y + 0.04, name, fontsize=9, fontweight='bold', ha='center', va='center', color='#311B92')
        ax.text(0.62, y + 0.04, dim, fontsize=8, ha='center', va='center', color='#4527A0')
        if i < len(layers) - 1:
            ax.annotate('', xy=(0.50, y - 0.03), xytext=(0.50, y), arrowprops=dict(facecolor='#512DA8', width=1.2, headwidth=4))

    ax.set_title("Figure 6: Proposed TCN-Transformer-Attention Neural Architecture for Threat Detection", fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig6_model_architecture.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[GENERATED] {path}")

def plot_fig7_confusion_matrix():
    cm_path = os.path.join(RESULTS_METRICS_DIR, "confusion_matrices.json")
    if not os.path.exists(cm_path):
        print(f"[WAIT] {cm_path} not yet available.")
        return
    with open(cm_path, "r") as fp:
        cms = json.load(fp)

    key = "Proposed_TCN_Transformer_Attention"
    if key not in cms:
        key = list(cms.keys())[0]

    cm = np.array(cms[key])
    # Normalize by row
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(cm_norm, annot=False, cmap='Blues', cbar=True, square=True)
    plt.title(f"Figure 7: Confusion Matrix (Normalized) - {key.replace('_', ' ')}", fontsize=11, fontweight='bold', pad=12)
    plt.xlabel("Predicted Class Index", fontsize=10)
    plt.ylabel("True Class Index", fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig7_confusion_matrix.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[GENERATED] {path}")

def plot_fig8_and_9_roc_pr():
    # Model comparison summary
    csv_path = os.path.join(RESULTS_TABLES_DIR, "table_model_comparison.csv")
    if not os.path.exists(csv_path):
        print(f"[WAIT] {csv_path} not yet available.")
        return
    df = pd.read_csv(csv_path)

    # Figure 8: ROC-AUC Comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    df_sorted = df.sort_values("roc_auc_mean", ascending=True)
    bars = ax.barh(df_sorted["model"].str.replace("_", " "), df_sorted["roc_auc_mean"], color='#1976D2', height=0.6)
    min_roc = max(0.0, df_sorted["roc_auc_mean"].min() - 0.05)
    ax.set_xlim(min_roc, 1.05)
    ax.set_xlabel("Macro-Averaged ROC-AUC", fontsize=10)
    ax.set_title("Figure 8: Receiver Operating Characteristic (ROC-AUC) across Evaluated Architectures", fontsize=11, fontweight='bold', pad=12)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va='center', fontsize=8)
    plt.tight_layout()
    path8 = os.path.join(FIGURES_DIR, "fig8_roc_curves.png")
    plt.savefig(path8, dpi=300)
    plt.close()
    print(f"[GENERATED] {path8}")

    # Figure 9: PR-AUC Comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    df_sorted = df.sort_values("pr_auc_mean", ascending=True)
    bars = ax.barh(df_sorted["model"].str.replace("_", " "), df_sorted["pr_auc_mean"], color='#388E3C', height=0.6)
    min_pr = max(0.0, df_sorted["pr_auc_mean"].min() - 0.05)
    ax.set_xlim(min_pr, 1.05)
    ax.set_xlabel("Macro-Averaged Precision-Recall AUC (PR-AUC)", fontsize=10)
    ax.set_title("Figure 9: Precision-Recall Area Under Curve (PR-AUC) across Evaluated Architectures", fontsize=11, fontweight='bold', pad=12)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va='center', fontsize=8)
    plt.tight_layout()
    path9 = os.path.join(FIGURES_DIR, "fig9_pr_curves.png")
    plt.savefig(path9, dpi=300)
    plt.close()
    print(f"[GENERATED] {path9}")

def plot_fig10_and_11_crypto():
    csv_path = os.path.join(RESULTS_TABLES_DIR, "table_crypto_benchmark.csv")
    if not os.path.exists(csv_path):
        print(f"[WAIT] {csv_path} not yet available.")
        return
    df = pd.read_csv(csv_path)

    # Figure 10: PQC Latency Comparison (at 10KB payload)
    df_10k = df[df["payload_target_kb"] == 10].copy()
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    suites = df_10k["suite"]
    x = np.arange(len(suites))
    width = 0.35

    ax.bar(x - width/2, df_10k["encryption_ms_mean"], width, label='Encryption / Packaging (ms)', color='#1565C0')
    ax.bar(x + width/2, df_10k["decryption_ms_mean"], width, label='Decryption / Unpackaging (ms)', color='#D84315')
    ax.set_ylabel("Latency (ms)", fontsize=10)
    ax.set_title("Figure 10: Cryptographic Primitive Latency (Classical vs Hybrid vs Post-Quantum at 10KB)", fontsize=11, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(suites, rotation=15, ha='right', fontsize=9)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    path10 = os.path.join(FIGURES_DIR, "fig10_pqc_latency_comparison.png")
    plt.savefig(path10, dpi=300)
    plt.close()
    print(f"[GENERATED] {path10}")

    # Figure 11: Total Overhead across payload sizes
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    for s in df["suite"].unique():
        sub = df[df["suite"] == s].sort_values("payload_target_kb")
        ax.plot(sub["payload_target_kb"], sub["total_latency_ms_mean"], marker='o', label=s, lw=1.8)
    ax.set_xlabel("Payload Size (KB)", fontsize=10)
    ax.set_ylabel("Total Packaging & Unpackaging Roundtrip (ms)", fontsize=10)
    ax.set_title("Figure 11: Cryptographic Processing Latency vs EHR Payload Size", fontsize=11, fontweight='bold', pad=12)
    ax.grid(True)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    path11 = os.path.join(FIGURES_DIR, "fig11_encryption_decryption_overhead.png")
    plt.savefig(path11, dpi=300)
    plt.close()
    print(f"[GENERATED] {path11}")

def plot_fig12_blockchain():
    csv_path = os.path.join(RESULTS_TABLES_DIR, "table_blockchain_ablation.csv")
    if not os.path.exists(csv_path):
        print(f"[WAIT] {csv_path} not yet available.")
        return
    df = pd.read_csv(csv_path)

    fig, ax1 = plt.subplots(figsize=(9, 4.5), dpi=300)
    modes = df["configuration"].str.replace("_", " ")
    x = np.arange(len(modes))

    ax1.set_xlabel("Blockchain Security Configuration", fontsize=10)
    ax1.set_ylabel("Transaction Latency (ms)", color='#B71C1C', fontsize=10)
    b1 = ax1.bar(x - 0.2, df["tx_latency_ms_mean"], 0.4, label='Latency (ms)', color='#EF5350')
    ax1.tick_params(axis='y', labelcolor='#B71C1C')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Throughput (Transactions / sec)", color='#0D47A1', fontsize=10)
    b2 = ax2.bar(x + 0.2, df["throughput_tps_mean"], 0.4, label='Throughput (TPS)', color='#42A5F5')
    ax2.tick_params(axis='y', labelcolor='#0D47A1')

    ax1.set_xticks(x)
    ax1.set_xticklabels(modes, rotation=15, ha='right', fontsize=9)
    ax1.set_title("Figure 12: Permissioned Blockchain Transaction Latency and Throughput", fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    path12 = os.path.join(FIGURES_DIR, "fig12_blockchain_latency.png")
    plt.savefig(path12, dpi=300)
    plt.close()
    print(f"[GENERATED] {path12}")

def plot_fig13_scalability():
    csv_path = os.path.join(RESULTS_TABLES_DIR, "table_scalability_benchmarks.csv")
    if not os.path.exists(csv_path):
        print(f"[WAIT] {csv_path} not yet available.")
        return
    df = pd.read_csv(csv_path)

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    scales = df["scale_records"] / 1000.0

    ax1.set_xlabel("Dataset Scale (Thousands of Synthetic Patient Records)", fontsize=10)
    ax1.set_ylabel("Throughput (records / sec)", color='#1B5E20', fontsize=10)
    l1 = ax1.plot(scales, df["fhir_throughput_rps"], marker='s', color='#2E7D32', lw=2, label='FHIR Minimization (rec/s)')
    l2 = ax1.plot(scales, df["encryption_throughput_rps"], marker='o', color='#388E3C', lw=2, label='PQC Encryption (rec/s)')
    ax1.tick_params(axis='y', labelcolor='#1B5E20')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Encrypted Storage Footprint (MB)", color='#4A148C', fontsize=10)
    l3 = ax2.plot(scales, df["storage_size_mb"], marker='^', color='#8E24AA', lw=2, linestyle='--', label='Storage Size (MB)')
    ax2.tick_params(axis='y', labelcolor='#4A148C')

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', frameon=True, fontsize=8.5)
    ax1.set_title("Figure 13: Large-Scale System Scalability and Storage Growth (10k to 100k Records)", fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    path13 = os.path.join(FIGURES_DIR, "fig13_scalability.png")
    plt.savefig(path13, dpi=300)
    plt.close()
    print(f"[GENERATED] {path13}")

def plot_fig14_ablation():
    csv_path = os.path.join(RESULTS_TABLES_DIR, "table_ablation_study.csv")
    if not os.path.exists(csv_path):
        print(f"[WAIT] {csv_path} not yet available.")
        return
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    models = df["model"].str.replace("Ablation_", "").str.replace("_", " ")
    x = np.arange(len(models))

    ax.bar(x, df["macro_f1_mean"], yerr=df["macro_f1_ci95"], capsize=4, color='#5E35B1', alpha=0.85, width=0.55)
    ax.set_ylabel("Macro-Averaged F1-Score (±95% CI)", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=8.5)
    ax.set_title("Figure 14: Neural Architecture Component Ablation Study (Macro-F1 across 5 Seeds)", fontsize=11, fontweight='bold', pad=12)
    
    for i, v in enumerate(df["macro_f1_mean"]):
        ax.text(i, v + 0.02, f"{v:.4f}", ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    path14 = os.path.join(FIGURES_DIR, "fig14_ablation_results.png")
    plt.savefig(path14, dpi=300)
    plt.close()
    print(f"[GENERATED] {path14}")

def plot_fig15_shap():
    json_path = os.path.join(RESULTS_METRICS_DIR, "shap_feature_importances.json")
    if not os.path.exists(json_path):
        print(f"[WAIT] {json_path} not yet available.")
        return
    with open(json_path, "r") as fp:
        shap_data = json.load(fp)

    df_shap = pd.DataFrame(shap_data[:15]).sort_values("mean_abs_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    bars = ax.barh(df_shap["feature"], df_shap["mean_abs_shap"], color='#00897B', height=0.6)
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Prediction Magnitude)", fontsize=10)
    ax.set_title("Figure 15: Top 15 Telemetry Features by Game-Theoretic SHAP Importance", fontsize=11, fontweight='bold', pad=12)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va='center', fontsize=8)
    plt.tight_layout()
    path15 = os.path.join(FIGURES_DIR, "fig15_shap_feature_importance.png")
    plt.savefig(path15, dpi=300)
    plt.close()
    print(f"[GENERATED] {path15}")

def plot_fig16_e2e_latency():
    csv_path = os.path.join(RESULTS_TABLES_DIR, "table_e2e_exchange.csv")
    if not os.path.exists(csv_path):
        print(f"[WAIT] {csv_path} not yet available.")
        return
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    clean_steps = df["step"].str.replace("step_", "").str.replace("_", " ").str.title()
    bars = ax.barh(clean_steps, df["mean_latency_ms"], xerr=df["std_latency_ms"], capsize=3, color='#F57C00', height=0.6)
    ax.set_xlabel("Latency (ms)", fontsize=10)
    ax.set_title("Figure 16: Complete Hospital A -> Hospital B End-to-End Exchange Latency Breakdown", fontsize=11, fontweight='bold', pad=12)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.05, bar.get_y() + bar.get_height()/2, f"{w:.3f} ms", va='center', fontsize=8)
    plt.tight_layout()
    path16 = os.path.join(FIGURES_DIR, "fig16_end_to_end_latency.png")
    plt.savefig(path16, dpi=300)
    plt.close()
    print(f"[GENERATED] {path16}")

def main():
    print("=" * 75)
    print("PHASE 30: AUTOMATIC PUBLICATION FIGURE GENERATION (FIGURES 1 TO 16)")
    print("=" * 75)
    plot_fig1_architecture()
    plot_fig2_fhir_pipeline()
    plot_fig3_crypto_workflow()
    plot_fig4_blockchain_workflow()
    plot_fig5_threat_model()
    plot_fig6_model_architecture()
    plot_fig7_confusion_matrix()
    plot_fig8_and_9_roc_pr()
    plot_fig10_and_11_crypto()
    plot_fig12_blockchain()
    plot_fig13_scalability()
    plot_fig14_ablation()
    plot_fig15_shap()
    plot_fig16_e2e_latency()
    print("=" * 75)
    print(f"[COMPLETE] Publication figures generated in: {FIGURES_DIR}")
    print("=" * 75)

if __name__ == "__main__":
    main()
