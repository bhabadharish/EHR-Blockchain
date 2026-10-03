import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.metrics import evaluate_classification_performance

os.makedirs("results", exist_ok=True)
os.makedirs("docs/figures", exist_ok=True)
sns.set_theme(style="darkgrid")

def main():
    print("=" * 70)
    print("STAGE 14: LOCKED TEST EVALUATION & RESEARCH FIGURE GENERATION")
    print("=" * 70)

    # 1. Load untouched Test Partition
    data = np.load("data/processed/processed_arrays.npz")
    X_test_num = data["X_test_num"]
    X_test_cat = data["X_test_cat"]
    X_test_full = np.hstack([X_test_num, X_test_cat])
    y_test = data["y_test"]

    print(f"Loaded {len(y_test)} untouched test samples for one-time locked evaluation.")

    # Load baseline models
    baseline_names = [
        "Logistic_Regression", "Decision_Tree", "Random_Forest",
        "Extra_Trees", "SVM", "XGBoost", "LightGBM", "CatBoost", "MLP"
    ]

    all_test_metrics = []

    for name in baseline_names:
        model_path = f"models/baselines/{name}.pkl"
        if not os.path.exists(model_path):
            continue
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            
        t0 = time.time()
        y_prob = model.predict_proba(X_test_full)
        inf_time = time.time() - t0
        y_pred = np.argmax(y_prob, axis=1)

        m = evaluate_classification_performance(y_test, y_pred, y_prob, inference_time=inf_time)
        m["Model"] = name.replace("_", " ")
        all_test_metrics.append(m)

    # Evaluate Proposed CA-HTDNet on Test Set
    # Load GBDT hints on Test Set
    with open("models/baselines/LightGBM.pkl", "rb") as f:
        lgb_m = pickle.load(f)
    with open("models/baselines/CatBoost.pkl", "rb") as f:
        cat_m = pickle.load(f)

    p_lgb_test = lgb_m.predict_proba(X_test_full)
    p_cat_test = cat_m.predict_proba(X_test_full)
    gbdt_test = np.hstack([p_lgb_test, p_cat_test]).astype(np.float32)

    import torch
    from src.models.ca_htdnet import CA_HTDNet
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ca_model = CA_HTDNet(num_numerical=X_test_num.shape[1]).to(device)
    ca_model.load_state_dict(torch.load("models/proposed/ca_htdnet.pt", map_location=device))
    ca_model.eval()

    # Load frozen validation threshold
    with open("models/proposed/threshold.json") as f:
        thresh_meta = json.load(f)
    opt_thresh = thresh_meta["optimal_threshold"]

    t0 = time.time()
    with torch.no_grad():
        bx = torch.tensor(X_test_num, dtype=torch.float32, device=device)
        bg = torch.tensor(gbdt_test, dtype=torch.float32, device=device)
        out = ca_model(bx, bg)
        probs_ca = torch.softmax(out["calibrated_logits"], dim=-1).cpu().numpy()
        inf_time_ca = time.time() - t0
        preds_ca = (probs_ca[:, 1] >= opt_thresh).astype(int)

    m_ca = evaluate_classification_performance(y_test, preds_ca, probs_ca, inference_time=inf_time_ca)
    m_ca["Model"] = "CA-HTDNet (Proposed)"
    all_test_metrics.append(m_ca)

    # 2. Build Final Table
    summary_rows = []
    for m in all_test_metrics:
        summary_rows.append({
            "Model": m["Model"],
            "Accuracy": f"{m['Accuracy']*100:.2f}%",
            "Macro Precision": f"{m['Macro_Precision']*100:.2f}%",
            "Macro Recall": f"{m['Macro_Recall']*100:.2f}%",
            "Macro F1": f"{m['Macro_F1']*100:.2f}%",
            "ROC-AUC": f"{m['ROC_AUC']:.4f}",
            "PR-AUC": f"{m['PR_AUC']:.4f}",
            "FPR": f"{m['FPR']*100:.3f}%",
            "FNR": f"{m['FNR']*100:.3f}%"
        })

    df_final = pd.DataFrame(summary_rows)
    df_final.to_csv("results/final_results.csv", index=False)
    with open("results/final_results.json", "w") as f:
        json.dump(all_test_metrics, f, indent=2)

    with open("results/final_results.md", "w") as f:
        f.write("# Final Locked Test Evaluation Results Table\n\n")
        f.write(df_final.to_markdown(index=False))
        f.write("\n")

    print("\nFINAL LOCKED TEST RESULTS TABLE:")
    print(df_final.to_string(index=False))

    # 3. Generate Research Figures
    print("\nGenerating publication-quality figures in docs/figures/...")

    # Fig 1: Model Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    df_plot = df_final.copy()
    df_plot["Acc_Num"] = df_plot["Accuracy"].str.rstrip("%").astype(float)
    df_plot["F1_Num"] = df_plot["Macro F1"].str.rstrip("%").astype(float)
    
    x = np.arange(len(df_plot))
    width = 0.35
    plt.bar(x - width/2, df_plot["Acc_Num"], width, label="Accuracy (%)", color="#1d4ed8")
    plt.bar(x + width/2, df_plot["F1_Num"], width, label="Macro-F1 (%)", color="#059669")
    plt.xticks(x, df_plot["Model"], rotation=25, ha="right")
    plt.ylabel("Performance Score (%)")
    plt.title("Comparative Performance on Untouched Test Split (CICIoT2023 + Edge-IIoTset + FHIR)")
    plt.ylim(40, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig("docs/figures/fig3_model_comparison.png", dpi=300)
    plt.close()

    # Fig 2: Confusion Matrix Heatmap for CA-HTDNet
    plt.figure(figsize=(7, 6))
    cm = np.array(m_ca["Confusion_Matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign / Normal", "Cyber Threat"], yticklabels=["Benign / Normal", "Cyber Threat"])
    plt.title("CA-HTDNet Test Set Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig("docs/figures/fig4_confusion_matrix.png", dpi=300)
    plt.close()

    # Fig 3: Cryptographic Benchmark Latency Comparison
    if os.path.exists("results/crypto_benchmarks.csv"):
        df_c = pd.read_csv("results/crypto_benchmarks.csv")
        plt.figure(figsize=(10, 5))
        plt.barh(df_c["Algorithm"], df_c["Primary_Op_ms"], color="#8b5cf6")
        plt.xlabel("Primary Cryptographic Operation Latency (ms)")
        plt.title("Post-Quantum vs Classical Cryptography Latency on Apple Silicon M2")
        plt.tight_layout()
        plt.savefig("docs/figures/fig10_pqc_latency_comparison.png", dpi=300)
        plt.close()

    # Fig 4: Ablation Step Improvement
    if os.path.exists("results/ablation_results.csv"):
        df_ab = pd.read_csv("results/ablation_results.csv")
        plt.figure(figsize=(10, 5))
        plt.plot(df_ab["Variant"], df_ab["Macro_F1"] * 100, marker="o", color="#ec4899", linewidth=2.5)
        plt.ylabel("Validation Macro-F1 (%)")
        plt.title("Ablation Trajectory: Incremental Impact of Architectural Components (A0 to A8)")
        plt.xticks(df_ab["Variant"], [f"{v}: {c}" for v, c in zip(df_ab["Variant"], df_ab["Component"])], rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig("docs/figures/fig8_ablation_trajectory.png", dpi=300)
        plt.close()

    # Fig 5: System Architecture Diagram (Publication Schematic)
    plt.figure(figsize=(12, 7))
    plt.text(0.5, 0.95, "Crypto-Agile FHIR-Blockchain Architecture (CA-HTDNet)", ha="center", va="center", fontsize=16, fontweight="bold", color="#1e3a8a")
    
    boxes = [
        ("Multimodal Data Inputs\n(CICIoT2023, Edge-IIoTset, Synthetic FHIR)", (0.15, 0.75), "#dbeafe"),
        ("Feature Harmonization & Preprocessing\n(RobustScaler, OrdinalEncoder)", (0.50, 0.75), "#dbeafe"),
        ("Tabular GBDT Experts\n(LightGBM + CatBoost)", (0.85, 0.75), "#dbeafe"),
        
        ("FT-Transformer\nFeature Tokenizer", (0.25, 0.50), "#fef3c7"),
        ("Causal TCN + BiGRU\nMulti-Head Attention", (0.75, 0.50), "#fef3c7"),
        
        ("Context Fusion & Tri-State Head\n(Normal, Suspicious, Attack)", (0.50, 0.30), "#fee2e2"),
        ("Post-Quantum Crypto-Agility\n(ML-KEM, ML-DSA, SLH-DSA, AES-256-GCM)", (0.25, 0.10), "#dcfce7"),
        ("Hyperledger Fabric Consortium Ledger\n(Consent, Access, Audit, Integrity, Threat)", (0.75, 0.10), "#dcfce7")
    ]
    for text, (bx, by), col in boxes:
        plt.text(bx, by, text, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.6", facecolor=col, edgecolor="#64748b", lw=1.5), fontsize=10)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig("docs/figures/full_architecture.png", dpi=300)
    plt.close()

    print("All publication figures successfully created in docs/figures/!")

if __name__ == "__main__":
    main()
