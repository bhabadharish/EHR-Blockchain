# Current Project Audit & Architectural Gap Analysis

**Project Title:** Crypto-Agile FHIR-Blockchain Architecture for Post-Quantum Electronic Health Record Exchange with Intelligent Threat Detection  
**Audit Date:** October 2026  
**Auditor:** CA-HTDNet Research Team  
**Scope:** Full repository audit, dataset inspection, legacy artifact mapping, and architectural gap analysis prior to rebuilding CA-HTDNet V2 (`CAHTDNET_V2_001`).

---

## 1. Executive Summary

This audit establishes a rigorous inventory and dependency mapping of all existing experimental assets across data, preprocessing, models, predictions, benchmarks, and application tiers. In the previous baseline evaluation (`CAHTDNET_FINAL_V001`), the model achieved 98.74% Accuracy, 95.56% Macro-F1 (96.53% at validation-optimal $\tau^*=0.35$), 0.802% FPR, and 1.296% FNR. While mathematically reconciled and leakage-free, tree-based ensembles (XGBoost at 97.04% Macro-F1 and Random Forest at 96.70% Macro-F1) outperformed the neural network baseline.

Critical root causes identified include:
1. **Severe Imbalance in Real Threat Telemetry:** CICIoT2023 train split contained only 2.26% benign traffic (2,720 normal vs 117,280 attack). Edge-IIoTset raw data was previously head-sliced (`nrows=40000`), inadvertently ingesting zero normal records from that domain because all 24,301 normal records reside in the lower half of `ML-EdgeIIoT-dataset.csv`.
2. **Synthetic FHIR Separability Artifacts:** In the legacy synthetic generator, `historical_risk` was bounded to $[0.01, 0.15]$ for normal events and $\ge 0.35$ for threat events, allowing trivial separability and giving the model an unrealistic shortcut during training.
3. **Architectural Gaps in V1:** The V1 architecture relied on a standard feature projection with parallel TCN and Self-Attention branches. Because attack false negatives (316 out of 24,380) leaked into the small normal pool (1,871 test samples), Normal precision dropped to 85.45%, severely depressing Macro-Precision (92.69%) and Macro-F1 (95.56%).
4. **Component Interference:** Ablation demonstrated that individual modules (such as Gating alone or Attention alone) occasionally yielded higher validation scores than the naive monolithic concatenation in V1, demonstrating the need for component selection and gated residual fusion in CA-HTDNet V2.

---

## 2. Inventory of Current Assets

### 2.1 Datasets
- `data/raw/cic_iot/` & `data/raw/ciciot2023/`: Large-scale IoT/network threat traffic (DDoS, DoS, Recon, Web, BruteForce, Mirai). 120k slice has 2,720 Benign and 117,280 Attack packets.
- `data/raw/edge_iiot/ML-EdgeIIoT-dataset.csv`: 157,800 rows total (24,301 Normal, 133,499 Attack across 14 attack types: DDoS UDP/ICMP/HTTP/TCP, Ransomware, SQL Injection, Uploading, Backdoor, Vulnerability Scanner, Port Scanning, XSS, Password, MITM, Fingerprinting).
- `data/raw/synthetic_fhir/`: FHIR R4 security access events. Previously generated with artificial feature boundaries. Must be redesigned for realistic distribution overlap.
- `data/splits/`: Parquet partitions for train (122,499), validation (26,250), and locked test (26,251).

### 2.2 Preprocessors & Feature Engineering
- `src/data/cleaning.py`: `UnifiedSecurityPreprocessor` implementing `RobustScaler` on numerical features and integer categorical encoding with out-of-vocabulary handling.
- `models/preprocessors/preprocessor.pkl`: Preprocessor fitted strictly on Train split.
- `models/preprocessors/feature_schema.json` & `feature_order.json`: Standardized 21 numeric features and 6 categorical features.

### 2.3 Models & Checkpoints
- `experiments/models/CAHTDNET_FINAL_V001.pt`: V1 PyTorch weights (144,356 parameters, 605 KB).
- `models/baselines/`: Trained scikit-learn, XGBoost, LightGBM, CatBoost, MLP, and FT-Transformer models.

### 2.4 Canonical Experiment Results & Metrics
- `results/result_engine.py`: Single canonical metric engine enforcing strict mathematical identities.
- `results/CAHTDNET_FINAL_V001_metrics.json`: Final locked metrics for V1.
- `results/baseline_comparison.csv`: Fair baseline comparison across 9 algorithms.
- `results/ablation_results.csv`: Ablation variants A0 through A8.
- `results/threshold_sweep.csv`: Validation-only threshold optimization ($\tau^* = 0.35$).
- `results/calibration_comparison.csv`: Platt & Temperature scaling calibration reports.
- `results/robustness_report.csv`: Perturbation benchmarks.
- `results/cross_dataset_generalization.csv`: Stratified domain shift evaluations.
- `results/FINAL_RESULTS_LOCK.json`: Cryptographic SHA-256 hash manifest.

### 2.5 Software & Application Tiers
- `app/streamlit_app.py`: Streamlit demonstration dashboard.
- `dashboard/inference/model_loader.py`: Cached loader validating hashes against `FINAL_RESULTS_LOCK.json`.
- `blockchain/`: Local Hyperledger Fabric simulation verifying cryptographic audit trail.
- `src/crypto/`: Kyber/ML-KEM, Dilithium/ML-DSA, SPHINCS+/SLH-DSA, AES-256-GCM, SHA3-256 implementations.

---

## 3. Dependency Graph

```text
Raw Datasets (CICIoT2023, Edge-IIoTset, Synthetic FHIR)
       │
       ▼
Data Harmonization & Stratified Sampling (Zero-Leakage)
       │
       ▼
Deterministic Splits (70% Train / 15% Val / 15% Locked Test)
       │
       ▼
Feature Pipeline (Fitted strictly on Train)
       │
       ├─────────────────────────┬─────────────────────────┐
       ▼                         ▼                         ▼
Baseline Training        CA-HTDNet Training       Cross-Dataset Sets
(XGBoost, RF, etc.)      (Train + Val selection)  (Domain shift eval)
       │                         │                         │
       └─────────────────────────┼─────────────────────────┘
                                 ▼
                    Raw Predictions Generation
                    (Locked Test Parquet / CSV)
                                 │
                                 ▼
                    Canonical Result Engine
                    (result_engine.py / metric_engine.py)
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
Final Results CSV         Ablation Tables       Paper Results & Plots
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    Verification & Hash Locking
                    (FINAL_RESULTS_LOCK.json)
                                 │
                                 ▼
                    Streamlit Dashboard & Inference
```

---

## 4. Archive Actions Performed

The following obsolete scripts and intermediate unversioned markdown reports have been safely moved to `archive/` to ensure no conflict with the canonical V1 and V2 experiment pipelines:
- `scripts/01_validate_data.py` → `archive/legacy_scripts/`
- `scripts/02_prepare_data.py` → `archive/legacy_scripts/`
- `scripts/03_generate_synthetic_fhir.py` → `archive/legacy_scripts/`
- `scripts/04_train_baselines.py` → `archive/legacy_scripts/`
- `scripts/06_train_ca_htdnet.py` → `archive/legacy_scripts/`
- `scripts/07_calibrate.py` → `archive/legacy_scripts/`
- `scripts/08_optimize_threshold.py` → `archive/legacy_scripts/`
- `scripts/09_cross_dataset.py` → `archive/legacy_scripts/`
- `scripts/10_ablation.py` → `archive/legacy_scripts/`
- `scripts/11_robustness.py` → `archive/legacy_scripts/`
- `scripts/12_crypto_benchmark.py` → `archive/legacy_scripts/`
- `scripts/13_blockchain_benchmark.py` → `archive/legacy_scripts/`
- `scripts/14_generate_results.py` → `archive/legacy_scripts/`
- `FAILURE_ANALYSIS.md` → `archive/legacy_reports/`
- `FINAL_EXPERIMENT_REPORT.md` → `archive/legacy_reports/`
- `PROJECT_AUDIT.md` → `archive/legacy_reports/`

---

## 5. Architectural Strategy for CA-HTDNet V2

To legitimately surpass 98% Macro-F1 without violating leakage or test-locking constraints, CA-HTDNet V2 must resolve the specific failure modes identified:
1. **Stratified Sampling Across All Raw Datasets:** Sample 20,000 Normal from Edge-IIoTset, 10,000 Normal from CICIoT, and 10,000 Normal from realistic FHIR, ensuring the model learns diverse, realistic benign boundaries.
2. **Realistic FHIR Generator (Zero Deterministic Leaks):** Overhaul the synthetic generator to eliminate arbitrary threshold cliffs (e.g. `historical_risk` separation) and simulate nuanced attack vectors with realistic benign overlap.
3. **Dual-Branch Architecture with Feature Gating & Cross-Feature Attention:**
   - Explicit numerical projection with RobustScaler + categorical entity embeddings.
   - Learnable Feature Gating layer (Sigmoid-weighted importance gating).
   - Cross-Feature Self-Attention block to model feature correlations.
   - Parallel Local Branch (Conv1D/TCN + Residual MLP) and Global Branch (Transformer Encoder).
   - Gated Fusion combining local and global representations.
4. **Class-Balanced Focal & Margin Loss:** Implement an asymmetric class-balanced loss with label smoothing and cosine margin penalty to penalize attack false negatives into the normal class, directly boosting Normal class precision.
5. **Rigorous Validation Protocol:** Optuna hyperparameter optimization on Train+Val, 5-seed stability validation (seeds 42, 123, 2024, 3407, 777), 5-fold cross-validation, and locked test evaluation performed strictly once at the end.
