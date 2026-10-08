# Repository Audit Report
**Project:** Adaptive Hierarchical Boosting Intrusion Detection System (HAB-IDS)  
**Date:** 2026-10-03  
**Status:** Pre-Rearchitecture Audit Complete  
**Backup Location:** `backup/pre_rearchitecture/`

---

## 1. Executive Summary

This repository was originally created for an experimental intrusion detection and cryptographic EHR prototype (`CA-HTDNet` / `CA-HTDNet V2`), combining a hybrid deep learning model (BiGRU, TCN, attention) with post-quantum cryptography and blockchain auditing. The repository has been preserved in its entirety in `backup/pre_rearchitecture/`.

The research architecture is transitioning to **HAB-IDS (Adaptive Hierarchical Boosting Intrusion Detection System)**:
- Core Base Learners: **XGBoost + LightGBM + CatBoost**
- Adaptive disagreement-aware meta-learner routing (low vs. high disagreement)
- Stage-1 binary detection (Normal vs. Attack) + Stage-2 multiclass attack-family classification
- Probability calibration (Platt scaling & Isotonic regression)
- Rigorous security threshold optimization (FPR ≤ 1%, FNR ≤ 1%, Macro-F1 maximization)
- FHIR contextual risk engine
- Post-quantum security layer (ML-KEM, ML-DSA, AES-256-GCM)
- Permissioned blockchain audit ledger integration
- Absolute test set isolation, leakage-free feature pipelines, and reproducible verification.

---

## 2. Inventory of Existing Assets

### 2.1 Existing Directories
- `data/`: Raw, interim, processed, splits, and metadata directories.
- `models/`: Checkpoints of legacy neural models (`CAHTDNET_FINAL_V001.pt`, `ca_htdnet_v2.pt`) and tabular baselines (`models/baselines/`).
- `scripts/`: Legacy training, evaluation, benchmark, and reconciliation scripts.
- `src/`: Legacy source modules for blockchain, crypto, data, evaluation, inference, and models.
- `experiments/`: Legacy configuration runs, splits, thresholds, and curves.
- `results/`: Legacy JSON and CSV benchmark records.
- `predictions/`: Legacy prediction files from CAHTDNet runs.
- `paper_results/`: Legacy generated tables for prior manuscript drafts.
- `figures/`: Legacy PNG figures.
- `reports/`: Prior experiment and leakage reports.
- `tests/`: Existing unit test suites for platform and dashboard integrity.
- `backup/pre_rearchitecture/`: Complete, byte-for-byte snapshot of the pre-rearchitecture repository.

### 2.2 Existing Datasets
1. **CICIoT2023**:
   - `data/raw/ciciot2023/train.csv` (1,623,446,296 bytes, 5,491,971 rows, 47 columns)
   - `data/raw/ciciot2023/validation.csv` (347,921,865 bytes, 1,176,851 rows, 47 columns)
   - `data/raw/ciciot2023/test.csv` (347,958,926 bytes, 1,176,851 rows, 47 columns)
   - Total rows: 7,845,673 across 34 classes (33 attack types + `BenignTraffic`).
2. **Edge-IIoTset**:
   - `data/raw/ML-EdgeIIoT-dataset.csv` (82,184,390 bytes, 157,800 rows, 63 columns)
   - Binary label: `Attack_label` (133,499 attack, 24,301 normal)
   - Multiclass: `Attack_type` (14 attack types + `Normal`).
3. **Synthetic FHIR Security Events**:
   - `data/raw/synthetic_fhir/fhir_security_large.csv` (13,334,291 bytes, 50,000 rows, 26 columns)
   - `data/raw/synthetic_fhir/fhir_security_med.csv` (10,000 rows)
   - `data/raw/synthetic_fhir/fhir_security_dev.csv` (1,000 rows)
   - Parquet versions also available.
   - Binary label: `binary_label` (32,500 normal, 17,500 attack)
   - Multiclass: `attack_category` (14 attack categories + `normal_access`).
4. **Synthea Clinical Data (Reference)**:
   - `data/raw/synthea_sample_data_fhir_r4_nov2021.zip` (94.9 MB) and unzipped JSONs in `data/raw/synthea_reference_fhir_r4/fhir/`.

### 2.3 Duplicate Datasets & Broken Links
- **Duplicate / Symlink in Edge-IIoT:** `data/raw/edge_iiot/ML-EdgeIIoT-dataset.csv` is a symlink pointing to `../ML-EdgeIIoT-dataset.csv`.
- **Broken Symlinks in CICIoT:** `data/raw/cic_iot/` contains broken symlinks `train.csv`, `validation.csv`, and `test.csv` targeting `../../../CICIOT23/...`. The actual datasets exist in `data/raw/ciciot2023/`.
- **Resolution:** Canonical path references must target `data/raw/ciciot2023/` and `data/raw/edge_iiot/ML-EdgeIIoT-dataset.csv`.

### 2.4 Existing Models & Checkpoints
- `models/CAHTDNET_FINAL_V001.pt` (0.60 MB, PyTorch hybrid model)
- `models/proposed/ca_htdnet_v2.pt` (1.96 MB)
- `models/baselines/Random_Forest.pkl` (6.65 MB)
- `models/baselines/Extra_Trees.pkl` (5.83 MB)
- `models/baselines/LightGBM.pkl` (0.51 MB)
- `models/baselines/XGBoost.pkl` (0.33 MB)
- `models/baselines/CatBoost.pkl` (0.18 MB)
- `models/baselines/MLP.pkl` (0.26 MB)
- `models/baselines/FT-Transformer.pt` (0.30 MB)
- `models/baselines/Decision_Tree.pkl`, `Logistic_Regression.pkl`, `SVM.pkl`
- Preprocessors: `models/preprocessors/preprocessor.pkl`, `feature_order.json`, `feature_schema.json`

### 2.5 Existing Scripts & Obsolete Code
- 41 scripts in `scripts/` primarily geared toward deep-learning attention training (`train_ca_htdnet.py`), baseline training, and generating static report tables.
- These scripts represent the prior CA-HTDNet iteration and are preserved in `backup/pre_rearchitecture/`.
- The new architecture requires modular, research-grade Python scripts specifically implementing the hierarchical boosting architecture, out-of-fold disagreement meta-learning, calibrated thresholds, and verification gates.

### 2.6 Existing Results, Metrics, and Suspicious Files
- `results/FINAL_RESULTS_LOCK.json` and `results/final_results.json`: Contain recorded metrics from CA-HTDNet V2 experiments.
- `predictions/`: Pre-computed prediction files for CAHTDNet models.
- **Finding:** In previous iterations, some scripts read and wrote to fixed JSON metric locks without full programmatic re-computation from predictions during validation audits.
- **Remediation:** Strict Phase 36 and Phase 37 compliance: All metrics will be calculated strictly from saved test predictions and ground truth, verified by `scripts/verify_results.py`.

---

## 3. Rearchitecture Strategy (HAB-IDS)

1. **Clean Project Structure**:
   - Establish dedicated `configs/` (`data.yaml`, `features.yaml`, `models.yaml`, `training.yaml`, `evaluation.yaml`).
   - Modular `src/` codebase: `data/`, `preprocessing/`, `features/`, `models/`, `ensemble/`, `calibration/`, `thresholding/`, `evaluation/`, `security/`, `utils/`.
   - Clear executable scripts in `scripts/`.
   - Standardized artifact storage in `models/` and `results/`.
2. **Leakage-Free Protocol**:
   - Absolute isolation of the test set during feature scaling, feature selection, hyperparameter tuning, meta-model training, calibration, and threshold searching.
   - Elimination of identity and post-event columns.
3. **Execution Plan**:
   - Efficient Parquet conversion and schema alignment.
   - Train/Val/Test canonical splits.
   - Baseline training (LR, RF, ET, XGB, LGBM, CB).
   - Tuned XGBoost, LightGBM, and CatBoost base models with early stopping.
   - Out-of-fold disagreement feature extraction and meta-learner training.
   - Adaptive routing evaluation (low vs. high disagreement).
   - Hierarchical Stage-1 (binary) and Stage-2 (attack family) classification.
   - Post-quantum and blockchain decoupled security benchmarks.
   - Multi-seed stability, statistical testing (McNemar, CI), and ablation studies.
   - Verification gate script (`scripts/verify_results.py`).
