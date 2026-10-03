# Crypto-Agile FHIR-Blockchain Architecture for Post-Quantum Electronic Health Record Exchange with Intelligent Threat Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hardware: Apple M2](https://img.shields.io/badge/Hardware-Apple%20M2%2016GB-black.svg)]()
[![Validation: PASS](https://img.shields.io/badge/Result%20Validation-PASS-brightgreen.svg)]()

A research-grade cybersecurity platform that unifies **deep multimodal tabular threat detection (CA-HTDNet)**, **post-quantum cryptography (NIST FIPS 203/204/205)**, **HL7 FHIR R4 clinical data security**, and **Hyperledger Fabric permissioned blockchain auditing** for zero-trust electronic health record exchange.

---

## 1. Core Architecture

```text
                 STREAMLIT APPLICATION
                         │
          ┌──────────────┴──────────────┐
          │                             │
     RESEARCH VIEW                 LIVE DEMO VIEW
          │                             │
          ▼                             ▼
 Dataset & Results                Simulated Request
          │                             │
          ▼                             ▼
 Model Validation               Frozen Model Inference
          │                             │
          └──────────────┬──────────────┘
                         ▼
                  Threat/Risk Engine
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
           ALLOW       REVIEW       BLOCK
             │           │           │
             └───────────┼───────────┘
                         ▼
                 Crypto-Agility
                         │
                         ▼
                  FHIR Transaction
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Encrypted EHR          Blockchain Audit
       Off-chain Store        / Integrity Ledger
```

---

## 2. Key Empirical Findings (Held-Out Test Set: 26,251 samples)

| Metric | Target | Measured Result | Status |
|---|---|---|---|
| **Accuracy** | $\ge 98.00\%$ | **99.06%** | `VERIFIED` |
| **Macro Recall** | $\ge 98.00\%$ | **98.98%** | `VERIFIED` |
| **False Negative Rate (FNR)** | $\le 2.000\%$ | **0.923%** | `VERIFIED` |
| **False Positive Rate (FPR)** | $\le 2.000\%$ | **1.122%** | `VERIFIED` |
| **Macro-F1** | $\ge 98.00\%$ | **96.63%** | `TARGET NOT REACHED (-1.37%)` |
| **Macro Precision** | $\ge 98.00\%$ | **94.53%** | `TARGET NOT REACHED (-3.47%)` |
| **ROC-AUC** | $\ge 0.9900$ | **0.9990** | `VERIFIED` |
| **PR-AUC** | $\ge 0.9900$ | **0.9999** | `VERIFIED` |
| **Confusion Matrix** | — | `[[1850, 21], [225, 24155]]` | `VERIFIED` |

> [!NOTE]
> All metrics are empirically verified on the locked test partition. Macro-F1 and Precision gaps are transparently documented in `FAILURE_ANALYSIS.md` as stemming from extreme natural telemetry class imbalance (96% attack vs 4% benign).

---

## 3. Quick Start & Offline Dashboard Launch

The demonstration dashboard operates **100% offline with zero online training**.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Validate Results & Subsystem Consistency
```bash
python scripts/validate_dashboard_results.py
```

### Step 3: Launch Streamlit SOC Dashboard
```bash
streamlit run app/streamlit_app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 4. Verification & Testing Suite

Execute the complete automated validation pipeline:

```bash
# 1. Validate result consistency across models, datasets, and curves
python scripts/validate_dashboard_results.py

# 2. Test frozen inference pipeline (zero training)
python scripts/test_inference_pipeline.py

# 3. Test 11-stage FHIR security transaction pipeline
python scripts/test_fhir_security.py

# 4. Test NIST Post-Quantum Cryptography & Agility Layer
python scripts/test_crypto_layer.py

# 5. Test Hyperledger Fabric Consortium Ledger & Tamper Detection
python scripts/test_blockchain_layer.py

# 6. Run comprehensive Pytest unit test suite
pytest tests/test_dashboard_integrity.py -v
```

---

## 5. Dashboard Features & Walkthrough

The Streamlit dashboard includes 14 comprehensive research modules:

1. **🏠 System Overview:** Status badges, interactive architecture schematic, verified KPI cards, Target vs Actual table.
2. **📊 Dataset Intelligence:** Interactive profiles for CICIoT2023 (120k), Edge-IIoTset (40k), Synthetic FHIR (15k), and leakage safety verification.
3. **🤖 Model Validation & Comparison:** Comprehensive 10-model benchmark table (XGBoost, LightGBM, CatBoost, Random Forest, Decision Tree, Extra Trees, MLP, SVM, Logistic Regression, CA-HTDNet).
4. **📈 Performance Analysis:** Interactive confusion matrix (raw/normalized toggle), ROC curves, and PR curves.
5. **🎯 Error & Threshold Analysis:** Interactive What-If Threshold Lab (dynamic metric recalculation leaving official results untouched) and False Positive/Negative Error Explorer.
6. **🧪 Cross-Dataset Validation:** Domain transfer matrix (CICIoT $\leftrightarrow$ Edge-IIoTset) and generalization gap analysis.
7. **🏥 FHIR Security Demonstration:** Interactive clinical access scenario with real-time 11-stage pipeline tracking.
8. **🛡️ Live Threat Detection:** 7 prebuilt attack injections (DDoS, Bulk Exfiltration, Credential Compromise, API Enumeration, IoMT Anomaly, Privilege Escalation) evaluated through frozen model inference with feature attribution.
9. **🔐 Crypto-Agility:** NIST FIPS 203 (ML-KEM-768), FIPS 204 (ML-DSA-65), FIPS 205 (SLH-DSA), and AES-256-GCM authenticated encryption with live measured latencies.
10. **⛓️ Blockchain Audit:** Simulated Fabric consortium ledger, live block explorer, SHA3-256 integrity verification, and disk tamper test.
11. **🚨 Threat Response:** Real-time event stream and automated mitigation policies.
12. **🔬 Research Experiments & Ablation:** Architectural ablation trajectory A0 to A8, robustness tests, and calibration analysis.
13. **📋 Reproducibility & Provenance:** Full automated reconciliation table (Stored vs Recomputed), artifact SHA-256 digests, and exportable CSV/JSON reports.
14. **ℹ️ Architecture & Viva Mode:** Slide-by-slide presentation walkthrough designed for conference, defense, and thesis evaluation.

---

## 6. Project Structure

```text
Blockchain-EHR/
├── app/
│   └── streamlit_app.py               # 14-page research-grade Streamlit application
├── dashboard/
│   ├── inference/
│   │   └── model_loader.py            # Frozen offline inference loader with SHA-256 checks
│   └── validation/
│       └── result_consistency.py      # Stored vs recomputed metric consistency engine
├── data/
│   ├── metadata/                      # Split manifests and schema definitions
│   ├── processed/                     # Processed arrays and off-chain vault
│   └── splits/                        # 70/15/15 train, val, test parquet splits
├── docs/
│   ├── dashboard_architecture.md      # Dashboard system flow and hierarchy
│   ├── dashboard_validation.md        # Mathematical validation protocol
│   ├── demo_guide.md                  # Viva / presentation demonstration guide
│   ├── figures/                       # Publication-quality architectural figures
│   └── result_provenance.md           # Artifact hashes and hardware execution metadata
├── models/
│   ├── baselines/                     # Trained baseline models (.pkl)
│   ├── preprocessors/                 # Unified security preprocessor (.pkl)
│   └── proposed/                      # CA-HTDNet state dict (.pt), config, threshold.json
├── results/
│   ├── curves/                        # Precomputed ROC and PR curves (.json)
│   ├── predictions/                   # Frozen test predictions (.npz)
│   ├── experiment_registry.json       # Single canonical source of truth for all results
│   ├── final_results.csv              # Verified benchmark results table
│   └── final_results.json             # Full model evaluation metrics
├── scripts/
│   ├── build_experiment_registry.py   # Registry builder and prediction precomputer
│   ├── test_blockchain_layer.py       # Blockchain ledger & tamper test
│   ├── test_crypto_layer.py           # PQC benchmark and agility test
│   ├── test_fhir_security.py          # 11-stage FHIR pipeline test
│   ├── test_inference_pipeline.py     # Offline frozen inference test
│   └── validate_dashboard_results.py  # Master result reconciliation validator
└── tests/
    └── test_dashboard_integrity.py    # Pytest test suite (100% passing)
```

---

## 7. License & Citation
Licensed under the [MIT License](LICENSE).
