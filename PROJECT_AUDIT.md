# Comprehensive Project Audit & Discovery Report
## Crypto-Agile FHIR-Blockchain Architecture for Post-Quantum EHR Exchange with Intelligent Threat Detection

**Date:** 2026-10-03  
**Auditor:** Antigravity AI Cybersecurity & Tabular Deep Learning Specialist  
**System Target:** Apple MacBook Air M2 (16 GB Unified Memory), macOS Darwin arm64

---

### 1. Executive Summary & Inventory

The repository was systematically audited prior to restructuring. All prior obsolete code and experiment runs have been cleanly flushed, leaving the foundational raw and partitioned datasets strictly preserved.

#### Existing Datasets Identified

| Dataset | File Location | Size | Rows | Columns | Purpose |
|---|---|---|---|---|---|
| **CICIoT2023 (Train)** | `CICIOT23/train/train.csv` | 1.5 GB | 5,491,971 | 47 | Primary network intrusion detection (Train partition) |
| **CICIoT2023 (Val)** | `CICIOT23/validation/validation.csv` | 332 MB | 1,176,851 | 47 | In-domain validation partition |
| **CICIoT2023 (Test)** | `CICIOT23/test/test.csv` | 332 MB | 1,176,851 | 47 | Untouched locked test partition |
| **Edge-IIoTset** | `data/raw/ML-EdgeIIoT-dataset.csv` | 78 MB | 157,800 | 63 | IoT telemetry & cross-dataset generalization benchmark |
| **Synthea FHIR R4** | `data/raw/synthea_sample_data_fhir_r4_nov2021.zip` | 90 MB | Multi-resource | Canonical FHIR | Real-world synthetic healthcare resource baseline |

---

### 2. Architecture & Design Requirements

The required research system brings together six interdependent subsystems into a unified platform:

1. **Heterogeneous Threat Detection Subsystem (CA-HTDNet)**:
   - Feature Preprocessing & Harmonization (Network + FHIR + Identity/Device)
   - Tabular Deep Learning: LightGBM, CatBoost, FT-Transformer
   - Temporal Sequence Encoding: Causal TCN + Bidirectional GRU + Multi-Head Self-Attention
   - Stacking / Learned Multi-Model Fusion
   - Cost-sensitive Focal Loss & Temperature/Platt Probability Calibration
   - Strict validation threshold optimization ($Macro-F1, FPR, FNR$)

2. **Healthcare & FHIR Telemetry Engine**:
   - FHIR R4 standard resources (`Patient`, `Observation`, `Encounter`, `MedicationRequest`, `DiagnosticReport`, `CarePlan`, `Device`, `Consent`, `Provenance`, `AuditEvent`)
   - Schema validation, Canonical JSON normalization, SHA3-256 integrity hashing
   - Security labeling, SMART-on-FHIR RBAC/ABAC policy engine

3. **Post-Quantum Cryptography & Crypto-Agility Engine**:
   - NIST PQC Algorithms: ML-KEM (FIPS 203), ML-DSA (FIPS 204), SLH-DSA (FIPS 205)
   - Classical & Hybrid Modes: ECDH + ML-KEM, ECDSA + ML-DSA, AES-256-GCM
   - Dynamic Risk-Driven Policy Engine: Threat score triggers automated cryptographic agility transitions

4. **Permissioned Consortium Blockchain (Hyperledger Fabric)**:
   - 4 Simulated Consortium Organizations: Hospital A, Hospital B, Hospital C, Research Org
   - Five Chaincodes:
     - `ConsentChaincode`: Patient granular consent lifecycle (create, read, update, revoke, verify)
     - `AccessChaincode`: Role/attribute verification and access tokens
     - `AuditChaincode`: Tamper-evident query and modification logging
     - `IntegrityChaincode`: On-chain SHA3-256 verification against off-chain encrypted EHR storage
     - `ThreatChaincode`: Incident recording, risk state escalation, device quarantine

5. **Threat-Aware Zero-Trust Risk & Response Engine**:
   - Contextual Risk Score = $f(\text{Threat Probability}, \text{Resource Sensitivity}, \text{Actor Role}, \text{Device Trust}, \text{Historical Risk})$
   - Autonomous Policy Actions: `ALLOW`, `LOG`, `STEP_UP_AUTH`, `REQUIRE_CONSENT`, `RATE_LIMIT`, `QUARANTINE_DEVICE`, `BLOCK`

6. **Interactive Research Dashboard & Automation**:
   - 10-page Streamlit application with live inference, zero fake data, and end-to-end audit trails
   - Comprehensive reproducible execution pipeline (`scripts/01_validate_data.py` to `scripts/14_generate_results.py`)

---

### 3. Data Leakage & Security Audit

| Risk Category | Potential Vulnerability | Mitigation in New Pipeline |
|---|---|---|
| **Preprocessing Leakage** | Scaling or feature selection on full dataset | Scalers, imputers, and encoders fitted strictly on `TRAIN` only. |
| **Temporal Leakage** | Random shuffling across time-series sequences | Chronological sequence windowing without lookahead. |
| **Identifier Leakage** | Source/Destination IP or timestamps memorized | IP hashing, port entropy, flow duration, and protocol feature abstraction. |
| **Threshold Leakage** | Optimizing decision boundary on test data | Multi-objective threshold search evaluated strictly on validation set. |
| **Synthetic Cross-Contamination** | Synthetic samples entering test partitions | Synthetic FHIR security data quarantined to training / synthetic testing only. |

---

### 4. Migration & Implementation Roadmap

1. **Project Re-structure**: Build directory structure matching Section 3 of prompt.
2. **Data Pipeline & Synthetic Generator**:
   - Validate CICIoT2023 and Edge-IIoTset schemas.
   - Implement `src/data/synthetic_fhir_generator.py` for realistic FHIR security events.
   - Build unified feature harmonization layer.
3. **Machine Learning & Deep Learning**:
   - Train baselines (Logistic Regression, Random Forest, Extra Trees, SVM, Decision Tree, XGBoost, LightGBM, CatBoost, MLP).
   - Implement proposed **CA-HTDNet** (LightGBM + CatBoost + FT-Transformer + TCN + BiGRU + Attention).
   - Perform calibration (ECE/Brier) and validation threshold optimization.
   - Run cross-dataset generalization (CICIoT2023 $\leftrightarrow$ Edge-IIoTset) and ablation studies (A0–A8).
4. **Crypto & Blockchain**:
   - Implement post-quantum cryptographic primitives (ML-KEM, ML-DSA, SLH-DSA, AES-256-GCM, SHA3-256).
   - Build crypto-agility dynamic engine.
   - Implement Hyperledger Fabric client simulation with 5 production chaincodes.
   - Build off-chain encrypted EHR storage vault with on-chain hash verification.
5. **Security & FHIR Services**:
   - FHIR R4 resource validation, SMART-on-FHIR authorization, RBAC/ABAC, Consent engine.
   - Threat-aware response and automated mitigation engine.
6. **Dashboard, Tests & Documentation**:
   - 10-page Streamlit dashboard (`app/streamlit_app.py`).
   - Unit and integration tests (`tests/`).
   - Full automated scripts (`scripts/01_validate_data.py` to `scripts/14_generate_results.py`).
   - `FINAL_EXPERIMENT_REPORT.md` and research publication tables.
