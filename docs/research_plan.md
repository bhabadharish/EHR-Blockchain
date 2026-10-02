# Research Plan: Crypto-Agile FHIR-Blockchain Architecture for Post-Quantum EHR Exchange with Intelligent Threat Detection

## 1. Executive Summary and Problem Statement
Electronic Health Record (EHR) exchange across heterogeneous healthcare organizations requires interoperability, confidentiality, integrity, fine-grained access control, and robust cyber-threat resilience. While HL7 FHIR (Fast Healthcare Interoperability Resources) R4 provides standard syntax and semantics, existing healthcare exchange architectures face two impending crises:
1. **The Quantum Threat**: Shor's algorithm will render classical public-key cryptography (RSA, ECDH, ECDSA) obsolete, exposing longitudinal patient records to "Harvest Now, Decrypt Later" (HNDL) attacks.
2. **Advanced Cyber Threats**: Healthcare APIs and IoT/telemetry interfaces are targeted by sophisticated network intrusions, unauthorized access escalations, and data tampering attempts that evade signature-based defenses.

This project designs, implements, and experimentally validates a **Crypto-Agile FHIR-Blockchain Architecture** integrating:
- HL7 FHIR R4 data models and validation
- Post-Quantum Cryptography (ML-KEM-768/1024, ML-DSA-65) with AES-256-GCM and SHA-3-256
- Permissioned blockchain state machine for consent, authorization, and immutable audit logs
- Encrypted off-chain object storage with integrity verification
- Temporal Convolutional Network (TCN) + Transformer with Attention for intelligent threat detection on network/security telemetry
- SHAP-based model interpretability
- Reproducible, empirical validation across large-scale synthetic FHIR datasets (>100,000 records) and benchmark telemetry datasets (Edge-IIoTset).

---

## 2. Research Questions (RQs)
- **RQ1 (PQC Overhead & Feasibility)**: What is the computational, storage, and transmission overhead of replacing classical public-key cryptography with NIST FIPS 203 (ML-KEM) and FIPS 204 (ML-DSA) in HL7 FHIR record encryption and digital signatures under varying clinical payload sizes?
- **RQ2 (Crypto-Agility & Hybrid Migration)**: How does a hybrid cryptographic scheme (ECDH + ML-KEM) balance quantum resistance against legacy interoperability and latency overhead compared to pure classical and pure post-quantum modes?
- **RQ3 (Off-Chain Blockchain Scalability & Consent Enforcement)**: How does an off-chain storage pattern coupled with a permissioned blockchain access-control and consent smart contract scale across transaction volumes from 10,000 to 100,000+ operations, in terms of latency, throughput, and state growth?
- **RQ4 (Deep Threat Detection Efficacy)**: To what extent does a hybrid TCN-Transformer architecture with multi-head attention improve detection accuracy, macro-F1, false positive rate (FPR), and false negative rate (FNR) on cyber telemetry compared to conventional baselines (Logistic Regression, Random Forest, LightGBM, MLP, 1D-CNN, BiLSTM)?
- **RQ5 (Data Leakage & Generalizability)**: Under strict non-leaking data splitting (fit-on-train only) and multi-seed statistical validation (>=5 seeds), are the performance gains of the proposed neural architecture statistically significant?
- **RQ6 (Interpretability & Actionability)**: How do SHAP explanations illuminate key network/telemetry features driving intrusion detection decisions, and how can they inform automated mitigation policies without introducing human error?

---

## 3. Hypotheses (H)
- **H1**: PQC encryption (ML-KEM-768 + AES-256-GCM) introduces an increase in end-to-end exchange latency of less than 35% compared to classical ECDH + AES-256-GCM, maintaining sub-second latency for standard FHIR resource bundles (<100 KB).
- **H2**: Digital signatures with ML-DSA-65 achieve 100% detection of unauthorized ciphertext, metadata, hash, and FHIR resource tampering with sub-5 millisecond verification times on modern computing hardware.
- **H3**: The off-chain architecture bounds blockchain ledger growth to cryptographic references (<1 KB per transaction), allowing sustained throughput without ledger bloat.
- **H4**: Combining Temporal Convolutional Networks (capturing local receptive temporal features) with Transformer encoders (capturing long-range contextual dependencies) yields a statistically significant improvement in Macro-F1 (p < 0.05) over individual TCN and Transformer ablations and classical ML baselines.
- **H5**: Data minimization policies (FULL, MINIMAL, RESEARCH, EMERGENCY) reduce transmission payload sizes by at least 25% for research and emergency contexts without degrading clinical utility or cryptographic integrity.

---

## 4. Experimental Variables
### Independent Variables:
1. **Cryptographic Suite**: Classical (ECDH P-256 + AES-256-GCM + ECDSA), Hybrid (ECDH P-256 + ML-KEM-768 + AES-256-GCM + ML-DSA-65), PQC-Standard (ML-KEM-768 + AES-256-GCM + ML-DSA-65), PQC-High (ML-KEM-1024 + AES-256-GCM + ML-DSA-87).
2. **Data Minimization Policy**: FULL, MINIMAL, RESEARCH, EMERGENCY.
3. **Dataset Scale**: 10,000; 25,000; 50,000; 100,000 records.
4. **Threat Detection Model**: Logistic Regression, Random Forest, LightGBM, MLP, 1D-CNN, BiLSTM, TCN-only, Transformer-only, Proposed (TCN + Transformer + Attention).
5. **Class Imbalance Mitigation**: Class weighting, Focal loss, SMOTE, Random undersampling.
6. **Attack Scenario**: Unauthorized access, revoked access, replay attack, ciphertext tampering, metadata tampering, signature tampering, hash tampering, volumetric network attack, injection attack.

### Dependent Variables:
1. **Cryptographic Metrics**: Key generation latency (ms), encapsulation latency (ms), decapsulation latency (ms), encryption latency (ms), decryption latency (ms), signature generation latency (ms), verification latency (ms), public key size (bytes), ciphertext size (bytes), signature size (bytes).
2. **Blockchain Metrics**: Transaction latency (ms), transaction throughput (TPS), block verification latency, ledger storage overhead (bytes/tx).
3. **Machine Learning Metrics**: Accuracy, Precision, Recall, Macro-F1, Weighted-F1, Balanced Accuracy, ROC-AUC, PR-AUC, False Positive Rate (FPR), False Negative Rate (FNR), inference latency per sample (ms), parameter count, FLOPs/MACs.
4. **End-to-End Metrics**: Total exchange latency (ms), payload transfer size (bytes), tamper detection accuracy (%), authorization decision latency (ms).

---

## 5. Work Breakdown Structure (Phases 0 - 40)
- **Phase 0**: Project Governance & Research Protocols (Current)
- **Phases 1-4**: Dataset Acquisition, Provenance Manifest, Synthetic FHIR Generation (>100,000 records), and Statistical Validation
- **Phases 5-6**: FHIR R4 Gateway, Pydantic Schema Validation, Data Minimization Layer
- **Phases 7-9**: Modular Crypto-Agility Engine (PQC ML-KEM/ML-DSA), Hybrid Comparison, SHA-3-256 Integrity & Tamper Detection
- **Phases 10-12**: Off-chain Storage Engine, Hyperledger Fabric State Machine/Chaincode, RBAC + ABAC Access Control
- **Phases 13-17**: Security Threat Modeling, Cybersecurity Telemetry Ingestion (Edge-IIoTset), Baselines & Proposed TCN-Transformer Architecture, Anti-Leakage Protocol, Class Imbalance Handling
- **Phases 18-20**: Deterministic Multi-Seed Model Training, Comprehensive Metrics Evaluation, Statistical Significance Testing
- **Phases 21-23**: Architectural, Cryptographic, and Blockchain Ablation Studies
- **Phases 24-26**: Scalability Benchmarking (up to 100,000+ records), End-to-End Clinical Exchange, Controlled Attack Simulation
- **Phases 27-29**: Streamlit Research Dashboard, Structured Experiment Database, Automated Reproduction Scripts
- **Phases 30-35**: Publication-Ready Figures, Baseline Literature Matrix, Error Analysis, Security Validation, Performance Profiling
- **Phases 36-40**: Packaging & Reproducibility, Test Suite Execution, Quality Gates Verification, Consistency Auditing, Final Research Reports

---

## 6. Research Governance & Quality Assurance
All experimental outputs must be derived strictly from live program execution with verified seeds. No synthetic metrics, cherry-picked splits, or untracked state will be permitted. Every phase must pass its explicit Quality Gate before subsequent phases execute.
