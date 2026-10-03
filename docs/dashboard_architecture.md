# Dashboard Architecture Specification

## 1. Overview
The **Crypto-Agile FHIR-Blockchain Security Platform Dashboard** is a research-grade, offline-inference interactive system designed to demonstrate end-to-end healthcare cybersecurity telemetry, post-quantum cryptography, and blockchain-audited electronic health record (EHR) exchange.

---

## 2. Structural Component Hierarchy

```
                             STREAMLIT INTERACTION LAYER
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             │                                                     │
      RESEARCH AUDIT VIEW                                  LIVE INFERENCE VIEW
             │                                                     │
    Canonical Registry                                     Prebuilt Clinical Scenarios
    (results/experiment_registry.json)                     (Normal, Suspicious, DDoS, FHIR Exfil)
             │                                                     │
    Frozen Test Predictions                                FrozenModelLoader
    (results/predictions/test_predictions.npz)             (dashboard/inference/model_loader.py)
             │                                                     │
    ResultConsistencyEngine                                PyTorch CA-HTDNet (Offline CPU)
    (dashboard/validation/result_consistency.py)           No Online Model Training (.fit() disabled)
             │                                                     │
             └──────────────────────────┬──────────────────────────┘
                                        ▼
                            Contextual Zero-Trust Engine
                        (src/security/response_engine.py)
                                        │
                         ┌──────────────┼──────────────┐
                         ▼              ▼              ▼
                       ALLOW         REVIEW          BLOCK
                         │              │              │
                         └──────────────┼──────────────┘
                                        ▼
                           Post-Quantum Crypto Agility
                          (src/crypto/crypto_agility.py)
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
          ML-KEM-768 (KEM)                              ML-DSA-65 (Sign)
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        ▼
                            AES-256-GCM + SHA3-256
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                 Off-Chain EHR Vault          Hyperledger Fabric Ledger
                 (data/processed/vault)       (src/blockchain/ledger.py)
```

---

## 3. Strict Operating Rules
1. **Offline Inference Only:** The application never executes model training (`.fit()`, `.train()`, `optimizer.step()`), Optuna optimization, or threshold fitting during dashboard runtime.
2. **Deterministic Artifact Integrity:** Calculates SHA-256 digests on startup and verifies them against `dashboard_artifact_audit.json` and `results/experiment_registry.json`.
3. **Single Canonical Registry:** All 10 evaluated models read verified performance metrics directly from `results/experiment_registry.json`.
4. **Result Consistency Enforcement:** `ResultConsistencyEngine` reconciles stored metrics against freshly recomputed values on the 26,251 test partition samples with numerical tolerance $\le 10^{-6}$.
