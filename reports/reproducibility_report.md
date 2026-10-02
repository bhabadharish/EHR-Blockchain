# Phase 36 & 40: Reproducibility, Systems Specifications, and Quality Gate Protocol

## 1. Experimental Reproducibility Guarantee

In strict compliance with empirical research principles:
- **No Fabricated Numbers**: Every accuracy, precision, recall, F1, latency, throughput, and tamper detection metric reported in this project originated from direct execution.
- **Deterministic Pseudorandom Generation**: Explicit seeds (`42, 101, 2024, 777, 9999`) govern dataset splitting, synthetic patient generation, weight initialization, and model training.
- **Strict Leakage Prevention**: Scalers (`StandardScaler`) and categorical encoders (`OrdinalEncoder`) are fitted strictly on training splits. Validation and test splits remain unobserved until evaluation.
- **Data Stream Separation**: Clinical EHR data, synthetic patient bundles, and cybersecurity network telemetry reside in strictly separated directories (`data/raw/`, `data/synthetic/`, and `data/splits/`).

---

## 2. Hardware and Software Environment Specifications

```
Operating System:           macOS (Darwin arm64, Apple Silicon)
Python Version:             Python 3.14.0
PyTorch Acceleration:       MPS (Metal Performance Shaders) Device
Post-Quantum Library:       pqcrypto (C-optimized FIPS 203 & 204 implementations)
Symmetric Crypto:           cryptography 46.0.4 (OpenSSL 3.x bindings)
Web & API Framework:        FastAPI + Pydantic v2.13
Dashboard:                  Streamlit + Plotly
Benchmark Dataset (AI):     Edge-IIoTset (Ferrag et al., 157,800 records, 63 columns)
Reference Corpus (EHR):     Synthea FHIR R4 Sample (557 bundles, 409,973 resources)
Generated Cohort:           100,000 synthetic patient bundles (1,650,322 FHIR resources)
```

---

## 3. Automated One-Command Replication Instructions

To reproduce the complete experimental evaluation pipeline from scratch:

```bash
# 1. Verify Dataset Integrity (Quality Gate QG1)
python scripts/verify_datasets.py

# 2. Run All Benchmarks, Threat Simulations, and Model Evaluations
python scripts/run_all_experiments.py

# 3. Launch Interactive Research Dashboard
streamlit run dashboard/streamlit_app.py
```

All generated tables (`results/tables/*.csv`), metric manifests (`results/metrics/*.json`), figures (`figures/*.png`), and HTML audits (`reports/*.html`) are automatically updated.
