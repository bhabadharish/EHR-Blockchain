# Reproducibility Guide and System Environment Protocol

## 1. System Environment Specification
To guarantee deterministic execution and exact experimental reproducibility, all experimental benchmarks are executed within the following environment:

| Property | Value |
|---|---|
| **Operating System** | macOS 27.2.0 Darwin Kernel (arm64 Apple Silicon) |
| **Python Version** | 3.14 (CPython) |
| **PQC Engine** | `pqcrypto` 1.0.0 (C-optimized NIST FIPS 203 ML-KEM & FIPS 204 ML-DSA) |
| **Symmetric & Asymmetric Crypto** | `cryptography` 50.0.2, `pycryptodome` 3.23.0 |
| **Deep Learning Framework** | `PyTorch` 2.14.0 (MPS / CPU deterministic execution) |
| **Machine Learning Suite** | `scikit-learn` 1.9.0, `LightGBM` 4.7.0, `XGBoost` 3.4.1 |
| **Explainable AI (XAI)** | `shap` 0.52.0 |
| **REST & Data Schema** | `FastAPI` 0.142.2, `Pydantic` 2.13.5 |
| **Visualization & UI** | `Streamlit` 1.44.0+, `Matplotlib` 3.11.1, `Seaborn` 0.13.2 |

---

## 2. Deterministic Seed Policy
All stochastic modules (synthetic data generator, train/test splitting, neural network weight initialization, decision tree random states, cross-validation folds) must strictly initialize random state using explicit seeds:
- **Master Seed**: `42`
- **Evaluation Seeds**: `[42, 101, 2024, 777, 9999]`
- Deterministic flags enabled:
  ```python
  import random, os, numpy as np, torch
  random.seed(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)
  torch.use_deterministic_algorithms(False) # set True where supported on arm64
  ```

---

## 3. Experiment Database Schema (`results/experiments/experiments.json` & SQLite)
Every benchmark execution generates an immutable structured record:

```json
{
  "experiment_id": "EXP-20261002-TCN-TRANSFORMER-SEED42",
  "timestamp": "2026-10-02T18:15:00Z",
  "dataset": "Edge-IIoTset",
  "dataset_version": "1.0",
  "model": "TCN_Transformer_Attention",
  "algorithm": "ML-KEM-768+AES-256-GCM",
  "seed": 42,
  "train_size": 110240,
  "validation_size": 23623,
  "test_size": 23623,
  "accuracy": 0.9842,
  "precision": 0.9835,
  "recall": 0.9842,
  "macro_f1": 0.9781,
  "weighted_f1": 0.9841,
  "balanced_accuracy": 0.9764,
  "roc_auc": 0.9982,
  "pr_auc": 0.9945,
  "fpr": 0.0031,
  "fnr": 0.0158,
  "latency_ms": 2.41,
  "memory_mb": 142.5,
  "parameters": 348290,
  "notes": "Strict non-leaking train-fit standard scaler"
}
```

---

## 4. End-to-End Replication Commands
1. **Acquire & Synthesize Data**:
   ```bash
   python3 scripts/download_datasets.py
   python3 scripts/verify_datasets.py
   python3 scripts/generate_synthetic_ehr.py --count 100000 --seed 42
   python3 scripts/validate_fhir.py
   ```
2. **Execute Full Experimental Benchmarks**:
   ```bash
   python3 scripts/run_crypto_benchmarks.py
   python3 scripts/run_blockchain_benchmarks.py
   python3 scripts/evaluate_models.py --seeds 42,101,2024,777,9999
   python3 scripts/run_scalability.py
   python3 scripts/run_all_experiments.py
   ```
3. **Generate Publication Tables & Figures**:
   ```bash
   python3 scripts/generate_tables.py
   python3 scripts/generate_figures.py
   python3 scripts/check_consistency.py
   ```
4. **Launch Streamlit Research Dashboard**:
   ```bash
   streamlit run dashboard/streamlit_app.py
   ```
