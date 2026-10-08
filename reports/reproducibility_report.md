# Reproducibility and Provenance Report
**Project:** HAB-IDS Architecture
**Standard:** Zero Data Leakage & Exact Reproducibility Gate

## 1. System Environment
- **OS Platform:** `macOS-27.2-arm64-arm-64bit-Mach-O`
- **Processor:** `arm`
- **Python Version:** `3.14.7`

## 2. Core Dependencies
| Package | Version |
| :--- | :--- |
| `numpy` | `2.5.3` |
| `pandas` | `3.0.6` |
| `pyarrow` | `25.0.1` |
| `scipy` | `1.18.1` |
| `sklearn` | `1.9.1` |
| `xgboost` | `3.4.1` |
| `lightgbm` | `4.7.0` |
| `catboost` | `1.2.10` |
| `torch` | `2.14.0` |
| `cryptography` | `50.0.2` |
| `joblib` | `1.6.0` |

## 3. Dataset Integrity and Hashes
| Dataset Key | Filename / Parquet | SHA-256 Digest |
| :--- | :--- | :--- |
| `edge_iiot` | `data/processed/edge_iiot.parquet` | `c0f74a56b4683b648e76...` |
| `fhir_security` | `data/processed/fhir_security.parquet` | `4fa0b49505e444ad8f3a...` |
| `ciciot2023_train` | `data/processed/ciciot_train.parquet` | `8c2a5796c8e310c12b1e...` |
| `ciciot2023_val` | `data/processed/ciciot_val.parquet` | `8f096c0a2ba045c66246...` |
| `ciciot2023_test` | `data/processed/ciciot_test.parquet` | `4ed37c31a36288c4d83a...` |

## 4. Random Seeds and Evaluation Protocol
- **Evaluated Seeds:** `[42, 123, 999]`
- **Test Set Status:** Strictly locked; isolated from preprocessing, hyperparameter search, threshold search, and calibration.
- **Deterministic Hash Verification:** Executed by `scripts/verify_results.py`.
