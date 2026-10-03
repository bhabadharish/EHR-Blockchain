# Result Provenance & Execution Audit

## 1. Primary Metadata
- **Experiment ID:** `CAHTDNet_final_locked_v001`
- **Model Architecture:** CA-HTDNet (FT-Transformer + Causal Dilated TCN + BiGRU Multi-Head Attention + Calibrated Temperature Head)
- **Target Hardware:** Apple MacBook Air (Apple M2, 16 GB Unified Memory, macOS Darwin arm64)
- **Primary Datasets:** CICIoT2023 (120,000 flows) + Edge-IIoTset (40,000 flows) + Synthetic FHIR R4 (15,000 records)
- **Harmonized Dataset Total:** 175,000 records (16 unified features: 13 numerical, 3 categorical)
- **Data Splits:** 70% Train (122,499), 15% Validation (26,250), 15% Held-out Test (26,251)
- **Deterministic Seed:** 42

---

## 2. Canonical Artifact Hashes
| Artifact | Relative Path | SHA-256 Digest | Size |
|---|---|---|---|
| **CA-HTDNet State Dict** | `models/proposed/ca_htdnet.pt` | `dc7fa83bba31b1ec0d80e663bed31a763b06b16f0bd51d99eb7d5a7b71fc282c` | 1,665,309 B |
| **Unified Preprocessor** | `models/preprocessors/preprocessor.pkl` | `5f5ec9629bfe018de686084bcd87d6fb7601fab7f2ff5e3edf1adb9a6158928f` | 1,769 B |
| **Optimal Threshold Meta** | `models/proposed/threshold.json` | `a6018bb2c0aeb8effdf43b004351716ed9dbcdfdf49a138e8943c8e1fb1ababf` | 396 B |
| **Model Configuration** | `models/proposed/config.json` | `6584764c343e5780faee186be1be361ccb35b36bbc747f9621667f39df7f98a3` | 290 B |
| **Locked Test Partition** | `data/splits/test.parquet` | `69169cd8b113b638f2d19af8f7281ecb3744023e996c443b8e1e06ea58eec99d` | 953,701 B |
| **Experiment Registry** | `results/experiment_registry.json` | Authenticated Canonical Registry | JSON |

---

## 3. Verified Empirical Metrics (26,251 Held-out Test Samples)
- **Accuracy:** `99.0629%`
- **Macro Recall:** `98.9774%`
- **Macro Precision:** `94.5349%`
- **Macro F1:** `96.6296%`
- **False Negative Rate (FNR):** `0.9229%`
- **False Positive Rate (FPR):** `1.1224%`
- **ROC-AUC:** `0.9990`
- **PR-AUC:** `0.9999`
- **Matthews Correlation Coefficient (MCC):** `0.9341`
- **Confusion Matrix:** `[[1850, 21], [225, 24155]]`
- **Inference Latency:** `~1.1 ms/sample` (Apple Silicon M2 / CPU)
