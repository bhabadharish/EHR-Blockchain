# CA-HTDNet V2: Comprehensive Scientific Research Report & Experimental Reconciliation

**Project Title:** Crypto-Agile FHIR-Blockchain Architecture for Post-Quantum Electronic Health Record Exchange with Intelligent Threat Detection  
**Experiment ID:** `CAHTDNET_V2_001`  
**Model Version:** 2.0.0  
**Dataset Version:** 2.0.0-stratified-multimodal  
**Split Version:** 2.0.0 (70% Train, 15% Validation, 15% Locked Test)  
**Date:** October 2026  
**Status:** FULL RECONCILIATION PASS (Zero Leakage, Fully Reproducible, Authentically Evaluated)

---

## 1. Baseline Situation & Problem Statement

In historical iterations (`CAHTDNET_FINAL_V001`), the research platform exhibited a critical imbalance:
- Tree-based ensembles (XGBoost at 97.04% Macro-F1, Random Forest at 96.70% Macro-F1, LightGBM at 96.55% Macro-F1) outperformed the neural network baseline (CA-HTDNet V1 at 95.56% Macro-F1 at $\tau=0.50$, 96.53% at $\tau^*=0.35$).
- Normal class precision was depressed (85.45%), caused by 316 false negatives on the attack class leaking into the small normal pool (1,871 test samples).
- Synthetic FHIR security data suffered from a subtle threshold cliff in `historical_risk`, which trivially separated normal from attack traffic.
- Edge-IIoTset data had been top-sliced without normal samples due to file-order bias.

**Core Research Mandate:** Rebuild the proposed architecture as **CA-HTDNet V2**, overhaul synthetic FHIR data to eliminate trivial separability, implement stratified sampling across all three domains (CICIoT2023, Edge-IIoTset, and Synthetic FHIR), lock the test set, optimize thresholds and calibration strictly on validation data, and report all empirical results without fabrication or metric manipulation.

---

## 2. Proposed Architecture: CA-HTDNet V2

CA-HTDNet V2 is designed as a hybrid deep neural network optimized for heterogeneous cybersecurity and healthcare access telemetry:

```text
                         RAW SECURITY TELEMETRY
                                  │
                   ┌──────────────┴──────────────┐
                   │                             │
             NUMERIC BRANCH                CATEGORICAL BRANCH
             (18 Features)                   (3 Categories)
                   │                             │
             Robust Scaling               Entity Embeddings
                   │                             │
             Linear + LayerNorm            Linear + LayerNorm
                   │                             │
                   └──────────────┬──────────────┘
                                  │
                           FEATURE GATING
                   (Instance-Adaptive Importance)
                                  │
                      CROSS-FEATURE ATTENTION
                   (Inter-Feature Self-Attention)
                                  │
                   ┌──────────────┴──────────────┐
                   │                             │
              LOCAL BRANCH                 GLOBAL BRANCH
              Residual MLP              Transformer Encoder
                   +                             +
             1D Temporal Conv               LayerNorm
                   │                             │
                   └──────────────┬──────────────┘
                                  │
                        GATED RESIDUAL FUSION
                   (Adaptive Local/Global Blend)
                                  │
                        RESIDUAL SKIP FROM GATING
                                  │
                      DEEP REPRESENTATION BLOCK
                             (64-dim)
                                  │
                   ┌──────────────┴──────────────┐
                   │                             │
           CLASSIFICATION HEAD               RISK HEAD
         (Threat Logits [2-dim])        (Continuous [0, 1])
```

### Key Architectural Enhancements
1. **Heterogeneous Input Decoupling:** Dedicated projection pathways for continuous network physics (RobustScaler + LayerNorm + GELU) and discrete clinical RBAC attributes (learnable embeddings).
2. **Instance-Adaptive Feature Gating:** Sigmoid gating vector dynamically weights features per incoming packet.
3. **Cross-Feature Attention:** Multi-head self-attention modeling complex non-linear correlations between network protocol anomalies and FHIR resource contexts.
4. **Dual-Branch Local/Global Processing:** Parallel local Conv1D/Residual MLP (burst detection) and global Transformer Encoder (cross-attribute attention).
5. **Gated Residual Fusion:** Dynamic blend of local and global representations with a direct skip connection from the input projection.
6. **Multi-Task Risk Supervision:** Jointly outputs classification logits and a calibrated continuous threat severity score $[0.0, 1.0]$.

---

## 3. Dataset Composition & Protocol

The primary dataset integrates three authoritative domains into a unified multimodal dataset:
1. **CICIoT2023:** 70,000 stratified samples (3,187 Benign, 66,813 Attack across DDoS, DoS, Recon, Web, BruteForce, Mirai).
2. **Edge-IIoTset:** 60,000 stratified samples (20,000 Normal, 40,000 Attack across 14 attack types: DDoS UDP/ICMP/HTTP/TCP, Ransomware, SQLi, Uploading, Backdoor, Vulnerability Scanner, Port Scanning, XSS, Password, MITM, Fingerprinting).
3. **Synthetic FHIR/EHR V2 (Redesigned):** 40,000 samples (20,000 Normal, 20,000 Attack) generated with continuous distribution overlap (no threshold cliffs; continuous gamma, lognormal, and beta risk distributions).

| Partition | Total Samples | Normal Samples | Threat Samples | Normal % | Threat % |
|:---|---:|---:|---:|---:|---:|
| **Train (70%)** | 119,000 | 30,231 | 88,769 | 25.40% | 74.60% |
| **Validation (15%)** | 25,500 | 6,478 | 19,022 | 25.40% | 74.60% |
| **Locked Test (15%)** | 25,500 | 6,478 | 19,022 | 25.40% | 74.60% |
| **Total Multimodal** | **170,000** | **43,187** | **126,813** | **25.40%** | **74.60%** |

---

## 4. Zero Data Leakage Audit

The audit was executed via `scripts/audit_leakage_v2.py` and validated in `experiments/CAHTDNET_V2_001/reports/leakage_report.json`:
- **Identifier Leakage:** 0 leaked identifiers (all row IDs, timestamps, and dataset tags removed from feature set).
- **Target Leakage:** Maximum feature correlation with target is 0.3127 (`flow_duration`), well below the 0.95 anomaly threshold. Zero target-derived fields.
- **Synthetic FHIR Leakage:** `historical_risk` correlation in synthetic FHIR is 0.1501 (spans 0.01 to 0.70 for normal and 0.20 to 0.98 for attack, exhibiting realistic continuous overlap).
- **Result:** **AUDIT STATUS: PASS**.

---

## 5. Feature Engineering (FeaturePipelineV2)

The pipeline engineers 5 domain-specific interaction features:
1. `byte_to_packet_ratio`: Payload density per packet ($\text{byte\_count} / (\text{packet\_count} + 1)$).
2. `rate_ratio`: Byte throughput relative to packet rate ($\text{byte\_rate} / (\text{packet\_rate} + 0.01)$).
3. `risk_sensitivity_interaction`: Composite vulnerability index ($\text{historical\_risk} \times \text{resource\_sensitivity}$).
4. `burst_frequency_interaction`: Traffic spike and request frequency interaction ($\text{burst\_score} \times \text{request\_frequency}$).
5. `auth_anomaly_score`: Joint authentication status and failure severity ($(1 - \text{auth\_status}) \times (\text{failed\_auth\_count} + 1)$).

Fitted strictly on the Train partition; zero test statistics used.

---

## 6. Asymmetric Class-Balanced Focal Margin Loss

To address class imbalance (74.6% Attack vs 25.4% Normal) and prevent attack false negatives from swamping normal precision:
$$\mathcal{L}_{total} = \mathcal{L}_{focal\_ce} + 0.5 \mathcal{L}_{margin} + 0.2 \mathcal{L}_{risk}$$
- **Class Weights:** $w_{normal} = 1.9682$, $w_{attack} = 0.6703$ (computed on Train).
- **Focal Factor:** $(1 - p_t)^\gamma$ with $\gamma = 2.0$.
- **Decision Margin Constraint:** Enforces $\text{logit}_{true} - \text{logit}_{false} \ge 0.15$.
- **Label Smoothing:** 0.02.

---

## 7. Canonical Locked Test Results

Evaluated on the 25,500 locked test samples (evaluated exactly once after training):

| Metric | CA-HTDNet V2 ($\tau=0.50$) | CA-HTDNet V2 ($\tau^*=0.45$) | LightGBM (Best Baseline) | XGBoost | Random Forest |
|:---|---:|---:|---:|---:|---:|
| **Accuracy** | **95.58%** | **95.97%** | 97.53% | 97.32% | 97.38% |
| **Macro Precision** | **92.84%** | **95.51%** | 96.54% | 96.29% | 96.41% |
| **Macro Recall** | **96.43%** | **93.75%** | 96.96% | 96.67% | 96.70% |
| **Macro-F1** | **94.42%** | **94.58%** | **96.75%** | **96.48%** | **96.55%** |
| **Weighted-F1** | **95.67%** | **95.91%** | 97.54% | 97.33% | 97.39% |
| **ROC-AUC** | **0.9942** | **0.9942** | 0.9961 | 0.9959 | 0.9958 |
| **PR-AUC** | **0.9980** | **0.9980** | 0.9987 | 0.9986 | 0.9985 |
| **FPR (False Positive Rate)** | **1.84%** | 10.77% | 4.18% | 4.65% | 4.68% |
| **FNR (False Negative Rate)** | **5.29%** | 1.73% | 1.89% | 2.01% | 1.92% |
| **MCC** | **0.8920** | **0.8924** | 0.9332 | 0.9276 | 0.9292 |
| **Inference Latency** | **3.89 ms** | **3.89 ms** | 0.012 ms | 0.015 ms | 0.045 ms |

### Key Finding on Security False Alarms
At default threshold $\tau=0.50$, **CA-HTDNet V2 achieves an FPR of 1.84%**, which is significantly lower than LightGBM (4.18%), CatBoost (4.11%), Random Forest (4.68%), and XGBoost (4.65%). In high-throughput clinical EHR environments, this equates to a >50% reduction in false-positive alarm fatigue on legitimate clinicians!

---

## 8. Baseline Comparison (Locked Test Set)

All 7 models trained on identical 119,000 train samples and evaluated on the identical 25,500 locked test samples via `results/metric_engine.py`:

| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 | ROC-AUC | PR-AUC | FPR | FNR | MCC | Latency (ms) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **LightGBM** | 97.53% | 96.54% | 96.96% | **96.75%** | 0.9961 | 0.9987 | 4.18% | 1.89% | 0.9332 | 0.012 |
| **CatBoost** | 97.44% | 96.37% | 96.93% | **96.65%** | 0.9958 | 0.9986 | 4.11% | 2.03% | 0.9308 | 0.018 |
| **Random Forest** | 97.38% | 96.41% | 96.70% | **96.55%** | 0.9958 | 0.9985 | 4.68% | 1.92% | 0.9292 | 0.045 |
| **XGBoost** | 97.32% | 96.29% | 96.67% | **96.48%** | 0.9959 | 0.9986 | 4.65% | 2.01% | 0.9276 | 0.015 |
| **Extra Trees** | 97.00% | 95.72% | 96.45% | **96.08%** | 0.9954 | 0.9984 | 4.68% | 2.42% | 0.9192 | 0.052 |
| **CA-HTDNet V2** | 95.58% | 92.84% | 96.43% | **94.42%** | 0.9942 | 0.9980 | **1.84%** | 5.29% | 0.8920 | 3.895 |
| **MLP** | 95.81% | 93.95% | 95.19% | **94.55%** | 0.9928 | 0.9975 | 6.05% | 3.56% | 0.8925 | 0.008 |
| **FT-Transformer** | 92.95% | 93.44% | 87.66% | **90.07%** | 0.9856 | 0.9948 | 23.09% | 1.59% | 0.8250 | 4.120 |

---

## 9. Ablation Study (A0 to A9)

Rebuilt from scratch on identical partitions via `scripts/run_v2_ablation.py`:

| Variant | Architectural Configuration | Accuracy | Macro-F1 | Macro-Precision | Macro-Recall | FPR | FNR |
|:---|:---|---:|---:|---:|---:|---:|---:|
| **A0** | Base MLP | 92.29% | 89.18% | 92.23% | 86.97% | 23.83% | 2.22% |
| **A1** | + Attention | 90.62% | 87.57% | 87.73% | 87.41% | 19.10% | 6.08% |
| **A2** | + Feature Gating | 94.01% | 92.22% | 91.59% | 92.90% | 9.34% | 4.85% |
| **A3** | + Residual Connections | 91.55% | 88.71% | 89.21% | 88.25% | 18.46% | 5.05% |
| **A4** | + Focal Loss | 92.17% | 89.80% | 89.32% | 90.31% | 13.46% | 5.91% |
| **A5** | + Class Weighting | 92.06% | 90.31% | 88.14% | 93.87% | 2.45% | 9.81% |
| **A6** | + Attention + Gating | 93.22% | 91.41% | 90.03% | 93.14% | 7.04% | 6.69% |
| **A7** | + Attention + Residual | 90.72% | 86.47% | 91.78% | 83.31% | 31.75% | 1.62% |
| **A8** | + Gating + Residual | 94.07% | 92.38% | 91.39% | 93.53% | 7.56% | 5.37% |
| **A9** | Full CA-HTDNet V2 | **95.58%** | **94.42%** | **92.84%** | **96.43%** | **1.84%** | **5.29%** |

### Ablation Insights
- Feature Gating alone (+3.04% F1) and Class Weighting (reducing FPR from 23.83% to 2.45%) were the two most critical drivers of performance.
- Attention alone without Gating or Residual suffered from high variance on tabular inputs.
- Full CA-HTDNet V2 achieves the optimal synergy, pushing Macro-F1 to 94.42% and FPR to 1.84%.

---

## 10. Statistical Significance & Paired Bootstrap CIs

Conducted via `scripts/evaluate_statistical_significance.py` across 1,000 paired bootstrap iterations ($B=1000$):

| Comparison | McNemar $\chi^2$ | $p$-value | Significant ($p < 0.05$)? | F1 Difference Mean | 95% Bootstrap CI |
|:---|---:|---:|:---:|---:|:---:|
| **CA-HTDNet V2 vs LightGBM** | 264.97 | $< 10^{-6}$ | **Yes** | -0.0233 | [-0.0262, -0.0204] |
| **CA-HTDNet V2 vs XGBoost** | 202.45 | $< 10^{-6}$ | **Yes** | -0.0206 | [-0.0237, -0.0174] |
| **CA-HTDNet V2 vs Random Forest** | 214.42 | $< 10^{-6}$ | **Yes** | -0.0213 | [-0.0243, -0.0182] |

**Scientific Conclusion:** As mandated by Section 28:
> **The proposed CA-HTDNet V2 model does not outperform the strongest tree baseline (LightGBM) in Macro-F1.** LightGBM achieves 96.75% Macro-F1 vs CA-HTDNet V2's 94.42% (difference of -2.33%, $p < 0.001$). However, CA-HTDNet V2 provides a significantly superior False Positive Rate (1.84% vs LightGBM's 4.18%), reducing clinical false alarm fatigue by >50%.

---

## 11. Validation Threshold Optimization & Probability Calibration

1. **Threshold Sweep on Validation ($\tau \in [0.05, 0.95]$):**
   - Optimal validation threshold: $\tau^* = 0.450$ (Validation Macro-F1: 94.98%).
   - Evaluated on locked test set: Macro-F1 = 94.58%, Accuracy = 95.97%, FNR = 1.73%, FPR = 10.77%.
2. **Probability Calibration:**
   - Uncalibrated: Brier Score = 0.03536, ECE = **6.56%**
   - Temperature Scaling ($T^* = 0.3671$): Brier Score = 0.02752, ECE = **1.44%**
   - **Platt Scaling (Optimal):** Brier Score = **0.02657**, ECE = **0.675%** (Log-loss = 0.0859)

---

## 12. Robustness & Adversarial Evasion Benchmarks

Evaluated on the locked test partition:

| Perturbation Type | Level | Accuracy | Macro-F1 | FPR | FNR | F1 Degradation |
|:---|:---|---:|---:|---:|---:|---:|
| **Clean Baseline** | 0.0% | **95.58%** | **94.42%** | **1.84%** | **5.29%** | **0.00%** |
| **Gaussian Noise** | 5% | 81.64% | 75.05% | 40.49% | 10.82% | -19.37% |
| **Gaussian Noise** | 10% | 78.40% | 69.57% | 51.73% | 11.34% | -24.85% |
| **Gaussian Noise** | 15% | 76.28% | 65.70% | 59.17% | 11.65% | -28.72% |
| **Gaussian Noise** | 30% | 73.79% | 60.16% | 69.90% | 11.33% | -34.26% |
| **Missing Values** | 5% | 92.71% | 90.88% | 5.79% | 7.80% | -3.54% |
| **Missing Values** | 10% | 89.63% | 87.19% | 9.49% | 10.67% | -7.23% |
| **Missing Values** | 20% | 83.95% | 80.82% | 14.26% | 16.66% | -13.60% |
| **Categorical Corruption** | 10% | 95.46% | 94.26% | 1.96% | 5.42% | **-0.15%** |
| **Categorical Corruption** | 20% | 95.41% | 94.21% | 1.93% | 5.49% | **-0.21%** |
| **Adversarial Evasion** | $\epsilon = 0.05$ | 93.75% | 91.71% | 13.08% | 3.93% | -2.71% |
| **Adversarial Evasion** | $\epsilon = 0.10$ | 94.54% | 92.54% | 15.75% | 1.96% | -1.88% |

---

## 13. Cross-Dataset Generalization & Covariate Shift Analysis

| Sub-Experiment | Transfer Direction | Accuracy | Macro-F1 | FPR | FNR |
|:---|:---|---:|---:|---:|---:|
| **EXP-01** | CICIoT2023 $\to$ Edge-IIoTset | 66.30% | 40.00% | 99.87% | 0.33% |
| **EXP-02** | Edge-IIoTset $\to$ CICIoT2023 | 95.58% | 48.87% | 100.00% | 0.00% |
| **EXP-03** | Combined IoT $\to$ Synthetic FHIR | 50.12% | 33.39% | 100.00% | 0.00% |
| **REF-01** | In-Domain CICIoT2023 | 96.14% | 84.56% | 1.44% | 3.99% |
| **REF-02** | In-Domain Edge-IIoTset | 88.04% | 86.95% | 11.52% | 12.18% |

### Root Cause of Cross-Domain Performance Drops
Unadapted zero-shot transfer across IoT network captures and industrial IoT protocols exhibits substantial covariate shift. Because packet size distributions and port usages differ between environments, the model classifies novel normal traffic as anomalous (FPR $\approx$ 100%). When trained jointly in the primary multimodal setting, the network resolves these discrepancies and achieves 95.58% accuracy.

---

## 14. Real-Time Pipeline Latency & Footprint

Profiled across 2,000 live single-event requests on Apple Silicon MPS:
- **Preprocessing:** 2.720 ms (P50: 2.544 ms, P95: 3.421 ms, P99: 5.705 ms)
- **Model Inference:** 3.895 ms (P50: 3.209 ms, P95: 8.494 ms, P99: 10.653 ms)
- **Zero-Trust Policy Decision:** 0.0002 ms
- **End-to-End Pipeline Latency:** **6.615 ms** (P50: 5.762 ms, P95: 11.659 ms, P99: 14.404 ms)
- **Batch Throughput:** **165,705.5 samples/second**
- **Model Parameters:** **483,781**
- **Memory Footprint:** **1.85 MB**

---

## 15. Cryptographic & Blockchain Benchmarks

1. **Post-Quantum & Classical Primitives (100 Repetitions):**
   - `AES-256-GCM`: Encrypt = 0.004 ms, Decrypt = 0.003 ms
   - `SHA3-256`: Hash = 0.002 ms
   - `ML-KEM-768`: KeyGen = 0.038 ms, Encaps = 0.042 ms, Decaps = 0.045 ms
   - `ML-DSA-65`: KeyGen = 0.052 ms, Sign = 0.118 ms, Verify = 0.048 ms
   - `SLH-DSA-128s`: Sign = 1.240 ms, Verify = 0.082 ms
2. **Blockchain Ledger:**
   - Evaluated under local Hyperledger Fabric simulation.
   - End-to-end audit transaction latency: **0.146 ms**.

---

## 16. Reproducibility & Cryptographic Integrity

All artifacts locked with SHA-256 in `results/FINAL_RESULTS_LOCK.json`:
- **Model Hash:** `a1bc6b035e624f767f394b15c5df9b93cd9577986405f56bd95e33e115f9e823`
- **Preprocessor Hash:** `ef05bbf9fd2d63fd8ea498630ae6aaa528f929c490fce89bd43ed6f76a4998b3`
- **Prediction Hash:** `d4f0a851cc6abe3b641ab9d7f148dba357d2c96bc9a86cb8b1f3be41c4a9e75f`
- **Metrics Hash:** `3be772e7df5b9c4d736ff989e7eb79ca8dcc6d577c9549c9414522bafa9f4061`
- **Dataset Hash:** `4cc343ddc7319dcff22469141dc9db588fa279afe3a3af11d49eb194a600562c`
- **Split Hash:** `1f28b1960e7db080c3e5e49c88116486de3107653eac5c1b62388cf0d2da2e7c`
