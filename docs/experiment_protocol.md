# Experiment Protocol: Models, Cryptography, Blockchain, and Scalability

## 1. Experimental Integrity Mandates
1. **Zero Data Fabrication**: All reported numbers, metrics, latencies, and figures must originate from actual code execution.
2. **Strict Leakage Prevention**:
   - Data is split into `TRAIN` (70%), `VALIDATION` (15%), and `TEST` (15%) before any preprocessing.
   - Scalers, encoders, and feature selectors are fitted **strictly** on the `TRAIN` set and applied transformatively to `VALIDATION` and `TEST`.
   - Under no circumstances is the test set oversampled, evaluated iteratively for hyperparameter tuning, or used to guide architecture decisions.
3. **Statistical Validity**:
   - All AI model evaluations and latency experiments must be evaluated across at least 5 distinct random seeds (`42`, `101`, `2024`, `777`, `9999`).
   - Reports must provide Mean, Standard Deviation, and 95% Confidence Intervals:
     $$\text{CI}_{95} = \bar{x} \pm 1.96 \cdot \frac{s}{\sqrt{n}}$$
   - Statistical significance between proposed model and baselines must be evaluated using paired t-tests or Wilcoxon signed-rank tests ($p < 0.05$).

---

## 2. Threat Detection Model Benchmark Suite

### 2.1 Model Specifications
- **Baseline 1: Logistic Regression (LR)**: L2 regularized, liblinear/lbfgs solver, max_iter=1000.
- **Baseline 2: Random Forest (RF)**: 100 estimators, max_depth=16, Gini criterion.
- **Baseline 3: LightGBM / XGBoost**: Gradient boosted decision trees, 100 trees, learning_rate=0.05, num_leaves=31.
- **Baseline 4: Multi-Layer Perceptron (MLP)**: 3 dense layers (128 -> 64 -> num_classes) with ReLU and Dropout(0.2).
- **Baseline 5: 1D-CNN**: Two 1D convolutional layers (kernel size 3, filters 64 and 128), BatchNorm, MaxPool1D, Dense head.
- **Baseline 6: BiLSTM**: Bidirectional LSTM (hidden dimension 64, 2 layers), Dropout(0.3), Dense classification head.
- **Proposed Architecture: TCN + Transformer + Multi-Head Attention**:
  - Temporal Convolutional Network (TCN): Dilated causal 1D convolutions (dilations: 1, 2, 4, 8; kernel size: 3; channels: 64) with residual connections.
  - Transformer Encoder: 2 transformer blocks, 4 attention heads, feed-forward dimension 128, GELU activation, layer normalization.
  - Multi-Head Attention Pooling: Attentive temporal aggregation across sequence representations.
  - Final Dense Head: Dense(64) -> Dropout(0.2) -> Dense(num_classes).

### 2.2 Metrics Definition
- **Accuracy**: $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$
- **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$
- **Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$
- **Macro-F1**: $\frac{1}{C} \sum_{c=1}^C F1_c$ (Equal weight across classes, crucial for imbalanced attacks)
- **Weighted-F1**: $\sum_{c=1}^C \frac{N_c}{N} F1_c$
- **Balanced Accuracy**: $\frac{1}{C} \sum_{c=1}^C \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$
- **False Positive Rate (FPR)**: $\frac{\text{FP}}{\text{FP} + \text{TN}}$
- **False Negative Rate (FNR)**: $\frac{\text{FN}}{\text{FN} + \text{TP}}$
- **ROC-AUC & PR-AUC**: Area under Receiver Operating Characteristic and Precision-Recall curves.

---

## 3. Cryptographic Benchmark Protocol

### 3.1 Evaluated Suites
1. **Classical Suite**: ECDH (NIST P-256) + AES-256-GCM + ECDSA P-256 + SHA-256.
2. **PQC Standard**: ML-KEM-768 (NIST Security Category 3) + AES-256-GCM + ML-DSA-65 (NIST Security Category 3) + SHA-3-256.
3. **PQC High**: ML-KEM-1024 (NIST Security Category 5) + AES-256-GCM + ML-DSA-87 (NIST Security Category 5) + SHA-3-256.
4. **Hybrid Suite**: ECDH P-256 + ML-KEM-768 combined key establishment + AES-256-GCM + ML-DSA-65 + SHA-3-256.

### 3.2 Evaluation Variables
- Key generation latency (ms)
- Encapsulation / Decapsulation latency (ms)
- Encryption / Decryption latency (ms)
- Digital signing and verification latency (ms)
- Public key, private key, ciphertext, and signature sizes (bytes)
- Memory usage and total transmission package overhead (bytes)

---

## 4. Blockchain & Access Control Benchmark Protocol

### 4.1 Chaincode Operations Evaluated
- `create_consent`: Register patient dynamic consent policy.
- `update_consent`: Modify authorized organizations or resource scopes.
- `revoke_consent`: Immediate consent invalidation.
- `request_access`: Evaluate RBAC/ABAC rules against ledger state.
- `record_audit`: Append immutable cryptographic access event.

### 4.2 Blockchain Ablation Configurations
1. **Ablation 0**: No blockchain (direct database baseline).
2. **Ablation 1**: Blockchain audit logging only.
3. **Ablation 2**: Blockchain audit + dynamic consent.
4. **Ablation 3**: Blockchain audit + consent + real-time revocation checking.

---

## 5. Scalability & End-to-End Evaluation Protocol
- **Dataset Scales**: Evaluated across synthetic corpus sizes:
  $N \in \{10,000, 25,000, 50,000, 100,000, 250,000\}$.
- **End-to-End Latency Breakdown**:
  $$T_{\text{total}} = T_{\text{validation}} + T_{\text{minimization}} + T_{\text{encapsulation}} + T_{\text{AES-enc}} + T_{\text{signature}} + T_{\text{storage}} + T_{\text{blockchain}} + T_{\text{retrieval}} + T_{\text{decapsulation}} + T_{\text{AES-dec}} + T_{\text{verify}}$$
- Measured throughput (records/sec), memory footprint (MB), and CPU utilization across load increments.
