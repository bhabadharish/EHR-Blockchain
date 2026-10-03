# Live Demonstration & Evaluator Guide

## 1. Quick Launch
Execute the offline demonstration dashboard using:
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 2. Recommended Walkthrough Flow (Viva / Presentation)

### Step 1: System Overview (Landing)
- Point to the **Status Bar**: Explain that all ML, PQC, and FHIR engines are `READY`, and Blockchain is in `SIMULATION MODE`.
- Point to **Verified Test Performance**: 99.06% Test Accuracy, 98.98% Macro Recall, 0.923% FNR on 26,251 test samples.
- Show the **Target vs Measured Table**: Highlight the transparency of documenting the Macro-F1 gap (-1.37%) caused by raw class imbalance.

### Step 2: Dataset Intelligence
- Review data harmonization across **CICIoT2023** (120,000 flows), **Edge-IIoTset** (40,000 records), and **Synthetic FHIR** (15,000 events).
- Highlight the **Leakage Validation Panel** confirming 0 sample overlap and training-only preprocessor fitting.

### Step 3: Performance & Error Analysis
- Inspect the interactive **Confusion Matrix** for CA-HTDNet: 1,850 True Negatives, 24,155 True Positives, and only 21 False Positives.
- Open the **Threshold Lab**: Adjust the What-If slider to demonstrate trade-offs between FPR and FNR without altering the official benchmark (0.57).

### Step 4: FHIR Security Pipeline (11 Stages)
- Configure an incoming request: Doctor requesting Observation Read for Patient #1024.
- Click **"🔎 Analyze Request & Execute Pipeline"**:
  - Watch the request pass through Schema Validation, Consent Verification, Threat Inference, Risk Scoring, Crypto Selection, Authenticated Encryption, and Fabric Ledger Anchoring.

### Step 5: Live Threat Detection
- Trigger prebuilt clinical scenarios:
  - `🟢 Normal Doctor Access` → Threat Probability ~0.20 → **ALLOW**
  - `🔴 Bulk EHR Extraction` → Threat Probability ~0.88 → **BLOCK**
  - `🔴 High-Risk IoMT Device` → Threat Probability ~0.95 → **QUARANTINE_DEVICE**
- Review the **Feature Attribution** table explaining which clinical parameters drove the decision.

### Step 6: Post-Quantum Crypto Agility
- Click **"▶ Run Live PQC Benchmark"**:
  - Show real-time measured timings: ML-KEM-768 KeyGen & Encapsulation (<1 ms), ML-DSA-65 signature verification (<0.1 ms).

### Step 7: Blockchain Audit & Tamper Test
- Click **"▶ Execute Live Tamper Test"**:
  - A real FHIR record is stored in the off-chain vault and anchored on-chain.
  - 4 bytes are corrupted on disk.
  - The ledger computes SHA3-256 and instantly flags `🚨 LIVE TAMPERING DETECTED`.

### Step 8: Reproducibility & Result Reconciliation
- Click **"▶ Run Full Automated Validation Sequence"**:
  - Reconciles all 10 models in real-time, displaying `✓ Result Validation Passed: All 10 models verified with 0 discrepancies!`.
