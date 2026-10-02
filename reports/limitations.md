# Scientific Limitations & Boundary Conditions

## 1. Scope of Synthetic Clinical Data
- **Synthetic Origin Disclaimer**: The 100,000 synthetic patient bundles (1,650,322 resources) were generated via parametric statistical sampling modeled after Synthea and MIMIC-IV distributions. While structurally and relationally valid HL7 FHIR R4 resources, they are **explicitly synthetic** and should not be used as clinical ground truth for treatment or therapeutic efficacy.
- **Parametric Bounds**: Statistical tests confirm similarity in resource frequencies and common vital signs, but complex rare disease multi-morbidities remain bounded by synthetic generator constraints.

---

## 2. Cryptographic & Quantum Boundaries
- **No Novel Mathematical Primitive Claims**: All post-quantum security claims rest on the peer-reviewed NIST standards (FIPS 203 for ML-KEM and FIPS 204 for ML-DSA) based on the hardness of Module Learning with Errors (M-LWE). No independent cryptographic security proofs are asserted by this prototype.
- **Hardware Architecture Dependency**: Benchmark throughputs reflect Apple Silicon hardware with native ARM NEON vector instructions and C-level optimization. Microcontrollers or resource-constrained IoT edges will experience higher encapsulation latencies.

---

## 3. Blockchain Consensus Assumptions
- **Permissioned vs Public Ledger**: This work models a permissioned consortium (Hyperledger Fabric) appropriate for HIPAA/GDPR health networks. It does not apply to permissionless Proof-of-Work or high-fee smart contract environments (e.g. public Ethereum).
- **Scale Horizon**: Evaluated experimentally up to 100,000 patient records. Systems managing hundreds of millions of historical encounters require tiered off-chain indexing and distributed archival pruning.

---

## 4. Threat Shift & Machine Learning Boundaries
- **Adversarial Drift**: Threat detection models trained on Edge-IIoTset reflect network telemetry up to the dataset publication date. Novel zero-day polymorphic attacks may require continual fine-tuning or semi-supervised anomaly detection.
- **Telemetry-Clinical Boundary**: Security telemetry is strictly separated from clinical EHR resources. The model predicts cyber threats from transport metadata, NOT from patient clinical diagnostic content.
