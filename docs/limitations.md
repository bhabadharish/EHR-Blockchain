# Scientific Limitations and Boundary Conditions

## 1. Synthetic Data & Clinical Generalizability
- **Explicit Labeling**: All generated FHIR patient bundles are strictly **synthetic** derivations based on Synthea statistical templates and HL7 FHIR R4 schema profiles. They do **not** represent real human subjects, and no identifiable private health information (PHI) is reproduced.
- **Pathophysiological Boundaries**: While demographic, diagnostic, and laboratory values respect valid clinical ranges (e.g., physiological boundaries for blood pressure and glucose), synthetic data lacks complex, unmeasured latent covariates, rare syndromic interactions, and real-world charting noise found in authentic clinical workflows.
- **Prognostic Disclaimer**: The synthetic EHR data is intended solely for interoperability validation, payload benchmarking, schema testing, and cryptographic throughput analysis. It must **not** be used for clinical decision-making or medical prognosis.

---

## 2. Cryptographic Guarantees & Assumptions
- **No Novel Primitives**: This project utilizes established implementations of NIST-standardized algorithms (ML-KEM-768/1024 via FIPS 203, ML-DSA-65/87 via FIPS 204, AES-256-GCM via NIST SP 800-38D, and SHA-3-256 via FIPS 202). No cryptographic primitives have been designed from scratch.
- **Absence of Formal Proofs**: We do not claim formal cryptographic security proofs (e.g., IND-CCA2 reduction theorems) for the composability of the complete application protocol, but rely on the standard model security reductions established by NIST.
- **Side-Channel Vulnerabilities**: Benchmarks measure algorithmic execution time and memory footprint on commodity Apple Silicon (arm64). Physical side-channel attacks (power analysis, electromagnetic radiation, cache-timing on shared virtualized clouds) are outside the empirical scope of this prototype.

---

## 3. Blockchain & Consensus Simulation
- **Permissioned Prototype Scope**: The blockchain tier implements a high-fidelity state machine and smart contract prototype reflecting Hyperledger Fabric's execute-order-validate flow and channel isolation semantics.
- **Network Latency Constraints**: While local execution accurately reflects transaction serialization, cryptographic signature verification, state database updates, and consensus logic, wide-area network (WAN) multi-cloud propagation delays, BFT network partition healing, and Byzantine peer churn are simulated through deterministic jitter models rather than multi-datacenter physical deployments.

---

## 4. Threat Detection & Telemetry Distribution Shift
- **Domain Boundaries**: Network intrusion models are trained and evaluated on the Edge-IIoTset benchmark. While Edge-IIoTset offers realistic healthcare-relevant protocols (MQTT, CoAP, HTTP, Modbus) and volumetric attacks (DDoS, injection, malware), operational hospital networks exhibit unique baseline traffic patterns (e.g., DICOM image transfers, HL7 v2 MLLP streams).
- **Adversarial Robustness**: The TCN-Transformer model demonstrates state-of-the-art anomaly classification under standard test splits; however, sophisticated white-box adversarial perturbations designed specifically to evade neural attention heads are not fully evaluated in this study.
