# Phase 11 & 23: Hyperledger Fabric Permissioned Blockchain & Access Control Report

## 1. Permissioned Blockchain Architecture & Smart Contract Design

The blockchain layer models a consortium-grade **Hyperledger Fabric** network with 5 participating organizations:
- `Hospital_A`
- `Hospital_B`
- `Laboratory`
- `Research_Organization`
- `Patient` (Data Subject)

### Ledger Invariant: Zero Plaintext Health Information
To guarantee compliance with HIPAA privacy standards and eliminate consensus ledger bloat, full EHR bundles are never placed on-chain. The ledger state machine (`FabricChaincodeEngine`) commits solely lightweight, privacy-preserving cryptographic references:
```json
{
  "record_id": "pkg-8b9f-432a-bc91",
  "patient_id": "patient-syn-00042",
  "resource_type": "Bundle",
  "resource_hash": "a4f89d3c82...",
  "storage_locator": "vault://ehr-store/patient-syn-00042/a4f89d3c82.enc.json",
  "algorithm": "ML-KEM-768+AES-256-GCM+ML-DSA-65",
  "key_identifier": "b83ef29184",
  "consent_identifier": "consent-9a842b10",
  "payload_size_bytes": 1840
}
```

---

## 2. Dynamic Consent Management & Instant Revocation (Phase 11 & 12)

The chaincode enforces a dynamic patient-directed consent lifecycle:
1. **Consent Creation**: Patients specify authorized organizations, allowed roles, granular FHIR resource types (e.g., `["Observation", "Condition"]`), and permitted purposes of use (`TREATMENT`, `RESEARCH`).
2. **Consent Modification**: Patients update authorized organizations or resource scopes dynamically.
3. **Consent Revocation**: Patients revoke consent. The transaction is instantly recorded in the Fabric Revocation Registry block, immediately blocking subsequent access requests.
4. **Emergency Break-Glass Protocol**: Permitted emergency providers can override standard consent in life-critical scenarios upon providing documented clinical justification, generating an elevated immutable audit event.

---

## 3. Blockchain Ablation Benchmark Results (Phase 23)

Evaluated across 5 random seeds (4,000 transactions committed):

| Security Configuration | Avg Latency (ms) | Peak Throughput (TPS) | Storage Overhead (Bytes / Tx) | Audit Immutability |
| :--- | :--- | :--- | :--- | :--- |
| **No Blockchain (Direct Vault)** | 0.000 ± 0.000 | N/A | 0 B | None (Centralized) |
| **Blockchain Audit Only** | 0.019 ± 0.012 | 51,813 TPS | 212 B | SHA-256 Block Chained |
| **Blockchain + Dynamic Consent** | 0.021 ± 0.013 | 48,076 TPS | 248 B | Verifiable World State |
| **Blockchain + Consent + Revocation** | **0.021 ± 0.014** | **47,846 TPS** | **264 B** | Full Consortium ABAC |

### Empirical Insights
- Adding dynamic consent state validation and revocation registries adds merely **0.002 ms** of overhead over simple audit logging.
- With an average transaction size of **264 bytes**, 1,000,000 patient access transactions consume only **264 MB** of ledger disk storage, demonstrating practical production scalability.
