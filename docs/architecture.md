# System Architecture: Crypto-Agile FHIR-Blockchain Architecture for Post-Quantum EHR Exchange with Intelligent Threat Detection

## 1. Architectural Overview
The architecture is structured into decoupled, modular tiers to ensure quantum resistance, cryptographic agility, strict access control, low-latency off-chain data exchange, and real-time cyber threat detection.

```
+-----------------------------------------------------------------------------------+
|                              Clinical / Client Tier                               |
|   Hospital A (Sender)        Hospital B (Receiver)       Research Org / Lab       |
+------------------------------------------+----------------------------------------+
                                           | HTTPS / REST
                                           v
+-----------------------------------------------------------------------------------+
|                                FHIR Gateway Tier                                  |
|   - FastAPI RESTful Endpoints (Patient, Observation, Condition, Bundle, etc.)     |
|   - Pydantic HL7 FHIR R4 Schema Validator                                         |
|   - Data Minimization Engine (FULL, MINIMAL, RESEARCH, EMERGENCY policies)        |
|   - Canonical JSON Serialization (RFC 8785)                                       |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                             Crypto-Agility Engine                                 |
|   - Pluggable Cryptographic Provider Interface (CryptoAgileProvider)              |
|   - Symmetric: AES-256-GCM (NIST SP 800-38D)                                      |
|   - Integrity: SHA-3-256 (FIPS 202)                                               |
|   - PQC Key Encapsulation: ML-KEM-768 / ML-KEM-1024 (FIPS 203)                    |
|   - PQC Digital Signatures: ML-DSA-65 / ML-DSA-87 (FIPS 204)                      |
|   - Classical / Hybrid Modes: ECDH P-256 + ML-KEM-768                             |
+---------------------+-------------------------------------+-----------------------+
                      |                                     |
                      v                                     v
+-----------------------------------+ +---------------------------------------------+
|    Off-Chain Storage Engine       | |      Permissioned Blockchain Tier           |
|  - Encrypted EHR Object Store     | |  - Hyperledger Fabric Smart Contract Model  |
|  - Content-Addressable Locators   | |  - Organizations: HospA, HospB, Lab, Res, Pt|
|  - Zero Plaintext on Ledger       | |  - Chaincode: Consent, Access, Revocation   |
|  - SHA-3-256 Tamper Verification  | |  - Immutable Security & Access Audit Trails |
+-----------------------------------+ +---------------------------------------------+
                      ^                                     ^
                      |                                     |
+---------------------+-------------------------------------+-----------------------+
|                    Intelligent Threat Detection & Analytics Tier                  |
|  - Telemetry Ingestion Engine (Network, API, Connection, Behavioral Features)      |
|  - Deep Learning Engine: Temporal Convolutional Network (TCN)                     |
|  - Transformer Encoder with Multi-Head Self-Attention                             |
|  - Global Temporal Pooling & Classification Head                                  |
|  - Explainability Engine: SHAP (SHapley Additive exPlanations)                     |
|  - Automated Policy Enforcement & Security Alerting                               |
+-----------------------------------------------------------------------------------+
```

---

## 2. Component Specifications

### 2.1 FHIR Gateway & Validation Layer
- **Standards Compliance**: HL7 FHIR Release 4 (R4).
- **Core Resources Supported**: `Patient`, `Encounter`, `Condition`, `Observation`, `MedicationRequest`, `Medication`, `Procedure`, `DiagnosticReport`, `AllergyIntolerance`, `CarePlan`, `Immunization`, `Practitioner`, `Organization`, and `Bundle`.
- **Validation Pipeline**:
  1. Structural and type validation via Pydantic FHIR R4 schema models.
  2. Mandatory field and code-system checks (LOINC, SNOMED-CT, RxNorm, ICD-10).
  3. Referential integrity validation across resource links.
  4. Canonical JSON serialization following RFC 8785 (deterministic key ordering, whitespace elimination) prior to hashing.

### 2.2 Data Minimization Layer
To enforce privacy-by-design (GDPR Art. 5(1)(c), HIPAA Safe Harbor), a configurable data minimization transformer operates between schema validation and cryptographic encapsulation:
- **`FULL`**: Preserves complete clinical, demographic, and administrative metadata (for primary care continuum).
- **`MINIMAL`**: Retains core identifiers, active condition codes, and current medications; strips secondary narratives and administrative history.
- **`RESEARCH`**: Anonymizes synthetic/clinical direct identifiers (removes synthetic names, telecom, addresses; rounds ages to 5-year buckets) while retaining codified observations, conditions, and procedures.
- **`EMERGENCY`**: Selectively isolates allergy alerts, blood type, active critical conditions, and resuscitation directives for rapid triaging.

### 2.3 Crypto-Agility Engine
A polymorphic design pattern allows runtime algorithm negotiation without altering application logic:
```
           +---------------------------------------------+
           |           CryptoEngineInterface             |
           | + encapsulate(recipient_pk) -> (ct, ss)     |
           | + decapsulate(sk, ct) -> ss                 |
           | + encrypt(plaintext, ss) -> (ct, nonce, tag)|
           | + decrypt(ct, nonce, tag, ss) -> plaintext  |
           | + sign(sk, data) -> signature               |
           | + verify(pk, data, signature) -> bool       |
           | + digest(data) -> hash                      |
           +----------------------+----------------------+
                                  |
         +------------------------+------------------------+
         |                        |                        |
+--------v-------+       +--------v-------+       +--------v-------+
| ClassicalSuite |       |  HybridSuite   |       |    PQCSuite    |
| - ECDH P-256   |       | - ECDH + ML-KEM|       | - ML-KEM-768   |
| - AES-256-GCM  |       | - AES-256-GCM  |       | - ML-DSA-65    |
| - ECDSA P-256  |       | - ML-DSA-65    |       | - AES-256-GCM  |
| - SHA-256      |       | - SHA-3-256    |       | - SHA-3-256    |
+----------------+       +----------------+       +----------------+
```
- **Encrypted Package Structure**:
  - `package_id`: Unique identifier (UUIDv4).
  - `algorithm_id`: e.g., `"PQC-MLKEM768-AES256GCM-MLDSA65-SHA3"`.
  - `key_id`: Recipient public key fingerprint.
  - `kem_ciphertext`: Encapsulated symmetric key bytes (Base64).
  - `aes_ciphertext`: Ciphertext of canonicalized FHIR payload (Base64).
  - `aes_nonce`: 12-byte initialization vector (Base64).
  - `aes_tag`: 16-byte authentication tag (Base64).
  - `sha3_digest`: SHA-3-256 hash of original unencrypted canonical FHIR payload.
  - `signature`: ML-DSA signature over `{package_id, algorithm_id, sha3_digest, aes_tag}`.
  - `timestamp`: UTC ISO-8601 creation timestamp.

### 2.4 Off-Chain Storage Engine
- **Decoupling Strategy**: High-volume, high-cardinality EHR payloads are stored in an encrypted off-chain object store (local persistent filesystem or S3/MinIO compatible store).
- **Addressing**: Objects are located via Content-Addressable Locators: `storage_uri = f"s3://ehr-vault/{patient_id}/{sha3_digest}.enc"`.
- **Integrity Guarantee**: On retrieval, the decrypted payload is re-hashed using SHA-3-256 and compared to the blockchain-anchored digest. Any bit-flip triggers an immediate cryptographic integrity alarm.

### 2.5 Permissioned Blockchain Tier (Smart Contract / Chaincode)
- **Role**: Decentralized consensus, identity verification, dynamic patient consent management, access authorization, and tamper-proof audit trails.
- **Participating Organizations**:
  1. `Hospital_A` (Primary Care Network)
  2. `Hospital_B` (Specialty & Tertiary Care)
  3. `Laboratory` (Diagnostic & Pathology Services)
  4. `Research_Organization` (Academic Health Center)
  5. `Patient` (Data Owner & Consent Controller)
- **Core Chaincode Functions**:
  - `create_consent(patient_id, authorized_org, role, allowed_resources, expiration)`
  - `modify_consent(consent_id, update_params)`
  - `revoke_consent(consent_id)`
  - `request_access(requestor_id, org, role, patient_id, resource_type, purpose)`
  - `log_audit_event(event_type, actor, patient_id, action, result, tx_hash)`
  - `verify_access(requestor_id, patient_id, resource_type, purpose) -> bool`

### 2.6 Intelligent Threat Detection Engine
- **Telemetry Ingestion**: Monitors network telemetry, API invocation frequencies, authorization failures, payload sizes, and header anomalies.
- **Architecture**:
  1. **Input Normalization**: Train-fit standard scaler with non-leaking preprocessors.
  2. **Temporal Convolutional Network (TCN)**: Dilated causal 1D convolutions with residual connections capturing short-to-medium range sequential access patterns.
  3. **Transformer Encoder**: Multi-head self-attention mechanisms capturing long-range contextual relationships across telemetry sequences.
  4. **Multi-Head Attention Layer**: Queries, keys, and values projecting hidden states to emphasize suspicious anomalous bursts.
  5. **Global Temporal Pooling & Dense Classification Head**: Outputs multi-class attack classification and binary anomaly confidence.
- **Explainability**: SHAP Tree/Deep Explainer generates local attribution values for security analysts.

---

## 3. End-to-End Cross-Institutional Data Exchange Flow
```
Hospital A (Sender)               FHIR Gateway / Crypto            Blockchain Ledger           Hospital B (Receiver)
       |                                   |                              |                              |
       | 1. POST /ehr (FHIR Resource)      |                              |                              |
       +---------------------------------->+                              |                              |
       |                                   | 2. Schema Validate           |                              |
       |                                   | 3. Apply Minimization Policy |                              |
       |                                   | 4. Canonicalize & SHA-3-256  |                              |
       |                                   | 5. ML-KEM Encapsulate Key    |                              |
       |                                   | 6. AES-256-GCM Encrypt       |                              |
       |                                   | 7. ML-DSA Sign Package       |                              |
       |                                   | 8. Off-chain Store Encrypted |                              |
       |                                   |                              |                              |
       |                                   | 9. Anchor Record Reference   |                              |
       |                                   +----------------------------->+                              |
       |                                   |                              | 10. Request Access           |
       |                                   |                              |<-----------------------------+
       |                                   |                              | 11. Evaluate ABAC / Consent  |
       |                                   |                              | 12. Record Audit Log         |
       |                                   | 13. Deliver Encrypted Pkg    |                              |
       |                                   +------------------------------------------------------------>+
       |                                   |                              |                              | 14. ML-KEM Decapsulate
       |                                   |                              |                              | 15. AES-256-GCM Decrypt
       |                                   |                              |                              | 16. Verify ML-DSA Sig
       |                                   |                              |                              | 17. Re-calculate SHA-3
       |                                   |                              |                              | 18. Verified FHIR EHR
```
