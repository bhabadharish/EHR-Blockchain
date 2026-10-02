# Threat Model: Post-Quantum Crypto-Agile FHIR-Blockchain Architecture

## 1. Threat Modeling Methodology
This threat model employs a hybrid STRIDE (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege) and DREAD (Damage, Reproducibility, Exploitability, Affected Users, Discoverability) framework tailored to distributed post-quantum healthcare record exchange.

---

## 2. Attacker Taxonomy

| Attacker ID | Classification | Capabilities & Resources | Primary Motivation |
|---|---|---|---|
| **A1** | **Network Attacker (Eavesdropper / MITM)** | Controls transmission path; captures encrypted traffic; capable of "Harvest Now, Decrypt Later" (HNDL) attacks using future Cryptanalytically Relevant Quantum Computers (CRQCs); can inject or drop network packets. | Mass interception of protected health information (PHI); quantum cryptanalysis. |
| **A2** | **Unauthorized Healthcare Worker** | Legitimate internal user with valid credentials but attempting privilege escalation or accessing patient records outside therapeutic relationship (curiosity, snooping, extortion). | Data exfiltration; unauthorized disclosure. |
| **A3** | **Malicious Organization** | Rogue peer organization participating in health exchange federation; attempts bulk data scraping, fraudulent consent attestation, or collusion. | Unfair competitive advantage; intellectual property theft; secondary data selling. |
| **A4** | **Compromised API Client** | Subverted third-party health app, patient portal, or hijacked bearer tokens; issues high-velocity automated queries, parameter tampering, and injection payloads. | API exploitation; credential abuse; denial of service. |
| **A5** | **Malicious Blockchain Participant** | Rogue peer node in permissioned network; attempts consensus manipulation, double-spending of consent state, or reading private data off ledger. | Sabotaging consensus; ledger pollution; unearned audit clearance. |
| **A6** | **Storage Attacker** | Compromises off-chain object storage (cloud bucket or on-prem filesystem); attempts direct exfiltration of stored blobs or silent bit-flipping/corruption. | Extortion via ransomware; silent record corruption; data destruction. |
| **A7** | **Replay Attacker** | Captures legitimate encrypted FHIR packages, authorization tokens, or blockchain transactions and resubmits them to cause state desynchronization or unauthorized repeat dispensing. | Fraudulent healthcare billing; unauthorized service consumption. |
| **A8** | **Tampering Attacker** | Active adversary modifying in-transit payloads, altering metadata, modifying digital signatures, or modifying resource identifiers. | Misleading clinical diagnosis; medical malpractice induction; covering illicit activity. |

---

## 3. Post-Quantum Cryptographic Threat Analysis
- **Shor's Algorithm (Polynomial Time)**: Classical asymmetric primitives (RSA-2048/4096, ECDH P-256/384, ECDSA, Ed25519) can be completely broken by a CRQC with a few thousand logical qubits. Classical ciphertext captured today will be decrypted in the future (HNDL).
  - *Defense*: Key Encapsulation using **ML-KEM-768 / ML-KEM-1024** (lattice-based Learning with Errors) and digital signatures using **ML-DSA-65 / ML-DSA-87** (lattice-based Fiat-Shamir with Aborts).
- **Grover's Algorithm (Square Root Speedup)**: Symmetric key search speedup from $O(2^n)$ to $O(2^{n/2})$.
  - *Defense*: Doubling symmetric key lengths. We use **AES-256-GCM** (retaining 128 bits of post-quantum security) and **SHA-3-256** (retaining 128 bits of collision resistance under quantum claw attacks).

---

## 4. Threat Matrix

| Threat | Attacker | Attack Vector | Attack Surface | Defense Mechanism | Detection Strategy | Empirical Evidence / Artifact |
|---|---|---|---|---|---|---|
| **T1: Quantum Eavesdropping (HNDL)** | A1 | Intercepting encrypted FHIR packages over TLS/WAN | Network transport layer | ML-KEM-768/1024 lattice key encapsulation + AES-256-GCM | Anomaly telemetry monitoring for bulk harvesting | Packet capture decryption failure test against CRQC simulator |
| **T2: Unauthorized Record Access** | A2 | Direct API query with unauthorized actor token | FastAPI FHIR endpoints | Strict RBAC + ABAC policy engine; smart-contract consent verification | Access denial logs; real-time authorization anomaly alert | Test suite `test_access_control.py` (denial matrix validation) |
| **T3: Consent Revocation Bypass** | A2, A3 | Attempting to access record after consent is revoked | Blockchain ledger & gateway cache | Dynamic on-chain query to Hyperledger state machine checking revoked status | State query validation; immediate 403 Forbidden | Test suite `test_consent.py` (revocation enforcement test) |
| **T4: In-Transit Data Tampering** | A8 | Flipping bits in ciphertext or modifying JSON payload | Gateway transit / HTTP body | SHA-3-256 integrity hash + AES-256-GCM authentication tag | GCM tag verification failure + SHA-3 digest mismatch | Test suite `test_integrity.py` (deliberate bit-flip tests) |
| **T5: Signature Forgery** | A8 | Crafting illegitimate digital signature on clinical notes | Cryptographic envelope | ML-DSA-65 post-quantum digital signature verification | ML-DSA `InvalidSignatureError` raised | Signature mutation test suite reporting 100% rejection |
| **T6: Replay of Valid Package** | A7 | Resending prior valid encrypted bundle | Ingestion API | Nonce uniqueness tracking, timestamp freshness window (<300s), UUID deduplication | Duplicate nonce / stale timestamp rejection | API replay attack simulation script |
| **T7: Plaintext Storage Exfiltration** | A6 | Breaching off-chain S3/MinIO bucket | Storage disk / bucket | Data is encrypted *before* reaching storage; zero plaintext at rest | Object storage access logs; zero-knowledge audit | Off-chain storage inspection verifying zero plaintext bytes |
| **T8: Ledger Data Leakage** | A5 | Inspecting blockchain blocks and state database | Hyperledger Fabric ledger | Zero PHI on-chain; only SHA-3 hashes and cryptographic locators stored | Blockchain state inspection audit | Ledger block deserializer verifying absence of FHIR attributes |
| **T9: Volumetric API Flooding / DoS** | A4 | High-frequency SYN/HTTP flooding | Gateway ingress ports | Rate-limiting middleware + Intelligent TCN-Transformer threat detector | Neural telemetry anomaly classification (alert generated < 20ms) | Attack simulation benchmark reporting FPR/FNR and detection latency |
| **T10: Malicious Insider Organization Scraping** | A3 | Querying thousands of patient IDs systematically | Federated FHIR exchange API | Behavioral telemetry anomaly detection + ABAC purpose-of-use binding | High-density access clustering detected by deep threat model | Threat detection ROC-AUC / Confusion matrix evaluation |

---

## 5. Security Principles Implemented
1. **Confidentiality**: Protected via hybrid/post-quantum envelope encryption (ML-KEM + AES-256-GCM).
2. **Integrity**: Assured end-to-end via cryptographic hash trees and SHA-3-256 digests.
3. **Authentication & Non-Repudiation**: Enforced by ML-DSA-65 digital signatures bound to certified organizational public keys.
4. **Authorization**: Governed by deterministic smart contract state machines enforcing RBAC and ABAC rules.
5. **Auditability**: Every access request, consent update, revocation, and security event produces an append-only blockchain transaction.
6. **Crypto-Agility**: Modular cipher-suite interfaces permit rapid algorithm migration without software refactoring.
