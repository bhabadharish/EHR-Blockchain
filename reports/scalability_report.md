# Phase 24: Large-Scale System Scalability and Storage Growth Report

## 1. Experimental Protocol & Target Checkpoints

To evaluate whether the proposed crypto-agile FHIR-blockchain system scales effectively to regional and national healthcare network demands, empirical benchmarks were conducted across four synthetic cohort scales:
- **10,000 patient records**
- **25,000 patient records**
- **50,000 patient records**
- **100,000 patient records**

For each scale, the complete pipeline was benchmarked:
1. **FHIR Ingestion & Minimization**: Ingesting and minimizing patient bundles under the `MINIMAL` policy.
2. **Post-Quantum Cryptographic Packaging**: AES-256-GCM encryption, ML-KEM-768 lattice encapsulation, ML-DSA-65 digital signature, and SHA-3-256 hashing.
3. **Storage Footprint**: Total serialized off-chain vault storage size.
4. **Blockchain State Commitment**: Anchoring zero-PHI cryptographic references into permissioned Fabric blocks.
5. **Decryption & Verification**: Unwrapping envelopes, verifying ML-DSA signatures, decapsulating shared secrets, decrypting AES-GCM, and verifying SHA-3-256 integrity digests.
6. **System Resource Utilization**: Tracking peak resident set size (RSS) memory consumption in MB.

---

## 2. Scalability Evaluation Results Matrix

| Dataset Scale (Records) | FHIR Minimization Throughput | PQC Encryption Throughput | Blockchain Anchor Throughput | PQC Decryption Throughput | Total Storage Footprint | Storage / Record | End-to-End Latency / Record | Peak System RAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **10,000** | 2,549.4 rec/s | 179.5 rec/s | 4,780.3 TPS | 1,842.1 rec/s | **176.22 MB** | 17.62 KB | **6.409 ms** | 5,457.6 MB |
| **25,000** | 2,862.4 rec/s | 183.1 rec/s | 7,051.8 TPS | 1,912.4 rec/s | **440.55 MB** | 17.62 KB | **6.183 ms** | 5,940.3 MB |
| **50,000** | 3,105.8 rec/s | 174.8 rec/s | 6,303.6 TPS | 1,795.6 rec/s | **881.10 MB** | 17.62 KB | **6.449 ms** | 6,743.2 MB |
| **100,000** | 2,780.0 rec/s | 179.9 rec/s | 5,588.9 TPS | 1,820.5 rec/s | **1,762.21 MB** | 17.62 KB | **6.330 ms** | 7,868.8 MB |

---

## 3. Analysis & Scaling Behavior

1. **Throughput Invariance**:
   - FHIR minimization maintains a consistent throughput of **~2,800 records/second**.
   - Post-quantum cryptographic processing achieves **~180 packaging operations/second** per core on Apple Silicon M-series hardware.
   - Decryption and tamper verification reaches **~1,820 operations/second**.
2. **Linear Storage Growth ($O(N)$)**:
   - Off-chain storage grows strictly linearly at **17.62 KB per encrypted patient bundle**.
   - 100,000 full synthetic patient records require only **1.76 GB** of disk storage.
3. **Sub-10ms End-to-End Latency**:
   - Total latency per record remains bounded between **6.18 ms and 6.45 ms** regardless of dataset size, demonstrating that the architecture suffers zero latency degradation as data volume scales by an order of magnitude.
