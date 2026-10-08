import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""Pipeline Script: Cryptographic (PQC) and Blockchain Layer Benchmarks (Phase 23 & 24)."""

import os
import sys
import json

from src.security.pqc_layer import benchmark_pqc_layer
from src.security.blockchain_adapter import benchmark_blockchain_adapter


def main():
    print("==================================================")
    print("PHASE 23: Benchmarking Post-Quantum Cryptography Layer")
    print("==================================================")
    pqc_metrics = benchmark_pqc_layer(n_iterations=100)
    print("PQC Benchmark Complete:")
    print(f"ML-KEM-768: KeyGen={pqc_metrics['ML_KEM_768']['KeyGen_Latency_ms']:.3f}ms | Encaps={pqc_metrics['ML_KEM_768']['Encaps_Latency_ms']:.3f}ms | Decaps={pqc_metrics['ML_KEM_768']['Decaps_Latency_ms']:.3f}ms")
    print(f"ML-DSA-65:  KeyGen={pqc_metrics['ML_DSA_65']['KeyGen_Latency_ms']:.3f}ms | Sign={pqc_metrics['ML_DSA_65']['Sign_Latency_ms']:.3f}ms | Verify={pqc_metrics['ML_DSA_65']['Verify_Latency_ms']:.3f}ms")
    print(f"AES-256-GCM: Encrypt={pqc_metrics['AES_256_GCM']['Encrypt_Latency_ms']:.3f}ms | Decrypt={pqc_metrics['AES_256_GCM']['Decrypt_Latency_ms']:.3f}ms")

    out_pqc = "results/final/crypto_benchmarks.json"
    os.makedirs(os.path.dirname(out_pqc), exist_ok=True)
    with open(out_pqc, "w") as f:
        json.dump(pqc_metrics, f, indent=2)

    print("\n==================================================")
    print("PHASE 24: Benchmarking Permissioned Blockchain Layer")
    print("==================================================")
    bc_metrics = benchmark_blockchain_adapter(n_tx=500)
    print("Blockchain Benchmark Complete:")
    print(f"Transactions: {bc_metrics['Total_Transactions']} | Throughput: {bc_metrics['Throughput_TPS']} TPS")
    print(f"Commit Latency Mean: {bc_metrics['Commit_Latency_Mean_ms']:.3f}ms | P95: {bc_metrics['Commit_Latency_P95_ms']:.3f}ms")
    print(f"Audit Verification per Block: {bc_metrics['Audit_Verification_Time_per_Block_ms']:.4f}ms")

    out_bc = "results/final/blockchain_benchmarks.json"
    with open(out_bc, "w") as f:
        json.dump(bc_metrics, f, indent=2)

    print(f"\nSaved security benchmark results to {out_pqc} and {out_bc}.")


if __name__ == "__main__":
    main()
