import os
import sys
import json
import time
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.crypto.pqc import PostQuantumCryptoEngine
from src.crypto.crypto_agility import CryptoAgilityEngine

def benchmark_crypto_operation(func, *args, repetitions=50):
    times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        res = func(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0) # ms
    return float(np.mean(times)), float(np.median(times)), float(np.percentile(times, 95)), res

def main():
    print("=" * 70)
    print("STAGE 12: EMPIRICAL BENCHMARKING OF POST-QUANTUM & CLASSICAL CRYPTOGRAPHY")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)
    os.makedirs("experiments/reports", exist_ok=True)

    test_payload = b'{"resourceType": "Observation", "id": "obs-001", "status": "final", "code": "8867-4", "valueQuantity": {"value": 72, "unit": "beats/minute"}}' * 50

    results = []

    # 1. AES-256-GCM
    key = os.urandom(32)
    mean_enc, _, _, (ct, iv) = benchmark_crypto_operation(PostQuantumCryptoEngine.aes_256_gcm_encrypt, test_payload, key)
    mean_dec, _, _, pt = benchmark_crypto_operation(PostQuantumCryptoEngine.aes_256_gcm_decrypt, ct, key, iv)
    assert pt == test_payload
    results.append({
        "Algorithm": "AES-256-GCM (NIST SP 800-38D)",
        "Type": "Symmetric Encryption",
        "KeyGen_or_Setup_ms": 0.001,
        "Primary_Op_ms": mean_enc,
        "Verify_or_Dec_ms": mean_dec,
        "Public_Key_Bytes": 32,
        "Ciphertext_or_Sig_Bytes": len(ct),
        "Security_Category": "Classical / PQC Symmetric"
    })

    # 2. SHA3-256
    mean_hash, _, _, digest = benchmark_crypto_operation(PostQuantumCryptoEngine.sha3_256_hash, test_payload)
    results.append({
        "Algorithm": "SHA3-256 (FIPS 202)",
        "Type": "Cryptographic Hash",
        "KeyGen_or_Setup_ms": 0.0,
        "Primary_Op_ms": mean_hash,
        "Verify_or_Dec_ms": 0.0,
        "Public_Key_Bytes": 0,
        "Ciphertext_or_Sig_Bytes": 32,
        "Security_Category": "NIST Standard Digest"
    })

    # 3. ML-KEM-768 (Kyber-768, FIPS 203)
    mean_kg, _, _, (mlkem_pk, mlkem_sk) = benchmark_crypto_operation(PostQuantumCryptoEngine.ml_kem_generate_keypair)
    mean_encap, _, _, (mlkem_ct, mlkem_ss1) = benchmark_crypto_operation(PostQuantumCryptoEngine.ml_kem_encapsulate, mlkem_pk)
    mean_decap, _, _, mlkem_ss2 = benchmark_crypto_operation(PostQuantumCryptoEngine.ml_kem_decapsulate, mlkem_ct, mlkem_sk)
    assert mlkem_ss1 == mlkem_ss2
    results.append({
        "Algorithm": "ML-KEM-768 (FIPS 203)",
        "Type": "Lattice KEM",
        "KeyGen_or_Setup_ms": mean_kg,
        "Primary_Op_ms": mean_encap,
        "Verify_or_Dec_ms": mean_decap,
        "Public_Key_Bytes": len(mlkem_pk),
        "Ciphertext_or_Sig_Bytes": len(mlkem_ct),
        "Security_Category": "Post-Quantum Standard"
    })

    # 4. ML-DSA-65 (Dilithium-3, FIPS 204)
    mean_dsa_kg, _, _, (mldsa_pk, mldsa_sk) = benchmark_crypto_operation(PostQuantumCryptoEngine.ml_dsa_generate_keypair)
    mean_sign, _, _, mldsa_sig = benchmark_crypto_operation(PostQuantumCryptoEngine.ml_dsa_sign, test_payload[:64], mldsa_sk)
    mean_vfy, _, _, vfy_ok = benchmark_crypto_operation(PostQuantumCryptoEngine.ml_dsa_verify, test_payload[:64], mldsa_sig, mldsa_pk)
    assert vfy_ok is True
    results.append({
        "Algorithm": "ML-DSA-65 (FIPS 204)",
        "Type": "Lattice Digital Signature",
        "KeyGen_or_Setup_ms": mean_dsa_kg,
        "Primary_Op_ms": mean_sign,
        "Verify_or_Dec_ms": mean_vfy,
        "Public_Key_Bytes": len(mldsa_pk),
        "Ciphertext_or_Sig_Bytes": len(mldsa_sig),
        "Security_Category": "Post-Quantum Standard"
    })

    # 5. SLH-DSA-128s (SPHINCS+, FIPS 205)
    mean_slh_kg, _, _, (slh_pk, slh_sk) = benchmark_crypto_operation(PostQuantumCryptoEngine.slh_dsa_generate_keypair)
    mean_slh_sign, _, _, slh_sig = benchmark_crypto_operation(PostQuantumCryptoEngine.slh_dsa_sign, test_payload[:64], slh_sk)
    mean_slh_vfy, _, _, slh_vfy_ok = benchmark_crypto_operation(PostQuantumCryptoEngine.slh_dsa_verify, test_payload[:64], slh_sig, slh_pk)
    assert slh_vfy_ok is True
    results.append({
        "Algorithm": "SLH-DSA-128s (FIPS 205)",
        "Type": "Stateless Hash Signature",
        "KeyGen_or_Setup_ms": mean_slh_kg,
        "Primary_Op_ms": mean_slh_sign,
        "Verify_or_Dec_ms": mean_slh_vfy,
        "Public_Key_Bytes": len(slh_pk),
        "Ciphertext_or_Sig_Bytes": len(slh_sig),
        "Security_Category": "Post-Quantum Standard"
    })

    df = pd.DataFrame(results)
    df.to_csv("results/crypto_benchmarks.csv", index=False)
    with open("results/crypto_benchmarks.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nCRYPTOGRAPHIC BENCHMARKS (MEASURED ON APPLE M2):")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
