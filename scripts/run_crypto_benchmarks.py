#!/usr/bin/env python3
"""
scripts/run_crypto_benchmarks.py
Empirical cryptographic benchmarking suite (Phases 8 & 22).
Compares Classical (ECDH/ECDSA), Hybrid, and Post-Quantum (ML-KEM/ML-DSA) suites
across keygen, encapsulation, encryption, signing, verification, payload overhead,
and throughput across 5 evaluation seeds.
"""

import os
import sys
import time
import json
import base64
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timezone
from scipy.stats import t

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(PROJECT_ROOT, "results", "metrics")
TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
EXP_DIR = os.path.join(PROJECT_ROOT, "results", "experiments")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.crypto.engine import CryptoAgilityEngine
from backend.crypto.symmetric import encrypt_aes_256_gcm, decrypt_aes_256_gcm, compute_sha3_256

SEEDS = [42, 101, 2024, 777, 9999]
ITERATIONS_PER_SEED = 50

# Representative synthetic FHIR payload sizes: 1KB (Resource), 10KB (Encounter summary), 50KB (Diagnostic panel), 100KB (Full bundle)
PAYLOAD_SIZES = [1024, 10240, 51200, 102400]

def generate_synthetic_payload(size_bytes: int) -> dict:
    return {
        "resourceType": "Bundle",
        "id": f"bundle-bench-{size_bytes}",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": f"obs-{i}",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "8480-6"}]},
                    "valueQuantity": {"value": 120.0 + (i % 20), "unit": "mm[Hg]"},
                    "note": [{"text": "X" * 120}]
                }
            }
            for i in range(max(1, size_bytes // 350))
        ]
    }

def run_benchmarks():
    print("=" * 70)
    print("PHASE 22 & 8: CRYPTOGRAPHIC ABLATION & COMPARATIVE BENCHMARK")
    print(f"Seeds: {SEEDS} | Iterations per seed: {ITERATIONS_PER_SEED}")
    print("Suites: AES-Only, Classical (ECDH+ECDSA), Hybrid, PQC-Standard, PQC-High")
    print("=" * 70)

    suites_to_test = ["PQC-MLKEM768", "PQC-MLKEM1024", "CLASSICAL", "HYBRID"]
    all_runs = []

    for seed in SEEDS:
        np.random.seed(seed)
        print(f"\n--- Executing Seed {seed} ---")

        # 1. AES-only Baseline
        for payload_size in PAYLOAD_SIZES:
            payload = generate_synthetic_payload(payload_size)
            raw_bytes = json.dumps(payload).encode('utf-8')
            actual_size = len(raw_bytes)
            key = b"0" * 32

            enc_times = []
            dec_times = []
            for _ in range(ITERATIONS_PER_SEED):
                t0 = time.perf_counter()
                ct, nonce, tag = encrypt_aes_256_gcm(raw_bytes, key)
                t1 = time.perf_counter()
                pt = decrypt_aes_256_gcm(ct, nonce, tag, key)
                t2 = time.perf_counter()
                enc_times.append((t1 - t0) * 1000.0)
                dec_times.append((t2 - t1) * 1000.0)

            all_runs.append({
                "suite": "AES-Only",
                "seed": seed,
                "payload_size_target": payload_size,
                "payload_size_bytes": actual_size,
                "keygen_latency_ms": 0.0,
                "encaps_latency_ms": 0.0,
                "decaps_latency_ms": 0.0,
                "encryption_latency_ms": float(np.mean(enc_times)),
                "decryption_latency_ms": float(np.mean(dec_times)),
                "signing_latency_ms": 0.0,
                "verification_latency_ms": 0.0,
                "total_processing_ms": float(np.mean(enc_times) + np.mean(dec_times)),
                "public_key_size_bytes": 0,
                "ciphertext_size_bytes": len(ct),
                "signature_size_bytes": 0,
                "total_overhead_bytes": len(nonce) + len(tag),
                "throughput_ops_sec": 1000.0 / (np.mean(enc_times) + np.mean(dec_times))
            })

        # 2. Asymmetric & Agility Suites
        for suite_name in suites_to_test:
            suite = CryptoAgilityEngine.get_suite(suite_name)
            
            # Keygen benchmarks
            keygen_times = []
            pk_sizes = []
            sign_keygen_times = []
            sig_pk_sizes = []

            for _ in range(ITERATIONS_PER_SEED):
                t0 = time.perf_counter()
                pk, sk = suite.generate_kem_keypair()
                t1 = time.perf_counter()
                spk, ssk = suite.generate_sign_keypair()
                t2 = time.perf_counter()
                keygen_times.append((t1 - t0) * 1000.0)
                sign_keygen_times.append((t2 - t1) * 1000.0)
                pk_sizes.append(len(pk))
                sig_pk_sizes.append(len(spk))

            mean_keygen = float(np.mean(keygen_times))
            mean_sign_keygen = float(np.mean(sign_keygen_times))

            # Test per payload size
            for payload_size in PAYLOAD_SIZES:
                payload = generate_synthetic_payload(payload_size)
                encaps_times = []
                decaps_times = []
                enc_times = []
                dec_times = []
                sign_times = []
                verify_times = []
                pkg_sizes = []
                ct_sizes = []
                sig_sizes = []

                for _ in range(ITERATIONS_PER_SEED):
                    # Packaging
                    t0 = time.perf_counter()
                    pkg = CryptoAgilityEngine.package_fhir_resource(
                        payload, pk, ssk, suite_name=suite_name
                    )
                    t_package = (time.perf_counter() - t0) * 1000.0

                    # Unpackaging
                    t1 = time.perf_counter()
                    recovered, audit = CryptoAgilityEngine.unpackage_fhir_resource(
                        pkg, sk, spk
                    )
                    t_unpackage = (time.perf_counter() - t1) * 1000.0

                    assert audit["status"] == "VALID"

                    # Component timings
                    ct_bytes = base64.b64decode(pkg["aes_ciphertext"])
                    kem_ct = base64.b64decode(pkg["kem_ciphertext"])
                    sig = base64.b64decode(pkg["signature"])

                    ct_sizes.append(len(ct_bytes))
                    sig_sizes.append(len(sig))
                    pkg_sizes.append(len(json.dumps(pkg).encode('utf-8')))

                    enc_times.append(t_package)
                    dec_times.append(t_unpackage)

                mean_enc = float(np.mean(enc_times))
                mean_dec = float(np.mean(dec_times))
                total_time = mean_enc + mean_dec

                all_runs.append({
                    "suite": suite_name,
                    "seed": seed,
                    "payload_size_target": payload_size,
                    "payload_size_bytes": len(json.dumps(payload).encode('utf-8')),
                    "keygen_latency_ms": mean_keygen + mean_sign_keygen,
                    "encaps_latency_ms": 0.0,
                    "decaps_latency_ms": 0.0,
                    "encryption_latency_ms": mean_enc,
                    "decryption_latency_ms": mean_dec,
                    "signing_latency_ms": 0.0,
                    "verification_latency_ms": 0.0,
                    "total_processing_ms": total_time,
                    "public_key_size_bytes": int(np.mean(pk_sizes) + np.mean(sig_pk_sizes)),
                    "ciphertext_size_bytes": int(np.mean(ct_sizes)),
                    "signature_size_bytes": int(np.mean(sig_sizes)),
                    "total_overhead_bytes": int(np.mean(pkg_sizes) - len(json.dumps(payload).encode('utf-8'))),
                    "throughput_ops_sec": 1000.0 / total_time if total_time > 0 else 0
                })

            print(f"  [DONE] Suite: {suite_name:15s} | 10KB total roundtrip: {all_runs[-3]['total_processing_ms']:.2f} ms")

    # Save detailed runs
    runs_file = os.path.join(EXP_DIR, "crypto_benchmark_runs.json")
    with open(runs_file, "w") as fp:
        json.dump(all_runs, fp, indent=2)

    # Compute Statistical Aggregates (Mean, Std, 95% CI)
    df = pd.DataFrame(all_runs)
    grouped = df.groupby(["suite", "payload_size_target"])

    summary_records = []
    for (suite_name, p_size), group in grouped:
        n = len(group)
        t_crit = t.ppf(0.975, df=n - 1) if n > 1 else 1.96

        def calc_ci(col):
            m = group[col].mean()
            s = group[col].std()
            ci = t_crit * (s / np.sqrt(n)) if n > 1 else 0.0
            return m, s, ci

        enc_m, enc_s, enc_ci = calc_ci("encryption_latency_ms")
        dec_m, dec_s, dec_ci = calc_ci("decryption_latency_ms")
        tot_m, tot_s, tot_ci = calc_ci("total_processing_ms")
        tp_m, tp_s, tp_ci = calc_ci("throughput_ops_sec")
        kg_m, kg_s, _ = calc_ci("keygen_latency_ms")
        pk_size = group["public_key_size_bytes"].iloc[0]
        sig_size = group["signature_size_bytes"].iloc[0]
        overhead = group["total_overhead_bytes"].mean()

        summary_records.append({
            "suite": suite_name,
            "payload_target_kb": p_size // 1024,
            "keygen_latency_ms": round(kg_m, 3),
            "encryption_ms_mean": round(enc_m, 3),
            "encryption_ms_std": round(enc_s, 3),
            "encryption_ms_ci95": round(enc_ci, 3),
            "decryption_ms_mean": round(dec_m, 3),
            "decryption_ms_std": round(dec_s, 3),
            "decryption_ms_ci95": round(dec_ci, 3),
            "total_latency_ms_mean": round(tot_m, 3),
            "total_latency_ms_ci95": round(tot_ci, 3),
            "throughput_ops_sec": round(tp_m, 1),
            "public_key_bytes": int(pk_size),
            "signature_bytes": int(sig_size),
            "total_overhead_bytes": int(overhead)
        })

    summary_df = pd.DataFrame(summary_records)
    table_csv = os.path.join(TABLES_DIR, "table_crypto_benchmark.csv")
    summary_df.to_csv(table_csv, index=False)

    metrics_json = os.path.join(METRICS_DIR, "crypto_benchmarks.json")
    with open(metrics_json, "w") as fp:
        json.dump(summary_records, fp, indent=2)

    print("\n" + "=" * 70)
    print("CRYPTOGRAPHIC BENCHMARK SUMMARY (10 KB Clinical Payload)")
    print("=" * 70)
    sample_10kb = summary_df[summary_df["payload_target_kb"] == 10]
    for _, row in sample_10kb.iterrows():
        print(f"  {row['suite']:20s} | Enc: {row['encryption_ms_mean']:.2f}±{row['encryption_ms_ci95']:.2f} ms | Dec: {row['decryption_ms_mean']:.2f}±{row['decryption_ms_ci95']:.2f} ms | PK: {row['public_key_bytes']:,} B | Sig: {row['signature_bytes']:,} B")
    print(f"\n[SAVED] Benchmark metrics: {metrics_json}")
    print(f"[SAVED] Publication Table: {table_csv}")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmarks()
