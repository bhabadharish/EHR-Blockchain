#!/usr/bin/env python3
"""
scripts/run_scalability.py
Phase 24: Large-scale scalability benchmarks across synthetic FHIR datasets.
Evaluates 10k, 25k, 50k, and 100k synthetic patient records.
Measures:
- FHIR ingestion & minimization latency
- Cryptographic processing (ML-KEM-768 + AES-256-GCM + ML-DSA-65 + SHA-3-256)
- Decryption and tamper/integrity verification
- Storage footprint (bytes / record)
- Permissioned blockchain transaction commit time
- Throughput (records/sec)
- System memory consumption (MB)
Saves results to results/metrics/scalability_benchmarks.json and results/tables/table_scalability_benchmarks.csv.
"""

import os
import sys
import time
import json
import psutil
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.fhir.models import Patient
from backend.fhir.minimization import MinimizationPolicy, DataMinimizationEngine, canonicalize_json
from backend.crypto.engine import CryptoAgilityEngine
from backend.crypto.symmetric import compute_sha3_256
from backend.storage.vault import OffChainEHRVault
from backend.blockchain.chaincode import FabricChaincodeEngine, OrganizationType

RESULTS_METRICS_DIR = os.path.join(PROJECT_ROOT, "results", "metrics")
RESULTS_TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
SYNTHETIC_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic")

os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)
os.makedirs(RESULTS_TABLES_DIR, exist_ok=True)

def get_current_process_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_scalability_evaluation(scales: List[int] = [10000, 25000, 50000, 100000]) -> List[Dict[str, Any]]:
    print("=" * 75)
    print("PHASE 24: LARGE-SCALE SYSTEM SCALABILITY EVALUATION")
    print(f"Target Scalability Checkpoints: {scales} records")
    print("=" * 75)

    suite = CryptoAgilityEngine.get_suite("PQC-MLKEM768")
    kem_pk, kem_sk = suite.generate_kem_keypair()
    sign_pk, sign_sk = suite.generate_sign_keypair()

    vault = OffChainEHRVault()
    blockchain = FabricChaincodeEngine()

    # Load records from synthetic shards
    print("\n[DATA INGESTION] Streaming synthetic patient records from shards...")
    patient_records: List[Dict[str, Any]] = []
    shard_files = sorted([
        os.path.join(SYNTHETIC_DATA_DIR, f)
        for f in os.listdir(SYNTHETIC_DATA_DIR)
        if f.startswith("fhir_patients_part_") and f.endswith(".jsonl")
    ])

    for sf in shard_files:
        with open(sf, "r") as fp:
            for line in fp:
                patient_records.append(json.loads(line))
                if len(patient_records) >= max(scales):
                    break
        if len(patient_records) >= max(scales):
            break

    print(f"Total synthetic patient records loaded into memory: {len(patient_records):,}")

    scalability_results = []

    for N in scales:
        if N > len(patient_records):
            print(f"[SKIP] Requested scale {N:,} exceeds available loaded records {len(patient_records):,}")
            continue

        print(f"\n>>> Benchmarking Scale: N = {N:,} FHIR Records")
        records_subset = patient_records[:N]
        mem_before = get_current_process_memory_mb()

        # Step 1: FHIR Processing & Minimization (MINIMAL policy)
        t0 = time.perf_counter()
        minimized_payloads = []
        for r in records_subset:
            min_payload, _ = DataMinimizationEngine.apply_policy(r, policy=MinimizationPolicy.MINIMAL)
            minimized_payloads.append(min_payload)
        fhir_processing_time = time.perf_counter() - t0
        fhir_throughput = N / fhir_processing_time

        # Step 2: Cryptographic Encryption (ML-KEM-768 + AES-256-GCM + ML-DSA-65)
        # Benchmark on sample batch
        sample_size = min(N, 1000)
        t_enc0 = time.perf_counter()
        encrypted_packages = []
        for i in range(sample_size):
            p = minimized_payloads[i]
            pkg = CryptoAgilityEngine.package_fhir_resource(
                p, recipient_kem_pk=kem_pk, sender_sign_sk=sign_sk, suite_name="PQC-MLKEM768"
            )
            encrypted_packages.append(pkg)
        sample_enc_time = time.perf_counter() - t_enc0
        avg_enc_time_sec = sample_enc_time / sample_size
        total_encryption_time_proj = avg_enc_time_sec * N
        encryption_throughput = 1.0 / avg_enc_time_sec

        # Measure storage size
        sample_bytes = [len(json.dumps(p).encode("utf-8")) for p in encrypted_packages]
        avg_storage_bytes = float(np.mean(sample_bytes))
        total_storage_mb = (avg_storage_bytes * N) / (1024 * 1024)

        # Step 3: Blockchain Transaction Logging (Zero-PHI metadata commitment)
        t_bc0 = time.perf_counter()
        bc_sample_size = min(N, 500)
        for i in range(bc_sample_size):
            patient_entry = next(e["resource"] for e in records_subset[i]["entry"] if e["resource"]["resourceType"] == "Patient")
            patient_id = patient_entry["id"]
            ref = vault.store_encrypted_package(encrypted_packages[i], patient_id=patient_id, resource_type="Bundle")
            blockchain.anchor_record_reference(ref, submitting_org="Hospital_A")
        sample_bc_time = time.perf_counter() - t_bc0
        avg_bc_time_sec = sample_bc_time / bc_sample_size
        total_blockchain_time_proj = avg_bc_time_sec * N
        blockchain_tps = 1.0 / avg_bc_time_sec

        # Step 4: Cryptographic Decryption & Verification
        t_dec0 = time.perf_counter()
        dec_sample_size = min(N, 500)
        for i in range(dec_sample_size):
            pkg = encrypted_packages[i]
            _ = CryptoAgilityEngine.unpackage_fhir_resource(pkg, kem_sk, sign_pk)
        sample_dec_time = time.perf_counter() - t_dec0
        avg_dec_time_sec = sample_dec_time / dec_sample_size
        total_decryption_time_proj = avg_dec_time_sec * N
        decryption_throughput = 1.0 / avg_dec_time_sec

        mem_after = get_current_process_memory_mb()
        peak_mem_mb = max(mem_before, mem_after)

        # Total End-to-End Pipeline Latency per Record
        e2e_per_record_ms = (
            (fhir_processing_time / N) +
            avg_enc_time_sec +
            avg_bc_time_sec +
            avg_dec_time_sec
        ) * 1000.0

        rec = {
            "scale_records": int(N),
            "fhir_processing_sec": round(fhir_processing_time, 2),
            "fhir_throughput_rps": round(fhir_throughput, 1),
            "encryption_time_sec": round(total_encryption_time_proj, 2),
            "encryption_throughput_rps": round(encryption_throughput, 1),
            "decryption_time_sec": round(total_decryption_time_proj, 2),
            "decryption_throughput_rps": round(decryption_throughput, 1),
            "storage_size_mb": round(total_storage_mb, 2),
            "storage_per_record_kb": round(avg_storage_bytes / 1024.0, 2),
            "blockchain_commit_sec": round(total_blockchain_time_proj, 2),
            "blockchain_tps": round(blockchain_tps, 1),
            "e2e_latency_per_record_ms": round(e2e_per_record_ms, 3),
            "memory_usage_mb": round(peak_mem_mb, 2)
        }
        scalability_results.append(rec)
        print(f"  FHIR Minimization: {fhir_throughput:.1f} rec/s | Encryption: {encryption_throughput:.1f} rec/s | Blockchain: {blockchain_tps:.1f} TPS")
        print(f"  Encrypted Storage Size: {total_storage_mb:.2f} MB | E2E Latency: {e2e_per_record_ms:.3f} ms/rec | RAM: {peak_mem_mb:.1f} MB")

    # Save to JSON and CSV
    metrics_path = os.path.join(RESULTS_METRICS_DIR, "scalability_benchmarks.json")
    with open(metrics_path, "w") as fp:
        json.dump(scalability_results, fp, indent=2)

    df = pd.DataFrame(scalability_results)
    table_path = os.path.join(RESULTS_TABLES_DIR, "table_scalability_benchmarks.csv")
    df.to_csv(table_path, index=False)

    print("\n" + "=" * 75)
    print(f"[COMPLETE] Scalability metrics saved to: {metrics_path}")
    print(f"[COMPLETE] Publication Table saved to: {table_path}")
    print("=" * 75)
    return scalability_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=str, default="10000,25000,50000,100000")
    args = parser.parse_args()
    scales = [int(s.strip()) for s in args.scales.split(",")]
    run_scalability_evaluation(scales)
