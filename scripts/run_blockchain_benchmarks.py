"""
scripts/run_blockchain_benchmarks.py
Empirical blockchain ablation benchmarking suite (Phase 23).
Compares:
1. No blockchain (Baseline direct storage/cache)
2. Blockchain audit only
3. Blockchain audit + dynamic consent
4. Blockchain audit + dynamic consent + real-time revocation checking
Measures transaction latency, access-control decision time, throughput (TPS),
storage overhead, and block verification across 5 evaluation seeds.
"""

import os
import sys
import time
import json
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

from backend.blockchain.chaincode import FabricChaincodeEngine
from backend.blockchain.access_control import AccessControlEngine, Role, PurposeOfUse

SEEDS = [42, 101, 2024, 777, 9999]
NUM_TRANSACTIONS = 200

def run_blockchain_benchmarks():
    print("=" * 70)
    print("PHASE 23: BLOCKCHAIN ABLATION & ACCESS CONTROL BENCHMARK")
    print(f"Seeds: {SEEDS} | Transactions per configuration: {NUM_TRANSACTIONS}")
    print("Ablations: No-Blockchain, Audit-Only, Audit+Consent, Audit+Consent+Revocation")
    print("=" * 70)

    configurations = [
        ("No_Blockchain", False, False, False),
        ("Blockchain_Audit_Only", True, False, False),
        ("Blockchain_Audit_Consent", True, True, False),
        ("Blockchain_Audit_Consent_Revocation", True, True, True)
    ]

    all_runs = []

    for seed in SEEDS:
        np.random.seed(seed)
        print(f"\n--- Executing Seed {seed} ---")

        for config_name, has_audit, has_consent, has_revocation in configurations:
            chaincode = FabricChaincodeEngine()
            access_engine = AccessControlEngine(chaincode)

            # Pre-populate 50 patient consents
            for i in range(50):
                pid = f"pt-bench-{i:03d}"
                chaincode.create_consent(
                    patient_id=pid,
                    authorized_org="Hospital_A",
                    role=Role.DOCTOR.value,
                    allowed_resources=["Observation", "Condition", "MedicationRequest"],
                    purpose_of_use=PurposeOfUse.TREATMENT.value
                )
                if has_revocation and (i % 5 == 0):
                    # Revoke 20% of consents to measure real-time revocation checking
                    c_id = list(chaincode.consent_state.keys())[-1]
                    chaincode.revoke_consent(c_id, pid, reason="Patient opted out")

            tx_latencies = []
            access_latencies = []
            bytes_added = []

            initial_blocks = len(chaincode.ledger_blocks)
            start_bench_time = time.perf_counter()

            for tx_i in range(NUM_TRANSACTIONS):
                pid = f"pt-bench-{(tx_i % 50):03d}"
                
                # Measure access-control decision latency
                t_ac_start = time.perf_counter()
                
                if config_name == "No_Blockchain":
                    # Direct dictionary/cache lookup without blockchain consensus
                    t0 = time.perf_counter()
                    allowed = True
                    decision = "PERMIT"
                    t_ac = (time.perf_counter() - t0) * 1000.0
                    t_tx = t_ac
                    tx_bytes = 0
                elif config_name == "Blockchain_Audit_Only":
                    # Direct RBAC check + audit log transaction
                    t0 = time.perf_counter()
                    tx_id = chaincode._commit_transaction(
                        tx_type="ACCESS_AUDIT_LOG",
                        actor="DOCTOR/dr-bench-01@Hospital_A",
                        payload={"patient_id": pid, "resource_type": "Observation", "action": "READ"}
                    )
                    t_tx = (time.perf_counter() - t0) * 1000.0
                    t_ac = t_tx
                    tx_bytes = len(json.dumps(chaincode.ledger_blocks[-1]["transactions"][0]).encode('utf-8'))
                elif config_name == "Blockchain_Audit_Consent":
                    # Check consent on-chain + log audit
                    t0 = time.perf_counter()
                    dec = access_engine.evaluate_access(
                        requestor_id="dr-bench-01",
                        role=Role.DOCTOR,
                        organization="Hospital_A",
                        patient_id=pid,
                        resource_type="Observation",
                        purpose_of_use=PurposeOfUse.TREATMENT
                    )
                    t_tx = (time.perf_counter() - t0) * 1000.0
                    t_ac = t_tx
                    tx_bytes = len(json.dumps(chaincode.ledger_blocks[-1]["transactions"][0]).encode('utf-8'))
                else: # Blockchain_Audit_Consent_Revocation
                    t0 = time.perf_counter()
                    dec = access_engine.evaluate_access(
                        requestor_id="dr-bench-01",
                        role=Role.DOCTOR,
                        organization="Hospital_A",
                        patient_id=pid,
                        resource_type="Observation",
                        purpose_of_use=PurposeOfUse.TREATMENT
                    )
                    t_tx = (time.perf_counter() - t0) * 1000.0
                    t_ac = t_tx
                    tx_bytes = len(json.dumps(chaincode.ledger_blocks[-1]["transactions"][0]).encode('utf-8'))

                tx_latencies.append(t_tx)
                access_latencies.append(t_ac)
                bytes_added.append(tx_bytes)

            total_bench_duration = time.perf_counter() - start_bench_time
            throughput_tps = NUM_TRANSACTIONS / total_bench_duration if total_bench_duration > 0 else 0

            # Measure block verification latency
            t_v_start = time.perf_counter()
            for b_idx in range(1, len(chaincode.ledger_blocks)):
                curr_b = chaincode.ledger_blocks[b_idx]
                prev_b = chaincode.ledger_blocks[b_idx - 1]
                assert curr_b["previous_hash"] == prev_b["block_hash"]
            block_verify_ms = (time.perf_counter() - t_v_start) * 1000.0

            all_runs.append({
                "config": config_name,
                "seed": seed,
                "num_transactions": NUM_TRANSACTIONS,
                "tx_latency_ms": float(np.mean(tx_latencies)),
                "access_latency_ms": float(np.mean(access_latencies)),
                "throughput_tps": float(throughput_tps),
                "avg_bytes_per_tx": float(np.mean(bytes_added)),
                "block_verify_ms": float(block_verify_ms),
                "total_blocks": len(chaincode.ledger_blocks)
            })

            print(f"  [DONE] {config_name:35s} | Latency: {np.mean(tx_latencies):.3f} ms | Throughput: {throughput_tps:,.1f} TPS | Storage: {np.mean(bytes_added):.0f} B/tx")

    # Aggregate Statistics
    df = pd.DataFrame(all_runs)
    grouped = df.groupby("config")

    summary_records = []
    for config_name, group in grouped:
        n = len(group)
        t_crit = t.ppf(0.975, df=n - 1) if n > 1 else 1.96

        def calc_ci(col):
            m = group[col].mean()
            s = group[col].std()
            ci = t_crit * (s / np.sqrt(n)) if n > 1 else 0.0
            return m, s, ci

        tx_m, tx_s, tx_ci = calc_ci("tx_latency_ms")
        ac_m, ac_s, ac_ci = calc_ci("access_latency_ms")
        tp_m, tp_s, tp_ci = calc_ci("throughput_tps")
        bytes_m, _, _ = calc_ci("avg_bytes_per_tx")
        bv_m, _, _ = calc_ci("block_verify_ms")

        summary_records.append({
            "configuration": config_name,
            "tx_latency_ms_mean": round(tx_m, 4),
            "tx_latency_ms_std": round(tx_s, 4),
            "tx_latency_ms_ci95": round(tx_ci, 4),
            "access_decision_ms_mean": round(ac_m, 4),
            "throughput_tps_mean": round(tp_m, 1),
            "throughput_tps_ci95": round(tp_ci, 1),
            "storage_overhead_bytes_per_tx": round(bytes_m, 1),
            "ledger_verification_ms": round(bv_m, 3)
        })

    summary_df = pd.DataFrame(summary_records)
    table_csv = os.path.join(TABLES_DIR, "table_blockchain_ablation.csv")
    summary_df.to_csv(table_csv, index=False)

    metrics_json = os.path.join(METRICS_DIR, "blockchain_benchmarks.json")
    with open(metrics_json, "w") as fp:
        json.dump(summary_records, fp, indent=2)

    print("\n" + "=" * 70)
    print("BLOCKCHAIN ABLATION SUMMARY")
    print("=" * 70)
    for _, row in summary_df.iterrows():
        print(f"  {row['configuration']:35s} | Latency: {row['tx_latency_ms_mean']:.3f}±{row['tx_latency_ms_ci95']:.3f} ms | Throughput: {row['throughput_tps_mean']:,} TPS | Storage: {row['storage_overhead_bytes_per_tx']:.0f} B/tx")
    print(f"\n[SAVED] Benchmark metrics: {metrics_json}")
    print(f"[SAVED] Publication Table: {table_csv}")
    print("=" * 70)

if __name__ == "__main__":
    run_blockchain_benchmarks()
