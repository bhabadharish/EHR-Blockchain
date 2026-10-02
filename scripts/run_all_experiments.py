#!/usr/bin/env python3
"""
scripts/run_all_experiments.py
Phase 29: Complete Master Orchestrator for 100% Reproducible Research Pipeline.

Sequentially executes:
1. Dataset verification & provenance audit (Quality Gate QG1)
2. Post-quantum cryptographic benchmarks (Phase 8 & 22)
3. Blockchain consensus & consent ablation benchmarks (Phase 11 & 23)
4. Large-scale synthetic FHIR scalability benchmarks (Phase 24)
5. End-to-end Hospital A -> Hospital B exchange & 10 attack vectors (Phases 25 & 26)
6. Multi-seed AI threat detection & component ablation suite (Phases 15-21)
7. Automated publication table consolidation (Phase 29)
8. Publication figure generation (Figures 1-16) (Phase 30)
9. Cross-artifact consistency audit (Phase 39, Quality Gate QG10)

Usage:
    python scripts/run_all_experiments.py
"""

import os
import sys
import time
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXE = sys.executable

PIPELINE_STEPS = [
    ("Step 1: Dataset Verification & Integrity Audit (QG1)", "scripts/verify_datasets.py"),
    ("Step 2: Cryptographic Agility & PQC Benchmarks (Phases 8 & 22)", "scripts/run_crypto_benchmarks.py"),
    ("Step 3: Blockchain Consensus & Smart Contract Benchmarks (Phases 11 & 23)", "scripts/run_blockchain_benchmarks.py"),
    ("Step 4: Large-Scale System Scalability Evaluation (Phase 24)", "scripts/run_scalability.py"),
    ("Step 5: End-to-End Exchange & Controlled Attack Simulation (Phases 25 & 26)", "scripts/run_attack_simulation.py"),
    ("Step 6: Threat Detection Model Evaluation & Ablation (Phases 15-21)", "scripts/evaluate_models.py"),
    ("Step 7: Automated Publication Table Generation (Phase 29)", "scripts/generate_tables.py"),
    ("Step 8: Publication Figure Generation (Figures 1 to 16) (Phase 30)", "scripts/generate_figures.py"),
    ("Step 9: Cross-Artifact Consistency & Quality Gate Audit (Phase 39)", "scripts/check_consistency.py")
]

def main():
    print("=" * 80)
    print("MASTER REPRODUCIBILITY ORCHESTRATOR: COMPLETE RESEARCH EVALUATION PIPELINE")
    print("Project: Crypto-Agile FHIR-Blockchain EHR Exchange with Intelligent Threat Detection")
    print("=" * 80)

    t_total_start = time.perf_counter()
    passed_steps = 0

    for step_desc, script_rel in PIPELINE_STEPS:
        script_path = os.path.join(PROJECT_ROOT, script_rel)
        if not os.path.exists(script_path):
            print(f"\n[ERROR] Required script missing: {script_path}")
            sys.exit(1)

        print(f"\n>>> Executing: {step_desc}")
        print(f"    Command: {PYTHON_EXE} {script_rel}")
        t0 = time.perf_counter()
        
        result = subprocess.run([PYTHON_EXE, script_path], cwd=PROJECT_ROOT)
        elapsed = time.perf_counter() - t0

        if result.returncode != 0:
            print(f"\n[FAILURE] {step_desc} exited with error code {result.returncode} after {elapsed:.2f}s!")
            sys.exit(result.returncode)

        print(f"    [COMPLETED] in {elapsed:.2f} seconds.")
        passed_steps += 1

    total_time = time.perf_counter() - t_total_start
    print("\n" + "=" * 80)
    print(f"ALL {passed_steps} EXPERIMENTAL PIPELINE STEPS SUCCESSFULLY EXECUTED & VALIDATED!")
    print(f"Total Execution Time: {total_time:.2f} seconds ({total_time/60.0:.2f} minutes).")
    print("All results, tables, metrics, and publication figures are fully reproduced.")
    print("=" * 80)

if __name__ == "__main__":
    main()
