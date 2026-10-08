import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""Master Pipeline Orchestrator for HAB-IDS (Phase 43).

Executes the complete end-to-end research pipeline deterministically in one unified run:
1. Data Ingestion & Manifests
2. Data Quality & Leakage Audit
3. Leakage-Controlled Splitting
4. Comparative Baselines Training
5. Tuned XGBoost (Multi-Seed)
6. Tuned LightGBM (Multi-Seed)
7. Tuned CatBoost (Multi-Seed)
8. OOF Predictions, Disagreement-Aware Meta-Learner, Routing, Hierarchical, Calibration, & Thresholding
9. Locked Test Set Evaluation & Cross-Dataset Benchmark Table
10. 12-Configuration Ablation Study
11. Real-Time Inference Latency Benchmarking (1,000+ iterations)
12. Cryptography & Blockchain Audit Benchmarking
13. Research Figure Generation
14. Comprehensive Error Analysis
15. Independent Result Verification & Production Validation Gate
"""

import sys
import time
import subprocess


def run_stage(step_num: int, title: str, script_name: str):
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {title.upper()}")
    print(f"{'='*70}")
    t0 = time.perf_counter()
    ret = subprocess.run([sys.executable, script_name], capture_output=False)
    elapsed = time.perf_counter() - t0
    if ret.returncode != 0:
        print(f"ERROR: Stage {step_num} ({script_name}) failed with exit code {ret.returncode}.")
        sys.exit(ret.returncode)
    print(f"-> Step {step_num} completed in {elapsed:.2f} seconds.")


def main():
    start_total = time.perf_counter()
    print("\n" + "#"*70)
    print("# LAUNCHING FULL HAB-IDS RESEARCH AND VALIDATION PIPELINE")
    print("#"*70)

    stages = [
        (1, "Data Preparation & Manifest Generation", "scripts/prepare_data.py"),
        (2, "Data Quality & Leakage Auditing", "scripts/validate_data.py"),
        (3, "Canonical Leakage-Controlled Splitting", "scripts/split_data.py"),
        (4, "Baseline Model Training", "scripts/train_baselines.py"),
        (5, "Tuned XGBoost & Multi-Seed Stability", "scripts/train_xgboost.py"),
        (6, "Tuned LightGBM & Multi-Seed Stability", "scripts/train_lightgbm.py"),
        (7, "Tuned CatBoost & Multi-Seed Stability", "scripts/train_catboost.py"),
        (8, "Meta-Learning, Routing, Hierarchical, Calibration & Threshold", "scripts/train_meta_model.py"),
        (9, "Test Set Evaluation & Cross-Dataset Tasks", "scripts/evaluate.py"),
        (10, "12-Configuration Ablation Study", "scripts/ablation.py"),
        (11, "Real-Time Inference Benchmarking (1000+ iters)", "scripts/benchmark_inference.py"),
        (12, "PQC and Blockchain Benchmarking", "scripts/run_security_benchmarks.py"),
        (13, "Publication Figures Generation", "scripts/generate_figures.py"),
        (14, "Comprehensive Error Analysis", "scripts/error_analysis.py"),
        (15, "Result Consistency Verification & Validation Gate", "scripts/verify_results.py"),
    ]

    for step, title, script in stages:
        run_stage(step, title, script)

    total_elapsed = time.perf_counter() - start_total
    print("\n" + "#"*70)
    print(f"# FULL PIPELINE EXECUTED SUCCESSFULLY IN {total_elapsed/60:.2f} MINUTES")
    print("# ALL ARTIFACTS, FIGURES, AND METRICS HAVE BEEN VALIDATED")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
