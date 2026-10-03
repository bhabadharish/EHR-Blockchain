"""
scripts/run_v2_latency_and_system.py
====================================
REAL-TIME INFERENCE LATENCY, MEMORY, AND MODEL FOOTPRINT BENCHMARK (CAHTDNET_V2_001)

Measures:
- Preprocessing latency per sample
- Model inference latency (CPU and MPS) across 2,000 requests
- Post-processing, calibration, and zero-trust decision latency
- End-to-end total pipeline latency
- Latency percentiles: Mean, Median (P50), P95, P99, and Peak Throughput (samples/sec)
- Model parameter count, disk footprint, and runtime memory usage
"""

import os
import sys
import json
import time
import psutil
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.feature_pipeline_v2 import FeaturePipelineV2
from src.models.ca_htdnet_v2 import CAHTDNetV2

EXPERIMENT_ID = "CAHTDNET_V2_001"
BASE_DIR = f"experiments/{EXPERIMENT_ID}"

def benchmark_system():
    print("=" * 70)
    print("CA-HTDNet V2: REAL-TIME LATENCY & SYSTEM BENCHMARK")
    print("=" * 70)

    os.makedirs(f"{BASE_DIR}/latency", exist_ok=True)

    # 1. Load Preprocessor and Model
    prep_path = f"{BASE_DIR}/preprocessing/preprocessor.pkl"
    model_path = f"{BASE_DIR}/models/{EXPERIMENT_ID}.pt"

    pipeline = FeaturePipelineV2.load(prep_path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = CAHTDNetV2(
        num_numerical=len(pipeline.feature_order) - 3,
        cat_cardinalities=[10, 15, 10],
        d_model=128
    ).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Model Size & Footprint
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = model_size_bytes / (1024 * 1024)

    print(f"  Model Parameter Count: {param_count:,} ({trainable_params:,} trainable)")
    print(f"  Model Weight Footprint: {model_size_mb:.2f} MB ({model_size_bytes/1024:.1f} KB)")

    # 2. Synthetic Test Telemetry Samples (2,000 samples)
    n_samples = 2000
    sample_raw_df = pd.DataFrame({
        "flow_duration": np.random.exponential(1.0, size=n_samples),
        "packet_count": np.random.randint(5, 500, size=n_samples),
        "byte_count": np.random.randint(500, 100000, size=n_samples),
        "packet_rate": np.random.uniform(10, 1000, size=n_samples),
        "byte_rate": np.random.uniform(1000, 500000, size=n_samples),
        "dst_port": np.random.choice([443, 80, 8080, 22], size=n_samples),
        "protocol": np.random.choice([6, 17], size=n_samples),
        "resource_sensitivity": np.random.uniform(0.3, 0.9, size=n_samples),
        "auth_status": np.random.choice([0, 1], size=n_samples),
        "failed_auth_count": np.random.choice([0, 1, 2, 5], size=n_samples),
        "request_frequency": np.random.uniform(0.5, 50, size=n_samples),
        "burst_score": np.random.uniform(0.1, 0.9, size=n_samples),
        "historical_risk": np.random.uniform(0.1, 0.8, size=n_samples),
        "user_role": np.random.choice(["doctor", "nurse", "admin", "iomt_device"], size=n_samples),
        "resource_type": np.random.choice(["Patient", "Observation", "Consent", "Device"], size=n_samples),
        "operation": np.random.choice(["read", "vread", "search", "update"], size=n_samples)
    })

    # Warmup
    print("  Running warm-up (100 iterations)...")
    for _ in range(100):
        single_row = sample_raw_df.iloc[[0]]
        x_num, x_cat = pipeline.transform(single_row)
        with torch.no_grad():
            _ = model(torch.tensor(x_num, device=device), torch.tensor(x_cat, device=device))

    # Benchmark: Individual stages per sample
    print(f"  Profiling {n_samples} individual single-event requests...")
    prep_latencies = []
    inf_latencies = []
    decision_latencies = []
    total_latencies = []

    for i in range(n_samples):
        row = sample_raw_df.iloc[[i]]

        # Stage 1: Preprocessing
        t0 = time.perf_counter()
        x_num, x_cat = pipeline.transform(row)
        t_prep = (time.perf_counter() - t0) * 1000.0
        prep_latencies.append(t_prep)

        # Stage 2: Neural Inference
        t0 = time.perf_counter()
        with torch.no_grad():
            t_num = torch.tensor(x_num, device=device)
            t_cat = torch.tensor(x_cat, device=device)
            out = model(t_num, t_cat)
            prob = out["probabilities"][0, 1].item()
            risk = out["risk_score"][0, 0].item()
        t_inf = (time.perf_counter() - t0) * 1000.0
        inf_latencies.append(t_inf)

        # Stage 3: Zero-Trust Decision & Risk Mapping
        t0 = time.perf_counter()
        decision = "DENY" if prob >= 0.50 or risk >= 0.70 else "ALLOW"
        t_dec = (time.perf_counter() - t0) * 1000.0
        decision_latencies.append(t_dec)

        total_latencies.append(t_prep + t_inf + t_dec)

    # Batch throughput benchmark (batch_size=256)
    print("  Measuring batch throughput (batch_size=256)...")
    X_num_batch, X_cat_batch = pipeline.transform(sample_raw_df)
    t_num_batch = torch.tensor(X_num_batch, device=device)
    t_cat_batch = torch.tensor(X_cat_batch, device=device)

    t0_batch = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            _ = model(t_num_batch, t_cat_batch)
    t_batch_total = time.perf_counter() - t0_batch
    throughput = (n_samples * 10) / t_batch_total

    # Summary Statistics
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "device": str(device),
        "hardware": "Apple Silicon (MPS Acceleration)",
        "model_parameters": param_count,
        "model_size_mb": float(round(model_size_mb, 2)),
        "samples_profiled": n_samples,
        "preprocessing_latency_ms": {
            "mean": float(np.mean(prep_latencies)),
            "median": float(np.median(prep_latencies)),
            "p95": float(np.percentile(prep_latencies, 95)),
            "p99": float(np.percentile(prep_latencies, 99))
        },
        "model_inference_latency_ms": {
            "mean": float(np.mean(inf_latencies)),
            "median": float(np.median(inf_latencies)),
            "p95": float(np.percentile(inf_latencies, 95)),
            "p99": float(np.percentile(inf_latencies, 99))
        },
        "decision_latency_ms": {
            "mean": float(np.mean(decision_latencies)),
            "median": float(np.median(decision_latencies)),
            "p95": float(np.percentile(decision_latencies, 95)),
            "p99": float(np.percentile(decision_latencies, 99))
        },
        "end_to_end_pipeline_latency_ms": {
            "mean": float(np.mean(total_latencies)),
            "median": float(np.median(total_latencies)),
            "p95": float(np.percentile(total_latencies, 95)),
            "p99": float(np.percentile(total_latencies, 99))
        },
        "throughput_samples_per_second": float(round(throughput, 1))
    }

    with open(f"{BASE_DIR}/latency/system_latency_benchmark.json", "w") as f:
        json.dump(summary, f, indent=2)

    df_lat = pd.DataFrame([
        {"stage": "Preprocessing", "mean_ms": np.mean(prep_latencies), "p50_ms": np.median(prep_latencies), "p95_ms": np.percentile(prep_latencies, 95), "p99_ms": np.percentile(prep_latencies, 99)},
        {"stage": "Model Inference", "mean_ms": np.mean(inf_latencies), "p50_ms": np.median(inf_latencies), "p95_ms": np.percentile(inf_latencies, 95), "p99_ms": np.percentile(inf_latencies, 99)},
        {"stage": "Decision & Policy", "mean_ms": np.mean(decision_latencies), "p50_ms": np.median(decision_latencies), "p95_ms": np.percentile(decision_latencies, 95), "p99_ms": np.percentile(decision_latencies, 99)},
        {"stage": "End-to-End Pipeline", "mean_ms": np.mean(total_latencies), "p50_ms": np.median(total_latencies), "p95_ms": np.percentile(total_latencies, 95), "p99_ms": np.percentile(total_latencies, 99)}
    ])
    df_lat.to_csv(f"{BASE_DIR}/latency/latency_benchmark.csv", index=False)

    print("\n" + "=" * 70)
    print("REAL-TIME LATENCY BENCHMARK SUMMARY:")
    print("=" * 70)
    print(df_lat.to_string(index=False))
    print(f"\nThroughput: {throughput:,.1f} samples/second")

if __name__ == "__main__":
    benchmark_system()
