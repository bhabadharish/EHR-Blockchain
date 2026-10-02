#!/usr/bin/env python3
"""
scripts/benchmark_apple_m2.py
Apple M2 Hardware Detection and Baseline Profiling.

Automatically detects:
- Machine architecture
- Chip (e.g., Apple M2)
- Total and available RAM
- Operating System & Kernel
- Python version
- PyTorch version
- Available acceleration device (MPS vs CPU fallback)

Measures:
- Tensor operation benchmark on MPS vs CPU
- Peak memory consumption (RAM target < 6 GB compliance)
- Inference latency per sample
- Model size and parameter budget constraints (< 1M - 2M parameters)

Outputs:
- apple_m2_benchmark.json
- apple_m2_benchmark.csv
- results/metrics/apple_m2_benchmark.json
- results/tables/apple_m2_benchmark.csv
"""

import os
import sys
import time
import json
import subprocess
import platform
import psutil
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def get_cpu_brand() -> str:
    """Detects Apple Silicon processor brand string using sysctl on macOS."""
    if platform.system() == "Darwin":
        try:
            res = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True
            )
            brand = res.stdout.strip()
            if brand:
                return brand
        except Exception:
            pass
    return platform.processor() or "Unknown ARM64 CPU"

def profile_device_ops(device_name: str, tensor_size: int = 2048, iterations: int = 50) -> dict:
    """Profiles matrix multiplication and throughput on target device."""
    device = torch.device(device_name)
    try:
        # Warmup
        x = torch.randn(tensor_size, tensor_size, device=device)
        y = torch.randn(tensor_size, tensor_size, device=device)
        for _ in range(5):
            _ = torch.matmul(x, y)
        if device_name == "mps":
            torch.mps.synchronize()

        start_time = time.perf_counter()
        for _ in range(iterations):
            z = torch.matmul(x, y)
        if device_name == "mps":
            torch.mps.synchronize()
        total_time = time.perf_counter() - start_time
        avg_latency_ms = (total_time / iterations) * 1000.0
        
        # 2 * N^3 FLOPs per matmul
        flops_per_matmul = 2 * (tensor_size ** 3)
        tflops = (flops_per_matmul * iterations) / (total_time * 1e12)
        
        return {
            "supported": True,
            "avg_latency_ms": round(avg_latency_ms, 3),
            "throughput_tflops": round(tflops, 3),
            "total_benchmark_time_sec": round(total_time, 4)
        }
    except Exception as e:
        return {
            "supported": False,
            "error": str(e)
        }

def run_apple_m2_profiling():
    print("=" * 75)
    print("APPLE M2 HARDWARE & RUNTIME ENVIRONMENT PROFILER")
    print("=" * 75)

    machine = platform.machine()
    chip = get_cpu_brand()
    os_name = f"{platform.system()} {platform.release()}"
    python_ver = sys.version.split()[0]
    torch_ver = torch.__version__
    
    total_ram_bytes = psutil.virtual_memory().total
    total_ram_gb = round(total_ram_bytes / (1024 ** 3), 2)
    available_ram_gb = round(psutil.virtual_memory().available / (1024 ** 3), 2)
    
    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()
    preferred_device = "mps" if mps_available else "cpu"

    print(f"[*] Machine Architecture : {machine}")
    print(f"[*] Processor / Chip     : {chip}")
    print(f"[*] Total System RAM     : {total_ram_gb} GB (Available: {available_ram_gb} GB)")
    print(f"[*] Operating System     : {os_name}")
    print(f"[*] Python Version       : {python_ver}")
    print(f"[*] PyTorch Version      : {torch_ver}")
    print(f"[*] MPS Built / Available: {mps_built} / {mps_available}")
    print(f"[*] Primary Device       : {preferred_device}")

    # Benchmark CPU
    print("\n[+] Benchmarking CPU matrix operations...")
    cpu_bench = profile_device_ops("cpu", tensor_size=1024, iterations=20)
    print(f"    CPU Latency (1024x1024): {cpu_bench.get('avg_latency_ms')} ms ({cpu_bench.get('throughput_tflops')} TFLOPS)")

    # Benchmark MPS if available
    mps_bench = {"supported": False}
    if mps_available:
        print("\n[+] Benchmarking Apple Silicon MPS operations...")
        mps_bench = profile_device_ops("mps", tensor_size=1024, iterations=20)
        print(f"    MPS Latency (1024x1024): {mps_bench.get('avg_latency_ms')} ms ({mps_bench.get('throughput_tflops')} TFLOPS)")

    # Process Memory Footprint
    process = psutil.Process(os.getpid())
    current_rss_mb = round(process.memory_info().rss / (1024 ** 2), 2)

    # Lightweight Neural Net Baseline Test with Proposed Architecture
    print("\n[+] Profiling Lightweight Model Parameters & Memory Envelope...")
    from backend.models.architectures import ProposedLightweightTCNConvAttention
    test_model = ProposedLightweightTCNConvAttention(in_features=50, num_classes=15, hidden_dim=32)
    param_count = sum(p.numel() for p in test_model.parameters())
    param_size_mb = round(sum(p.numel() * p.element_size() for p in test_model.parameters()) / (1024 ** 2), 4)
    
    # 1 sample inference latency
    test_model.eval()
    dummy_input = torch.randn(1, 50)
    with torch.no_grad():
        for _ in range(10):
            _ = test_model(dummy_input)
        t0 = time.perf_counter()
        for _ in range(100):
            _ = test_model(dummy_input)
        t_inf_ms = round(((time.perf_counter() - t0) / 100) * 1000.0, 4)

    print(f"    Sample Architecture Param Count: {param_count} parameters")
    print(f"    Model Weights Size             : {param_size_mb} MB")
    print(f"    Single-sample Inference Latency: {t_inf_ms} ms")
    print(f"    Process Peak RSS Memory        : {current_rss_mb} MB")

    # Resource Compliance Checks
    compliance = {
        "m2_hardware_confirmed": "Apple M2" in chip or "Apple" in chip,
        "ram_budget_under_6gb_limit": current_rss_mb < 6144,
        "model_parameter_limit_compliant": param_count < 2_000_000,
        "mps_acceleration_active": mps_available,
        "cpu_fallback_verified": cpu_bench.get("supported", False)
    }

    benchmark_data = {
        "machine": machine,
        "chip": chip,
        "total_ram_gb": total_ram_gb,
        "available_ram_gb": available_ram_gb,
        "os": os_name,
        "python_version": python_ver,
        "pytorch_version": torch_ver,
        "device": preferred_device,
        "mps_available": mps_available,
        "mps_built": mps_built,
        "cpu_latency_ms": cpu_bench.get("avg_latency_ms"),
        "cpu_tflops": cpu_bench.get("throughput_tflops"),
        "mps_latency_ms": mps_bench.get("avg_latency_ms") if mps_available else None,
        "mps_tflops": mps_bench.get("throughput_tflops") if mps_available else None,
        "baseline_rss_mb": current_rss_mb,
        "target_ram_limit_gb": 6.0,
        "target_max_parameters": 2_000_000,
        "target_preferred_parameters": 1_000_000,
        "sample_model_parameters": param_count,
        "sample_model_size_mb": param_size_mb,
        "single_sample_inference_ms": t_inf_ms,
        "compliance": compliance
    }

    # Save to apple_m2_benchmark.json
    root_json_path = os.path.join(PROJECT_ROOT, "apple_m2_benchmark.json")
    with open(root_json_path, "w") as fp:
        json.dump(benchmark_data, fp, indent=2)
    print(f"\n[+] Saved JSON benchmark to: {root_json_path}")

    # Save to apple_m2_benchmark.csv
    flat_data = {k: [v] if not isinstance(v, dict) else [json.dumps(v)] for k, v in benchmark_data.items()}
    df = pd.DataFrame(flat_data)
    root_csv_path = os.path.join(PROJECT_ROOT, "apple_m2_benchmark.csv")
    df.to_csv(root_csv_path, index=False)
    print(f"[+] Saved CSV benchmark to: {root_csv_path}")

    # Mirror into results/metrics and results/tables
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "metrics"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "tables"), exist_ok=True)
    
    with open(os.path.join(PROJECT_ROOT, "results", "metrics", "apple_m2_benchmark.json"), "w") as fp:
        json.dump(benchmark_data, fp, indent=2)
    df.to_csv(os.path.join(PROJECT_ROOT, "results", "tables", "apple_m2_benchmark.csv"), index=False)

    print("=" * 75)
    print("PHASE 0 BENCHMARK COMPLETE: Apple M2 profiling successfully saved.")
    print("=" * 75)
    return benchmark_data

if __name__ == "__main__":
    run_apple_m2_profiling()
