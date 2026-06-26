"""
Spec2Prop-Edge: Edge Benchmark
===============================
Benchmark preprocessing and inference latency for edge deployment assessment.

IMPORTANT: If running on a laptop/workstation CPU, the device is reported as
"Laptop/Workstation CPU". Do NOT claim Raspberry Pi or Jetson deployment
unless actually benchmarked on that hardware.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import platform
import time

import numpy as np
import torch

from deploy.preprocess_runtime import preprocess_raman_file
from deploy.infer_single_sample import load_model, run_inference_torchscript


def detect_device_name():
    """Detect a human-readable device name."""
    proc = platform.processor() or platform.machine()
    system = platform.system()
    # Check for known edge devices
    machine = platform.machine().lower()
    if "aarch64" in machine or "arm" in machine:
        return f"ARM-based device ({proc})"
    return f"Laptop/Workstation CPU ({proc}, {system})"


def benchmark(
    model_path: str,
    sample_path: str,
    task: str = "family",
    num_runs: int = 500,
    warmup_runs: int = 20,
    device: str = "cpu",
):
    """Run the benchmark."""
    # Model size
    model_size_bytes = os.path.getsize(model_path)
    model_size_mb = model_size_bytes / (1024 * 1024)

    # Load model
    model = load_model(model_path, "torchscript")

    # Preprocess once to get the vector
    raman_vector, _ = preprocess_raman_file(sample_path)
    raman_tensor = torch.from_numpy(raman_vector).float().unsqueeze(0).unsqueeze(0)

    # --- Preprocessing benchmark ---
    preprocess_times = []
    for _ in range(warmup_runs):
        preprocess_raman_file(sample_path)

    for _ in range(num_runs):
        t0 = time.perf_counter()
        preprocess_raman_file(sample_path)
        t1 = time.perf_counter()
        preprocess_times.append((t1 - t0) * 1000)

    # --- Inference benchmark ---
    inference_times = []
    for _ in range(warmup_runs):
        run_inference_torchscript(model, raman_tensor)

    for _ in range(num_runs):
        t0 = time.perf_counter()
        run_inference_torchscript(model, raman_tensor)
        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000)

    # --- Total pipeline ---
    total_times = []
    for _ in range(min(num_runs, 100)):
        t0 = time.perf_counter()
        vec, _ = preprocess_raman_file(sample_path)
        tensor = torch.from_numpy(vec).float().unsqueeze(0).unsqueeze(0)
        run_inference_torchscript(model, tensor)
        t1 = time.perf_counter()
        total_times.append((t1 - t0) * 1000)

    preprocess_times = np.array(preprocess_times)
    inference_times = np.array(inference_times)
    total_times = np.array(total_times)

    device_name = detect_device_name()

    results = {
        "device": device_name,
        "model_name": os.path.basename(model_path),
        "task": task,
        "num_runs": num_runs,
        "preprocessing_time_ms_mean": round(float(np.mean(preprocess_times)), 3),
        "preprocessing_time_ms_std": round(float(np.std(preprocess_times)), 3),
        "inference_time_ms_mean": round(float(np.mean(inference_times)), 3),
        "inference_time_ms_std": round(float(np.std(inference_times)), 3),
        "total_time_ms_mean": round(float(np.mean(total_times)), 3),
        "total_time_ms_std": round(float(np.std(total_times)), 3),
        "throughput_samples_per_sec": round(1000.0 / float(np.mean(total_times)), 1),
        "model_size_mb": round(model_size_mb, 3),
        "input_shape": [1, 1, 2048],
        "notes": (
            "Benchmarked on " + device_name + ". "
            "If not tested on actual edge hardware (Raspberry Pi, Jetson), "
            "these results should be reported as 'edge-oriented CPU inference benchmark'."
        ),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Spec2Prop-Edge: Edge Benchmark")
    parser.add_argument("--model", required=True, help="Path to TorchScript model")
    parser.add_argument("--task", default="family", choices=["family", "property", "multimodal"])
    parser.add_argument("--sample", required=True, help="Path to sample Raman CSV")
    parser.add_argument("--num-runs", type=int, default=500)
    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="deploy/outputs/edge_benchmark.json")
    args = parser.parse_args()

    print(f"Running edge benchmark: {args.num_runs} runs...")
    results = benchmark(
        args.model, args.sample, args.task,
        args.num_runs, args.warmup_runs, args.device
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n--- Edge Benchmark Results ---")
    print(f"Device: {results['device']}")
    print(f"Preprocessing: {results['preprocessing_time_ms_mean']:.2f} ± {results['preprocessing_time_ms_std']:.2f} ms")
    print(f"Inference: {results['inference_time_ms_mean']:.2f} ± {results['inference_time_ms_std']:.2f} ms")
    print(f"Total Pipeline: {results['total_time_ms_mean']:.2f} ± {results['total_time_ms_std']:.2f} ms")
    print(f"Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"Model Size: {results['model_size_mb']:.3f} MB")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
