import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Spec2Prop-Inorg: Export Edge Models
===================================
Exports the lightweight CNN models (e.g. LiteSpecNet) to TorchScript and ONNX
for deployment on edge devices (Raspberry Pi, Jetson Nano).
"""

import argparse
import os
import time
import json
import torch
import numpy as np
from models.cnn1d import LiteSpecNet, model_size_mb, count_parameters

def export_model(checkpoint_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # Instantiate architecture (assumes default LiteSpecNet params for now, 
    # in practice we'd load config or infer from checkpoint)
    # We use num_classes=12 just as a placeholder since we don't know the exact task here
    # A robust script would load the encoder config.
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # We will just benchmark a dummy LiteSpecNet if checkpoint is not found/specified
    model = LiteSpecNet(in_channels=1, num_classes=12)
    
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            print("Loaded weights successfully.")
        except Exception as e:
            print(f"Could not load weights, using random init for benchmark: {e}")
            
    model.eval()
    
    # Parameter count and size
    params = count_parameters(model)
    size_mb = model_size_mb(model)
    
    print(f"Model: LiteSpecNet | Parameters: {params:,} | Size: {size_mb:.2f} MB")
    
    # TorchScript Export
    dummy_input = torch.randn(1, 1, 2048)
    ts_path = os.path.join(out_dir, "litespecnet_torchscript.pt")
    try:
        traced_script_module = torch.jit.trace(model, dummy_input)
        traced_script_module.save(ts_path)
        print(f"Exported TorchScript to {ts_path}")
    except Exception as e:
        print(f"TorchScript export failed: {e}")

    # ONNX Export
    try:
        import onnx
        onnx_path = os.path.join(out_dir, "litespecnet.onnx")
        torch.onnx.export(
            model, dummy_input, onnx_path, 
            export_params=True, opset_version=12,
            do_constant_folding=True,
            input_names=['raman_input'], output_names=['output'],
            dynamic_axes={'raman_input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Exported ONNX to {onnx_path}")
    except ImportError:
        print("ONNX not installed. Skipping ONNX export. (pip install onnx)")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        
    # CPU Inference Benchmark
    print("Benchmarking CPU inference latency (100 passes)...")
    latencies = []
    with torch.no_grad():
        for _ in range(100):
            t0 = time.time()
            _ = model(dummy_input)
            latencies.append(time.time() - t0)
            
    avg_latency = np.mean(latencies) * 1000 # ms
    p95_latency = np.percentile(latencies, 95) * 1000
    
    benchmark = {
        "model": "LiteSpecNet",
        "parameters": params,
        "size_mb": round(size_mb, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "target_devices": ["Raspberry Pi 4/5", "Jetson Nano", "Laptop CPU (Benchmarked)"]
    }
    
    with open(os.path.join(out_dir, "edge_benchmark.json"), "w") as f:
        json.dump(benchmark, f, indent=2)
        
    print(f"Benchmark results: {benchmark['avg_latency_ms']} ms/sample")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/family_classification/LiteSpecNet_best.pt")
    parser.add_argument("--out-dir", default="results/edge")
    args = parser.parse_args()
    export_model(args.checkpoint, args.out_dir)
