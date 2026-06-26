"""
Spec2Prop-Edge: Batch Inference
================================
Process a folder of Raman/XRD spectral files and generate batch predictions.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import glob

import numpy as np
import pandas as pd
import torch

from deploy.preprocess_runtime import preprocess_raman_file
from deploy.infer_single_sample import load_model, run_inference_torchscript, decode_predictions
from deploy.generate_prediction_report import DISCLAIMER


def main():
    parser = argparse.ArgumentParser(description="Spec2Prop-Edge: Batch Inference")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--model-type", default="torchscript", choices=["torchscript", "onnx"])
    parser.add_argument("--task", required=True, choices=["family", "property", "multimodal"])
    parser.add_argument("--input-dir", required=True, help="Folder containing Raman CSV/TXT files")
    parser.add_argument("--label-encoder", required=True, help="Path to label encoders JSON")
    parser.add_argument("--output-csv", default="deploy/outputs/batch_predictions.csv")
    parser.add_argument("--output-json", default="deploy/outputs/batch_predictions.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    # Load label encoder
    with open(args.label_encoder) as f:
        all_encoders = json.load(f)
    if "chemical_family_model" in all_encoders:
        label_encoder = all_encoders["chemical_family_model"]
    else:
        label_encoder = all_encoders

    # Load model
    print(f"Loading model: {args.model}")
    model = load_model(args.model, args.model_type)

    # Find spectral files
    patterns = ["*.csv", "*.txt", "*.CSV", "*.TXT"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(args.input_dir, p)))
    files = sorted(set(files))
    print(f"Found {len(files)} spectral files in {args.input_dir}")

    if not files:
        print("No files found. Exiting.")
        return

    results = []
    failures = []

    for filepath in files:
        sample_id = os.path.splitext(os.path.basename(filepath))[0]
        print(f"  Processing: {sample_id}...", end=" ")

        try:
            # Preprocess
            raman_vector, raman_meta = preprocess_raman_file(filepath)
            raman_tensor = torch.from_numpy(raman_vector).float().unsqueeze(0).unsqueeze(0)

            # Inference
            t0 = time.perf_counter()
            output = run_inference_torchscript(model, raman_tensor)
            t1 = time.perf_counter()
            inference_time_ms = (t1 - t0) * 1000

            probs = list(output.values())[0]
            top_preds = decode_predictions(probs, label_encoder, args.top_k)

            row = {
                "sample_id": sample_id,
                "file": os.path.basename(filepath),
                "predicted_class": top_preds[0]["class"],
                "confidence": top_preds[0]["confidence"],
                "inference_time_ms": round(inference_time_ms, 3),
                "status": "success",
                "error": "",
            }
            for i, p in enumerate(top_preds):
                row[f"top{i+1}_class"] = p["class"]
                row[f"top{i+1}_confidence"] = round(p["confidence"], 4)

            results.append(row)
            print(f"-> {top_preds[0]['class']} ({top_preds[0]['confidence']:.3f})")

        except Exception as e:
            failures.append({"sample_id": sample_id, "file": os.path.basename(filepath), "error": str(e)})
            results.append({
                "sample_id": sample_id,
                "file": os.path.basename(filepath),
                "predicted_class": "",
                "confidence": 0.0,
                "inference_time_ms": 0,
                "status": "failed",
                "error": str(e),
            })
            print(f"-> FAILED: {e}")

    # Save CSV
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nBatch CSV saved: {args.output_csv}")

    # Save JSON
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    batch_report = {
        "total_files": len(files),
        "successful": len(files) - len(failures),
        "failed": len(failures),
        "model": os.path.basename(args.model),
        "task": args.task,
        "results": results,
        "failures": failures,
        "disclaimer": DISCLAIMER,
    }
    with open(args.output_json, "w") as f:
        json.dump(batch_report, f, indent=2, default=str)
    print(f"Batch JSON saved: {args.output_json}")

    # Summary
    print(f"\n--- Batch Summary ---")
    print(f"Total: {len(files)} | Success: {len(files) - len(failures)} | Failed: {len(failures)}")


if __name__ == "__main__":
    main()
