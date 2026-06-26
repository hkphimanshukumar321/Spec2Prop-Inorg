"""
Spec2Prop-Edge: Single-Sample Inference
========================================
CLI tool for running inference on a single Raman/XRD spectral file.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time

import numpy as np
import torch

from deploy.preprocess_runtime import preprocess_raman_file, preprocess_xrd_file, build_descriptor_vector
from deploy.generate_prediction_report import generate_report, DISCLAIMER
from deploy.saliency_runtime import compute_gradient_saliency, plot_saliency


def load_model(model_path: str, model_type: str = "torchscript"):
    """Load a deployed model."""
    if model_type == "torchscript":
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()
        return model
    elif model_type == "onnx":
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(model_path)
            return session
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX inference.")
    elif model_type == "pytorch":
        raise NotImplementedError(
            "Raw PyTorch checkpoint loading requires model class specification. "
            "Use TorchScript export instead."
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_inference_torchscript(model, input_tensor: torch.Tensor) -> dict:
    """Run inference with a TorchScript model."""
    with torch.no_grad():
        output = model(input_tensor)

    if isinstance(output, dict):
        results = {}
        for key, logits in output.items():
            probs = torch.softmax(logits, dim=1)
            results[key] = probs.cpu().numpy()[0]
        return results
    else:
        probs = torch.softmax(output, dim=1)
        return {"default": probs.cpu().numpy()[0]}


def run_inference_onnx(session, input_array: np.ndarray) -> dict:
    """Run inference with an ONNX model."""
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_array})
    from scipy.special import softmax
    probs = softmax(output[0], axis=1)
    return {"default": probs[0]}


def decode_predictions(probs: np.ndarray, label_encoder: dict, top_k: int = 3):
    """Decode probability vector to top-k class predictions."""
    inv_encoder = {int(v): k for k, v in label_encoder.items()}
    top_indices = np.argsort(probs)[::-1][:top_k]

    predictions = []
    for rank, idx in enumerate(top_indices, 1):
        predictions.append({
            "rank": rank,
            "class": inv_encoder.get(idx, f"class_{idx}"),
            "confidence": float(probs[idx]),
        })
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Spec2Prop-Edge: Single-Sample Inference")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--model-type", default="torchscript", choices=["torchscript", "onnx"])
    parser.add_argument("--task", required=True, choices=["family", "property", "multimodal"])
    parser.add_argument("--raman-file", required=True, help="Path to Raman CSV/TXT")
    parser.add_argument("--xrd-file", default=None, help="Path to XRD CSV/TXT (for multimodal)")
    parser.add_argument("--descriptor-json", default=None, help="Path to descriptor JSON (for property)")
    parser.add_argument("--label-encoder", required=True, help="Path to label encoders JSON")
    parser.add_argument("--output", default="deploy/outputs/prediction_report.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--saliency", action="store_true", help="Generate saliency plot")
    args = parser.parse_args()

    # Load label encoders
    with open(args.label_encoder) as f:
        all_encoders = json.load(f)

    # Determine primary encoder
    if args.task == "family":
        primary_encoder_key = "chemical_family_model"
    else:
        primary_encoder_key = list(all_encoders.keys())[0]

    label_encoder = all_encoders.get(primary_encoder_key, all_encoders)
    if not isinstance(label_encoder, dict) or not all(isinstance(v, int) for v in label_encoder.values()):
        # The top-level dict IS the encoder
        label_encoder = all_encoders

    # Preprocess Raman
    print("Preprocessing Raman spectrum...")
    raman_vector, raman_meta = preprocess_raman_file(args.raman_file)
    raman_tensor = torch.from_numpy(raman_vector).float().unsqueeze(0).unsqueeze(0)  # [1, 1, 2048]

    # Load model
    print(f"Loading model: {args.model}")
    model = load_model(args.model, args.model_type)

    # Inference
    t0 = time.perf_counter()
    if args.model_type == "torchscript":
        results = run_inference_torchscript(model, raman_tensor)
    else:
        results = run_inference_onnx(model, raman_tensor.numpy())
    t1 = time.perf_counter()
    inference_time_ms = (t1 - t0) * 1000

    # Decode
    probs = list(results.values())[0]
    top_preds = decode_predictions(probs, label_encoder, args.top_k)

    # Build report
    warnings_list = []
    if np.max(raman_vector) < 0.01:
        warnings_list.append("Spectrum appears very weak — check input quality.")

    report = {
        "sample_id": os.path.splitext(os.path.basename(args.raman_file))[0],
        "task": args.task,
        "model_name": os.path.basename(args.model),
        "preprocessing": raman_meta.get("preprocessing", {}),
        "top_predictions": top_preds,
        "inference_time_ms": round(inference_time_ms, 3),
        "device": "cpu",
        "warnings": warnings_list,
    }

    # Task-specific fields
    if args.task == "family" and top_preds:
        report["predicted_family"] = top_preds[0]["class"]

    # Generate reports
    output_dir = os.path.dirname(args.output) or "deploy/outputs"
    prefix = os.path.splitext(os.path.basename(args.output))[0]
    generate_report(report, output_dir, prefix)

    # Optional saliency
    if args.saliency and args.model_type == "torchscript":
        print("Computing saliency...")
        try:
            saliency = compute_gradient_saliency(model, raman_tensor)
            saliency_path = os.path.join(output_dir, "saliency_plot.png")
            plot_saliency(raman_vector, saliency, saliency_path)
        except Exception as e:
            print(f"Saliency computation failed: {e}")

    print(f"\nTop prediction: {top_preds[0]['class']} ({top_preds[0]['confidence']:.4f})")
    print(f"Inference time: {inference_time_ms:.2f} ms")


if __name__ == "__main__":
    main()
