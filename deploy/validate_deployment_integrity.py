"""
Spec2Prop-Edge: Deployment Integrity Validation
=================================================
Verifies that the deployment pipeline is consistent with training.

If validation passes, prints:
    DEPLOYMENT INTEGRITY: PASSED

If validation fails, prints the reason and exits with non-zero code.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time

import numpy as np
import torch


def _check(condition, message):
    """Assert a condition, raising with a clear message on failure."""
    if not condition:
        raise AssertionError(f"INTEGRITY CHECK FAILED: {message}")


def validate(
    model_path: str,
    label_encoder_path: str,
    sample_raman_path: str,
    deployment_config_path: str = None,
    model_card_path: str = None,
    task: str = "family",
    benchmark_device: str = None,
):
    """
    Run all deployment integrity checks.

    Raises AssertionError on first failure.
    """
    checks_passed = 0

    # ===== 1. FILE CHECKS =====
    print("[1/8] File checks...")
    _check(os.path.isfile(model_path), f"Model file not found: {model_path}")
    _check(os.path.isfile(label_encoder_path), f"Label encoder not found: {label_encoder_path}")
    _check(os.path.isfile(sample_raman_path), f"Sample input not found: {sample_raman_path}")
    if deployment_config_path:
        _check(os.path.isfile(deployment_config_path), f"Deployment config not found: {deployment_config_path}")
    checks_passed += 1
    print("  -> File checks PASSED")

    # ===== 2. PREPROCESSING CHECKS =====
    print("[2/8] Preprocessing checks...")
    from deploy.preprocess_runtime import preprocess_raman_file
    vector, meta = preprocess_raman_file(sample_raman_path)
    _check(len(vector) == 2048, f"Output length {len(vector)} != 2048")
    _check(not np.any(np.isnan(vector)), "Preprocessed vector contains NaN")
    _check(not np.any(np.isinf(vector)), "Preprocessed vector contains Inf")
    _check(np.all(np.isfinite(vector)), "Preprocessed vector has non-finite values")
    _check(np.max(vector) <= 1.5, f"Max intensity {np.max(vector):.3f} > 1.5 after normalization")
    _check(meta["preprocessing"]["n_points"] == 2048, "n_points mismatch in metadata")
    checks_passed += 1
    print("  -> Preprocessing checks PASSED")

    # ===== 3. MODEL INPUT CHECKS =====
    print("[3/8] Model input checks...")
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    raman_tensor = torch.from_numpy(vector).float().unsqueeze(0).unsqueeze(0)  # [1, 1, 2048]
    _check(raman_tensor.shape == (1, 1, 2048), f"Input shape {raman_tensor.shape} != (1, 1, 2048)")

    try:
        with torch.no_grad():
            output = model(raman_tensor)
    except Exception as e:
        _check(False, f"Model forward pass failed: {e}")
    checks_passed += 1
    print("  -> Model input checks PASSED")

    # ===== 4. LABEL ENCODER CHECKS =====
    print("[4/8] Label encoder checks...")
    with open(label_encoder_path) as f:
        all_encoders = json.load(f)

    if "chemical_family_model" in all_encoders:
        encoder = all_encoders["chemical_family_model"]
    else:
        encoder = all_encoders

    _check(isinstance(encoder, dict), "Label encoder is not a dictionary")
    _check(len(encoder) > 0, "Label encoder is empty")

    if isinstance(output, dict):
        first_key = list(output.keys())[0]
        output_dim = output[first_key].shape[1]
    else:
        output_dim = output.shape[1]
    _check(
        output_dim == len(encoder),
        f"Model output dim {output_dim} != label encoder size {len(encoder)}"
    )

    # Top-k decoding test
    inv_encoder = {int(v): k for k, v in encoder.items()}
    _check(0 in inv_encoder, "Label encoder missing class index 0")
    checks_passed += 1
    print("  -> Label encoder checks PASSED")

    # ===== 5. INFERENCE CHECKS =====
    print("[5/8] Inference checks...")
    if isinstance(output, dict):
        logits = list(output.values())[0]
    else:
        logits = output

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    prob_sum = float(np.sum(probs))
    _check(abs(prob_sum - 1.0) < 0.01, f"Softmax probabilities sum to {prob_sum:.4f}, not ~1.0")
    _check(np.all(np.isfinite(probs)), "Confidence values contain non-finite numbers")
    _check(np.all(probs >= 0), "Negative probability values detected")
    checks_passed += 1
    print("  -> Inference checks PASSED")

    # ===== 6. MULTITASK CHECKS =====
    print("[6/8] Multitask/regression checks...")
    # For classification, verify heads are present
    if isinstance(output, dict):
        for key in output:
            _check(output[key].shape[0] == 1, f"Batch dim mismatch for head '{key}'")
    checks_passed += 1
    print("  -> Multitask checks PASSED")

    # ===== 7. PERFORMANCE CHECKS =====
    print("[7/8] Performance checks...")
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(raman_tensor)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000

    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Inference latency: {latency_ms:.2f} ms")
    print(f"  Model size: {model_size:.3f} MB")
    _check(latency_ms < 10000, f"Inference latency {latency_ms:.0f} ms is unreasonably high")
    checks_passed += 1
    print("  -> Performance checks PASSED")

    # ===== 8. SCIENTIFIC INTEGRITY CHECKS =====
    print("[8/8] Scientific integrity checks...")
    if model_card_path and os.path.isfile(model_card_path):
        with open(model_card_path) as f:
            card = json.load(f)
        limitations = card.get("limitations", [])
        _check(len(limitations) > 0, "Model card has no listed limitations")
        has_screening = any("screening" in l.lower() for l in limitations)
        _check(has_screening, "Model card must include 'screening-level' disclaimer")

    # Device honesty check
    if benchmark_device:
        import platform
        machine = platform.machine().lower()
        if benchmark_device.lower() in ["raspberry pi", "jetson"]:
            if "arm" not in machine and "aarch64" not in machine:
                _check(
                    False,
                    f"Claiming '{benchmark_device}' deployment but running on '{machine}'. "
                    "Report as 'Laptop/Workstation CPU benchmark' instead."
                )
    checks_passed += 1
    print("  -> Scientific integrity checks PASSED")

    return checks_passed


def main():
    parser = argparse.ArgumentParser(description="Spec2Prop-Edge: Deployment Integrity Validation")
    parser.add_argument("--model", default="deploy/exported/family_litespecnet_torchscript.pt")
    parser.add_argument("--label-encoder", default="deploy/exported/label_encoders.json")
    parser.add_argument("--sample", default="deploy/sample_inputs/example_raman.csv")
    parser.add_argument("--config", default="deploy/exported/deployment_config.json")
    parser.add_argument("--model-card", default="deploy/exported/model_card.json")
    parser.add_argument("--task", default="family")
    parser.add_argument("--benchmark-device", default=None,
                        help="If set, validates the device claim is honest.")
    args = parser.parse_args()

    print("=" * 60)
    print("  Spec2Prop-Edge: Deployment Integrity Validation")
    print("=" * 60)

    try:
        checks = validate(
            model_path=args.model,
            label_encoder_path=args.label_encoder,
            sample_raman_path=args.sample,
            deployment_config_path=args.config if os.path.isfile(args.config) else None,
            model_card_path=args.model_card if os.path.isfile(args.model_card) else None,
            task=args.task,
            benchmark_device=args.benchmark_device,
        )
        print("\n" + "=" * 60)
        print(f"  DEPLOYMENT INTEGRITY: PASSED ({checks}/8 categories)")
        print("=" * 60)
        sys.exit(0)

    except AssertionError as e:
        print(f"\n{'=' * 60}")
        print(f"  DEPLOYMENT INTEGRITY: FAILED")
        print(f"  Reason: {e}")
        print(f"{'=' * 60}")
        sys.exit(1)

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"  DEPLOYMENT INTEGRITY: ERROR")
        print(f"  {type(e).__name__}: {e}")
        print(f"{'=' * 60}")
        sys.exit(2)


if __name__ == "__main__":
    main()
