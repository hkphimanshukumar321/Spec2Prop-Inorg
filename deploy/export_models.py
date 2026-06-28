"""
Spec2Prop-Edge: Model Export
=============================
Export trained PyTorch models to TorchScript and ONNX for deployment.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import shutil
import warnings
from datetime import datetime

import torch
import numpy as np

from models.cnn1d import SimpleCNN1D, LiteSpecNet, ResidualCNN1D, MultiScaleSpecNet, count_parameters, model_size_mb
from models.fusion_models import FusionSpec2PropNet, Spec2PropLite
from models.multimodal_cnn import DualBranchRamanXRDNet


MODEL_REGISTRY = {
    "SimpleCNN1D": SimpleCNN1D,
    "LiteSpecNet": LiteSpecNet,
    "ResidualCNN1D": ResidualCNN1D,
    "MultiScaleSpecNet": MultiScaleSpecNet,
}


def _load_label_encoder(path):
    """Load a label encoder JSON file."""
    with open(path) as f:
        return json.load(f)


def _build_model(model_class_name, num_classes, task="family", descriptor_dim=0):
    """Instantiate a model by class name."""
    if model_class_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_class_name](in_channels=1, num_classes=num_classes)
    elif model_class_name == "FusionSpec2PropNet":
        tasks = {"output": num_classes}  # simplified for export
        return FusionSpec2PropNet(
            tasks=tasks,
            descriptor_dim=descriptor_dim,
        )
    elif model_class_name == "Spec2PropLite":
        tasks = {"output": num_classes}
        return Spec2PropLite(
            tasks=tasks,
            descriptor_dim=descriptor_dim,
        )
    elif model_class_name == "DualBranchRamanXRDNet":
        tasks = {"output": num_classes}
        return DualBranchRamanXRDNet(
            tasks=tasks,
            descriptor_dim=descriptor_dim,
        )
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")


def export_family_model(
    checkpoint_path: str,
    model_class_name: str,
    label_encoder_path: str,
    output_dir: str,
    n_points: int = 2048,
):
    """Export a family classification model."""
    os.makedirs(output_dir, exist_ok=True)

    # Load label encoder
    label_encoder = _load_label_encoder(label_encoder_path)
    num_classes = len(label_encoder)
    inv_encoder = {v: k for k, v in label_encoder.items()}

    # Build and load model
    model = MODEL_REGISTRY[model_class_name](in_channels=1, num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # TorchScript export
    dummy_input = torch.randn(1, 1, n_points)
    ts_model = torch.jit.trace(model, dummy_input)
    ts_path = os.path.join(output_dir, f"family_{model_class_name.lower()}_torchscript.pt")
    ts_model.save(ts_path)
    print(f"TorchScript saved: {ts_path}")

    # ONNX export (optional)
    onnx_path = os.path.join(output_dir, f"family_{model_class_name.lower()}.onnx")
    try:
        import onnx
        torch.onnx.export(
            model, dummy_input, onnx_path,
            input_names=["raman"],
            output_names=["logits"],
            dynamic_axes={"raman": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=13,
        )
        print(f"ONNX saved: {onnx_path}")
    except ImportError:
        warnings.warn("onnx not installed — skipping ONNX export.")
    except Exception as e:
        warnings.warn(f"ONNX export failed: {e}")

    # Copy label encoder
    enc_path = os.path.join(output_dir, "label_encoders.json")
    with open(enc_path, "w") as f:
        json.dump({"chemical_family_model": label_encoder}, f, indent=2)
    print(f"Label encoder saved: {enc_path}")

    # Model card
    model_card = {
        "model_name": model_class_name,
        "task": "chemical_family_classification",
        "input_shape": [1, 1, n_points],
        "output_heads": ["chemical_family_model"],
        "num_classes": num_classes,
        "class_labels": inv_encoder,
        "training_subset": "spec2prop_clean_inorganic.pkl (4027 samples)",
        "preprocessing": {
            "wn_min": 100, "wn_max": 4000, "n_points": n_points,
            "baseline_correction": True, "smoothing": True,
            "normalization": "max"
        },
        "parameter_count": count_parameters(model),
        "model_size_mb": round(model_size_mb(model), 3),
        "export_date": datetime.now().isoformat(),
        "limitations": [
            "Trained on RRUFF Raman spectra only",
            "Screening-level prediction, not experimental confirmation",
            "Performance may degrade on spectra from different instruments or preprocessing",
            "This is a screening-level decision-support tool, not a final identification method."
        ]
    }
    card_path = os.path.join(output_dir, "model_card.json")
    with open(card_path, "w") as f:
        json.dump(model_card, f, indent=2)
    print(f"Model card saved: {card_path}")

    # Deployment config
    deploy_cfg = {
        "task": "family",
        "model_class": model_class_name,
        "model_file": os.path.basename(ts_path),
        "label_encoder_file": "label_encoders.json",
        "input_shape": [1, 1, n_points],
        "num_classes": num_classes,
        "device": "cpu",
    }
    cfg_path = os.path.join(output_dir, "deployment_config.json")
    with open(cfg_path, "w") as f:
        json.dump(deploy_cfg, f, indent=2)
    print(f"Deployment config saved: {cfg_path}")

    return ts_path


def main():
    parser = argparse.ArgumentParser(description="Export Spec2Prop models for deployment.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--model-class", required=True, help="Model class name")
    parser.add_argument("--label-encoder", required=True, help="Path to label encoder JSON")
    parser.add_argument("--task", default="family", choices=["family", "property", "multimodal"])
    parser.add_argument("--output-dir", default="deploy/exported")
    parser.add_argument("--n-points", type=int, default=2048)
    args = parser.parse_args()

    if args.task == "family":
        export_family_model(
            args.checkpoint, args.model_class,
            args.label_encoder, args.output_dir, args.n_points
        )
    else:
        print(f"Export for task '{args.task}' uses the same flow — adjust model instantiation as needed.")
        export_family_model(
            args.checkpoint, args.model_class,
            args.label_encoder, args.output_dir, args.n_points
        )


if __name__ == "__main__":
    main()
