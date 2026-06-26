"""
Spec2Prop-Edge: Deployment Pipeline Tests
==========================================
Smoke, functional, and integration tests for the deployment system.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_RAMAN = os.path.join(PROJECT_ROOT, "deploy", "sample_inputs", "example_raman.csv")
SAMPLE_XRD = os.path.join(PROJECT_ROOT, "deploy", "sample_inputs", "example_xrd.csv")
SAMPLE_DESC = os.path.join(PROJECT_ROOT, "deploy", "sample_inputs", "example_descriptors.json")
EXPORTED_DIR = os.path.join(PROJECT_ROOT, "deploy", "exported")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "deploy", "outputs")


def _has_exported_model():
    """Check if any TorchScript model exists in the exported directory."""
    if not os.path.isdir(EXPORTED_DIR):
        return False
    return any(f.endswith("_torchscript.pt") for f in os.listdir(EXPORTED_DIR))


def _get_exported_model():
    """Return path to first available exported model."""
    for f in os.listdir(EXPORTED_DIR):
        if f.endswith("_torchscript.pt"):
            return os.path.join(EXPORTED_DIR, f)
    return None


# =====================================================================
# SMOKE TESTS
# =====================================================================

class TestSmoke:
    """Basic import and sanity checks."""

    def test_import_preprocess(self):
        from deploy.preprocess_runtime import preprocess_raman_file, preprocess_xrd_file
        assert callable(preprocess_raman_file)
        assert callable(preprocess_xrd_file)

    def test_import_inference(self):
        from deploy.infer_single_sample import load_model, decode_predictions
        assert callable(load_model)
        assert callable(decode_predictions)

    def test_import_report(self):
        from deploy.generate_prediction_report import generate_report
        assert callable(generate_report)

    def test_import_saliency(self):
        from deploy.saliency_runtime import compute_gradient_saliency, plot_saliency
        assert callable(compute_gradient_saliency)

    def test_import_benchmark(self):
        from deploy.edge_benchmark import benchmark
        assert callable(benchmark)

    def test_import_integrity(self):
        from deploy.validate_deployment_integrity import validate
        assert callable(validate)

    def test_sample_raman_exists(self):
        assert os.path.isfile(SAMPLE_RAMAN), f"Missing: {SAMPLE_RAMAN}"

    def test_sample_xrd_exists(self):
        assert os.path.isfile(SAMPLE_XRD), f"Missing: {SAMPLE_XRD}"


# =====================================================================
# PREPROCESSING TESTS
# =====================================================================

class TestPreprocessing:
    """Test the runtime preprocessing module."""

    def test_raman_preprocessing_output_shape(self):
        from deploy.preprocess_runtime import preprocess_raman_file
        vector, meta = preprocess_raman_file(SAMPLE_RAMAN)
        assert vector.shape == (2048,), f"Expected (2048,), got {vector.shape}"

    def test_raman_preprocessing_no_nan(self):
        from deploy.preprocess_runtime import preprocess_raman_file
        vector, _ = preprocess_raman_file(SAMPLE_RAMAN)
        assert not np.any(np.isnan(vector))

    def test_raman_preprocessing_finite(self):
        from deploy.preprocess_runtime import preprocess_raman_file
        vector, _ = preprocess_raman_file(SAMPLE_RAMAN)
        assert np.all(np.isfinite(vector))

    def test_raman_preprocessing_range(self):
        from deploy.preprocess_runtime import preprocess_raman_file
        vector, _ = preprocess_raman_file(SAMPLE_RAMAN)
        assert np.max(vector) <= 1.5
        assert np.min(vector) >= -0.1

    def test_raman_metadata_keys(self):
        from deploy.preprocess_runtime import preprocess_raman_file
        _, meta = preprocess_raman_file(SAMPLE_RAMAN)
        assert "input_file" in meta
        assert "preprocessing" in meta
        assert "n_points" in meta["preprocessing"]

    def test_xrd_preprocessing_output_shape(self):
        from deploy.preprocess_runtime import preprocess_xrd_file
        vector, meta = preprocess_xrd_file(SAMPLE_XRD)
        assert vector.shape == (2048,)

    def test_descriptor_vector(self):
        from deploy.preprocess_runtime import build_descriptor_vector
        with open(SAMPLE_DESC) as f:
            desc = json.load(f)
        vec = build_descriptor_vector(desc)
        assert vec.dtype == np.float32
        assert len(vec) == 7  # 4 bool + 3 float, no family encoder

    def test_descriptor_vector_with_encoder(self):
        from deploy.preprocess_runtime import build_descriptor_vector
        with open(SAMPLE_DESC) as f:
            desc = json.load(f)
        fake_encoder = {"Oxide": 0, "Sulfide": 1, "Other/Rare": 2}
        vec = build_descriptor_vector(desc, family_encoder=fake_encoder)
        assert len(vec) == 7 + 3  # 7 base + 3 one-hot

    def test_dummy_model_forward(self):
        """Test that a dummy CNN can process a 2048-point vector."""
        from models.cnn1d import LiteSpecNet
        model = LiteSpecNet(in_channels=1, num_classes=5)
        model.eval()
        dummy = torch.randn(1, 1, 2048)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 5)


# =====================================================================
# REPORT TESTS
# =====================================================================

class TestReport:
    """Test prediction report generation."""

    def test_generate_report(self, tmp_path):
        from deploy.generate_prediction_report import generate_report
        predictions = {
            "sample_id": "test_sample",
            "task": "family",
            "model_name": "test_model",
            "top_predictions": [
                {"rank": 1, "class": "Oxide", "confidence": 0.85},
                {"rank": 2, "class": "Sulfide", "confidence": 0.10},
            ],
            "inference_time_ms": 5.2,
            "device": "cpu",
            "warnings": [],
        }
        paths = generate_report(predictions, str(tmp_path), "test_report")
        assert os.path.isfile(paths["json"])
        assert os.path.isfile(paths["csv"])
        assert os.path.isfile(paths["md"])

        with open(paths["json"]) as f:
            data = json.load(f)
        assert "disclaimer" in data


# =====================================================================
# INTEGRATION TESTS (require exported model)
# =====================================================================

class TestIntegration:
    """Integration tests that require an exported model."""

    @pytest.mark.skipif(not _has_exported_model(),
                        reason="Trained checkpoint not found. Run training/export first.")
    def test_single_sample_inference(self):
        from deploy.infer_single_sample import load_model, run_inference_torchscript, decode_predictions
        from deploy.preprocess_runtime import preprocess_raman_file

        model_path = _get_exported_model()
        model = load_model(model_path, "torchscript")
        vector, _ = preprocess_raman_file(SAMPLE_RAMAN)
        tensor = torch.from_numpy(vector).float().unsqueeze(0).unsqueeze(0)
        results = run_inference_torchscript(model, tensor)
        probs = list(results.values())[0]

        assert abs(np.sum(probs) - 1.0) < 0.01
        assert len(probs) > 0

    @pytest.mark.skipif(not _has_exported_model(),
                        reason="Trained checkpoint not found. Run training/export first.")
    def test_edge_benchmark(self):
        from deploy.edge_benchmark import benchmark

        model_path = _get_exported_model()
        results = benchmark(model_path, SAMPLE_RAMAN, num_runs=10, warmup_runs=2)
        assert "inference_time_ms_mean" in results
        assert results["inference_time_ms_mean"] > 0
        assert "throughput_samples_per_sec" in results
