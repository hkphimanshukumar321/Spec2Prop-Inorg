"""
Spec2Prop-Edge: Deployment Integrity Tests
===========================================
Tests the integrity validator itself.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest
import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_RAMAN = os.path.join(PROJECT_ROOT, "deploy", "sample_inputs", "example_raman.csv")
EXPORTED_DIR = os.path.join(PROJECT_ROOT, "deploy", "exported")


def _has_exported_model():
    if not os.path.isdir(EXPORTED_DIR):
        return False
    return any(f.endswith("_torchscript.pt") for f in os.listdir(EXPORTED_DIR))


def _get_exported_paths():
    model = encoder = config = card = None
    if os.path.isdir(EXPORTED_DIR):
        for f in os.listdir(EXPORTED_DIR):
            path = os.path.join(EXPORTED_DIR, f)
            if f.endswith("_torchscript.pt"):
                model = path
            elif f == "label_encoders.json":
                encoder = path
            elif f == "deployment_config.json":
                config = path
            elif f == "model_card.json":
                card = path
    return model, encoder, config, card


class TestIntegrityValidator:
    """Test the deployment integrity validation module."""

    def test_missing_model_file(self):
        from deploy.validate_deployment_integrity import validate
        with pytest.raises(AssertionError, match="Model file not found"):
            validate(
                model_path="nonexistent_model.pt",
                label_encoder_path="nonexistent.json",
                sample_raman_path=SAMPLE_RAMAN,
            )

    def test_missing_label_encoder(self, tmp_path):
        """Create a dummy torchscript model, then fail on missing encoder."""
        from deploy.validate_deployment_integrity import validate
        from models.cnn1d import LiteSpecNet

        model = LiteSpecNet(in_channels=1, num_classes=5)
        model.eval()
        dummy = torch.randn(1, 1, 2048)
        ts = torch.jit.trace(model, dummy)
        model_path = str(tmp_path / "test_model.pt")
        ts.save(model_path)

        with pytest.raises(AssertionError, match="Label encoder not found"):
            validate(
                model_path=model_path,
                label_encoder_path="nonexistent_encoder.json",
                sample_raman_path=SAMPLE_RAMAN,
            )

    def test_label_encoder_size_mismatch(self, tmp_path):
        """Model outputs 5 classes but encoder has 3 — should fail."""
        from deploy.validate_deployment_integrity import validate
        from models.cnn1d import LiteSpecNet

        model = LiteSpecNet(in_channels=1, num_classes=5)
        model.eval()
        dummy = torch.randn(1, 1, 2048)
        ts = torch.jit.trace(model, dummy)
        model_path = str(tmp_path / "test_model.pt")
        ts.save(model_path)

        enc = {"A": 0, "B": 1, "C": 2}  # 3 classes, model expects 5
        enc_path = str(tmp_path / "bad_encoder.json")
        with open(enc_path, "w") as f:
            json.dump(enc, f)

        with pytest.raises(AssertionError, match="output dim.*label encoder size"):
            validate(
                model_path=model_path,
                label_encoder_path=enc_path,
                sample_raman_path=SAMPLE_RAMAN,
            )

    def test_valid_setup_passes(self, tmp_path):
        """Create a matching model + encoder and verify it passes."""
        from deploy.validate_deployment_integrity import validate
        from models.cnn1d import LiteSpecNet

        num_classes = 5
        model = LiteSpecNet(in_channels=1, num_classes=num_classes)
        model.eval()
        dummy = torch.randn(1, 1, 2048)
        ts = torch.jit.trace(model, dummy)
        model_path = str(tmp_path / "test_model.pt")
        ts.save(model_path)

        enc = {f"Class_{i}": i for i in range(num_classes)}
        enc_path = str(tmp_path / "encoder.json")
        with open(enc_path, "w") as f:
            json.dump(enc, f)

        checks = validate(
            model_path=model_path,
            label_encoder_path=enc_path,
            sample_raman_path=SAMPLE_RAMAN,
        )
        assert checks == 8  # All 8 categories should pass

    @pytest.mark.skipif(not _has_exported_model(),
                        reason="No exported model found. Run export first.")
    def test_full_integrity_with_exported_model(self):
        from deploy.validate_deployment_integrity import validate

        model, encoder, config, card = _get_exported_paths()
        assert model is not None
        assert encoder is not None

        checks = validate(
            model_path=model,
            label_encoder_path=encoder,
            sample_raman_path=SAMPLE_RAMAN,
            deployment_config_path=config,
            model_card_path=card,
        )
        assert checks == 8
