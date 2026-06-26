import os
import pytest
import json

@pytest.mark.integration
@pytest.mark.dataset
def test_family_integration_mini_run(mini_config_generator):
    """Run a tiny end-to-end training cycle for family classification."""
    cfg_path, out_dir = mini_config_generator("family_classification")
    
    # Run main script
    from scripts.train_family_cnn import main
    try:
        main(cfg_path)
    except Exception as e:
        pytest.fail(f"train_family_cnn.py main() crashed: {e}")
        
    # Verify outputs
    assert os.path.exists(out_dir)
    assert os.path.exists(os.path.join(out_dir, "metrics.json"))
    
    with open(os.path.join(out_dir, "metrics.json")) as f:
        metrics = json.load(f)
        assert len(metrics) > 0

@pytest.mark.integration
@pytest.mark.dataset
def test_property_integration_mini_run(mini_config_generator):
    """Run a tiny end-to-end training cycle for property prediction."""
    cfg_path, out_dir = mini_config_generator("property_prediction")
    
    from scripts.train_property_cnn import main
    try:
        main(cfg_path)
    except Exception as e:
        pytest.fail(f"train_property_cnn.py main() crashed: {e}")
        
    assert os.path.exists(os.path.join(out_dir, "metrics.json"))

@pytest.mark.integration
@pytest.mark.dataset
def test_multimodal_integration_mini_run(mini_config_generator):
    """Run a tiny end-to-end training cycle for multimodal Raman+XRD."""
    cfg_path, out_dir = mini_config_generator("multimodal_raman_xrd")
    
    from scripts.train_multimodal_raman_xrd import main
    try:
        main(cfg_path)
    except Exception as e:
        pytest.fail(f"train_multimodal_raman_xrd.py main() crashed: {e}")
        
    assert os.path.exists(os.path.join(out_dir, "metrics.json"))

@pytest.mark.integration
def test_edge_export_integration(tmp_path):
    """Verify exporting works (doesn't require dataset, just instantiates model)."""
    from scripts.export_edge_models import export_model
    
    out_dir = str(tmp_path / "edge")
    # Provide a dummy non-existent path to force it to use random weights
    export_model("dummy_non_existent.pt", out_dir)
    
    assert os.path.exists(os.path.join(out_dir, "litespecnet_torchscript.pt"))
    assert os.path.exists(os.path.join(out_dir, "edge_benchmark.json"))
    
    with open(os.path.join(out_dir, "edge_benchmark.json")) as f:
        bench = json.load(f)
        assert "avg_latency_ms" in bench
        assert bench["avg_latency_ms"] > 0
