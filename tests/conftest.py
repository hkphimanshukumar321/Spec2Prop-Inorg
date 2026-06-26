import os
import pytest
import yaml
import tempfile
import json
import torch
import numpy as np
import pandas as pd

@pytest.fixture(scope="session")
def project_root():
    """Returns the absolute path to the project root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@pytest.fixture(scope="session")
def data_dir(project_root):
    """Returns the processed data directory, or skips tests if missing."""
    path = os.path.join(project_root, "data", "processed")
    if not os.path.exists(path) or not os.path.exists(os.path.join(path, "spec2prop_clean_inorganic.pkl")):
        pytest.skip("Processed dataset not found. Run preprocessing/build_training_cache first.")
    return path

@pytest.fixture(scope="session")
def cache_dir(data_dir):
    """Returns the cache directory, or skips tests if missing."""
    path = os.path.join(data_dir, "cache")
    if not os.path.exists(path) or not os.path.exists(os.path.join(path, "clean_inorganic", "train", "meta.pkl")):
        pytest.skip("Training cache not found. Run scripts/build_training_cache.py first.")
    return path

@pytest.fixture(scope="session")
def default_configs(project_root):
    """Loads all default configs."""
    configs = {}
    config_dir = os.path.join(project_root, "configs")
    if not os.path.exists(config_dir):
        pytest.skip("Configs directory not found.")
        
    for name in ["family_classification", "property_prediction", "multimodal_raman_xrd", "baselines"]:
        path = os.path.join(config_dir, f"{name}.yaml")
        if os.path.exists(path):
            with open(path) as f:
                configs[name] = yaml.safe_load(f)
        else:
            pytest.skip(f"Config missing: {path}")
            
    return configs

@pytest.fixture
def mini_config_generator(tmp_path, default_configs):
    """Returns a function to generate minimal configs for integration tests."""
    def _generate(task_name, overrides=None):
        cfg = default_configs[task_name].copy()
        
        # Default overrides for speed
        cfg["training"]["epochs"] = 1
        cfg["training"]["batch_size"] = 8
        cfg["output"]["results_dir"] = str(tmp_path / f"results_{task_name}")
        
        # Limit models for speed
        if "models" in cfg:
            cfg["models"] = [cfg["models"][0]] # Just test the first model
            
        if overrides:
            for k, v in overrides.items():
                if k in cfg["training"]:
                    cfg["training"][k] = v
                elif k in cfg["output"]:
                    cfg["output"][k] = v
                    
        out_path = tmp_path / f"{task_name}_mini.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f)
            
        return str(out_path), cfg["output"]["results_dir"]
        
    return _generate
