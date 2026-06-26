import subprocess
import os
import pytest

SCRIPTS = [
    "scripts/train_family_cnn.py",
    "scripts/train_property_cnn.py",
    "scripts/train_multimodal_raman_xrd.py",
    "scripts/train_baselines.py",
    "scripts/evaluate_models.py",
    "scripts/plot_results.py",
    "scripts/export_edge_models.py",
    "scripts/summarize_experiments.py"
]

@pytest.mark.smoke
@pytest.mark.parametrize("script_path", SCRIPTS)
def test_script_help(project_root, script_path):
    """Test that all CLI scripts can be invoked with --help without crashing."""
    full_path = os.path.join(project_root, script_path)
    if not os.path.exists(full_path):
        pytest.skip(f"Script not found: {full_path}")
        
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root
    result = subprocess.run(["python", full_path, "--help"], capture_output=True, text=True, env=env)
    assert result.returncode == 0, f"Script {script_path} failed on --help. Stderr: {result.stderr}"

@pytest.mark.smoke
@pytest.mark.dataset
def test_check_dataset_integrity(project_root, data_dir):
    """Test the dataset integrity checker."""
    script_path = os.path.join(project_root, "scripts", "check_dataset_integrity.py")
    if not os.path.exists(script_path):
        pytest.skip("check_dataset_integrity.py not found.")
        
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root
    result = subprocess.run(["python", script_path, "--processed-dir", data_dir], capture_output=True, text=True, env=env)
    assert result.returncode == 0, f"Integrity check failed. Stderr: {result.stderr}"
    assert "PASSED" in result.stdout or "OK" in result.stdout, "Integrity check did not output PASSED or OK."
