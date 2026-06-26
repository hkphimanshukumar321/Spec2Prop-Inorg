import os
import pytest
import pandas as pd

SPLIT_DIRS = [
    "clean_inorganic",
    "property_matched_inorganic",
    "xrd_linked_inorganic"
]

@pytest.mark.dataset
@pytest.mark.parametrize("split_name", SPLIT_DIRS)
def test_data_leakage(data_dir, split_name):
    """Check for sample_id and rruff_id leakage between train/val/test splits."""
    d = os.path.join(data_dir, "splits", split_name)
    if not os.path.exists(d):
        pytest.skip(f"Split dir not found: {d}")
        
    train_file = os.path.join(d, "train.csv")
    val_file = os.path.join(d, "val.csv")
    test_file = os.path.join(d, "test.csv")
    
    assert os.path.exists(train_file), "train.csv missing"
    assert os.path.exists(val_file), "val.csv missing"
    assert os.path.exists(test_file), "test.csv missing"
    
    train = pd.read_csv(train_file)
    val = pd.read_csv(val_file)
    test = pd.read_csv(test_file)
    
    assert len(train) > 0, "Train split empty"
    assert len(val) > 0, "Val split empty"
    assert len(test) > 0, "Test split empty"
    
    # 1. Sample ID Leakage
    train_ids = set(train["sample_id"].astype(str))
    val_ids = set(val["sample_id"].astype(str))
    test_ids = set(test["sample_id"].astype(str))
    
    assert len(train_ids.intersection(val_ids)) == 0, f"Leakage: train/val sample_ids overlap in {split_name}"
    assert len(train_ids.intersection(test_ids)) == 0, f"Leakage: train/test sample_ids overlap in {split_name}"
    assert len(val_ids.intersection(test_ids)) == 0, f"Leakage: val/test sample_ids overlap in {split_name}"
    
    # 2. RRUFF ID Leakage (Group Leakage)
    if "rruff_id" in train.columns:
        train_rruff = set(train["rruff_id"].astype(str).dropna())
        val_rruff = set(val["rruff_id"].astype(str).dropna())
        test_rruff = set(test["rruff_id"].astype(str).dropna())
        
        # Remove "nan" if it got parsed as a string
        train_rruff.discard("nan")
        val_rruff.discard("nan")
        test_rruff.discard("nan")
        
        assert len(train_rruff.intersection(val_rruff)) == 0, f"Leakage: train/val rruff_ids overlap in {split_name}"
        assert len(train_rruff.intersection(test_rruff)) == 0, f"Leakage: train/test rruff_ids overlap in {split_name}"
        assert len(val_rruff.intersection(test_rruff)) == 0, f"Leakage: val/test rruff_ids overlap in {split_name}"
