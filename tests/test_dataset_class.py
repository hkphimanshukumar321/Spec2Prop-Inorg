import os
import pytest
import torch
from models.dataset import Spec2PropDataset, build_label_encoder

@pytest.mark.functional
@pytest.mark.dataset
def test_dataset_family_task(data_dir, cache_dir):
    """Test Spec2PropDataset for the family classification task."""
    meta_pkl = os.path.join(data_dir, "spec2prop_clean_inorganic.pkl")
    raman_pqt = os.path.join(data_dir, "spec2prop_raman.parquet")
    split_csv = os.path.join(data_dir, "splits", "clean_inorganic", "train.csv")
    
    import pandas as pd
    from models.dataset import add_model_family_column
    meta_df = pd.read_pickle(meta_pkl)
    meta_df = add_model_family_column(meta_df)
    
    encoder = build_label_encoder(meta_df["chemical_family_model"])
    
    ds = Spec2PropDataset(
        metadata_pkl=meta_pkl,
        raman_parquet=raman_pqt,
        split_csv=split_csv,
        target_cols=["chemical_family_model"],
        label_encoders={"chemical_family_model": encoder},
        cache_dir=os.path.join(cache_dir, "clean_inorganic")
    )
    
    # Check length
    assert len(ds) > 0
    
    # Check item
    item = ds[0]
    assert "raman" in item
    assert "target_chemical_family_model" in item
    assert "mask_chemical_family_model" in item
    
    # Check shapes
    assert item["raman"].shape == (1, 2048)
    assert item["mask_chemical_family_model"] == 1.0 # Should be present
    
    # Check rare mapping logic
    mapped_families = meta_df["chemical_family_model"].unique()
    assert "Carbide" not in mapped_families
    assert "Other/Rare" in mapped_families

@pytest.mark.functional
def test_dataset_property_task(data_dir, cache_dir):
    """Test dataset for multi-task property prediction."""
    meta_pkl = os.path.join(data_dir, "spec2prop_property_matched_inorganic.pkl")
    raman_pqt = os.path.join(data_dir, "spec2prop_raman.parquet")
    split_csv = os.path.join(data_dir, "splits", "property_matched_inorganic", "train.csv")
    
    import pandas as pd
    meta_df = pd.read_pickle(meta_pkl)
    
    targets = ["band_gap_class", "is_metal", "formation_energy_class"]
    encoders = {t: build_label_encoder(meta_df[t]) for t in targets}
    
    ds = Spec2PropDataset(
        metadata_pkl=meta_pkl,
        raman_parquet=raman_pqt,
        split_csv=split_csv,
        target_cols=targets,
        label_encoders=encoders,
        cache_dir=os.path.join(cache_dir, "property_matched")
    )
    
    item = ds[0]
    for t in targets:
        assert f"target_{t}" in item
        assert f"mask_{t}" in item
