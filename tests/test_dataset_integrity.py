import os
import pytest
import pandas as pd
import numpy as np

@pytest.mark.dataset
def test_clean_inorganic_integrity(data_dir):
    """Test clean inorganic dataset rules."""
    df = pd.read_pickle(os.path.join(data_dir, "spec2prop_clean_inorganic.pkl"))
    
    assert len(df) > 0
    assert df["sample_id"].is_unique, "sample_id is not unique"
    
    from models.dataset import add_model_family_column
    df = add_model_family_column(df)
    
    # Exclude organic
    if "chemical_family_original" in df.columns:
        assert not any(df["chemical_family_original"].str.contains("Organic", na=False)), "Organics found"
        
    assert "chemical_family_model" in df.columns or "chemical_family_original" in df.columns
    assert "raman_length" not in df.columns or (df["raman_length"] == 2048).all()

@pytest.mark.dataset
def test_property_matched_integrity(data_dir):
    """Test property matched rules."""
    df = pd.read_pickle(os.path.join(data_dir, "spec2prop_property_matched_inorganic.pkl"))
    
    assert len(df) > 0
    assert df["sample_id"].is_unique, "sample_id is not unique"
    
    assert df[["band_gap_class", "is_metal", "formation_energy_class"]].notna().any(axis=1).all(), "Some rows have no properties at all"

@pytest.mark.dataset
def test_xrd_linked_integrity(data_dir):
    """Test XRD dataset rules."""
    df = pd.read_pickle(os.path.join(data_dir, "spec2prop_xrd_linked_inorganic.pkl"))
    
    assert len(df) > 0
    assert df["sample_id"].is_unique, "sample_id is not unique"
    
    xrd_cols = [c for c in df.columns if c.startswith("xrd_") and c[4:].isdigit()]
    assert len(xrd_cols) == 2048, "XRD vector length is not 2048"

@pytest.mark.dataset
def test_spectral_vectors_parquet(data_dir):
    """Test actual Raman parquet data."""
    df = pd.read_parquet(os.path.join(data_dir, "spec2prop_raman.parquet"))
    
    assert len(df) > 0
    
    raman_cols = [c for c in df.columns if c.startswith("raman_") and c[6:].isdigit()]
    assert len(raman_cols) == 2048, "Raman vector length is not 2048"
    
    # Check bounds (roughly)
    # Take a small sample to avoid huge memory test
    sample = df.head(100)[raman_cols].values
    assert np.isfinite(sample).all(), "Non-finite values found in Raman spectra"
    # Assuming normalized
    assert sample.max() <= 1.01, "Values seem not max-normalized"
