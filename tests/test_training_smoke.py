import os
import pytest
import torch

@pytest.mark.smoke
def test_imports():
    """Verify core modules can be imported without syntax errors."""
    import models.dataset
    import models.cnn1d
    import models.descriptor_mlp
    import models.fusion_models
    import models.multimodal_cnn
    import models.baselines

@pytest.mark.smoke
def test_dataset_files_exist(data_dir):
    """Verify key dataset files exist."""
    files = [
        "spec2prop_clean_inorganic.pkl",
        "spec2prop_property_matched_inorganic.pkl",
        "spec2prop_xrd_linked_inorganic.pkl",
        "spec2prop_raman.parquet"
    ]
    for f in files:
        assert os.path.exists(os.path.join(data_dir, f)), f"Missing dataset file: {f}"

@pytest.mark.smoke
def test_tiny_training_step():
    """Verify a tiny forward+backward+optimizer step works."""
    from models.cnn1d import SimpleCNN1D
    import torch.nn as nn
    
    model = SimpleCNN1D(1, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    x = torch.randn(4, 1, 2048) # Batch of 4, 1 channel, 2048 length
    y = torch.randint(0, 10, (4,))
    
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    
    assert loss.item() > 0, "Loss should be > 0"
    assert logits.shape == (4, 10), "Logits shape mismatch"
    assert not torch.isnan(loss), "Loss is NaN"
