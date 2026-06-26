import pytest
import torch
import torch.nn as nn
from models.cnn1d import SimpleCNN1D

@pytest.mark.functional
def test_family_loss_calculation():
    """Verify standard class-weighted CrossEntropy works correctly."""
    model = SimpleCNN1D(1, 10)
    # Give some dummy class weights
    weights = torch.tensor([1.0, 2.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    x = torch.randn(4, 1, 2048)
    y = torch.tensor([0, 1, 2, 3]) # Targets
    
    logits = model(x)
    loss = criterion(logits, y)
    
    assert loss.item() > 0
    assert not torch.isnan(loss)
    
    # Backward should work
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None
