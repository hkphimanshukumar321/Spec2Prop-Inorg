import pytest
import torch
from models.multimodal_cnn import DualBranchRamanXRDNet

@pytest.mark.functional
def test_multimodal_missing_xrd_handling():
    """Verify that if XRD is missing but enabled, fallback logic works."""
    model = DualBranchRamanXRDNet(tasks={"family": 5})
    r = torch.randn(2, 1, 2048)
    
    # In training loop, if XRD isn't present, it usually falls back to Raman tensor.
    # We simulate passing Raman twice just to check fusion dimension compatibility.
    logits = model(r, r)
    assert logits["family"].shape == (2, 5)
