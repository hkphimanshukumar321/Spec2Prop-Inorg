import pytest
import torch
from models.cnn1d import SimpleCNN1D, LiteSpecNet, ResidualCNN1D
from models.descriptor_mlp import DescriptorMLP
from models.fusion_models import FusionSpec2PropNet, Spec2PropLite
from models.multimodal_cnn import DualBranchRamanXRDNet

@pytest.mark.functional
@pytest.mark.parametrize("model_class", [SimpleCNN1D, LiteSpecNet, ResidualCNN1D])
def test_1d_cnn_forward(model_class):
    """Test standard 1D CNNs."""
    model = model_class(in_channels=1, num_classes=10)
    x = torch.randn(8, 1, 2048)
    out = model(x)
    assert out.shape == (8, 10)

@pytest.mark.functional
def test_descriptor_mlp_forward():
    """Test descriptor MLP."""
    model = DescriptorMLP(input_dim=32, embed_dim=16)
    x = torch.randn(8, 32)
    out = model(x)
    assert out.shape == (8, 16) # embed dim output, not classes

@pytest.mark.functional
def test_fusion_net_forward():
    """Test multi-task fusion net."""
    tasks = {"task_a": 3, "task_b": 2}
    model = FusionSpec2PropNet(tasks=tasks, descriptor_dim=32)
    x_raman = torch.randn(8, 1, 2048)
    x_desc = torch.randn(8, 32)
    
    out = model(x_raman, x_desc)
    assert isinstance(out, dict)
    assert out["task_a"].shape == (8, 3)
    assert out["task_b"].shape == (8, 2)

@pytest.mark.functional
def test_multimodal_cnn_forward():
    """Test dual-branch Raman+XRD net."""
    tasks = {"family": 12}
    model = DualBranchRamanXRDNet(tasks=tasks)
    x_raman = torch.randn(4, 1, 2048)
    x_xrd = torch.randn(4, 1, 2048)
    
    out = model(x_raman, x_xrd)
    assert isinstance(out, dict)
    assert out["family"].shape == (4, 12)
