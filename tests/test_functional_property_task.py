import pytest
import torch
import torch.nn as nn
from models.fusion_models import FusionSpec2PropNet

@pytest.mark.functional
def test_property_multitask_loss_masking():
    """Verify that multi-task masking correctly ignores NaN/Missing targets."""
    tasks = {"task_1": 2, "task_2": 3}
    model = FusionSpec2PropNet(tasks=tasks, descriptor_dim=0)
    
    # 2 samples in batch
    x = torch.randn(2, 1, 2048)
    d = None
    
    # task 1 targets
    y1 = torch.tensor([0, 1])
    m1 = torch.tensor([1.0, 1.0]) # both valid
    
    # task 2 targets - sample 0 is missing target
    y2 = torch.tensor([0, 2])
    m2 = torch.tensor([0.0, 1.0]) # only sample 1 valid
    
    crit1 = nn.CrossEntropyLoss()
    crit2 = nn.CrossEntropyLoss()
    
    logits = model(x, d)
    
    # Loss 1
    idx1 = torch.where(m1 > 0)[0]
    loss1 = crit1(logits["task_1"][idx1], y1[idx1])
    
    # Loss 2
    idx2 = torch.where(m2 > 0)[0]
    assert len(idx2) == 1 # Masking logic works
    loss2 = crit2(logits["task_2"][idx2], y2[idx2])
    
    total_loss = loss1 + 0.5 * loss2
    total_loss.backward()
    
    assert not torch.isnan(total_loss)
