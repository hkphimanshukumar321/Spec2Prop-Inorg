"""
Spec2Prop-Inorg: Fusion Models
==============================
Models that combine Raman spectral encoding with chemistry descriptor MLP.
"""

import torch
import torch.nn as nn
from models.cnn1d import SimpleCNN1D, LiteSpecNet
from models.descriptor_mlp import DescriptorMLP


class FusionMultiTaskHead(nn.Module):
    """
    Multi-task classification and regression heads.
    """
    def __init__(self, embed_dim: int, tasks: dict):
        """
        tasks: dict mapping task_name -> num_classes
               (num_classes = 1 for regression)
        """
        super().__init__()
        self.heads = nn.ModuleDict()
        for t_name, n_cls in tasks.items():
            self.heads[t_name] = nn.Linear(embed_dim, n_cls)

    def forward(self, x):
        return {t_name: head(x) for t_name, head in self.heads.items()}


class FusionSpec2PropNet(nn.Module):
    """
    Raman CNN Encoder + Descriptor MLP Encoder -> Fusion -> Multi-task Heads.
    """

    def __init__(
        self,
        tasks: dict,
        raman_channels: int = 1,
        raman_embed_dim: int = 128,
        descriptor_dim: int = 0,
        descriptor_embed_dim: int = 32,
        fusion_dim: int = 128,
        dropout: float = 0.3,
        cnn_type: str = "SimpleCNN1D"
    ):
        super().__init__()
        self.use_descriptors = descriptor_dim > 0

        # Spectral Encoder
        if cnn_type == "LiteSpecNet":
            self.raman_enc = LiteSpecNet(raman_channels, raman_embed_dim, dropout=dropout)
        else:
            self.raman_enc = SimpleCNN1D(raman_channels, raman_embed_dim, dropout=dropout)

        # Descriptor Encoder
        if self.use_descriptors:
            self.desc_enc = DescriptorMLP(
                input_dim=descriptor_dim, 
                embed_dim=descriptor_embed_dim, 
                dropout=dropout
            )
            concat_dim = raman_embed_dim + descriptor_embed_dim
        else:
            concat_dim = raman_embed_dim

        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Multi-task Head
        self.head = FusionMultiTaskHead(fusion_dim, tasks)

    def forward(self, raman, descriptors=None):
        r_emb = self.raman_enc.forward_features(raman)
        
        if self.use_descriptors and descriptors is not None:
            d_emb = self.desc_enc.forward_features(descriptors)
            fused = torch.cat([r_emb, d_emb], dim=1)
        else:
            fused = r_emb
            
        fused = self.fusion(fused)
        return self.head(fused)


class Spec2PropLite(FusionSpec2PropNet):
    """
    Alias for FusionSpec2PropNet using LiteSpecNet.
    """
    def __init__(self, *args, **kwargs):
        kwargs["cnn_type"] = "LiteSpecNet"
        super().__init__(*args, **kwargs)

