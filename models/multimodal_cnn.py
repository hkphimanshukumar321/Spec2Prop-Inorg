"""
Spec2Prop-Inorg: Multimodal CNN
===============================
Dual-branch CNN for processing aligned Raman and XRD spectra.
"""

import torch
import torch.nn as nn
from models.cnn1d import SimpleCNN1D, LiteSpecNet
from models.descriptor_mlp import DescriptorMLP
from models.fusion_models import FusionMultiTaskHead


class DualBranchRamanXRDNet(nn.Module):
    """
    Raman CNN Encoder + XRD CNN Encoder + Optional Descriptor MLP -> Multi-task Heads.
    """

    def __init__(
        self,
        tasks: dict,
        raman_channels: int = 1,
        xrd_channels: int = 1,
        embed_dim: int = 128,
        descriptor_dim: int = 0,
        descriptor_embed_dim: int = 32,
        fusion_dim: int = 128,
        dropout: float = 0.3,
        cnn_type: str = "SimpleCNN1D"
    ):
        super().__init__()
        self.use_descriptors = descriptor_dim > 0

        # Encoders
        CNNClass = LiteSpecNet if cnn_type == "LiteSpecNet" else SimpleCNN1D
        self.raman_enc = CNNClass(raman_channels, embed_dim, dropout=dropout)
        self.xrd_enc = CNNClass(xrd_channels, embed_dim, dropout=dropout)

        concat_dim = embed_dim * 2

        if self.use_descriptors:
            self.desc_enc = DescriptorMLP(
                input_dim=descriptor_dim, 
                embed_dim=descriptor_embed_dim, 
                dropout=dropout
            )
            concat_dim += descriptor_embed_dim

        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Multi-task Head
        self.head = FusionMultiTaskHead(fusion_dim, tasks)

    def forward(self, raman, xrd, descriptors=None):
        r_emb = self.raman_enc.forward_features(raman)
        x_emb = self.xrd_enc.forward_features(xrd)
        
        if self.use_descriptors and descriptors is not None:
            d_emb = self.desc_enc.forward_features(descriptors)
            fused = torch.cat([r_emb, x_emb, d_emb], dim=1)
        else:
            fused = torch.cat([r_emb, x_emb], dim=1)
            
        fused = self.fusion(fused)
        return self.head(fused)
