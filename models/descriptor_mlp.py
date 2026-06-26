"""
Spec2Prop-Inorg: Descriptor MLP
================================
MLP encoder for chemistry/formula descriptor features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DescriptorMLP(nn.Module):
    """
    Small MLP for encoding chemistry descriptor features.

    Input: (batch, descriptor_dim)
    Output: (batch, embed_dim) or (batch, num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        num_classes: int = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(embed_dim, num_classes) if num_classes else None

    def forward_features(self, x):
        """Return embedding (batch, embed_dim)."""
        return self.encoder(x)

    def forward(self, x):
        emb = self.forward_features(x)
        if self.head is not None:
            return self.head(emb)
        return emb
