"""
Spec2Prop-Inorg: 1D CNN Architectures for spectral encoding.
=============================================================
SimpleCNN1D   — standard Conv1D stack
LiteSpecNet   — depthwise-separable, edge-friendly
ResidualCNN1D — small residual blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv1D → BatchNorm → ReLU → optional MaxPool → Dropout."""

    def __init__(self, in_ch, out_ch, kernel_size=7, pool_size=4, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.pool(F.relu(self.bn(self.conv(x)))))
        return x


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution: depthwise + pointwise."""

    def __init__(self, in_ch, out_ch, kernel_size=7, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size, padding=padding, groups=in_ch
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ResidualBlock(nn.Module):
    """Small 1D residual block with two conv layers."""

    def __init__(self, channels, kernel_size=5, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(self.bn2(self.conv2(out)))
        return F.relu(out + residual)


# ---------------------------------------------------------------------------
# SimpleCNN1D
# ---------------------------------------------------------------------------

class SimpleCNN1D(nn.Module):
    """
    Standard 1D-CNN for spectral classification/encoding.

    Architecture:
        Conv(1→32, k=7) → BN → ReLU → Pool(4) → Drop
        Conv(32→64, k=5) → BN → ReLU → Pool(4) → Drop
        Conv(64→128, k=3) → BN → ReLU → Pool(4) → Drop
        GlobalAveragePooling → FC → embed_dim
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 128,
        num_classes: int = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=7, pool_size=4, dropout=dropout),
            ConvBlock(32, 64, kernel_size=5, pool_size=4, dropout=dropout),
            ConvBlock(64, 128, kernel_size=3, pool_size=4, dropout=dropout),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc_embed = nn.Linear(128, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes else None

    def forward_features(self, x):
        """Return embedding vector (batch, embed_dim)."""
        x = self.encoder(x)
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc_embed(x))
        return x

    def forward(self, x):
        emb = self.forward_features(x)
        if self.head is not None:
            return self.head(emb)
        return emb


# ---------------------------------------------------------------------------
# LiteSpecNet — lightweight, edge-friendly
# ---------------------------------------------------------------------------

class LiteSpecNet(nn.Module):
    """
    Lightweight spectral CNN using depthwise-separable convolutions.

    Designed for edge deployment (Raspberry Pi, Jetson Nano).
    ~15K parameters with default settings.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 64,
        num_classes: int = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        # Initial standard conv to expand channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=11, padding=5),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )
        # Depthwise separable blocks
        self.dw1 = nn.Sequential(
            DepthwiseSeparableConv1d(16, 32, kernel_size=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
        )
        self.dw2 = nn.Sequential(
            DepthwiseSeparableConv1d(32, 48, kernel_size=5),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc_embed = nn.Linear(48, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes else None

    def forward_features(self, x):
        x = self.stem(x)
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc_embed(x))
        return x

    def forward(self, x):
        emb = self.forward_features(x)
        if self.head is not None:
            return self.head(emb)
        return emb


# ---------------------------------------------------------------------------
# ResidualCNN1D
# ---------------------------------------------------------------------------

class ResidualCNN1D(nn.Module):
    """
    1D-CNN with residual blocks for spectral classification.

    Architecture:
        Conv(1→32) → Pool → ResBlock(32) → Pool
        Conv(32→64) → Pool → ResBlock(64) → Pool
        GAP → FC → embed_dim
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 128,
        num_classes: int = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )
        self.res1 = ResidualBlock(32, kernel_size=5, dropout=dropout)

        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )
        self.res2 = ResidualBlock(64, kernel_size=3, dropout=dropout)

        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc_embed = nn.Linear(128, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes else None

    def forward_features(self, x):
        x = self.block1(x)
        x = self.res1(x)
        x = self.block2(x)
        x = self.res2(x)
        x = self.block3(x)
        x = self.gap(x).squeeze(-1)
        x = self.drop(F.relu(self.fc_embed(x)))
        return x

    def forward(self, x):
        emb = self.forward_features(x)
        if self.head is not None:
            return self.head(emb)
        return emb


# ---------------------------------------------------------------------------
# Utility: model summary
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """Estimate model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)
