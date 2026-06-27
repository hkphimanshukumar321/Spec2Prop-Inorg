"""
raman_former.py — RamanFormer1D v2 (SupCon-Ready)

Key changes from v1:
  1. CLS token pooling  →  replaces mean pooling; gives the transformer a
     dedicated summary slot rather than averaging over all patch positions.
  2. Projection head    →  mandatory for SupCon training (128-d L2-normalised
     sphere), detachable at inference so forward_features() still returns the
     d_model embedding for downstream GBDT.
  3. Deeper / wider tokenizer  →  4-stage instead of 3, with residual bypass
     on the last two stages to stabilise gradient flow on small batches.
  4. Learnable positional embeddings  →  sinusoidal PE is suboptimal for
     sequences <256 tokens; learned PE converges faster at low-N.
  5. Label-smoothed CE head  →  kept for fine-tuning after SupCon Stage 1.
  6. Stochastic depth (drop_path)  →  replaces uniform dropout inside
     transformer layers; regularises transformer on small datasets.

Architecture overview
─────────────────────
  Raw spectrum  (B, 1, 2048)
       │
  Tokenizer CNN  → (B, d_model, 128)          4-stage strided conv
       │
  Patch embedding  → (B, 128+1, d_model)       prepend [CLS] token
       │
  Learned PE  → add positional embeddings
       │
  Transformer Encoder  ×N  (StochDepth)
       │
  [CLS] slice  → (B, d_model)                  ← forward_features() output
       │
  ┌────┴─────────────────────┐
  │                          │
  Proj Head (SupCon)    Linear Head (CE)
  (B, proj_dim, L2norm)   (B, num_classes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Stochastic Depth (drop-path) — per-sample residual zeroing
# ─────────────────────────────────────────────────────────────────────────────
class DropPath(nn.Module):
    """Stochastic depth: randomly drop entire residual branches during training."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device).floor_().add_(keep).clamp_(0, 1)
        return x * mask / keep


# ─────────────────────────────────────────────────────────────────────────────
# Residual Conv Block for Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
class ResConv1d(nn.Module):
    """Conv → BN → GELU with optional stride-matched residual shortcut."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        # Shortcut: 1×1 conv if dimensions change
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride), nn.BatchNorm1d(out_ch))
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Encoder Layer with Drop-Path
# ─────────────────────────────────────────────────────────────────────────────
class TransformerLayerWithDropPath(nn.Module):
    """Standard Pre-LN transformer block + stochastic depth."""
    def __init__(self, d_model: int, nhead: int, dim_ff: int,
                 dropout: float, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dp1 = DropPath(drop_path)
        self.dp2 = DropPath(drop_path)

    def forward(self, x, src_key_padding_mask=None):
        # Pre-LN attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=src_key_padding_mask)
        x = x + self.dp1(h)
        # Pre-LN FFN
        x = x + self.dp2(self.ff(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────
class RamanFormer1D(nn.Module):
    """
    RamanFormer1D v2 — SupCon-ready CNN-Transformer for Raman spectral classification.

    Parameters
    ----------
    in_channels        : raw spectrum channels (default 1)
    num_classes        : if None, no CE head is built (embedding-only mode)
    d_model            : transformer / CLS embedding dimension (default 128)
    nhead              : attention heads (default 4)
    num_encoder_layers : transformer depth (default 4; was 3 in v1)
    dim_feedforward    : FFN width (default 512)
    dropout            : dropout rate inside transformer FFN and attention
    drop_path_rate     : max stochastic-depth rate (linearly scaled per layer)
    proj_dim           : SupCon projection head output dimension (default 128)
    label_smoothing    : label smoothing for CE head (default 0.1)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = None,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        drop_path_rate: float = 0.1,
        proj_dim: int = 128,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # ── 1. CNN Tokenizer ─────────────────────────────────────────────────
        # 4-stage: 2048 → 1024 → 512 → 256 → 128
        #   Stage 1-2: plain strided conv (no residual; channel expansion phase)
        #   Stage 3-4: ResConv1d (residual; feature refinement phase)
        self.tokenizer = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),

            ResConv1d(64, d_model, kernel=3, stride=2),   # 512 → 256
            ResConv1d(d_model, d_model, kernel=3, stride=2),  # 256 → 128
        )
        # Output shape: (B, d_model, 128)

        # ── 2. CLS Token + Learned Positional Embedding ───────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 128 patch positions + 1 CLS = 129
        self.pos_emb = nn.Parameter(torch.zeros(1, 129, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.emb_drop = nn.Dropout(dropout)

        # ── 3. Transformer Encoder with linear stoch-depth schedule ────────
        dp_rates = [r.item() for r in torch.linspace(0, drop_path_rate, num_encoder_layers)]
        self.encoder = nn.ModuleList([
            TransformerLayerWithDropPath(d_model, nhead, dim_feedforward, dropout, dp_rates[i])
            for i in range(num_encoder_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # ── 4. Projection Head (SupCon) ────────────────────────────────────
        # 2-layer MLP → L2-norm onto unit hypersphere
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

        # ── 5. Classification Head (CE fine-tuning) ───────────────────────
        self.head = (
            nn.Sequential(
                nn.Linear(d_model, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )
            if num_classes
            else None
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Core feature extraction (used by both training modes + GBDT export) ─
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the CLS-token embedding of shape (B, d_model).
        This is the vector you pass to XGBoost / LightGBM at inference.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)   # (B, 2048) → (B, 1, 2048)

        # Tokenize: (B, 1, 2048) → (B, d_model, 128)
        x = self.tokenizer(x)

        # Rearrange: (B, d_model, 128) → (B, 128, d_model)
        x = x.permute(0, 2, 1)

        # Prepend CLS token: (B, 129, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add positional embeddings
        x = self.emb_drop(x + self.pos_emb)

        # Transformer
        for layer in self.encoder:
            x = layer(x)
        x = self.norm(x)

        # Return CLS token only
        return x[:, 0, :]   # (B, d_model)

    # ── Projection (SupCon training only) ──────────────────────────────────
    def forward_proj(self, x: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised projection for SupCon loss."""
        feat = self.forward_features(x)
        z = self.proj_head(feat)
        return F.normalize(z, dim=-1)   # unit hypersphere

    # ── Full forward (CE fine-tuning or standalone inference) ──────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward_features(x)
        if self.head is not None:
            return self.head(emb)
        return emb
