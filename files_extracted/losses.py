"""
losses.py — Supervised Contrastive Loss for Raman spectral classification

SupConLoss (Khosla et al., NeurIPS 2020)
─────────────────────────────────────────
Standard cross-entropy trains the network to push logits for the correct
class above others. The resulting embedding space has no geometry constraints:
same-class samples can be spread anywhere as long as the linear head separates
them. This is catastrophic for downstream GBDTs, which need axis-aligned,
locally consistent structure.

SupCon fixes this by optimising a contrastive objective directly on the
embedding space:
  • All samples sharing a label are attracted to each other
  • All samples with different labels are repelled
  • Operates on L2-normalised projections (unit hypersphere)
  • Temperature τ controls cluster tightness

Two-stage training protocol:
  Stage 1 (this loss): train backbone + proj_head with SupConLoss only.
                       No classification head involved.
  Stage 2 (CE loss):   freeze backbone, fine-tune linear head with standard
                       label-smoothed cross-entropy.

Usage in your training loop:
  projections = model.forward_proj(x)           # (B, proj_dim), L2-normed
  projections = projections.unsqueeze(1)         # (B, 1, proj_dim) — single-view
  loss = supcon_loss(projections, labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Parameters
    ----------
    temperature : float
        τ in the SupCon paper. Lower → tighter clusters, higher gradient
        magnitude, but can be unstable. 0.07–0.1 is typical for spectral data.
    contrast_mode : str
        'all'  — every sample in the batch is used as an anchor (recommended)
        'one'  — only the first view is used as anchor
    base_temperature : float
        Normalisation constant; keep equal to temperature unless you have a
        specific reason to decouple them.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "all",
        base_temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (B, n_views, proj_dim)  — already L2-normalised
            For single-view (no augmentation) use n_views=1:
                features = model.forward_proj(x).unsqueeze(1)
        labels   : (B,) — integer class indices

        Returns
        -------
        scalar loss
        """
        device = features.device
        B, n_views, _ = features.shape

        # Flatten to (B*n_views, proj_dim)
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)  # (B*n, proj_dim)
        batch_size = B

        if self.contrast_mode == "all":
            anchor_features = contrast_features
            anchor_count = n_views
        elif self.contrast_mode == "one":
            anchor_features = features[:, 0]
            anchor_count = 1
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        # Build positive mask: 1 where anchor_i and contrast_j share a label
        labels = labels.contiguous().view(-1, 1)  # (B, 1)
        # Tile labels for all views
        mask = torch.eq(
            labels.repeat(anchor_count, 1),                          # (B*n, 1)
            labels.T.repeat(1, n_views).view(1, -1).repeat(anchor_count * batch_size, 1) if n_views > 1
            else labels.T.expand(batch_size * anchor_count, -1)
        ).float().to(device)

        # Simpler correct mask construction:
        labels_tiled = labels.repeat(n_views, 1)     # (B*n, 1)
        contrast_labels = labels.repeat(1, n_views).view(-1, 1)  # (B*n, 1)
        mask = torch.eq(labels_tiled, contrast_labels.T).float().to(device)  # (B*n, B*n)

        # Compute cosine similarity matrix (features already L2-normed)
        sim = torch.matmul(anchor_features, contrast_features.T) / self.temperature  # (B*n, B*n)

        # For numerical stability
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Remove self-contrast from denominator
        logits_mask = torch.ones_like(mask) - torch.eye(anchor_count * batch_size, device=device)
        mask = mask * logits_mask

        # Log-sum-exp denominator (all non-self pairs)
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean over positive pairs
        n_positives = mask.sum(dim=1)
        # Guard against classes with no other sample in the batch
        valid = n_positives > 0
        mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid] / n_positives[valid]

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss — handles class imbalance better than weighted CE
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) with optional per-class weights.

    γ=2 is the standard; increase to 3–4 if your minority classes are very rare.
    Use this for Stage 2 (CE fine-tuning) instead of plain cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
