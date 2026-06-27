"""
Spec2Prop-Inorg: Supervised Contrastive Loss
=============================================
Implementation of SupCon (Khosla et al., NeurIPS 2020).

Forces same-class embeddings to cluster and different-class embeddings
to repel. When downstream GBDT classifiers operate on these embeddings,
tree splits become geometrically meaningful — each split corresponds to
a real boundary between class clusters, not an arbitrary hyperplane.

Reference: https://arxiv.org/abs/2004.11362
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    Given a batch of L2-normalized embedding vectors and their class labels,
    pulls together same-class embeddings and pushes apart different-class
    embeddings using a temperature-scaled cross-entropy formulation.

    Parameters
    ----------
    temperature : float
        Scaling factor for the similarity logits. Lower values sharpen
        the distribution (default: 0.07).
    base_temperature : float
        Used for normalization (default: 0.07).
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute SupCon loss.

        Parameters
        ----------
        features : torch.Tensor
            Shape (batch_size, embed_dim). Will be L2-normalized internally.
        labels : torch.Tensor
            Shape (batch_size,). Integer class labels.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # L2 normalize
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix: (B, B)
        similarity = torch.matmul(features, features.T) / self.temperature

        # Mask: 1 where labels match (positive pairs), 0 elsewhere
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Remove self-similarity from consideration
        logits_mask = 1.0 - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # Check if any positive pairs exist
        positives_per_row = mask.sum(dim=1)
        has_positives = positives_per_row > 0

        if not has_positives.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Numerical stability: subtract max logit
        logits_max, _ = similarity.detach().max(dim=1, keepdim=True)
        logits = similarity - logits_max

        # Compute log-softmax over negatives
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Mean log-prob over positive pairs (only for rows with positives)
        mean_log_prob = torch.zeros(batch_size, device=device)
        mean_log_prob[has_positives] = (
            (mask[has_positives] * log_prob[has_positives]).sum(dim=1)
            / positives_per_row[has_positives]
        )

        # Loss
        loss = -(self.base_temperature / self.temperature) * mean_log_prob
        loss = loss[has_positives].mean()

        return loss


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.

    Projects embeddings to a lower-dimensional space where the
    contrastive loss is applied. This head is discarded after
    pre-training; only the encoder is kept.

    Architecture: Linear → BN → ReLU → Linear
    """

    def __init__(self, in_dim: int = 128, hidden_dim: int = 256,
                 out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
