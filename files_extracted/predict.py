"""
predict.py — Inference wrapper for the SupCon → LightGBM hybrid pipeline

Loads:
  1. RamanFormer1D backbone (supcon_backbone.pt or stage2_full.pt)
  2. Domain feature pipeline (domain_pipeline.pkl)
  3. LightGBM classifier (lightgbm_hybrid.pkl)

Usage:
  from predict import HybridPredictor
  predictor = HybridPredictor("./checkpoints", num_classes=22, device="cuda")

  # X: np.ndarray of shape (N, 2048) — raw spectra
  preds, probs = predictor.predict(X)
"""

import numpy as np
import torch
import joblib

from raman_former import RamanFormer1D
from domain_features import extract_domain_features, transform_features


class HybridPredictor:
    """
    Full inference pipeline: raw spectrum → class prediction + probabilities.

    Parameters
    ----------
    checkpoint_dir : path to directory containing the three saved artefacts
    num_classes    : number of chemical families
    d_model        : must match training configuration (default 128)
    device         : 'cuda' or 'cpu'
    use_stage2     : if True, load stage2_full.pt (has CE head);
                     if False, load supcon_backbone.pt (embedding-only)
    """

    def __init__(self, checkpoint_dir: str, num_classes: int,
                 d_model: int = 128, device: str = "cpu", use_stage2: bool = False):
        import os
        self.device = device
        self.d_model = d_model

        # ── Load backbone ────────────────────────────────────────────────────
        ckpt = "stage2_full.pt" if use_stage2 else "supcon_backbone.pt"
        backbone_path = os.path.join(checkpoint_dir, ckpt)

        self.backbone = RamanFormer1D(
            in_channels=1,
            num_classes=num_classes if use_stage2 else None,
            d_model=d_model,
            nhead=4,
            num_encoder_layers=4,
            dim_feedforward=512,
            dropout=0.0,      # eval mode: dropout=0 anyway
            drop_path_rate=0.0,
        )
        state = torch.load(backbone_path, map_location=device)
        self.backbone.load_state_dict(state, strict=False)
        self.backbone.to(device).eval()

        # ── Load domain pipeline ─────────────────────────────────────────────
        self.dom_pipeline = joblib.load(os.path.join(checkpoint_dir, "domain_pipeline.pkl"))

        # ── Load LightGBM ────────────────────────────────────────────────────
        self.lgbm = joblib.load(os.path.join(checkpoint_dir, "lightgbm_hybrid.pkl"))

        print(f"HybridPredictor loaded from {checkpoint_dir}")

    def _extract_embeddings(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        X_t = torch.FloatTensor(X)
        embs = []
        for i in range(0, len(X), batch_size):
            xb = X_t[i: i + batch_size].to(self.device)
            with torch.no_grad():
                e = self.backbone.forward_features(xb)
            embs.append(e.cpu().numpy())
        return np.concatenate(embs, axis=0)

    def predict(self, X: np.ndarray):
        """
        Parameters
        ----------
        X : (N, 2048) float32 — raw Raman spectra (same normalisation as training)

        Returns
        -------
        preds : (N,) int64  — predicted class indices
        probs : (N, C) float64 — class probabilities from LightGBM
        """
        emb = self._extract_embeddings(X)
        dom_raw = extract_domain_features(X)
        dom = transform_features(self.dom_pipeline, dom_raw)
        X_hybrid = np.concatenate([emb, dom], axis=1)

        probs = self.lgbm.predict_proba(X_hybrid)
        preds = probs.argmax(axis=1)
        return preds, probs
