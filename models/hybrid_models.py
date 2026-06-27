"""
Spec2Prop-Inorg: Hybrid Models — Heterogeneous Feature Extraction
==================================================================
Builds rich feature vectors for downstream GBDT/TabPFN classifiers:

    [CNN_embedding (128-d)]            ← learned nonlinear features
  + [PCA_components (32-d)]            ← global variance structure
  + [domain_descriptors (126-d)]       ← peak stats, moments, band ratios
  = total ~286-d heterogeneous feature vector
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from typing import Optional

from models.domain_features import extract_domain_features


class HeterogeneousFeatureExtractor:
    """
    Extracts a heterogeneous feature vector combining:
      1. CNN embeddings (from a trained encoder)
      2. PCA components (global variance structure)
      3. Domain-engineered spectral features (peaks, bands, ratios)
    """

    def __init__(self, model=None, pca=None, device='cpu', n_peaks: int = 10):
        self.model = model
        self.pca = pca
        self.device = device
        self.n_peaks = n_peaks

        if self.model is not None:
            self.model.eval()
            self.model.to(self.device)

    def extract_cnn_embeddings_from_array(self, X_raw: np.ndarray) -> np.ndarray:
        """Extract CNN embeddings from raw numpy array."""
        if self.model is None:
            return None

        embeddings = []
        self.model.eval()

        with torch.no_grad():
            batch_size = 64
            for start in range(0, len(X_raw), batch_size):
                end = min(start + batch_size, len(X_raw))
                x = torch.from_numpy(X_raw[start:end]).float()
                # Reshape to (batch, 1, 2048) for Conv1d
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                x = x.to(self.device)
                emb = self.model.forward_features(x)
                embeddings.append(emb.cpu().numpy())

        return np.vstack(embeddings).astype(np.float32)

    def extract_pca_features(self, X_raw: np.ndarray) -> Optional[np.ndarray]:
        """Apply PCA transform to raw spectra."""
        if self.pca is None:
            return None
        return self.pca.transform(X_raw).astype(np.float32)

    def extract_domain_feats(self, X_raw: np.ndarray) -> np.ndarray:
        """Extract domain-engineered spectral features using batch API."""
        return extract_domain_features(X_raw, n_peaks=self.n_peaks)

    def extract(self, X_raw: np.ndarray) -> np.ndarray:
        components = []

        cnn_emb = self.extract_cnn_embeddings_from_array(X_raw)
        if cnn_emb is not None:
            components.append(cnn_emb)

        pca_feats = self.extract_pca_features(X_raw)
        if pca_feats is not None:
            components.append(pca_feats)

        domain_feats = self.extract_domain_feats(X_raw)
        components.append(domain_feats)

        return np.hstack(components)

    def get_feature_names(self, embed_dim: int = 128, pca_dim: int = 32) -> list:
        """Return human-readable feature names for interpretability."""
        names = []

        if self.model is not None:
            names += [f"cnn_embed_{i}" for i in range(embed_dim)]

        if self.pca is not None:
            names += [f"pca_{i}" for i in range(pca_dim)]

        # Domain feature names (126 features)
        for i in range(self.n_peaks):
            names.append(f"peak_pos_{i}")
        for i in range(self.n_peaks):
            names.append(f"peak_intensity_{i}")
        for i in range(self.n_peaks):
            names.append(f"peak_width_{i}")
            
        stat_names = ["mean", "std", "skew", "kurtosis"]
        names += [f"stat_{s}" for s in stat_names]

        band_windows = ["silicate_main", "silicate_stretch", "carbonate", "phosphate", "oxide_low", "ch_stretch", "oh_stretch"]
        for b in band_windows:
            names.append(f"band_mean_{b}")
            
        for i in range(len(band_windows)):
            for j in range(i + 1, len(band_windows)):
                names.append(f"ratio_{band_windows[i]}_{band_windows[j]}")

        for i in range(64):
            names.append(f"subsample_{i}")

        return names
