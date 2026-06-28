"""
prototype_features.py — Class Prototype Similarity Features
============================================================
Computes per-sample similarity to train-only class prototypes.

Scientific motivation:
  Traditional Raman identification uses spectral library matching.
  Instead of matching to individual reference spectra, we compute
  similarity to the MEAN spectrum of each class (the "prototype").
  This gives the GBDT a direct, axis-aligned feature that encodes
  "how much does this spectrum look like a Silicate vs an Oxide?"

Features (per sample, per class):
  - Cosine similarity to class prototype
  - Pearson correlation to class prototype
  - Euclidean distance to class prototype

Total: 3 * num_classes features.

CRITICAL: Prototypes are computed from TRAINING data only.
          The .fit() method stores prototypes; .transform() uses them.
          Never call .fit() on validation or test data.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class PrototypeExtractor:
    """
    Compute class-prototype similarity features.

    Usage:
        pe = PrototypeExtractor()
        pe.fit(X_train, y_train)          # fit on training data only
        feats_train = pe.transform(X_train)
        feats_val   = pe.transform(X_val)   # transform only, no fitting
        feats_test  = pe.transform(X_test)
    """

    def __init__(self):
        self.prototypes = None  # (num_classes, L) mean spectra
        self.class_labels = None  # sorted unique labels
        self.num_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Compute class prototypes (mean spectrum per class) from training data.

        Parameters
        ----------
        X : (N, L) raw spectra
        y : (N,) integer class labels
        """
        self.class_labels = np.sort(np.unique(y))
        self.num_classes = len(self.class_labels)
        self.prototypes = np.zeros((self.num_classes, X.shape[1]), dtype=np.float64)

        for i, c in enumerate(self.class_labels):
            mask = y == c
            if mask.sum() > 0:
                self.prototypes[i] = X[mask].mean(axis=0)

        # Also store normalised prototypes for cosine/correlation
        norms = np.linalg.norm(self.prototypes, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._prototypes_normed = self.prototypes / norms

        # For Pearson correlation: center each prototype
        self._prototypes_centered = self.prototypes - self.prototypes.mean(axis=1, keepdims=True)
        centered_norms = np.linalg.norm(self._prototypes_centered, axis=1, keepdims=True)
        centered_norms[centered_norms == 0] = 1.0
        self._prototypes_centered_normed = self._prototypes_centered / centered_norms

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute similarity features for each sample against all prototypes.

        Parameters
        ----------
        X : (N, L) raw spectra

        Returns
        -------
        features : (N, 3 * num_classes) numpy array
            [cosine_0, ..., cosine_K, corr_0, ..., corr_K, dist_0, ..., dist_K]
        """
        if self.prototypes is None:
            raise RuntimeError("PrototypeExtractor.fit() must be called before transform().")

        N = X.shape[0]
        K = self.num_classes

        # 1. Cosine similarity: (N, K)
        cos_sim = cosine_similarity(X, self.prototypes).astype(np.float32)

        # 2. Pearson correlation: center each sample, then cosine with centered prototypes
        X_centered = X - X.mean(axis=1, keepdims=True)
        X_centered_norms = np.linalg.norm(X_centered, axis=1, keepdims=True)
        X_centered_norms[X_centered_norms == 0] = 1.0
        X_centered_normed = X_centered / X_centered_norms
        pearson = (X_centered_normed @ self._prototypes_centered_normed.T).astype(np.float32)

        # 3. Euclidean distance: (N, K)
        # Using broadcasting: ||x - p||^2 = ||x||^2 + ||p||^2 - 2*x.p
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (N, 1)
        P_sq = np.sum(self.prototypes ** 2, axis=1, keepdims=True).T  # (1, K)
        dists_sq = X_sq + P_sq - 2.0 * (X @ self.prototypes.T)
        dists_sq = np.maximum(dists_sq, 0.0)  # numerical safety
        euclidean = np.sqrt(dists_sq).astype(np.float32)

        return np.hstack([cos_sim, pearson, euclidean])  # (N, 3*K)

    def get_feature_names(self, class_names=None):
        """Return human-readable feature names."""
        if class_names is None:
            class_names = [str(c) for c in self.class_labels]
        names = []
        for prefix in ["cos_sim", "pearson", "euclidean"]:
            for cname in class_names:
                names.append(f"{prefix}_{cname}")
        return names
