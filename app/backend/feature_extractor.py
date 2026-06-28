"""
Spec2Prop-Edge: Feature Extractor
===================================
Reproduces the exact same feature extraction used during training.
Uses the FITTED PCA/scaler/prototype objects — never refits on test data.
"""
import sys
import os
import numpy as np
from typing import Optional

# Ensure project root is on path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.domain_features import extract_domain_features


def extract_features(
    raman_vector: np.ndarray,
    pca=None,
    scaler=None,
    prototype_extractor=None,
) -> np.ndarray:
    """
    Extract the full heterogeneous feature vector for a single sample.

    This mirrors the exact pipeline from train_deployment_model.py:
      1. Domain features (163-d): peaks, bands, moments, derivatives, entropy
      2. PCA features (32-d): fitted on training spectra
      3. Prototype similarity features (27-d): fitted on training spectra

    Parameters
    ----------
    raman_vector : (2048,) raw Raman spectrum
    pca : fitted PCA object (from training)
    scaler : fitted StandardScaler (from training)
    prototype_extractor : fitted PrototypeExtractor (from training)

    Returns
    -------
    features : (1, F) feature matrix ready for model input
    """
    # Ensure 2D input
    if raman_vector.ndim == 1:
        X = raman_vector.reshape(1, -1)
    else:
        X = raman_vector

    # 1. Domain features (no fitting needed)
    domain_feats = extract_domain_features(X)  # (1, 163)

    feature_parts = [domain_feats]

    # 2. PCA features (transform only — never fit)
    if pca is not None:
        pca_feats = pca.transform(X).astype(np.float32)  # (1, 32)
        feature_parts.append(pca_feats)

    # 3. Prototype similarity features (transform only — never fit)
    if prototype_extractor is not None:
        proto_feats = prototype_extractor.transform(X)  # (1, 27)
        feature_parts.append(proto_feats)

    # Concatenate all features
    X_features = np.hstack(feature_parts)  # (1, 222)

    # 4. Scale using fitted scaler (transform only — never fit)
    if scaler is not None:
        X_features = scaler.transform(X_features).astype(np.float32)

    return X_features


def extract_features_fallback(
    raman_vector: np.ndarray,
    pca=None,
) -> np.ndarray:
    """
    Simplified feature extraction for fallback model (no scaler/prototypes).
    Used when only PCA + raw spectrum features are available.
    """
    if raman_vector.ndim == 1:
        X = raman_vector.reshape(1, -1)
    else:
        X = raman_vector

    if pca is not None:
        return pca.transform(X).astype(np.float32)
    else:
        # Return raw spectrum if no PCA
        return X.astype(np.float32)
