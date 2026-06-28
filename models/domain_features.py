"""
domain_features.py — Hand-crafted Raman Spectral Descriptors

Scientific motivation
─────────────────────
XGBoost makes axis-aligned splits. A feature that directly encodes
"intensity ratio at 464 cm⁻¹ / 1085 cm⁻¹" is separable in ONE split.
The CNN encodes this implicitly across 128 dimensions, requiring many
splits to recover. Mixing direct domain features with learned CNN
embeddings gives LightGBM both precision (domain) and coverage (CNN).

Features extracted (per spectrum)
───────────────────────────────────
  1. Peak statistics : positions, heights, widths of top-K local maxima
  2. Band ratios     : ratios between intensities at fixed wavenumber windows
  3. Spectral moments: mean, std, skewness, kurtosis of the full spectrum
  4. Subsampled raw  : spectrum downsampled to 64 points (stride=32)
  5. PCA scores      : first 32 components of sklearn PCA fit on training set

This file provides:
  • extract_domain_features(spectra_np)   → np.ndarray (N, n_features)
  • build_feature_pipeline(X_train)       → fitted sklearn Pipeline
  • transform_features(pipeline, X)       → np.ndarray ready for LightGBM
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Peak-based descriptors
# ─────────────────────────────────────────────────────────────────────────────
def _peak_features(spectrum: np.ndarray, n_peaks: int = 10) -> np.ndarray:
    """
    Extract top-N peaks by prominence.
    Returns: [positions (n_peaks), heights (n_peaks), widths (n_peaks)]
             = 3 * n_peaks features, zero-padded if fewer peaks found.
    """
    peaks, props = find_peaks(spectrum, height=0.01, prominence=0.01, distance=10)
    if len(peaks) == 0:
        return np.zeros(3 * n_peaks)

    # Sort by prominence, keep top-N
    proms = props["prominences"]
    order = np.argsort(proms)[::-1][:n_peaks]
    top_peaks = peaks[order]
    top_heights = spectrum[top_peaks]
    widths, *_ = peak_widths(spectrum, top_peaks, rel_height=0.5)

    # Normalise peak positions to [0, 1]
    pos = top_peaks / len(spectrum)

    # Zero-pad to fixed size
    def pad(arr):
        out = np.zeros(n_peaks)
        out[: len(arr)] = arr
        return out

    return np.concatenate([pad(pos), pad(top_heights), pad(widths / len(spectrum))])


# ─────────────────────────────────────────────────────────────────────────────
# Spectral moment descriptors
# ─────────────────────────────────────────────────────────────────────────────
def _moment_features(spectrum: np.ndarray) -> np.ndarray:
    """4 statistical moments of the intensity distribution."""
    s = spectrum
    mean = np.mean(s)
    std = np.std(s) + 1e-8
    skew = np.mean(((s - mean) / std) ** 3)
    kurt = np.mean(((s - mean) / std) ** 4)
    return np.array([mean, std, skew, kurt])


# ─────────────────────────────────────────────────────────────────────────────
# Band-ratio descriptors (chemical-family sensitive windows)
# ─────────────────────────────────────────────────────────────────────────────
# These indices assume a 2048-point spectrum spanning roughly 0–3200 cm⁻¹.
# Adjust if your wavenumber axis differs.
_BAND_WINDOWS = {
    "silicate_main": (200, 280),     # Si–O bending ~465 cm⁻¹
    "silicate_stretch": (830, 900),  # Si–O–Si stretch
    "carbonate": (560, 620),         # CO3 bending ~1085 cm⁻¹ (folded)
    "phosphate": (530, 590),         # P–O stretch
    "oxide_low": (100, 180),         # Metal-oxygen lattice modes
    "ch_stretch": (1780, 1950),      # C–H stretch (organics)
    "oh_stretch": (1950, 2048),      # O–H stretch (hydrated phases)
}

def _band_features(spectrum: np.ndarray) -> np.ndarray:
    """
    Mean intensity in each diagnostic band + pairwise ratios between bands.
    """
    n = len(spectrum)
    band_means = {}
    for name, (lo, hi) in _BAND_WINDOWS.items():
        # Convert to index, clip to valid range
        i_lo = min(max(int(lo * n / 2048), 0), n - 1)
        i_hi = min(max(int(hi * n / 2048), 1), n)
        band_means[name] = spectrum[i_lo:i_hi].mean() + 1e-8

    means_arr = np.array(list(band_means.values()))   # (7,)

    # All pairwise ratios (7 choose 2 = 21)
    keys = list(band_means.keys())
    ratios = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ratios.append(band_means[keys[i]] / band_means[keys[j]])

    return np.concatenate([means_arr, np.array(ratios)])   # 7 + 21 = 28


# ─────────────────────────────────────────────────────────────────────────────
# Subsampled raw spectrum
# ─────────────────────────────────────────────────────────────────────────────
def _subsample_features(spectrum: np.ndarray, n_out: int = 64) -> np.ndarray:
    """Downsample spectrum to n_out points by average pooling."""
    n = len(spectrum)
    stride = max(n // n_out, 1)
    out = np.array([spectrum[i * stride: (i + 1) * stride].mean()
                    for i in range(n_out)])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Derivative features (first and second derivative statistics)
# ─────────────────────────────────────────────────────────────────────────────
def _derivative_features(spectrum: np.ndarray) -> np.ndarray:
    """
    Statistics of first and second derivatives.
    Captures peak sharpness, inflection density, and spectral roughness.
    Returns 12 features: [d1_mean, d1_std, d1_max, d1_min, d1_zcr, d1_energy,
                          d2_mean, d2_std, d2_max, d2_min, d2_zcr, d2_energy]
    """
    d1 = np.diff(spectrum)
    d2 = np.diff(d1)

    def _stats(d):
        zcr = np.sum(np.diff(np.sign(d)) != 0) / (len(d) + 1e-8)  # zero-crossing rate
        energy = np.sum(d ** 2) / (len(d) + 1e-8)
        return np.array([np.mean(d), np.std(d) + 1e-8, np.max(d), np.min(d), zcr, energy])

    return np.concatenate([_stats(d1), _stats(d2)])  # 12 features


# ─────────────────────────────────────────────────────────────────────────────
# Spectral entropy
# ─────────────────────────────────────────────────────────────────────────────
def _spectral_entropy(spectrum: np.ndarray) -> np.ndarray:
    """Shannon entropy of the normalised intensity distribution. 1 feature."""
    s = np.abs(spectrum) + 1e-12
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return np.array([entropy])


# ─────────────────────────────────────────────────────────────────────────────
# Per-band peak count and integrated area
# ─────────────────────────────────────────────────────────────────────────────
def _band_peak_count_and_area(spectrum: np.ndarray) -> np.ndarray:
    """
    For each diagnostic band window:
      - count of local maxima within the band
      - integrated (summed) area within the band
    Returns 14 features: 7 counts + 7 areas.
    """
    n = len(spectrum)
    all_peaks, _ = find_peaks(spectrum, height=0.01, prominence=0.01, distance=5)
    all_peaks_set = set(all_peaks)

    counts = []
    areas = []
    for name, (lo, hi) in _BAND_WINDOWS.items():
        i_lo = min(max(int(lo * n / 2048), 0), n - 1)
        i_hi = min(max(int(hi * n / 2048), 1), n)
        # Count peaks in this band
        band_peak_count = sum(1 for p in all_peaks_set if i_lo <= p < i_hi)
        counts.append(float(band_peak_count))
        # Integrated area
        areas.append(float(spectrum[i_lo:i_hi].sum()))

    return np.concatenate([np.array(counts), np.array(areas)])  # 14 features


# ─────────────────────────────────────────────────────────────────────────────
# Peak prominence features
# ─────────────────────────────────────────────────────────────────────────────
def _peak_prominence_features(spectrum: np.ndarray, n_peaks: int = 10) -> np.ndarray:
    """
    Prominence values of top-N peaks (sorted by prominence).
    Returns n_peaks features, zero-padded.
    """
    peaks, props = find_peaks(spectrum, height=0.01, prominence=0.01, distance=10)
    if len(peaks) == 0:
        return np.zeros(n_peaks)
    proms = props["prominences"]
    proms_sorted = np.sort(proms)[::-1][:n_peaks]
    out = np.zeros(n_peaks)
    out[:len(proms_sorted)] = proms_sorted
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def extract_domain_features(spectra: np.ndarray, n_peaks: int = 10) -> np.ndarray:
    """
    Extract hand-crafted domain features from a batch of spectra.

    Parameters
    ----------
    spectra : (N, L) numpy array — raw spectra, any length L
    n_peaks : number of peaks to track per spectrum

    Returns
    -------
    features : (N, F) numpy array
        F = 3*n_peaks + 4 + 28 + 64 + 12 + 1 + 14 + n_peaks
          = 30 + 4 + 28 + 64 + 12 + 1 + 14 + 10 = 163 (default n_peaks=10)
    """
    # Normalise each spectrum to [0, 1] before feature extraction
    s_norm = spectra - spectra.min(axis=1, keepdims=True)
    denom = s_norm.max(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    s_norm = s_norm / denom

    rows = []
    for i in range(len(s_norm)):
        s = s_norm[i]
        row = np.concatenate([
            _peak_features(s, n_peaks),            # 30
            _moment_features(s),                    #  4
            _band_features(s),                      # 28
            _subsample_features(s, 64),             # 64
            _derivative_features(s),                # 12
            _spectral_entropy(s),                   #  1
            _band_peak_count_and_area(s),           # 14
            _peak_prominence_features(s, n_peaks),  # 10
        ])
        rows.append(row)

    return np.array(rows, dtype=np.float32)


def build_feature_pipeline(X_train: np.ndarray, pca_components: int = 32) -> Pipeline:
    """
    Fit a sklearn Pipeline: StandardScaler → PCA on training domain features.
    Call this ONCE on training data; use transform_features() for val/test.

    Parameters
    ----------
    X_train       : (N_train, F) domain features from extract_domain_features()
    pca_components: number of PCA components to retain

    Returns
    -------
    Fitted sklearn Pipeline
    """
    n_components = min(pca_components, X_train.shape[0], X_train.shape[1])
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, whiten=True, random_state=42)),
    ])
    pipeline.fit(X_train)
    return pipeline


def transform_features(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Apply fitted pipeline to new domain features."""
    return pipeline.transform(X).astype(np.float32)
