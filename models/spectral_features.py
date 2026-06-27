"""
Spec2Prop-Inorg: Domain-Engineered Spectral Features
=====================================================
Extracts mineralogist-style features from Raman spectra:
  - Peak positions, intensities, and FWHM
  - Peak intensity ratios (diagnostic pairs)
  - Integrated area under characteristic Raman windows
  - Subsampled raw spectrum for direct tree splits

These features complement learned CNN embeddings in the
heterogeneous feature vector for GBDT/TabPFN classifiers.
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths


# ---------------------------------------------------------------------------
# Characteristic Raman windows (cm⁻¹ mapped to 2048-point index)
# Assumes linear mapping from ~100 cm⁻¹ to ~4000 cm⁻¹
# ---------------------------------------------------------------------------
RAMAN_RANGE = (100.0, 4000.0)
N_POINTS = 2048

def _cm_to_idx(cm_value: float) -> int:
    """Convert wavenumber (cm⁻¹) to 2048-point array index."""
    frac = (cm_value - RAMAN_RANGE[0]) / (RAMAN_RANGE[1] - RAMAN_RANGE[0])
    return int(np.clip(frac * N_POINTS, 0, N_POINTS - 1))


# Diagnostic Raman windows for inorganic minerals
RAMAN_WINDOWS = [
    (100, 300),    # Lattice modes, heavy-atom vibrations
    (300, 500),    # Metal-oxygen stretching (e.g., TiO₂, Fe₂O₃)
    (500, 700),    # Si-O-Si bending, sulfate bending
    (700, 900),    # Carbonate internal modes
    (900, 1200),   # Si-O stretching, phosphate, sulfate
    (1200, 1600),  # Carbonate asymmetric stretch, organics
    (1600, 2000),  # Overtones, combination bands
    (2800, 3600),  # O-H stretching, water
]


def _safe_peak_detection(spectrum: np.ndarray, n_peaks: int = 20,
                         prominence: float = 0.02,
                         distance: int = 10) -> tuple:
    """Detect peaks with fallback for flat spectra."""
    peaks, properties = find_peaks(
        spectrum,
        prominence=prominence,
        distance=distance,
    )

    if len(peaks) == 0:
        # Fallback: use top-n intensity positions
        peaks = np.argsort(spectrum)[-n_peaks:]
        peaks = np.sort(peaks)
        properties = {"prominences": spectrum[peaks]}

    # Sort by prominence and take top-n
    prominences = properties.get("prominences", spectrum[peaks])
    order = np.argsort(prominences)[::-1][:n_peaks]
    peaks = peaks[order]
    prominences = prominences[order]

    return peaks, prominences


def extract_peak_features(spectrum: np.ndarray, n_peaks: int = 20) -> np.ndarray:
    """
    Extract peak-based features from a 1D spectrum.

    Returns:
        Fixed-length vector: [peak_positions(n), peak_intensities(n), fwhm(n)]
        Total: 3 * n_peaks features.
    """
    peaks, prominences = _safe_peak_detection(spectrum, n_peaks=n_peaks)

    # Pad if fewer peaks than n_peaks
    n_found = len(peaks)
    positions = np.zeros(n_peaks, dtype=np.float32)
    intensities = np.zeros(n_peaks, dtype=np.float32)
    fwhm_vals = np.zeros(n_peaks, dtype=np.float32)

    positions[:n_found] = peaks[:n_found] / N_POINTS  # Normalize to [0, 1]
    intensities[:n_found] = spectrum[peaks[:n_found]]

    # Compute FWHM for found peaks
    if n_found > 0:
        try:
            widths, _, _, _ = peak_widths(spectrum, peaks[:n_found], rel_height=0.5)
            fwhm_vals[:n_found] = widths / N_POINTS  # Normalize
        except Exception:
            pass  # Leave as zeros if FWHM computation fails

    return np.concatenate([positions, intensities, fwhm_vals])


def extract_peak_ratios(spectrum: np.ndarray, n_peaks: int = 20,
                        n_ratios: int = 10) -> np.ndarray:
    """
    Compute intensity ratios between top peaks.

    Ratios between the strongest peaks are highly diagnostic for
    mineral identification (e.g., quartz 464/206 cm⁻¹ ratio).

    Returns:
        Fixed-length vector of n_ratios pairwise intensity ratios.
    """
    peaks, _ = _safe_peak_detection(spectrum, n_peaks=n_peaks)
    peak_intensities = spectrum[peaks]

    ratios = np.zeros(n_ratios, dtype=np.float32)
    idx = 0
    for i in range(min(len(peak_intensities), n_peaks)):
        for j in range(i + 1, min(len(peak_intensities), n_peaks)):
            if idx >= n_ratios:
                break
            denom = peak_intensities[j]
            if denom > 1e-8:
                ratios[idx] = peak_intensities[i] / denom
            idx += 1
            if idx >= n_ratios:
                break

    return ratios


def extract_band_areas(spectrum: np.ndarray) -> np.ndarray:
    """
    Compute integrated area under characteristic Raman windows.

    Returns:
        Vector of len(RAMAN_WINDOWS) integrated areas (trapezoidal rule).
    """
    areas = np.zeros(len(RAMAN_WINDOWS), dtype=np.float32)
    for i, (lo, hi) in enumerate(RAMAN_WINDOWS):
        idx_lo = _cm_to_idx(lo)
        idx_hi = _cm_to_idx(hi)
        if idx_hi > idx_lo:
            areas[i] = np.trapz(spectrum[idx_lo:idx_hi])
    return areas


def extract_statistical_features(spectrum: np.ndarray) -> np.ndarray:
    """
    Basic statistical features of the spectrum.

    Returns:
        [mean, std, skewness, kurtosis, max, min, range, energy]
    """
    mean = np.mean(spectrum)
    std = np.std(spectrum)
    # Skewness
    if std > 1e-8:
        skew = np.mean(((spectrum - mean) / std) ** 3)
        kurt = np.mean(((spectrum - mean) / std) ** 4) - 3.0
    else:
        skew = 0.0
        kurt = 0.0
    energy = np.sum(spectrum ** 2)
    return np.array([mean, std, skew, kurt,
                     np.max(spectrum), np.min(spectrum),
                     np.max(spectrum) - np.min(spectrum),
                     energy], dtype=np.float32)


def subsample_spectrum(spectrum: np.ndarray, stride: int = 32) -> np.ndarray:
    """
    Subsample spectrum at fixed stride for direct tree splits.

    2048 / 32 = 64 points. Each point is directly splittable by a tree
    node, unlike CNN embeddings which encode information implicitly.
    """
    return spectrum[::stride].astype(np.float32)


def extract_spectral_features(spectrum: np.ndarray,
                               n_peaks: int = 20,
                               n_ratios: int = 10,
                               subsample_stride: int = 32) -> np.ndarray:
    """
    Extract full domain-engineered feature vector from a Raman spectrum.

    Composition:
        Peak features:     3 * n_peaks  = 60
        Peak ratios:       n_ratios     = 10
        Band areas:        8 windows    =  8
        Statistics:        8 features   =  8
        Subsampled:        2048/stride  = 64
        ─────────────────────────────────
        Total:                          ~150

    Parameters
    ----------
    spectrum : np.ndarray
        1D array of shape (2048,) — the Raman spectrum.
    n_peaks : int
        Number of top peaks to extract.
    n_ratios : int
        Number of pairwise intensity ratios.
    subsample_stride : int
        Stride for raw spectrum subsampling.

    Returns
    -------
    np.ndarray
        Fixed-length feature vector.
    """
    # Ensure 1D
    spectrum = spectrum.flatten().astype(np.float32)

    features = np.concatenate([
        extract_peak_features(spectrum, n_peaks=n_peaks),
        extract_peak_ratios(spectrum, n_peaks=n_peaks, n_ratios=n_ratios),
        extract_band_areas(spectrum),
        extract_statistical_features(spectrum),
        subsample_spectrum(spectrum, stride=subsample_stride),
    ])

    return features


def get_spectral_feature_dim(n_peaks: int = 20, n_ratios: int = 10,
                              subsample_stride: int = 32) -> int:
    """Return the dimensionality of the spectral feature vector."""
    peak_dim = 3 * n_peaks        # positions + intensities + FWHM
    ratio_dim = n_ratios
    band_dim = len(RAMAN_WINDOWS)
    stat_dim = 8
    subsample_dim = N_POINTS // subsample_stride
    return peak_dim + ratio_dim + band_dim + stat_dim + subsample_dim
