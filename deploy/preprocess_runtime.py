"""
Spec2Prop-Edge: Runtime Preprocessing
======================================
Preprocesses raw Raman and XRD spectral files into fixed-length vectors
identical to the training pipeline.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_spectral_csv(file_path: str) -> tuple:
    """
    Read a two-column spectral file (CSV or TXT).
    Auto-detects delimiter and header. Returns (x_array, y_array).
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Spectral file not found: {file_path}")

    # Try common delimiters
    for sep in [",", "\t", ";", " "]:
        try:
            df = pd.read_csv(file_path, sep=sep, comment="#", engine="python")
            if df.shape[1] >= 2:
                break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not parse spectral file with any common delimiter: {file_path}")

    # If headers are numeric, the file has no header — re-read
    try:
        float(df.columns[0])
        df = pd.read_csv(file_path, sep=sep, header=None, comment="#", engine="python")
    except (ValueError, TypeError):
        pass

    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns, got {df.shape[1]} in {file_path}")

    # Take first two numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        # Try converting
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        raise ValueError(f"Need at least 2 numeric columns, found {len(numeric_cols)}")

    x = df[numeric_cols[0]].values.astype(np.float64)
    y = df[numeric_cols[1]].values.astype(np.float64)

    # Remove NaN/inf
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < 10:
        raise ValueError(f"Too few valid data points ({len(x)}) after cleaning in {file_path}")

    return x, y


def _remove_duplicates(x, y):
    """Remove duplicate x values by averaging y."""
    df = pd.DataFrame({"x": x, "y": y})
    df = df.groupby("x", as_index=False).mean()
    df = df.sort_values("x")
    return df["x"].values, df["y"].values


def _baseline_correction_polynomial(x, y, degree=4):
    """Simple polynomial baseline correction."""
    coeffs = np.polyfit(x, y, degree)
    baseline = np.polyval(coeffs, x)
    corrected = y - baseline
    return np.clip(corrected, 0, None)


def _normalize(y, method="max"):
    """Normalize intensity."""
    if method == "max":
        mx = np.max(np.abs(y))
        if mx > 0:
            return y / mx
        return y
    elif method == "minmax":
        mn, mx = np.min(y), np.max(y)
        if mx - mn > 0:
            return (y - mn) / (mx - mn)
        return y - mn
    elif method == "l2":
        norm = np.linalg.norm(y)
        if norm > 0:
            return y / norm
        return y
    else:
        return y


def _validate_output(vector, n_points, file_path):
    """Validate the preprocessed output vector."""
    issues = []
    if len(vector) != n_points:
        issues.append(f"Output length {len(vector)} != expected {n_points}")
    if np.any(np.isnan(vector)):
        issues.append("Output contains NaN values")
    if np.any(np.isinf(vector)):
        issues.append("Output contains Inf values")
    if np.all(vector == 0):
        issues.append("Output is all zeros — spectrum may be empty or corrupt")
    if np.std(vector) < 1e-8:
        issues.append("Output is nearly flat — spectrum may lack features")
    if np.max(vector) > 1.5:
        warnings.warn(f"Max intensity {np.max(vector):.3f} > 1.5 after normalization for {file_path}")

    if issues:
        raise ValueError(f"Preprocessing validation failed for {file_path}: " + "; ".join(issues))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_raman_file(
    file_path: str,
    wn_min: float = 100,
    wn_max: float = 4000,
    n_points: int = 2048,
    baseline_correction: bool = True,
    smoothing: bool = True,
    smoothing_window: int = 11,
    smoothing_polyorder: int = 3,
    normalization: str = "max",
) -> tuple:
    """
    Preprocess a raw Raman spectral file to a fixed-length vector.

    Parameters
    ----------
    file_path : str
        Path to a CSV/TXT file with (wavenumber, intensity) columns.
    wn_min, wn_max : float
        Wavenumber range to keep.
    n_points : int
        Number of output points (must match training = 2048).
    baseline_correction : bool
        Apply polynomial baseline correction.
    smoothing : bool
        Apply Savitzky-Golay smoothing.
    normalization : str
        Normalization method: 'max', 'minmax', 'l2', or None.

    Returns
    -------
    vector : np.ndarray, shape (n_points,)
        Preprocessed spectral vector.
    metadata : dict
        Preprocessing metadata for audit trail.
    """
    x, y = _read_spectral_csv(file_path)
    x, y = _remove_duplicates(x, y)

    # Sort ascending
    order = np.argsort(x)
    x, y = x[order], y[order]

    metadata = {
        "input_file": os.path.basename(file_path),
        "raw_points": len(x),
        "raw_wn_range": [float(x[0]), float(x[-1])],
        "preprocessing": {},
    }

    # Clip to wavenumber range
    mask = (x >= wn_min) & (x <= wn_max)
    x, y = x[mask], y[mask]
    metadata["preprocessing"]["wn_range"] = [wn_min, wn_max]
    metadata["preprocessing"]["points_after_clip"] = len(x)

    if len(x) < 10:
        raise ValueError(f"Only {len(x)} points in [{wn_min}, {wn_max}] range for {file_path}")

    # Baseline correction
    if baseline_correction:
        y = _baseline_correction_polynomial(x, y)
        metadata["preprocessing"]["baseline_correction"] = "polynomial_deg4"

    # Smoothing
    if smoothing and len(y) > smoothing_window:
        y = savgol_filter(y, window_length=smoothing_window, polyorder=smoothing_polyorder)
        y = np.clip(y, 0, None)
        metadata["preprocessing"]["smoothing"] = f"savgol_w{smoothing_window}_p{smoothing_polyorder}"

    # Interpolate to fixed grid
    target_x = np.linspace(wn_min, wn_max, n_points)
    interp_fn = interp1d(x, y, kind="linear", bounds_error=False, fill_value=0.0)
    vector = interp_fn(target_x).astype(np.float32)

    # Normalize
    if normalization:
        vector = _normalize(vector, normalization)
        metadata["preprocessing"]["normalization"] = normalization

    metadata["preprocessing"]["n_points"] = n_points
    metadata["output_range"] = [float(np.min(vector)), float(np.max(vector))]

    # Validate
    _validate_output(vector, n_points, file_path)

    return vector, metadata


def preprocess_xrd_file(
    file_path: str,
    x_min: float = None,
    x_max: float = None,
    n_points: int = 2048,
    baseline_correction: bool = False,
    smoothing: bool = True,
    smoothing_window: int = 11,
    smoothing_polyorder: int = 3,
    normalization: str = "max",
) -> tuple:
    """
    Preprocess a raw XRD spectral file to a fixed-length vector.

    Parameters
    ----------
    file_path : str
        Path to a CSV/TXT file with (2theta, intensity) columns.
    x_min, x_max : float or None
        2theta range. If None, uses full data range.
    n_points : int
        Number of output points (must match training = 2048).

    Returns
    -------
    vector : np.ndarray, shape (n_points,)
    metadata : dict
    """
    x, y = _read_spectral_csv(file_path)
    x, y = _remove_duplicates(x, y)

    order = np.argsort(x)
    x, y = x[order], y[order]

    if x_min is None:
        x_min = float(x[0])
    if x_max is None:
        x_max = float(x[-1])

    metadata = {
        "input_file": os.path.basename(file_path),
        "raw_points": len(x),
        "raw_x_range": [float(x[0]), float(x[-1])],
        "preprocessing": {},
    }

    mask = (x >= x_min) & (x <= x_max)
    x, y = x[mask], y[mask]
    metadata["preprocessing"]["x_range"] = [x_min, x_max]
    metadata["preprocessing"]["points_after_clip"] = len(x)

    if len(x) < 10:
        raise ValueError(f"Only {len(x)} points in [{x_min}, {x_max}] range for {file_path}")

    if baseline_correction:
        y = _baseline_correction_polynomial(x, y)
        metadata["preprocessing"]["baseline_correction"] = "polynomial_deg4"

    if smoothing and len(y) > smoothing_window:
        y = savgol_filter(y, window_length=smoothing_window, polyorder=smoothing_polyorder)
        y = np.clip(y, 0, None)
        metadata["preprocessing"]["smoothing"] = f"savgol_w{smoothing_window}_p{smoothing_polyorder}"

    target_x = np.linspace(x_min, x_max, n_points)
    interp_fn = interp1d(x, y, kind="linear", bounds_error=False, fill_value=0.0)
    vector = interp_fn(target_x).astype(np.float32)

    if normalization:
        vector = _normalize(vector, normalization)
        metadata["preprocessing"]["normalization"] = normalization

    metadata["preprocessing"]["n_points"] = n_points
    metadata["output_range"] = [float(np.min(vector)), float(np.max(vector))]

    _validate_output(vector, n_points, file_path)

    return vector, metadata


def build_descriptor_vector(
    descriptor_json: dict,
    family_encoder: dict = None,
) -> np.ndarray:
    """
    Build a descriptor feature vector from a JSON dictionary,
    matching the training-time feature extraction order in models/dataset.py.

    Parameters
    ----------
    descriptor_json : dict
        Keys matching DESCRIPTOR_BOOL_COLS and DESCRIPTOR_FLOAT_COLS.
    family_encoder : dict
        Label encoder for chemical_family_model (for one-hot encoding).

    Returns
    -------
    np.ndarray
        Descriptor feature vector.
    """
    BOOL_COLS = [
        "contains_transition_metal",
        "contains_3d_transition_metal",
        "contains_4d_transition_metal",
        "contains_5d_transition_metal",
    ]
    FLOAT_COLS = [
        "n_elements",
        "avg_electronegativity",
        "avg_atomic_mass",
    ]

    feats = []
    for col in BOOL_COLS:
        val = descriptor_json.get(col, False)
        feats.append(float(bool(val)))
    for col in FLOAT_COLS:
        val = descriptor_json.get(col, 0.0)
        feats.append(float(val) if val is not None else 0.0)

    if family_encoder is not None:
        fam = str(descriptor_json.get("chemical_family_model", ""))
        onehot = [0.0] * len(family_encoder)
        idx = family_encoder.get(fam, -1)
        if idx >= 0:
            onehot[idx] = 1.0
        feats.extend(onehot)

    return np.array(feats, dtype=np.float32)
