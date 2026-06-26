"""
Spec2Prop-Inorg Dataset
=======================
PyTorch Dataset class that loads metadata PKL, joins with Raman parquet
by sample_id at runtime, and provides spectral + descriptor + target tensors.

Supports optional .npy cache for faster repeated loading.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAMAN_PREFIX = "raman_"
XRD_PREFIX = "xrd_"

DESCRIPTOR_BOOL_COLS = [
    "contains_transition_metal",
    "contains_3d_transition_metal",
    "contains_4d_transition_metal",
    "contains_5d_transition_metal",
]
DESCRIPTOR_FLOAT_COLS = [
    "n_elements",
    "avg_electronegativity",
    "avg_atomic_mass",
]

# Rare families to merge into "Other/Rare" for model training (v1)
RARE_FAMILIES = {"Carbide", "Intermetallic/Alloy", "Telluride/Selenide", "Elemental/Native"}

# Minimum class threshold — families below this are merged
MIN_FAMILY_SIZE = 20


# ---------------------------------------------------------------------------
# Label encoding helpers
# ---------------------------------------------------------------------------

def build_label_encoder(series: pd.Series, drop_nan: bool = True) -> Dict:
    """Build label -> int mapping from a pandas Series."""
    vals = series.dropna().unique() if drop_nan else series.unique()
    vals = sorted([str(v) for v in vals])
    return {v: i for i, v in enumerate(vals)}


def save_label_encoder(encoder: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(encoder, f, indent=2)


def load_label_encoder(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Descriptor feature extraction
# ---------------------------------------------------------------------------

def extract_descriptors(row: pd.Series, family_encoder: Optional[Dict] = None) -> np.ndarray:
    """Extract a fixed-length descriptor vector from a metadata row."""
    feats = []
    # Boolean flags
    for col in DESCRIPTOR_BOOL_COLS:
        val = row.get(col, False)
        feats.append(float(bool(val)))
    # Float features
    for col in DESCRIPTOR_FLOAT_COLS:
        val = row.get(col, 0.0)
        feats.append(float(val) if pd.notna(val) else 0.0)
    # Chemical family one-hot
    if family_encoder is not None:
        fam = str(row.get("chemical_family_model", row.get("chemical_family", "")))
        onehot = [0.0] * len(family_encoder)
        idx = family_encoder.get(fam, -1)
        if idx >= 0:
            onehot[idx] = 1.0
        feats.extend(onehot)
    return np.array(feats, dtype=np.float32)


def get_descriptor_dim(family_encoder: Optional[Dict] = None) -> int:
    """Return dimensionality of the descriptor vector."""
    dim = len(DESCRIPTOR_BOOL_COLS) + len(DESCRIPTOR_FLOAT_COLS)
    if family_encoder is not None:
        dim += len(family_encoder)
    return dim


# ---------------------------------------------------------------------------
# Model-level family mapping
# ---------------------------------------------------------------------------

def add_model_family_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add chemical_family_model column, merging rare families into Other/Rare."""
    df = df.copy()
    df["chemical_family_original"] = df["chemical_family"].copy()
    counts = df["chemical_family"].value_counts()
    rare = set(counts[counts < MIN_FAMILY_SIZE].index) | RARE_FAMILIES
    df["chemical_family_model"] = df["chemical_family"].apply(
        lambda x: "Other/Rare" if x in rare else x
    )
    return df


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class Spec2PropDataset(Dataset):
    """
    PyTorch Dataset for Spec2Prop-Inorg.

    Loads metadata pkl, joins with Raman parquet by sample_id,
    optionally loads XRD vectors, extracts descriptors and targets.

    Parameters
    ----------
    metadata_pkl : str
        Path to the subset metadata pkl file.
    split_csv : str
        Path to the split CSV (contains sample_id column).
    raman_parquet : str
        Path to spec2prop_raman.parquet with raman_0..raman_2047.
    target_cols : list of str
        Target column names for classification/regression.
    label_encoders : dict of {col_name: {label: int}}
        Pre-built label encoders for categorical targets.
    use_descriptors : bool
        Whether to extract chemistry descriptor features.
    family_encoder : dict, optional
        Label encoder for chemical_family (used in descriptors one-hot).
    use_xrd : bool
        Whether to load XRD vectors.
    xrd_pkl : str, optional
        Path to the XRD-linked pkl if use_xrd is True.
    cache_dir : str, optional
        If set, load/save .npy cache for fast access.
    n_raman_points : int
        Expected number of Raman spectral points (default 2048).
    n_xrd_points : int
        Expected number of XRD spectral points (default 2048).
    """

    def __init__(
        self,
        metadata_pkl: str,
        split_csv: str,
        raman_parquet: str,
        target_cols: List[str],
        label_encoders: Dict[str, Dict],
        use_descriptors: bool = False,
        family_encoder: Optional[Dict] = None,
        use_xrd: bool = False,
        xrd_pkl: Optional[str] = None,
        cache_dir: Optional[str] = None,
        n_raman_points: int = 2048,
        n_xrd_points: int = 2048,
        random_shift: bool = False,
        add_noise: bool = False,
    ):
        super().__init__()
        self.target_cols = target_cols
        self.label_encoders = label_encoders
        self.use_descriptors = use_descriptors
        self.family_encoder = family_encoder
        self.use_xrd = use_xrd
        self.n_raman_points = n_raman_points
        self.n_xrd_points = n_xrd_points
        self.random_shift = random_shift
        self.add_noise = add_noise

        # --- Load split sample IDs ---
        split_df = pd.read_csv(split_csv)
        split_ids = set(split_df["sample_id"].astype(str).values)

        # --- Check for .npy cache ---
        cache_loaded = False
        if cache_dir and os.path.isdir(cache_dir):
            cache_loaded = self._try_load_cache(cache_dir, split_ids)

        if not cache_loaded:
            self._load_from_source(
                metadata_pkl, raman_parquet, split_ids,
                xrd_pkl, cache_dir
            )

    def _try_load_cache(self, cache_dir: str, split_ids: set) -> bool:
        """Attempt to load pre-built .npy cache."""
        raman_path = os.path.join(cache_dir, "raman.npy")
        meta_path = os.path.join(cache_dir, "meta.pkl")
        if not (os.path.isfile(raman_path) and os.path.isfile(meta_path)):
            return False

        self.raman_data = np.load(raman_path)
        meta = pd.read_pickle(meta_path)
        # Filter to split
        mask = meta["sample_id"].astype(str).isin(split_ids)
        indices = np.where(mask.values)[0]
        self.raman_data = self.raman_data[indices]
        self.meta = meta.iloc[indices].reset_index(drop=True)

        if self.use_xrd:
            xrd_path = os.path.join(cache_dir, "xrd.npy")
            if os.path.isfile(xrd_path):
                self.xrd_data = np.load(xrd_path)[indices]
            else:
                self.xrd_data = np.zeros(
                    (len(self.meta), self.n_xrd_points), dtype=np.float32
                )

        self._build_targets()
        self._build_descriptors()
        return True

    def _load_from_source(
        self, metadata_pkl, raman_parquet, split_ids, xrd_pkl, cache_dir
    ):
        """Load data from original pkl + parquet sources."""
        # --- Load metadata ---
        meta = pd.read_pickle(metadata_pkl)
        meta["sample_id"] = meta["sample_id"].astype(str)

        # Add model-level family column
        meta = add_model_family_column(meta)

        # Filter to split
        meta = meta[meta["sample_id"].isin(split_ids)].reset_index(drop=True)

        # --- Load Raman vectors ---
        raman_df = pd.read_parquet(raman_parquet)
        # The parquet may not have sample_id; construct it from rruff_id
        # or match by index position. Let's check what columns are available.
        raman_cols = sorted(
            [c for c in raman_df.columns if c.startswith(RAMAN_PREFIX)],
            key=lambda c: int(c.replace(RAMAN_PREFIX, ""))
        )

        if "sample_id" not in raman_df.columns:
            # Build sample_id from the raman parquet the same way the
            # preprocessing script does: rruff_id is the sample_id for
            # single-variant groups, else rruff_id_v{n}.
            # Since the parquet was saved from df_raman which HAS sample_id
            # as a column originally, let's check other ID columns.
            if "rruff_id" in raman_df.columns:
                # We need to match by rruff_id + mineral_name or similar.
                # But the simplest approach: re-read the full spectra pkl
                # which has sample_id, and use it as an index bridge.
                full_pkl_path = os.path.join(
                    os.path.dirname(metadata_pkl), "spec2prop_full_spectra.pkl"
                )
                if os.path.isfile(full_pkl_path):
                    full_meta = pd.read_pickle(full_pkl_path)
                    full_meta["sample_id"] = full_meta["sample_id"].astype(str)
                    # The parquet and full_meta should have same row order
                    if len(full_meta) == len(raman_df):
                        raman_df["sample_id"] = full_meta["sample_id"].values
                    else:
                        # Fallback: merge on rruff_id + source_file
                        raman_df["sample_id"] = raman_df.get(
                            "rruff_id", pd.Series(range(len(raman_df)))
                        ).astype(str)
                else:
                    raman_df["sample_id"] = raman_df["rruff_id"].astype(str)
            else:
                raman_df["sample_id"] = [str(i) for i in range(len(raman_df))]

        raman_df["sample_id"] = raman_df["sample_id"].astype(str)

        # Filter to split IDs and merge
        raman_split = raman_df[raman_df["sample_id"].isin(split_ids)][
            ["sample_id"] + raman_cols
        ].reset_index(drop=True)

        # Merge meta with raman
        merged = meta.merge(raman_split, on="sample_id", how="inner")

        self.meta = merged.drop(columns=raman_cols).reset_index(drop=True)
        self.raman_data = merged[raman_cols].values.astype(np.float32)

        # --- Load XRD vectors if needed ---
        if self.use_xrd and xrd_pkl:
            xrd_df = pd.read_pickle(xrd_pkl)
            xrd_cols = sorted(
                [c for c in xrd_df.columns if c.startswith(XRD_PREFIX) and c[len(XRD_PREFIX):].isdigit()],
                key=lambda c: int(c.replace(XRD_PREFIX, ""))
            )
            if "sample_id" not in xrd_df.columns:
                if "rruff_id" in xrd_df.columns:
                    # XRD is keyed by rruff_id; merge through meta
                    pass  # handled below
            
            # Match XRD by rruff_id through metadata
            if xrd_cols:
                xrd_subset = xrd_df[["rruff_id"] + xrd_cols].drop_duplicates("rruff_id")
                rruff_to_xrd = {}
                for _, row in xrd_subset.iterrows():
                    rruff_to_xrd[str(row["rruff_id"])] = row[xrd_cols].values.astype(np.float32)
                
                xrd_arrays = []
                for _, row in self.meta.iterrows():
                    rid = str(row.get("rruff_id", ""))
                    if rid in rruff_to_xrd:
                        xrd_arrays.append(rruff_to_xrd[rid])
                    else:
                        xrd_arrays.append(np.zeros(len(xrd_cols), dtype=np.float32))
                self.xrd_data = np.stack(xrd_arrays)
            else:
                self.xrd_data = np.zeros(
                    (len(self.meta), self.n_xrd_points), dtype=np.float32
                )
        elif self.use_xrd:
            self.xrd_data = np.zeros(
                (len(self.meta), self.n_xrd_points), dtype=np.float32
            )

        self._build_targets()
        self._build_descriptors()

    def _build_targets(self):
        """Encode target columns into integer/float arrays with NaN masks."""
        self.targets = {}
        self.target_masks = {}

        for col in self.target_cols:
            if col not in self.meta.columns:
                # All NaN
                self.targets[col] = np.full(len(self.meta), -1, dtype=np.int64)
                self.target_masks[col] = np.zeros(len(self.meta), dtype=np.float32)
                continue

            series = self.meta[col]
            valid_mask = series.notna()

            if col in self.label_encoders:
                enc = self.label_encoders[col]
                encoded = np.full(len(series), -1, dtype=np.int64)
                for i, val in enumerate(series):
                    if pd.notna(val) and str(val) in enc:
                        encoded[i] = enc[str(val)]
                    else:
                        valid_mask.iloc[i] = False
                self.targets[col] = encoded
            else:
                # Treat as float regression target
                vals = series.fillna(0.0).astype(np.float32).values
                self.targets[col] = vals

            self.target_masks[col] = valid_mask.astype(np.float32).values

    def _build_descriptors(self):
        """Pre-compute descriptor vectors for all rows."""
        if self.use_descriptors:
            descs = []
            for i in range(len(self.meta)):
                descs.append(extract_descriptors(
                    self.meta.iloc[i], self.family_encoder
                ))
            self.descriptor_data = np.stack(descs)
        else:
            self.descriptor_data = None

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = {}
        # Raman spectrum: shape (1, n_points) for Conv1d
        raman_arr = self.raman_data[idx: idx + 1] if self.raman_data.ndim == 1 else self.raman_data[idx]
        
        # Apply augmentations if enabled
        if getattr(self, "random_shift", False) and np.random.rand() < 0.5:
            shift = np.random.randint(-15, 15)
            raman_arr = np.roll(raman_arr, shift)
        if getattr(self, "add_noise", False) and np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.05, size=raman_arr.shape)
            raman_arr = raman_arr + noise
            
        result["raman"] = torch.from_numpy(raman_arr).float().unsqueeze(0)

        # XRD spectrum
        if self.use_xrd and hasattr(self, "xrd_data"):
            result["xrd"] = torch.from_numpy(
                self.xrd_data[idx]
            ).float().unsqueeze(0)

        # Descriptors
        if self.use_descriptors and self.descriptor_data is not None:
            result["descriptors"] = torch.from_numpy(
                self.descriptor_data[idx]
            ).float()

        # Targets + masks
        for col in self.target_cols:
            t = self.targets[col][idx]
            m = self.target_masks[col][idx]
            if isinstance(t, (np.integer, int)):
                result[f"target_{col}"] = torch.tensor(t, dtype=torch.long)
            else:
                result[f"target_{col}"] = torch.tensor(t, dtype=torch.float)
            result[f"mask_{col}"] = torch.tensor(m, dtype=torch.float)

        result["sample_id"] = str(self.meta.iloc[idx]["sample_id"])
        return result

    def get_class_weights(self, col: str) -> Optional[torch.Tensor]:
        """Compute inverse-frequency class weights for a classification target."""
        if col not in self.targets or col not in self.label_encoders:
            return None
        labels = self.targets[col]
        mask = self.target_masks[col]
        valid_labels = labels[mask > 0]
        if len(valid_labels) == 0:
            return None
        n_classes = len(self.label_encoders[col])
        counts = np.bincount(valid_labels[valid_labels >= 0], minlength=n_classes).astype(np.float32)
        counts = np.maximum(counts, 1.0)  # avoid div-by-zero
        weights = 1.0 / counts
        weights = weights / weights.sum() * n_classes  # normalize
        return torch.from_numpy(weights).float()

    @property
    def num_raman_points(self) -> int:
        return self.raman_data.shape[1] if self.raman_data.ndim == 2 else self.n_raman_points

    @property
    def num_xrd_points(self) -> int:
        if hasattr(self, "xrd_data") and self.xrd_data is not None:
            return self.xrd_data.shape[1] if self.xrd_data.ndim == 2 else self.n_xrd_points
        return self.n_xrd_points

    @property
    def descriptor_dim(self) -> int:
        if self.descriptor_data is not None:
            return self.descriptor_data.shape[1]
        return get_descriptor_dim(self.family_encoder)
