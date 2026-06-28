"""
Spec2Prop-Edge: Sample Loader
==============================
Loads real test samples from the processed dataset.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple

from app.backend.config import (
    METADATA_PKL, FULL_SPECTRA_PKL, RAMAN_PARQUET, XRD_PARQUET,
    SPLITS_DIR, RAMAN_WN_MIN, RAMAN_WN_MAX, RAMAN_N_POINTS,
    XRD_2THETA_MIN, XRD_2THETA_MAX, XRD_N_POINTS, TARGET_COL,
)

MAP_12_TO_9 = {
    'Silicate': 'Silicate', 'Oxide': 'Oxide', 'Sulfate': 'Sulfate',
    'Phosphate': 'Phosphate', 'Sulfide': 'Sulfide', 'Halide': 'Halide',
    'Carbonate': 'Carbonate', 'Borate': 'Borate', 'Elemental/Native': 'Other/Rare',
    'Telluride/Selenide': 'Other/Rare', 'Intermetallic/Alloy': 'Other/Rare', 'Carbide': 'Other/Rare'
}

MAP_12_TO_5 = {
    'Silicate': 'Silicate', 'Oxide': 'Oxide', 'Sulfate': 'Oxyanion Group',
    'Phosphate': 'Oxyanion Group', 'Carbonate': 'Oxyanion Group', 'Borate': 'Oxyanion Group',
    'Sulfide': 'Sulfide/Halide Group', 'Halide': 'Sulfide/Halide Group',
    'Telluride/Selenide': 'Sulfide/Halide Group', 'Elemental/Native': 'Other/Rare',
    'Intermetallic/Alloy': 'Other/Rare', 'Carbide': 'Other/Rare'
}


class SampleLoader:
    def __init__(self):
        self.meta: Optional[pd.DataFrame] = None
        self.raman_vectors: Optional[Dict[str, np.ndarray]] = None
        self.xrd_vectors: Optional[Dict[str, np.ndarray]] = None
        self.test_ids: Optional[set] = None
        self._loaded = False

    def load(self) -> bool:
        try:
            test_csv = SPLITS_DIR / "test.csv"
            if not test_csv.exists():
                print(f"[SampleLoader] ERROR: test split not found: {test_csv}")
                return False
            test_df = pd.read_csv(test_csv)
            self.test_ids = set(test_df["sample_id"].astype(str).values)

            if not METADATA_PKL.exists():
                print(f"[SampleLoader] ERROR: metadata not found: {METADATA_PKL}")
                return False
            meta = pd.read_pickle(METADATA_PKL)
            meta["sample_id"] = meta["sample_id"].astype(str)

            self.meta = meta[meta["sample_id"].isin(self.test_ids)].reset_index(drop=True)
            self._load_raman_vectors()
            self._load_xrd_vectors()
            self._loaded = True
            return True
        except Exception as e:
            print(f"[SampleLoader] ERROR during loading: {e}")
            return False

    def _load_raman_vectors(self):
        if not RAMAN_PARQUET.exists():
            self.raman_vectors = {}
            return

        raman_df = pd.read_parquet(RAMAN_PARQUET)
        raman_cols = sorted(
            [c for c in raman_df.columns if c.startswith("raman_")],
            key=lambda c: int(c.replace("raman_", ""))
        )

        if "sample_id" not in raman_df.columns and FULL_SPECTRA_PKL.exists():
            full_meta = pd.read_pickle(FULL_SPECTRA_PKL)
            full_meta["sample_id"] = full_meta["sample_id"].astype(str)
            if len(full_meta) == len(raman_df):
                raman_df["sample_id"] = full_meta["sample_id"].values
            else:
                if "rruff_id" in raman_df.columns:
                    raman_df["sample_id"] = raman_df["rruff_id"].astype(str)
        elif "sample_id" not in raman_df.columns:
            if "rruff_id" in raman_df.columns:
                raman_df["sample_id"] = raman_df["rruff_id"].astype(str)
            else:
                raman_df["sample_id"] = [str(i) for i in range(len(raman_df))]

        raman_df["sample_id"] = raman_df["sample_id"].astype(str)
        raman_test = raman_df[raman_df["sample_id"].isin(self.test_ids)]
        self.raman_vectors = {}
        for _, row in raman_test.iterrows():
            self.raman_vectors[str(row["sample_id"])] = row[raman_cols].values.astype(np.float32)

    def _load_xrd_vectors(self):
        self.xrd_vectors = {}
        if not XRD_PARQUET.exists(): return
        xrd_df = pd.read_parquet(XRD_PARQUET)
        xrd_cols = sorted(
            [c for c in xrd_df.columns if c.startswith("xrd_") and c[4:].isdigit()],
            key=lambda c: int(c.replace("xrd_", ""))
        )
        if not xrd_cols or "rruff_id" not in xrd_df.columns: return

        rruff_to_sid = {}
        for _, row in self.meta.iterrows():
            rruff_to_sid[str(row.get("rruff_id", ""))] = str(row["sample_id"])

        for _, row in xrd_df.iterrows():
            rid = str(row["rruff_id"])
            if rid in rruff_to_sid:
                vec = row[xrd_cols].values.astype(np.float32)
                if np.any(vec != 0):
                    self.xrd_vectors[rruff_to_sid[rid]] = vec

    @property
    def is_loaded(self) -> bool: return self._loaded
    @property
    def num_samples(self) -> int: return len(self.meta) if self.meta is not None else 0

    def get_sample_list(self, limit: int = 50) -> list:
        if not self._loaded: return []
        samples = []
        for _, row in self.meta.head(limit).iterrows():
            sid = str(row["sample_id"])
            orig_label = str(row.get("chemical_family", "Unknown"))
            samples.append({
                "sample_id": sid,
                "rruff_id": str(row.get("rruff_id", sid)),
                "mineral_name": str(row.get("mineral_name", "Unknown")),
                "original_12class_label": orig_label,
                "true_9class_label": MAP_12_TO_9.get(orig_label, "Other/Rare"),
                "true_5class_label": MAP_12_TO_5.get(orig_label, "Other/Rare"),
                "has_raman": sid in self.raman_vectors if self.raman_vectors else False,
                "has_xrd": sid in self.xrd_vectors if self.xrd_vectors else False,
                "subset": "clean_inorganic",
            })
        return samples

    def get_sample_detail(self, sample_id: str) -> Optional[dict]:
        if not self._loaded or self.meta is None: return None
        row = self.meta[self.meta["sample_id"] == sample_id]
        if row.empty or sample_id not in self.test_ids: return None
        row = row.iloc[0]

        raman_spec = None
        if self.raman_vectors and sample_id in self.raman_vectors:
            vec = self.raman_vectors[sample_id]
            raman_spec = {
                "x": np.linspace(RAMAN_WN_MIN, RAMAN_WN_MAX, len(vec)).tolist(),
                "y": vec.tolist(),
                "x_label": "Raman Shift (cm⁻¹)",
                "y_label": "Intensity (a.u.)",
            }

        xrd_spec = None
        if self.xrd_vectors and sample_id in self.xrd_vectors:
            vec = self.xrd_vectors[sample_id]
            xrd_spec = {
                "x": np.linspace(XRD_2THETA_MIN, XRD_2THETA_MAX, len(vec)).tolist(),
                "y": vec.tolist(),
                "x_label": "2θ (degrees)",
                "y_label": "Intensity (a.u.)",
            }

        meta_dict = {}
        for col in ["rruff_id", "mineral_name", "compound_name", "chemical_family", "original_formula", "has_xrd"]:
            val = row.get(col, None)
            if pd.notna(val): meta_dict[col] = str(val)

        orig_label = str(row.get("chemical_family", "Unknown"))
        return {
            "sample_id": sample_id,
            "rruff_id": str(row.get("rruff_id", sample_id)),
            "mineral_name": str(row.get("mineral_name", "Unknown")),
            "formula": str(row.get("original_formula", row.get("display_formula_clean", "N/A"))),
            "original_12class_label": orig_label,
            "true_9class_label": MAP_12_TO_9.get(orig_label, "Other/Rare"),
            "true_5class_label": MAP_12_TO_5.get(orig_label, "Other/Rare"),
            "has_xrd": sample_id in self.xrd_vectors if self.xrd_vectors else False,
            "raman_spectrum": raman_spec,
            "xrd_pattern": xrd_spec,
            "metadata": meta_dict,
        }

    def get_raman_vector(self, sample_id: str) -> Optional[np.ndarray]:
        if self.raman_vectors and sample_id in self.raman_vectors: return self.raman_vectors[sample_id]
        return None

    def get_true_label(self, sample_id: str) -> Optional[str]:
        if self.meta is None: return None
        row = self.meta[self.meta["sample_id"] == sample_id]
        if row.empty: return None
        return str(row.iloc[0].get("chemical_family", "Unknown"))

    def get_random_sample(self) -> Optional[dict]:
        if not self._loaded or self.meta is None: return None
        valid_ids = [sid for sid in self.test_ids if self.raman_vectors and sid in self.raman_vectors]
        if not valid_ids: return None
        sid = np.random.choice(valid_ids)
        row = self.meta[self.meta["sample_id"] == sid].iloc[0]
        orig_label = str(row.get("chemical_family", "Unknown"))
        return {
            "sample_id": sid,
            "rruff_id": str(row.get("rruff_id", sid)),
            "mineral_name": str(row.get("mineral_name", "Unknown")),
            "original_12class_label": orig_label,
            "true_9class_label": MAP_12_TO_9.get(orig_label, "Other/Rare"),
            "true_5class_label": MAP_12_TO_5.get(orig_label, "Other/Rare"),
            "has_raman": True,
            "has_xrd": sid in self.xrd_vectors if self.xrd_vectors else False,
            "subset": "clean_inorganic",
        }

    def validate_is_test_sample(self, sample_id: str) -> bool:
        return self.test_ids is not None and sample_id in self.test_ids
