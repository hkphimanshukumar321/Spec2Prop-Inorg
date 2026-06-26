import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Spec2Prop-Inorg: Build Training Cache
=====================================
Pre-compiles the joined Raman/XRD arrays and metadata into .npy/.pkl caches
for fast data loading during model training. This avoids doing the pandas
merges at runtime.
"""

import argparse
import os
import pandas as pd
import numpy as np

RAMAN_PREFIX = "raman_"
XRD_PREFIX = "xrd_"

def build_cache(metadata_pkl, split_csv, raman_parquet, out_dir, xrd_pkl=None):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Building cache for {os.path.basename(metadata_pkl)} -> {out_dir}")
    
    # 1. Load splits
    split_df = pd.read_csv(split_csv)
    # Get train, val, test subsets
    for split_name in ["train", "val", "test"]:
        split_path = split_csv.replace("train.csv", f"{split_name}.csv")
        if not os.path.exists(split_path):
            continue
            
        print(f"  -> Processing {split_name} split...")
        sub_split_df = pd.read_csv(split_path)
        split_ids = set(sub_split_df["sample_id"].astype(str))
        
        # We save separate caches for each split to make loading extremely fast
        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # 2. Load Metadata
        meta = pd.read_pickle(metadata_pkl)
        meta["sample_id"] = meta["sample_id"].astype(str)
        meta = meta[meta["sample_id"].isin(split_ids)].reset_index(drop=True)
        
        # 3. Load Raman
        raman_df = pd.read_parquet(raman_parquet)
        if "sample_id" not in raman_df.columns:
            # Reconstruct sample_id (fallback to index string)
            full_pkl_path = os.path.join(os.path.dirname(metadata_pkl), "spec2prop_full_spectra.pkl")
            if os.path.exists(full_pkl_path):
                full_meta = pd.read_pickle(full_pkl_path)
                if len(full_meta) == len(raman_df):
                    raman_df["sample_id"] = full_meta["sample_id"].astype(str).values
                else:
                    raman_df["sample_id"] = raman_df.get("rruff_id", pd.Series(range(len(raman_df)))).astype(str)
            else:
                raman_df["sample_id"] = raman_df.get("rruff_id", pd.Series(range(len(raman_df)))).astype(str)
        
        raman_df["sample_id"] = raman_df["sample_id"].astype(str)
        
        # Filter and extract Raman array
        raman_split = raman_df[raman_df["sample_id"].isin(split_ids)]
        raman_cols = sorted(
            [c for c in raman_df.columns if c.startswith(RAMAN_PREFIX)],
            key=lambda c: int(c.replace(RAMAN_PREFIX, ""))
        )
        
        # Merge exactly according to meta order
        merged = meta.merge(raman_split[["sample_id"] + raman_cols], on="sample_id", how="left")
        
        raman_array = merged[raman_cols].values.astype(np.float32)
        final_meta = merged.drop(columns=raman_cols)
        
        # 4. Extract XRD (if provided)
        if xrd_pkl:
            xrd_df = pd.read_pickle(xrd_pkl)
            xrd_cols = sorted(
                [c for c in xrd_df.columns if c.startswith(XRD_PREFIX) and c[len(XRD_PREFIX):].isdigit()],
                key=lambda c: int(c.replace(XRD_PREFIX, ""))
            )
            rruff_to_xrd = {}
            if xrd_cols:
                xrd_subset = xrd_df[["rruff_id"] + xrd_cols].drop_duplicates("rruff_id")
                for _, row in xrd_subset.iterrows():
                    rruff_to_xrd[str(row["rruff_id"])] = row[xrd_cols].values.astype(np.float32)
            
            xrd_arrays = []
            for _, row in final_meta.iterrows():
                rid = str(row.get("rruff_id", ""))
                if rid in rruff_to_xrd:
                    xrd_arrays.append(rruff_to_xrd[rid])
                else:
                    xrd_arrays.append(np.zeros(2048, dtype=np.float32))
            
            xrd_array = np.stack(xrd_arrays)
            np.save(os.path.join(split_dir, "xrd.npy"), xrd_array)
        
        # 5. Save
        np.save(os.path.join(split_dir, "raman.npy"), raman_array)
        final_meta.to_pickle(os.path.join(split_dir, "meta.pkl"))
        
        print(f"    Saved {len(final_meta)} rows. Raman shape: {raman_array.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()
    
    d = args.data_dir
    raman_pqt = os.path.join(d, "spec2prop_raman.parquet")
    xrd_pkl = os.path.join(d, "spec2prop_xrd_linked_inorganic.pkl")
    
    # Clean inorganic cache
    build_cache(
        metadata_pkl=os.path.join(d, "spec2prop_clean_inorganic.pkl"),
        split_csv=os.path.join(d, "splits/clean_inorganic/train.csv"),
        raman_parquet=raman_pqt,
        out_dir=os.path.join(d, "cache", "clean_inorganic")
    )
    
    # Property matched cache
    build_cache(
        metadata_pkl=os.path.join(d, "spec2prop_property_matched_inorganic.pkl"),
        split_csv=os.path.join(d, "splits/property_matched_inorganic/train.csv"),
        raman_parquet=raman_pqt,
        out_dir=os.path.join(d, "cache", "property_matched")
    )
    
    # XRD linked cache
    build_cache(
        metadata_pkl=os.path.join(d, "spec2prop_xrd_linked_inorganic.pkl"),
        split_csv=os.path.join(d, "splits/xrd_linked_inorganic/train.csv"),
        raman_parquet=raman_pqt,
        xrd_pkl=xrd_pkl,
        out_dir=os.path.join(d, "cache", "xrd_linked")
    )
    
    print("Done building caches.")
