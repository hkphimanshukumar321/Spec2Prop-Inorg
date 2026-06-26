import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

LOG_FMT = "[%(asctime)s] %(levelname)-7s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
log = logging.getLogger("make_splits")

def get_group_key(row):
    if pd.notna(row.get("rruff_id")) and row.get("rruff_id") != "":
        return row["rruff_id"]
    if pd.notna(row.get("reduced_formula")) and row.get("reduced_formula") != "":
        return row["reduced_formula"]
    return row.get("mineral_name", "unknown")

def make_splits(df, target_col=None):
    df = df.copy()
    df["group_key"] = df.apply(get_group_key, axis=1)
    
    # Stratify column just for logging/inspection, group split doesn't strict stratify
    if target_col and target_col in df.columns:
        stratify_col = df[target_col].astype(str) + "_" + df["chemical_family"].astype(str)
    else:
        stratify_col = df["chemical_family"].astype(str)
        
    df["stratify_key"] = stratify_col
    
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    
    train_idx, temp_idx = next(gss1.split(df, groups=df["group_key"]))
    
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["group_key"]))
    
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]
    
    return train_df, val_df, test_df

def process_subset(csv_path, out_dir, name, target_col=None):
    log.info(f"Processing subset: {name}")
    if not os.path.exists(csv_path):
        log.warning(f"{csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    train_df, val_df, test_df = make_splits(df, target_col)
    
    subset_dir = out_dir / "splits" / name
    subset_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.drop(columns=["group_key", "stratify_key"]).to_csv(subset_dir / "train.csv", index=False)
    val_df.drop(columns=["group_key", "stratify_key"]).to_csv(subset_dir / "val.csv", index=False)
    test_df.drop(columns=["group_key", "stratify_key"]).to_csv(subset_dir / "test.csv", index=False)
    
    summary = {
        "total": len(df),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "train_groups": train_df["group_key"].nunique(),
        "val_groups": val_df["group_key"].nunique(),
        "test_groups": test_df["group_key"].nunique(),
    }
    
    with open(subset_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    log.info(f"Done: {name}. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()
    
    out_dir = Path(args.processed_dir)
    
    process_subset(out_dir / "spec2prop_clean_inorganic_metadata.csv", out_dir, "clean_inorganic")
    process_subset(out_dir / "spec2prop_property_matched_inorganic_metadata.csv", out_dir, "property_matched_inorganic", target_col="band_gap_class")
    process_subset(out_dir / "spec2prop_xrd_linked_inorganic_metadata.csv", out_dir, "xrd_linked_inorganic")

if __name__ == "__main__":
    main()
