import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import logging
import os
from pathlib import Path
import pandas as pd

LOG_FMT = "[%(asctime)s] %(levelname)-7s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
log = logging.getLogger("check_leakage")

def check_leakage(subset_dir):
    log.info(f"Checking leakage in {subset_dir.name}")
    train_path = subset_dir / "train.csv"
    val_path = subset_dir / "val.csv"
    test_path = subset_dir / "test.csv"
    
    if not train_path.exists():
        log.warning(f"No splits found in {subset_dir}")
        return False
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    def get_group_keys(df):
        keys = set()
        for _, row in df.iterrows():
            if pd.notna(row.get("rruff_id")) and row.get("rruff_id") != "":
                keys.add("rruff:" + str(row["rruff_id"]))
            elif pd.notna(row.get("reduced_formula")) and row.get("reduced_formula") != "":
                keys.add("formula:" + str(row["reduced_formula"]))
            else:
                keys.add("mineral:" + str(row.get("mineral_name", "unknown")))
        return keys

    train_keys = get_group_keys(train_df)
    val_keys = get_group_keys(val_df)
    test_keys = get_group_keys(test_df)
    
    leakage_train_val = train_keys.intersection(val_keys)
    leakage_train_test = train_keys.intersection(test_keys)
    leakage_val_test = val_keys.intersection(test_keys)
    
    has_leakage = False
    if leakage_train_val:
        log.error(f"Leakage between train and val: {len(leakage_train_val)} keys")
        has_leakage = True
    if leakage_train_test:
        log.error(f"Leakage between train and test: {len(leakage_train_test)} keys")
        has_leakage = True
    if leakage_val_test:
        log.error(f"Leakage between val and test: {len(leakage_val_test)} keys")
        has_leakage = True
        
    train_samples = set(train_df["sample_id"])
    val_samples = set(val_df["sample_id"])
    test_samples = set(test_df["sample_id"])
    
    if train_samples.intersection(val_samples) or train_samples.intersection(test_samples) or val_samples.intersection(test_samples):
        log.error("Sample ID leakage detected!")
        has_leakage = True

    if not has_leakage:
        log.info("No leakage detected! Splits are safe.")
        
    # Check distributions
    for col in ["chemical_family", "band_gap_class"]:
        if col in train_df.columns:
            train_dist = train_df[col].value_counts(normalize=True).to_dict()
            val_dist = val_df[col].value_counts(normalize=True).to_dict()
            test_dist = test_df[col].value_counts(normalize=True).to_dict()
            
            all_classes = set(train_dist.keys()) | set(val_dist.keys()) | set(test_dist.keys())
            missing = False
            for c in all_classes:
                if c not in train_dist or c not in val_dist or c not in test_dist:
                    missing = True
                    log.warning(f"Class '{c}' in column '{col}' is missing in one of the splits")
            
            if not missing:
                log.info(f"{col} distribution is present across all splits.")

    return not has_leakage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()
    
    out_dir = Path(args.processed_dir) / "splits"
    
    all_safe = True
    for subset in ["clean_inorganic", "property_matched_inorganic", "xrd_linked_inorganic"]:
        if (out_dir / subset).exists():
            safe = check_leakage(out_dir / subset)
            all_safe = all_safe and safe
    
    if all_safe:
        log.info("All subsets passed leakage checks.")
    else:
        log.error("Leakage checks failed for some subsets.")
        exit(1)

if __name__ == "__main__":
    main()
