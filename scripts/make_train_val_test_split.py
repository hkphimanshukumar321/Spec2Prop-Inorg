import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#!/usr/bin/env python3
"""
Spec2Prop-Inorg: Train/Val/Test Splitter
=========================================
Performs a stratified split of the processed dataset based on chemical families.

Usage:
    python scripts/make_train_val_test_split.py --master data/processed/spec2prop_master.csv --out-dir data/processed/splits
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Spec2Prop-Inorg Train/Val/Test Splitter")
    parser.add_argument("--master", default="data/processed/spec2prop_master.csv",
                        help="Path to the master metadata CSV")
    parser.add_argument("--out-dir", default="data/processed/splits",
                        help="Output directory for split CSVs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splitting")
    parser.add_argument("--val-size", type=float, default=0.15,
                        help="Validation set fraction")
    parser.add_argument("--test-size", type=float, default=0.15,
                        help="Test set fraction")
    args = parser.parse_args()

    master_path = Path(args.master)
    out_dir = Path(args.out_dir)

    print("=" * 70)
    print("Spec2Prop-Inorg Data Splitter")
    print("=" * 70)

    if not master_path.is_file():
        print(f"ERROR: Master file not found at {master_path}")
        sys.exit(1)

    print(f"Loading master table: {master_path}")
    df = pd.read_csv(master_path)
    print(f"Loaded {len(df)} rows.")

    # Drop rows without a valid ID or chemical family
    df = df.dropna(subset=["mineral_name", "chemical_family"])
    print(f"Rows after dropping nulls in key columns: {len(df)}")

    if len(df) == 0:
        print("ERROR: No valid rows to split.")
        sys.exit(1)

    # For stratification, if a family has too few samples (e.g. < 5),
    # train_test_split might fail. Let's group rare families into "Other".
    counts = df["chemical_family"].value_counts()
    rare_families = counts[counts < 5].index
    if len(rare_families) > 0:
        print(f"Grouping rare families into 'Other': {list(rare_families)}")
        df.loc[df["chemical_family"].isin(rare_families), "chemical_family"] = "Other"

    out_dir.mkdir(parents=True, exist_ok=True)

    # We want train / val / test. 
    # E.g., 70% train, 15% val, 15% test
    # First split into train and temp (val + test)
    temp_size = args.val_size + args.test_size
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=args.seed,
        stratify=df["chemical_family"]
    )

    # Now split temp into val and test
    test_frac_of_temp = args.test_size / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_frac_of_temp,
        random_state=args.seed,
        stratify=temp_df["chemical_family"]
    )

    print("\nSplit complete:")
    print(f"  Train: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} rows ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")

    # Save splits (just IDs is usually enough, but we'll save full metadata for convenience)
    train_out = out_dir / "train.csv"
    val_out = out_dir / "val.csv"
    test_out = out_dir / "test.csv"

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f"\nSaved splits to {out_dir}")

    # Generate a summary
    summary_out = out_dir / "split_summary.txt"
    with open(summary_out, "w") as f:
        f.write("Spec2Prop-Inorg Data Split Summary\n")
        f.write("==================================\n\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Train samples: {len(train_df)}\n")
        f.write(f"Val samples:   {len(val_df)}\n")
        f.write(f"Test samples:  {len(test_df)}\n\n")
        f.write("Chemical Family Distribution (Train/Val/Test):\n")
        f.write("-" * 50 + "\n")
        
        families = df["chemical_family"].unique()
        for fam in sorted(families):
            tr_cnt = sum(train_df["chemical_family"] == fam)
            vl_cnt = sum(val_df["chemical_family"] == fam)
            ts_cnt = sum(test_df["chemical_family"] == fam)
            tot = tr_cnt + vl_cnt + ts_cnt
            f.write(f"{fam:20s}: {tr_cnt:6d} / {vl_cnt:6d} / {ts_cnt:6d} (Total: {tot})\n")

    print(f"Saved summary to {summary_out}")
    print("=" * 70)

if __name__ == "__main__":
    main()
