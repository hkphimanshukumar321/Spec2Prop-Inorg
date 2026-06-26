import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#!/usr/bin/env python3
"""
Spec2Prop-Inorg: Dataset Integrity Checker
===========================================
Validates the processed dataset subsets with row-explosion detection.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def check_metadata_csv(filepath: str, name: str) -> dict:
    """Validate a CSV metadata file."""
    report = {"name": name, "status": "OK", "issues": [], "warnings": []}

    if not os.path.isfile(filepath):
        report["status"] = "MISSING"
        report["issues"].append(f"File not found: {filepath}")
        return report

    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        report["status"] = "LOAD_ERROR"
        report["issues"].append(f"Failed to load: {e}")
        return report

    report["rows"] = len(df)
    report["columns"] = len(df.columns)
    report["size_mb"] = round(os.path.getsize(filepath) / (1024 * 1024), 2)

    # --- Uniqueness checks ---
    if "sample_id" in df.columns:
        n_unique_sid = df["sample_id"].nunique()
        n_dup_sid = len(df) - n_unique_sid
        report["unique_sample_id"] = n_unique_sid
        report["duplicate_sample_id"] = n_dup_sid
        if n_dup_sid > 0:
            report["warnings"].append(f"CRITICAL: {n_dup_sid} duplicate sample_id values!")

    if "raw_file_path" in df.columns:
        n_unique_rawfp = df["raw_file_path"].nunique()
        n_dup_rawfp = len(df) - n_unique_rawfp
        report["unique_raw_file_path"] = n_unique_rawfp
        report["duplicate_raw_file_path"] = n_dup_rawfp
        if n_dup_rawfp > 0:
            report["warnings"].append(f"WARNING: {n_dup_rawfp} duplicate raw_file_path values")
        if len(df) > 2 * n_unique_rawfp:
            report["warnings"].append("POSSIBLE MANY-TO-MANY MERGE EXPLOSION DETECTED")

    if "rruff_id" in df.columns:
        non_empty = df[df["rruff_id"].notna() & (df["rruff_id"] != "")]
        n_unique_rruff = non_empty["rruff_id"].nunique()
        report["unique_rruff_id"] = n_unique_rruff
        if n_unique_rruff > 0:
            avg_per_rruff = len(non_empty) / n_unique_rruff
            rruff_counts = non_empty["rruff_id"].value_counts()
            report["avg_rows_per_rruff_id"] = round(avg_per_rruff, 2)
            report["max_rows_per_rruff_id"] = int(rruff_counts.max())
            report["actual_duplicate_rruff_ids"] = int((rruff_counts > 1).sum())

    if "mineral_name" in df.columns:
        report["unique_minerals"] = df["mineral_name"].nunique()

    if "chemical_family" in df.columns:
        report["family_distribution"] = df["chemical_family"].value_counts().to_dict()

        unknown_df = df[df["chemical_family"].isin(["Unknown", "Mixed/Other"])]
        if not unknown_df.empty and "mineral_name" in unknown_df.columns:
            report["top_unknown_minerals"] = unknown_df["mineral_name"].value_counts().head(20).to_dict()

    if "formula_parse_success" in df.columns:
        n_success = df["formula_parse_success"].sum()
        report["formula_parse_success_rate"] = f"{100.0 * n_success / len(df):.1f}%"

    if "eligible_for_matbench_matching" in df.columns:
        report["eligible_count"] = int(df["eligible_for_matbench_matching"].sum())

    if "matbench_match_success" in df.columns:
        report["matbench_match_count"] = int(df["matbench_match_success"].sum())
        eligible_count = report.get("eligible_count", len(df))
        if eligible_count > 0:
            report["matbench_match_rate_eligible"] = f"{100.0 * report['matbench_match_count'] / eligible_count:.1f}%"

    if "has_xrd" in df.columns:
        report["samples_with_xrd"] = int(df["has_xrd"].sum())

    # Check flags
    flags = ["has_variable_formula", "has_vacancy_symbol", "has_mixed_occupancy", "has_hydrate", "has_html_tag"]
    report["formula_flags"] = {}
    for f in flags:
        if f in df.columns:
            report["formula_flags"][f] = int(df[f].sum())

    # Sub-family and Organic exclusions
    if "sub_family" in df.columns:
        report["silica_count"] = int((df["sub_family"] == "Silica").sum())
    if "exclude_from_inorganic_main" in df.columns:
        report["organic_excluded"] = int(df["exclude_from_inorganic_main"].sum())

    if "eligible_for_matbench_matching" in df.columns and "formula_parse_success" in df.columns:
        report["excluded_from_prop_prediction"] = int(
            df["formula_parse_success"].sum() - df["eligible_for_matbench_matching"].sum()
        )

    return report


def main():
    parser = argparse.ArgumentParser(description="Spec2Prop-Inorg Dataset Integrity Checker")
    parser.add_argument("--processed-dir", default="data/processed",
                        help="Directory with processed outputs")
    args = parser.parse_args()

    pdir = Path(args.processed_dir)

    print("=" * 80)
    print("Spec2Prop-Inorg Final Dataset Integrity Check")
    print("=" * 80)
    print(f"Processed directory: {pdir.resolve()}\n")

    subsets = [
        ("spec2prop_full_spectra_metadata.csv", "Full Spectra Subset"),
        ("spec2prop_clean_inorganic_metadata.csv", "Clean Inorganic Subset"),
        ("spec2prop_property_matched_inorganic_metadata.csv", "Property Matched Inorganic Subset"),
        ("spec2prop_xrd_linked_inorganic_metadata.csv", "XRD Linked Inorganic Subset"),
    ]

    all_ok = True
    subset_row_counts = {}

    for fname, title in subsets:
        r = check_metadata_csv(str(pdir / fname), title)

        status = r.get("status", "?")
        icon = "[OK]" if status == "OK" else ("[WARN]" if status == "WARNINGS" else "[ERR]")
        print(f"  {icon} {r['name']:40s}  [{status}]")

        if status != "OK":
            for issue in r.get("issues", []):
                print(f"       [!] {issue}")
            print()
            continue

        subset_row_counts[title] = r["rows"]
        print(f"       Rows: {r['rows']:,}  |  Cols: {r.get('columns', '?')}  |  Size: {r.get('size_mb', '?')} MB")

        if "unique_sample_id" in r:
            print(f"       Unique sample_id: {r['unique_sample_id']:,}  |  Duplicate: {r['duplicate_sample_id']:,}")

        if "unique_raw_file_path" in r:
            print(f"       Unique raw_file_path: {r['unique_raw_file_path']:,}  |  Duplicate: {r['duplicate_raw_file_path']:,}")

        if "unique_rruff_id" in r:
            extra = ""
            if "avg_rows_per_rruff_id" in r:
                extra = f"  |  Avg rows/rruff_id: {r['avg_rows_per_rruff_id']}  |  Max: {r['max_rows_per_rruff_id']}"
            print(f"       Unique rruff_id: {r['unique_rruff_id']:,}{extra}")

        if "actual_duplicate_rruff_ids" in r:
            print(f"       Actual duplicate RRUFF IDs (rows > 1): {r['actual_duplicate_rruff_ids']:,}")

        if "unique_minerals" in r:
            print(f"       Unique minerals: {r['unique_minerals']:,}")

        if "formula_parse_success_rate" in r:
            print(f"       Formula parsing success: {r['formula_parse_success_rate']}")

        if "eligible_count" in r:
            print(f"       Matbench Eligible rows: {r['eligible_count']:,}")

        if "matbench_match_count" in r:
            print(f"       Matbench Matches (among eligible): {r['matbench_match_count']:,} ({r.get('matbench_match_rate_eligible', '0%')})")

        if "samples_with_xrd" in r and "XRD" not in title:
            print(f"       Samples with XRD: {r['samples_with_xrd']:,}")

        if r.get("excluded_from_prop_prediction"):
            print(f"       Excluded from prop prediction (formula flags): {r['excluded_from_prop_prediction']:,}")

        if r.get("formula_flags"):
            print(f"\n       Formula Quality Flags:")
            for flag, count in r["formula_flags"].items():
                if count > 0:
                    print(f"         {flag:25s}: {count:,}")

        if r.get("family_distribution") and title == "Full Spectra Subset":
            print(f"\n       Chemical Families (Class Distribution):")
            for fam, cnt in sorted(r["family_distribution"].items(), key=lambda x: -x[1]):
                print(f"         {fam:25s} {cnt:>6,}")

        if r.get("silica_count"):
            print(f"       Silica Sub-Family Count: {r['silica_count']:,}")

        if r.get("organic_excluded"):
            print(f"       Organic/Polymer excluded: {r['organic_excluded']:,}")

        # Leakage-safe splits
        split_dir = pdir / "splits" / title.replace(" Subset", "").replace(" ", "_").lower()
        if split_dir.exists() and (split_dir / "train.csv").exists():
            print(f"       Leakage-safe splits available: Yes")
        else:
            print(f"       Leakage-safe splits available: No")

        # Warnings
        for w in r.get("warnings", []):
            print(f"       *** {w} ***")

        if r.get("top_unknown_minerals") and title == "Full Spectra Subset":
            print(f"\n       Top 20 Unknown/Mixed Minerals:")
            for min_name, cnt in r["top_unknown_minerals"].items():
                print(f"         {min_name:25s} {cnt:>6,}")

        if r.get("top_unmatched_formulas") and title == "Full Spectra Subset":
            print(f"\n       Top 20 Unmatched Clean Formulas:")
            for form, cnt in r["top_unmatched_formulas"].items():
                print(f"         {str(form):25s} {cnt:>6,}")

        print()

    # --- Cross-subset explosion checks ---
    print("-" * 80)
    print("  Cross-Subset Row Explosion Checks:")
    full_rows = subset_row_counts.get("Full Spectra Subset", 0)
    clean_rows = subset_row_counts.get("Clean Inorganic Subset", 0)
    prop_rows = subset_row_counts.get("Property Matched Inorganic Subset", 0)
    xrd_rows = subset_row_counts.get("XRD Linked Inorganic Subset", 0)

    if prop_rows > clean_rows and clean_rows > 0:
        print(f"  *** ROW EXPLOSION: property_matched ({prop_rows:,}) > clean_inorganic ({clean_rows:,})")
        print(f"  *** Do not train model on this dataset until fixed!")
        all_ok = False
    else:
        print(f"  [OK] property_matched ({prop_rows:,}) <= clean_inorganic ({clean_rows:,})")

    if xrd_rows > full_rows and full_rows > 0:
        print(f"  *** ROW EXPLOSION: xrd_linked ({xrd_rows:,}) > full_spectra ({full_rows:,})")
        print(f"  *** Do not train model on this dataset until fixed!")
        all_ok = False
    else:
        print(f"  [OK] xrd_linked ({xrd_rows:,}) <= full_spectra ({full_rows:,})")

    if xrd_rows > clean_rows and clean_rows > 0:
        print(f"  *** ROW EXPLOSION: xrd_linked ({xrd_rows:,}) > clean_inorganic ({clean_rows:,})")
        all_ok = False
    else:
        print(f"  [OK] xrd_linked ({xrd_rows:,}) <= clean_inorganic ({clean_rows:,})")

    print()

    # tmQM Warnings
    for fname in ["tmqm_without_smiles_but_properties.csv", "tmqmg_without_smiles_but_properties.csv"]:
        path = pdir / fname
        if path.is_file():
            df = pd.read_csv(path)
            print(f"  [WARN] {fname} created with {len(df):,} null SMILES rows.")

    print("-" * 80)
    if all_ok:
        print("  Dataset integrity: PASSED. Safe for model training.")
    else:
        print("  Dataset integrity: FAILED. Fix row explosion before training.")
    print("=" * 80)


if __name__ == "__main__":
    main()
