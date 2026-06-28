import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Spec2Prop-Inorg: Deployment Evaluation — Comprehensive Metrics
===============================================================
Reports ALL deployment-relevant metrics for a trained classifier:

  1. Top-k accuracy (k=1,2,3,5)
  2. Balanced accuracy
  3. Macro / Weighted F1
  4. Per-class F1 with class names
  5. Confidence calibration (reliability diagram data)
  6. Reject-option accuracy at multiple confidence thresholds
     (coverage, accuracy, macro-F1, abstention rate)

Usage:
  python scripts/evaluate_deployment.py \\
      --proba_npy results/hybrid/lgbm_deployment_test_proba.npy \\
      --y_true_npy results/hybrid/y_test.npy \\
      --encoder_json results/hybrid/chemical_family_model_encoder.json \\
      --output_dir results/hybrid
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report
)


# ─────────────────────────────────────────────────────────────────────────────
# Top-k accuracy
# ─────────────────────────────────────────────────────────────────────────────
def top_k_accuracy(y_true, proba, k):
    """Fraction of samples where the true label is in the top-k predictions."""
    top_k_preds = np.argsort(-proba, axis=1)[:, :k]
    correct = np.any(top_k_preds == y_true[:, None], axis=1)
    return float(correct.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Confidence calibration
# ─────────────────────────────────────────────────────────────────────────────
def calibration_bins(y_true, proba, n_bins=10):
    """
    Compute reliability diagram data.
    Returns a DataFrame with columns: bin_center, accuracy, confidence, count.
    """
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_center": (lo + hi) / 2,
            "accuracy": correct[mask].mean(),
            "confidence": confidences[mask].mean(),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Reject-option evaluation
# ─────────────────────────────────────────────────────────────────────────────
def reject_option_eval(y_true, proba, thresholds=(0.50, 0.60, 0.70, 0.80)):
    """
    Evaluate with abstention: only predict when max confidence > threshold.
    Returns a DataFrame with columns: threshold, coverage, accuracy,
    macro_f1, abstention_rate.
    """
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    n_total = len(y_true)

    rows = []
    for thresh in thresholds:
        mask = confidences >= thresh
        n_accepted = mask.sum()

        if n_accepted == 0:
            rows.append({
                "threshold": thresh,
                "coverage": 0.0,
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "abstention_rate": 1.0,
                "n_accepted": 0,
            })
            continue

        y_accepted = y_true[mask]
        p_accepted = predictions[mask]

        acc = accuracy_score(y_accepted, p_accepted)
        f1 = f1_score(y_accepted, p_accepted, average="macro", zero_division=0)
        coverage = n_accepted / n_total
        abstention = 1.0 - coverage

        rows.append({
            "threshold": thresh,
            "coverage": round(coverage, 4),
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "abstention_rate": round(abstention, 4),
            "n_accepted": int(n_accepted),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation report
# ─────────────────────────────────────────────────────────────────────────────
def full_evaluation(y_true, proba, class_names=None, model_name="Model"):
    """
    Run the complete deployment evaluation suite.
    Returns a dict with all results.
    """
    predictions = proba.argmax(axis=1)
    results = {"model_name": model_name}

    # --- Top-k accuracy ---
    print(f"\n{'=' * 60}")
    print(f"  DEPLOYMENT EVALUATION: {model_name}")
    print(f"{'=' * 60}")

    print("\n  Top-k Accuracy:")
    for k in [1, 2, 3, 5]:
        acc = top_k_accuracy(y_true, proba, k)
        results[f"top_{k}_accuracy"] = round(acc, 4)
        print(f"    Top-{k}: {acc:.4f} ({acc*100:.1f}%)")

    # --- Standard metrics ---
    acc = accuracy_score(y_true, predictions)
    bal_acc = balanced_accuracy_score(y_true, predictions)
    macro_f1 = f1_score(y_true, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, predictions, average="weighted", zero_division=0)

    results["accuracy"] = round(acc, 4)
    results["balanced_accuracy"] = round(bal_acc, 4)
    results["macro_f1"] = round(macro_f1, 4)
    results["weighted_f1"] = round(weighted_f1, 4)

    print(f"\n  Standard Metrics:")
    print(f"    Accuracy:          {acc:.4f}")
    print(f"    Balanced Accuracy: {bal_acc:.4f}")
    print(f"    Macro F1:          {macro_f1:.4f}")
    print(f"    Weighted F1:       {weighted_f1:.4f}")

    # --- Per-class F1 ---
    if class_names is not None:
        target_names = [class_names.get(str(i), f"class_{i}")
                        for i in range(proba.shape[1])]
    else:
        target_names = [f"class_{i}" for i in range(proba.shape[1])]

    print(f"\n  Per-Class Report:")
    report = classification_report(
        y_true, predictions,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    report_str = classification_report(
        y_true, predictions,
        target_names=target_names,
        zero_division=0,
    )
    print(report_str)
    results["per_class_report"] = report

    # --- Confidence calibration ---
    cal_df = calibration_bins(y_true, proba)
    results["calibration"] = cal_df.to_dict(orient="records")
    print("  Confidence Calibration:")
    print(cal_df.to_string(index=False))

    # --- Reject-option ---
    reject_df = reject_option_eval(y_true, proba)
    results["reject_option"] = reject_df.to_dict(orient="records")
    print(f"\n  Reject-Option Evaluation:")
    print(reject_df.to_string(index=False))

    # --- Confidence distribution ---
    confidences = proba.max(axis=1)
    print(f"\n  Confidence Distribution:")
    print(f"    Mean:   {confidences.mean():.4f}")
    print(f"    Median: {np.median(confidences):.4f}")
    print(f"    Std:    {confidences.std():.4f}")
    print(f"    Min:    {confidences.min():.4f}")
    print(f"    Max:    {confidences.max():.4f}")
    results["confidence_stats"] = {
        "mean": round(float(confidences.mean()), 4),
        "median": round(float(np.median(confidences)), 4),
        "std": round(float(confidences.std()), 4),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Deployment Evaluation")
    parser.add_argument("--proba_npy", required=True, help="Path to predicted probabilities .npy")
    parser.add_argument("--y_true_npy", required=True, help="Path to true labels .npy")
    parser.add_argument("--encoder_json", default=None, help="Path to label encoder JSON")
    parser.add_argument("--model_name", default="Deployment Model", help="Name for reporting")
    parser.add_argument("--output_dir", default="results/hybrid", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    proba = np.load(args.proba_npy)
    y_true = np.load(args.y_true_npy)

    # Load class names if available
    class_names = None
    if args.encoder_json and os.path.isfile(args.encoder_json):
        with open(args.encoder_json) as f:
            enc = json.load(f)
        # Invert: {label: idx} -> {idx: label}
        class_names = {str(v): k for k, v in enc.items()}

    results = full_evaluation(y_true, proba, class_names, args.model_name)

    # Save results
    out_path = os.path.join(args.output_dir, f"deployment_eval_{args.model_name.replace(' ', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    main()
