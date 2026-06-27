import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Spec2Prop-Inorg: Evaluate Hybrid Pipeline — Stage 3
=====================================================
Loads saved probability outputs from all stages and computes:
  1. Per-model accuracy and macro F1
  2. Probability-averaged ensemble
  3. Comparison table: Pure NN vs Pure ML vs Hybrid vs Ensemble
"""

import argparse
import glob
import json
import os
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from models.stacking_ensemble import ProbabilityEnsemble, evaluate_ensemble


def load_proba_files(results_dir):
    """Find and load all *_test_proba.npy files."""
    pattern = os.path.join(results_dir, "*_test_proba.npy")
    files = sorted(glob.glob(pattern))
    probas = {}
    for f in files:
        name = os.path.basename(f).replace("_test_proba.npy", "")
        probas[name] = np.load(f)
    return probas


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg.get("output", {}).get("results_dir", "results/hybrid")

    # Load test labels
    y_test_path = os.path.join(out_dir, "y_test.npy")
    if not os.path.isfile(y_test_path):
        print(f"ERROR: {y_test_path} not found. Run train_hybrid.py first.")
        return

    y_test = np.load(y_test_path)

    # Load label encoder for class names
    target_col = cfg["data"]["target_col"]
    encoder_path = os.path.join(out_dir, f"{target_col}_encoder.json")
    if os.path.isfile(encoder_path):
        with open(encoder_path) as f:
            encoder = json.load(f)
        class_names = [k for k, v in sorted(encoder.items(), key=lambda x: x[1])]
    else:
        class_names = None

    # Load all probability files
    probas = load_proba_files(out_dir)

    if not probas:
        print("No probability files found. Run train_hybrid.py and/or train_tabpfn.py first.")
        return

    print(f"\n{'=' * 70}")
    print("  HYBRID PIPELINE — FULL EVALUATION")
    print(f"{'=' * 70}")
    print(f"\n  Found {len(probas)} model probability outputs:")
    for name in probas:
        print(f"    - {name}")

    # --- Per-model results ---
    print(f"\n{'=' * 70}")
    print("  PER-MODEL RESULTS")
    print(f"{'=' * 70}")

    results = []
    for name, proba in probas.items():
        preds = np.argmax(proba, axis=1)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Macro_F1": f1,
        })
        print(f"  {name:40s} | Acc: {acc:.4f} | F1: {f1:.4f}")

    # --- Ensemble ---
    print(f"\n{'=' * 70}")
    print("  ENSEMBLE RESULTS")
    print(f"{'=' * 70}")

    proba_list = list(probas.values())
    model_names = list(probas.keys())

    # Simple average
    ensemble = ProbabilityEnsemble()
    ensemble_preds = ensemble.predict(proba_list)
    ensemble_proba = ensemble.predict_proba(proba_list)
    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    ensemble_f1 = f1_score(y_test, ensemble_preds, average="macro")

    print(f"  {'Ensemble (avg all)':40s} | Acc: {ensemble_acc:.4f} | F1: {ensemble_f1:.4f}")
    results.append({
        "Model": "Ensemble (avg all)",
        "Accuracy": ensemble_acc,
        "Macro_F1": ensemble_f1,
    })

    # Try pairwise ensembles
    if len(probas) >= 2:
        keys = list(probas.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                pair_proba = [probas[keys[i]], probas[keys[j]]]
                pair_ens = ProbabilityEnsemble()
                pair_preds = pair_ens.predict(pair_proba)
                pair_acc = accuracy_score(y_test, pair_preds)
                pair_f1 = f1_score(y_test, pair_preds, average="macro")
                pair_name = f"Ensemble ({keys[i]} + {keys[j]})"
                print(f"  {pair_name:40s} | Acc: {pair_acc:.4f} | F1: {pair_f1:.4f}")
                results.append({
                    "Model": pair_name,
                    "Accuracy": pair_acc,
                    "Macro_F1": pair_f1,
                })

    # --- Comparison with baselines ---
    print(f"\n{'=' * 70}")
    print("  COMPARISON WITH BASELINES")
    print(f"{'=' * 70}")

    # Load baseline results if available
    baseline_path = os.path.join("results", "baselines", "baseline_comparison.csv")
    if os.path.isfile(baseline_path):
        baselines = pd.read_csv(baseline_path)
        best_baseline = baselines.loc[baselines["Macro_F1"].idxmax()]
        print(f"  Best classical baseline: {best_baseline['Model']} (PCA:{best_baseline['PCA_Dim']})")
        print(f"    Acc: {best_baseline['Accuracy']:.4f} | F1: {best_baseline['Macro_F1']:.4f}")

    # Sort and display final ranking
    df_results = pd.DataFrame(results).sort_values("Macro_F1", ascending=False)
    print(f"\n{'=' * 70}")
    print("  FINAL RANKING")
    print(f"{'=' * 70}")
    print(df_results.to_string(index=False))

    # Save
    out_path = os.path.join(out_dir, "stage3_final_comparison.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")

    # --- Best model confusion matrix ---
    best_model = df_results.iloc[0]["Model"]
    if "Ensemble" in best_model:
        best_preds = ensemble_preds
    else:
        best_preds = np.argmax(probas.get(best_model, ensemble_proba), axis=1)

    print(f"\n{'=' * 70}")
    print(f"  CLASSIFICATION REPORT — {best_model}")
    print(f"{'=' * 70}")
    if class_names:
        # Filter to classes that appear in test set
        labels = sorted(set(y_test))
        target_names = [class_names[i] for i in labels if i < len(class_names)]
        print(classification_report(y_test, best_preds, labels=labels, target_names=target_names))
    else:
        print(classification_report(y_test, best_preds))

    # Save ensemble probabilities
    np.save(os.path.join(out_dir, "ensemble_test_proba.npy"), ensemble_proba)
    print(f"\nEnsemble probabilities saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Evaluate Hybrid Pipeline")
    parser.add_argument("--config", default="configs/hybrid.yaml")
    args = parser.parse_args()
    main(args.config)
