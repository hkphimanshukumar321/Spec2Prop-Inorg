import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Spec2Prop-Inorg: Hard Baseline — Stage 0
=========================================
Establishes the classical ML ceiling with properly tuned XGBoost on:
  1. Raw spectra (no PCA) — tests whether PCA was limiting performance
  2. Domain-engineered features only — tests mineralogist-style features
  3. Raw + domain features — the heterogeneous baseline

This is the reference against which all hybrid models are measured.
"""

import argparse
import json
import os
import time
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.decomposition import PCA

from models import Spec2PropDataset
from models.dataset import build_label_encoder, add_model_family_column
from models.domain_features import extract_domain_features

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def get_arrays(dataset, target_col):
    """Extract raw Raman arrays and labels from a dataset."""
    X, y = [], []
    for i in range(len(dataset)):
        b = dataset[i]
        if b[f"mask_{target_col}"].item() > 0:
            X.append(b["raman"].numpy().flatten())
            y.append(b[f"target_{target_col}"].item())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def compute_domain_features(X_raw):
    """Extract domain-engineered spectral features for all samples."""
    print(f"  Computing domain features for {len(X_raw)} samples...")
    return extract_domain_features(X_raw)

def train_and_evaluate(name, X_train, y_train, X_test, y_test, num_classes, results):
    """Train XGBoost and evaluate."""
    if not HAS_XGBOOST:
        print(f"  [SKIP] {name}: XGBoost not installed")
        return

    print(f"\n  Training: {name} (features: {X_train.shape[1]})")
    t0 = time.time()

    clf = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.3,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    elapsed = time.time() - t0
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print(f"    -> Acc: {acc:.4f} | F1: {f1:.4f} | Time: {elapsed:.1f}s | Trees: {clf.best_iteration}")
    results.append({
        "Model": name,
        "Features": X_train.shape[1],
        "Accuracy": float(acc),
        "Macro_F1": float(f1),
        "Best_Iteration": int(clf.best_iteration) if hasattr(clf, 'best_iteration') else -1,
        "Train_Time_s": round(elapsed, 1),
    })

    return clf


def train_lightgbm(name, X_train, y_train, X_test, y_test, num_classes, results):
    """Train LightGBM DART and evaluate."""
    if not HAS_LIGHTGBM:
        print(f"  [SKIP] {name}: LightGBM not installed")
        return None

    print(f"\n  Training: {name} (features: {X_train.shape[1]})")
    t0 = time.time()

    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        num_class=num_classes,
        objective="multiclass",
        boosting_type="dart",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )

    elapsed = time.time() - t0
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print(f"    -> Acc: {acc:.4f} | F1: {f1:.4f} | Time: {elapsed:.1f}s")
    results.append({
        "Model": name,
        "Features": X_train.shape[1],
        "Accuracy": float(acc),
        "Macro_F1": float(f1),
        "Best_Iteration": int(clf.best_iteration_) if hasattr(clf, 'best_iteration_') else -1,
        "Train_Time_s": round(elapsed, 1),
    })

    return clf


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Use data section from config
    data_cfg = cfg["data"]
    out_dir = cfg.get("output", {}).get("results_dir", "results/hybrid")
    os.makedirs(out_dir, exist_ok=True)

    target_col = data_cfg["target_col"]

    # Build encoder
    meta_df = pd.read_pickle(data_cfg["metadata_pkl"])
    meta_df = add_model_family_column(meta_df)
    encoder = build_label_encoder(meta_df[target_col])
    num_classes = len(encoder)
    print(f"Target: {target_col} | Classes: {num_classes}")

    # Load datasets
    kwargs = dict(
        metadata_pkl=data_cfg["metadata_pkl"],
        raman_parquet=data_cfg["raman_parquet"],
        target_cols=[target_col],
        label_encoders={target_col: encoder},
        cache_dir=data_cfg.get("cache_dir"),
    )

    train_dataset = Spec2PropDataset(
        split_csv=os.path.join(data_cfg["splits_dir"], "train.csv"), **kwargs
    )
    test_dataset = Spec2PropDataset(
        split_csv=os.path.join(data_cfg["splits_dir"], "test.csv"), **kwargs
    )

    print("Extracting arrays...")
    X_train_raw, y_train = get_arrays(train_dataset, target_col)
    X_test_raw, y_test = get_arrays(test_dataset, target_col)
    print(f"  Train: {X_train_raw.shape} | Test: {X_test_raw.shape}")

    # Compute domain features
    X_train_domain = compute_domain_features(X_train_raw)
    X_test_domain = compute_domain_features(X_test_raw)
    print(f"  Domain features: {X_train_domain.shape[1]}-d")

    # PCA features
    pca = PCA(n_components=32, random_state=42)
    X_train_pca = pca.fit_transform(X_train_raw).astype(np.float32)
    X_test_pca = pca.transform(X_test_raw).astype(np.float32)

    # Heterogeneous: raw + domain + PCA
    X_train_hetero = np.hstack([X_train_raw, X_train_domain, X_train_pca])
    X_test_hetero = np.hstack([X_test_raw, X_test_domain, X_test_pca])

    results = []

    print("\n" + "=" * 60)
    print("  STAGE 0: HARD BASELINE")
    print("=" * 60)

    # --- XGBoost variants ---
    train_and_evaluate(
        "XGBoost-Raw (no PCA)",
        X_train_raw, y_train, X_test_raw, y_test,
        num_classes, results
    )

    train_and_evaluate(
        "XGBoost-Domain (peaks/FWHM/ratios)",
        X_train_domain, y_train, X_test_domain, y_test,
        num_classes, results
    )

    train_and_evaluate(
        "XGBoost-Raw+Domain",
        np.hstack([X_train_raw, X_train_domain]),
        y_train,
        np.hstack([X_test_raw, X_test_domain]),
        y_test,
        num_classes, results
    )

    train_and_evaluate(
        "XGBoost-Heterogeneous (Raw+Domain+PCA)",
        X_train_hetero, y_train, X_test_hetero, y_test,
        num_classes, results
    )

    # --- LightGBM DART variants ---
    train_lightgbm(
        "LightGBM-DART-Raw",
        X_train_raw, y_train, X_test_raw, y_test,
        num_classes, results
    )

    train_lightgbm(
        "LightGBM-DART-Domain",
        X_train_domain, y_train, X_test_domain, y_test,
        num_classes, results
    )

    train_lightgbm(
        "LightGBM-DART-Heterogeneous",
        X_train_hetero, y_train, X_test_hetero, y_test,
        num_classes, results
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  STAGE 0 RESULTS SUMMARY")
    print("=" * 60)

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("Macro_F1", ascending=False)
    print(df_res.to_string(index=False))

    out_path = os.path.join(out_dir, "stage0_hard_baseline.csv")
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")

    # Save PCA for later use
    import joblib
    pca_path = os.path.join(out_dir, "pca_32.joblib")
    joblib.dump(pca, pca_path)
    print(f"PCA saved to: {pca_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 0: Hard Baseline")
    parser.add_argument("--config", default="configs/hybrid.yaml")
    args = parser.parse_args()
    main(args.config)
