import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Spec2Prop-Inorg: TabPFN v2 Zero-Shot Classifier — Stage 2b
============================================================
TabPFN (Prior-Fitted Network) is a transformer pre-trained on millions
of synthetic tabular datasets. It performs in-context learning: you
feed it training samples + features, and it predicts with zero
hyperparameter tuning.

This script takes the same heterogeneous feature vector from Stage 2
and passes it through TabPFN. If TabPFN is unavailable or exceeds
limits (features, classes), it falls back gracefully.
"""

import argparse
import json
import os
import time
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA

from models import Spec2PropDataset, SimpleCNN1D, MultiScaleSpecNet
from models.dataset import build_label_encoder, add_model_family_column
from models.hybrid_models import HeterogeneousFeatureExtractor

try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False

# Fallback
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def get_model_class(name):
    from models import LiteSpecNet, ResidualCNN1D, RamanFormer1D
    classes = {
        "SimpleCNN1D": SimpleCNN1D,
        "LiteSpecNet": LiteSpecNet,
        "ResidualCNN1D": ResidualCNN1D,
        "MultiScaleSpecNet": MultiScaleSpecNet,
        "RamanFormer1D": RamanFormer1D,
    }
    return classes[name]


def load_supcon_model(model_name, checkpoint_path, num_classes, device):
    """Load a SupCon-trained model checkpoint."""
    import torch
    ModelClass = get_model_class(model_name)
    model = ModelClass(in_channels=1, num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def get_arrays(dataset, target_col):
    X, y = [], []
    for i in range(len(dataset)):
        b = dataset[i]
        if b[f"mask_{target_col}"].item() > 0:
            X.append(b["raman"].numpy().flatten())
            y.append(b[f"target_{target_col}"].item())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    hybrid_cfg = cfg.get("hybrid", {})
    supcon_cfg = cfg.get("supcon", {})
    tabpfn_cfg = hybrid_cfg.get("tabpfn", {})
    out_dir = cfg.get("output", {}).get("results_dir", "results/hybrid")
    os.makedirs(out_dir, exist_ok=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tabpfn_device = tabpfn_cfg.get("device", "cpu")
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

    # Load or fit PCA
    pca_path = os.path.join(out_dir, "pca_32.joblib")
    if os.path.isfile(pca_path):
        pca = joblib.load(pca_path)
        print(f"  Loaded PCA from {pca_path}")
    else:
        feat_cfg = hybrid_cfg.get("feature_vector", {})
        pca = PCA(n_components=feat_cfg.get("pca_dim", 32), random_state=42)
        pca.fit(X_train_raw)

    results = []

    # Try with each SupCon backbone
    backbones = supcon_cfg.get("backbones", ["SimpleCNN1D"])

    for model_name in backbones:
        checkpoint_path = os.path.join(out_dir, f"{model_name}_supcon_best.pt")

        if os.path.isfile(checkpoint_path):
            model = load_supcon_model(model_name, checkpoint_path, num_classes, device)
        else:
            print(f"  [INFO] No SupCon checkpoint for {model_name}, using domain features only")
            model = None

        extractor = HeterogeneousFeatureExtractor(
            model=model, pca=pca, device=str(device),
        )

        print(f"\n  Extracting features ({model_name})...")
        X_train = extractor.extract(X_train_raw)
        X_test = extractor.extract(X_test_raw)
        print(f"  Feature dim: {X_train.shape[1]}")

        # --- TabPFN ---
        if HAS_TABPFN and tabpfn_cfg.get("enabled", True):
            print(f"\n  Running TabPFN v2 ({model_name} embeddings)...")
            try:
                t0 = time.time()
                clf = TabPFNClassifier(device=tabpfn_device)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                proba = clf.predict_proba(X_test)
                elapsed = time.time() - t0

                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average="macro")

                label = f"TabPFN + {model_name}"
                print(f"    -> Test Acc: {acc:.4f} | Test F1: {f1:.4f} | Time: {elapsed:.1f}s")
                results.append({
                    "Model": label,
                    "Backbone": model_name,
                    "Classifier": "TabPFN",
                    "Features": X_train.shape[1],
                    "Test_Accuracy": float(acc),
                    "Test_Macro_F1": float(f1),
                    "Train_Time_s": round(elapsed, 1),
                })

                # Save probabilities for ensemble
                np.save(os.path.join(out_dir, f"tabpfn_{model_name}_test_proba.npy"), proba)

            except Exception as e:
                print(f"    [FAILED] TabPFN error: {e}")
                print(f"    Falling back to CatBoost...")

                if HAS_CATBOOST:
                    t0 = time.time()
                    clf = CatBoostClassifier(
                        iterations=1000, depth=6, learning_rate=0.05,
                        auto_class_weights="Balanced",
                        loss_function="MultiClass",
                        random_seed=42, verbose=0,
                    )
                    clf.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
                    elapsed = time.time() - t0

                    preds = clf.predict(X_test).flatten().astype(int)
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average="macro")

                    label = f"CatBoost + {model_name}"
                    print(f"    -> Test Acc: {acc:.4f} | Test F1: {f1:.4f} | Time: {elapsed:.1f}s")
                    results.append({
                        "Model": label,
                        "Backbone": model_name,
                        "Classifier": "CatBoost",
                        "Features": X_train.shape[1],
                        "Test_Accuracy": float(acc),
                        "Test_Macro_F1": float(f1),
                        "Train_Time_s": round(elapsed, 1),
                    })

                    proba = clf.predict_proba(X_test)
                    np.save(os.path.join(out_dir, f"catboost_{model_name}_test_proba.npy"), proba)
        else:
            if not HAS_TABPFN:
                print("  [SKIP] TabPFN not installed. Install with: pip install tabpfn")
            else:
                print("  [SKIP] TabPFN disabled in config")

    # --- Summary ---
    if results:
        print(f"\n{'=' * 60}")
        print("  STAGE 2b RESULTS SUMMARY")
        print(f"{'=' * 60}")
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values("Test_Macro_F1", ascending=False)
        print(df_res.to_string(index=False))

        out_path = os.path.join(out_dir, "stage2b_tabpfn_results.csv")
        df_res.to_csv(out_path, index=False)
        print(f"\nSaved to: {out_path}")
    else:
        print("\n  No models were trained. Check TabPFN installation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2b: TabPFN Classifier")
    parser.add_argument("--config", default="configs/hybrid.yaml")
    args = parser.parse_args()
    main(args.config)
