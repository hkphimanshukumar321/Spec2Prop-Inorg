import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Spec2Prop-Inorg: Hybrid GBDT Training — Stage 2
=================================================
Loads SupCon-trained CNN checkpoints, extracts heterogeneous feature
vectors, and trains LightGBM DART + XGBoost classifiers.

Optionally tunes hyperparameters with Optuna.
"""

import argparse
import json
import os
import time
import yaml
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.decomposition import PCA

from models import Spec2PropDataset, SimpleCNN1D, MultiScaleSpecNet, RamanFormer1D, LiteSpecNet, ResidualCNN1D
from models.dataset import build_label_encoder, add_model_family_column
from models.hybrid_models import HeterogeneousFeatureExtractor

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def get_model_class(name):
    """Return the model class by name."""
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
    ModelClass = get_model_class(model_name)
    model = ModelClass(in_channels=1, num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"  Loaded {model_name} from {checkpoint_path}")
    return model


def get_arrays(dataset, target_col):
    """Extract raw Raman arrays and labels from a dataset."""
    X, y = [], []
    for i in range(len(dataset)):
        b = dataset[i]
        if b[f"mask_{target_col}"].item() > 0:
            X.append(b["raman"].numpy().flatten())
            y.append(b[f"target_{target_col}"].item())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def optuna_lgbm_objective(trial, X_train, y_train, X_val, y_val, num_classes):
    """Optuna objective for LightGBM hyperparameter tuning."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'boosting_type': 'dart',
        'class_weight': 'balanced',
        'objective': 'multiclass',
        'num_class': num_classes,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1,
    }

    clf = lgb.LGBMClassifier(**params)
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )
    preds = clf.predict(X_val)
    return f1_score(y_val, preds, average="macro")


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    hybrid_cfg = cfg.get("hybrid", {})
    supcon_cfg = cfg.get("supcon", {})
    out_dir = cfg.get("output", {}).get("results_dir", "results/hybrid")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    val_dataset = Spec2PropDataset(
        split_csv=os.path.join(data_cfg["splits_dir"], "val.csv"), **kwargs
    )
    test_dataset = Spec2PropDataset(
        split_csv=os.path.join(data_cfg["splits_dir"], "test.csv"), **kwargs
    )

    print("Extracting raw arrays...")
    X_train_raw, y_train = get_arrays(train_dataset, target_col)
    X_val_raw, y_val = get_arrays(val_dataset, target_col)
    X_test_raw, y_test = get_arrays(test_dataset, target_col)
    print(f"  Train: {X_train_raw.shape} | Val: {X_val_raw.shape} | Test: {X_test_raw.shape}")

    # Fit PCA on training data
    feat_cfg = hybrid_cfg.get("feature_vector", {})
    pca_dim = feat_cfg.get("pca_dim", 32)
    pca = PCA(n_components=pca_dim, random_state=42)
    pca.fit(X_train_raw)
    print(f"  PCA fitted: {pca_dim} components, {pca.explained_variance_ratio_.sum():.2%} variance explained")

    results = []

    # --- Try each SupCon backbone ---
    backbones = supcon_cfg.get("backbones", ["SimpleCNN1D", "MultiScaleSpecNet"])

    for model_name in backbones:
        checkpoint_path = os.path.join(out_dir, f"{model_name}_supcon_best.pt")

        if not os.path.isfile(checkpoint_path):
            print(f"\n  [SKIP] {model_name}: No SupCon checkpoint found at {checkpoint_path}")
            print(f"         Run train_supcon.py first.")
            # Fall back to no CNN embeddings
            model = None
        else:
            model = load_supcon_model(model_name, checkpoint_path, num_classes, device)

        extractor = HeterogeneousFeatureExtractor(
            model=model,
            pca=pca,
            device=str(device),
            n_peaks=feat_cfg.get("n_domain_peaks", 10)
        )

        print(f"\n  Extracting heterogeneous features ({model_name})...")
        X_train = extractor.extract(X_train_raw)
        X_val = extractor.extract(X_val_raw)
        X_test = extractor.extract(X_test_raw)
        print(f"  Feature dim: {X_train.shape[1]}")

        # --- LightGBM DART ---
        if HAS_LIGHTGBM:
            lgbm_cfg = hybrid_cfg.get("lightgbm", {})
            n_optuna = lgbm_cfg.get("optuna_trials", 0)

            if n_optuna > 0 and HAS_OPTUNA:
                print(f"\n  Tuning LightGBM with Optuna ({n_optuna} trials)...")
                study = optuna.create_study(direction="maximize")
                study.optimize(
                    lambda trial: optuna_lgbm_objective(
                        trial, X_train, y_train, X_val, y_val, num_classes
                    ),
                    n_trials=n_optuna,
                    show_progress_bar=True,
                )
                best_params = study.best_params
                best_params.update({
                    'boosting_type': 'dart',
                    'class_weight': 'balanced',
                    'objective': 'multiclass',
                    'num_class': num_classes,
                    'n_jobs': -1,
                    'random_state': 42,
                    'verbose': -1,
                })
                print(f"  Best Optuna F1: {study.best_value:.4f}")
            else:
                best_params = {
                    'n_estimators': lgbm_cfg.get("n_estimators", 1000),
                    'max_depth': lgbm_cfg.get("max_depth", 6),
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.5,
                    'boosting_type': lgbm_cfg.get("boosting_type", "dart"),
                    'class_weight': lgbm_cfg.get("class_weight", "balanced"),
                    'objective': 'multiclass',
                    'num_class': num_classes,
                    'n_jobs': -1,
                    'random_state': 42,
                    'verbose': -1,
                }

            print(f"\n  Training LightGBM DART ({model_name} embeddings)...")
            t0 = time.time()
            clf_lgbm = lgb.LGBMClassifier(**best_params)
            clf_lgbm.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
            )
            elapsed = time.time() - t0

            preds_test = clf_lgbm.predict(X_test)
            acc = accuracy_score(y_test, preds_test)
            f1 = f1_score(y_test, preds_test, average="macro")

            label = f"LightGBM-DART + {model_name}"
            print(f"    -> Test Acc: {acc:.4f} | Test F1: {f1:.4f} | Time: {elapsed:.1f}s")
            results.append({
                "Model": label,
                "Backbone": model_name,
                "Classifier": "LightGBM-DART",
                "Features": X_train.shape[1],
                "Test_Accuracy": float(acc),
                "Test_Macro_F1": float(f1),
                "Train_Time_s": round(elapsed, 1),
            })

            # Save model
            lgbm_path = os.path.join(out_dir, f"lgbm_{model_name}.joblib")
            joblib.dump(clf_lgbm, lgbm_path)
            print(f"    Saved: {lgbm_path}")

            # Save probabilities for ensemble
            proba_test = clf_lgbm.predict_proba(X_test)
            np.save(os.path.join(out_dir, f"lgbm_{model_name}_test_proba.npy"), proba_test)

        # --- XGBoost ---
        if HAS_XGBOOST:
            print(f"\n  Training XGBoost ({model_name} embeddings)...")
            t0 = time.time()
            clf_xgb = XGBClassifier(
                n_estimators=1000, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.3,
                objective="multi:softprob", num_class=num_classes,
                eval_metric="mlogloss", early_stopping_rounds=50,
                n_jobs=-1, random_state=42, use_label_encoder=False, verbosity=0,
            )
            clf_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            elapsed = time.time() - t0

            preds_test = clf_xgb.predict(X_test)
            acc = accuracy_score(y_test, preds_test)
            f1 = f1_score(y_test, preds_test, average="macro")

            label = f"XGBoost + {model_name}"
            print(f"    -> Test Acc: {acc:.4f} | Test F1: {f1:.4f} | Time: {elapsed:.1f}s")
            results.append({
                "Model": label,
                "Backbone": model_name,
                "Classifier": "XGBoost",
                "Features": X_train.shape[1],
                "Test_Accuracy": float(acc),
                "Test_Macro_F1": float(f1),
                "Train_Time_s": round(elapsed, 1),
            })

            xgb_path = os.path.join(out_dir, f"xgb_{model_name}.joblib")
            joblib.dump(clf_xgb, xgb_path)

            proba_test = clf_xgb.predict_proba(X_test)
            np.save(os.path.join(out_dir, f"xgb_{model_name}_test_proba.npy"), proba_test)

    # --- Also train on domain-only features (no CNN) for comparison ---
    extractor_no_cnn = HeterogeneousFeatureExtractor(
        model=None, pca=pca, device=str(device),
    )
    X_train_nocnn = extractor_no_cnn.extract(X_train_raw)
    X_val_nocnn = extractor_no_cnn.extract(X_val_raw)
    X_test_nocnn = extractor_no_cnn.extract(X_test_raw)

    if HAS_LIGHTGBM:
        print(f"\n  Training LightGBM DART (PCA + Domain only, no CNN)...")
        clf_nocnn = lgb.LGBMClassifier(
            n_estimators=1000, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5,
            boosting_type="dart", class_weight="balanced",
            objective="multiclass", num_class=num_classes,
            n_jobs=-1, random_state=42, verbose=-1,
        )
        clf_nocnn.fit(
            X_train_nocnn, y_train,
            eval_set=[(X_val_nocnn, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
        )
        preds = clf_nocnn.predict(X_test_nocnn)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        print(f"    -> Test Acc: {acc:.4f} | Test F1: {f1:.4f}")
        results.append({
            "Model": "LightGBM-DART (PCA+Domain only)",
            "Backbone": "None",
            "Classifier": "LightGBM-DART",
            "Features": X_train_nocnn.shape[1],
            "Test_Accuracy": float(acc),
            "Test_Macro_F1": float(f1),
            "Train_Time_s": 0,
        })

    # Save PCA + extractor config for downstream use
    joblib.dump(pca, os.path.join(out_dir, "pca_32.joblib"))

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("  STAGE 2 RESULTS SUMMARY")
    print(f"{'=' * 60}")

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("Test_Macro_F1", ascending=False)
    print(df_res.to_string(index=False))

    out_path = os.path.join(out_dir, "stage2_hybrid_results.csv")
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")

    # Save test labels for ensemble
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Hybrid GBDT Training")
    parser.add_argument("--config", default="configs/hybrid.yaml")
    args = parser.parse_args()
    main(args.config)
