import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Spec2Prop-Inorg: Deployment Model Training
============================================
The main classical-first deployment model.

Pipeline:
  Train split:
    1. fit PCA on X_train_raw
    2. fit PrototypeExtractor on (X_train_raw, y_train)
    3. extract: PCA(32) + domain(163) + prototype(27) = ~222 features
    4. train LightGBM-DART with Optuna (50 trials)
    5. train CatBoost as comparison
    6. (optional) train TabPFN as comparison

  Validation split:
    1. transform using train-fitted PCA
    2. transform using train-fitted PrototypeExtractor
    3. extract domain features (no fitting)
    4. tune hyperparameters / early stopping

  Test split:
    1. transform once
    2. evaluate once with full deployment metrics
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
from sklearn.preprocessing import StandardScaler

from models import Spec2PropDataset
from models.dataset import build_label_encoder, add_model_family_column
from models.domain_features import extract_domain_features
from models.prototype_features import PrototypeExtractor

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False


def get_arrays(dataset, target_col):
    """Extract raw Raman arrays and labels from a dataset."""
    X, y = [], []
    for i in range(len(dataset)):
        b = dataset[i]
        if b[f"mask_{target_col}"].item() > 0:
            X.append(b["raman"].numpy().flatten())
            y.append(b[f"target_{target_col}"].item())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def build_features(X_raw, y, pca, scaler, proto_ext, fit=False):
    """
    Build the full heterogeneous feature vector.

    Parameters
    ----------
    X_raw : (N, 2048) raw spectra
    y : (N,) labels (only needed if fit=True)
    pca : PCA object
    scaler : StandardScaler object
    proto_ext : PrototypeExtractor object
    fit : if True, fit PCA/scaler/proto on this data (training only)

    Returns
    -------
    X_features : (N, F) feature matrix
    """
    # 1. Domain features (no fitting needed, purely spectral)
    print(f"    Extracting domain features for {len(X_raw)} samples...")
    domain = extract_domain_features(X_raw)
    print(f"    Domain features: {domain.shape[1]}-d")

    # 2. PCA features
    if fit:
        pca_feats = pca.fit_transform(X_raw).astype(np.float32)
    else:
        pca_feats = pca.transform(X_raw).astype(np.float32)
    print(f"    PCA features: {pca_feats.shape[1]}-d")

    # 3. Prototype similarity features
    if fit:
        proto_ext.fit(X_raw, y)
    proto_feats = proto_ext.transform(X_raw)
    print(f"    Prototype features: {proto_feats.shape[1]}-d")

    # Stack all features
    X_features = np.hstack([domain, pca_feats, proto_feats])

    # 4. Scale the full feature vector
    if fit:
        X_features = scaler.fit_transform(X_features).astype(np.float32)
    else:
        X_features = scaler.transform(X_features).astype(np.float32)

    print(f"    Total feature dim: {X_features.shape[1]}")
    return X_features


def optuna_lgbm_objective(trial, X_train, y_train, X_val, y_val, num_classes):
    """Optuna objective for LightGBM hyperparameter tuning."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'boosting_type': 'dart',
        'drop_rate': trial.suggest_float('drop_rate', 0.05, 0.3),
        'class_weight': 'balanced',
        'objective': 'multiclass',
        'num_class': num_classes,
        'n_jobs': 2,
        'random_state': 42,
        'verbose': -1,
    }

    clf = lgb.LGBMClassifier(**params)
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    preds = clf.predict(X_val)
    return f1_score(y_val, preds, average="macro", zero_division=0)


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    hybrid_cfg = cfg.get("hybrid", {})
    out_dir = cfg.get("output", {}).get("results_dir", "results/hybrid")
    deploy_dir = os.path.join(out_dir, "deployment")
    os.makedirs(deploy_dir, exist_ok=True)

    target_col = data_cfg["target_col"]

    # Build encoder from official mappings
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "label_mappings.yaml")
    with open(config_path, 'r') as f:
        mappings = yaml.safe_load(f)
    fine_labels = mappings.get("fine_9_labels", [])
    encoder = {v: k for k, v in enumerate(fine_labels)}
    
    meta_df = pd.read_pickle(data_cfg["metadata_pkl"])
    meta_df = add_model_family_column(meta_df)
    num_classes = len(encoder)
    inv_encoder = {v: k for k, v in encoder.items()}
    print(f"Target: {target_col} | Classes: {num_classes}")
    print(f"Class mapping: {encoder}")

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

    # --- Build features with strict train/val/test separation ---
    feat_cfg = hybrid_cfg.get("feature_vector", {})
    pca_dim = feat_cfg.get("pca_dim", 32)
    pca = PCA(n_components=pca_dim, random_state=42)
    scaler = StandardScaler()
    proto_ext = PrototypeExtractor()

    print("\n  Building TRAIN features (fit PCA + scaler + prototypes):")
    X_train = build_features(X_train_raw, y_train, pca, scaler, proto_ext, fit=True)

    print("\n  Building VAL features (transform only):")
    X_val = build_features(X_val_raw, y_val, pca, scaler, proto_ext, fit=False)

    print("\n  Building TEST features (transform only):")
    X_test = build_features(X_test_raw, y_test, pca, scaler, proto_ext, fit=False)

    # Save fitted objects for inference
    joblib.dump(pca, os.path.join(deploy_dir, "pca.joblib"))
    joblib.dump(scaler, os.path.join(deploy_dir, "scaler.joblib"))
    joblib.dump(proto_ext, os.path.join(deploy_dir, "prototype_extractor.joblib"))
    with open(os.path.join(deploy_dir, "encoder.json"), "w") as f:
        json.dump(encoder, f, indent=2)
    print("  Saved: PCA, scaler, prototype extractor, encoder")

    results = []
    all_probas = {}

    # ═══════════════════════════════════════════════════════════════════════
    # LightGBM-DART with Optuna
    # ═══════════════════════════════════════════════════════════════════════
    if HAS_LIGHTGBM:
        lgbm_cfg = hybrid_cfg.get("lightgbm", {})
        n_optuna = lgbm_cfg.get("optuna_trials", 50)

        if n_optuna > 0 and HAS_OPTUNA:
            print(f"\n  Tuning LightGBM-DART with Optuna ({n_optuna} trials)...")
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=42))
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
            print(f"  Best Optuna Val F1: {study.best_value:.4f}")
            print(f"  Best params: {study.best_params}")
        else:
            best_params = {
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.5,
                'boosting_type': 'dart',
                'class_weight': 'balanced',
                'objective': 'multiclass',
                'num_class': num_classes,
                'n_jobs': -1,
                'random_state': 42,
                'verbose': -1,
            }

        # Final training on train, evaluate on test ONCE
        print("\n  Training final LightGBM-DART...")
        t0 = time.time()
        clf_lgbm = lgb.LGBMClassifier(**best_params)
        clf_lgbm.fit(X_train, y_train)
        elapsed = time.time() - t0

        # Test evaluation (single shot)
        proba_test = clf_lgbm.predict_proba(X_test)
        preds_test = proba_test.argmax(axis=1)
        acc = accuracy_score(y_test, preds_test)
        f1 = f1_score(y_test, preds_test, average="macro", zero_division=0)

        print(f"    -> Test Acc: {acc:.4f} | Test F1: {f1:.4f} | Time: {elapsed:.1f}s")
        results.append({
            "Model": "LightGBM-DART (PCA+Domain+Prototype)",
            "Features": X_train.shape[1],
            "Test_Accuracy": round(acc, 4),
            "Test_Macro_F1": round(f1, 4),
            "Train_Time_s": round(elapsed, 1),
        })

        # Save model and probabilities
        joblib.dump(clf_lgbm, os.path.join(deploy_dir, "lgbm_deployment.joblib"))
        np.save(os.path.join(deploy_dir, "lgbm_deployment_test_proba.npy"), proba_test)
        all_probas["LightGBM-DART"] = proba_test

        # Feature importance
        imp = clf_lgbm.feature_importances_
        top_idx = np.argsort(imp)[::-1][:20]
        print("\n  Top-20 Feature Importances:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1:2d}. feature_{idx:3d} = {imp[idx]:.0f}")

    # ═══════════════════════════════════════════════════════════════════════
    # CatBoost
    # ═══════════════════════════════════════════════════════════════════════
    if HAS_CATBOOST:
        print("\n  Training CatBoost...")
        t0 = time.time()
        clf_cat = CatBoostClassifier(
            iterations=1000, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced",
            loss_function="MultiClass",
            random_seed=42, verbose=0,
        )
        clf_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        elapsed = time.time() - t0

        proba_test = clf_cat.predict_proba(X_test)
        preds_test = proba_test.argmax(axis=1)
        acc = accuracy_score(y_test, preds_test)
        f1 = f1_score(y_test, preds_test, average="macro", zero_division=0)

        print(f"    -> Test Acc: {acc:.4f} | Test F1: {f1:.4f} | Time: {elapsed:.1f}s")
        results.append({
            "Model": "CatBoost (PCA+Domain+Prototype)",
            "Features": X_train.shape[1],
            "Test_Accuracy": round(acc, 4),
            "Test_Macro_F1": round(f1, 4),
            "Train_Time_s": round(elapsed, 1),
        })

        np.save(os.path.join(deploy_dir, "catboost_test_proba.npy"), proba_test)
        all_probas["CatBoost"] = proba_test

    # ═══════════════════════════════════════════════════════════════════════
    # TabPFN
    # ═══════════════════════════════════════════════════════════════════════
    if HAS_TABPFN:
        print("\n  Training TabPFN...")
        try:
            t0 = time.time()
            clf_tab = TabPFNClassifier(device="cpu")
            clf_tab.fit(X_train, y_train)
            proba_test = clf_tab.predict_proba(X_test)
            elapsed = time.time() - t0

            preds_test = proba_test.argmax(axis=1)
            acc = accuracy_score(y_test, preds_test)
            f1 = f1_score(y_test, preds_test, average="macro", zero_division=0)

            print(f"    -> Test Acc: {acc:.4f} | Test F1: {f1:.4f} | Time: {elapsed:.1f}s")
            results.append({
                "Model": "TabPFN (PCA+Domain+Prototype)",
                "Features": X_train.shape[1],
                "Test_Accuracy": round(acc, 4),
                "Test_Macro_F1": round(f1, 4),
                "Train_Time_s": round(elapsed, 1),
            })

            np.save(os.path.join(deploy_dir, "tabpfn_test_proba.npy"), proba_test)
            all_probas["TabPFN"] = proba_test
        except Exception as e:
            print(f"    [FAILED] TabPFN error: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # Save test labels and summary
    # ═══════════════════════════════════════════════════════════════════════
    np.save(os.path.join(deploy_dir, "y_test.npy"), y_test)
    np.save(os.path.join(deploy_dir, "y_val.npy"), y_val)

    print(f"\n{'=' * 60}")
    print("  DEPLOYMENT MODEL RESULTS SUMMARY")
    print(f"{'=' * 60}")

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("Test_Macro_F1", ascending=False)
    print(df_res.to_string(index=False))

    out_path = os.path.join(deploy_dir, "deployment_results.csv")
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # Run full deployment evaluation on best model
    # ═══════════════════════════════════════════════════════════════════════
    if all_probas:
        from scripts.evaluate_deployment import full_evaluation
        best_name = df_res.iloc[0]["Model"]
        best_proba_key = best_name.split("(")[0].strip()

        # Find matching proba
        best_proba = None
        for key, proba in all_probas.items():
            if key in best_proba_key:
                best_proba = proba
                break
        if best_proba is None:
            best_proba = list(all_probas.values())[0]
            best_name = list(all_probas.keys())[0]

        eval_results = full_evaluation(y_test, best_proba, inv_encoder, best_name)

        eval_path = os.path.join(deploy_dir, "deployment_full_eval.json")
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2, default=str)
        print(f"\n  Full eval saved to: {eval_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deployment Model")
    parser.add_argument("--config", default="configs/hybrid.yaml")
    args = parser.parse_args()
    main(args.config)
