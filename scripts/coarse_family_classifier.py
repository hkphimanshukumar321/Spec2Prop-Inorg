import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Spec2Prop-Inorg: Coarse-Family Deployment Classifier
======================================================
Creates a coarse 5-class mapping and optionally a hierarchical two-stage
classifier. Target: >80% accuracy for practical deployment.

Coarse mapping:
  Silicate                         -> Silicate
  Oxide                            -> Oxide
  Sulfate, Phosphate, Carbonate    -> Salt-like
  Sulfide                          -> Sulfide
  Halide, Borate, Other/Rare       -> Other

Also runs a hierarchical two-stage evaluation:
  Stage 1: Predict coarse class
  Stage 2: Within predicted coarse group, predict fine class
"""

import argparse
import json
import os
import time
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report
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


# ─────────────────────────────────────────────────────────────────────────────
# Coarse family mapping
# ─────────────────────────────────────────────────────────────────────────────
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "label_mappings.yaml")
with open(config_path, 'r') as f:
    mappings = yaml.safe_load(f)

COARSE_MAPPING = mappings.get("fine_9_to_coarse_5", {})
COARSE_CLASSES = mappings.get("coarse_5_labels", [])
COARSE_ENCODER = {c: i for i, c in enumerate(COARSE_CLASSES)}

# Which fine classes belong to each coarse class
COARSE_TO_FINE = {}
for fine, coarse in COARSE_MAPPING.items():
    COARSE_TO_FINE.setdefault(coarse, []).append(fine)


def fine_to_coarse_labels(y_fine, fine_encoder):
    """Convert fine-grained labels to coarse labels."""
    inv_enc = {v: k for k, v in fine_encoder.items()}
    y_coarse = np.zeros_like(y_fine)
    for i, label_idx in enumerate(y_fine):
        fine_name = inv_enc.get(int(label_idx), "Other/Rare")
        coarse_name = COARSE_MAPPING.get(fine_name, "Other")
        y_coarse[i] = COARSE_ENCODER[coarse_name]
    return y_coarse


def get_arrays(dataset, target_col):
    X, y = [], []
    for i in range(len(dataset)):
        b = dataset[i]
        if b[f"mask_{target_col}"].item() > 0:
            X.append(b["raman"].numpy().flatten())
            y.append(b[f"target_{target_col}"].item())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def build_features(X_raw, y, pca, scaler, proto_ext, fit=False):
    """Build full feature vector with strict separation."""
    domain = extract_domain_features(X_raw)

    if fit:
        pca_feats = pca.fit_transform(X_raw).astype(np.float32)
    else:
        pca_feats = pca.transform(X_raw).astype(np.float32)

    if fit:
        proto_ext.fit(X_raw, y)
    proto_feats = proto_ext.transform(X_raw)

    X_features = np.hstack([domain, pca_feats, proto_feats])

    if fit:
        X_features = scaler.fit_transform(X_features).astype(np.float32)
    else:
        X_features = scaler.transform(X_features).astype(np.float32)

    return X_features


def optuna_lgbm(X_train, y_train, X_val, y_val, num_classes, n_trials=30):
    """Quick Optuna search for LightGBM."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'num_leaves': trial.suggest_int('num_leaves', 15, 100),
            'boosting_type': 'dart',
            'class_weight': 'balanced',
            'objective': 'multiclass',
            'num_class': num_classes,
            'n_jobs': 2,
            'random_state': 42,
            'verbose': -1,
        }
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                callbacks=[lgb.log_evaluation(period=0)])
        preds = clf.predict(X_val)
        return f1_score(y_val, preds, average="macro", zero_division=0)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({
        'boosting_type': 'dart', 'class_weight': 'balanced',
        'objective': 'multiclass', 'num_class': num_classes,
        'n_jobs': 2, 'random_state': 42, 'verbose': -1,
    })
    return best_params, study.best_value


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    hybrid_cfg = cfg.get("hybrid", {})
    out_dir = cfg.get("output", {}).get("results_dir", "results/hybrid")
    coarse_dir = os.path.join(out_dir, "coarse")
    os.makedirs(coarse_dir, exist_ok=True)

    target_col = data_cfg["target_col"]

    # Build fine-grained encoder
    meta_df = pd.read_pickle(data_cfg["metadata_pkl"])
    meta_df = add_model_family_column(meta_df)
    fine_encoder = build_label_encoder(meta_df[target_col])
    num_fine = len(fine_encoder)

    kwargs = dict(
        metadata_pkl=data_cfg["metadata_pkl"],
        raman_parquet=data_cfg["raman_parquet"],
        target_cols=[target_col],
        label_encoders={target_col: fine_encoder},
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

    X_train_raw, y_train_fine = get_arrays(train_dataset, target_col)
    X_val_raw, y_val_fine = get_arrays(val_dataset, target_col)
    X_test_raw, y_test_fine = get_arrays(test_dataset, target_col)

    # Convert to coarse labels
    y_train_coarse = fine_to_coarse_labels(y_train_fine, fine_encoder)
    y_val_coarse = fine_to_coarse_labels(y_val_fine, fine_encoder)
    y_test_coarse = fine_to_coarse_labels(y_test_fine, fine_encoder)

    num_coarse = len(COARSE_CLASSES)
    print(f"Fine classes: {num_fine} | Coarse classes: {num_coarse}")
    print(f"Coarse mapping: {COARSE_ENCODER}")
    print(f"Train: {len(y_train_coarse)} | Val: {len(y_val_coarse)} | Test: {len(y_test_coarse)}")

    # Coarse class distribution
    for c_name, c_idx in COARSE_ENCODER.items():
        n = (y_train_coarse == c_idx).sum()
        print(f"  {c_name}: {n} train samples")

    # ═══════════════════════════════════════════════════════════════════════
    # Build features with COARSE labels for prototype extraction
    # ═══════════════════════════════════════════════════════════════════════
    pca = PCA(n_components=32, random_state=42)
    scaler = StandardScaler()
    proto_ext = PrototypeExtractor()

    print("\n  Building COARSE features...")
    X_train = build_features(X_train_raw, y_train_coarse, pca, scaler, proto_ext, fit=True)
    X_val = build_features(X_val_raw, y_val_coarse, pca, scaler, proto_ext, fit=False)
    X_test = build_features(X_test_raw, y_test_coarse, pca, scaler, proto_ext, fit=False)
    print(f"  Feature dim: {X_train.shape[1]}")

    results = []

    # ═══════════════════════════════════════════════════════════════════════
    # Train COARSE classifier
    # ═══════════════════════════════════════════════════════════════════════
    if HAS_LIGHTGBM and HAS_OPTUNA:
        print(f"\n  Tuning coarse LightGBM-DART (30 Optuna trials)...")
        best_params, best_val_f1 = optuna_lgbm(
            X_train, y_train_coarse, X_val, y_val_coarse, num_coarse, n_trials=30
        )
        print(f"  Best Val F1: {best_val_f1:.4f}")

        # Train final model
        clf_coarse = lgb.LGBMClassifier(**best_params)
        clf_coarse.fit(X_train, y_train_coarse)

        proba_coarse = clf_coarse.predict_proba(X_test)
        preds_coarse = proba_coarse.argmax(axis=1)
        acc = accuracy_score(y_test_coarse, preds_coarse)
        f1 = f1_score(y_test_coarse, preds_coarse, average="macro", zero_division=0)

        print(f"\n  COARSE Test Acc: {acc:.4f} | Test F1: {f1:.4f}")
        results.append({
            "Task": "Coarse 5-class",
            "Model": "LightGBM-DART",
            "Test_Accuracy": round(acc, 4),
            "Test_Macro_F1": round(f1, 4),
        })

        # Per-class report
        print("\n  Coarse Per-Class Report:")
        print(classification_report(
            y_test_coarse, preds_coarse,
            target_names=COARSE_CLASSES, zero_division=0
        ))

        # Top-k accuracy for coarse
        for k in [1, 2, 3]:
            top_k = np.argsort(-proba_coarse, axis=1)[:, :k]
            top_k_acc = np.any(top_k == y_test_coarse[:, None], axis=1).mean()
            print(f"  Coarse Top-{k}: {top_k_acc:.4f}")

        # Save
        joblib.dump(clf_coarse, os.path.join(coarse_dir, "lgbm_coarse.joblib"))
        np.save(os.path.join(coarse_dir, "coarse_test_proba.npy"), proba_coarse)
        np.save(os.path.join(coarse_dir, "y_test_coarse.npy"), y_test_coarse)

        # Confidence threshold evaluation
        print("\n  Coarse Reject-Option:")
        for thresh in [0.50, 0.60, 0.70, 0.80]:
            conf = proba_coarse.max(axis=1)
            mask = conf >= thresh
            if mask.sum() == 0:
                print(f"    Threshold {thresh}: No samples accepted")
                continue
            acc_t = accuracy_score(y_test_coarse[mask], preds_coarse[mask])
            cov = mask.mean()
            print(f"    Threshold {thresh:.2f}: Coverage={cov:.4f} | Acc={acc_t:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Hierarchical two-stage: Coarse -> Fine
    # ═══════════════════════════════════════════════════════════════════════
    if HAS_LIGHTGBM:
        print(f"\n{'=' * 60}")
        print("  HIERARCHICAL TWO-STAGE CLASSIFIER")
        print(f"{'=' * 60}")

        # Train per-coarse fine-grained classifiers
        stage2_models = {}
        inv_fine_enc = {v: k for k, v in fine_encoder.items()}

        for coarse_name in COARSE_CLASSES:
            coarse_idx = COARSE_ENCODER[coarse_name]
            fine_names = COARSE_TO_FINE.get(coarse_name, [])
            fine_indices = [fine_encoder[fn] for fn in fine_names if fn in fine_encoder]

            if len(fine_indices) <= 1:
                # Only one fine class in this coarse group, no stage-2 needed
                print(f"  {coarse_name}: single fine class, skipping stage-2")
                stage2_models[coarse_name] = None
                continue

            # Filter training data to this coarse group
            train_mask = np.isin(y_train_fine, fine_indices)
            val_mask = np.isin(y_val_fine, fine_indices)

            if train_mask.sum() < 10:
                print(f"  {coarse_name}: too few samples ({train_mask.sum()}), skipping")
                stage2_models[coarse_name] = None
                continue

            # Remap labels to 0..K-1 within this group
            label_remap = {old: new for new, old in enumerate(sorted(fine_indices))}
            y_train_sub = np.array([label_remap[l] for l in y_train_fine[train_mask]])
            y_val_sub = np.array([label_remap[l] for l in y_val_fine[val_mask]])
            n_sub = len(label_remap)

            # Use same feature extraction (but with group-specific prototypes)
            pca_sub = PCA(n_components=min(32, train_mask.sum() - 1), random_state=42)
            scaler_sub = StandardScaler()
            proto_sub = PrototypeExtractor()

            X_train_sub = build_features(
                X_train_raw[train_mask], y_train_sub,
                pca_sub, scaler_sub, proto_sub, fit=True
            )
            X_val_sub = build_features(
                X_val_raw[val_mask], y_val_sub,
                pca_sub, scaler_sub, proto_sub, fit=False
            )

            # Quick LightGBM
            clf_sub = lgb.LGBMClassifier(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                boosting_type='dart', class_weight='balanced',
                objective='multiclass', num_class=n_sub,
                n_jobs=2, random_state=42, verbose=-1,
            )
            clf_sub.fit(X_train_sub, y_train_sub,
                        eval_set=[(X_val_sub, y_val_sub)],
                        callbacks=[lgb.log_evaluation(period=0)])

            val_preds = clf_sub.predict(X_val_sub)
            val_f1 = f1_score(y_val_sub, val_preds, average="macro", zero_division=0)
            print(f"  {coarse_name} ({n_sub} fine classes): Val F1 = {val_f1:.4f}")

            stage2_models[coarse_name] = {
                "clf": clf_sub, "pca": pca_sub, "scaler": scaler_sub,
                "proto": proto_sub, "label_remap": label_remap,
                "inv_remap": {v: k for k, v in label_remap.items()},
            }

        # Hierarchical test evaluation
        print("\n  Hierarchical Test Evaluation:")
        hierarchical_preds = np.zeros(len(y_test_fine), dtype=np.int64)

        for i in range(len(y_test_fine)):
            coarse_pred = preds_coarse[i]
            coarse_name = COARSE_CLASSES[coarse_pred]
            fine_names = COARSE_TO_FINE.get(coarse_name, [])
            fine_indices = [fine_encoder[fn] for fn in fine_names if fn in fine_encoder]

            if len(fine_indices) <= 1 or stage2_models.get(coarse_name) is None:
                # Single fine class or no model
                hierarchical_preds[i] = fine_indices[0] if fine_indices else 0
            else:
                # Run stage-2 model
                m = stage2_models[coarse_name]
                x_feat = build_features(
                    X_test_raw[i:i+1], np.array([0]),
                    m["pca"], m["scaler"], m["proto"], fit=False
                )
                fine_pred_local = m["clf"].predict(x_feat)[0]
                hierarchical_preds[i] = m["inv_remap"][fine_pred_local]

        h_acc = accuracy_score(y_test_fine, hierarchical_preds)
        h_f1 = f1_score(y_test_fine, hierarchical_preds, average="macro", zero_division=0)
        print(f"\n  Hierarchical Fine-Grained Test Acc: {h_acc:.4f} | F1: {h_f1:.4f}")
        results.append({
            "Task": "Hierarchical Fine 9-class",
            "Model": "Coarse->Fine LightGBM",
            "Test_Accuracy": round(h_acc, 4),
            "Test_Macro_F1": round(h_f1, 4),
        })

        # Per-class
        fine_class_names = [inv_fine_enc[i] for i in range(num_fine)]
        print(classification_report(
            y_test_fine, hierarchical_preds,
            target_names=fine_class_names, zero_division=0
        ))

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  COARSE + HIERARCHICAL RESULTS")
    print(f"{'=' * 60}")
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    df_res.to_csv(os.path.join(coarse_dir, "coarse_results.csv"), index=False)

    with open(os.path.join(coarse_dir, "coarse_encoder.json"), "w") as f:
        json.dump(COARSE_ENCODER, f, indent=2)
    with open(os.path.join(coarse_dir, "coarse_mapping.json"), "w") as f:
        json.dump(COARSE_MAPPING, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coarse-Family Classifier")
    parser.add_argument("--config", default="configs/hybrid.yaml")
    args = parser.parse_args()
    main(args.config)
