"""
train_hybrid.py — Two-stage SupCon → LightGBM Hybrid Pipeline

Pipeline overview
─────────────────
Stage 1 | SupCon pretraining
  • Train RamanFormer1D backbone + projection head with Supervised Contrastive Loss
  • Objective: learn a geometrically structured embedding space where same-class
    Raman spectra cluster tightly on the unit hypersphere
  • NO classification head involved; purely self-supervised on label information
  • Augmentations: Gaussian noise, random scaling, spectral masking

Stage 2 | CE fine-tuning (optional but recommended)
  • Freeze backbone, train only the linear classification head
  • Use Focal Loss (γ=2) with label smoothing to handle class imbalance
  • Early stopping on val F1

Stage 3 | GBDT on heterogeneous features
  • Extract CLS embeddings from Stage 1 backbone (forward_features())
  • Concatenate with domain features (peaks, band ratios, subsampled spectrum)
  • Train LightGBM with DART booster + Optuna hyperparameter search
  • This is the final, highest-accuracy classifier

Usage
─────
  python train_hybrid.py \
      --data_path /path/to/your/dataset.npz \
      --num_classes 22 \
      --output_dir ./checkpoints \
      --stage1_epochs 200 \
      --stage2_epochs 50 \
      --n_optuna_trials 50

Expected data format (.npz):
  X_train : (N, 2048) float32   — raw spectra
  y_train : (N,) int64          — class labels 0..num_classes-1
  X_val   : (M, 2048) float32
  y_val   : (M,) int64
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import joblib

# ── Imports from this package ──────────────────────────────────────────────
from raman_former import RamanFormer1D
from losses import SupConLoss, FocalLoss
from domain_features import extract_domain_features, build_feature_pipeline, transform_features


# ─────────────────────────────────────────────────────────────────────────────
# Spectral Augmentations (training-time only)
# ─────────────────────────────────────────────────────────────────────────────
def augment_spectrum(x: torch.Tensor, noise_std: float = 0.02,
                     scale_range=(0.9, 1.1), mask_frac: float = 0.05) -> torch.Tensor:
    """
    Three augmentations applied randomly in sequence:
      1. Additive Gaussian noise      — simulates detector noise
      2. Random multiplicative scale  — simulates laser power variation
      3. Random contiguous masking    — simulates cosmic ray removal gaps

    All are physically motivated for Raman spectroscopy.
    """
    # Noise
    x = x + torch.randn_like(x) * noise_std
    # Scale
    scale = torch.FloatTensor(1).uniform_(*scale_range).to(x.device)
    x = x * scale
    # Mask
    L = x.size(-1)
    mask_len = int(L * mask_frac)
    if mask_len > 0:
        start = torch.randint(0, L - mask_len, (1,)).item()
        x = x.clone()
        x[..., start: start + mask_len] = 0.0
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Supervised Contrastive Pretraining
# ─────────────────────────────────────────────────────────────────────────────
def train_supcon(model, X_train, y_train, X_val, y_val,
                 epochs=200, batch_size=64, lr=3e-4, temperature=0.07,
                 device="cuda", output_dir="./checkpoints"):
    """
    Train backbone with SupCon loss.
    Two augmented views of each spectrum are generated on-the-fly.
    """
    print("\n=== Stage 1: Supervised Contrastive Pretraining ===")
    os.makedirs(output_dir, exist_ok=True)

    X_t = torch.FloatTensor(X_train)
    y_t = torch.LongTensor(y_train)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=0)

    # Only backbone + proj_head parameters; head is None during Stage 1
    params = list(model.tokenizer.parameters()) + \
             list(model.encoder.parameters()) + \
             list(model.norm.parameters()) + \
             list(model.proj_head.parameters()) + \
             [model.cls_token, model.pos_emb]

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = SupConLoss(temperature=temperature)

    model = model.to(device)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Two augmented views per sample
            view1 = augment_spectrum(x_batch)
            view2 = augment_spectrum(x_batch)

            z1 = model.forward_proj(view1)   # (B, proj_dim), L2-normed
            z2 = model.forward_proj(view2)   # (B, proj_dim)

            # Stack as (B, 2, proj_dim) for SupCon
            features = torch.stack([z1, z2], dim=1)
            loss = criterion(features, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:>3d}/{epochs} | SupCon Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "supcon_backbone.pt"))

    print(f"  Best SupCon loss: {best_loss:.4f}  →  saved to {output_dir}/supcon_backbone.pt")
    # Load best checkpoint
    model.load_state_dict(torch.load(os.path.join(output_dir, "supcon_backbone.pt"), map_location=device))
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Linear Head Fine-tuning (Focal Loss)
# ─────────────────────────────────────────────────────────────────────────────
def finetune_head(model, X_train, y_train, X_val, y_val,
                  num_classes, epochs=50, batch_size=64, lr=1e-3,
                  device="cuda", output_dir="./checkpoints"):
    """
    Freeze backbone; train only classification head with Focal Loss.
    Early stops on validation macro F1.
    """
    print("\n=== Stage 2: Linear Head Fine-tuning ===")

    # Freeze backbone
    for name, p in model.named_parameters():
        if "head" not in name:
            p.requires_grad_(False)

    # Build head if not present (model was built with num_classes=None for Stage 1)
    if model.head is None:
        model.head = nn.Sequential(
            nn.Linear(model.d_model, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        ).to(device)

    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.LongTensor(y_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.LongTensor(y_val)

    # Compute class weights for imbalance
    counts = np.bincount(y_train, minlength=num_classes).astype(float)
    weights = (1.0 / (counts + 1e-8))
    weights = weights / weights.sum() * num_classes
    weight_tensor = torch.FloatTensor(weights).to(device)

    criterion = FocalLoss(gamma=2.0, weight=weight_tensor, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_f1, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for x_b, y_b in loader:
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits_v = model(X_v)
            preds = logits_v.argmax(dim=1).cpu().numpy()

        acc = accuracy_score(y_v.numpy(), preds)
        f1 = f1_score(y_v.numpy(), preds, average="macro", zero_division=0)
        scheduler.step(1 - f1)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:>3d}/{epochs} | Val Acc: {acc*100:.2f}% | Val Macro F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    print(f"  Best Val Macro F1: {best_f1:.4f}")

    # Unfreeze for future use
    for p in model.parameters():
        p.requires_grad_(True)

    return model, best_f1


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: LightGBM on Heterogeneous Features
# ─────────────────────────────────────────────────────────────────────────────
def extract_embeddings(model, X: np.ndarray, device="cuda", batch_size=128) -> np.ndarray:
    """Extract CLS-token embeddings from the backbone in batches."""
    model.eval()
    model = model.to(device)
    embs = []
    X_t = torch.FloatTensor(X)
    for i in range(0, len(X), batch_size):
        x_b = X_t[i: i + batch_size].to(device)
        with torch.no_grad():
            e = model.forward_features(x_b)
        embs.append(e.cpu().numpy())
    return np.concatenate(embs, axis=0)


def build_heterogeneous_features(model, X_train, X_val, device, pca_components=32):
    """
    Build the full heterogeneous feature matrix for LightGBM:
      [CNN embedding (d_model)] + [domain PCA (pca_components)]
    """
    print("\n  Extracting CNN embeddings...")
    emb_train = extract_embeddings(model, X_train, device)   # (N, d_model)
    emb_val   = extract_embeddings(model, X_val,   device)   # (M, d_model)

    print("  Extracting domain features...")
    dom_train_raw = extract_domain_features(X_train)         # (N, 126)
    dom_val_raw   = extract_domain_features(X_val)           # (M, 126)

    print("  Fitting PCA on domain features...")
    dom_pipeline = build_feature_pipeline(dom_train_raw, pca_components=pca_components)
    dom_train = transform_features(dom_pipeline, dom_train_raw)   # (N, 32)
    dom_val   = transform_features(dom_pipeline, dom_val_raw)     # (M, 32)

    X_hybrid_train = np.concatenate([emb_train, dom_train], axis=1)   # (N, d_model+32)
    X_hybrid_val   = np.concatenate([emb_val,   dom_val],   axis=1)   # (M, d_model+32)

    print(f"  Heterogeneous feature dim: {X_hybrid_train.shape[1]}")
    return X_hybrid_train, X_hybrid_val, dom_pipeline


def train_lightgbm(X_train, y_train, X_val, y_val, num_classes,
                   n_trials=50, output_dir="./checkpoints"):
    """
    Train LightGBM with Optuna hyperparameter optimisation.
    Uses DART booster for small-dataset regularisation.
    """
    try:
        import lightgbm as lgb
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Install lightgbm and optuna: pip install lightgbm optuna")

    print(f"\n=== Stage 3: LightGBM (DART) with {n_trials} Optuna trials ===")

    # Class weights for imbalance
    counts = np.bincount(y_train, minlength=num_classes).astype(float)
    sample_weights = (1.0 / counts[y_train])
    sample_weights = sample_weights / sample_weights.sum() * len(y_train)

    def objective(trial):
        params = {
            "boosting_type": "dart",
            "objective": "multiclass",
            "num_class": num_classes,
            "metric": "multi_logloss",
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "num_leaves": trial.suggest_int("num_leaves", 20, 127),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "drop_rate": trial.suggest_float("drop_rate", 0.05, 0.3),  # DART-specific
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1,
        }
        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        preds = clf.predict(X_val)
        return f1_score(y_val, preds, average="macro", zero_division=0)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best trial F1: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Retrain with best params
    best_params = study.best_params
    best_params.update({
        "boosting_type": "dart",
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    })

    import lightgbm as lgb  # re-import for final model
    final_clf = lgb.LGBMClassifier(**best_params)
    final_clf.fit(X_train, y_train, sample_weight=sample_weights)

    preds = final_clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1  = f1_score(y_val, preds, average="macro", zero_division=0)
    print(f"\n  Final LightGBM | Val Acc: {acc*100:.2f}% | Val Macro F1: {f1:.4f}")

    # Save
    model_path = os.path.join(output_dir, "lightgbm_hybrid.pkl")
    joblib.dump(final_clf, model_path)
    print(f"  Saved → {model_path}")

    return final_clf, acc, f1


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SupCon → LightGBM Hybrid Training")
    parser.add_argument("--data_path",       type=str, default="./data.npz")
    parser.add_argument("--num_classes",     type=int, default=22)
    parser.add_argument("--output_dir",      type=str, default="./checkpoints")
    parser.add_argument("--d_model",         type=int, default=128)
    parser.add_argument("--stage1_epochs",   type=int, default=200)
    parser.add_argument("--stage2_epochs",   type=int, default=50)
    parser.add_argument("--batch_size",      type=int, default=64)
    parser.add_argument("--supcon_temp",     type=float, default=0.07)
    parser.add_argument("--n_optuna_trials", type=int, default=50)
    parser.add_argument("--skip_stage1",     action="store_true",
                        help="Load existing supcon_backbone.pt and skip Stage 1")
    parser.add_argument("--skip_stage2",     action="store_true")
    parser.add_argument("--device",          type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading data from {args.data_path}...")
    data = np.load(args.data_path)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    X_val   = data["X_val"].astype(np.float32)
    y_val   = data["y_val"].astype(np.int64)
    print(f"  Train: {X_train.shape}  |  Val: {X_val.shape}  |  Classes: {args.num_classes}")

    # ── Instantiate model (no head for Stage 1) ──────────────────────────────
    model = RamanFormer1D(
        in_channels=1,
        num_classes=None,          # no head during SupCon pretraining
        d_model=args.d_model,
        nhead=4,
        num_encoder_layers=4,
        dim_feedforward=512,
        dropout=0.2,
        drop_path_rate=0.1,
        proj_dim=128,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # ── Stage 1: SupCon pretraining ──────────────────────────────────────────
    ckpt_path = os.path.join(args.output_dir, "supcon_backbone.pt")
    if args.skip_stage1 and os.path.exists(ckpt_path):
        print(f"\n  [skip] Loading existing backbone from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    else:
        model = train_supcon(
            model, X_train, y_train, X_val, y_val,
            epochs=args.stage1_epochs,
            batch_size=args.batch_size,
            lr=3e-4,
            temperature=args.supcon_temp,
            device=args.device,
            output_dir=args.output_dir,
        )

    # ── Stage 2: CE fine-tuning ───────────────────────────────────────────────
    if not args.skip_stage2:
        model, stage2_f1 = finetune_head(
            model, X_train, y_train, X_val, y_val,
            num_classes=args.num_classes,
            epochs=args.stage2_epochs,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir,
        )
        torch.save(model.state_dict(), os.path.join(args.output_dir, "stage2_full.pt"))

    # ── Stage 3: LightGBM on heterogeneous features ──────────────────────────
    X_h_train, X_h_val, dom_pipeline = build_heterogeneous_features(
        model, X_train, X_val, device=args.device, pca_components=32
    )
    joblib.dump(dom_pipeline, os.path.join(args.output_dir, "domain_pipeline.pkl"))

    lgbm_clf, lgbm_acc, lgbm_f1 = train_lightgbm(
        X_h_train, y_train, X_h_val, y_val,
        num_classes=args.num_classes,
        n_trials=args.n_optuna_trials,
        output_dir=args.output_dir,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Stage 2 (CE head only)  : F1 = {stage2_f1:.4f}" if not args.skip_stage2 else "")
    print(f"  Stage 3 (LightGBM hybrid): Acc = {lgbm_acc*100:.2f}% | F1 = {lgbm_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
