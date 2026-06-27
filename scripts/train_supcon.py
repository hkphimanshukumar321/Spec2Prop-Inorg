import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Spec2Prop-Inorg: Supervised Contrastive Pre-training — Stage 1
===============================================================
Trains CNN backbones with SupCon loss to learn geometrically coherent
embeddings where same-class spectra cluster and different-class spectra
repel. These embeddings are then used by downstream GBDT classifiers.

Supports two modes:
  1. Two-stage: Pre-train encoder with SupCon → freeze → linear probe
  2. Joint:     SupCon + FocalLoss simultaneously (simpler, single pass)

The joint mode is default as it requires no separate fine-tuning step.
"""

import argparse
import json
import os
import time
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score

from models import Spec2PropDataset, SimpleCNN1D, LiteSpecNet, ResidualCNN1D, MultiScaleSpecNet, RamanFormer1D
from models.dataset import build_label_encoder, save_label_encoder, add_model_family_column
from models.losses import SupConLoss, FocalLoss

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=1)





def get_model(model_name, num_classes, dropout=0.3):
    """Instantiate a model by name."""
    models = {
        "SimpleCNN1D": lambda: SimpleCNN1D(in_channels=1, num_classes=num_classes, dropout=dropout),
        "LiteSpecNet": lambda: LiteSpecNet(in_channels=1, num_classes=num_classes, dropout=dropout),
        "ResidualCNN1D": lambda: ResidualCNN1D(in_channels=1, num_classes=num_classes, dropout=dropout),
        "MultiScaleSpecNet": lambda: MultiScaleSpecNet(in_channels=1, num_classes=num_classes, dropout=dropout),
        "RamanFormer1D": lambda: RamanFormer1D(in_channels=1, num_classes=num_classes, dropout=dropout),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    return models[model_name]()


def get_embed_dim(model_name):
    """Return the embedding dimension for each model."""
    dims = {
        "SimpleCNN1D": 128,
        "LiteSpecNet": 64,
        "ResidualCNN1D": 128,
        "MultiScaleSpecNet": 128,
        "RamanFormer1D": 128,
    }
    return dims.get(model_name, 128)


def train_epoch_supcon(model, proj_head, dataloader, supcon_criterion,
                       ce_criterion, optimizer, device, target_col,
                       supcon_weight=0.5):
    """Train one epoch with joint SupCon + CE/Focal loss."""
    model.train()
    proj_head.train()

    total_loss = 0.0
    total_supcon = 0.0
    total_ce = 0.0
    all_preds, all_targets = [], []
    n_samples = 0

    for batch in dataloader:
        x = batch["raman"].to(device)
        y = batch[f"target_{target_col}"].to(device)
        mask = batch[f"mask_{target_col}"].to(device)

        valid_idx = torch.where(mask > 0)[0]
        if len(valid_idx) < 2:  # Need at least 2 for contrastive
            continue

        x, y = x[valid_idx], y[valid_idx]

        optimizer.zero_grad()

        # Forward: get embeddings and logits
        embeddings = model.forward_features(x)
        logits = model.head(embeddings) if model.head is not None else None

        # SupCon on projected embeddings
        if hasattr(model, 'forward_proj'):
            # RamanFormer1D has built-in projection and normalisation
            projected = model.forward_proj(x)
        else:
            projected = proj_head(embeddings)
            
        projected_views = projected.unsqueeze(1) # shape: (B, 1, proj_dim)
        supcon_loss = supcon_criterion(projected_views, y)

        # CE/Focal loss on logits
        if logits is not None and ce_criterion is not None:
            ce_loss = ce_criterion(logits, y)
            loss = supcon_weight * supcon_loss + (1.0 - supcon_weight) * ce_loss
        else:
            ce_loss = torch.tensor(0.0, device=device)
            loss = supcon_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(proj_head.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item() * len(y)
        total_supcon += supcon_loss.item() * len(y)
        total_ce += ce_loss.item() * len(y)
        n_samples += len(y)

        if logits is not None:
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / max(1, n_samples)
    avg_supcon = total_supcon / max(1, n_samples)
    avg_ce = total_ce / max(1, n_samples)

    if len(all_targets) > 0:
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average="macro")
    else:
        acc = 0.0
        f1 = 0.0

    return avg_loss, avg_supcon, avg_ce, acc, f1


def evaluate_model(model, dataloader, ce_criterion, device, target_col):
    """Evaluate model on validation set."""
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["raman"].to(device)
            y = batch[f"target_{target_col}"].to(device)
            mask = batch[f"mask_{target_col}"].to(device)

            valid_idx = torch.where(mask > 0)[0]
            if len(valid_idx) == 0:
                continue

            x, y = x[valid_idx], y[valid_idx]
            logits = model(x)

            if ce_criterion is not None:
                loss = ce_criterion(logits, y)
                total_loss += loss.item() * len(y)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            n_samples += len(y)

    avg_loss = total_loss / max(1, n_samples)
    acc = accuracy_score(all_targets, all_preds) if all_targets else 0.0
    f1 = f1_score(all_targets, all_preds, average="macro") if all_targets else 0.0

    return avg_loss, acc, f1


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    supcon_cfg = cfg.get("supcon", {})
    out_dir = cfg.get("output", {}).get("results_dir", "results/hybrid")
    os.makedirs(out_dir, exist_ok=True)

    seed = supcon_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    target_col = data_cfg["target_col"]

    # Build encoder
    meta_df = pd.read_pickle(data_cfg["metadata_pkl"])
    meta_df = add_model_family_column(meta_df)
    encoder = build_label_encoder(meta_df[target_col])
    num_classes = len(encoder)
    save_label_encoder(encoder, os.path.join(out_dir, f"{target_col}_encoder.json"))
    print(f"Target: {target_col} | Classes: {num_classes}")

    # Load datasets
    kwargs = dict(
        metadata_pkl=data_cfg["metadata_pkl"],
        raman_parquet=data_cfg["raman_parquet"],
        target_cols=[target_col],
        label_encoders={target_col: encoder},
        cache_dir=data_cfg.get("cache_dir"),
        random_shift=True,    # Augmentation as "views" for contrastive learning
        add_noise=True,
    )

    train_dataset = Spec2PropDataset(
        split_csv=os.path.join(data_cfg["splits_dir"], "train.csv"), **kwargs
    )
    val_dataset = Spec2PropDataset(
        split_csv=os.path.join(data_cfg["splits_dir"], "val.csv"), **kwargs
    )

    # Class-balanced sampling
    weights = train_dataset.get_class_weights(target_col)
    sample_weights = torch.zeros(len(train_dataset), dtype=torch.float)
    for i in range(len(train_dataset)):
        t = train_dataset.targets[target_col][i]
        if t >= 0:
            sample_weights[i] = weights[t]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    batch_size = supcon_cfg.get("batch_size", 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Class weights for focal loss
    class_weights = train_dataset.get_class_weights(target_col)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    # Training config
    supcon_weight = supcon_cfg.get("supcon_weight", 0.5)
    temperature = supcon_cfg.get("temperature", 0.07)
    projection_dim = supcon_cfg.get("projection_dim", 128)
    epochs = supcon_cfg.get("epochs", 200)
    lr = supcon_cfg.get("lr", 0.0003)
    patience = supcon_cfg.get("early_stopping_patience", 20)
    backbones = supcon_cfg.get("backbones", ["SimpleCNN1D", "MultiScaleSpecNet"])

    all_results = {}

    for model_name in backbones:
        print(f"\n{'=' * 60}")
        print(f"  SupCon Training: {model_name}")
        print(f"  lambda_supcon={supcon_weight} | tau={temperature} | proj={projection_dim}")
        print(f"{'=' * 60}")

        embed_dim = get_embed_dim(model_name)
        model = get_model(model_name, num_classes).to(device)
        proj_head = ProjectionHead(in_dim=embed_dim, hidden_dim=256, out_dim=projection_dim).to(device)

        # Losses
        supcon_criterion = SupConLoss(temperature=temperature)
        ce_criterion = FocalLoss(weight=class_weights, gamma=2.0)

        # Optimizer for both model + projection head
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(proj_head.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=7, factor=0.5
        )

        best_val_f1 = 0.0
        patience_counter = 0
        best_model_path = os.path.join(out_dir, f"{model_name}_supcon_best.pt")
        history = []

        for epoch in range(epochs):
            t0 = time.time()

            tr_loss, tr_supcon, tr_ce, tr_acc, tr_f1 = train_epoch_supcon(
                model, proj_head, train_loader,
                supcon_criterion, ce_criterion, optimizer,
                device, target_col, supcon_weight=supcon_weight,
            )

            val_loss, val_acc, val_f1 = evaluate_model(
                model, val_loader, ce_criterion, device, target_col,
            )

            elapsed = time.time() - t0
            scheduler.step(val_f1)

            history.append({
                "epoch": epoch + 1,
                "train_loss": tr_loss, "train_supcon": tr_supcon, "train_ce": tr_ce,
                "train_acc": tr_acc, "train_f1": tr_f1,
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                "time": elapsed,
            })

            tag = ""
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                # Save encoder weights only (discard projection head)
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
                tag = " [Best]"
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or tag:
                print(
                    f"  Epoch {epoch + 1:03d} | "
                    f"SupCon: {tr_supcon:.4f} | CE: {tr_ce:.4f} | "
                    f"Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f} | "
                    f"Val F1: {val_f1:.4f}{tag}"
                )

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        all_results[model_name] = {
            "best_val_f1": float(best_val_f1),
            "checkpoint": best_model_path,
        }

        pd.DataFrame(history).to_csv(
            os.path.join(out_dir, f"{model_name}_supcon_history.csv"), index=False
        )
        print(f"  Best Val F1: {best_val_f1:.4f}")
        print(f"  Checkpoint: {best_model_path}")

    # Save summary
    with open(os.path.join(out_dir, "stage1_supcon_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("  STAGE 1 COMPLETE")
    print(f"{'=' * 60}")
    for name, res in all_results.items():
        print(f"  {name}: Best Val F1 = {res['best_val_f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: SupCon Training")
    parser.add_argument("--config", default="configs/hybrid.yaml")
    args = parser.parse_args()
    main(args.config)
