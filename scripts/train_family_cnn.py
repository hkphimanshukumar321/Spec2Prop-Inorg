import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Spec2Prop-Inorg: Train Family Classification
============================================
Trains 1D-CNNs for chemical family classification using Raman spectra.
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
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from models import Spec2PropDataset, SimpleCNN1D, LiteSpecNet, ResidualCNN1D, MultiScaleSpecNet, RamanPCAMLP, RamanFormer1D
from models.dataset import build_label_encoder, save_label_encoder

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_model(model_name, num_classes):
    if model_name == "SimpleCNN1D":
        return SimpleCNN1D(in_channels=1, num_classes=num_classes)
    elif model_name == "LiteSpecNet":
        return LiteSpecNet(in_channels=1, num_classes=num_classes)
    elif model_name == "ResidualCNN1D":
        return ResidualCNN1D(in_channels=1, num_classes=num_classes)
    elif model_name == "MultiScaleSpecNet":
        return MultiScaleSpecNet(in_channels=1, num_classes=num_classes)
    elif model_name == "RamanPCAMLP":
        # Note: pca_dim is injected later, we just need the class here
        # It defaults to 128 if not passed
        return RamanPCAMLP(in_channels=1, num_classes=num_classes)
    elif model_name == "RamanFormer1D":
        return RamanFormer1D(in_channels=1, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_epoch(model, dataloader, criterion, optimizer, device, target_col):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    for batch in dataloader:
        x = batch["raman"].to(device)
        y = batch[f"target_{target_col}"].to(device)
        mask = batch[f"mask_{target_col}"].to(device)
        
        valid_idx = torch.where(mask > 0)[0]
        if len(valid_idx) == 0:
            continue
            
        x, y = x[valid_idx], y[valid_idx]
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(y)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
        
    avg_loss = total_loss / max(1, len(all_targets))
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")
    return avg_loss, acc, f1

def evaluate(model, dataloader, criterion, device, target_col):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
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
            loss = criterion(logits, y)
            
            total_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    avg_loss = total_loss / max(1, len(all_targets))
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")
    
    return avg_loss, acc, f1, all_targets, all_preds

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    out_dir = cfg["output"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    
    meta_df = pd.read_pickle(cfg["data"]["metadata_pkl"])
    target_col = cfg["data"]["target_col"]
    from models.dataset import add_model_family_column
    meta_df = add_model_family_column(meta_df)
    
    encoder = build_label_encoder(meta_df[target_col])
    save_label_encoder(encoder, os.path.join(out_dir, f"{target_col}_encoder.json"))
    num_classes = len(encoder)
    print(f"Number of classes: {num_classes}")
    
    kwargs = dict(
        metadata_pkl=cfg["data"]["metadata_pkl"],
        raman_parquet=cfg["data"]["raman_parquet"],
        target_cols=[target_col],
        label_encoders={target_col: encoder},
        cache_dir=cfg["data"]["cache_dir"],
        random_shift=cfg["data"].get("random_shift", False),
        add_noise=cfg["data"].get("add_noise", False)
    )
    
    train_dataset = Spec2PropDataset(split_csv=os.path.join(cfg["data"]["splits_dir"], "train.csv"), **kwargs)
    val_dataset = Spec2PropDataset(split_csv=os.path.join(cfg["data"]["splits_dir"], "val.csv"), **kwargs)
    
    if cfg["training"].get("use_class_balanced_sampler"):
        from torch.utils.data import WeightedRandomSampler
        weights = train_dataset.get_class_weights(target_col)
        sample_weights = torch.zeros(len(train_dataset), dtype=torch.float)
        for i in range(len(train_dataset)):
            t = train_dataset.targets[target_col][i]
            if t >= 0:
                sample_weights[i] = weights[t]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], sampler=sampler, drop_last=True)
        print("Using Class-Balanced Sampler.")
    else:
        train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=True)
        
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)
    
    class_weights = None
    if cfg["training"].get("use_class_weights"):
        class_weights = train_dataset.get_class_weights(target_col)
        if class_weights is not None:
            class_weights = class_weights.to(device)
            print("Using class weights.")
            
    if cfg["training"].get("use_focal_loss"):
        criterion = FocalLoss(weight=class_weights, gamma=cfg["training"].get("focal_gamma", 2.0))
        print("Using Focal Loss.")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    all_results = {}
    
    for model_name in cfg["models"]:
        print(f"\n{'='*50}\nTraining {model_name}\n{'='*50}")
        
        # Handle RamanPCAMLP specifically
        if model_name == "RamanPCAMLP":
            pca_dim = cfg["training"].get("pca_dim", 128)
            model = RamanPCAMLP(in_channels=1, pca_dim=pca_dim, num_classes=num_classes).to(device)
            
            print(f"Computing PCA ({pca_dim} components) on training set for RamanPCAMLP initialization...")
            from sklearn.decomposition import PCA
            
            # Extract all train Raman spectra
            X_train = np.zeros((len(train_dataset), 2048), dtype=np.float32)
            for i in range(len(train_dataset)):
                X_train[i] = train_dataset[i]["raman"].numpy().flatten()
            
            pca = PCA(n_components=pca_dim)
            pca.fit(X_train)
            
            with torch.no_grad():
                model.pca_proj.weight.copy_(torch.from_numpy(pca.components_).float())
                bias = -np.dot(pca.mean_, pca.components_.T)
                model.pca_proj.bias.copy_(torch.from_numpy(bias).float())
            print("PCA weights frozen and injected successfully.")
        else:
            model = get_model(model_name, num_classes).to(device)
            
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=cfg["training"]["lr"], 
            weight_decay=cfg["training"]["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        
        best_val_f1 = 0.0
        patience_counter = 0
        best_model_path = os.path.join(out_dir, f"{model_name}_best.pt")
        history = []
        
        for epoch in range(cfg["training"]["epochs"]):
            t0 = time.time()
            tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, device, target_col)
            val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device, target_col)
            t1 = time.time()
            
            scheduler.step(val_f1)
            
            history.append({
                "epoch": epoch+1,
                "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                "time": t1-t0
            })
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
                print(f"Epoch {epoch+1:03d} | Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f} | Train F1: {tr_f1:.4f} | Val F1: {val_f1:.4f} | Loss: {val_loss:.4f} [Best]")
            else:
                patience_counter += 1
                print(f"Epoch {epoch+1:03d} | Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f} | Train F1: {tr_f1:.4f} | Val F1: {val_f1:.4f} | Loss: {val_loss:.4f}")
                
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
                
        # Load best and evaluate on val for final metrics
        model.load_state_dict(torch.load(best_model_path))
        _, _, final_val_f1, val_targets, val_preds = evaluate(model, val_loader, criterion, device, target_col)
        
        # Calculate per-class F1
        per_class_f1 = f1_score(val_targets, val_preds, average=None)
        all_results[model_name] = {
            "best_val_f1": float(final_val_f1),
            "per_class_f1": per_class_f1.tolist()
        }
        
        pd.DataFrame(history).to_csv(os.path.join(out_dir, f"{model_name}_history.csv"), index=False)
        
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)
        
    print("\nTraining completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/family_classification.yaml")
    args = parser.parse_args()
    main(args.config)
