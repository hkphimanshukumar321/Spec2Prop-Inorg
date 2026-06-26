import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Spec2Prop-Inorg: Train Multimodal Raman+XRD CNN
===============================================
Trains dual-branch CNN models to predict chemical family using both Raman and XRD.
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
from sklearn.metrics import accuracy_score, f1_score

from models import Spec2PropDataset, SimpleCNN1D, DualBranchRamanXRDNet
from models.dataset import build_label_encoder, save_label_encoder

def get_model(model_name, num_classes):
    if model_name == "SimpleCNN1D":
        # Raman only baseline
        class RamanOnly(nn.Module):
            def __init__(self, nc):
                super().__init__()
                self.cnn = SimpleCNN1D(1, num_classes=nc)
            def forward(self, raman, xrd):
                return {"family": self.cnn(raman)}
        return RamanOnly(num_classes)
        
    elif model_name == "DualBranchRamanXRDNet":
        return DualBranchRamanXRDNet(tasks={"family": num_classes}, descriptor_dim=0)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    for batch in dataloader:
        r = batch["raman"].to(device)
        x = batch.get("xrd", r).to(device) # fallback to raman if not using xrd
        y = batch["target_chemical_family_model"].to(device)
        mask = batch["mask_chemical_family_model"].to(device)
        
        valid_idx = torch.where(mask > 0)[0]
        if len(valid_idx) == 0: continue
            
        r, x, y = r[valid_idx], x[valid_idx], y[valid_idx]
        
        optimizer.zero_grad()
        logits = model(r, x)["family"]
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(y)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
        
    return total_loss / max(1, len(all_targets)), accuracy_score(all_targets, all_preds), f1_score(all_targets, all_preds, average="macro")

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            r = batch["raman"].to(device)
            x = batch.get("xrd", r).to(device)
            y = batch["target_chemical_family_model"].to(device)
            mask = batch["mask_chemical_family_model"].to(device)
            
            valid_idx = torch.where(mask > 0)[0]
            if len(valid_idx) == 0: continue
                
            r, x, y = r[valid_idx], x[valid_idx], y[valid_idx]
            logits = model(r, x)["family"]
            loss = criterion(logits, y)
            
            total_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    return total_loss / max(1, len(all_targets)), accuracy_score(all_targets, all_preds), f1_score(all_targets, all_preds, average="macro")

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    torch.manual_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg["output"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    
    meta_df = pd.read_pickle(cfg["data"]["metadata_pkl"])
    target_col = cfg["data"]["target_col"]
    from models.dataset import add_model_family_column
    meta_df = add_model_family_column(meta_df)
    
    encoder = build_label_encoder(meta_df[target_col])
    save_label_encoder(encoder, os.path.join(out_dir, "family_encoder.json"))
    num_classes = len(encoder)
    
    kwargs = dict(
        metadata_pkl=cfg["data"]["metadata_pkl"],
        raman_parquet=cfg["data"]["raman_parquet"],
        target_cols=[target_col],
        label_encoders={target_col: encoder},
        cache_dir=cfg["data"]["cache_dir"],
        use_xrd=True,
        xrd_pkl=cfg["data"].get("xrd_pkl")
    )
    
    train_dataset = Spec2PropDataset(split_csv=os.path.join(cfg["data"]["splits_dir"], "train.csv"), **kwargs)
    val_dataset = Spec2PropDataset(split_csv=os.path.join(cfg["data"]["splits_dir"], "val.csv"), **kwargs)
    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)
    
    w = train_dataset.get_class_weights(target_col) if cfg["training"].get("use_class_weights") else None
    criterion = nn.CrossEntropyLoss(weight=w.to(device) if w is not None else None)
    
    all_results = {}
    for model_name in cfg["models"]:
        print(f"\nTraining {model_name}")
        model = get_model(model_name, num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
        
        best_f1 = 0.0
        for epoch in range(cfg["training"]["epochs"]):
            tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), os.path.join(out_dir, f"{model_name}_best.pt"))
                print(f"Epoch {epoch+1:03d} | Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} [Best]")
            else:
                print(f"Epoch {epoch+1:03d} | Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
                
        all_results[model_name] = {"val_macro_f1": float(best_f1)}
        
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multimodal_raman_xrd.yaml")
    args = parser.parse_args()
    main(args.config)
