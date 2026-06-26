import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Spec2Prop-Inorg: Train Property Prediction
==========================================
Trains models for multi-task property prediction (band gap class, metallicity, formation energy class).
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

from models import Spec2PropDataset, SimpleCNN1D, DescriptorMLP, FusionSpec2PropNet, Spec2PropLite
from models.dataset import build_label_encoder, save_label_encoder, get_descriptor_dim

def get_model(model_name, tasks, raman_dim=128, desc_dim=32):
    if model_name == "SimpleCNN1D":
        return FusionSpec2PropNet(tasks=tasks, descriptor_dim=0, cnn_type="SimpleCNN1D")
    elif model_name == "DescriptorMLP":
        # A bit hacky: reuse FusionSpec2PropNet but we will just pass dummy raman or mock it
        # Actually better: a custom wrapper that just ignores raman
        class DescOnly(nn.Module):
            def __init__(self, desc_in, desc_out, tasks):
                super().__init__()
                self.mlp = DescriptorMLP(desc_in, embed_dim=desc_out)
                from models.fusion_models import FusionMultiTaskHead
                self.head = FusionMultiTaskHead(desc_out, tasks)
            def forward(self, raman, descriptors):
                return self.head(self.mlp.forward_features(descriptors))
        return DescOnly(desc_in=desc_dim, desc_out=32, tasks=tasks)
    elif model_name == "FusionSpec2PropNet":
        return FusionSpec2PropNet(tasks=tasks, descriptor_dim=desc_dim, cnn_type="SimpleCNN1D")
    elif model_name == "Spec2PropLite":
        return Spec2PropLite(tasks=tasks, descriptor_dim=desc_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def calculate_metrics(y_true, y_pred, mask):
    valid_idx = np.where(mask > 0)[0]
    if len(valid_idx) == 0:
        return 0.0, 0.0
    y_t = y_true[valid_idx]
    y_p = y_pred[valid_idx]
    acc = accuracy_score(y_t, y_p)
    f1 = f1_score(y_t, y_p, average="macro")
    return acc, f1

def train_epoch(model, dataloader, criterions, loss_weights, optimizer, device):
    model.train()
    total_loss = 0.0
    
    # Store predictions for metrics
    all_targets = {t: [] for t in criterions}
    all_preds = {t: [] for t in criterions}
    all_masks = {t: [] for t in criterions}
    
    for batch in dataloader:
        x = batch["raman"].to(device)
        d = batch.get("descriptors", None)
        if d is not None: d = d.to(device)
        
        optimizer.zero_grad()
        logits_dict = model(x, d)
        
        loss = 0.0
        batch_has_valid = False
        
        for t_name, criterion in criterions.items():
            y = batch[f"target_{t_name}"].to(device)
            mask = batch[f"mask_{t_name}"].to(device)
            
            valid_idx = torch.where(mask > 0)[0]
            if len(valid_idx) > 0:
                batch_has_valid = True
                task_loss = criterion(logits_dict[t_name][valid_idx], y[valid_idx])
                loss += task_loss * loss_weights.get(t_name, 1.0)
            
            preds = torch.argmax(logits_dict[t_name], dim=1)
            all_preds[t_name].extend(preds.cpu().numpy())
            all_targets[t_name].extend(y.cpu().numpy())
            all_masks[t_name].extend(mask.cpu().numpy())
            
        if batch_has_valid:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
    metrics = {}
    for t_name in criterions:
        acc, f1 = calculate_metrics(np.array(all_targets[t_name]), np.array(all_preds[t_name]), np.array(all_masks[t_name]))
        metrics[t_name] = {"acc": acc, "f1": f1}
        
    return total_loss / len(dataloader), metrics

def evaluate(model, dataloader, criterions, loss_weights, device):
    model.eval()
    total_loss = 0.0
    
    all_targets = {t: [] for t in criterions}
    all_preds = {t: [] for t in criterions}
    all_masks = {t: [] for t in criterions}
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch["raman"].to(device)
            d = batch.get("descriptors", None)
            if d is not None: d = d.to(device)
            
            logits_dict = model(x, d)
            loss = 0.0
            batch_has_valid = False
            
            for t_name, criterion in criterions.items():
                y = batch[f"target_{t_name}"].to(device)
                mask = batch[f"mask_{t_name}"].to(device)
                
                valid_idx = torch.where(mask > 0)[0]
                if len(valid_idx) > 0:
                    batch_has_valid = True
                    task_loss = criterion(logits_dict[t_name][valid_idx], y[valid_idx])
                    loss += task_loss * loss_weights.get(t_name, 1.0)
                
                preds = torch.argmax(logits_dict[t_name], dim=1)
                all_preds[t_name].extend(preds.cpu().numpy())
                all_targets[t_name].extend(y.cpu().numpy())
                all_masks[t_name].extend(mask.cpu().numpy())
                
            if batch_has_valid:
                total_loss += loss.item()
                
    metrics = {}
    for t_name in criterions:
        acc, f1 = calculate_metrics(np.array(all_targets[t_name]), np.array(all_preds[t_name]), np.array(all_masks[t_name]))
        metrics[t_name] = {"acc": acc, "f1": f1}
        
    # Calculate macro average F1 across all tasks for early stopping
    avg_val_f1 = np.mean([metrics[t]["f1"] for t in metrics])
        
    return total_loss / len(dataloader), metrics, avg_val_f1

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    out_dir = cfg["output"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    
    # --- Prepare Encoders ---
    meta_df = pd.read_pickle(cfg["data"]["metadata_pkl"])
    target_cols = cfg["data"]["target_cols"]
    
    from models.dataset import add_model_family_column
    meta_df = add_model_family_column(meta_df)
    
    encoders = {}
    tasks_dict = {}
    for col in target_cols:
        enc = build_label_encoder(meta_df[col])
        encoders[col] = enc
        tasks_dict[col] = len(enc)
        save_label_encoder(enc, os.path.join(out_dir, f"{col}_encoder.json"))
        
    # Also need family encoder for descriptors
    fam_enc = build_label_encoder(meta_df["chemical_family_model"])
    desc_dim = get_descriptor_dim(fam_enc)
    
    # --- Datasets ---
    kwargs = dict(
        metadata_pkl=cfg["data"]["metadata_pkl"],
        raman_parquet=cfg["data"]["raman_parquet"],
        target_cols=target_cols,
        label_encoders=encoders,
        use_descriptors=True,
        family_encoder=fam_enc,
        cache_dir=cfg["data"]["cache_dir"]
    )
    
    train_dataset = Spec2PropDataset(split_csv=os.path.join(cfg["data"]["splits_dir"], "train.csv"), **kwargs)
    val_dataset = Spec2PropDataset(split_csv=os.path.join(cfg["data"]["splits_dir"], "val.csv"), **kwargs)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)
    
    criterions = {}
    for col in target_cols:
        weights = None
        if cfg["training"].get("use_class_weights"):
            weights = train_dataset.get_class_weights(col)
            if weights is not None:
                weights = weights.to(device)
        criterions[col] = nn.CrossEntropyLoss(weight=weights)
        
    loss_weights = cfg["training"].get("loss_weights", {col: 1.0 for col in target_cols})
    
    # --- Training Loop ---
    all_results = {}
    
    for model_name in cfg["models"]:
        print(f"\n{'='*50}\nTraining {model_name}\n{'='*50}")
        model = get_model(model_name, tasks_dict, desc_dim=desc_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        
        best_val_f1 = 0.0
        patience_counter = 0
        best_model_path = os.path.join(out_dir, f"{model_name}_best.pt")
        history = []
        
        for epoch in range(cfg["training"]["epochs"]):
            tr_loss, tr_metrics = train_epoch(model, train_loader, criterions, loss_weights, optimizer, device)
            val_loss, val_metrics, avg_val_f1 = evaluate(model, val_loader, criterions, loss_weights, device)
            
            avg_tr_acc = np.mean([tr_metrics[t]["acc"] for t in tr_metrics])
            avg_val_acc = np.mean([val_metrics[t]["acc"] for t in val_metrics])
            
            scheduler.step(avg_val_f1)
            
            history.append({
                "epoch": epoch+1,
                "train_loss": tr_loss, "val_loss": val_loss, "val_avg_f1": avg_val_f1, "val_avg_acc": avg_val_acc
            })
            
            if avg_val_f1 > best_val_f1:
                best_val_f1 = avg_val_f1
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
                print(f"Epoch {epoch+1:03d} | Train Acc: {avg_tr_acc:.4f} | Val Acc: {avg_val_acc:.4f} | Avg Val F1: {avg_val_f1:.4f} | Loss: {val_loss:.4f} [Best]")
            else:
                patience_counter += 1
                print(f"Epoch {epoch+1:03d} | Train Acc: {avg_tr_acc:.4f} | Val Acc: {avg_val_acc:.4f} | Avg Val F1: {avg_val_f1:.4f} | Loss: {val_loss:.4f}")
                
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                print("Early stopping triggered.")
                break
                
        # Final evaluation
        model.load_state_dict(torch.load(best_model_path))
        _, final_val_metrics, final_avg_f1 = evaluate(model, val_loader, criterions, loss_weights, device)
        all_results[model_name] = {
            "avg_val_f1": float(final_avg_f1),
            "tasks": {k: {"f1": float(v["f1"]), "acc": float(v["acc"])} for k, v in final_val_metrics.items()}
        }
        pd.DataFrame(history).to_csv(os.path.join(out_dir, f"{model_name}_history.csv"), index=False)
        
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/property_prediction.yaml")
    args = parser.parse_args()
    main(args.config)
