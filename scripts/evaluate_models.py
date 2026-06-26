import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Spec2Prop-Inorg: Evaluate Models
================================
Runs evaluation on the test set for all trained models across all tasks.
Calculates Accuracy, Precision, Recall, F1, and Confusion Matrices.
"""

import argparse
import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from models import Spec2PropDataset
from models.cnn1d import SimpleCNN1D, LiteSpecNet, ResidualCNN1D, MultiScaleSpecNet, RamanPCAMLP
from models.raman_former import RamanFormer1D
from models.fusion_models import FusionSpec2PropNet, Spec2PropLite, DescriptorMLP
from models.multimodal_cnn import DualBranchRamanXRDNet
from models.dataset import build_label_encoder, add_model_family_column, get_descriptor_dim

# --- Model Factories ---
def get_family_model(model_name, num_classes):
    if model_name == "SimpleCNN1D": return SimpleCNN1D(in_channels=1, num_classes=num_classes)
    elif model_name == "LiteSpecNet": return LiteSpecNet(in_channels=1, num_classes=num_classes)
    elif model_name == "ResidualCNN1D": return ResidualCNN1D(in_channels=1, num_classes=num_classes)
    elif model_name == "MultiScaleSpecNet": return MultiScaleSpecNet(in_channels=1, num_classes=num_classes)
    elif model_name == "RamanPCAMLP": return RamanPCAMLP(in_channels=1, num_classes=num_classes)
    elif model_name == "RamanFormer1D": return RamanFormer1D(in_channels=1, num_classes=num_classes)
    else: raise ValueError(f"Unknown family model: {model_name}")

def get_property_model(model_name, tasks, desc_dim):
    if model_name == "SimpleCNN1D": return FusionSpec2PropNet(tasks=tasks, descriptor_dim=0, cnn_type="SimpleCNN1D")
    elif model_name == "DescriptorMLP":
        import torch.nn as nn
        class DescOnly(nn.Module):
            def __init__(self, desc_in, desc_out, tasks):
                super().__init__()
                self.mlp = DescriptorMLP(desc_in, embed_dim=desc_out)
                from models.fusion_models import FusionMultiTaskHead
                self.head = FusionMultiTaskHead(desc_out, tasks)
            def forward(self, raman, descriptors):
                return self.head(self.mlp.forward_features(descriptors))
        return DescOnly(desc_in=desc_dim, desc_out=32, tasks=tasks)
    elif model_name == "FusionSpec2PropNet": return FusionSpec2PropNet(tasks=tasks, descriptor_dim=desc_dim, cnn_type="SimpleCNN1D")
    elif model_name == "Spec2PropLite": return Spec2PropLite(tasks=tasks, descriptor_dim=desc_dim)
    else: raise ValueError(f"Unknown property model: {model_name}")

def get_multimodal_model(model_name, tasks):
    if model_name == "SimpleCNN1D":
        import torch.nn as nn
        class RamanOnly(nn.Module):
            def __init__(self, tasks):
                super().__init__()
                # Assuming first task is family for num_classes
                nc = list(tasks.values())[0]
                self.cnn = SimpleCNN1D(1, num_classes=nc)
            def forward(self, raman, xrd):
                # Return dict matching target_col name
                task_name = list(tasks.keys())[0]
                return {task_name: self.cnn(raman)}
        return RamanOnly(tasks)
    elif model_name == "DualBranchRamanXRDNet": 
        return DualBranchRamanXRDNet(tasks=tasks, descriptor_dim=0)
    else: 
        raise ValueError(f"Unknown multimodal model: {model_name}")

# --- Core Evaluation Logic ---
def calc_all_metrics(y_true, y_pred, mask):
    valid_idx = np.where(mask > 0)[0]
    if len(valid_idx) == 0:
        return None, None
    y_t = y_true[valid_idx]
    y_p = y_pred[valid_idx]
    
    metrics = {
        "accuracy": float(accuracy_score(y_t, y_p)),
        "macro_precision": float(precision_score(y_t, y_p, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_t, y_p, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_t, y_p, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y_t, y_p, average=None, zero_division=0).tolist(),
    }
    cm = confusion_matrix(y_t, y_p).tolist()
    return metrics, cm

def run_inference(model, dataloader, target_cols, device, is_multimodal=False, is_property=False):
    model.eval()
    all_targets = {t: [] for t in target_cols}
    all_preds = {t: [] for t in target_cols}
    all_masks = {t: [] for t in target_cols}
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch["raman"].to(device)
            d = batch.get("descriptors", None)
            if d is not None: d = d.to(device)
            x_xrd = batch.get("xrd", None)
            if x_xrd is not None: x_xrd = x_xrd.to(device)
            
            if is_multimodal:
                logits = model(x, x_xrd)
                if "family" in logits:
                    logits[target_cols[0]] = logits["family"]
            elif is_property:
                logits = model(x, d)
            else:
                logits = {target_cols[0]: model(x)} # Wrap single output in dict for consistency
                
            for t_name in target_cols:
                y = batch[f"target_{t_name}"]
                mask = batch[f"mask_{t_name}"]
                
                preds = torch.argmax(logits[t_name], dim=1)
                all_preds[t_name].extend(preds.cpu().numpy())
                all_targets[t_name].extend(y.numpy())
                all_masks[t_name].extend(mask.numpy())
                
    results_metrics = {}
    results_cms = {}
    
    for t_name in target_cols:
        m, cm = calc_all_metrics(np.array(all_targets[t_name]), np.array(all_preds[t_name]), np.array(all_masks[t_name]))
        if m is not None:
            results_metrics[t_name] = m
            results_cms[t_name] = cm
            
    return results_metrics, results_cms

# --- Task Runners ---
def evaluate_family(cfg_path, device):
    with open(cfg_path) as f: cfg = yaml.safe_load(f)
    out_dir = cfg["output"]["results_dir"]
    if not os.path.exists(out_dir): return
    print("Evaluating Family Classification...")
    
    meta_df = add_model_family_column(pd.read_pickle(cfg["data"]["metadata_pkl"]))
    target_col = cfg["data"]["target_col"]
    encoder = build_label_encoder(meta_df[target_col])
    
    kwargs = dict(
        metadata_pkl=cfg["data"]["metadata_pkl"], raman_parquet=cfg["data"]["raman_parquet"],
        target_cols=[target_col], label_encoders={target_col: encoder}, cache_dir=cfg["data"]["cache_dir"]
    )
    test_dataset = Spec2PropDataset(split_csv=os.path.join(cfg["data"]["splits_dir"], "test.csv"), **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)
    
    final_metrics = {}
    final_cms = {}
    
    for model_name in cfg["models"]:
        pt_path = os.path.join(out_dir, f"{model_name}_best.pt")
        if not os.path.exists(pt_path): continue
        
        model = get_family_model(model_name, len(encoder)).to(device)
        model.load_state_dict(torch.load(pt_path, map_location=device))
        
        m, cm = run_inference(model, test_loader, [target_col], device)
        final_metrics[model_name] = m[target_col]
        final_cms[model_name] = {target_col: cm[target_col]}
        
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f: json.dump(final_metrics, f, indent=2)
    with open(os.path.join(out_dir, "confusion_matrices.json"), "w") as f: json.dump(final_cms, f, indent=2)

def evaluate_property(cfg_path, device):
    with open(cfg_path) as f: cfg = yaml.safe_load(f)
    out_dir = cfg["output"]["results_dir"]
    if not os.path.exists(out_dir): return
    print("Evaluating Property Prediction...")
    
    meta_df = add_model_family_column(pd.read_pickle(cfg["data"]["metadata_pkl"]))
    target_cols = cfg["data"]["target_cols"]
    encoders = {col: build_label_encoder(meta_df[col]) for col in target_cols}
    tasks_dict = {col: len(enc) for col, enc in encoders.items()}
    
    fam_enc = build_label_encoder(meta_df["chemical_family_model"])
    desc_dim = get_descriptor_dim(fam_enc)
    
    kwargs = dict(
        metadata_pkl=cfg["data"]["metadata_pkl"], raman_parquet=cfg["data"]["raman_parquet"],
        target_cols=target_cols, label_encoders=encoders, use_descriptors=True,
        family_encoder=fam_enc, cache_dir=cfg["data"]["cache_dir"]
    )
    test_dataset = Spec2PropDataset(split_csv=os.path.join(cfg["data"]["splits_dir"], "test.csv"), **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)
    
    final_metrics = {}
    final_cms = {}
    
    for model_name in cfg["models"]:
        pt_path = os.path.join(out_dir, f"{model_name}_best.pt")
        if not os.path.exists(pt_path): continue
        
        model = get_property_model(model_name, tasks_dict, desc_dim).to(device)
        model.load_state_dict(torch.load(pt_path, map_location=device))
        
        m, cm = run_inference(model, test_loader, target_cols, device, is_property=True)
        final_metrics[model_name] = m
        final_cms[model_name] = cm
        
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f: json.dump(final_metrics, f, indent=2)
    with open(os.path.join(out_dir, "confusion_matrices.json"), "w") as f: json.dump(final_cms, f, indent=2)

def evaluate_multimodal(cfg_path, device):
    with open(cfg_path) as f: cfg = yaml.safe_load(f)
    out_dir = cfg["output"]["results_dir"]
    if not os.path.exists(out_dir): return
    print("Evaluating Multimodal Models...")
    
    meta_df = add_model_family_column(pd.read_pickle(cfg["data"]["metadata_pkl"]))
    target_col = cfg["data"]["target_col"]
    encoder = build_label_encoder(meta_df[target_col])
    
    kwargs = dict(
        metadata_pkl=cfg["data"]["metadata_pkl"], raman_parquet=cfg["data"]["raman_parquet"],
        target_cols=[target_col], label_encoders={target_col: encoder}, use_xrd=True,
        xrd_pkl=cfg["data"].get("xrd_pkl"), cache_dir=cfg["data"]["cache_dir"]
    )
    test_dataset = Spec2PropDataset(split_csv=os.path.join(cfg["data"]["splits_dir"], "test.csv"), **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)
    
    final_metrics = {}
    final_cms = {}
    
    for model_name in cfg["models"]:
        pt_path = os.path.join(out_dir, f"{model_name}_best.pt")
        if not os.path.exists(pt_path): continue
        
        model = get_multimodal_model(model_name, {"family": len(encoder)}).to(device)
        model.load_state_dict(torch.load(pt_path, map_location=device))
        
        m, cm = run_inference(model, test_loader, [target_col], device, is_multimodal=True)
        final_metrics[model_name] = m[target_col]
        final_cms[model_name] = {target_col: cm[target_col]}
        
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f: json.dump(final_metrics, f, indent=2)
    with open(os.path.join(out_dir, "confusion_matrices.json"), "w") as f: json.dump(final_cms, f, indent=2)

def main():
    print("="*60)
    print("Spec2Prop-Inorg Test-Set Evaluation")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating using device: {device}\n")
    
    evaluate_family("configs/family_classification.yaml", device)
    evaluate_property("configs/property_prediction.yaml", device)
    evaluate_multimodal("configs/multimodal_raman_xrd.yaml", device)
    
    print("\nEvaluation Complete! Results saved to test_metrics.json and confusion_matrices.json")

if __name__ == "__main__":
    main()
