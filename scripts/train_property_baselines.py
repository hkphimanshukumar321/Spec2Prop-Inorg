import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Spec2Prop-Inorg: Train Property Prediction Baselines
====================================================
Trains RandomForest and XGBoost on descriptors and PCA-reduced Raman spectra.
"""

import argparse
import os
import yaml
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from models import Spec2PropDataset
from models.baselines import BaselineModel
from models.dataset import build_label_encoder, add_model_family_column

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["property_baselines"]
        
    out_dir = cfg["output"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    
    meta_df = pd.read_pickle(cfg["data"]["metadata_pkl"])
    meta_df = add_model_family_column(meta_df)
    
    target_cols = cfg["data"]["target_cols"]
    encoders = {col: build_label_encoder(meta_df[col]) for col in target_cols}
    fam_enc = build_label_encoder(meta_df["chemical_family_model"])
    
    kwargs = dict(
        metadata_pkl=cfg["data"]["metadata_pkl"],
        raman_parquet=cfg["data"]["raman_parquet"],
        target_cols=target_cols,
        label_encoders=encoders,
        use_descriptors=True,
        family_encoder=fam_enc,
        cache_dir=cfg["data"]["cache_dir"],
    )
    
    train_dataset = Spec2PropDataset(split_csv=os.path.join(cfg["data"]["splits_dir"], "train.csv"), **kwargs)
    test_dataset = Spec2PropDataset(split_csv=os.path.join(cfg["data"]["splits_dir"], "test.csv"), **kwargs)
    
    def get_arrays(ds, target_col, include_raman=False):
        X, y = [], []
        for i in range(len(ds)):
            b = ds[i]
            if b[f"mask_{target_col}"].item() > 0:
                features = [b["descriptors"].numpy()]
                if include_raman:
                    features.append(b["raman"].numpy().flatten())
                X.append(np.concatenate(features))
                y.append(b[f"target_{target_col}"].item())
        return np.array(X), np.array(y)
        
    final_metrics = {}

    for target_col in target_cols:
        print(f"\n{'='*40}\nTask: {target_col}\n{'='*40}")
        
        # Descriptors Only
        print("Extracting train arrays (Descriptors Only)...")
        X_train_desc, y_train_desc = get_arrays(train_dataset, target_col, include_raman=False)
        X_test_desc, y_test_desc = get_arrays(test_dataset, target_col, include_raman=False)
        
        # Descriptors + Raman
        print("Extracting train arrays (Descriptors + Raman)...")
        X_train_full, y_train_full = get_arrays(train_dataset, target_col, include_raman=True)
        X_test_full, y_test_full = get_arrays(test_dataset, target_col, include_raman=True)
        
        for model_name in cfg["models"]:
            if model_name not in final_metrics:
                final_metrics[model_name] = {}
                
            # 1. Descriptor-Only Model (No PCA needed for just descriptors)
            print(f"\nTraining {model_name} (Descriptors Only)")
            try:
                model_desc = BaselineModel(model_name)
                model_desc.fit(X_train_desc, y_train_desc)
                preds = model_desc.predict(X_test_desc)
                
                acc = accuracy_score(y_test_desc, preds)
                f1 = f1_score(y_test_desc, preds, average="macro")
                print(f"  -> Acc: {acc:.4f} | F1: {f1:.4f}")
                
                final_metrics[model_name][f"{target_col}_desc_only"] = {"accuracy": acc, "macro_f1": f1}
            except Exception as e:
                print(f"  -> Failed: {e}")
                
            # 2. Descriptor + Raman Model (with PCA over the Raman features part)
            for pca_dim in cfg["pca_dims"]:
                print(f"\nTraining {model_name} (Descriptors + Raman PCA: {pca_dim})")
                try:
                    # We only PCA the raman part of the array
                    from sklearn.decomposition import PCA
                    X_train_r = X_train_full[:, X_train_desc.shape[1]:]
                    X_test_r = X_test_full[:, X_test_desc.shape[1]:]
                    
                    if pca_dim:
                        pca = PCA(n_components=pca_dim, random_state=42)
                        X_train_r_pca = pca.fit_transform(X_train_r)
                        X_test_r_pca = pca.transform(X_test_r)
                    else:
                        X_train_r_pca = X_train_r
                        X_test_r_pca = X_test_r
                        
                    X_tr_final = np.concatenate([X_train_full[:, :X_train_desc.shape[1]], X_train_r_pca], axis=1)
                    X_te_final = np.concatenate([X_test_full[:, :X_test_desc.shape[1]], X_test_r_pca], axis=1)
                    
                    model_full = BaselineModel(model_name) # Internal PCA disabled since we did it manually
                    model_full.fit(X_tr_final, y_train_full)
                    preds = model_full.predict(X_te_final)
                    
                    acc = accuracy_score(y_test_full, preds)
                    f1 = f1_score(y_test_full, preds, average="macro")
                    print(f"  -> Acc: {acc:.4f} | F1: {f1:.4f}")
                    
                    key_name = f"{target_col}_pca_{pca_dim}" if pca_dim else f"{target_col}_raw"
                    final_metrics[model_name][key_name] = {"accuracy": acc, "macro_f1": f1}
                except Exception as e:
                    print(f"  -> Failed: {e}")
                    
    with open(os.path.join(out_dir, "property_baselines_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    print("\nSaved property baselines results to:", os.path.join(out_dir, "property_baselines_metrics.json"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baselines.yaml")
    args = parser.parse_args()
    main(args.config)
