import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Spec2Prop-Inorg: Train Traditional Baselines
============================================
Trains SVM, Random Forest, Logistic Regression on Raman spectra (raw and PCA).
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from models import Spec2PropDataset
from models.baselines import BaselineModel
from models.dataset import build_label_encoder

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    out_dir = cfg["output"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    
    meta_df = pd.read_pickle(cfg["data"]["metadata_pkl"])
    target_col = cfg["data"]["target_col"]
    from models.dataset import add_model_family_column
    meta_df = add_model_family_column(meta_df)
    
    encoder = build_label_encoder(meta_df[target_col])
    
    kwargs = dict(
        metadata_pkl=cfg["data"]["metadata_pkl"],
        raman_parquet=cfg["data"]["raman_parquet"],
        target_cols=[target_col],
        label_encoders={target_col: encoder},
        cache_dir=cfg["data"]["cache_dir"],
    )
    
    train_dataset = Spec2PropDataset(split_csv=os.path.join(cfg["data"]["splits_dir"], "train.csv"), **kwargs)
    test_dataset = Spec2PropDataset(split_csv=os.path.join(cfg["data"]["splits_dir"], "test.csv"), **kwargs)
    
    def get_arrays(ds):
        X, y = [], []
        for i in range(len(ds)):
            b = ds[i]
            if b[f"mask_{target_col}"].item() > 0:
                X.append(b["raman"].numpy().flatten())
                y.append(b[f"target_{target_col}"].item())
        return np.array(X), np.array(y)
        
    print("Extracting train arrays...")
    X_train, y_train = get_arrays(train_dataset)
    print("Extracting test arrays...")
    X_test, y_test = get_arrays(test_dataset)
    
    results = []
    
    for model_name in cfg["baselines"]["models"]:
        for pca_dim in cfg["baselines"]["pca_dims"]:
            print(f"\nTraining {model_name} (PCA: {pca_dim})")
            try:
                model = BaselineModel(model_name, pca_dim=pca_dim)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average="macro")
                
                print(f"  -> Acc: {acc:.4f} | F1: {f1:.4f}")
                results.append({
                    "Model": model_name,
                    "PCA_Dim": pca_dim if pca_dim else "Raw",
                    "Accuracy": acc,
                    "Macro_F1": f1
                })
            except Exception as e:
                print(f"  -> Failed: {e}")
                
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(out_dir, "baseline_comparison.csv"), index=False)
    print("\nSaved baseline results to:", os.path.join(out_dir, "baseline_comparison.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baselines.yaml")
    args = parser.parse_args()
    main(args.config)
