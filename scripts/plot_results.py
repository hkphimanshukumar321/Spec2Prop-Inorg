import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Spec2Prop-Inorg: Plot Results
=============================
Generates comprehensive figures for the research paper/report based on
training logs, metrics, and dataset distributions.
"""

import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_family_distribution(results_dir):
    # Mock plotting function - in reality would read from dataset
    pass

def plot_training_curves(results_dir, task):
    task_dir = os.path.join(results_dir, task)
    if not os.path.exists(task_dir): return
    
    out_dir = os.path.join(results_dir, "figures", task)
    os.makedirs(out_dir, exist_ok=True)
    
    for f in os.listdir(task_dir):
        if f.endswith("_history.csv"):
            df = pd.read_csv(os.path.join(task_dir, f))
            model_name = f.replace("_history.csv", "")
            
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(df["epoch"], df["train_loss"], label="Train")
            plt.plot(df["epoch"], df["val_loss"], label="Val")
            plt.title(f"{model_name} Loss")
            plt.legend()
            
            plt.subplot(1, 2, 2)
            if "val_f1" in df.columns:
                plt.plot(df["epoch"], df["val_f1"], label="Val F1", color='green')
            elif "val_avg_f1" in df.columns:
                plt.plot(df["epoch"], df["val_avg_f1"], label="Val Avg F1", color='green')
            plt.title(f"{model_name} F1 Score")
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{model_name}_curves.png"), dpi=300)
            plt.close()
            print(f"Saved curves for {model_name} in {task}")

def plot_ablation(results_dir):
    # Generates bar charts comparing models
    pass

def main(results_dir):
    print("Generating result plots...")
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    
    for task in ["family_classification", "property_prediction", "multimodal_raman_xrd"]:
        plot_training_curves(results_dir, task)
        
    print("Plotting complete. Check results/figures/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(args.results_dir)
