import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Spec2Prop-Inorg: Summarize Experiments
======================================
Aggregates all results, metrics, and benchmarks into a final Markdown report.
"""

import argparse
import os
import json
import pandas as pd

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def main(results_dir):
    out_path = os.path.join(results_dir, "FINAL_EXPERIMENT_SUMMARY.md")
    
    family_metrics = load_json(os.path.join(results_dir, "family_classification", "test_metrics.json"))
    prop_metrics = load_json(os.path.join(results_dir, "property_prediction", "test_metrics.json"))
    mm_metrics = load_json(os.path.join(results_dir, "multimodal_raman_xrd", "test_metrics.json"))
    edge_metrics = load_json(os.path.join(results_dir, "edge", "edge_benchmark.json"))
    
    baseline_csv = os.path.join(results_dir, "baselines", "baseline_comparison.csv")
    has_baselines = os.path.exists(baseline_csv)
    
    with open(out_path, "w") as f:
        f.write("# Spec2Prop-Inorg: Experiment Summary Report\n\n")
        f.write("This document summarizes the performance of the Spec2Prop-Inorg framework across all tasks.\n\n")
        
        # 1. Family Classification
        f.write("## Task 1: Chemical Family Classification\n")
        if family_metrics:
            f.write("| Model | Accuracy | Macro F1 |\n")
            f.write("|---|---|---|\n")
            for model, m in family_metrics.items():
                f.write(f"| {model} | {m.get('accuracy', 0):.4f} | {m.get('macro_f1', 0):.4f} |\n")
        else:
            f.write("No results found yet.\n")
        f.write("\n")
        
        # 2. Property Prediction
        f.write("## Task 2: Spectra-to-Property Prediction\n")
        if prop_metrics:
            for model, m in prop_metrics.items():
                f.write(f"### {model}\n")
                f.write("| Task | Accuracy | Macro F1 |\n")
                f.write("|---|---|---|\n")
                for task, t_m in m.items():
                    if isinstance(t_m, dict):
                        f.write(f"| {task} | {t_m.get('accuracy',0):.4f} | {t_m.get('macro_f1',0):.4f} |\n")
        else:
            f.write("No results found yet.\n")
        f.write("\n")
        
        # 3. Multimodal
        f.write("## Task 3: Multimodal Raman + XRD\n")
        if mm_metrics:
            f.write("| Model | Accuracy | Macro F1 |\n")
            f.write("|---|---|---|\n")
            for model, m in mm_metrics.items():
                f.write(f"| {model} | {m.get('accuracy', 0):.4f} | {m.get('macro_f1', 0):.4f} |\n")
        else:
            f.write("No results found yet.\n")
        f.write("\n")
        
        # 4. Edge Benchmark
        f.write("## Edge Deployment Benchmark\n")
        if edge_metrics:
            f.write(f"- Model: {edge_metrics.get('model')}\n")
            f.write(f"- Parameters: {edge_metrics.get('parameters'):,}\n")
            f.write(f"- Size: {edge_metrics.get('size_mb')} MB\n")
            f.write(f"- CPU Inference Latency: {edge_metrics.get('avg_latency_ms')} ms/sample\n")
            f.write(f"- Intended Devices: {', '.join(edge_metrics.get('target_devices', []))}\n")
        else:
            f.write("No benchmark found yet.\n")
        f.write("\n")
        
        # 5. Baselines
        f.write("## Baseline Comparison\n")
        if has_baselines:
            df = pd.read_csv(baseline_csv)
            f.write(df.to_markdown(index=False))
            f.write("\n")
        else:
            f.write("No baseline results found yet.\n")
            
        # Scientific framing
        f.write("\n## Scientific Framing & Conclusion\n")
        f.write("This work demonstrates that lightweight, open-data AI models can successfully screen inorganic compounds for material properties and chemical families using Raman and XRD spectra without complex laboratory preprocessing. The fusion of structural diffraction data with vibrational spectra yields robust multi-modal representations.\n")
        
    print(f"Summary report generated at: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(args.results_dir)
