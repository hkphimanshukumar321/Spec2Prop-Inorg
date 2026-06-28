"""
Spec2Prop-Edge: Static Demo Exporter
======================================
Precomputes real inference results on the held-out test set and saves them
as static JSON files for GitHub Pages deployment.
"""
import os
import sys
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.backend.sample_loader import SampleLoader
from app.backend.model_loader import ModelLoader
from app.backend.inference import run_inference
from app.backend.config import RAMAN_WN_MIN, RAMAN_WN_MAX

def main():
    print("Initializing components...")
    sample_loader = SampleLoader()
    model_loader = ModelLoader()

    if not sample_loader.load():
        print("Failed to load dataset.")
        return
    if not model_loader.load():
        print("Failed to load model.")
        return

    out_dir = os.path.join(PROJECT_ROOT, "app", "frontend", "public", "demo_samples")
    os.makedirs(out_dir, exist_ok=True)
    
    samples = sample_loader.get_sample_list(limit=50)  # Export 50 samples for demo
    print(f"Exporting {len(samples)} samples to {out_dir} ...")

    index_data = {"samples": samples, "total": len(samples)}
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index_data, f, indent=2)

    for i, s in enumerate(samples):
        sid = s["sample_id"]
        print(f"Processing [{i+1}/{len(samples)}] {sid}...")
        
        # Get true labels from sample
        detail = sample_loader.get_sample_detail(sid)
        if not detail:
            continue
            
        # Run real inference
        res = run_inference(sid, sample_loader, model_loader)
        if not res or "error" in res:
            print(f"  Inference failed for {sid}")
            continue
            
        # Get processed Raman
        raman_vec = sample_loader.get_raman_vector(sid)
        x_axis = np.linspace(RAMAN_WN_MIN, RAMAN_WN_MAX, len(raman_vec)).tolist()
        
        # Subsample to keep file sizes manageable for GitHub Pages
        step = max(1, len(x_axis) // 500)
        x_sub = x_axis[::step]
        y_sub = raman_vec.tolist()[::step]

        out_json = {
            "sample_id": sid,
            "rruff_id": detail["rruff_id"],
            "mineral_name": detail["mineral_name"],
            "original_formula": detail["formula"],
            "original_12class_label": detail["original_12class_label"],
            "true_9class_label": detail["true_9class_label"],
            "true_5class_label": detail["true_5class_label"],
            "processed_raman_x": [round(x, 1) for x in x_sub],
            "processed_raman_y": [round(y, 4) for y in y_sub],
            "predicted_5class_label": res["predicted_5class_label"],
            "predicted_5class_confidence": res["predicted_5class_confidence"],
            "predicted_9class_label": res["predicted_9class_label"],
            "predicted_9class_confidence": res["predicted_9class_confidence"],
            "top3_9class": res["top3_9class"],
            "inference_time_ms": res["inference_time_ms"],
            "is_correct_5class": res["is_correct_5class"],
            "is_correct_9class": res["is_correct_9class"],
            "recommendation": res["recommendation"],
            "disclaimer": res["disclaimer"],
            "model_name": res["model_name"]
        }

        with open(os.path.join(out_dir, f"{sid}.json"), "w") as f:
            json.dump(out_json, f, indent=2)

    print("Done exporting static demo files.")

if __name__ == "__main__":
    main()
