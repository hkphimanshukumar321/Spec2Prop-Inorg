"""
Spec2Prop-Edge: Inference Engine
==================================
Runs real model inference on real test samples.
Includes both 9-class fine predictions and 5-class broad grouping.
"""
import time
import numpy as np
from typing import Optional, Dict, List

from app.backend.feature_extractor import extract_features, extract_features_fallback
from app.backend.model_loader import ModelLoader
from app.backend.sample_loader import SampleLoader
import yaml
import os

# Mappings
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "label_mappings.yaml"))
with open(config_path, "r") as f:
    mappings = yaml.safe_load(f)

MAP_12_TO_9 = mappings.get("original_to_fine_9", {})
MAP_12_TO_5 = mappings.get("original_to_coarse_5", {})
MAP_9_TO_5 = mappings.get("fine_9_to_coarse_5", {})



def run_inference(
    sample_id: str,
    sample_loader: SampleLoader,
    model_loader: ModelLoader,
    task: str = "family",
    modality: str = "raman",
) -> Optional[Dict]:
    """
    Run real model inference on a real test sample.
    """
    if not sample_loader.validate_is_test_sample(sample_id):
        return {"error": f"Sample {sample_id} is not in the test split. Only test samples are allowed."}

    raman_vector = sample_loader.get_raman_vector(sample_id)
    if raman_vector is None:
        return {"error": f"Raman vector not found for sample {sample_id}"}

    true_12class = sample_loader.get_true_label(sample_id)
    if true_12class is None:
        true_12class = "Unknown"
        
    true_9class = MAP_12_TO_9.get(true_12class, 'Other/Rare')
    true_5class = MAP_12_TO_5.get(true_12class, 'Other/Rare')

    t_start = time.perf_counter()

    if model_loader.has_full_pipeline:
        features = extract_features(
            raman_vector,
            pca=model_loader.pca,
            scaler=model_loader.scaler,
            prototype_extractor=model_loader.prototype_extractor,
        )
    else:
        features = extract_features_fallback(
            raman_vector,
            pca=model_loader.pca,
        )

    # Output from LightGBM trained on 9 classes
    probabilities = model_loader.model.predict_proba(features)  # (1, num_classes)
    t_end = time.perf_counter()
    inference_time_ms = (t_end - t_start) * 1000.0

    proba = probabilities[0]
    
    # 9-class predictions
    predicted_idx = int(np.argmax(proba))
    confidence_9class = float(proba[predicted_idx])
    predicted_9class_label = model_loader.inv_encoder.get(predicted_idx, f"Class_{predicted_idx}")

    # Top-K 9-class
    top_k_indices = np.argsort(proba)[::-1]
    top3_9class = []
    for idx in top_k_indices[:3]:
        label = model_loader.inv_encoder.get(int(idx), f"Class_{idx}")
        prob = float(proba[idx])
        if prob > 0.001:
            top3_9class.append({"label": label, "probability": round(prob, 4)})

    # 5-class broad grouping by summing probabilities
    prob_5class = {}
    for idx, p in enumerate(proba):
        label_9 = model_loader.inv_encoder.get(idx, f"Class_{idx}")
        label_5 = MAP_9_TO_5.get(label_9, "Other/Rare")
        prob_5class[label_5] = prob_5class.get(label_5, 0.0) + float(p)
        
    # Get highest 5-class prob
    predicted_5class_label = max(prob_5class.items(), key=lambda x: x[1])[0]
    predicted_5class_confidence = prob_5class[predicted_5class_label]

    # Quality based on 5-class broad group confidence
    if predicted_5class_confidence >= 0.70:
        quality = "High"
    elif predicted_5class_confidence >= 0.50:
        quality = "Medium"
    else:
        quality = "Low"

    # Recommendation
    groups_members = {
        'Oxyanion Group': 'sulfate, phosphate, carbonate, and borate',
        'Sulfide/Halide Group': 'sulfide and halide',
        'Silicate': 'silicate',
        'Oxide': 'oxide',
        'Other/Rare': 'rare classes (elemental, intermetallic, etc)'
    }
    
    recommendation = f"This sample likely belongs to the {predicted_5class_label}. Verify against {groups_members.get(predicted_5class_label, 'reference')} reference spectra."

    return {
        "sample_id": sample_id,
        "task": task,
        "modality": modality,
        "model_name": model_loader.model_name,
        
        "predicted_5class_label": predicted_5class_label,
        "predicted_5class_confidence": round(predicted_5class_confidence, 4),
        "is_correct_5class": (predicted_5class_label == true_5class),
        
        "predicted_9class_label": predicted_9class_label,
        "predicted_9class_confidence": round(confidence_9class, 4),
        "is_correct_9class": (predicted_9class_label == true_9class),
        "top3_9class": top3_9class,
        
        "prediction_quality": quality,
        "inference_time_ms": round(inference_time_ms, 2),
        "feature_dim": int(features.shape[1]),
        "recommendation": recommendation,
        "disclaimer": "Screening-level prediction, not final experimental confirmation.",
    }

def run_custom_inference(
    raman_vector: np.ndarray,
    model_loader: ModelLoader,
    filename: str = "custom_upload"
) -> Optional[Dict]:
    """
    Run model inference on a custom uploaded Raman spectrum.
    Assumes raman_vector is already interpolated to the 2048-point standard grid.
    """
    t_start = time.perf_counter()

    if model_loader.has_full_pipeline:
        features = extract_features(
            raman_vector,
            pca=model_loader.pca,
            scaler=model_loader.scaler,
            prototype_extractor=model_loader.prototype_extractor,
        )
    else:
        features = extract_features_fallback(
            raman_vector,
            pca=model_loader.pca,
        )

    probabilities = model_loader.model.predict_proba(features)
    t_end = time.perf_counter()
    inference_time_ms = (t_end - t_start) * 1000.0

    proba = probabilities[0]
    
    # 9-class predictions
    predicted_idx = int(np.argmax(proba))
    confidence_9class = float(proba[predicted_idx])
    predicted_9class_label = model_loader.inv_encoder.get(predicted_idx, f"Class_{predicted_idx}")

    # Top-K 9-class
    top_k_indices = np.argsort(proba)[::-1]
    top3_9class = []
    for idx in top_k_indices[:3]:
        label = model_loader.inv_encoder.get(int(idx), f"Class_{idx}")
        prob = float(proba[idx])
        if prob > 0.001:
            top3_9class.append({"label": label, "probability": round(prob, 4)})

    # 5-class broad grouping by summing probabilities
    prob_5class = {}
    for idx, p in enumerate(proba):
        label_9 = model_loader.inv_encoder.get(idx, f"Class_{idx}")
        label_5 = MAP_9_TO_5.get(label_9, "Other/Rare")
        prob_5class[label_5] = prob_5class.get(label_5, 0.0) + float(p)
        
    predicted_5class_label = max(prob_5class.items(), key=lambda x: x[1])[0]
    predicted_5class_confidence = prob_5class[predicted_5class_label]

    if predicted_5class_confidence >= 0.70:
        quality = "High"
    elif predicted_5class_confidence >= 0.50:
        quality = "Medium"
    else:
        quality = "Low"

    groups_members = {
        'Oxyanion Group': 'sulfate, phosphate, carbonate, and borate',
        'Sulfide/Halide Group': 'sulfide and halide',
        'Silicate': 'silicate',
        'Oxide': 'oxide',
        'Other/Rare': 'rare classes (elemental, intermetallic, etc)'
    }
    recommendation = f"This sample likely belongs to the {predicted_5class_label}. Verify against {groups_members.get(predicted_5class_label, 'reference')} reference spectra."

    return {
        "sample_id": filename,
        "task": "family",
        "modality": "raman",
        "model_name": model_loader.model_name,
        
        "predicted_5class_label": predicted_5class_label,
        "predicted_5class_confidence": round(predicted_5class_confidence, 4),
        "is_correct_5class": None,
        
        "predicted_9class_label": predicted_9class_label,
        "predicted_9class_confidence": round(confidence_9class, 4),
        "is_correct_9class": None,
        "top3_9class": top3_9class,
        
        "prediction_quality": quality,
        "inference_time_ms": round(inference_time_ms, 2),
        "feature_dim": int(features.shape[1]),
        "recommendation": recommendation,
        "disclaimer": "Screening-level prediction, not final experimental confirmation.",
    }
