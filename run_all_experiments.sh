#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "======================================================================"
echo " Starting Spec2Prop-InorgBench Experiments"
echo "======================================================================"

echo ""
echo "--- 1. Training Chemical Family Classification (Raman only) ---"
python scripts/train_family_cnn.py --config configs/family_classification.yaml

echo ""
echo "--- 2. Training Property Prediction (Raman + Descriptors) ---"
python scripts/train_property_cnn.py --config configs/property_prediction.yaml

echo ""
echo "--- 3. Training Multimodal Models (Raman + XRD) ---"
python scripts/train_multimodal_raman_xrd.py --config configs/multimodal_raman_xrd.yaml

echo ""
echo "--- 4. Training Baseline Models (RF, XGBoost) ---"
python scripts/train_baselines.py --config configs/baselines.yaml

echo ""
echo "--- 5. Evaluating Models ---"
# Assuming evaluation uses the same config or runs against everything by default
python scripts/evaluate_models.py

echo ""
echo "--- 6. Plotting Results ---"
python scripts/plot_results.py

echo ""
echo "--- 7. Summarizing Experiments ---"
python scripts/summarize_experiments.py

echo ""
echo "--- 8. Exporting Edge Models (TorchScript/ONNX) ---"
# This might require passing the best model path, running as default
python scripts/export_edge_models.py

echo ""
echo "======================================================================"
echo " All experiments completed successfully!"
echo "======================================================================"
