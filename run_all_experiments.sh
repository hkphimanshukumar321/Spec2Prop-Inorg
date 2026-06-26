#!/bin/bash
# ======================================================================
#  Spec2Prop-InorgBench: Full Pipeline
#  Training → Evaluation → Deployment → Validation
# ======================================================================
# Usage:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh           # run everything
#   ./run_all_experiments.sh --skip-training   # skip training, run deploy only
# ======================================================================

set -e

SKIP_TRAINING=false
for arg in "$@"; do
  case $arg in
    --skip-training) SKIP_TRAINING=true ;;
  esac
done

echo "======================================================================"
echo " Spec2Prop-InorgBench: Full Pipeline"
echo " $(date)"
echo "======================================================================"

# ------------------------------------------------------------------
# PHASE 1: TRAINING
# ------------------------------------------------------------------
if [ "$SKIP_TRAINING" = false ]; then

echo ""
echo "=== PHASE 1: TRAINING ==="
echo ""

echo "--- 1a. Chemical Family Classification (CNN + MultiScaleSpecNet) ---"
python scripts/train_family_cnn.py --config configs/family_classification.yaml

echo ""
echo "--- 1b. Property Prediction (CNN + Fusion + Descriptor MLP) ---"
python scripts/train_property_cnn.py --config configs/property_prediction.yaml

echo ""
echo "--- 1c. Multimodal Raman + XRD ---"
python scripts/train_multimodal_raman_xrd.py --config configs/multimodal_raman_xrd.yaml

echo ""
echo "--- 1d. Traditional Baselines (SVM, RF, kNN, XGBoost) ---"
python scripts/train_baselines.py --config configs/baselines.yaml

echo ""
echo "--- 1e. Property Prediction Baselines (XGBoost, RF on descriptors) ---"
python scripts/train_property_baselines.py --config configs/baselines.yaml

else
  echo ""
  echo "[SKIPPED] Training phase (--skip-training flag set)"
fi

# ------------------------------------------------------------------
# PHASE 2: EVALUATION & REPORTING
# ------------------------------------------------------------------
echo ""
echo "=== PHASE 2: EVALUATION & REPORTING ==="
echo ""

echo "--- 2a. Evaluating all trained models on test set ---"
python scripts/evaluate_models.py

echo ""
echo "--- 2b. Plotting results (training curves, confusion matrices) ---"
python scripts/plot_results.py

echo ""
echo "--- 2c. Summarizing experiments (Markdown tables) ---"
python scripts/summarize_experiments.py

# ------------------------------------------------------------------
# PHASE 3: DEPLOYMENT EXPORT
# ------------------------------------------------------------------
echo ""
echo "=== PHASE 3: DEPLOYMENT EXPORT ==="
echo ""

echo "--- 3a. Exporting LiteSpecNet (Family Classification) ---"
python deploy/export_models.py \
  --checkpoint results/family_classification/LiteSpecNet_best.pt \
  --model-class LiteSpecNet \
  --label-encoder results/family_classification/chemical_family_model_encoder.json \
  --task family \
  --output-dir deploy/exported

echo ""
echo "--- 3b. Exporting SimpleCNN1D (Family Classification) ---"
python deploy/export_models.py \
  --checkpoint results/family_classification/SimpleCNN1D_best.pt \
  --model-class SimpleCNN1D \
  --label-encoder results/family_classification/chemical_family_model_encoder.json \
  --task family \
  --output-dir deploy/exported

echo ""
echo "--- 3c. Exporting MultiScaleSpecNet (Family Classification) ---"
python deploy/export_models.py \
  --checkpoint results/family_classification/MultiScaleSpecNet_best.pt \
  --model-class MultiScaleSpecNet \
  --label-encoder results/family_classification/chemical_family_model_encoder.json \
  --task family \
  --output-dir deploy/exported

# ------------------------------------------------------------------
# PHASE 4: DEPLOYMENT INFERENCE
# ------------------------------------------------------------------
echo ""
echo "=== PHASE 4: DEPLOYMENT INFERENCE ==="
echo ""

echo "--- 4a. Single-sample inference (with saliency) ---"
python deploy/infer_single_sample.py \
  --model deploy/exported/family_litespecnet_torchscript.pt \
  --model-type torchscript \
  --task family \
  --raman-file deploy/sample_inputs/example_raman.csv \
  --label-encoder deploy/exported/label_encoders.json \
  --output deploy/outputs/prediction_report.json \
  --saliency

echo ""
echo "--- 4b. Batch inference ---"
python deploy/infer_batch.py \
  --model deploy/exported/family_litespecnet_torchscript.pt \
  --task family \
  --input-dir deploy/sample_inputs/ \
  --label-encoder deploy/exported/label_encoders.json \
  --output-csv deploy/outputs/batch_predictions.csv

# ------------------------------------------------------------------
# PHASE 5: EDGE BENCHMARK
# ------------------------------------------------------------------
echo ""
echo "=== PHASE 5: EDGE BENCHMARK ==="
echo ""

echo "--- 5a. Benchmarking LiteSpecNet (500 runs) ---"
python deploy/edge_benchmark.py \
  --model deploy/exported/family_litespecnet_torchscript.pt \
  --task family \
  --sample deploy/sample_inputs/example_raman.csv \
  --num-runs 500 \
  --output deploy/outputs/edge_benchmark.json

# ------------------------------------------------------------------
# PHASE 6: DEPLOYMENT INTEGRITY VALIDATION
# ------------------------------------------------------------------
echo ""
echo "=== PHASE 6: DEPLOYMENT INTEGRITY VALIDATION ==="
echo ""

python deploy/validate_deployment_integrity.py \
  --model deploy/exported/family_litespecnet_torchscript.pt \
  --label-encoder deploy/exported/label_encoders.json \
  --sample deploy/sample_inputs/example_raman.csv \
  --model-card deploy/exported/model_card.json

# ------------------------------------------------------------------
# PHASE 7: TESTS
# ------------------------------------------------------------------
echo ""
echo "=== PHASE 7: RUNNING TESTS ==="
echo ""

python -m pytest tests/test_deployment_pipeline.py tests/test_deployment_integrity.py -v --tb=short

# ------------------------------------------------------------------
# DONE
# ------------------------------------------------------------------
echo ""
echo "======================================================================"
echo " ALL PHASES COMPLETED SUCCESSFULLY"
echo " $(date)"
echo "======================================================================"
echo ""
echo "Output files:"
echo "  results/FINAL_EXPERIMENT_SUMMARY.md    - Experiment results tables"
echo "  results/figures/                        - Training curves & confusion matrices"
echo "  deploy/exported/                        - TorchScript & ONNX models"
echo "  deploy/outputs/prediction_report.json   - Sample inference report"
echo "  deploy/outputs/prediction_report.md     - Markdown inference report"
echo "  deploy/outputs/saliency_plot.png        - Gradient saliency visualization"
echo "  deploy/outputs/edge_benchmark.json      - Edge benchmark results"
echo "  deploy/outputs/batch_predictions.csv    - Batch inference results"
echo ""
echo "To launch Streamlit app:"
echo "  streamlit run deploy/app_streamlit.py"
echo ""
