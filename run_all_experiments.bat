@echo off
REM ======================================================================
REM  Spec2Prop-InorgBench: Full Pipeline (Windows)
REM  Training -> Evaluation -> Deployment -> Validation
REM ======================================================================
REM Usage:
REM   run_all_experiments.bat                  -- run everything
REM   run_all_experiments.bat --skip-training  -- skip training, run deploy only
REM ======================================================================

setlocal enabledelayedexpansion

set SKIP_TRAINING=false
for %%a in (%*) do (
    if "%%a"=="--skip-training" set SKIP_TRAINING=true
)

echo ======================================================================
echo  Spec2Prop-InorgBench: Full Pipeline
echo  %date% %time%
echo ======================================================================

REM ------------------------------------------------------------------
REM PHASE 1: TRAINING
REM ------------------------------------------------------------------
if "%SKIP_TRAINING%"=="true" (
    echo.
    echo [SKIPPED] Training phase --skip-training flag set
    goto :phase2
)

echo.
echo === PHASE 1: TRAINING ===
echo.

echo --- 1a. Chemical Family Classification ---
python scripts/train_family_cnn.py --config configs/family_classification.yaml
if errorlevel 1 goto :error

echo.
echo --- 1b. Property Prediction ---
python scripts/train_property_cnn.py --config configs/property_prediction.yaml
if errorlevel 1 goto :error

echo.
echo --- 1c. Multimodal Raman + XRD ---
python scripts/train_multimodal_raman_xrd.py --config configs/multimodal_raman_xrd.yaml
if errorlevel 1 goto :error

echo.
echo --- 1d. Traditional Baselines ---
python scripts/train_baselines.py --config configs/baselines.yaml
if errorlevel 1 goto :error

echo.
echo --- 1e. Property Prediction Baselines ---
python scripts/train_property_baselines.py --config configs/baselines.yaml
if errorlevel 1 goto :error

:phase2
REM ------------------------------------------------------------------
REM PHASE 2: EVALUATION
REM ------------------------------------------------------------------
echo.
echo === PHASE 2: EVALUATION ===
echo.

echo --- 2a. Evaluating models ---
python scripts/evaluate_models.py
if errorlevel 1 goto :error

echo.
echo --- 2b. Plotting results ---
python scripts/plot_results.py
if errorlevel 1 goto :error

echo.
echo --- 2c. Summarizing experiments ---
python scripts/summarize_experiments.py
if errorlevel 1 goto :error

REM ------------------------------------------------------------------
REM PHASE 3: DEPLOYMENT EXPORT
REM ------------------------------------------------------------------
echo.
echo === PHASE 3: DEPLOYMENT EXPORT ===
echo.

echo --- 3a. Exporting LiteSpecNet ---
python deploy/export_models.py --checkpoint results/family_classification/LiteSpecNet_best.pt --model-class LiteSpecNet --label-encoder results/family_classification/chemical_family_model_encoder.json --task family --output-dir deploy/exported
if errorlevel 1 goto :error

echo.
echo --- 3b. Exporting SimpleCNN1D ---
python deploy/export_models.py --checkpoint results/family_classification/SimpleCNN1D_best.pt --model-class SimpleCNN1D --label-encoder results/family_classification/chemical_family_model_encoder.json --task family --output-dir deploy/exported
if errorlevel 1 goto :error

echo.
echo --- 3c. Exporting MultiScaleSpecNet ---
python deploy/export_models.py --checkpoint results/family_classification/MultiScaleSpecNet_best.pt --model-class MultiScaleSpecNet --label-encoder results/family_classification/chemical_family_model_encoder.json --task family --output-dir deploy/exported
if errorlevel 1 goto :error

REM ------------------------------------------------------------------
REM PHASE 4: INFERENCE
REM ------------------------------------------------------------------
echo.
echo === PHASE 4: DEPLOYMENT INFERENCE ===
echo.

echo --- 4a. Single-sample inference ---
python deploy/infer_single_sample.py --model deploy/exported/family_litespecnet_torchscript.pt --model-type torchscript --task family --raman-file deploy/sample_inputs/example_raman.csv --label-encoder deploy/exported/label_encoders.json --output deploy/outputs/prediction_report.json --saliency
if errorlevel 1 goto :error

echo.
echo --- 4b. Batch inference ---
python deploy/infer_batch.py --model deploy/exported/family_litespecnet_torchscript.pt --task family --input-dir deploy/sample_inputs/ --label-encoder deploy/exported/label_encoders.json --output-csv deploy/outputs/batch_predictions.csv
if errorlevel 1 goto :error

REM ------------------------------------------------------------------
REM PHASE 5: EDGE BENCHMARK
REM ------------------------------------------------------------------
echo.
echo === PHASE 5: EDGE BENCHMARK ===
echo.

python deploy/edge_benchmark.py --model deploy/exported/family_litespecnet_torchscript.pt --task family --sample deploy/sample_inputs/example_raman.csv --num-runs 500 --output deploy/outputs/edge_benchmark.json
if errorlevel 1 goto :error

REM ------------------------------------------------------------------
REM PHASE 6: INTEGRITY VALIDATION
REM ------------------------------------------------------------------
echo.
echo === PHASE 6: DEPLOYMENT INTEGRITY VALIDATION ===
echo.

python deploy/validate_deployment_integrity.py --model deploy/exported/family_litespecnet_torchscript.pt --label-encoder deploy/exported/label_encoders.json --sample deploy/sample_inputs/example_raman.csv --model-card deploy/exported/model_card.json
if errorlevel 1 goto :error

REM ------------------------------------------------------------------
REM PHASE 7: TESTS
REM ------------------------------------------------------------------
echo.
echo === PHASE 7: RUNNING TESTS ===
echo.

python -m pytest tests/test_deployment_pipeline.py tests/test_deployment_integrity.py -v --tb=short
if errorlevel 1 goto :error

REM ------------------------------------------------------------------
REM DONE
REM ------------------------------------------------------------------
echo.
echo ======================================================================
echo  ALL PHASES COMPLETED SUCCESSFULLY
echo  %date% %time%
echo ======================================================================
echo.
echo Output files:
echo   results/FINAL_EXPERIMENT_SUMMARY.md    - Experiment results tables
echo   results/figures/                        - Training curves + confusion matrices
echo   deploy/exported/                        - TorchScript + ONNX models
echo   deploy/outputs/prediction_report.json   - Inference report
echo   deploy/outputs/saliency_plot.png        - Saliency visualization
echo   deploy/outputs/edge_benchmark.json      - Edge benchmark results
echo   deploy/outputs/batch_predictions.csv    - Batch inference results
echo.
echo To launch Streamlit app:
echo   streamlit run deploy/app_streamlit.py
echo.
goto :eof

:error
echo.
echo ======================================================================
echo  PIPELINE FAILED - see error above
echo ======================================================================
exit /b 1
