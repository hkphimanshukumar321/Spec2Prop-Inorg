@echo off
echo ======================================================================
echo  Starting Spec2Prop-InorgBench Experiments
echo ======================================================================

echo.
echo --- 1. Training Chemical Family Classification (Raman only) ---
python scripts\train_family_cnn.py --config configs\family_classification.yaml
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo --- 2. Training Property Prediction (Raman + Descriptors) ---
python scripts\train_property_cnn.py --config configs\property_prediction.yaml
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo --- 3. Training Multimodal Models (Raman + XRD) ---
python scripts\train_multimodal_raman_xrd.py --config configs\multimodal_raman_xrd.yaml
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo --- 4. Training Baseline Models (RF, XGBoost) ---
python scripts\train_baselines.py --config configs\baselines.yaml
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo --- 5. Evaluating Models ---
python scripts\evaluate_models.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo --- 6. Plotting Results ---
python scripts\plot_results.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo --- 7. Summarizing Experiments ---
python scripts\summarize_experiments.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo --- 8. Exporting Edge Models (TorchScript/ONNX) ---
python scripts\export_edge_models.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ======================================================================
echo  All experiments completed successfully!
echo ======================================================================
