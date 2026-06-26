# Spec2Prop-Edge: Deployment System

**Edge-oriented deployment prototype for Spec2Prop-InorgBench models.**

> Spec2Prop-Edge provides an edge-oriented deployment prototype. Unless benchmarked on the target hardware, results should be reported as CPU/desktop inference benchmarks rather than real edge deployment.

## What is Spec2Prop-Edge?

Spec2Prop-Edge is the deployment module for the Spec2Prop-InorgBench project. It takes trained model checkpoints and provides:

- **Runtime preprocessing** matching the training pipeline
- **Single-sample and batch inference** via CLI
- **TorchScript/ONNX model export** for portable deployment
- **Edge benchmarking** with honest device reporting
- **Deployment integrity validation** ensuring consistency with training
- **Streamlit web UI** for interactive use
- **Gradient saliency visualization** for model interpretability

## Supported Tasks

| Task | Input | Model | Output |
|------|-------|-------|--------|
| Family Classification | Raman CSV | LiteSpecNet / SimpleCNN1D | Chemical family (e.g., Oxide, Sulfide) |
| Property Prediction | Raman CSV + Descriptors | FusionSpec2PropNet | band_gap_class, is_metal, formation_energy_class |
| Multimodal | Raman CSV + XRD CSV | DualBranchRamanXRDNet | Chemical family / properties |

## Supported Devices

- ✅ **Laptop/Workstation CPU** — tested and benchmarked
- 🔲 **Raspberry Pi 5-class CPU** — architecture supports it, not yet benchmarked
- 🔲 **NVIDIA Jetson Nano/Orin Nano** — architecture supports it, not yet benchmarked

## Input Formats

### Raman CSV/TXT
Two-column file: `wavenumber_cm-1, intensity`
```
wavenumber_cm-1,intensity
100.00,0.012345
107.82,0.015678
...
```

### XRD CSV/TXT
Two-column file: `two_theta_deg, intensity`
```
two_theta_deg,intensity
5.000,0.001234
5.188,0.002345
...
```

## Quick Start

### 1. Export a Trained Model

```bash
python deploy/export_models.py \
  --checkpoint results/family_classification/LiteSpecNet_best.pt \
  --model-class LiteSpecNet \
  --label-encoder results/family_classification/chemical_family_model_encoder.json \
  --task family \
  --output-dir deploy/exported
```

### 2. Single-Sample Inference

```bash
python deploy/infer_single_sample.py \
  --model deploy/exported/family_litespecnet_torchscript.pt \
  --model-type torchscript \
  --task family \
  --raman-file deploy/sample_inputs/example_raman.csv \
  --label-encoder deploy/exported/label_encoders.json \
  --output deploy/outputs/prediction_report.json \
  --saliency
```

### 3. Batch Inference

```bash
python deploy/infer_batch.py \
  --model deploy/exported/family_litespecnet_torchscript.pt \
  --task family \
  --input-dir deploy/sample_inputs/ \
  --label-encoder deploy/exported/label_encoders.json \
  --output-csv deploy/outputs/batch_predictions.csv
```

### 4. Edge Benchmark

```bash
python deploy/edge_benchmark.py \
  --model deploy/exported/family_litespecnet_torchscript.pt \
  --task family \
  --sample deploy/sample_inputs/example_raman.csv \
  --num-runs 500 \
  --output deploy/outputs/edge_benchmark.json
```

### 5. Deployment Integrity Validation

```bash
python deploy/validate_deployment_integrity.py \
  --model deploy/exported/family_litespecnet_torchscript.pt \
  --label-encoder deploy/exported/label_encoders.json \
  --sample deploy/sample_inputs/example_raman.csv
```

### 6. Streamlit App

```bash
streamlit run deploy/app_streamlit.py
```

### 7. Run Tests

```bash
python -m pytest tests/test_deployment_pipeline.py -v
python -m pytest tests/test_deployment_integrity.py -v
```

## Output Report Fields

Each prediction report contains:

| Field | Description |
|-------|-------------|
| `sample_id` | Input filename |
| `task` | family / property / multimodal |
| `model_name` | Model file used |
| `top_predictions` | Top-K classes with confidence scores |
| `inference_time_ms` | Inference latency in milliseconds |
| `device` | Device used for inference |
| `preprocessing` | Preprocessing parameters used |
| `warnings` | Any data quality warnings |
| `disclaimer` | Screening-level prediction notice |
| `timestamp` | When the prediction was made |

## Limitations and Disclaimers

1. **Screening-level only**: Predictions are screening-level and do NOT constitute experimental confirmation.
2. **Instrument sensitivity**: Performance may degrade on spectra from instruments different from the training data (RRUFF database).
3. **Edge hardware**: Unless actually benchmarked on Raspberry Pi or Jetson, results are "edge-oriented CPU inference benchmarks" only.
4. **Saliency**: Highlighted spectral regions are "model-attributed" — they show what the model uses, not necessarily physically meaningful peaks.
5. **No diagnostic use**: This system is NOT validated for diagnostic or safety-critical applications.

## File Structure

```
deploy/
├── __init__.py
├── preprocess_runtime.py        # Runtime preprocessing
├── export_models.py             # TorchScript/ONNX export
├── infer_single_sample.py       # Single-sample CLI
├── infer_batch.py               # Batch inference CLI
├── edge_benchmark.py            # Latency/throughput benchmark
├── validate_deployment_integrity.py  # Integrity validation
├── generate_prediction_report.py     # Report generation
├── saliency_runtime.py          # Gradient saliency
├── app_streamlit.py             # Streamlit web UI
├── sample_inputs/
│   ├── example_raman.csv
│   ├── example_xrd.csv
│   └── example_descriptors.json
├── exported/                    # Exported models go here
├── outputs/                     # Inference outputs go here
└── README_DEPLOYMENT.md         # This file
```
