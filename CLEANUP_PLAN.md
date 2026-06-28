# Spec2Prop-InorgBench / Spec2Prop-Edge Cleanup Plan

## Objective
The objective of this cleanup was to perform a complete consistency check, cleanup, and fix pass across the whole project (dataset, learning pipeline, deployment pipeline, frontend demo, documentation, and final claims). The goal is to ensure all components align with the final verified 9-class/5-class hierarchy and strictly enforce scientific integrity.

## Execution Steps Taken

### 1. Unified Label Mapping
- **Issue**: Label hierarchies (e.g., merging rare classes) were hardcoded and duplicated across the preprocessing scripts, model training, deployment backend, and report generators, causing potential mismatches and maintenance issues.
- **Action**: Created `configs/label_mappings.yaml` as the single source of truth for the official 9-class fine hierarchy and the 5-class coarse hierarchy, including their display names.

### 2. Dataset Pipeline Alignment
- **Issue**: The dataset pipeline (`models/dataset.py`) dynamically pruned rare families based on batch sizes and frequencies, overriding the official 9-class subset.
- **Action**: Removed dynamic thresholding (`MIN_FAMILY_SIZE`, `RARE_FAMILIES`) and updated `add_model_family_column` to strictly pull the `original_to_fine_9` mapping from `label_mappings.yaml`.

### 3. Model Training Alignment
- **Issue**: Training scripts generated their label encoders on the fly based on the data they ingested, creating brittleness if some rare classes were missing in a specific split.
- **Action**: Modified `scripts/coarse_family_classifier.py` and `scripts/train_deployment_model.py` to construct their label encoders strictly from the `fine_9_labels` list in `label_mappings.yaml`.

### 4. Deployment Pipeline Updates
- **Issue**: The prediction report generator outputted raw labels without explicitly mapping to the required 5-class coarse prediction. The integrity checker lacked enforcement of "decision-support" wording.
- **Action**:
  - Updated `deploy/generate_prediction_report.py` to ingest the `label_mappings.yaml` and add `predicted_coarse_family` to all output reports (JSON, CSV, MD) when evaluating the family task.
  - Added specific checks in `deploy/validate_deployment_integrity.py` to ensure the term `decision-support` is in the model card, and scanned the `README.md` and `README_DEPLOYMENT.md` to guarantee the absence of "final identification", "replaces scientist", "perfect classification", and "fully automated".

### 5. Frontend UI / Backend API Alignment
- **Issue**: The FastAPI backend (`app/backend/inference.py`) used hardcoded `MAP_12_TO_5` and `MAP_9_TO_5` dicts. The frontend needed to display properly formatted 5-class broad families.
- **Action**: Replaced the backend's hardcoded mappings with the central `configs/label_mappings.yaml`, ensuring that the `run_inference` function strictly outputs the standard `predicted_5class_label` and `predicted_9class_label`. The frontend now dynamically receives the standardized names without any client-side hardcoding.

### 6. Documentation and Claims Verification
- **Issue**: The READMEs contained unverified claims about edge deployment and lacked strict descriptions for auxiliary datasets (COD, MLROD, tmQM).
- **Action**:
  - **Removed** unsupported claims regarding Raspberry Pi / Jetson deployment from `deploy/README_DEPLOYMENT.md`.
  - **Clarified** in `data/processed/README_Spec2Prop_Inorg.md` that COD was deferred as it does not provide paired Raman spectra, MLROD is an auxiliary resource, and tmQM/tmQMg are optional, non-core datasets.

## Status
**Completed.** All system components are now tightly aligned to a single configuration file, preventing dataset drift, label mismatches, and overclaiming of scientific capabilities.
