"""
Spec2Prop-Edge: Backend Configuration
=======================================
All paths are relative to the project root (FICS/).
"""
import os
from pathlib import Path

# Project root: go up from app/backend/ to FICS/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ─── Data paths ───
DATA_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_PKL = DATA_DIR / "spec2prop_clean_inorganic.pkl"
FULL_SPECTRA_PKL = DATA_DIR / "spec2prop_full_spectra.pkl"
RAMAN_PARQUET = DATA_DIR / "spec2prop_raman.parquet"
XRD_PARQUET = DATA_DIR / "spec2prop_xrd.parquet"
XRD_LINKED_PKL = DATA_DIR / "spec2prop_xrd_linked_inorganic.pkl"
SPLITS_DIR = DATA_DIR / "splits" / "clean_inorganic"
CACHE_DIR = DATA_DIR / "cache" / "clean_inorganic"

# ─── Model artifacts ───
MODEL_DIR = PROJECT_ROOT / "results" / "hybrid" / "deployment"
LGBM_MODEL_PATH = MODEL_DIR / "lgbm_deployment.joblib"
PCA_PATH = MODEL_DIR / "pca.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
PROTOTYPE_PATH = MODEL_DIR / "prototype_extractor.joblib"
ENCODER_PATH = MODEL_DIR / "encoder.json"

# ─── Fallback model (stage0 baseline) ───
FALLBACK_MODEL_DIR = PROJECT_ROOT / "results" / "hybrid"
FALLBACK_LGBM_PATH = FALLBACK_MODEL_DIR / "lgbm_SimpleCNN1D.joblib"
FALLBACK_PCA_PATH = FALLBACK_MODEL_DIR / "pca_32.joblib"
FALLBACK_ENCODER_PATH = FALLBACK_MODEL_DIR / "chemical_family_model_encoder.json"

# ─── Raman axis reconstruction ───
RAMAN_WN_MIN = 100.0
RAMAN_WN_MAX = 4000.0
RAMAN_N_POINTS = 2048

# ─── XRD axis reconstruction ───
XRD_2THETA_MIN = 5.0
XRD_2THETA_MAX = 90.0
XRD_N_POINTS = 2048

# ─── API config ───
API_HOST = os.getenv("SPEC2PROP_HOST", "0.0.0.0")
API_PORT = int(os.getenv("SPEC2PROP_PORT", "8000"))
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]

# ─── Model info ───
MODEL_NAME = "LightGBM-DART + PCA + Expert Spectral Features"
MODEL_VERSION = "v1.0-deployment"
DATASET_NAME = "Spec2Prop-InorgBench"
TARGET_COL = "chemical_family_model"
