"""
Spec2Prop-Edge: Model Loader
==============================
Loads the trained LightGBM-DART model + fitted PCA/scaler/prototype objects.
All artifacts come from the real training pipeline — no fake models.
"""
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Optional, Dict

from app.backend.config import (
    LGBM_MODEL_PATH, PCA_PATH, SCALER_PATH, PROTOTYPE_PATH, ENCODER_PATH,
    FALLBACK_LGBM_PATH, FALLBACK_PCA_PATH, FALLBACK_ENCODER_PATH,
    MODEL_NAME, MODEL_VERSION,
)


class ModelLoader:
    """
    Loads and validates the deployment model pipeline.
    
    Components:
      - LightGBM-DART classifier
      - PCA transform (fitted on train data)
      - StandardScaler (fitted on train data)
      - PrototypeExtractor (fitted on train data)
      - Label encoder (class name <-> index mapping)
    """

    def __init__(self):
        self.model = None
        self.pca = None
        self.scaler = None
        self.prototype_extractor = None
        self.encoder: Optional[Dict[str, int]] = None
        self.inv_encoder: Optional[Dict[int, str]] = None
        self.num_classes: int = 0
        self.model_name: str = MODEL_NAME
        self.model_version: str = MODEL_VERSION
        self._loaded = False
        self._using_fallback = False

    def load(self) -> bool:
        """Load all model artifacts. Returns True on success."""
        try:
            # Try primary deployment model first
            if LGBM_MODEL_PATH.exists():
                print(f"[ModelLoader] Loading deployment model: {LGBM_MODEL_PATH}")
                self.model = joblib.load(LGBM_MODEL_PATH)
                self._using_fallback = False
            elif FALLBACK_LGBM_PATH.exists():
                print(f"[ModelLoader] Deployment model not found, using fallback: {FALLBACK_LGBM_PATH}")
                self.model = joblib.load(FALLBACK_LGBM_PATH)
                self._using_fallback = True
                self.model_name = "LightGBM (Fallback — SimpleCNN1D embeddings)"
            else:
                print("[ModelLoader] ERROR: No trained model found!")
                print(f"  Expected: {LGBM_MODEL_PATH}")
                print(f"  Fallback: {FALLBACK_LGBM_PATH}")
                return False

            # Load PCA
            if PCA_PATH.exists():
                self.pca = joblib.load(PCA_PATH)
                print(f"[ModelLoader] PCA loaded: {PCA_PATH}")
            elif FALLBACK_PCA_PATH.exists():
                self.pca = joblib.load(FALLBACK_PCA_PATH)
                print(f"[ModelLoader] PCA fallback loaded: {FALLBACK_PCA_PATH}")

            # Load scaler
            if SCALER_PATH.exists():
                self.scaler = joblib.load(SCALER_PATH)
                print(f"[ModelLoader] Scaler loaded: {SCALER_PATH}")

            # Load prototype extractor
            if PROTOTYPE_PATH.exists():
                self.prototype_extractor = joblib.load(PROTOTYPE_PATH)
                print(f"[ModelLoader] PrototypeExtractor loaded: {PROTOTYPE_PATH}")

            # Load encoder
            enc_path = ENCODER_PATH if ENCODER_PATH.exists() else FALLBACK_ENCODER_PATH
            if enc_path.exists():
                with open(enc_path) as f:
                    self.encoder = json.load(f)
                self.inv_encoder = {int(v): k for k, v in self.encoder.items()}
                self.num_classes = len(self.encoder)
                print(f"[ModelLoader] Encoder loaded: {self.num_classes} classes")
            else:
                print("[ModelLoader] WARNING: No encoder found, using hardcoded fallback")
                self.encoder = {
                    "Borate": 0, "Carbonate": 1, "Halide": 2,
                    "Other/Rare": 3, "Oxide": 4, "Phosphate": 5,
                    "Silicate": 6, "Sulfate": 7, "Sulfide": 8,
                }
                self.inv_encoder = {v: k for k, v in self.encoder.items()}
                self.num_classes = 9

            self._loaded = True
            print(f"[ModelLoader] ✓ Model pipeline ready ({self.model_name})")
            return True

        except Exception as e:
            print(f"[ModelLoader] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    @property
    def has_full_pipeline(self) -> bool:
        """True if we have PCA + scaler + prototypes (deployment model)."""
        return (self.pca is not None and
                self.scaler is not None and
                self.prototype_extractor is not None and
                not self._using_fallback)

    def get_class_names(self) -> list:
        """Return ordered list of class names."""
        if self.inv_encoder:
            return [self.inv_encoder[i] for i in range(self.num_classes)]
        return []
