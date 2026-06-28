"""
Spec2Prop-Edge: Pydantic Schemas
=================================
Request/response models for the API.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


# ─── Response models ───

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    dataset_loaded: bool
    model_name: str
    num_test_samples: int
    num_classes: int


class SampleSummary(BaseModel):
    sample_id: str
    rruff_id: str
    mineral_name: str
    original_12class_label: str
    true_9class_label: str
    true_5class_label: str
    has_raman: bool
    has_xrd: bool
    subset: str = "clean_inorganic"


class SampleListResponse(BaseModel):
    samples: List[SampleSummary]
    total: int


class SpectrumData(BaseModel):
    x: List[float]
    y: List[float]
    x_label: str
    y_label: str


class SampleDetailResponse(BaseModel):
    sample_id: str
    rruff_id: str
    mineral_name: str
    formula: str
    original_12class_label: str
    true_9class_label: str
    true_5class_label: str
    has_xrd: bool
    raman_spectrum: SpectrumData
    xrd_pattern: Optional[SpectrumData] = None
    metadata: Dict[str, Any]


class TopKPrediction(BaseModel):
    label: str
    probability: float


class InferenceRequest(BaseModel):
    sample_id: str
    task: str = "family"
    modality: str = "raman"


class InferenceResponse(BaseModel):
    sample_id: str
    task: str
    modality: str
    model_name: str
    
    # 5-class Broad Group
    predicted_5class_label: str
    predicted_5class_confidence: float
    is_correct_5class: bool
    
    # 9-class Fine Candidates
    predicted_9class_label: str
    predicted_9class_confidence: float
    is_correct_9class: bool
    top3_9class: List[TopKPrediction]
    
    prediction_quality: str
    inference_time_ms: float
    feature_dim: int
    recommendation: str
    disclaimer: str = "Screening-level prediction, not final experimental confirmation."


class RandomSampleResponse(BaseModel):
    sample: SampleSummary
