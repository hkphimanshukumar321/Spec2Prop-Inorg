import sys
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

from app.backend.config import CORS_ORIGINS
from app.backend.schemas import (
    HealthResponse, SampleListResponse, SampleSummary,
    SampleDetailResponse, SpectrumData,
    InferenceRequest, InferenceResponse, TopKPrediction,
    RandomSampleResponse,
)
from app.backend.sample_loader import SampleLoader
from app.backend.model_loader import ModelLoader
from app.backend.inference import run_inference

sample_loader = SampleLoader()
model_loader = ModelLoader()

@asynccontextmanager
async def lifespan(app: FastAPI):
    sample_loader.load()
    model_loader.load()
    yield

app = FastAPI(title="Spec2Prop-Edge", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok", model_loaded=model_loader.is_loaded, dataset_loaded=sample_loader.is_loaded,
        model_name=model_loader.model_name if model_loader.is_loaded else "Not loaded",
        num_test_samples=sample_loader.num_samples, num_classes=model_loader.num_classes,
    )

@app.get("/api/samples", response_model=SampleListResponse)
async def list_samples(limit: int = Query(50, ge=1, le=500)):
    if not sample_loader.is_loaded: raise HTTPException(status_code=503, detail="Dataset not loaded")
    samples = sample_loader.get_sample_list(limit=limit)
    return SampleListResponse(samples=[SampleSummary(**s) for s in samples], total=sample_loader.num_samples)

@app.get("/api/sample/{sample_id}", response_model=SampleDetailResponse)
async def get_sample(sample_id: str):
    if not sample_loader.is_loaded: raise HTTPException(status_code=503, detail="Dataset not loaded")
    detail = sample_loader.get_sample_detail(sample_id)
    if detail is None: raise HTTPException(status_code=404, detail="Sample not found")
    raman_spec = SpectrumData(**detail["raman_spectrum"]) if detail.get("raman_spectrum") else None
    xrd_spec = SpectrumData(**detail["xrd_pattern"]) if detail.get("xrd_pattern") else None
    return SampleDetailResponse(
        sample_id=detail["sample_id"], rruff_id=detail["rruff_id"], mineral_name=detail["mineral_name"],
        formula=detail["formula"], original_12class_label=detail["original_12class_label"],
        true_9class_label=detail["true_9class_label"], true_5class_label=detail["true_5class_label"],
        has_xrd=detail["has_xrd"], raman_spectrum=raman_spec, xrd_pattern=xrd_spec, metadata=detail["metadata"],
    )

@app.post("/api/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    if not model_loader.is_loaded: raise HTTPException(status_code=503, detail="Model not loaded")
    if not sample_loader.is_loaded: raise HTTPException(status_code=503, detail="Dataset not loaded")
    result = run_inference(request.sample_id, sample_loader, model_loader, request.task, request.modality)
    if result is None: raise HTTPException(status_code=500, detail="Inference failed")
    if "error" in result: raise HTTPException(status_code=400, detail=result["error"])
    return InferenceResponse(
        sample_id=result["sample_id"], task=result["task"], modality=result["modality"], model_name=result["model_name"],
        predicted_5class_label=result["predicted_5class_label"], predicted_5class_confidence=result["predicted_5class_confidence"],
        is_correct_5class=result["is_correct_5class"],
        predicted_9class_label=result["predicted_9class_label"], predicted_9class_confidence=result["predicted_9class_confidence"],
        is_correct_9class=result["is_correct_9class"], top3_9class=[TopKPrediction(**tk) for tk in result["top3_9class"]],
        prediction_quality=result["prediction_quality"], inference_time_ms=result["inference_time_ms"],
        feature_dim=result["feature_dim"], recommendation=result["recommendation"], disclaimer=result["disclaimer"],
    )

@app.get("/api/random-sample", response_model=RandomSampleResponse)
async def random_sample():
    if not sample_loader.is_loaded: raise HTTPException(status_code=503, detail="Dataset not loaded")
    sample = sample_loader.get_random_sample()
    if sample is None: raise HTTPException(status_code=500, detail="No valid samples")
    return RandomSampleResponse(sample=SampleSummary(**sample))
