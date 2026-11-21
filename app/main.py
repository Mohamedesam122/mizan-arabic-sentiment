"""FastAPI inference service for the Arabic sentiment model.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
import json
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.preprocessing import normalize_arabic

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "sentiment_pipeline.joblib")
METRICS_PATH = os.path.join(BASE, "reports", "metrics.json")

app = FastAPI(
    title="Mizan - Arabic Sentiment API",
    description="Binary sentiment (POS/NEG) for MSA and Egyptian-dialect Arabic tweets.",
    version="1.0.0",
)

model = None
metrics = None


@app.on_event("startup")
def load_artifacts():
    global model, metrics
    model = joblib.load(MODEL_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000,
                      example="الفيلم ده كان تحفه بجد")


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=500)


def predict_frame(texts):
    clean = [normalize_arabic(t) for t in texts]
    if any(len(c) < 2 for c in clean):
        raise HTTPException(
            422, "Each text must contain Arabic content after normalization.")
    frame = pd.DataFrame({"clean": clean})
    proba = model.predict_proba(frame)
    classes = list(model.classes_)
    out = []
    for raw, cleaned, p in zip(texts, clean, proba):
        i = int(p.argmax())
        out.append({
            "text": raw,
            "label": classes[i],
            "confidence": round(float(p[i]), 4),
            "probabilities": {c: round(float(v), 4) for c, v in zip(classes, p)},
        })
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model_info")
def model_info():
    if metrics is None:
        raise HTTPException(503, "metrics not loaded")
    return {
        "model": metrics["best_model"],
        "train_size": metrics["train_size"],
        "test_accuracy": metrics["test_metrics"]["accuracy"],
        "test_macro_f1": metrics["test_metrics"]["macro_f1"],
        "egyptian_dialect_accuracy": metrics["egyptian_dialect_metrics"]["accuracy"],
        "egyptian_dialect_macro_f1": metrics["egyptian_dialect_metrics"]["macro_f1"],
    }


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    result = predict_frame([req.text])[0]
    result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    return {"predictions": predict_frame(req.texts), "count": len(req.texts)}
