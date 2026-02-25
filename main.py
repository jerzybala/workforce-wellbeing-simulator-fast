from __future__ import annotations

import io
import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from config import (
    CATEGORY_LABELS,
    DEFAULT_MODEL_SOURCE,
    EXERCISE_LABELS,
    FEATURE_CONFIG_PATH,
    MODEL_SOURCES,
    UPF_LABELS,
)
from models_loader import get_models, get_models_status, get_shap_weights, get_shap_weights_unprod, load_all_models
from optimization import search_best_levers, search_best_levers_team
from prediction import predict_targets, predict_targets_batch
from schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    OptimizeRequest,
    OptimizeResponse,
    OptimizeResult,
    PredictRequest,
    PredictResponse,
    SensitivityRequest,
    SensitivityResponse,
    UploadResponse,
)
from sensitivity import compute_sensitivity, compute_sensitivity_team
from team import apply_team_deltas, compute_team_averages, compute_teamq, validate_team_csv


def _load_features_config() -> List[Dict[str, Any]]:
    if FEATURE_CONFIG_PATH.exists():
        with FEATURE_CONFIG_PATH.open("r") as f:
            return json.load(f)
    return []


_features_config: List[Dict[str, Any]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _features_config
    _features_config = _load_features_config()
    load_all_models()
    yield


app = FastAPI(title="Workforce Wellbeing Simulator API", lifespan=lifespan)


class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


app.add_middleware(NoCacheStaticMiddleware)


# ── Health ────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "models": get_models_status()}


# ── Features config ───────────────────────────────────────────────────────

@app.get("/api/features-config")
async def features_config():
    return {
        "features": _features_config,
        "exercise_labels": EXERCISE_LABELS,
        "upf_labels": UPF_LABELS,
        "categories": CATEGORY_LABELS,
    }


# ── Model sources ─────────────────────────────────────────────────────────

@app.get("/api/models")
async def models_list():
    status = get_models_status()
    sources = []
    for source_id, info in MODEL_SOURCES.items():
        s = status.get(source_id, {})
        sources.append({
            "id": source_id,
            "label": info["label"],
            "mhq_loaded": s.get("mhq_loaded", False),
            "unprod_loaded": s.get("unprod_loaded", False),
        })
    return {"sources": sources, "default": DEFAULT_MODEL_SOURCE}


# ── Single prediction ─────────────────────────────────────────────────────

@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    mhq_model, unprod_model = get_models(req.model_source)
    mhq, unprod = predict_targets(req.features, mhq_model, unprod_model)
    return PredictResponse(mhq=mhq, unproductive_days=unprod)


# ── Batch prediction ──────────────────────────────────────────────────────

@app.post("/api/predict-batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest):
    mhq_model, unprod_model = get_models(req.model_source)
    team_df = pd.DataFrame(req.team_data, columns=req.feature_names)

    # Apply slider deltas
    adjusted_df = apply_team_deltas(
        team_df, req.team_averages, req.slider_values, _features_config,
    )

    mhq_preds, unprod_preds = predict_targets_batch(adjusted_df, mhq_model, unprod_model)

    shap_weights = get_shap_weights(req.model_source)
    teamq = compute_teamq(req.slider_values, shap_weights, _features_config) if shap_weights else 0.0

    shap_weights_unprod = get_shap_weights_unprod(req.model_source)
    teamp = compute_teamq(req.slider_values, shap_weights_unprod, _features_config) if shap_weights_unprod else 0.0

    return BatchPredictResponse(
        avg_mhq=float(mhq_preds.mean()),
        avg_unproductive_days=float(unprod_preds.mean()),
        teamq=teamq,
        teamp=teamp,
        individual_mhq=mhq_preds.tolist(),
        individual_unproductive_days=unprod_preds.tolist(),
    )


# ── CSV upload ────────────────────────────────────────────────────────────

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...), model_source: str = "models_west"):
    contents = await file.read()
    df, error = validate_team_csv(io.BytesIO(contents), _features_config)
    if error:
        return {"error": error}

    mhq_model, unprod_model = get_models(model_source)
    team_averages = compute_team_averages(df, _features_config)
    team_raw_averages = {cfg["name"]: float(df[cfg["name"]].mean()) for cfg in _features_config}
    feature_names = [cfg["name"] for cfg in _features_config]

    mhq_base, unprod_base = predict_targets_batch(df, mhq_model, unprod_model)

    shap_weights = get_shap_weights(model_source)
    # Use rounded team_averages (same values the sliders start at) so baseline matches initial prediction
    baseline_teamq = compute_teamq(team_averages, shap_weights, _features_config) if shap_weights else 0.0

    shap_weights_unprod = get_shap_weights_unprod(model_source)
    baseline_teamp = compute_teamq(team_averages, shap_weights_unprod, _features_config) if shap_weights_unprod else 0.0

    return UploadResponse(
        team_data=df.values.tolist(),
        feature_names=feature_names,
        team_averages=team_averages,
        team_raw_averages=team_raw_averages,
        row_count=len(df),
        baseline_mhq=float(mhq_base.mean()),
        baseline_unproductive_days=float(unprod_base.mean()),
        baseline_teamq=baseline_teamq,
        baseline_teamp=baseline_teamp,
        baseline_individual_mhq=mhq_base.tolist(),
        baseline_individual_unproductive_days=unprod_base.tolist(),
    )


# ── Optimization ──────────────────────────────────────────────────────────

@app.post("/api/optimize", response_model=OptimizeResponse)
async def optimize(req: OptimizeRequest):
    mhq_model, unprod_model = get_models(req.model_source)

    weight_map = {"mhq": 1.0, "productivity": 0.0, "balanced": 0.6}
    weight_mhq = weight_map.get(req.goal, 0.6)

    if req.mode == "team" and req.team_data is not None and req.team_averages is not None:
        feature_names = [cfg["name"] for cfg in _features_config]
        team_df = pd.DataFrame(req.team_data, columns=feature_names)
        results = search_best_levers_team(
            team_df, req.team_averages, _features_config,
            mhq_model, unprod_model,
            k=req.k, weight_mhq=weight_mhq, top_n=1,
        )
    else:
        results = search_best_levers(
            req.current_inputs, _features_config,
            mhq_model, unprod_model,
            k=req.k, weight_mhq=weight_mhq, top_n=1,
        )

    top_results = []
    for r in results:
        top_results.append(OptimizeResult(
            levers=list(r["levers"]),
            values=[float(v) for v in r["values"]],
            mhq_pred=r["mhq_pred"],
            unprod_pred=r["unprod_pred"],
            delta_mhq=r["delta_mhq"],
            delta_unprod=r["delta_unprod"],
            score=r["score"],
        ))

    return OptimizeResponse(top_results=top_results)


# ── Sensitivity analysis ─────────────────────────────────────────────────

@app.post("/api/sensitivity", response_model=SensitivityResponse)
async def sensitivity(req: SensitivityRequest):
    mhq_model, unprod_model = get_models(req.model_source)

    if req.mode == "team" and req.team_data is not None and req.team_averages is not None:
        feature_names = [cfg["name"] for cfg in _features_config]
        team_df = pd.DataFrame(req.team_data, columns=feature_names)
        results = compute_sensitivity_team(
            team_df, req.team_averages, _features_config,
            mhq_model, unprod_model,
        )
    else:
        results = compute_sensitivity(
            req.current_inputs, _features_config,
            mhq_model, unprod_model,
        )

    return SensitivityResponse(features=results)


# ── SHAP weights (for report) ────────────────────────────────────────────

@app.get("/api/shap-weights")
async def shap_weights(model_source: str = "models_west"):
    weights = get_shap_weights(model_source)
    return {"weights": weights}


# ── Static files & SPA fallback ──────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")
