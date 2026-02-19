from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class PredictRequest(BaseModel):
    features: Dict[str, float]
    model_source: str = "models_west"


class PredictResponse(BaseModel):
    mhq: float
    unproductive_days: float


class BatchPredictRequest(BaseModel):
    team_data: List[List[float]]
    feature_names: List[str]
    slider_values: Dict[str, float]
    team_averages: Dict[str, float]
    model_source: str = "models_west"


class BatchPredictResponse(BaseModel):
    avg_mhq: float
    avg_unproductive_days: float
    individual_mhq: List[float]
    individual_unproductive_days: List[float]


class OptimizeRequest(BaseModel):
    mode: str  # "individual" or "team"
    current_inputs: Dict[str, float]
    model_source: str = "models_west"
    k: int = 2
    goal: str = "balanced"  # "mhq", "productivity", "balanced"
    team_data: Optional[List[List[float]]] = None
    team_averages: Optional[Dict[str, float]] = None


class OptimizeResult(BaseModel):
    levers: List[str]
    values: List[float]
    mhq_pred: float
    unprod_pred: float
    delta_mhq: float
    delta_unprod: float
    score: float


class OptimizeResponse(BaseModel):
    top_results: List[OptimizeResult]


class UploadResponse(BaseModel):
    team_data: List[List[float]]
    feature_names: List[str]
    team_averages: Dict[str, float]
    team_raw_averages: Dict[str, float]
    row_count: int
    baseline_mhq: float
    baseline_unproductive_days: float
    baseline_individual_mhq: List[float]
    baseline_individual_unproductive_days: List[float]
