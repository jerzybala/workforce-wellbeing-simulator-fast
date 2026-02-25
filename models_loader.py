from __future__ import annotations

import json
from typing import Any, Dict, Optional

import joblib

from config import MODEL_SOURCES

# Global model registry: {source_id: {"mhq": model, "unprod": model}}
_models: Dict[str, Dict[str, Any]] = {}


def load_all_models() -> None:
    for source_id, source_info in MODEL_SOURCES.items():
        path = source_info["path"]
        mhq_path = path / "mhq_model.pkl"
        unprod_path = path / "unproductive_days_model.pkl"
        entry = {"mhq": None, "unprod": None, "shap_weights": None, "shap_weights_unprod": None}
        if mhq_path.exists():
            entry["mhq"] = joblib.load(mhq_path)
        if unprod_path.exists():
            entry["unprod"] = joblib.load(unprod_path)
        shap_path = path / "shap_weights.json"
        if shap_path.exists():
            with open(shap_path) as f:
                entry["shap_weights"] = json.load(f)
        shap_unprod_path = path / "shap_weights_unprod.json"
        if shap_unprod_path.exists():
            with open(shap_unprod_path) as f:
                entry["shap_weights_unprod"] = json.load(f)
        _models[source_id] = entry


def get_models(source_id: str) -> tuple:
    entry = _models.get(source_id, {"mhq": None, "unprod": None})
    return entry["mhq"], entry["unprod"]


def get_shap_weights(source_id: str) -> Optional[Dict[str, float]]:
    entry = _models.get(source_id, {})
    return entry.get("shap_weights")


def get_shap_weights_unprod(source_id: str) -> Optional[Dict[str, float]]:
    entry = _models.get(source_id, {})
    return entry.get("shap_weights_unprod")


def get_models_status() -> Dict[str, Dict[str, bool]]:
    status = {}
    for source_id, entry in _models.items():
        status[source_id] = {
            "mhq_loaded": entry["mhq"] is not None,
            "unprod_loaded": entry["unprod"] is not None,
        }
    return status
