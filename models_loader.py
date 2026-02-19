from __future__ import annotations

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
        entry = {"mhq": None, "unprod": None}
        if mhq_path.exists():
            entry["mhq"] = joblib.load(mhq_path)
        if unprod_path.exists():
            entry["unprod"] = joblib.load(unprod_path)
        _models[source_id] = entry


def get_models(source_id: str) -> tuple:
    entry = _models.get(source_id, {"mhq": None, "unprod": None})
    return entry["mhq"], entry["unprod"]


def get_models_status() -> Dict[str, Dict[str, bool]]:
    status = {}
    for source_id, entry in _models.items():
        status[source_id] = {
            "mhq_loaded": entry["mhq"] is not None,
            "unprod_loaded": entry["unprod"] is not None,
        }
    return status
