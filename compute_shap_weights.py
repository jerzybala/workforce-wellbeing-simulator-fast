"""Pre-compute mean |SHAP| weights for each model source's MHQ and productivity models.

Run once from the app_ui_fast/ directory:
    python compute_shap_weights.py

Produces shap_weights.json and shap_weights_unprod.json in each model directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from config import FEATURE_CONFIG_PATH, MODEL_SOURCES

# Load feature config
with open(FEATURE_CONFIG_PATH) as f:
    FEATURES_CONFIG = json.load(f)

FEATURE_NAMES = [cfg["name"] for cfg in FEATURES_CONFIG]
N_SAMPLES = 1000


def generate_synthetic_data() -> pd.DataFrame:
    """Create random samples spanning each feature's range."""
    rng = np.random.default_rng(42)
    data = {}
    for cfg in FEATURES_CONFIG:
        lo, hi = cfg["min"], cfg["max"]
        data[cfg["name"]] = rng.integers(lo, hi + 1, size=N_SAMPLES).astype(float)
    return pd.DataFrame(data)


def _compute_weights_for_model(model_path: Path, pkl_name: str) -> dict[str, float] | None:
    pkl = model_path / pkl_name
    if not pkl.exists():
        print(f"  Skipping {model_path.name}: {pkl_name} not found")
        return None

    model = joblib.load(pkl)
    pre_model = model[:-1]  # imputer + restore_cols
    booster = model.named_steps["model"]

    X_synthetic = generate_synthetic_data()
    X_prepared = pre_model.transform(X_synthetic)

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_prepared)

    mean_abs = np.abs(shap_values).mean(axis=0)
    feature_names = list(X_prepared.columns) if hasattr(X_prepared, "columns") else FEATURE_NAMES

    return {name: round(float(val), 6) for name, val in zip(feature_names, mean_abs)}


def compute_weights(model_path: Path) -> dict[str, float] | None:
    return _compute_weights_for_model(model_path, "mhq_model.pkl")


def compute_weights_unprod(model_path: Path) -> dict[str, float] | None:
    return _compute_weights_for_model(model_path, "unproductive_days_model.pkl")


def main():
    for source_id, source_info in MODEL_SOURCES.items():
        model_path = source_info["path"]
        print(f"Processing {source_id} ({model_path}) ...")

        weights = compute_weights(model_path)
        if weights is not None:
            out_path = model_path / "shap_weights.json"
            with open(out_path, "w") as f:
                json.dump(weights, f, indent=2)
            print(f"  Saved {out_path}")

        weights_unprod = compute_weights_unprod(model_path)
        if weights_unprod is not None:
            out_path = model_path / "shap_weights_unprod.json"
            with open(out_path, "w") as f:
                json.dump(weights_unprod, f, indent=2)
            print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
