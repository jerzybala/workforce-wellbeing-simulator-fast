"""Pre-compute mean |SHAP| weights for each model source's MHQ model.

Run once from the app_ui_fast/ directory:
    python compute_shap_weights.py

Produces shap_weights.json in each model directory.
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


def compute_weights(model_path: Path) -> dict[str, float] | None:
    mhq_path = model_path / "mhq_model.pkl"
    if not mhq_path.exists():
        print(f"  Skipping {model_path.name}: mhq_model.pkl not found")
        return None

    model = joblib.load(mhq_path)
    pre_model = model[:-1]  # imputer + restore_cols
    booster = model.named_steps["model"]

    X_synthetic = generate_synthetic_data()
    X_prepared = pre_model.transform(X_synthetic)

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_prepared)

    mean_abs = np.abs(shap_values).mean(axis=0)
    feature_names = list(X_prepared.columns) if hasattr(X_prepared, "columns") else FEATURE_NAMES

    weights = {name: round(float(val), 6) for name, val in zip(feature_names, mean_abs)}
    return weights


def main():
    for source_id, source_info in MODEL_SOURCES.items():
        model_path = source_info["path"]
        print(f"Processing {source_id} ({model_path}) ...")
        weights = compute_weights(model_path)
        if weights is None:
            continue

        out_path = model_path / "shap_weights.json"
        with open(out_path, "w") as f:
            json.dump(weights, f, indent=2)
        print(f"  Saved {out_path}")
        print(f"  Weights: {weights}")


if __name__ == "__main__":
    main()
