from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from prediction import predict_targets, predict_targets_batch
from team import apply_team_deltas


def compute_sensitivity(
    current_inputs: Dict[str, Any],
    features_config: List[Dict[str, Any]],
    mhq_model,
    unprod_model,
) -> List[Dict[str, Any]]:
    """Sweep each feature min→max independently, return response curves sorted by slope."""
    results: List[Dict[str, Any]] = []

    for cfg in features_config:
        name = cfg["name"]
        lo, hi, step = cfg["min"], cfg["max"], cfg.get("step", 1)
        current = float(current_inputs[name])

        curve = []
        for v in _range_inclusive(lo, hi, step):
            trial = dict(current_inputs)
            trial[name] = float(v)
            mhq, unprod = predict_targets(trial, mhq_model, unprod_model)
            curve.append({"value": float(v), "mhq": mhq, "unprod": unprod})

        # Find prediction at current value
        cur_point = _closest_point(curve, current)
        max_point = curve[-1]

        units = max_point["value"] - current
        slope_mhq = (max_point["mhq"] - cur_point["mhq"]) / units if units > 0 else 0.0
        slope_unprod = (max_point["unprod"] - cur_point["unprod"]) / units if units > 0 else 0.0

        results.append({
            "name": name,
            "current": current,
            "curve": curve,
            "slope_mhq": round(slope_mhq, 4),
            "slope_unprod": round(slope_unprod, 4),
            "total_delta_mhq": round(max_point["mhq"] - cur_point["mhq"], 4),
            "total_delta_unprod": round(max_point["unprod"] - cur_point["unprod"], 4),
        })

    results.sort(key=lambda r: r["slope_mhq"], reverse=True)
    return results


def compute_sensitivity_team(
    team_df: pd.DataFrame,
    team_averages: Dict[str, float],
    features_config: List[Dict[str, Any]],
    mhq_model,
    unprod_model,
) -> List[Dict[str, Any]]:
    """Sweep each feature min→max for team mode, return response curves sorted by slope."""
    results: List[Dict[str, Any]] = []

    for cfg in features_config:
        name = cfg["name"]
        lo, hi, step = cfg["min"], cfg["max"], cfg.get("step", 1)
        current = float(team_averages[name])

        curve = []
        for v in _range_inclusive(lo, hi, step):
            trial = dict(team_averages)
            trial[name] = float(v)
            adj = apply_team_deltas(team_df, team_averages, trial, features_config)
            mhq_preds, unprod_preds = predict_targets_batch(adj, mhq_model, unprod_model)
            curve.append({
                "value": float(v),
                "mhq": round(float(mhq_preds.mean()), 4),
                "unprod": round(float(unprod_preds.mean()), 4),
            })

        cur_point = _closest_point(curve, current)
        max_point = curve[-1]

        units = max_point["value"] - current
        slope_mhq = (max_point["mhq"] - cur_point["mhq"]) / units if units > 0 else 0.0
        slope_unprod = (max_point["unprod"] - cur_point["unprod"]) / units if units > 0 else 0.0

        results.append({
            "name": name,
            "current": current,
            "curve": curve,
            "slope_mhq": round(slope_mhq, 4),
            "slope_unprod": round(slope_unprod, 4),
            "total_delta_mhq": round(max_point["mhq"] - cur_point["mhq"], 4),
            "total_delta_unprod": round(max_point["unprod"] - cur_point["unprod"], 4),
        })

    results.sort(key=lambda r: r["slope_mhq"], reverse=True)
    return results


def _range_inclusive(lo: float, hi: float, step: float) -> List[float]:
    vals = []
    v = lo
    while v <= hi + 1e-9:
        vals.append(v)
        v += step
    return vals


def _closest_point(curve: List[Dict], target: float) -> Dict:
    return min(curve, key=lambda p: abs(p["value"] - target))
