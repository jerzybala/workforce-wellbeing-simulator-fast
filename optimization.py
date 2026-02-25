from __future__ import annotations

import itertools
from typing import Any, Dict, List

import pandas as pd

from prediction import predict_targets, predict_targets_batch
from team import apply_team_deltas


def candidate_levels(name: str, current: float, cfg: Dict[str, Any]) -> List[float]:
    if name == "exercise_freq_ord" or name == "UPF_freq_ord":
        return sorted({float(current), float(cfg.get("max", 5))})
    max_v = cfg.get("max", 10)
    cap7 = max(7.0, float(current))
    return sorted({float(current), min(cap7, max_v), float(max_v)})


def search_best_levers(
    current_inputs: Dict[str, Any],
    features_config: List[Dict[str, Any]],
    mhq_model,
    unprod_model,
    k: int = 2,
    weight_mhq: float = 0.6,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    base_mhq, base_unprod = predict_targets(current_inputs, mhq_model, unprod_model)
    cfg_map = {c["name"]: c for c in features_config}
    names = list(cfg_map.keys())
    rows: List[Dict[str, Any]] = []

    for levers in itertools.combinations(names, k):
        level_lists = [candidate_levels(name, current_inputs[name], cfg_map[name]) for name in levers]
        for values in itertools.product(*level_lists):
            trial = dict(current_inputs)
            for name, val in zip(levers, values):
                trial[name] = val
            mhq_pred, unprod_pred = predict_targets(trial, mhq_model, unprod_model)
            delta_mhq = mhq_pred - base_mhq
            delta_unprod = unprod_pred - base_unprod
            score = weight_mhq * delta_mhq + (1 - weight_mhq) * (-delta_unprod)
            rows.append({
                "levers": levers,
                "values": values,
                "mhq_pred": mhq_pred,
                "unprod_pred": unprod_pred,
                "delta_mhq": delta_mhq,
                "delta_unprod": delta_unprod,
                "score": score,
            })

    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    return rows_sorted[:top_n]


def search_best_levers_team(
    team_df: pd.DataFrame,
    team_averages: Dict[str, float],
    features_config: List[Dict[str, Any]],
    mhq_model,
    unprod_model,
    k: int = 2,
    weight_mhq: float = 0.6,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    base_mhq, base_unprod = predict_targets_batch(team_df, mhq_model, unprod_model)
    base_avg_mhq = float(base_mhq.mean())
    base_avg_unprod = float(base_unprod.mean())

    cfg_map = {c["name"]: c for c in features_config}
    names = list(cfg_map.keys())
    rows: List[Dict[str, Any]] = []

    for levers in itertools.combinations(names, k):
        level_lists = [
            candidate_levels(name, team_averages[name], cfg_map[name])
            for name in levers
        ]
        for values in itertools.product(*level_lists):
            trial = dict(team_averages)
            for name, val in zip(levers, values):
                trial[name] = val
            adj = apply_team_deltas(team_df, team_averages, trial, features_config)
            mhq_preds, unprod_preds = predict_targets_batch(adj, mhq_model, unprod_model)
            avg_mhq = float(mhq_preds.mean())
            avg_unprod = float(unprod_preds.mean())
            delta_mhq = avg_mhq - base_avg_mhq
            delta_unprod = avg_unprod - base_avg_unprod
            score = weight_mhq * delta_mhq + (1 - weight_mhq) * (-delta_unprod)
            rows.append({
                "levers": levers,
                "values": values,
                "mhq_pred": avg_mhq,
                "unprod_pred": avg_unprod,
                "delta_mhq": delta_mhq,
                "delta_unprod": delta_unprod,
                "score": score,
            })

    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    return rows_sorted[:top_n]
