from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import CEN_COLUMN_MAP, EXERCISE_TEXT_TO_ORD, EXTRA_COLUMNS, UPF_TEXT_TO_ORD


def validate_team_csv(
    file_bytes,
    features_config: List[Dict[str, Any]],
) -> Tuple[Optional[pd.DataFrame], Optional[str], Dict[str, float]]:
    try:
        df = pd.read_csv(file_bytes)
    except Exception as exc:
        return None, f"Could not read CSV: {exc}", {}

    if len(df) == 0:
        return None, "CSV file is empty.", {}
    if len(df) > 10_000:
        return None, f"CSV has {len(df):,} rows. Maximum supported is 10,000.", {}

    # Auto-rename CEN survey columns to app feature names
    df = df.rename(columns=CEN_COLUMN_MAP)

    # Extract extra column averages before filtering
    extra_col_averages: Dict[str, float] = {}
    for col in EXTRA_COLUMNS:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().any():
                extra_col_averages[col] = float(numeric.mean(skipna=True))

    expected_cols = [cfg["name"] for cfg in features_config]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        return None, f"Missing columns: {', '.join(missing)}", extra_col_averages

    df = df[expected_cols].copy()

    # Convert text labels to ordinals for lifestyle features
    if "exercise_freq_ord" in df.columns and df["exercise_freq_ord"].dtype == object:
        df["exercise_freq_ord"] = df["exercise_freq_ord"].map(EXERCISE_TEXT_TO_ORD)
    if "UPF_freq_ord" in df.columns and df["UPF_freq_ord"].dtype == object:
        df["UPF_freq_ord"] = df["UPF_freq_ord"].map(UPF_TEXT_TO_ORD)

    for col in expected_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    nan_count = int(df.isna().sum().sum())
    if nan_count > 0:
        df = df.dropna()
        if len(df) == 0:
            return None, "All rows contained non-numeric values and were dropped.", extra_col_averages

    cfg_map = {c["name"]: c for c in features_config}
    for col in expected_cols:
        lo, hi = cfg_map[col]["min"], cfg_map[col]["max"]
        df[col] = df[col].clip(lo, hi)

    return df.reset_index(drop=True), None, extra_col_averages


def compute_team_averages(
    team_df: pd.DataFrame,
    features_config: List[Dict[str, Any]],
) -> Dict[str, float]:
    averages: Dict[str, float] = {}
    for cfg in features_config:
        name = cfg["name"]
        step = cfg.get("step", 1)
        lo, hi = cfg["min"], cfg["max"]
        raw_mean = float(team_df[name].mean())
        rounded = round(raw_mean / step) * step
        averages[name] = max(lo, min(hi, rounded))
    return averages


def compute_teamq(
    feature_averages: Dict[str, float],
    shap_weights: Dict[str, float],
    features_config: List[Dict[str, Any]],
) -> float:
    """SHAP-weighted normalized average of feature values, scaled 0-100."""
    if not shap_weights:
        return 0.0
    cfg_map = {c["name"]: c for c in features_config}
    weighted_sum = 0.0
    weight_total = 0.0
    for name, avg_val in feature_averages.items():
        w = shap_weights.get(name, 0.0)
        if w == 0.0:
            continue
        lo, hi = cfg_map[name]["min"], cfg_map[name]["max"]
        norm = (float(avg_val) - lo) / (hi - lo) if hi != lo else 0.5
        norm = max(0.0, min(1.0, norm))
        weighted_sum += w * norm
        weight_total += w
    if weight_total == 0.0:
        return 0.0
    return (weighted_sum / weight_total) * 100.0


def apply_team_deltas(
    team_df: pd.DataFrame,
    team_averages: Dict[str, float],
    slider_values: Dict[str, Any],
    features_config: List[Dict[str, Any]],
) -> pd.DataFrame:
    adjusted = team_df.copy()
    cfg_map = {c["name"]: c for c in features_config}
    for name, avg in team_averages.items():
        delta = float(slider_values.get(name, avg)) - avg
        if delta != 0:
            lo, hi = cfg_map[name]["min"], cfg_map[name]["max"]
            adjusted[name] = (team_df[name] + delta).clip(lo, hi)
    return adjusted
