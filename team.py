from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import CEN_COLUMN_MAP, EXERCISE_TEXT_TO_ORD, UPF_TEXT_TO_ORD


def validate_team_csv(
    file_bytes,
    features_config: List[Dict[str, Any]],
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        df = pd.read_csv(file_bytes)
    except Exception as exc:
        return None, f"Could not read CSV: {exc}"

    if len(df) == 0:
        return None, "CSV file is empty."
    if len(df) > 10_000:
        return None, f"CSV has {len(df):,} rows. Maximum supported is 10,000."

    # Auto-rename CEN survey columns to app feature names
    df = df.rename(columns=CEN_COLUMN_MAP)

    expected_cols = [cfg["name"] for cfg in features_config]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        return None, f"Missing columns: {', '.join(missing)}"

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
            return None, "All rows contained non-numeric values and were dropped."

    cfg_map = {c["name"]: c for c in features_config}
    for col in expected_cols:
        lo, hi = cfg_map[col]["min"], cfg_map[col]["max"]
        df[col] = df[col].clip(lo, hi)

    return df.reset_index(drop=True), None


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
