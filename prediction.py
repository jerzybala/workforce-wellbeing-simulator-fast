from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def heuristic_predict(row: pd.Series) -> Tuple[float, float]:
    sleep = row.get("sleep_hours", 7)
    exercise = row.get("exercise_days", 2)
    social = row.get("social_hours", 6)
    work = row.get("work_hours", 45)
    screen = row.get("screen_hours", 3)

    mhq = 50 + 2.5 * (sleep - 7) + 1.5 * exercise + 0.8 * (social / 5) - 0.4 * ((work - 45) / 5) - 0.6 * screen
    unprod = 8 - 0.7 * exercise - 0.5 * (sleep - 7) + 0.3 * ((work - 45) / 5) + 0.4 * screen - 0.2 * (social / 5)

    return float(mhq), float(max(unprod, 0))


def predict_targets(features: Dict[str, Any], mhq_model, unprod_model) -> Tuple[float, float]:
    df = pd.DataFrame([features])

    if mhq_model is not None:
        mhq_pred = float(mhq_model.predict(df)[0])
    else:
        mhq_pred, _ = heuristic_predict(df.iloc[0])

    if unprod_model is not None:
        unprod_pred = float(unprod_model.predict(df)[0])
    else:
        _, unprod_pred = heuristic_predict(df.iloc[0])

    return mhq_pred, unprod_pred


def predict_targets_batch(
    team_df: pd.DataFrame,
    mhq_model,
    unprod_model,
) -> Tuple[np.ndarray, np.ndarray]:
    if mhq_model is not None:
        mhq_preds = mhq_model.predict(team_df)
    else:
        mhq_preds = np.full(len(team_df), 50.0)
    if unprod_model is not None:
        unprod_preds = unprod_model.predict(team_df)
    else:
        unprod_preds = np.full(len(team_df), 5.0)
    return np.asarray(mhq_preds, dtype=float), np.asarray(unprod_preds, dtype=float)
