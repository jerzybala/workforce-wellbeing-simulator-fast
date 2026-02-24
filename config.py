from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FEATURE_CONFIG_PATH = DATA_DIR / "feature_config.json"

MODEL_SOURCES: Dict[str, Dict[str, Any]] = {
    "models_all": {"label": "All Data (32k samples)", "path": ROOT / "models_all"},
    "models_west": {"label": "Western Countries (5.8k samples)", "path": ROOT / "models_west"},
    "models_west_above45": {"label": "Western Age >45 (3.2k samples)", "path": ROOT / "models_west_above45"},
    "models_west_below45": {"label": "Western Age ≤45 (2.6k samples)", "path": ROOT / "models_west_below45"},
    "models": {"label": "App UI Models (current)", "path": ROOT / "models"},
}

DEFAULT_MODEL_SOURCE = "models_west"

EXERCISE_LABELS = [
    "Rarely/Never",
    "Less than once a week",
    "Once a week",
    "Few days a week",
    "Everyday",
]

UPF_LABELS = [
    "Several times a day",
    "Once a day",
    "Several days a week",
    "A few times a month",
    "Rarely/Never",
]

CATEGORY_LABELS = {
    "work": "Work Environment",
    "lifestyle": "Lifestyle Factors",
}

# CEN survey column names → app feature names
CEN_COLUMN_MAP = {
    "Please rate how much control and flexibility you have over your time (e.g. to leave work to take care of a personal matter)": "work_control_time",
    "Please rate how safe and secure you feel in your job": "work_control_job",
    "Please rate how you feel about the amount of work you have to do": "work_amount",
    "Please rate how you feel about the nature of your work": "work_purpose",
    "Please rate how you feel about learning and growth opportunities at work": "work_learning",
    "Please rate the nature of your social relationships with your work colleagues": "work_colleagues",
    "Please rate the nature of your relationship with your manager or supervisor": "work_manager",
    "Please rate how informed you feel about what is going on in your team": "work_informed",
    "Please rate how you perceive your work is valued and recognized": "work_recognition",
    "How regularly do you engage in physical exercise (30 minutes or more)?": "exercise_freq_ord",
    "How often do you eat processed, packaged, or fast food that is not made from fresh ingredients?": "UPF_freq_ord",
}

EXERCISE_TEXT_TO_ORD = {
    "Rarely/Never": 1,
    "Less than once a week": 2,
    "Once a week": 3,
    "Several days a week": 4,
    "Few days a week": 4,
    "Everyday": 5,
    "Every day": 5,
}

UPF_TEXT_TO_ORD = {
    "Several times a day": 1,
    "Once a day": 2,
    "Several days a week": 3,
    "Few days a week": 3,
    "A few times a month": 4,
    "Rarely/Never": 5,
}
