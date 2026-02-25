"""
Work factors uplift analysis.

This script trains regression models for overall MHQ and productivity composite
using the nine work factors plus exercise and UPF frequency. It then estimates
individual and aggregate uplift when all work factors are set to maximum,
exercise is maximized, and UPF is minimized.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

try:
    from catboost import CatBoostRegressor  # type: ignore
    CATBOOST_AVAILABLE = True
except ImportError:
    CatBoostRegressor = None  # type: ignore
    CATBOOST_AVAILABLE = False
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

DATA_PATH = Path("work_factors_subset.csv")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
MODEL_TYPE = os.getenv("MODEL_TYPE", "lightgbm")  # options: catboost, lightgbm
USE_EXTENDED_FEATURES = os.getenv("USE_EXTENDED_FEATURES", "1") != "0"
PLOT_TITLE_TAG = os.getenv("PLOT_TITLE_TAG", "")

CATEGORICAL_COLS = [
    "employment",
    "team_situation",
    "country",
    "biological_sex",
    "education",
    "job_features",
    "work_situation",
]

WORK_FACTOR_COLS = [
    "work_control_time",
    "work_control_job",
    "work_amount",
    "work_purpose",
    "work_learning",
    "work_colleagues",
    "work_manager",
    "work_informed",
    "work_recognition",
]

EXERCISE_ORDER = [
    "Rarely/Never",
    "Less than once a week",
    "Once a week",
    "Several days a week",
    "Everyday",
]

UPF_ORDER = [
    "Several times a day",
    "Once a day",
    "Several days a week",
    "A few times a month",
    "Rarely/Never",
]

SLEEP_ORDER = [
    "Hardly ever",
    "Some of the time",
    "Most of the time",
    "All of the time",
]

SCENARIOS = [
    ("cap7", 7),  # recommended: lift low factors to at least 7, set lifestyle best
    ("full", None),  # optimistic: set all factors to max
]
RECOMMENDED_SCENARIO = "cap7"


def _with_title_tag(title: str, tag: str = PLOT_TITLE_TAG) -> str:
    if not tag:
        return title
    return f"{title} ({tag})"


class RestoreCols(BaseEstimator, TransformerMixin):
    """Wrap numpy output back into a DataFrame with stored column names."""

    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        # Use incoming columns if available; fallback to provided list.
        if hasattr(X, "columns"):
            self.columns_ = list(X.columns)
        else:
            self.columns_ = list(self.columns)
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns_)


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Simple smoothing target encoder for categorical columns."""

    def __init__(self, columns: List[str], smoothing: float = 10.0):
        self.columns = columns
        self.smoothing = smoothing

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)
        self.global_mean_ = float(y_series.mean())
        self.maps_: Dict[str, pd.Series] = {}
        for col in self.columns:
            if col not in X_df.columns:
                continue
            tmp = pd.DataFrame({"col": X_df[col], "y": y_series})
            agg = tmp.groupby("col")["y"].agg(["mean", "count"]).reset_index()
            enc = (agg["mean"] * agg["count"] + self.global_mean_ * self.smoothing) / (agg["count"] + self.smoothing)
            self.maps_[col] = pd.Series(enc.values, index=agg["col"].astype(str))
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in self.columns:
            if col not in X_df.columns:
                continue
            mapping = self.maps_.get(col)
            if mapping is None:
                X_df[col] = self.global_mean_
                continue
            X_df[col] = X_df[col].astype(str).map(mapping).fillna(self.global_mean_)
        return X_df


def _make_frequency_maps() -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Create ordinal maps for exercise, UPF, and sleep (all higher is better)."""
    exercise_map = {label: idx for idx, label in enumerate(EXERCISE_ORDER)}
    # For UPF: UPF_ORDER goes from worst (most UPF = "Several times a day") to best (least UPF = "Rarely/Never").
    # We want higher ordinal values to mean healthier (less UPF), so we assign values in natural order:
    # "Several times a day" (worst) = 0, ..., "Rarely/Never" (best) = 4
    upf_map = {label: idx for idx, label in enumerate(UPF_ORDER)}
    # Normalize casing variant
    upf_map["Rarely/never"] = upf_map["Rarely/Never"]
    # Sleep: "Hardly ever" (worst) = 0, ..., "All of the time" (best) = 3
    sleep_map = {label: idx for idx, label in enumerate(SLEEP_ORDER)}
    return exercise_map, upf_map, sleep_map


def _midpoint_from_text(val: object) -> float:
    """Extract a numeric midpoint from range-like strings (e.g., "25-34", "10+ years")."""
    if pd.isna(val):
        return float("nan")
    text = str(val)
    numbers = [float(x) for x in re.findall(r"\d+", text)]
    if not numbers:
        return float("nan")
    if "+" in text and len(numbers) == 1:
        return numbers[0]
    if len(numbers) >= 2:
        return float(np.mean(numbers[:2]))
    return numbers[0]


def load_and_prepare(
    path: Path = DATA_PATH,
    use_extended: bool = USE_EXTENDED_FEATURES,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    exercise_map, upf_map, sleep_map = _make_frequency_maps()
    df = pd.read_csv(path)
    # Preserve original row order for traceability.
    df = df.reset_index().rename(columns={"index": "row_id"}).set_index("row_id")

    # Standardize column name
    if "productivity_composite" not in df.columns and "productavity_composite" in df.columns:
        df = df.rename(columns={"productavity_composite": "productivity_composite"})

    # Normalize UPF label variant
    df["UPF_freq"] = df["UPF_freq"].str.replace("Rarely/never", "Rarely/Never", case=False)

    df["exercise_freq_ord"] = df["exercise_freq"].map(exercise_map)
    df["UPF_freq_ord"] = df["UPF_freq"].map(upf_map)
    df["sleep_freq_ord"] = df["sleep_freq"].map(sleep_map)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    # Drop rows where mapping failed
    df = df.dropna(subset=["exercise_freq_ord", "UPF_freq_ord", "sleep_freq_ord", "overall_mhq_score", "productivity_composite"])

    feature_cols = WORK_FACTOR_COLS + [
        "exercise_freq_ord",
        "UPF_freq_ord",
        "sleep_freq_ord",
    ]

    if use_extended:
        df["age_mid"] = df["age"].apply(_midpoint_from_text)
        df["job_duration_years"] = df["job_duration"].apply(_midpoint_from_text)
        df["org_size_mid"] = df["organization_size"].apply(_midpoint_from_text)
        feature_cols += [
            "age_mid",
            "job_duration_years",
            "org_size_mid",
        ]
        feature_cols += [col for col in CATEGORICAL_COLS if col in df.columns]
    X = df[feature_cols].copy()
    y_mhq = df["overall_mhq_score"].copy()
    y_prod = df["productivity_composite"].copy()
    return df, X, y_mhq, y_prod


def build_model(
    feature_names: List[str],
    model_type: str = MODEL_TYPE,
    monotone_sign: int = 1,
    monotone_constraints: Optional[List[int]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Pipeline:
    mono = monotone_constraints or ([monotone_sign] * len(feature_names))
    model_choice = model_type.lower()

    if model_choice == "catboost" and CATBOOST_AVAILABLE:
        reg = CatBoostRegressor(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_seed=42,
            monotone_constraints=mono,
            verbose=0,
            allow_writing_files=False,
        )
        chosen = "catboost"
    else:
        reg = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_samples=120,
            min_split_gain=0.1,
            random_state=42,
            n_jobs=-1,
            monotone_constraints=mono,
            verbose=-1,
        )
        chosen = "lightgbm"

    steps: List[Tuple[str, object]] = []
    if categorical_cols:
        steps.append(("target_encoder", TargetEncoder(categorical_cols, smoothing=10.0)))
    steps.extend([
        ("imputer", SimpleImputer(strategy="median")),
        ("restore_cols", RestoreCols(feature_names)),
        ("model", reg),
    ])

    model = Pipeline(steps)
    model.chosen_model = chosen  # type: ignore
    return model


def evaluate_cv(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds) ** 0.5
    r_val = float(pd.Series(y).corr(pd.Series(preds)))
    if pd.isna(r_val):
        r_val = 0.0
    return {"mae": mae, "rmse": rmse, "r": r_val}


def compute_uplift(
    model: Pipeline,
    X: pd.DataFrame,
    strategy: str = "full",
    floor: int = 7,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Compute counterfactual uplift.

    strategy:
        - "full": set all work factors to 9, exercise to best, UPF to healthiest.
        - "cap": only raise work factors (and lifestyle ordinals) up to at least `floor`.
    """
    X_base = X.copy()
    X_cf = X.copy()

    exercise_max = max(_make_frequency_maps()[0].values())
    upf_best = max(_make_frequency_maps()[1].values())
    sleep_best = max(_make_frequency_maps()[2].values())

    if strategy == "cap":
        X_cf[WORK_FACTOR_COLS] = np.maximum(X_cf[WORK_FACTOR_COLS], floor)
    else:
        X_cf[WORK_FACTOR_COLS] = 9

    # Lifestyle always pushed to healthiest observed ordinal
    X_cf["exercise_freq_ord"] = exercise_max
    X_cf["UPF_freq_ord"] = upf_best
    X_cf["sleep_freq_ord"] = sleep_best

    base_pred = model.predict(X_base)
    cf_pred = model.predict(X_cf)
    # For targets where lower is better, an improvement is a reduction (base - cf).
    uplift = cf_pred - base_pred if higher_is_better else base_pred - cf_pred

    return pd.DataFrame({
        "pred_base": base_pred,
        "pred_cf": cf_pred,
        "uplift": uplift,
    })


def summarize_uplift(uplift: pd.Series) -> pd.Series:
    return uplift.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])


def save_shap_summary(model: Pipeline, X: pd.DataFrame, target_name: str, out_dir: Path = OUTPUT_DIR) -> Tuple[Path, Path]:
    import matplotlib.pyplot as plt
    import shap

    out_dir.mkdir(exist_ok=True, parents=True)
    # Sample to keep plotting fast
    sample_size = min(2000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)

    pre_model = model[:-1]  # all steps before the model
    booster = model.named_steps["model"]

    X_prepared = pre_model.transform(X_sample)
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_prepared)

    # Standard SHAP summary (beeswarm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_prepared, show=False)
    shap_title = f"SHAP summary - {target_name}"
    if target_name == "productivity":
        shap_title = "SHAP summary - productivity (unproductive days)"
    shap_title = _with_title_tag(shap_title)
    plt.title(shap_title)
    outfile = out_dir / f"shap_{target_name}.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

    # Mean |SHAP| impact bar plot
    shap_abs = np.abs(shap_values)
    mean_abs = shap_abs.mean(axis=0)
    feature_names = list(X_prepared.columns)
    order = np.argsort(mean_abs)[-20:][::-1]
    plt.figure(figsize=(8, 6))
    plt.barh([feature_names[i] for i in order], mean_abs[order])
    plt.gca().invert_yaxis()
    plt.xlabel("mean |SHAP|")
    plt.title(shap_title + " (impact)")
    outfile_abs = out_dir / f"shap_abs_{target_name}.png"
    plt.tight_layout()
    plt.savefig(outfile_abs, dpi=200)
    plt.close()

    return outfile, outfile_abs


def _make_age_order(age_series: pd.Series) -> List[str]:
    unique = age_series.dropna().unique()
    def _key(val: str) -> int:
        try:
            if "-" in val:
                return int(val.split("-")[0])
            return int(val)
        except Exception:
            return 10_000  # push unknowns to the end

    return sorted(unique, key=_key)


def _make_category_order(series: pd.Series) -> List[str]:
    return sorted(series.dropna().unique())


def plot_uplift_hist(uplift_df: pd.DataFrame, target_name: str, out_dir: Path = OUTPUT_DIR) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(8, 5))
    scenarios = list(uplift_df["scenario"].unique())
    colors = sns.color_palette("tab10", n_colors=len(scenarios))
    x_label = "uplift" if target_name != "productivity" else "uplift (days fewer unproductive days)"
    title = f"Uplift distribution - {target_name}"
    if target_name == "productivity":
        title = "Lift in productive days (reduced unproductive days)"
    # Use common bin edges and draw vertical guides at each edge.
    bins = 30
    data_min, data_max = uplift_df["uplift"].min(), uplift_df["uplift"].max()
    bin_edges = np.histogram_bin_edges(uplift_df["uplift"], bins=bins, range=(data_min, data_max))
    for scenario, color in zip(scenarios, colors):
        subset = uplift_df[uplift_df["scenario"] == scenario]
        sns.histplot(
            subset["uplift"],
            bins=bin_edges,
            stat="percent",
            element="step",
            fill=False,
            label=scenario,
            color=color,
            linewidth=1.0,
        )
    # Add bin edge guides (visible but subtle)
    for edge in bin_edges:
        plt.axvline(edge, color="#94a3b8", linestyle=":", linewidth=0.7, alpha=0.8)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel(x_label)
    plt.ylabel("percent")
    plt.title(_with_title_tag(title))
    plt.legend(title="scenario")
    outfile = out_dir / f"uplift_hist_{target_name}.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    return outfile


def plot_uplift_by_age(uplift_df: pd.DataFrame, target_name: str, out_dir: Path = OUTPUT_DIR) -> Optional[Path]:
    if "age" not in uplift_df.columns:
        return None

    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(10, 5))
    age_order = _make_age_order(uplift_df["age"])
    y_label = "uplift" if target_name != "productivity" else "uplift (days fewer unproductive days)"
    title = f"Uplift by age - {target_name}"
    if target_name == "productivity":
        title = "Lift by age - productive days (reduced unproductive days)"
    sns.boxplot(
        data=uplift_df,
        x="age",
        y="uplift",
        hue="scenario",
        order=age_order,
        fliersize=1.5,
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("age group")
    plt.ylabel(y_label)
    plt.title(_with_title_tag(title))
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="scenario")
    outfile = out_dir / f"uplift_age_{target_name}.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    return outfile


def plot_uplift_by_group(
    uplift_df: pd.DataFrame,
    target_name: str,
    group_col: str,
    title_prefix: str,
    out_dir: Path = OUTPUT_DIR,
) -> Optional[Path]:
    if group_col not in uplift_df.columns:
        return None

    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(10, 5))
    group_order = _make_category_order(uplift_df[group_col])
    y_label = "uplift" if target_name != "productivity" else "uplift (days fewer unproductive days)"
    title = f"{title_prefix} - {target_name}"
    if target_name == "productivity":
        title = f"{title_prefix} - productive days (reduced unproductive days)"

    sns.boxplot(
        data=uplift_df,
        x=group_col,
        y="uplift",
        hue="scenario",
        order=group_order,
        fliersize=1.5,
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel(group_col.replace("_", " "))
    plt.ylabel(y_label)
    plt.title(_with_title_tag(title))
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="scenario")
    outfile = out_dir / f"uplift_{group_col}_{target_name}.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    return outfile


def _relative_name(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    return path.name


def write_algorithm_report(out_dir: Path = OUTPUT_DIR) -> Path:
    out_dir.mkdir(exist_ok=True, parents=True)
    lines: List[str] = []
    lines.append("# Model Algorithms and Settings")
    lines.append("")

    lines.append("## Estimator architecture")
    if USE_EXTENDED_FEATURES:
        lines.append("- Pipeline: target encoding for categorical features -> median imputation -> column restoration -> LightGBM regressor (or CatBoost when selected).")
        lines.append("- Rationale: target encoding folds in categorical signal without high-cardinality one-hotting; imputation handles missingness; restoring column names keeps SHAP/constraints aligned; LightGBM provides fast, monotone-capable gradient boosting.")
    else:
        lines.append("- Pipeline: median imputation -> column restoration -> LightGBM regressor (no categorical encoding step when extended features are off).")
        lines.append("- Rationale: simpler pipeline when only numeric work factors + lifestyle ordinals are used; imputation handles missingness; restoring column names keeps SHAP/constraints aligned; LightGBM provides fast, monotone-capable gradient boosting.")
    lines.append("")

    lines.append("## Monotone constraints")
    if USE_EXTENDED_FEATURES:
        lines.append("- Signs: +1 for MHQ (higher is better), -1 for productivity (lower is better) on work factors + lifestyle ordinals; 0 (neutral) on added demographic/work-size/categorical encodings.")
        lines.append("- Why: enforces directionally plausible effects: higher work factors and exercise should not lower MHQ or raise unproductive days; healthier UPF should not worsen outcomes.")
        lines.append("- Benefit: prevents sign flips from noise/collinearity, improving interpretability and trust, while keeping non-ordered covariates unconstrained.")
    else:
        lines.append("- Signs: +1 for MHQ (higher is better), -1 for productivity (lower is better) on work factors + lifestyle ordinals (no additional neutral features).")
        lines.append("- Why: enforces directionally plausible effects for the core factors only.")
        lines.append("- Benefit: keeps effects directionally consistent for the Likert/ordinal drivers without adding unconstrained covariates.")
    lines.append("")

    lines.append("## Key hyperparameters")
    lines.append("- n_estimators=400, learning_rate=0.05: many small trees for smooth fits.")
    lines.append("- num_leaves=15, max_depth=4: shallow, small trees reduce variance and ease monotonicity satisfaction.")
    lines.append("- subsample=0.9, colsample_bytree=0.9: row/feature subsampling for bagging-like robustness.")
    lines.append("- min_child_samples=120, min_split_gain=0.1: regularization to avoid brittle splits." )
    lines.append("- random_state=42, n_jobs=-1: reproducibility and parallelism." )
    lines.append("")

    lines.append("## Why these choices")
    lines.append("- Stability vs. flexibility: shallow trees and small leaves constrain variance while still capturing nonlinearities.")
    lines.append("- Monotonicity feasibility: limited depth/leaves help the booster honor monotone constraints without oscillations.")
    lines.append("- Robustness: subsample and colsample decorrelate trees, reducing overfitting under monotone restrictions.")
    lines.append("- Interpretability: enforced direction ensures feature effects align with domain expectations (better work factors/exercise help; better UPF helps; the reverse would flag data issues).")
    lines.append("")

    lines.append("## Evaluation protocol")
    lines.append("- 5-fold shuffled KFold CV with MAE, RMSE, and r on out-of-fold predictions.")
    lines.append("- Purpose: lower variance estimate than a single split; consistent with constrained models where overfitting risk is lower but still possible.")
    lines.append("")

    lines.append("## Scenarios and counterfactuals")
    lines.append("- cap7: lift work factors to >=7; set exercise to best, UPF to healthiest.")
    lines.append("- full: set all work factors to 9; exercise best; UPF healthiest.")
    lines.append("- Uplift = pred_cf - pred_base (MHQ) or pred_base - pred_cf (productivity, where lower is better).")

    algo_path = out_dir / "algorithms_settings.md"
    algo_path.write_text("\n".join(lines))
    return algo_path


def plot_target_hist(series: pd.Series, label: str, out_dir: Path = OUTPUT_DIR) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(8, 4.5))
    sns.histplot(series.dropna(), bins=40, stat="percent", color="#2563eb", edgecolor="#0f172a", linewidth=0.8)
    for x in np.linspace(series.min(), series.max(), 9):
        plt.axvline(x, color="#94a3b8", linestyle="--", linewidth=0.7, alpha=0.7)
    plt.axvline(series.median(), color="orange", linestyle="--", linewidth=1.5, label="median")
    plt.axvline(series.mean(), color="green", linestyle="--", linewidth=1.0, label="mean")
    plt.xlabel(label.replace("_", " "))
    plt.ylabel("percent")
    plt.title(_with_title_tag(f"Distribution of {label.replace('_', ' ')}"))
    plt.legend()
    outfile = out_dir / f"target_hist_{label}.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    return outfile


REPORT_TITLE = "Work Model Uplift Analysis"


def _write_markdown_report(
    results: Dict[str, Dict],
    out_dir: Path = OUTPUT_DIR,
    data_info: Optional[Dict[str, object]] = None,
    appendix: Optional[Dict[str, object]] = None,
) -> Tuple[Path, Path]:
    """Write a rich markdown (and HTML) report with embedded plots."""

    def _target_label(target: str) -> str:
        return "MHQ" if target == "mhq" else "Productivity (fewer unproductive days)"

    def _uplift_definition(target: str) -> str:
        if target == "mhq":
            return "MHQ uplift = counterfactual – baseline (positive is better)."
        return "Productivity uplift = baseline – counterfactual (positive means fewer unproductive days)."

    md_lines: List[str] = []
    md_lines.append(f"# {REPORT_TITLE}")
    md_lines.append(f"Generated: {datetime.now():%Y-%m-%d}")
    md_lines.append("")

    if data_info:
        md_lines.append("## Data")
        md_lines.append(f"- Source: {data_info.get('source', 'n/a')}")
        md_lines.append(f"- Rows used: {data_info.get('n_rows', 'n/a')}")
        md_lines.append(f"- Features modeled: {data_info.get('n_features', 'n/a')} (work factors + lifestyle ordinals)")
        features_list = data_info.get("feature_list")
        if features_list:
            md_lines.append(f"- Feature columns: {', '.join(features_list)}")
        md_lines.append("- Work factors are 1–9 Likert-style scores; exercise_freq_ord, UPF_freq_ord, and sleep_freq_ord are ordinal (higher is better).")
        md_lines.append("- exercise_freq_ord levels (0->4): Rarely/Never, Less than once a week, Once a week, Several days a week, Everyday.")
        md_lines.append("- UPF_freq_ord levels (0->4; higher is healthier): Several times a day, Once a day, Several days a week, A few times a month, Rarely/Never.")
        md_lines.append("- sleep_freq_ord levels (0->3): Hardly ever, Some of the time, Most of the time, All of the time.")
        age_counts = data_info.get("age_counts") or {}
        over24 = data_info.get("age_over24_share")
        if age_counts:
            md_lines.append("- Age groups present (counts):")
            for age_label, count in age_counts.items():
                md_lines.append(f"  - {age_label}: {count}")
        if over24 is not None:
            md_lines.append(f"- Share of rows age >24: {over24:.1%}")
        md_lines.append("")

        # Target descriptions and distributions
        target_desc = data_info.get("target_descriptions") or {}
        target_stats = data_info.get("target_stats") or {}
        if target_desc:
            md_lines.append("### Targets")
            for name, desc in target_desc.items():
                md_lines.append(f"- {name}: {desc}")
            md_lines.append("")
        hist_paths = data_info.get("target_hist_paths") if data_info else None
        if hist_paths:
            md_lines.append("### Target distribution histograms")
            for name, path in hist_paths.items():
                rel = _relative_name(path)
                if rel:
                    md_lines.append(f"![Target distribution - {name}]({rel})")
            md_lines.append("")
        md_lines.append("")
    md_lines.append("## What we modeled")
    md_lines.append("- Targets: overall_mhq_score (higher is better) and productivity_composite (lower is better; represents unproductive days).")
    if USE_EXTENDED_FEATURES:
        md_lines.append("- Features: nine work factors, exercise_freq_ord, UPF_freq_ord, numeric encodings for age, job_duration, organization_size, plus target-encoded employment, team_situation, and country (emitted for segmentation too).")
    else:
        md_lines.append("- Features: nine work factors, exercise_freq_ord, and UPF_freq_ord (no demographic or categorical extensions).")
    md_lines.append("- Scenarios: cap7 (raise factors to at least 7, push lifestyle to healthiest) and full (set factors to max, lifestyle to healthiest).")
    md_lines.append("")

    md_lines.append("## Algorithms and settings")
    if USE_EXTENDED_FEATURES:
        md_lines.append("- Estimator: LightGBM regressor inside a scikit-learn Pipeline with target encoding for categorical features, median imputation, column restoration, then the booster.")
        md_lines.append("- Monotone constraints: +1 for MHQ (higher is better), -1 for productivity (lower is better) on work factors + lifestyle ordinals; neutral (0) on added demographic/work-size/categorical encodings.")
    else:
        md_lines.append("- Estimator: LightGBM regressor inside a scikit-learn Pipeline with median imputation, column restoration, then the booster (no categorical target encoding step).")
        md_lines.append("- Monotone constraints: +1 for MHQ (higher is better), -1 for productivity (lower is better) on work factors + lifestyle ordinals.")
    md_lines.append("- Key hyperparameters: n_estimators=400, learning_rate=0.05, num_leaves=15, max_depth=4, subsample=0.9, colsample_bytree=0.9, min_child_samples=120, min_split_gain=0.1, random_state=42, monotone constraints applied, n_jobs=-1.")
    md_lines.append("")

    md_lines.append("## Evaluation approach")
    md_lines.append("- Cross-validation: 5-fold KFold (shuffle, random_state=42).")
    md_lines.append("- Metrics: MAE and RMSE on out-of-fold predictions.")
    md_lines.append("- Baseline vs. uplift: baseline is the model prediction on original features; counterfactual applies the scenario adjustments; uplift is the difference (sign flips for productivity because lower is better).")
    md_lines.append("")

    md_lines.append("## Predictive performance")
    md_lines.append("| Target | MAE | RMSE | r | Model |")
    md_lines.append("| --- | --- | --- | --- | --- |")
    for target, res in results.items():
        label = _target_label(target)
        md_lines.append(
            f"| {label} | {res['metrics']['mae']:.3f} | {res['metrics']['rmse']:.3f} | {res['metrics']['r']:.3f} | {res['chosen_model']} |"
        )
    md_lines.append("")

    md_lines.append("## Uplift definitions")
    for target in results:
        md_lines.append(f"- {_uplift_definition(target)}")
    md_lines.append("")

    md_lines.append("## Uplift summaries by scenario")
    for target, res in results.items():
        md_lines.append(f"### {_target_label(target)}")
        stats_order = ["min", "10%", "25%", "50%", "75%", "90%", "max", "mean", "std"]
        md_lines.append("| Scenario | " + " | ".join(stats_order) + " |")
        md_lines.append("| --- | " + " | ".join(["---"] * len(stats_order)) + " |")
        for scenario, sres in res["scenarios"].items():
            summary = sres["uplift_summary"]
            scenario_label = scenario + (" (recommended)" if scenario == RECOMMENDED_SCENARIO else "")
            vals = [f"{summary[stat]:.3f}" if stat in summary.index else "" for stat in stats_order]
            md_lines.append("| " + scenario_label + " | " + " | ".join(vals) + " |")
        md_lines.append("")
        md_lines.append("## Uplift band means/medians")
        for target, res in results.items():
            md_lines.append(f"### {_target_label(target)}")
            for scenario, sres in res["scenarios"].items():
                band_stats = sres.get("band_stats")
                if not band_stats:
                    continue
                scenario_label = scenario + (" (recommended)" if scenario == RECOMMENDED_SCENARIO else "")
                md_lines.append(f"#### {scenario_label}")
                md_lines.append("| Band | Mean | Median | Count |")
                md_lines.append("| --- | --- | --- | --- |")
                for band in band_stats:
                    md_lines.append(
                        f"| {band['band']} | {band['mean']:.3f} | {band['median']:.3f} | {band['count']} |"
                    )
                md_lines.append("")

    md_lines.append("## Plots")

    md_lines.append("### Uplift histograms")
    for target, res in results.items():
        hist_path = _relative_name(res.get("hist"))
        if hist_path:
            md_lines.append(f"![Uplift histogram - {_target_label(target)}]({hist_path})")
    md_lines.append("")

    md_lines.append("## Artifacts and outputs")
    md_lines.append("- Scenario predictions: predictions_mhq_cap7.csv, predictions_mhq_full.csv, predictions_productivity_cap7.csv, predictions_productivity_full.csv (all in this folder).")
    md_lines.append("- Consolidated workbook: Work_Model_Uplift_Productivity_MHQ.xlsx (all scenarios).")
    md_lines.append("- Data subset: work_factors_data_subset.xlsx (data used for modeling).")
    md_lines.append("- Plots above are saved alongside this report in the same folder.")
    md_lines.append("")

    md_lines.append("## SHAP summaries")
    for target, res in results.items():
        md_lines.append(f"![SHAP summary - {_target_label(target)}]({ _relative_name(res['shap']) })")
        if res.get("shap_abs"):
            md_lines.append(f"![SHAP impact (mean |SHAP|) - {_target_label(target)}]({ _relative_name(res['shap_abs']) })")
    md_lines.append("")

    if appendix:
        md_lines.append("\n---\n")
        md_lines.append("# Appendix — Extended feature experiment")
        data_diff = appendix.get("data_info", {})
        md_lines.append("## Data differences")
        md_lines.append(f"- Features modeled: {data_diff.get('n_features', 'n/a')} (extended set)")
        feats = data_diff.get("feature_list")
        if feats:
            md_lines.append(f"- Feature columns: {', '.join(feats)}")
        md_lines.append("")

        ext_results: Dict[str, Dict] = appendix["results"]
        md_lines.append("## Predictive performance (extended)")
        md_lines.append("| Target | MAE | RMSE | r | Model |")
        md_lines.append("| --- | --- | --- | --- | --- |")
        for target, res in ext_results.items():
            label = _target_label(target)
            md_lines.append(
                f"| {label} | {res['metrics']['mae']:.3f} | {res['metrics']['rmse']:.3f} | {res['metrics']['r']:.3f} | {res['chosen_model']} |"
            )
        md_lines.append("")

        md_lines.append("## Uplift summaries by scenario (extended)")
        for target, res in ext_results.items():
            md_lines.append(f"### {_target_label(target)}")
            stats_order = ["min", "10%", "25%", "50%", "75%", "90%", "max", "mean", "std"]
            md_lines.append("| Scenario | " + " | ".join(stats_order) + " |")
            md_lines.append("| --- | " + " | ".join(["---"] * len(stats_order)) + " |")
            for scenario, sres in res["scenarios"].items():
                summary = sres["uplift_summary"]
                scenario_label = scenario + (" (recommended)" if scenario == RECOMMENDED_SCENARIO else "")
                vals = [f"{summary[stat]:.3f}" if stat in summary.index else "" for stat in stats_order]
                md_lines.append("| " + scenario_label + " | " + " | ".join(vals) + " |")
            md_lines.append("")

        md_lines.append("## Plots (extended)")
        md_lines.append("### Uplift histograms")
        for target, res in ext_results.items():
            hist_path = _relative_name(res.get("hist"))
            if hist_path:
                md_lines.append(f"![Uplift histogram - {_target_label(target)}]({hist_path})")
        md_lines.append("")

        md_lines.append("### Uplift by age")
        for target, res in ext_results.items():
            age_path = _relative_name(res.get("age_plot"))
            if age_path:
                md_lines.append(f"![Uplift by age - {_target_label(target)}]({age_path})")
        md_lines.append("")

        md_lines.append("## SHAP summaries (extended)")
        for target, res in ext_results.items():
            md_lines.append(f"![SHAP summary - {_target_label(target)}]({ _relative_name(res['shap']) })")
        md_lines.append("")

    md_text = "\n".join(md_lines)
    out_dir.mkdir(exist_ok=True, parents=True)
    report_md = out_dir / "report.md"
    report_md.write_text(md_text)

    report_html = out_dir / "report.html"
    try:
        import markdown  # type: ignore

        html_body = markdown.markdown(md_text, extensions=["tables"])
        report_html.write_text(
            """
<!doctype html>
<html lang=\"en\">
<head><meta charset=\"UTF-8\"><title>{title}</title></head>
<body>
"""
            .format(title=REPORT_TITLE)
            + html_body
            + "\n</body></html>"
        )
    except ImportError:
        md_js = json.dumps(md_text)
        report_html.write_text(
            """
<!doctype html>
<html lang=\"en\">
<head><meta charset=\"UTF-8\"><title>{title}</title><script src=\"https://cdn.jsdelivr.net/npm/marked/marked.min.js\"></script></head>
<body><div id=\"app\"></div><script>const md = """
            + md_js
            + """;document.getElementById('app').innerHTML = marked.parse(md);</script></body></html>
"""
            .format(title=REPORT_TITLE)
        )

    return report_md, report_html


def run_experiment() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    # Core (official) run
    df_core, X_core, y_mhq_core, y_prod_core = load_and_prepare(use_extended=False)
    feature_names_core = list(X_core.columns)
    # Keep original targets aligned to the filtered rows so we can emit them with predictions.
    orig_targets = df_core.loc[X_core.index, ["overall_mhq_score", "productivity_composite", "age", "team_situation", "employment"]]

    target_hist_paths = {
        "overall_mhq_score": plot_target_hist(y_mhq_core, "overall_mhq_score", OUTPUT_DIR),
        "productivity_composite": plot_target_hist(y_prod_core, "productivity_composite", OUTPUT_DIR),
    }

    def _summary(series: pd.Series) -> Dict[str, float]:
        stats = series.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        wanted = {key: float(stats[key]) for key in ["min", "10%", "25%", "50%", "75%", "90%", "max", "mean", "std"]}
        return wanted

    results = {}
    for target_name, y in [("mhq", y_mhq_core), ("productivity", y_prod_core)]:
        # MHQ: higher is better; Productivity (unproductive days): lower is better.
        higher_is_better = target_name == "mhq"
        mono_sign = 1 if higher_is_better else -1
        base_feature_count = len(WORK_FACTOR_COLS) + 3  # work factors + lifestyle ordinals
        extra_feature_count = max(len(feature_names_core) - base_feature_count, 0)
        monotone_list = [mono_sign] * base_feature_count + [0] * extra_feature_count

        model = build_model(
            feature_names_core,
            MODEL_TYPE,
            monotone_sign=mono_sign,
            monotone_constraints=monotone_list,
            categorical_cols=[],
        )
        metrics = evaluate_cv(model, X_core, y)
        model.fit(X_core, y)
        shap_path, shap_abs_path = save_shap_summary(model, X_core, target_name)
        chosen_model = getattr(model, "chosen_model", MODEL_TYPE)

        uplift_frames = []
        scenario_out = {}
        for scenario, floor in SCENARIOS:
            uplift_df = compute_uplift(
                model,
                X_core,
                strategy="full" if scenario == "full" else "cap",
                floor=floor or 7,
                higher_is_better=higher_is_better,
            )
            # Attach originals for reference.
            uplift_df = uplift_df.join(orig_targets)
            uplift_df = uplift_df.reset_index().rename(columns={"index": "row_id"})
            uplift_summary = summarize_uplift(uplift_df["uplift"])
            # Banded stats to expose mean/median within percentile slices
            q = uplift_df["uplift"].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
            bounds = list(q.values)
            band_labels = [
                "0-10%",
                "10-25%",
                "25-50%",
                "50-75%",
                "75-90%",
                "90-100%",
            ]
            band_stats = []
            for i, label in enumerate(band_labels):
                lower = bounds[i]
                upper = bounds[i + 1]
                if i < len(band_labels) - 1:
                    mask = (uplift_df["uplift"] >= lower) & (uplift_df["uplift"] < upper)
                else:
                    mask = (uplift_df["uplift"] >= lower) & (uplift_df["uplift"] <= upper)
                subset = uplift_df.loc[mask, "uplift"]
                band_stats.append(
                    {
                        "band": label,
                        "mean": float(subset.mean()) if len(subset) else float("nan"),
                        "median": float(subset.median()) if len(subset) else float("nan"),
                        "count": int(len(subset)),
                    }
                )
            # Save band stats to CSV for this scenario/target
            band_df = pd.DataFrame(band_stats)
            band_df.insert(0, "scenario", scenario)
            band_df.insert(0, "target", target_name)
            band_csv = OUTPUT_DIR / f"band_stats_{target_name}_{scenario}.csv"
            band_df.to_csv(band_csv, index=False)
            uplift_df.assign(target=target_name, scenario=scenario).to_csv(
                OUTPUT_DIR / f"predictions_{target_name}_{scenario}.csv", index=False
            )
            uplift_frames.append(uplift_df.assign(target=target_name, scenario=scenario))
            scenario_out[scenario] = {
                "uplift_summary": uplift_summary,
                "band_stats": band_stats,
            }

        combined_uplift = pd.concat(uplift_frames, ignore_index=True)
        hist_path = plot_uplift_hist(combined_uplift, target_name)
        age_path = None  # age plots disabled per request
        team_mean = None
        if "team_situation" in combined_uplift.columns:
            team_mean = combined_uplift.groupby(["scenario", "team_situation"])["uplift"].mean().unstack("team_situation")
        emp_path = None
        if target_name == "mhq":
            emp_path = plot_uplift_by_group(combined_uplift, target_name, "employment", "Lift by employment")

        results[target_name] = {
            "metrics": metrics,
            "shap": shap_path,
            "shap_abs": shap_abs_path,
            "scenarios": scenario_out,
            "chosen_model": chosen_model,
            "hist": hist_path,
            "age_plot": age_path,
            "team_mean": team_mean,
            "employment_plot": emp_path,
        }

    # Save a compact text report
    lines = []
    for name, res in results.items():
        lines.append(f"Target: {name} (model: {res['chosen_model']})")
        lines.append(f"  MAE: {res['metrics']['mae']:.3f}")
        lines.append(f"  RMSE: {res['metrics']['rmse']:.3f}")
        lines.append(f"  r: {res['metrics']['r']:.3f}")
        lines.append(f"  SHAP plot: {res['shap']}")
        if res.get("hist"):
            lines.append(f"  Uplift histogram: {res['hist']}")
        if res.get("age_plot"):
            lines.append(f"  Uplift by age: {res['age_plot']}")
        if res.get("team_plot"):
            lines.append(f"  Uplift by team_situation: {res['team_plot']}")
        if res.get("employment_plot"):
            lines.append(f"  Uplift by employment: {res['employment_plot']}")
        for scenario, sres in res["scenarios"].items():
            label = " (recommended)" if scenario == RECOMMENDED_SCENARIO else ""
            lines.append(f"  Scenario: {scenario}{label}")
            summary = sres["uplift_summary"]
            if name == "productivity":
                lines.append("    Uplift summary (base - counterfactual; positive=reduction in unproductive days):")
            else:
                lines.append("    Uplift summary (pred_cf - pred_base):")
            for stat_name in ["min", "10%", "25%", "50%", "75%", "90%", "max", "mean", "std"]:
                if stat_name in summary.index:
                    lines.append(f"      {stat_name}: {summary[stat_name]:.3f}")
        lines.append("")

    report_path = OUTPUT_DIR / "report.txt"
    report_path.write_text("\n".join(lines))

    data_info = {
        "source": str(DATA_PATH),
        "n_rows": len(X_core),
        "n_features": len(feature_names_core),
        "feature_list": feature_names_core,
        "target_descriptions": {
            "overall_mhq_score": "Higher is better mental health score (points).",
            "productivity_composite": "Unproductive days metric (lower is better).",
        },
        "target_stats": {
            "overall_mhq_score": _summary(y_mhq_core),
            "productivity_composite": _summary(y_prod_core),
        },
        "target_hist_paths": target_hist_paths,
        "age_counts": {age: int(count) for age, count in orig_targets["age"].value_counts().sort_index().items()},
        "age_over24_share": float((orig_targets["age"] != "21-24").mean()),
    }

    # Export the data subset for traceability.
    data_subset_path = OUTPUT_DIR / "work_factors_data_subset.xlsx"
    df_core.to_excel(data_subset_path, index=False)

    report_md, report_html = _write_markdown_report(results, OUTPUT_DIR, data_info=data_info)
    print(f"Wrote report to {report_path}")
    print(f"Wrote markdown report to {report_md}")
    print(f"Wrote HTML report to {report_html}")
    algo_report = write_algorithm_report(OUTPUT_DIR)
    print(f"Wrote algorithm settings report to {algo_report}")


if __name__ == "__main__":
    run_experiment()
