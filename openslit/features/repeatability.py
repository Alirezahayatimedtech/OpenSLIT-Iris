"""Repeatability, ICC, coefficient-of-variation, and Bland-Altman summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NON_FEATURE_COLUMNS = {
    "image_id",
    "image_file",
    "mask_file",
    "blinded_patient_id",
    "patient_id",
    "subject_id",
    "eye",
    "laterality",
    "session_id",
    "repeat_group_id",
    "feature_version",
    "feature_source",
    "feature_gate_passed",
    "feature_gate_flags",
    "source_review_status",
    "source_gradable",
    "source_mask_hash_present",
}


def _icc_2_1(matrix: np.ndarray) -> float | None:
    """Two-way random-effects, absolute-agreement, single-measure ICC(2,1)."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None
    if not np.isfinite(matrix).all():
        return None
    n, k = matrix.shape
    grand = float(matrix.mean())
    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    ss_rows = k * float(((row_means - grand) ** 2).sum())
    ss_columns = n * float(((column_means - grand) ** 2).sum())
    ss_total = float(((matrix - grand) ** 2).sum())
    ss_error = ss_total - ss_rows - ss_columns
    ms_rows = ss_rows / (n - 1)
    ms_columns = ss_columns / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    if denominator == 0:
        return None
    return float((ms_rows - ms_error) / denominator)


def _balanced_matrix(
    data: pd.DataFrame,
    group_column: str,
    feature: str,
) -> np.ndarray | None:
    grouped = []
    repeat_count: int | None = None
    for _, group in data.groupby(group_column, sort=True):
        values = (
            pd.to_numeric(group[feature], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(values) < 2:
            continue
        if repeat_count is None:
            repeat_count = len(values)
        if len(values) != repeat_count:
            continue
        grouped.append(values)
    if len(grouped) < 2:
        return None
    return np.vstack(grouped)


def _within_group_cv(
    data: pd.DataFrame,
    group_column: str,
    feature: str,
) -> float | None:
    cvs: list[float] = []
    for _, group in data.groupby(group_column):
        values = (
            pd.to_numeric(group[feature], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(values) < 2:
            continue
        mean = float(np.mean(values))
        if mean == 0:
            continue
        cvs.append(float(np.std(values, ddof=1) / abs(mean)))
    return float(np.mean(cvs)) if cvs else None


def _repeatability_coefficient(
    data: pd.DataFrame,
    group_column: str,
    feature: str,
) -> float | None:
    differences: list[float] = []
    for _, group in data.groupby(group_column):
        values = (
            pd.to_numeric(group[feature], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(values) == 2:
            differences.append(float(values[1] - values[0]))
    if len(differences) < 2:
        return None
    return float(1.96 * np.std(differences, ddof=1))


def _bland_altman(
    data: pd.DataFrame,
    group_column: str,
    feature: str,
) -> dict[str, Any] | None:
    means: list[float] = []
    differences: list[float] = []
    for _, group in data.groupby(group_column):
        values = (
            pd.to_numeric(group[feature], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(values) != 2:
            continue
        means.append(float(values.mean()))
        differences.append(float(values[1] - values[0]))
    if len(differences) < 2:
        return None
    bias = float(np.mean(differences))
    standard_deviation = float(np.std(differences, ddof=1))
    return {
        "feature": feature,
        "pairs": len(differences),
        "bias": bias,
        "lower_limit": bias - 1.96 * standard_deviation,
        "upper_limit": bias + 1.96 * standard_deviation,
        "mean_measurement": float(np.mean(means)),
    }


def numeric_feature_columns(
    data: pd.DataFrame,
    extra_exclude: set[str] | None = None,
) -> list[str]:
    excluded = NON_FEATURE_COLUMNS | (extra_exclude or set())
    selected: list[str] = []
    for column in data.columns:
        if column in excluded:
            continue
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.notna().sum() >= 2:
            selected.append(column)
    return selected


def analyze_repeatability(
    feature_table_path: Path,
    group_column: str,
    output_dir: Path,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    data = pd.read_csv(feature_table_path, dtype=str, keep_default_na=False)
    if group_column not in data.columns:
        raise ValueError(f"Feature table has no repeat-group column {group_column!r}")
    if "feature_gate_passed" in data.columns:
        data = data[
            data["feature_gate_passed"].str.lower().isin({"true", "1", "yes"})
        ].copy()
    if data.empty:
        raise ValueError("No quality-approved feature rows are available")
    features = feature_columns or numeric_feature_columns(data, {group_column})
    if not features:
        raise ValueError("No numeric feature columns were found")

    rows: list[dict[str, Any]] = []
    bland_rows: list[dict[str, Any]] = []
    for feature in features:
        matrix = _balanced_matrix(data, group_column, feature)
        icc = None if matrix is None else _icc_2_1(matrix)
        values = pd.to_numeric(data[feature], errors="coerce")
        valid_group_data = data.assign(_value=values).dropna(subset=["_value"])
        repeat_groups = int(
            valid_group_data.groupby(group_column)
            .filter(lambda frame: len(frame) >= 2)[group_column]
            .nunique()
        )
        rows.append(
            {
                "feature": feature,
                "observations": int(values.notna().sum()),
                "repeat_groups": repeat_groups,
                "icc_2_1": icc,
                "mean_within_group_cv": _within_group_cv(
                    data,
                    group_column,
                    feature,
                ),
                "repeatability_coefficient": _repeatability_coefficient(
                    data,
                    group_column,
                    feature,
                ),
            }
        )
        bland = _bland_altman(data, group_column, feature)
        if bland is not None:
            bland_rows.append(bland)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "repeatability_summary.csv"
    bland_path = output_dir / "bland_altman_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    pd.DataFrame(bland_rows).to_csv(bland_path, index=False)
    summary = {
        "group_column": group_column,
        "features": len(rows),
        "rows": len(data),
        "summary_path": str(summary_path),
        "bland_altman_path": str(bland_path),
    }
    (output_dir / "repeatability_run.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
