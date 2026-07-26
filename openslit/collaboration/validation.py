"""Validate blinded grader submissions and merge independent reviews."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    ALLOWED_VALUES,
    FORBIDDEN_SHARED_COLUMNS,
    REFERENCE_COLUMNS,
    REQUIRED_RESPONSE_COLUMNS,
    RESPONSE_COLUMNS,
)
from .workbook import read_grading_table


def validate_submission(
    path: Path, expected_index: Path, require_complete: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    data = read_grading_table(path)
    expected = pd.read_csv(expected_index, dtype=str, keep_default_na=False)
    errors: list[str] = []
    required_columns = set(REFERENCE_COLUMNS + RESPONSE_COLUMNS)
    missing = required_columns - set(data.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")
        return data, errors

    forbidden = FORBIDDEN_SHARED_COLUMNS.intersection(data.columns)
    if forbidden:
        errors.append(f"Forbidden source columns present: {sorted(forbidden)}")

    if data["blinded_image_id"].duplicated().any():
        values = data.loc[
            data["blinded_image_id"].duplicated(keep=False), "blinded_image_id"
        ].tolist()
        errors.append(f"Duplicate blinded image IDs: {values}")

    expected_ids = set(expected["blinded_image_id"])
    observed_ids = set(data["blinded_image_id"])
    if expected_ids != observed_ids:
        errors.append(
            "Submission image IDs differ from pilot index: "
            f"missing={sorted(expected_ids - observed_ids)}, "
            f"unexpected={sorted(observed_ids - expected_ids)}"
        )

    expected_reference = expected.set_index("blinded_image_id")
    for _, row in data.iterrows():
        image_id = row["blinded_image_id"]
        if image_id not in expected_reference.index:
            continue
        for column in REFERENCE_COLUMNS:
            if column == "blinded_image_id":
                continue
            expected_value = str(expected_reference.loc[image_id, column])
            observed_value = str(row[column])
            # A blank URL in the frozen index permits the custodian to add a
            # Drive link after upload without changing image identity.
            if column == "drive_url" and expected_value == "":
                continue
            if observed_value != expected_value:
                errors.append(
                    f"{image_id}: locked reference field changed: {column}"
                )

    for column, allowed in ALLOWED_VALUES.items():
        invalid = sorted(
            set(data.loc[~data[column].isin(allowed + [""]), column].astype(str))
        )
        if invalid:
            errors.append(f"{column}: invalid values {invalid}")

    if require_complete:
        for column in REQUIRED_RESPONSE_COLUMNS:
            blank_ids = data.loc[
                data[column].astype(str).str.strip().eq(""), "blinded_image_id"
            ].tolist()
            if blank_ids:
                errors.append(f"{column}: blank for {blank_ids}")

    grader_ids = set(data["grader_id"].astype(str).str.strip()) - {""}
    if len(grader_ids) != 1:
        errors.append(f"Expected exactly one grader_id, found {sorted(grader_ids)}")

    for _, row in data.iterrows():
        value = str(row["review_date_yyyy_mm_dd"]).strip()
        if not value:
            continue
        try:
            date.fromisoformat(value)
        except ValueError:
            errors.append(
                f"{row['blinded_image_id']}: invalid review date {value!r}"
            )

    return data, errors


def weighted_kappa(values_a: list[str], values_b: list[str], order: list[str]) -> float:
    if len(values_a) != len(values_b) or not values_a:
        return float("nan")
    index = {value: position for position, value in enumerate(order)}
    pairs = [(index[a], index[b]) for a, b in zip(values_a, values_b)]
    size = len(order)
    observed = np.zeros((size, size), dtype=float)
    for a, b in pairs:
        observed[a, b] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    if size == 1:
        return 1.0
    weights = np.fromfunction(
        lambda i, j: ((i - j) / (size - 1)) ** 2, (size, size)
    )
    numerator = float((weights * observed).sum())
    denominator = float((weights * expected).sum())
    return 1.0 if denominator == 0 and numerator == 0 else 1 - numerator / denominator


def merge_submissions(
    first_path: Path,
    second_path: Path,
    expected_index: Path,
    output_dir: Path,
) -> dict[str, object]:
    first, first_errors = validate_submission(first_path, expected_index, True)
    second, second_errors = validate_submission(second_path, expected_index, True)
    if first_errors or second_errors:
        raise ValueError(
            json.dumps(
                {"first_errors": first_errors, "second_errors": second_errors},
                indent=2,
            )
        )
    first = first.sort_values("blinded_image_id").reset_index(drop=True)
    second = second.sort_values("blinded_image_id").reset_index(drop=True)
    if first["grader_id"].iloc[0] == second["grader_id"].iloc[0]:
        raise ValueError("Independent submissions must use different grader IDs")

    output_dir.mkdir(parents=True, exist_ok=True)
    long = pd.concat([first, second], ignore_index=True)
    long.to_csv(output_dir / "grades_long.csv", index=False)

    consensus = first[REFERENCE_COLUMNS].copy()
    disagreement_fields = [
        "acquisition_eligible",
        "quality_grade",
        "segmentation_feasibility",
        "include_for_mask_annotation",
    ]
    for field in disagreement_fields:
        consensus[f"{field}_grader_1"] = first[field]
        consensus[f"{field}_grader_2"] = second[field]
        consensus[f"{field}_agreement"] = first[field].eq(second[field])
    consensus["requires_adjudication"] = ~consensus[
        [f"{field}_agreement" for field in disagreement_fields]
    ].all(axis=1)
    consensus["adjudicated_acquisition_eligible"] = ""
    consensus["adjudicated_quality_grade"] = ""
    consensus["adjudicated_segmentation_feasibility"] = ""
    consensus["adjudicated_include_for_mask_annotation"] = ""
    consensus["adjudicator_id"] = ""
    consensus["adjudication_date_yyyy_mm_dd"] = ""
    consensus["adjudication_notes"] = ""
    consensus.to_csv(output_dir / "adjudication_queue.csv", index=False)

    metrics: dict[str, object] = {
        "n_images": len(first),
        "grader_1": first["grader_id"].iloc[0],
        "grader_2": second["grader_id"].iloc[0],
        "requires_adjudication": int(consensus["requires_adjudication"].sum()),
        "agreement": {},
    }
    for field in disagreement_fields:
        metrics["agreement"][field] = float(first[field].eq(second[field]).mean())
    metrics["quality_grade_quadratic_weighted_kappa"] = weighted_kappa(
        first["quality_grade"].tolist(),
        second["quality_grade"].tolist(),
        ["D", "C", "B", "A"],
    )
    (output_dir / "agreement_summary.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    return metrics
