"""Human-factors design for comparing blank and AI-assisted annotation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AIWorkflowConfig


CORRECTION_CATEGORIES = {
    "ACCEPTED_WITHOUT_CHANGE",
    "MINOR_CORRECTION",
    "MAJOR_CORRECTION",
    "REJECTED_AND_REDRAWN",
    "UNGRADABLE",
    "SEND_TO_SENIOR",
}


def create_crossover_plan(
    config: AIWorkflowConfig,
    batch_manifest_path: Path,
    grader_ids: tuple[str, str],
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Randomize each image to blank or AI-assisted mode for each grader.

    Each image is annotated once in each mode across the two graders. This avoids
    exposing one grader to both versions of the same image while balancing arms.
    """

    data = pd.read_csv(batch_manifest_path, dtype=str, keep_default_na=False)
    required = {"image_id", "image_file"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Batch manifest is missing columns: {sorted(missing)}")
    if data["image_id"].duplicated().any():
        raise ValueError("Crossover batch contains duplicate image IDs")
    if len(grader_ids) != 2 or grader_ids[0] == grader_ids[1]:
        raise ValueError("Exactly two unique grader IDs are required")

    rng = np.random.default_rng(config.random_seed)
    order = np.arange(len(data))
    rng.shuffle(order)
    rows: list[dict[str, Any]] = []
    for rank, row_index in enumerate(order, start=1):
        row = data.iloc[int(row_index)]
        first_arm = "AI_ASSISTED" if rank % 2 else "MANUAL_BLANK"
        second_arm = "MANUAL_BLANK" if first_arm == "AI_ASSISTED" else "AI_ASSISTED"
        for grader_id, arm in zip(grader_ids, [first_arm, second_arm]):
            rows.append(
                {
                    "image_id": str(row["image_id"]),
                    "image_file": str(row["image_file"]),
                    "grader_id": grader_id,
                    "arm": arm,
                    "assignment_order": rank,
                    "active_annotation_seconds": "",
                    "correction_category": "",
                    "sent_to_senior": "",
                    "final_mask_file": "",
                    "notes": "",
                }
            )
    plan = pd.DataFrame(rows).sort_values(["grader_id", "assignment_order"])
    output_path = output_path or (
        config.output_dir / "human_factors" / "crossover_plan.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output_path, index=False)
    summary = {
        "images": len(data),
        "assignments": len(plan),
        "graders": list(grader_ids),
        "arms": plan["arm"].value_counts().to_dict(),
        "output_path": str(output_path),
        "seed": config.random_seed,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def summarize_crossover_results(
    completed_plan_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Summarize time, corrections, and referrals by annotation arm."""

    data = pd.read_csv(completed_plan_path, dtype=str, keep_default_na=False)
    required = {
        "image_id",
        "grader_id",
        "arm",
        "active_annotation_seconds",
        "correction_category",
        "sent_to_senior",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Completed crossover table is missing: {sorted(missing)}")
    allowed_arms = {"MANUAL_BLANK", "AI_ASSISTED"}
    invalid_arms = sorted(set(data["arm"]) - allowed_arms)
    if invalid_arms:
        raise ValueError(f"Unknown crossover arms: {invalid_arms}")
    invalid_categories = sorted(
        set(data["correction_category"]) - CORRECTION_CATEGORIES
    )
    if invalid_categories:
        raise ValueError(f"Unknown correction categories: {invalid_categories}")
    seconds = pd.to_numeric(data["active_annotation_seconds"], errors="coerce")
    if seconds.isna().any() or (seconds < 0).any():
        raise ValueError("Every assignment needs a non-negative active annotation time")
    data = data.copy()
    data["active_annotation_seconds"] = seconds
    data["sent_to_senior_bool"] = data["sent_to_senior"].str.lower().isin(
        {"true", "1", "yes"}
    )
    data["major_or_redraw"] = data["correction_category"].isin(
        {"MAJOR_CORRECTION", "REJECTED_AND_REDRAWN"}
    )

    arm_summary = (
        data.groupby("arm")
        .agg(
            assignments=("image_id", "count"),
            median_seconds=("active_annotation_seconds", "median"),
            mean_seconds=("active_annotation_seconds", "mean"),
            senior_referral_rate=("sent_to_senior_bool", "mean"),
            major_or_redraw_rate=("major_or_redraw", "mean"),
        )
        .reset_index()
    )
    category_summary = (
        data.groupby(["arm", "correction_category"])
        .size()
        .rename("count")
        .reset_index()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_path = output_dir / "arm_summary.csv"
    category_path = output_dir / "correction_categories.csv"
    arm_summary.to_csv(arm_path, index=False)
    category_summary.to_csv(category_path, index=False)
    summary = {
        "assignments": len(data),
        "arm_summary": arm_summary.to_dict(orient="records"),
        "arm_summary_path": str(arm_path),
        "correction_categories_path": str(category_path),
    }
    (output_dir / "productivity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
