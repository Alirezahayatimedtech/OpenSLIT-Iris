"""Patient-level train, validation, and untouched test split generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AIWorkflowConfig


def _load_consensus(config: AIWorkflowConfig) -> pd.DataFrame:
    data = pd.read_csv(config.consensus_manifest_path, dtype=str, keep_default_na=False)
    required = {"image_id", "image_file", "mask_file", config.split.group_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Consensus manifest is missing columns: {sorted(missing)}")
    if data["image_id"].duplicated().any():
        raise ValueError("Consensus manifest contains duplicate image_id values")
    if data[config.split.group_column].astype(str).str.strip().eq("").any():
        raise ValueError(f"Consensus manifest has blank {config.split.group_column} values")
    return data.copy()


def create_grouped_splits(
    config: AIWorkflowConfig,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Create deterministic group-level splits with no participant leakage."""

    data = _load_consensus(config)
    groups = sorted(data[config.split.group_column].unique().tolist())
    rng = np.random.default_rng(config.random_seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)

    n_groups = len(shuffled)
    if n_groups < 3:
        raise ValueError("At least three independent groups are required for train/val/test")
    n_train = max(1, int(round(n_groups * config.split.train_fraction)))
    n_validation = max(1, int(round(n_groups * config.split.validation_fraction)))
    if n_train + n_validation >= n_groups:
        n_validation = 1
        n_train = n_groups - 2

    assignments: dict[str, str] = {}
    for group in shuffled[:n_train]:
        assignments[group] = "train"
    for group in shuffled[n_train : n_train + n_validation]:
        assignments[group] = "validation"
    for group in shuffled[n_train + n_validation :]:
        assignments[group] = "test"

    data["split"] = data[config.split.group_column].map(assignments)
    if data["split"].isna().any():
        raise AssertionError("Some groups were not assigned to a split")

    output_path = output_path or (config.output_dir / "split_manifest.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

    summary = {
        "seed": config.random_seed,
        "group_column": config.split.group_column,
        "manifest_path": str(output_path),
        "images": data["split"].value_counts().sort_index().to_dict(),
        "groups": data.groupby("split")[config.split.group_column].nunique().to_dict(),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def verify_split_manifest(path: Path, group_column: str) -> dict[str, Any]:
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"image_id", "split", group_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    allowed = {"train", "validation", "test"}
    observed = set(data["split"])
    if not observed <= allowed:
        raise ValueError(f"Unknown split labels: {sorted(observed - allowed)}")
    group_counts = data.groupby(group_column)["split"].nunique()
    leaking = sorted(group_counts[group_counts > 1].index.tolist())
    if leaking:
        raise ValueError(f"Participant/group leakage detected: {leaking[:20]}")
    return {
        "images": data["split"].value_counts().sort_index().to_dict(),
        "groups": data.groupby("split")[group_column].nunique().to_dict(),
        "leakage": False,
    }
