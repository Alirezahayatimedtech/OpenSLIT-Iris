"""Balanced active-learning selection for the next annotation batch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AIWorkflowConfig


def _quota(total: int, fraction: float) -> int:
    return max(0, int(round(total * fraction)))


def _normalize(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if maximum <= minimum:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return (numeric - minimum) / (maximum - minimum)


def _farthest_first(
    candidates: pd.DataFrame,
    embeddings: np.ndarray,
    count: int,
    seed: int,
) -> list[int]:
    if count <= 0 or len(candidates) == 0:
        return []
    if embeddings.shape[0] != len(candidates):
        raise ValueError("Embedding rows must match candidate rows")
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(0, len(candidates)))]
    distances = np.linalg.norm(embeddings - embeddings[selected[0]], axis=1)
    while len(selected) < min(count, len(candidates)):
        next_index = int(np.argmax(distances))
        if next_index in selected:
            break
        selected.append(next_index)
        distances = np.minimum(
            distances,
            np.linalg.norm(embeddings - embeddings[next_index], axis=1),
        )
    return selected


def select_active_learning_batch(
    config: AIWorkflowConfig,
    candidate_table_path: Path,
    output_path: Path | None = None,
    embeddings_path: Path | None = None,
) -> dict[str, Any]:
    """Select a mixed batch using uncertainty, committee disagreement, diversity and random controls.

    Required columns: image_id, split, labelled, uncertainty_score,
    model_disagreement_score. Optional quality/artifact columns are retained for audit.
    """

    data = pd.read_csv(candidate_table_path, dtype=str, keep_default_na=False)
    required = {
        "image_id",
        "split",
        "labelled",
        "uncertainty_score",
        "model_disagreement_score",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")
    if data["image_id"].duplicated().any():
        raise ValueError("Candidate table contains duplicate image IDs")

    available = data[
        ~data["labelled"].str.strip().str.lower().isin({"true", "1", "yes"})
    ].copy()
    if config.active_learning.exclude_test_set:
        available = available[available["split"] != "test"].copy()
    if available.empty:
        raise ValueError("No eligible unlabelled images remain")

    batch_size = min(config.active_learning.batch_size, len(available))
    uncertainty_n = _quota(batch_size, config.active_learning.uncertainty_fraction)
    disagreement_n = _quota(
        batch_size, config.active_learning.model_disagreement_fraction
    )
    diversity_n = _quota(batch_size, config.active_learning.diversity_fraction)
    random_n = batch_size - uncertainty_n - disagreement_n - diversity_n
    random_n = max(random_n, config.active_learning.minimum_random_images)
    while uncertainty_n + disagreement_n + diversity_n + random_n > batch_size:
        for name in ["uncertainty", "disagreement", "diversity"]:
            if name == "uncertainty" and uncertainty_n > 0:
                uncertainty_n -= 1
                break
            if name == "disagreement" and disagreement_n > 0:
                disagreement_n -= 1
                break
            if name == "diversity" and diversity_n > 0:
                diversity_n -= 1
                break
        else:
            random_n -= 1

    available["uncertainty_norm"] = _normalize(available["uncertainty_score"])
    available["disagreement_norm"] = _normalize(
        available["model_disagreement_score"]
    )
    selected_ids: set[str] = set()
    rows: list[pd.DataFrame] = []

    def take_ranked(column: str, count: int, reason: str) -> None:
        remaining = available[~available["image_id"].isin(selected_ids)]
        chosen = remaining.sort_values(column, ascending=False).head(count).copy()
        if not chosen.empty:
            chosen["selection_reason"] = reason
            selected_ids.update(chosen["image_id"])
            rows.append(chosen)

    take_ranked("uncertainty_norm", uncertainty_n, "highest_uncertainty")
    take_ranked("disagreement_norm", disagreement_n, "model_disagreement")

    remaining = available[~available["image_id"].isin(selected_ids)].copy()
    if diversity_n > 0 and not remaining.empty:
        if embeddings_path is not None:
            embedding_table = pd.read_csv(embeddings_path, dtype=str, keep_default_na=False)
            if "image_id" not in embedding_table.columns:
                raise ValueError("Embeddings table must contain image_id")
            feature_columns = [column for column in embedding_table.columns if column != "image_id"]
            merged = remaining.merge(embedding_table, on="image_id", how="inner")
            if len(merged) != len(remaining):
                raise ValueError("Embeddings are missing eligible candidate images")
            embeddings = merged[feature_columns].astype(float).to_numpy()
            indices = _farthest_first(
                merged,
                embeddings,
                diversity_n,
                config.random_seed,
            )
            chosen = merged.iloc[indices][remaining.columns].copy()
        else:
            combined = remaining["uncertainty_norm"] + remaining["disagreement_norm"]
            chosen = remaining.assign(_combined=combined).sort_values(
                "_combined", ascending=True
            ).head(diversity_n).drop(columns="_combined")
        chosen["selection_reason"] = "diversity"
        selected_ids.update(chosen["image_id"])
        rows.append(chosen)

    remaining = available[~available["image_id"].isin(selected_ids)].copy()
    if len(selected_ids) < batch_size and not remaining.empty:
        needed = batch_size - len(selected_ids)
        chosen = remaining.sample(
            n=min(needed, len(remaining)),
            random_state=config.random_seed,
        ).copy()
        chosen["selection_reason"] = "random_control"
        selected_ids.update(chosen["image_id"])
        rows.append(chosen)

    selected = pd.concat(rows, ignore_index=True).head(batch_size)
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    output_path = output_path or (config.output_dir / "active_learning" / "next_batch.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)
    summary = {
        "batch_size": len(selected),
        "output_path": str(output_path),
        "selection_reasons": selected["selection_reason"].value_counts().to_dict(),
        "test_set_excluded": config.active_learning.exclude_test_set,
        "seed": config.random_seed,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
