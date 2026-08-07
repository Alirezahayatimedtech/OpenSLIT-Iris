"""Benchmark AI and human masks against the frozen senior consensus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from openslit.annotation.schema import load_annotation_schema

from .config import AIWorkflowConfig
from .metrics import multiclass_metrics


def _load_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        mask = np.asarray(image)
    if mask.ndim != 2:
        raise ValueError(f"Expected indexed 2-D mask at {path}; got {mask.shape}")
    return mask.astype(np.uint8, copy=False)


def _source_manifest(path: Path, source_name: str) -> pd.DataFrame:
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"image_id", "mask_file"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"{source_name} manifest is missing columns: {sorted(missing)}"
        )
    if data["image_id"].duplicated().any():
        raise ValueError(f"{source_name} manifest has duplicate image IDs")
    columns = ["image_id", "mask_file"]
    if "split" in data.columns:
        columns.append("split")
    return data[columns].rename(columns={"mask_file": "source_mask_file"})


def compare_source_to_consensus(
    config: AIWorkflowConfig,
    source_name: str,
    source_manifest_path: Path,
    source_masks_dir: Path,
    consensus_masks_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Compare one source (AI or human) against senior consensus masks."""

    schema = load_annotation_schema(config.schema_path)
    consensus = pd.read_csv(
        config.consensus_manifest_path,
        dtype=str,
        keep_default_na=False,
    )
    required = {"image_id", "mask_file"}
    missing = required - set(consensus.columns)
    if missing:
        raise ValueError(f"Consensus manifest is missing columns: {sorted(missing)}")
    if consensus["image_id"].duplicated().any():
        raise ValueError("Consensus manifest has duplicate image IDs")
    source = _source_manifest(source_manifest_path, source_name)
    if source.empty:
        raise ValueError(f"{source_name} manifest contains no images")
    unknown_ids = sorted(set(source["image_id"]) - set(consensus["image_id"]))
    if unknown_ids:
        raise ValueError(
            f"{source_name} contains images absent from consensus: {unknown_ids[:20]}"
        )
    joined = source.merge(
        consensus,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    split_value: str | None = None
    split_counts: dict[str, int] = {}
    if "split" in source.columns:
        split_labels = source["split"].astype(str).str.strip()
        if split_labels.eq("").any():
            raise ValueError(f"{source_name} manifest contains blank split labels")
        split_counts = {
            str(key): int(value)
            for key, value in split_labels.value_counts().sort_index().items()
        }
        split_value = next(iter(split_counts)) if len(split_counts) == 1 else "mixed"

    safe_source = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in source_name
    )
    output_dir = output_dir or (config.output_dir / "benchmarks" / safe_source)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    class_ids = sorted(schema.class_ids)

    for row in joined.itertuples(index=False):
        image_id = str(row.image_id)
        reference = _load_mask(consensus_masks_dir / str(row.mask_file))
        prediction = _load_mask(source_masks_dir / str(row.source_mask_file))
        if reference.shape != prediction.shape:
            raise ValueError(
                f"Mask dimensions differ for {image_id}: "
                f"reference={reference.shape}, source={prediction.shape}"
            )
        unknown_reference = sorted(set(np.unique(reference).tolist()) - set(class_ids))
        unknown_prediction = sorted(
            set(np.unique(prediction).tolist()) - set(class_ids)
        )
        if unknown_reference or unknown_prediction:
            raise ValueError(
                f"Unknown mask class IDs for {image_id}: "
                f"reference={unknown_reference}, source={unknown_prediction}"
            )
        per_class = multiclass_metrics(reference, prediction, class_ids)
        foreground_dice = [
            values["dice"] for class_id, values in per_class.items() if class_id != 0
        ]
        image_rows.append(
            {
                "image_id": image_id,
                "source": source_name,
                "macro_foreground_dice": float(np.mean(foreground_dice)),
                "minimum_foreground_dice": float(np.min(foreground_dice)),
                "requires_expert_review": bool(np.min(foreground_dice) < 0.70),
            }
        )
        for class_id, values in per_class.items():
            item = schema.class_by_id[class_id]
            class_rows.append(
                {
                    "image_id": image_id,
                    "source": source_name,
                    "class_id": class_id,
                    "class_name": item.name,
                    **values,
                }
            )

    image_table = pd.DataFrame(image_rows)
    class_table = pd.DataFrame(class_rows)
    image_path = output_dir / "image_metrics.csv"
    class_path = output_dir / "class_metrics.csv"
    image_table.to_csv(image_path, index=False)
    class_table.to_csv(class_path, index=False)

    summary = {
        "source": source_name,
        "reference": "senior_consensus",
        "split": split_value,
        "split_counts": split_counts,
        "images": len(image_table),
        "macro_foreground_dice_mean": float(
            image_table["macro_foreground_dice"].mean()
        ),
        "macro_foreground_dice_median": float(
            image_table["macro_foreground_dice"].median()
        ),
        "expert_review_images": int(image_table["requires_expert_review"].sum()),
        "class_summary": class_table.groupby(["class_id", "class_name"])[
            ["dice", "iou", "precision", "recall"]
        ]
        .mean()
        .reset_index()
        .to_dict(orient="records"),
        "image_metrics_path": str(image_path),
        "class_metrics_path": str(class_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_comparison_matrix(
    summaries: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Combine AI-vs-consensus and human-vs-consensus summaries."""

    rows = []
    for path in summaries:
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "source": raw["source"],
                "reference": raw["reference"],
                "split": raw.get("split"),
                "images": raw["images"],
                "macro_foreground_dice_mean": raw["macro_foreground_dice_mean"],
                "macro_foreground_dice_median": raw["macro_foreground_dice_median"],
                "expert_review_images": raw["expert_review_images"],
            }
        )
    table = pd.DataFrame(rows).sort_values("source")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return {"rows": len(table), "output_path": str(output_path)}
