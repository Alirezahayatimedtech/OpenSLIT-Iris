"""Validate OpenSLIT-Iris indexed-PNG annotation masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .schema import AnnotationSchema, load_annotation_schema


def _truthy(value: object, default: bool = True) -> bool:
    text = str(value).strip().lower()
    if text == "" or text == "nan":
        return default
    if text in {"1", "true", "yes", "y", "gradable"}:
        return True
    if text in {"0", "false", "no", "n", "ungradable"}:
        return False
    raise ValueError(f"Unrecognized boolean value: {value!r}")


def _validate_manifest(manifest: pd.DataFrame, schema: AnnotationSchema) -> list[str]:
    errors: list[str] = []
    missing = set(schema.required_manifest_columns) - set(manifest.columns)
    if missing:
        errors.append(f"Missing required manifest columns: {sorted(missing)}")
    forbidden = set(schema.forbidden_shared_fields).intersection(manifest.columns)
    if forbidden:
        errors.append(f"Forbidden shared fields are present: {sorted(forbidden)}")
    if "image_id" in manifest.columns and manifest["image_id"].duplicated().any():
        duplicate_ids = sorted(manifest.loc[manifest["image_id"].duplicated(), "image_id"].unique())
        errors.append(f"Duplicate image_id values: {duplicate_ids}")
    return errors


def validate_one(
    row: pd.Series,
    schema: AnnotationSchema,
    images_dir: Path,
    masks_dir: Path,
) -> list[dict[str, object]]:
    image_id = str(row.get("image_id", ""))
    issues: list[dict[str, object]] = []

    def add(level: str, code: str, message: str) -> None:
        issues.append(
            {
                "image_id": image_id,
                "level": level,
                "code": code,
                "message": message,
            }
        )

    image_path = images_dir / str(row["image_file"])
    mask_path = masks_dir / str(row["mask_file"])

    if not image_path.is_file():
        add("error", "missing_image", str(image_path))
        return issues
    if not mask_path.is_file():
        add("error", "missing_mask", str(mask_path))
        return issues

    try:
        with Image.open(image_path) as image:
            image_size = image.size
    except Exception as exc:
        add("error", "unreadable_image", str(exc))
        return issues

    try:
        with Image.open(mask_path) as mask_image:
            mask_size = mask_image.size
            mask = np.asarray(mask_image)
    except Exception as exc:
        add("error", "unreadable_mask", str(exc))
        return issues

    if mask.ndim == 3:
        add(
            "error",
            "non_indexed_mask",
            "Mask must be a single-channel indexed PNG, not RGB/RGBA.",
        )
        return issues

    if image_size != mask_size:
        add(
            "error",
            "dimension_mismatch",
            f"Image size {image_size} differs from mask size {mask_size}.",
        )

    unique_ids, counts = np.unique(mask, return_counts=True)
    observed = {int(value): int(count) for value, count in zip(unique_ids, counts)}
    invalid_ids = sorted(set(observed) - set(schema.class_ids))
    if invalid_ids:
        add("error", "invalid_class_ids", f"Unexpected mask values: {invalid_ids}")

    gradable = True
    try:
        gradable = _truthy(row.get("gradable", ""), default=True)
    except ValueError as exc:
        add("error", "invalid_gradable_value", str(exc))

    if gradable:
        missing_required = sorted(schema.required_class_ids - set(observed))
        if missing_required:
            add(
                "error",
                "missing_required_classes",
                f"Gradable mask is missing class IDs: {missing_required}",
            )

    expected_version = schema.protocol_version
    actual_version = str(row.get("protocol_version", "")).strip()
    if actual_version != expected_version:
        add(
            "error",
            "protocol_version_mismatch",
            f"Expected {expected_version!r}; found {actual_version!r}.",
        )

    if not str(row.get("annotator_id", "")).strip():
        add("error", "missing_annotator_id", "annotator_id is empty.")

    pupil_id = schema.class_by_name["pupil"].id
    iris_id = schema.class_by_name["iris"].id
    pupil_area = observed.get(pupil_id, 0)
    iris_area = observed.get(iris_id, 0)
    total = int(mask.size)

    if pupil_area > iris_area and iris_area > 0:
        add(
            "warning",
            "pupil_larger_than_iris",
            "Pupil pixel area exceeds visible iris pixel area; review the mask.",
        )
    if pupil_area / total > 0.5:
        add("warning", "large_pupil_fraction", "Pupil occupies more than 50% of the image.")
    if iris_area / total > 0.9:
        add("warning", "large_iris_fraction", "Iris occupies more than 90% of the image.")

    if not issues:
        add("ok", "valid", "Mask passed all implemented checks.")
    return issues


def validate_dataset(
    schema_path: Path,
    manifest_path: Path,
    images_dir: Path,
    masks_dir: Path,
    report_path: Path,
) -> dict[str, int]:
    schema = load_annotation_schema(schema_path)
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    manifest_errors = _validate_manifest(manifest, schema)
    if manifest_errors:
        raise ValueError("\n".join(manifest_errors))

    records: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        records.extend(validate_one(row, schema, images_dir, masks_dir))

    report = pd.DataFrame(records)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    summary = {
        "images": int(len(manifest)),
        "errors": int((report["level"] == "error").sum()),
        "warnings": int((report["level"] == "warning").sum()),
        "valid_images": int(report.loc[report["level"] == "ok", "image_id"].nunique()),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openslit-validate-masks",
        description="Validate OpenSLIT-Iris annotation manifests and indexed PNG masks.",
    )
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = validate_dataset(
        args.schema, args.manifest, args.images, args.masks, args.report
    )
    print(json.dumps(summary, indent=2))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
