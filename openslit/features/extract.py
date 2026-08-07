"""End-to-end, versioned extraction of interpretable iris features."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from openslit.annotation.schema import load_annotation_schema
from openslit.workflow.config import load_workflow_config
from openslit.workflow.state import WorkflowState

from .color import extract_color_features
from .common import (
    json_ready,
    load_indexed_mask,
    load_rgb_image,
    sha256_file,
    tissue_masks,
    validate_image_mask_pair,
)
from .config import FeatureExtractionConfig
from .dictionary import build_feature_dictionary
from .geometry import extract_geometry_features
from .normalization import PolarIris, normalize_iris
from .quality import FeatureEligibility, assess_feature_eligibility
from .report import create_feature_preview, write_html_report
from .texture import extract_texture_features


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_run_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not normalized:
        raise ValueError("run_id cannot be empty")
    return normalized


def _load_manifest(config: FeatureExtractionConfig) -> pd.DataFrame:
    data = pd.read_csv(config.manifest_path, dtype=str, keep_default_na=False)
    required = {"image_id", "image_file", "mask_file", "review_status", "gradable"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Feature source manifest is missing columns: {sorted(missing)}"
        )
    if (
        config.source_requirements.require_mask_sha256
        and "mask_sha256" not in data.columns
    ):
        raise ValueError("Feature source manifest must include mask_sha256")
    if data["image_id"].duplicated().any():
        duplicates = sorted(
            data.loc[data["image_id"].duplicated(), "image_id"].unique()
        )
        raise ValueError(
            f"Feature source manifest contains duplicate image IDs: {duplicates}"
        )
    return data.sort_values("image_id").reset_index(drop=True)


def _feature_state(config: FeatureExtractionConfig) -> tuple[Any, WorkflowState]:
    workflow_config = load_workflow_config(config.workflow_config_path)
    state = WorkflowState.load_or_create(workflow_config)
    adjudication_status = state.data.get("adjudication", {}).get("status")
    ai_status = state.data.get("ai", {}).get("status")
    if adjudication_status != "FINALIZED" and ai_status not in {
        "CONSENSUS_READY",
        "ASSISTED_ANNOTATION_READY",
        "FEATURES_READY",
    }:
        raise RuntimeError(
            "Feature extraction is locked until senior adjudication is finalized "
            "and a versioned consensus or approved corrected-mask manifest exists."
        )
    return workflow_config, state


def _base_provenance(
    row: pd.Series,
    config: FeatureExtractionConfig,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "image_id": str(row["image_id"]),
        "image_file": str(row["image_file"]),
        "mask_file": str(row["mask_file"]),
        "feature_version": config.feature_version,
        "feature_source": str(
            row.get("consensus_source", row.get("review_status", "unknown"))
        ),
        "source_review_status": str(row.get("review_status", "")),
    }
    for column in [
        "blinded_patient_id",
        "patient_id",
        "subject_id",
        "eye",
        "laterality",
        "session_id",
        "repeat_group_id",
        "camera",
        "site",
        "acquisition_date",
    ]:
        if column in row.index:
            output[column] = str(row[column])
    return output


def _top_flags(quality_table: pd.DataFrame) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    if "feature_gate_flags" not in quality_table.columns:
        return []
    for value in quality_table["feature_gate_flags"].fillna(""):
        for flag in str(value).split("|"):
            flag = flag.strip()
            if flag:
                counts[flag] = counts.get(flag, 0) + 1
    return [
        {"flag": flag, "count": count}
        for flag, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def extract_feature_table(
    config: FeatureExtractionConfig,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Extract geometry, color, texture, quality, and normalized-iris features."""

    config.validate(require_runtime_files=True)
    workflow_config, state = _feature_state(config)
    schema = load_annotation_schema(config.schema_path)
    manifest = _load_manifest(config)
    run_id = _safe_run_id(
        run_id or f"features_v{config.feature_version}_{_utc_stamp()}"
    )
    run_dir = config.output_dir / run_id
    if run_dir.exists():
        raise FileExistsError(
            f"Feature run already exists and cannot be overwritten: {run_dir}"
        )
    previews_dir = run_dir / "previews"
    run_dir.mkdir(parents=True, exist_ok=True)

    feature_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    preview_paths: list[Path] = []
    errors: list[dict[str, str]] = []

    for index, row in manifest.iterrows():
        provenance = _base_provenance(row, config)
        image_id = provenance["image_id"]
        image_path = config.image_dir / provenance["image_file"]
        mask_path = config.masks_dir / provenance["mask_file"]
        try:
            image = load_rgb_image(image_path)
            mask = load_indexed_mask(mask_path)
            validate_image_mask_pair(image, mask, schema)
            expected_hash = str(row.get("mask_sha256", "")).strip()
            if config.source_requirements.require_mask_sha256:
                actual_hash = sha256_file(mask_path)
                if expected_hash != actual_hash:
                    raise ValueError(
                        f"Mask SHA-256 mismatch for {image_id}: "
                        f"expected {expected_hash}, got {actual_hash}"
                    )
            try:
                polar: PolarIris | None = normalize_iris(
                    image,
                    mask,
                    schema,
                    angular_samples=config.normalization.angular_samples,
                    radial_samples=config.normalization.radial_samples,
                )
            except ValueError:
                polar = None
            eligibility = assess_feature_eligibility(
                image,
                mask,
                schema,
                polar,
                {column: str(row[column]) for column in row.index},
                config.source_requirements,
                config.quality,
            )
            if polar is not None and (
                polar.valid_angle_fraction
                < config.normalization.minimum_valid_angle_fraction
            ):
                eligibility = FeatureEligibility(
                    accepted=False,
                    flags=tuple([*eligibility.flags, "INSUFFICIENT_VALID_ANGLES"]),
                    measurements=eligibility.measurements,
                )

            geometry = extract_geometry_features(mask, schema)
            feature_row: dict[str, Any] = {
                **provenance,
                **eligibility.measurements,
                **geometry,
                "feature_gate_passed": eligibility.accepted,
                "feature_gate_flags": "|".join(eligibility.flags),
            }
            if eligibility.accepted:
                masks = tissue_masks(mask, schema)
                feature_row.update(
                    extract_color_features(
                        image,
                        masks["iris"],
                        polar,
                        normalization=config.color.normalization,
                        radial_zones=config.color.radial_zones,
                        angular_sectors=config.color.angular_sectors,
                    )
                )
                feature_row.update(
                    extract_texture_features(
                        polar,
                        gray_levels=config.texture.gray_levels,
                        glcm_distances=config.texture.glcm_distances,
                        glcm_angles_degrees=config.texture.glcm_angles_degrees,
                        haar_levels=config.texture.haar_levels,
                    )
                )

            quality_row = {
                **provenance,
                **eligibility.measurements,
                "feature_gate_passed": eligibility.accepted,
                "feature_gate_flags": "|".join(eligibility.flags),
                "extraction_error": "",
            }
            if config.preview.enabled and index < config.preview.max_images:
                try:
                    preview = create_feature_preview(
                        image_id,
                        image,
                        mask,
                        schema,
                        polar,
                        previews_dir / f"{image_id}_feature_preview.jpg",
                        eligibility.flags,
                    )
                    preview_paths.append(preview)
                except Exception as preview_error:
                    quality_row["preview_error"] = (
                        f"{type(preview_error).__name__}: {preview_error}"
                    )
            feature_rows.append(feature_row)
            quality_rows.append(quality_row)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append({"image_id": image_id, "error": error})
            quality_rows.append(
                {
                    **provenance,
                    "feature_gate_passed": False,
                    "feature_gate_flags": "EXTRACTION_ERROR",
                    "extraction_error": error,
                }
            )
            feature_rows.append(
                {
                    **provenance,
                    "feature_gate_passed": False,
                    "feature_gate_flags": "EXTRACTION_ERROR",
                }
            )

    features = pd.DataFrame(feature_rows)
    quality = pd.DataFrame(quality_rows)
    features_path = run_dir / "iris_features.csv"
    quality_path = run_dir / "feature_quality.csv"
    dictionary_path = run_dir / "feature_dictionary.csv"
    workbook_path = run_dir / "iris_features.xlsx"
    errors_path = run_dir / "extraction_errors.csv"
    features.to_csv(features_path, index=False)
    quality.to_csv(quality_path, index=False)
    dictionary = build_feature_dictionary(
        list(features.columns),
        config.feature_version,
    )
    dictionary.to_csv(dictionary_path, index=False)
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        features.to_excel(writer, sheet_name="features", index=False)
        quality.to_excel(writer, sheet_name="quality", index=False)
        dictionary.to_excel(writer, sheet_name="dictionary", index=False)
    pd.DataFrame(errors, columns=["image_id", "error"]).to_csv(
        errors_path,
        index=False,
    )

    gate_series = quality.get(
        "feature_gate_passed",
        pd.Series(dtype=bool),
    ).astype(bool)
    summary = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_version": config.feature_version,
        "workflow_id": workflow_config.workflow_id,
        "source_manifest_path": str(config.manifest_path),
        "source_manifest_sha256": sha256_file(config.manifest_path),
        "schema_path": str(config.schema_path),
        "schema_sha256": sha256_file(config.schema_path),
        "images": len(features),
        "feature_gate_passed": int(gate_series.sum()),
        "feature_gate_failed": int(len(quality) - gate_series.sum()),
        "extraction_errors": len(errors),
        "top_quality_flags": _top_flags(quality)[:20],
        "features_path": str(features_path),
        "quality_path": str(quality_path),
        "dictionary_path": str(dictionary_path),
        "workbook_path": str(workbook_path),
        "previews_dir": str(previews_dir),
        "configuration": config.validate(require_runtime_files=False),
    }
    manifest_path = run_dir / "feature_run.json"
    manifest_path.write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = write_html_report(
        summary,
        features,
        quality,
        [path.relative_to(run_dir) for path in preview_paths],
        run_dir / "feature_report.html",
    )

    feature_state = state.data.setdefault(
        "features",
        {"status": "LOCKED", "runs": [], "latest_run_id": None},
    )
    record = {
        "run_id": run_id,
        "created_utc": summary["created_utc"],
        "feature_version": config.feature_version,
        "run_dir": str(run_dir),
        "features_path": str(features_path),
        "quality_path": str(quality_path),
        "report_path": str(report_path),
        "manifest_path": str(manifest_path),
        "images": len(features),
        "passed": summary["feature_gate_passed"],
        "errors": len(errors),
        "drive": None,
    }
    feature_state["status"] = "EXTRACTED"
    feature_state["latest_run_id"] = run_id
    feature_state.setdefault("runs", []).append(record)
    state.record_event(
        "iris_features_extracted",
        "data_custodian",
        {
            "run_id": run_id,
            "feature_version": config.feature_version,
            "images": len(features),
            "passed": summary["feature_gate_passed"],
            "errors": len(errors),
        },
    )
    state.save()
    return {
        **summary,
        "report_path": str(report_path),
        "run_dir": str(run_dir),
    }
