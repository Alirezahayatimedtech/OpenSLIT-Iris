"""Bridge the gated grader workflow to two isolated CVAT projects."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from openslit.annotation.schema import AnnotationSchema
from openslit.annotation.validate_masks import validate_dataset
from openslit.cvat.api import authenticated_client, setup_cvat_workspace
from openslit.cvat.config import CvatSetupConfig, CvatTaskPlan

from .config import GraderConfig, WorkflowConfig
from .state import WorkflowState, utc_now


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_independent_cvat_projects(
    config: WorkflowConfig,
    state: WorkflowState,
    allow_existing: bool = False,
) -> dict[str, Any]:
    """Create one isolated CVAT project and task per grader."""

    state.require_both_grading_frozen()
    results: dict[str, Any] = {}
    for grader in config.graders:
        setup_config = CvatSetupConfig(
            config_path=config.config_path,
            project_name=config.cvat.project_name(grader),
            schema_path=config.schema_path,
            image_dir=config.image_dir,
            manifest_path=config.mask_manifest_path,
            image_column="image_file",
            selection_column="independent_double_annotation",
            selection_values=("true", "1", "yes"),
            segment_size=config.cvat.segment_size,
            tasks=(
                CvatTaskPlan(
                    name=config.cvat.task_name(grader, version=1),
                    assignee_username=grader.cvat_username,
                ),
            ),
        )
        result = setup_cvat_workspace(
            config=setup_config,
            host=config.cvat.host,
            allow_existing=allow_existing,
        )
        task_result = result["task_results"][0]
        grader_state = state.grader_state(grader.grader_id)
        grader_state["cvat"] = {
            "project_id": int(result["project_id"]),
            "project_name": setup_config.project_name,
            "task_id": int(task_result["id"]),
            "task_name": task_result["name"],
            "revision_tasks": [],
        }
        grader_state["segmentation_status"] = "ASSIGNED"
        state.record_event(
            "cvat_project_assigned",
            "data_custodian",
            {
                "grader_id": grader.grader_id,
                "project_id": int(result["project_id"]),
                "task_id": int(task_result["id"]),
            },
        )
        results[grader.grader_id] = result
    state.save()
    return results


def _parse_labelmap_names(text: str) -> list[str]:
    names: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line.split(":", 1)[0].strip())
    return names


def _mask_to_ids(
    mask: np.ndarray,
    schema: AnnotationSchema,
    label_names: list[str],
) -> np.ndarray:
    by_name = schema.class_by_name
    if mask.ndim == 2:
        output = np.zeros(mask.shape, dtype=np.uint8)
        unique = {int(value) for value in np.unique(mask)}
        for value in unique:
            if value == 0 and (not label_names or label_names[0] == "background"):
                output[mask == value] = 0
                continue
            if value >= len(label_names):
                raise ValueError(
                    f"Mask index {value} has no entry in the CVAT labelmap"
                )
            name = label_names[value]
            if name not in by_name:
                raise ValueError(f"Unknown CVAT label {name!r} in labelmap")
            output[mask == value] = by_name[name].id
        return output

    if mask.ndim != 3 or mask.shape[2] not in {3, 4}:
        raise ValueError(f"Unsupported exported mask shape: {mask.shape}")
    rgb = mask[:, :, :3]
    output = np.zeros(rgb.shape[:2], dtype=np.uint8)
    color_to_id = {tuple(item.color_rgb): item.id for item in schema.classes}
    unique_colors = np.unique(rgb.reshape(-1, 3), axis=0)
    for color_array in unique_colors:
        color = tuple(int(value) for value in color_array)
        if color == (0, 0, 0):
            class_id = 0
        elif color in color_to_id:
            class_id = color_to_id[color]
        else:
            raise ValueError(
                f"Exported mask contains RGB color {color} not present in the protocol schema"
            )
        output[np.all(rgb == color_array, axis=2)] = class_id
    return output


def normalize_segmentation_export(
    archive_path: Path,
    output_dir: Path,
    selected_manifest: pd.DataFrame,
    schema: AnnotationSchema,
    grader_id: str,
) -> tuple[Path, Path]:
    """Convert CVAT Segmentation Mask export to OpenSLIT indexed PNG masks."""

    extract_dir = output_dir / "raw_export"
    masks_dir = output_dir / "masks"
    extract_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

    labelmap_paths = list(extract_dir.rglob("labelmap.txt"))
    label_names: list[str] = []
    if labelmap_paths:
        if len(labelmap_paths) > 1:
            raise ValueError("Multiple labelmap.txt files found in CVAT export")
        label_names = _parse_labelmap_names(
            labelmap_paths[0].read_text(encoding="utf-8")
        )

    segmentation_dirs = [
        path for path in extract_dir.rglob("SegmentationClass") if path.is_dir()
    ]
    if len(segmentation_dirs) != 1:
        raise ValueError(
            "Expected exactly one SegmentationClass directory in CVAT export"
        )
    source_masks = {
        path.stem: path for path in segmentation_dirs[0].glob("*.png")
    }

    manifest_rows: list[dict[str, str]] = []
    for row in selected_manifest.itertuples(index=False):
        image_id = str(getattr(row, "blinded_image_id"))
        image_file = str(getattr(row, "image_file"))
        stem = Path(image_file).stem
        source_mask = source_masks.get(stem)
        if source_mask is None:
            raise FileNotFoundError(
                f"CVAT export has no SegmentationClass mask for {image_file}"
            )
        with Image.open(source_mask) as image:
            converted = _mask_to_ids(np.asarray(image), schema, label_names)
        mask_file = f"{image_id}_mask.png"
        Image.fromarray(converted, mode="L").save(masks_dir / mask_file)
        manifest_rows.append(
            {
                "image_id": image_id,
                "image_file": image_file,
                "mask_file": mask_file,
                "annotator_id": grader_id,
                "protocol_version": schema.protocol_version,
                "gradable": "true",
                "review_status": "not_reviewed",
                "comments": "",
            }
        )

    manifest_path = output_dir / "annotation_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return masks_dir, manifest_path


def export_and_freeze_segmentation(
    config: WorkflowConfig,
    state: WorkflowState,
    grader_id: str,
) -> dict[str, Any]:
    """Export one CVAT task, normalize its masks, validate, hash, and freeze it."""

    grader: GraderConfig = config.grader(grader_id)
    grader_state = state.grader_state(grader_id)
    if grader_state["segmentation_status"] not in {
        "ASSIGNED",
        "IN_PROGRESS",
        "SUBMITTED",
        "REVISION_REQUESTED",
    }:
        raise RuntimeError(
            f"Cannot freeze segmentation for {grader_id} from "
            f"{grader_state['segmentation_status']}"
        )
    task_id = grader_state["cvat"].get("task_id")
    if grader_state["cvat"].get("revision_tasks"):
        latest_revision = grader_state["cvat"]["revision_tasks"][-1]
        if int(latest_revision["version"]) == int(grader_state["segmentation_version"]) + 1:
            task_id = latest_revision["task_id"]
    if not task_id:
        raise RuntimeError(f"No CVAT task is recorded for {grader_id}")

    version = int(grader_state["segmentation_version"]) + 1
    output_dir = (
        config.state_path.parent
        / "submissions"
        / grader_id
        / f"segmentation_v{version}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "cvat_segmentation_export.zip"

    with authenticated_client(config.cvat.host) as client:
        task = client.tasks.retrieve(int(task_id))
        task.export_dataset(
            config.cvat.export_format,
            filename=str(archive_path),
            include_images=False,
        )

    schema = config.load_schema()
    selected = config.selected_mask_manifest()
    if version > 1:
        revision_ids = {
            str(item["image_id"])
            for item in grader_state.get("revision_requests", [])
            if int(item.get("revision_version", version)) == version
        }
        if revision_ids:
            selected = selected[selected["blinded_image_id"].isin(revision_ids)].copy()
    masks_dir, manifest_path = normalize_segmentation_export(
        archive_path=archive_path,
        output_dir=output_dir,
        selected_manifest=selected,
        schema=schema,
        grader_id=grader_id,
    )
    validation_report = output_dir / "validation_report.csv"
    summary = validate_dataset(
        schema_path=config.schema_path,
        manifest_path=manifest_path,
        images_dir=config.image_dir,
        masks_dir=masks_dir,
        report_path=validation_report,
    )
    if summary["errors"]:
        raise ValueError(
            f"Normalized CVAT export contains {summary['errors']} validation errors; "
            f"see {validation_report}"
        )

    archive_hash = _sha256(archive_path)
    manifest_hash = _sha256(manifest_path)
    grader_state["segmentation_version"] = version
    grader_state["segmentation_submissions"].append(
        {
            "version": version,
            "submitted_utc": utc_now(),
            "archive_path": str(archive_path),
            "archive_sha256": archive_hash,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "masks_dir": str(masks_dir),
            "validation_report": str(validation_report),
            "validation_summary": summary,
        }
    )
    if grader_state["segmentation_status"] != "SUBMITTED":
        state.set_segmentation_status(
            grader_id,
            "SUBMITTED",
            actor=grader_id,
            details={"version": version},
        )
    state.set_segmentation_status(
        grader_id,
        "FROZEN",
        actor="data_custodian",
        details={
            "version": version,
            "archive_sha256": archive_hash,
            "manifest_sha256": manifest_hash,
        },
    )
    return {
        "grader_id": grader_id,
        "version": version,
        "archive_path": str(archive_path),
        "archive_sha256": archive_hash,
        "masks_dir": str(masks_dir),
        "manifest_path": str(manifest_path),
        "validation": summary,
        "both_segmentations_frozen": all(
            item["segmentation_status"] == "FROZEN"
            for item in state.data["graders"].values()
        ),
    }


def _build_revision_import_archive(
    config: WorkflowConfig,
    state: WorkflowState,
    grader_id: str,
    image_ids: list[str],
    output_path: Path,
) -> Path:
    """Build a CVAT Segmentation Mask archive from latest frozen masks."""

    schema = config.load_schema()
    submission = state.grader_state(grader_id)["segmentation_submissions"][-1]
    masks_dir = Path(submission["masks_dir"])
    manifest = pd.read_csv(submission["manifest_path"], dtype=str, keep_default_na=False)
    manifest = manifest[manifest["image_id"].isin(image_ids)].copy()
    if set(manifest["image_id"]) != set(image_ids):
        missing = sorted(set(image_ids) - set(manifest["image_id"]))
        raise ValueError(f"Frozen masks are missing revision images: {missing}")

    staging = output_path.parent / "revision_import"
    segmentation_dir = staging / "SegmentationClass"
    image_set_dir = staging / "ImageSets" / "Segmentation"
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    image_set_dir.mkdir(parents=True, exist_ok=True)

    labelmap_lines = [
        f"{item.name}:{item.color_rgb[0]},{item.color_rgb[1]},{item.color_rgb[2]}::"
        for item in sorted(schema.classes, key=lambda item: item.id)
    ]
    (staging / "labelmap.txt").write_text(
        "\n".join(labelmap_lines) + "\n",
        encoding="utf-8",
    )
    stems: list[str] = []
    color_lookup = np.asarray(
        [item.color_rgb for item in sorted(schema.classes, key=lambda item: item.id)],
        dtype=np.uint8,
    )
    for row in manifest.itertuples(index=False):
        image_file = str(row.image_file)
        stem = Path(image_file).stem
        stems.append(stem)
        with Image.open(masks_dir / str(row.mask_file)) as image:
            indexed = np.asarray(image)
        if indexed.max(initial=0) >= len(color_lookup):
            raise ValueError(f"Mask contains an invalid class ID for {row.image_id}")
        rgb = color_lookup[indexed]
        Image.fromarray(rgb, mode="RGB").save(segmentation_dir / f"{stem}.png")
    (image_set_dir / "revision.txt").write_text(
        "\n".join(stems) + "\n",
        encoding="utf-8",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(staging))
    return output_path


def create_revision_task(
    config: WorkflowConfig,
    state: WorkflowState,
    grader_id: str,
) -> dict[str, Any]:
    """Create a pre-populated CVAT task for open senior-requested images."""

    grader = config.grader(grader_id)
    grader_state = state.grader_state(grader_id)
    if grader_state["segmentation_status"] != "REVISION_REQUESTED":
        raise RuntimeError(f"No open revision is recorded for {grader_id}")
    open_requests = [
        request
        for request in grader_state.get("revision_requests", [])
        if request.get("status") == "open"
    ]
    image_ids = sorted({str(request["image_id"]) for request in open_requests})
    if not image_ids:
        raise RuntimeError(f"No open revision images are recorded for {grader_id}")

    selected = config.selected_mask_manifest()
    revision_manifest = selected[selected["blinded_image_id"].isin(image_ids)].copy()
    if len(revision_manifest) != len(image_ids):
        raise ValueError("Revision requests include images outside the locked subset")
    revision_dir = config.state_path.parent / "revisions" / grader_id
    revision_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = revision_dir / "revision_manifest.csv"
    revision_manifest["selected_for_revision"] = "true"
    revision_manifest.to_csv(manifest_path, index=False)

    version = int(grader_state["segmentation_version"]) + 1
    setup_config = CvatSetupConfig(
        config_path=config.config_path,
        project_name=config.cvat.project_name(grader),
        schema_path=config.schema_path,
        image_dir=config.image_dir,
        manifest_path=manifest_path,
        image_column="image_file",
        selection_column="selected_for_revision",
        selection_values=("true",),
        segment_size=config.cvat.segment_size,
        tasks=(
            CvatTaskPlan(
                name=config.cvat.task_name(grader, version=version),
                assignee_username=grader.cvat_username,
            ),
        ),
    )
    result = setup_cvat_workspace(
        setup_config,
        host=config.cvat.host,
        allow_existing=True,
    )
    task_result = result["task_results"][0]
    task_id = int(task_result["id"])
    import_archive = _build_revision_import_archive(
        config,
        state,
        grader_id,
        image_ids,
        revision_dir / f"revision_v{version}_initial_masks.zip",
    )
    with authenticated_client(config.cvat.host) as client:
        task = client.tasks.retrieve(task_id)
        task.import_annotations(
            config.cvat.export_format,
            filename=str(import_archive),
        )

    grader_state["cvat"].setdefault("revision_tasks", []).append(
        {
            "version": version,
            "task_id": task_id,
            "task_name": task_result["name"],
            "image_ids": image_ids,
            "initial_masks_archive": str(import_archive),
        }
    )
    for request in open_requests:
        request["status"] = "assigned"
        request["revision_task_id"] = task_id
        request["revision_version"] = version
    grader_state["segmentation_status"] = "ASSIGNED"
    state.record_event(
        "cvat_revision_task_created",
        "data_custodian",
        {
            "grader_id": grader_id,
            "task_id": task_id,
            "version": version,
            "image_ids": image_ids,
        },
    )
    state.save()
    return {
        "grader_id": grader_id,
        "version": version,
        "task_id": task_id,
        "task_name": task_result["name"],
        "image_ids": image_ids,
        "initial_masks_archive": str(import_archive),
    }
