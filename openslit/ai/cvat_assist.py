"""Create isolated CVAT correction tasks pre-populated with AI masks."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from openslit.annotation.schema import load_annotation_schema
from openslit.cvat.api import authenticated_client, setup_cvat_workspace
from openslit.cvat.config import CvatSetupConfig, CvatTaskPlan
from openslit.workflow.config import WorkflowConfig
from openslit.workflow.state import WorkflowState, utc_now

from .config import AIWorkflowConfig


def build_cvat_segmentation_archive(
    ai_config: AIWorkflowConfig,
    prediction_manifest_path: Path,
    prediction_masks_dir: Path,
    output_path: Path,
) -> Path:
    """Create a CVAT Segmentation Mask archive from protocol-indexed AI masks."""

    schema = load_annotation_schema(ai_config.schema_path)
    manifest = pd.read_csv(prediction_manifest_path, dtype=str, keep_default_na=False)
    required = {"image_id", "image_file", "mask_file"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Prediction manifest is missing columns: {sorted(missing)}")
    staging = output_path.parent / f".{output_path.stem}_staging"
    if staging.exists():
        shutil.rmtree(staging)
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
    for row in manifest.itertuples(index=False):
        source = prediction_masks_dir / str(row.mask_file)
        if not source.is_file():
            raise FileNotFoundError(source)
        with Image.open(source) as image:
            mask = image.convert("L")
            observed = set(mask.getdata())
            unknown = observed - set(schema.class_ids)
            if unknown:
                raise ValueError(
                    f"AI mask {source} contains unknown class IDs: {unknown}"
                )
            destination = segmentation_dir / f"{Path(str(row.image_file)).stem}.png"
            mask.save(destination)
        stems.append(Path(str(row.image_file)).stem)
    (image_set_dir / "default.txt").write_text(
        "\n".join(stems) + "\n",
        encoding="utf-8",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in staging.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(staging))
    shutil.rmtree(staging)
    return output_path


def _extract_request_id(response: Any) -> str | None:
    parsed = response[0] if isinstance(response, tuple) else response
    raw = response[1] if isinstance(response, tuple) and len(response) > 1 else None
    for candidate in [parsed, raw]:
        if candidate is None:
            continue
        for attribute in ["rq_id", "request_id", "id"]:
            value = getattr(candidate, attribute, None)
            if value:
                return str(value)
        headers = getattr(candidate, "headers", None)
        if headers:
            for key in ["X-Request-Id", "x-request-id"]:
                if key in headers:
                    return str(headers[key])
    return None


def create_ai_assisted_task(
    workflow_config: WorkflowConfig,
    ai_config: AIWorkflowConfig,
    state: WorkflowState,
    grader_id: str,
    model_id: str,
    batch_manifest_path: Path,
    prediction_manifest_path: Path,
    prediction_masks_dir: Path,
    batch_id: str,
    allow_existing: bool = False,
) -> dict[str, Any]:
    """Create a correction task and initiate import of AI pre-annotations.

    This stage is blocked until a model has a recorded independent-test approval.
    The original manual pilot tasks remain unchanged.
    """

    grader = workflow_config.grader(grader_id)
    ai_state = state.data.setdefault(
        "ai",
        {
            "status": "LOCKED",
            "models": {},
            "assisted_tasks": [],
            "active_learning_batches": [],
        },
    )
    model_state = ai_state.setdefault("models", {}).get(model_id, {})
    if not model_state.get("approved_for_assistance", False):
        raise RuntimeError(
            f"Model {model_id!r} is not approved for AI-assisted annotation. "
            "Record an independent test benchmark and senior approval first."
        )

    batch = pd.read_csv(batch_manifest_path, dtype=str, keep_default_na=False)
    required = {"image_id", "image_file"}
    missing = required - set(batch.columns)
    if missing:
        raise ValueError(f"Batch manifest is missing columns: {sorted(missing)}")
    if batch["image_id"].duplicated().any():
        raise ValueError("Batch manifest contains duplicate image IDs")

    prediction = pd.read_csv(
        prediction_manifest_path,
        dtype=str,
        keep_default_na=False,
    )
    prediction_required = {"image_id", "image_file", "mask_file"}
    prediction_missing = prediction_required - set(prediction.columns)
    if prediction_missing:
        raise ValueError(
            f"Prediction manifest is missing columns: {sorted(prediction_missing)}"
        )
    filtered_prediction = prediction[
        prediction["image_id"].isin(set(batch["image_id"]))
    ].copy()
    if set(filtered_prediction["image_id"]) != set(batch["image_id"]):
        missing_predictions = sorted(
            set(batch["image_id"]) - set(filtered_prediction["image_id"])
        )
        raise ValueError(
            f"AI predictions are missing batch images: {missing_predictions}"
        )
    expected_files = batch.set_index("image_id")["image_file"].to_dict()
    mismatched = [
        image_id
        for image_id, image_file in filtered_prediction.set_index("image_id")[
            "image_file"
        ]
        .to_dict()
        .items()
        if expected_files.get(image_id) != image_file
    ]
    if mismatched:
        raise ValueError(f"Prediction image filenames differ from batch: {mismatched}")

    local_dir = ai_config.output_dir / "cvat" / batch_id
    local_manifest = local_dir / "task_manifest.csv"
    local_prediction_manifest = local_dir / "prediction_manifest.csv"
    local_dir.mkdir(parents=True, exist_ok=True)
    prepared = batch[["image_id", "image_file"]].rename(
        columns={"image_id": "blinded_image_id"}
    )
    prepared["independent_double_annotation"] = "true"
    prepared.to_csv(local_manifest, index=False)
    filtered_prediction.to_csv(local_prediction_manifest, index=False)

    project_template = str(
        ai_config.ai_assisted_cvat.get(
            "project_name_template",
            "OpenSLIT-Iris AI-assisted - {grader_id} - {model_id}",
        )
    )
    task_template = str(
        ai_config.ai_assisted_cvat.get(
            "task_name_template",
            "OpenSLIT-Iris AI-assisted - {grader_id} - {model_id} - batch {batch_id}",
        )
    )
    project_name = project_template.format(
        grader_id=grader_id,
        model_id=model_id,
        batch_id=batch_id,
    )
    task_name = task_template.format(
        grader_id=grader_id,
        model_id=model_id,
        batch_id=batch_id,
    )
    setup = CvatSetupConfig(
        config_path=ai_config.config_path,
        project_name=project_name,
        schema_path=ai_config.schema_path,
        image_dir=ai_config.image_dir,
        manifest_path=local_manifest,
        image_column="image_file",
        selection_column="independent_double_annotation",
        selection_values=("true",),
        segment_size=workflow_config.cvat.segment_size,
        tasks=(CvatTaskPlan(task_name, grader.cvat_username),),
    )
    result = setup_cvat_workspace(
        setup,
        host=workflow_config.cvat.host,
        allow_existing=allow_existing,
    )
    task_id = int(result["task_results"][0]["id"])
    archive_path = local_manifest.parent / "ai_preannotations.zip"
    build_cvat_segmentation_archive(
        ai_config,
        local_prediction_manifest,
        prediction_masks_dir,
        archive_path,
    )

    try:
        from cvat_sdk import models
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Install CVAT dependencies with: python -m pip install -e '.[cvat]'"
        ) from exc
    with (
        archive_path.open("rb") as handle,
        authenticated_client(workflow_config.cvat.host) as client,
    ):
        response = client.api_client.tasks_api.create_annotations(
            task_id,
            format=workflow_config.cvat.export_format,
            import_mode="replace",
            annotation_file_request=models.AnnotationFileRequest(
                annotation_file=handle,
            ),
        )
    request_id = _extract_request_id(response)
    record = {
        "created_utc": utc_now(),
        "grader_id": grader_id,
        "model_id": model_id,
        "batch_id": batch_id,
        "images": len(batch),
        "project_id": int(result["project_id"]),
        "task_id": task_id,
        "project_name": project_name,
        "task_name": task_name,
        "prediction_archive": str(archive_path),
        "import_request_id": request_id,
        "status": "IMPORT_REQUESTED",
    }
    ai_state.setdefault("assisted_tasks", []).append(record)
    state.record_event("ai_assisted_task_created", "data_custodian", record)
    state.save()
    return record


def approve_model_for_assistance(
    state: WorkflowState,
    model_id: str,
    benchmark_summary_path: Path,
    approved_by: str,
    notes: str = "",
) -> dict[str, Any]:
    """Record senior approval after independent held-out evaluation."""

    if not benchmark_summary_path.is_file():
        raise FileNotFoundError(benchmark_summary_path)
    if not approved_by.strip():
        raise ValueError("approved_by is required")
    benchmark = json.loads(benchmark_summary_path.read_text(encoding="utf-8"))
    if benchmark.get("reference") != "senior_consensus":
        raise ValueError("Model approval requires a senior-consensus benchmark")
    if int(benchmark.get("images", 0)) < 1:
        raise ValueError("Model approval requires a non-empty independent benchmark")
    if benchmark.get("split") != "test":
        raise ValueError(
            "Model approval requires a benchmark restricted to the untouched test split"
        )
    if benchmark.get("source") not in {model_id, f"ai_{model_id}"}:
        raise ValueError(
            "Benchmark source does not match the model being approved: "
            f"{benchmark.get('source')!r} vs {model_id!r}"
        )

    ai_state = state.data.setdefault(
        "ai",
        {
            "status": "EVALUATION",
            "models": {},
            "assisted_tasks": [],
            "active_learning_batches": [],
        },
    )
    record = ai_state.setdefault("models", {}).setdefault(model_id, {})
    record.update(
        {
            "approved_for_assistance": True,
            "approved_utc": utc_now(),
            "approved_by": approved_by.strip(),
            "benchmark_summary_path": str(benchmark_summary_path),
            "benchmark_images": int(benchmark["images"]),
            "benchmark_macro_foreground_dice_mean": benchmark.get(
                "macro_foreground_dice_mean"
            ),
            "notes": notes,
        }
    )
    ai_state["status"] = "ASSISTED_ANNOTATION_READY"
    state.record_event(
        "ai_model_approved_for_assistance",
        approved_by.strip(),
        {"model_id": model_id, "benchmark_summary_path": str(benchmark_summary_path)},
    )
    state.save()
    return record
