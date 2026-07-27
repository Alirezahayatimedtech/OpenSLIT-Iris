"""Generate senior-review packages from frozen independent submissions."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from openslit.collaboration.validation import merge_submissions

from .config import WorkflowConfig
from .state import WorkflowState, utc_now


ADJUDICATION_OUTCOMES = [
    "",
    "ACCEPT_A",
    "ACCEPT_B",
    "CREATE_CONSENSUS",
    "REVISION_REQUESTED_FROM_A",
    "REVISION_REQUESTED_FROM_B",
    "REVISION_REQUESTED_FROM_BOTH",
    "UNGRADABLE",
    "PROTOCOL_CLARIFICATION_REQUIRED",
]


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 1.0


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    intersection = int(np.logical_and(a, b).sum())
    return _safe_divide(2 * intersection, int(a.sum()) + int(b.sum()))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return _safe_divide(intersection, union)


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return None
    y, x = coordinates.mean(axis=0)
    return float(x), float(y)


def _center_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    center_a = _centroid(a)
    center_b = _centroid(b)
    if center_a is None or center_b is None:
        return None
    return math.dist(center_a, center_b)


def _boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    eroded = (
        center
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return center & ~eroded


def _overlay_disagreement(
    image_path: Path,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    output_path: Path,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    if image.size != (mask_a.shape[1], mask_a.shape[0]):
        raise ValueError(f"Image-mask size mismatch for {image_path.name}")
    disagreement = mask_a != mask_b
    disagreement_rgba = np.zeros((*disagreement.shape, 4), dtype=np.uint8)
    disagreement_rgba[disagreement] = [255, 0, 0, 115]
    overlay = Image.fromarray(disagreement_rgba, mode="RGBA")

    draw = ImageDraw.Draw(overlay)
    boundary_a = np.argwhere(_boundary(mask_a > 0))
    boundary_b = np.argwhere(_boundary(mask_b > 0))
    for y, x in boundary_a:
        draw.point((int(x), int(y)), fill=(255, 255, 255, 230))
    for y, x in boundary_b:
        draw.point((int(x), int(y)), fill=(0, 0, 0, 230))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(
        output_path,
        quality=92,
    )


def _latest_submission(state: WorkflowState, grader_id: str, stage: str) -> dict[str, Any]:
    key = f"{stage}_submissions"
    submissions = state.grader_state(grader_id).get(key, [])
    if not submissions:
        raise RuntimeError(f"No frozen {stage} submission found for {grader_id}")
    return submissions[-1]


def build_mask_disagreement_package(
    config: WorkflowConfig,
    state: WorkflowState,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Compare frozen masks and prepare a senior adjudication queue."""

    state.require_both_segmentations_frozen()
    output_dir = output_dir or (config.state_path.parent / "adjudication")
    overlays_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)

    grader_a, grader_b = config.graders
    submission_a = _latest_submission(state, grader_a.grader_id, "segmentation")
    submission_b = _latest_submission(state, grader_b.grader_id, "segmentation")
    masks_a_dir = Path(submission_a["masks_dir"])
    masks_b_dir = Path(submission_b["masks_dir"])
    manifest_a = pd.read_csv(submission_a["manifest_path"], dtype=str, keep_default_na=False)
    manifest_b = pd.read_csv(submission_b["manifest_path"], dtype=str, keep_default_na=False)
    map_a = manifest_a.set_index("image_id")["mask_file"].to_dict()
    map_b = manifest_b.set_index("image_id")["mask_file"].to_dict()
    selected = config.selected_mask_manifest()
    schema = config.load_schema()

    metric_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        image_id = str(getattr(row, "blinded_image_id"))
        image_file = str(getattr(row, "image_file"))
        if image_id not in map_a or image_id not in map_b:
            raise ValueError(f"Both graders must provide a mask for {image_id}")
        with Image.open(masks_a_dir / map_a[image_id]) as image_a:
            mask_a = np.asarray(image_a)
        with Image.open(masks_b_dir / map_b[image_id]) as image_b:
            mask_b = np.asarray(image_b)
        if mask_a.shape != mask_b.shape:
            raise ValueError(f"Mask dimensions differ for {image_id}")

        disagreement_fraction = float((mask_a != mask_b).mean())
        per_class: dict[str, dict[str, float]] = {}
        for class_item in schema.classes:
            a = mask_a == class_item.id
            b = mask_b == class_item.id
            dice = _dice(a, b)
            iou = _iou(a, b)
            per_class[class_item.name] = {"dice": dice, "iou": iou}
            metric_rows.append(
                {
                    "image_id": image_id,
                    "class_id": class_item.id,
                    "class_name": class_item.name,
                    "dice": round(dice, 6),
                    "iou": round(iou, 6),
                    "pixels_a": int(a.sum()),
                    "pixels_b": int(b.sum()),
                }
            )

        pupil_id = schema.class_by_name["pupil"].id
        iris_id = schema.class_by_name["iris"].id
        pupil_center_difference = _center_distance(
            mask_a == pupil_id,
            mask_b == pupil_id,
        )
        iris_area_a = int((mask_a == iris_id).sum())
        iris_area_b = int((mask_b == iris_id).sum())
        iris_area_difference_fraction = _safe_divide(
            abs(iris_area_a - iris_area_b),
            max(iris_area_a, iris_area_b),
        )
        overlay_path = overlays_dir / f"{image_id}_disagreement.jpg"
        _overlay_disagreement(
            config.image_dir / image_file,
            mask_a,
            mask_b,
            overlay_path,
        )

        queue_rows.append(
            {
                "image_id": image_id,
                "image_file": image_file,
                "overlay_file": str(overlay_path.relative_to(output_dir)),
                "disagreement_fraction": round(disagreement_fraction, 6),
                "pupil_dice": round(per_class["pupil"]["dice"], 6),
                "iris_dice": round(per_class["iris"]["dice"], 6),
                "reflection_dice": round(per_class["reflection"]["dice"], 6),
                "slit_beam_dice": round(per_class["slit_beam"]["dice"], 6),
                "eyelid_dice": round(per_class["eyelid"]["dice"], 6),
                "eyelash_dice": round(per_class["eyelash"]["dice"], 6),
                "uncertain_dice": round(per_class["uncertain"]["dice"], 6),
                "pupil_center_difference_px": (
                    ""
                    if pupil_center_difference is None
                    else round(pupil_center_difference, 3)
                ),
                "iris_area_difference_fraction": round(
                    iris_area_difference_fraction,
                    6,
                ),
                "senior_outcome": "",
                "revision_requested_from": "",
                "protocol_reference": "",
                "senior_comment": "",
                "consensus_mask_file": "",
                "adjudication_status": "pending",
            }
        )

    metrics_path = output_dir / "mask_agreement_long.csv"
    queue_path = output_dir / "senior_adjudication_queue.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(queue_rows).to_csv(queue_path, index=False)

    quality_a = _latest_submission(state, grader_a.grader_id, "grading")
    quality_b = _latest_submission(state, grader_b.grader_id, "grading")
    quality_output = output_dir / "quality_agreement"
    merge_submissions(
        Path(quality_a["local_path"]),
        Path(quality_b["local_path"]),
        config.image_index_path,
        quality_output,
    )

    state.data["adjudication"].update(
        {
            "status": "READY",
            "quality_queue_path": str(quality_output / "adjudication_queue.csv"),
            "mask_queue_path": str(queue_path),
            "package_dir": str(output_dir),
        }
    )
    state.record_event(
        "adjudication_package_built",
        "data_custodian",
        {
            "images": len(queue_rows),
            "queue_path": str(queue_path),
        },
    )
    state.save()
    return {
        "images": len(queue_rows),
        "package_dir": str(output_dir),
        "mask_metrics_path": str(metrics_path),
        "mask_queue_path": str(queue_path),
        "quality_queue_path": str(quality_output / "adjudication_queue.csv"),
    }


def record_revision_request(
    config: WorkflowConfig,
    state: WorkflowState,
    image_id: str,
    requested_from: str,
    reason: str,
    protocol_reference: str,
) -> dict[str, Any]:
    """Record a versioned senior revision request without overwriting originals."""

    valid_targets = {
        config.graders[0].grader_id,
        config.graders[1].grader_id,
        "both",
    }
    if requested_from not in valid_targets:
        raise ValueError(f"requested_from must be one of {sorted(valid_targets)}")
    selected_ids = set(config.selected_mask_manifest()["blinded_image_id"])
    if image_id not in selected_ids:
        raise ValueError(f"Unknown selected image ID: {image_id}")
    if not reason.strip():
        raise ValueError("A revision reason is required")

    request = {
        "request_id": f"REV-{len(state.data['events']) + 1:05d}",
        "created_utc": utc_now(),
        "image_id": image_id,
        "requested_from": requested_from,
        "reason": reason.strip(),
        "protocol_reference": protocol_reference.strip(),
        "status": "open",
    }
    targets = (
        [grader.grader_id for grader in config.graders]
        if requested_from == "both"
        else [requested_from]
    )
    for grader_id in targets:
        grader_state = state.grader_state(grader_id)
        grader_state["revision_requests"].append(request.copy())
        if grader_state["segmentation_status"] == "FROZEN":
            grader_state["segmentation_status"] = "REVISION_REQUESTED"

    output_path = config.state_path.parent / "adjudication" / "revision_requests.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if output_path.is_file():
        with output_path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    pd.DataFrame([*existing, request]).to_csv(output_path, index=False)
    state.data["adjudication"]["status"] = "REVISION_REQUESTED"
    state.data["adjudication"]["revision_requests_path"] = str(output_path)
    state.record_event(
        "revision_requested",
        config.adjudicator.adjudicator_id,
        request,
    )
    state.save()
    return request


def finalize_adjudication(
    config: WorkflowConfig,
    state: WorkflowState,
    adjudication_queue: Path,
) -> dict[str, Any]:
    """Validate senior decisions and freeze the adjudication record."""

    queue = pd.read_csv(adjudication_queue, dtype=str, keep_default_na=False)
    required = {"image_id", "senior_outcome", "senior_comment", "adjudication_status"}
    missing = required - set(queue.columns)
    if missing:
        raise ValueError(f"Adjudication queue is missing columns: {sorted(missing)}")
    invalid = sorted(set(queue["senior_outcome"]) - set(ADJUDICATION_OUTCOMES))
    if invalid:
        raise ValueError(f"Unknown senior outcomes: {invalid}")
    incomplete = queue[queue["senior_outcome"].eq("")]
    if not incomplete.empty:
        raise ValueError(
            f"Senior outcome is missing for {len(incomplete)} images"
        )
    open_revisions = queue[
        queue["senior_outcome"].str.startswith("REVISION_REQUESTED")
        & ~queue["adjudication_status"].eq("resolved")
    ]
    if not open_revisions.empty:
        raise ValueError(
            f"{len(open_revisions)} revision-requested images are not resolved"
        )

    frozen = config.state_path.parent / "adjudication" / "final_adjudication.csv"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(frozen, index=False)
    state.data["adjudication"]["status"] = "FINALIZED"
    state.data["adjudication"]["final_consensus_path"] = str(frozen)
    state.record_event(
        "adjudication_finalized",
        config.adjudicator.adjudicator_id,
        {"images": len(queue), "path": str(frozen)},
    )
    state.save()
    return {"images": len(queue), "final_adjudication_path": str(frozen)}
