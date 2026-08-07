"""Materialize senior-adjudicated consensus masks for the AI stage."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from openslit.annotation.validate_masks import validate_dataset
from openslit.workflow.config import WorkflowConfig
from openslit.workflow.state import WorkflowState, utc_now
from openslit.workflow.submissions import resolve_segmentation_masks

from .config import AIWorkflowConfig

FINAL_OUTCOMES = {"ACCEPT_A", "ACCEPT_B", "CREATE_CONSENSUS", "UNGRADABLE"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _patient_lookup(workflow_config: WorkflowConfig) -> dict[str, str]:
    selected = workflow_config.selected_mask_manifest()
    if "blinded_patient_id" in selected.columns:
        return selected.set_index("blinded_image_id")["blinded_patient_id"].to_dict()
    index = pd.read_csv(
        workflow_config.image_index_path,
        dtype=str,
        keep_default_na=False,
    )
    required = {"blinded_image_id", "blinded_patient_id"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(
            "Patient-level AI splitting requires blinded_patient_id in the mask manifest "
            f"or pilot image index; missing {sorted(missing)}"
        )
    return index.set_index("blinded_image_id")["blinded_patient_id"].to_dict()


def materialize_consensus_dataset(
    workflow_config: WorkflowConfig,
    ai_config: AIWorkflowConfig,
    state: WorkflowState,
    adjudication_queue_path: Path,
) -> dict[str, Any]:
    """Create immutable consensus masks and a training-ready manifest.

    Revision and protocol-clarification outcomes must first be replaced by one of
    the four final outcomes. Ungradable images remain in the audit table but are
    excluded from the training manifest.
    """

    adjudication = state.data.get("adjudication", {})
    if adjudication.get("status") != "FINALIZED":
        raise RuntimeError(
            "Senior adjudication must be finalized before consensus materialization"
        )
    frozen_queue_value = adjudication.get("final_adjudication_path")
    if not frozen_queue_value:
        legacy_value = adjudication.get("final_consensus_path")
        if legacy_value and str(legacy_value).endswith(".csv"):
            frozen_queue_value = legacy_value
    if not frozen_queue_value:
        raise RuntimeError("Workflow state does not record a frozen adjudication file")
    frozen_queue = Path(str(frozen_queue_value)).expanduser().resolve()
    supplied_queue = adjudication_queue_path.expanduser().resolve()
    if supplied_queue != frozen_queue:
        raise ValueError(
            "Consensus must be materialized from the frozen adjudication file: "
            f"{frozen_queue}"
        )
    expected_queue_hash = str(adjudication.get("final_adjudication_sha256", ""))
    if expected_queue_hash and _sha256(frozen_queue) != expected_queue_hash:
        raise ValueError("Frozen adjudication SHA-256 does not match workflow state")

    queue = pd.read_csv(frozen_queue, dtype=str, keep_default_na=False)
    required = {
        "image_id",
        "image_file",
        "senior_outcome",
        "consensus_mask_file",
        "adjudication_status",
    }
    missing = required - set(queue.columns)
    if missing:
        raise ValueError(f"Adjudication queue is missing columns: {sorted(missing)}")
    invalid = sorted(set(queue["senior_outcome"]) - FINAL_OUTCOMES)
    if invalid:
        raise ValueError(
            "Consensus cannot be materialized until all cases have a final outcome. "
            f"Non-final outcomes: {invalid}"
        )
    if not queue["adjudication_status"].isin({"resolved", "finalized"}).all():
        unresolved = queue.loc[
            ~queue["adjudication_status"].isin({"resolved", "finalized"}),
            "image_id",
        ].tolist()
        raise ValueError(f"Adjudication is not resolved for: {unresolved[:20]}")

    grader_a, grader_b = workflow_config.graders
    masks_a = resolve_segmentation_masks(state, grader_a.grader_id)
    masks_b = resolve_segmentation_masks(state, grader_b.grader_id)
    patient_by_image = _patient_lookup(workflow_config)
    schema = workflow_config.load_schema()

    adjudication_dir = workflow_config.state_path.parent / "adjudication"
    masks_dir = adjudication_dir / "consensus_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, str]] = []

    for row in queue.itertuples(index=False):
        image_id = str(row.image_id)
        image_file = str(row.image_file)
        outcome = str(row.senior_outcome)
        source_path: Path | None = None
        source_label = outcome
        if outcome == "ACCEPT_A":
            if image_id not in masks_a:
                raise ValueError(f"Grader A has no frozen mask for {image_id}")
            source_path = masks_a[image_id].path
        elif outcome == "ACCEPT_B":
            if image_id not in masks_b:
                raise ValueError(f"Grader B has no frozen mask for {image_id}")
            source_path = masks_b[image_id].path
        elif outcome == "CREATE_CONSENSUS":
            raw_path = str(row.consensus_mask_file).strip()
            if not raw_path:
                raise ValueError(
                    f"CREATE_CONSENSUS requires consensus_mask_file for {image_id}"
                )
            candidate = Path(raw_path).expanduser()
            source_path = (
                candidate
                if candidate.is_absolute()
                else (frozen_queue.parent / candidate).resolve()
            )
        elif outcome == "UNGRADABLE":
            audit_rows.append(
                {
                    "image_id": image_id,
                    "image_file": image_file,
                    "blinded_patient_id": patient_by_image.get(image_id, ""),
                    "senior_outcome": outcome,
                    "mask_file": "",
                    "include_for_training": False,
                    "mask_sha256": "",
                }
            )
            continue

        if source_path is None or not source_path.is_file():
            raise FileNotFoundError(source_path or image_id)
        with Image.open(source_path) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8)
        unknown = sorted(set(np.unique(mask).tolist()) - set(schema.class_ids))
        if unknown:
            raise ValueError(
                f"Consensus source for {image_id} contains unknown IDs: {unknown}"
            )
        destination_name = f"{image_id}_mask.png"
        destination = masks_dir / destination_name
        Image.fromarray(mask, mode="L").save(destination)
        mask_hash = _sha256(destination)
        patient_id = patient_by_image.get(image_id, "")
        if not patient_id:
            raise ValueError(f"No blinded patient ID is available for {image_id}")
        training_rows.append(
            {
                "image_id": image_id,
                "image_file": image_file,
                "mask_file": destination_name,
                "annotator_id": workflow_config.adjudicator.adjudicator_id,
                "protocol_version": schema.protocol_version,
                "gradable": "true",
                "review_status": "senior_consensus",
                "comments": source_label,
                "blinded_patient_id": patient_id,
                "consensus_source": source_label,
                "mask_sha256": mask_hash,
            }
        )
        audit_rows.append(
            {
                "image_id": image_id,
                "image_file": image_file,
                "blinded_patient_id": patient_id,
                "senior_outcome": outcome,
                "mask_file": destination_name,
                "include_for_training": True,
                "mask_sha256": mask_hash,
            }
        )

    manifest_path = ai_config.consensus_manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(training_rows).to_csv(manifest_path, index=False)
    audit_path = adjudication_dir / "final_consensus_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    validation_report = adjudication_dir / "final_consensus_validation.csv"
    validation = validate_dataset(
        schema_path=workflow_config.schema_path,
        manifest_path=manifest_path,
        images_dir=workflow_config.image_dir,
        masks_dir=masks_dir,
        report_path=validation_report,
    )
    if validation["errors"]:
        raise ValueError(
            f"Consensus dataset contains {validation['errors']} validation errors; "
            f"see {validation_report}"
        )

    state.data["adjudication"]["final_consensus_path"] = str(manifest_path)
    state.data["adjudication"]["final_consensus_masks_dir"] = str(masks_dir)
    state.data["adjudication"]["final_consensus_audit_path"] = str(audit_path)
    ai_state = state.data.setdefault(
        "ai",
        {
            "status": "LOCKED",
            "models": {},
            "assisted_tasks": [],
            "active_learning_batches": [],
        },
    )
    ai_state["status"] = "CONSENSUS_READY"
    ai_state["consensus_manifest_path"] = str(manifest_path)
    ai_state["consensus_masks_dir"] = str(masks_dir)
    state.record_event(
        "ai_consensus_dataset_materialized",
        workflow_config.adjudicator.adjudicator_id,
        {
            "created_utc": utc_now(),
            "gradable_images": len(training_rows),
            "ungradable_images": len(audit_rows) - len(training_rows),
            "manifest_path": str(manifest_path),
            "masks_dir": str(masks_dir),
        },
    )
    state.save()
    return {
        "gradable_images": len(training_rows),
        "ungradable_images": len(audit_rows) - len(training_rows),
        "manifest_path": str(manifest_path),
        "masks_dir": str(masks_dir),
        "audit_path": str(audit_path),
        "validation": validation,
    }
