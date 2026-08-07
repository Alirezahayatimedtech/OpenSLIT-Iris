from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from openslit.ai.config import load_ai_workflow_config
from openslit.ai.consensus import materialize_consensus_dataset
from openslit.workflow.adjudication import finalize_adjudication
from openslit.workflow.config import (
    AdjudicatorConfig,
    CvatWorkflowConfig,
    DriveWorkflowConfig,
    GraderConfig,
    WorkflowConfig,
)
from openslit.workflow.cvat_bridge import (
    _build_revision_import_archive,
    export_and_freeze_segmentation,
)
from openslit.workflow.revisions import (
    mark_revision_requests_resolved,
    unresolved_revision_requests,
)
from openslit.workflow.state import WorkflowState
from openslit.workflow.submissions import resolve_segmentation_masks


def _write_schema(path: Path) -> None:
    names = [
        "background",
        "pupil",
        "iris",
        "reflection",
        "slit_beam",
        "eyelid",
        "eyelash",
        "uncertain",
    ]
    path.write_text(
        json.dumps(
            {
                "protocol_name": "test",
                "protocol_version": "1.0.0",
                "task_type": "single-label semantic segmentation",
                "class_precedence_high_to_low": list(reversed(names)),
                "classes": [
                    {
                        "id": class_id,
                        "name": name,
                        "display_name": name,
                        "color_rgb": [class_id * 20] * 3,
                        "required_per_gradable_image": name in {"pupil", "iris"},
                        "description": name,
                    }
                    for class_id, name in enumerate(names)
                ],
                "required_manifest_columns": [],
                "optional_manifest_columns": [],
                "forbidden_shared_fields": [],
            }
        ),
        encoding="utf-8",
    )


def _config(tmp_path: Path) -> WorkflowConfig:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for image_id in ["I1", "I2"]:
        Image.new("RGB", (8, 8)).save(image_dir / f"{image_id}.jpg")
    pd.DataFrame(
        [
            {
                "blinded_image_id": image_id,
                "blinded_patient_id": f"P{index}",
                "image_file": f"{image_id}.jpg",
                "drive_url": "",
            }
            for index, image_id in enumerate(["I1", "I2"], start=1)
        ]
    ).to_csv(tmp_path / "index.csv", index=False)
    pd.DataFrame(
        [
            {
                "blinded_image_id": image_id,
                "image_file": f"{image_id}.jpg",
                "independent_double_annotation": "true",
            }
            for image_id in ["I1", "I2"]
        ]
    ).to_csv(tmp_path / "mask_manifest.csv", index=False)
    _write_schema(tmp_path / "schema.json")
    return WorkflowConfig(
        config_path=tmp_path / "config.json",
        workflow_id="pilot",
        pilot_dir=tmp_path,
        image_dir=image_dir,
        image_index_path=tmp_path / "index.csv",
        mask_manifest_path=tmp_path / "mask_manifest.csv",
        schema_path=tmp_path / "schema.json",
        state_path=tmp_path / "workflow" / "state.json",
        graders=(
            GraderConfig("grader_01", "a@x.org", "a", tmp_path / "a.xlsx"),
            GraderConfig("grader_02", "b@x.org", "b", tmp_path / "b.xlsx"),
        ),
        adjudicator=AdjudicatorConfig("senior", "senior@x.org", "senior"),
        drive=DriveWorkflowConfig("root", "Pilot"),
        cvat=CvatWorkflowConfig(
            "http://localhost:8080",
            "Pilot - {grader_id}",
            "Task - {grader_id} - v{version}",
        ),
    )


def _add_submission(
    state: WorkflowState,
    tmp_path: Path,
    grader_id: str,
    version: int,
    masks: dict[str, int],
) -> None:
    submission_dir = tmp_path / grader_id / f"v{version}"
    masks_dir = submission_dir / "masks"
    masks_dir.mkdir(parents=True)
    rows = []
    for image_id, class_id in masks.items():
        mask_file = f"{image_id}.png"
        Image.fromarray(np.full((8, 8), class_id, dtype=np.uint8)).save(
            masks_dir / mask_file
        )
        rows.append({"image_id": image_id, "mask_file": mask_file})
    manifest_path = submission_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    state.grader_state(grader_id)["segmentation_submissions"].append(
        {
            "version": version,
            "manifest_path": str(manifest_path),
            "masks_dir": str(masks_dir),
        }
    )
    state.grader_state(grader_id)["segmentation_version"] = version


def _add_array_submission(
    state: WorkflowState,
    tmp_path: Path,
    grader_id: str,
    version: int,
    masks: dict[str, np.ndarray],
) -> None:
    submission_dir = tmp_path / f"arrays_{grader_id}" / f"v{version}"
    masks_dir = submission_dir / "masks"
    masks_dir.mkdir(parents=True)
    rows = []
    for image_id, mask in masks.items():
        mask_file = f"{image_id}.png"
        Image.fromarray(mask.astype(np.uint8)).save(masks_dir / mask_file)
        rows.append({"image_id": image_id, "mask_file": mask_file})
    manifest_path = submission_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    state.grader_state(grader_id)["segmentation_submissions"].append(
        {
            "version": version,
            "manifest_path": str(manifest_path),
            "masks_dir": str(masks_dir),
        }
    )
    state.grader_state(grader_id)["segmentation_version"] = version


def _ai_config(tmp_path: Path):
    path = tmp_path / "ai.json"
    path.write_text(
        json.dumps(
            {
                "workflow_config_path": "config.json",
                "consensus_manifest_path": "consensus.csv",
                "image_dir": "images",
                "output_dir": "ai_output",
                "schema_path": "schema.json",
                "split": {
                    "group_column": "blinded_patient_id",
                    "train_fraction": 0.6,
                    "validation_fraction": 0.2,
                    "test_fraction": 0.2,
                },
                "models": [
                    {
                        "model_id": "unet",
                        "family": "segmentation_models_pytorch",
                        "role": "baseline",
                        "enabled": True,
                        "parameters": {},
                    }
                ],
                "active_learning": {
                    "batch_size": 4,
                    "uncertainty_fraction": 0.4,
                    "model_disagreement_fraction": 0.25,
                    "diversity_fraction": 0.2,
                    "random_fraction": 0.15,
                    "minimum_random_images": 1,
                    "exclude_test_set": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return load_ai_workflow_config(path)


def _write_adjudication_queue(
    path: Path,
    outcomes: tuple[str, str] = ("ACCEPT_A", "ACCEPT_B"),
    comment: str = "reviewed",
) -> None:
    pd.DataFrame(
        [
            {
                "image_id": image_id,
                "senior_outcome": outcome,
                "senior_comment": comment,
                "adjudication_status": "resolved",
                "consensus_mask_file": "",
            }
            for image_id, outcome in zip(["I1", "I2"], outcomes)
        ]
    ).to_csv(path, index=False)


def test_partial_revision_preserves_unchanged_masks(tmp_path: Path) -> None:
    config = _config(tmp_path)
    state = WorkflowState.load_or_create(config)
    _add_submission(state, tmp_path, "grader_01", 1, {"I1": 1, "I2": 2})
    _add_submission(state, tmp_path, "grader_01", 2, {"I1": 3})

    resolved = resolve_segmentation_masks(
        state,
        "grader_01",
        ["I1", "I2"],
    )

    assert resolved["I1"].submission_version == 2
    assert resolved["I2"].submission_version == 1
    assert np.asarray(Image.open(resolved["I1"].path))[0, 0] == 3
    assert np.asarray(Image.open(resolved["I2"].path))[0, 0] == 2


def test_revision_import_uses_latest_mask_for_each_image(tmp_path: Path) -> None:
    config = _config(tmp_path)
    state = WorkflowState.load_or_create(config)
    _add_submission(state, tmp_path, "grader_01", 1, {"I1": 1, "I2": 2})
    _add_submission(state, tmp_path, "grader_01", 2, {"I1": 3})
    archive_path = tmp_path / "revision.zip"

    _build_revision_import_archive(
        config,
        state,
        "grader_01",
        ["I2"],
        archive_path,
    )

    with (
        zipfile.ZipFile(archive_path) as archive,
        archive.open("SegmentationClass/I2.png") as handle,
    ):
        rgb = np.asarray(Image.open(handle).convert("RGB"))
    assert tuple(rgb[0, 0]) == (40, 40, 40)


def test_revision_cannot_freeze_before_revision_task_is_created(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    state = WorkflowState.load_or_create(config)
    grader_state = state.grader_state("grader_01")
    grader_state["segmentation_status"] = "REVISION_REQUESTED"
    grader_state["segmentation_version"] = 1

    with pytest.raises(RuntimeError, match="Cannot freeze segmentation"):
        export_and_freeze_segmentation(config, state, "grader_01")


def test_revision_state_resolves_only_after_all_targets_freeze(tmp_path: Path) -> None:
    config = _config(tmp_path)
    state = WorkflowState.load_or_create(config)
    state.data["adjudication"]["status"] = "REVISION_REQUESTED"
    request = {
        "request_id": "REV-00001",
        "created_utc": "2026-01-01T00:00:00+00:00",
        "image_id": "I1",
        "requested_from": "both",
        "reason": "boundary",
        "protocol_reference": "v1",
        "status": "assigned",
        "revision_version": 2,
    }
    for grader_id in ["grader_01", "grader_02"]:
        state.grader_state(grader_id)["revision_requests"].append(request.copy())

    assert (
        mark_revision_requests_resolved(
            config,
            state,
            "grader_01",
            2,
            ["I1"],
        )
        == 1
    )
    assert len(unresolved_revision_requests(state)) == 1
    assert state.data["adjudication"]["status"] == "REVISION_REQUESTED"

    assert (
        mark_revision_requests_resolved(
            config,
            state,
            "grader_02",
            2,
            ["I1"],
        )
        == 1
    )
    assert unresolved_revision_requests(state) == []
    assert state.data["adjudication"]["status"] == "IN_PROGRESS"
    log = pd.read_csv(
        state.data["adjudication"]["revision_requests_path"],
        dtype=str,
        keep_default_na=False,
    )
    assert log.loc[0, "status"] == "resolved"


def test_adjudication_rejects_nonfinal_outcomes_and_state_revisions(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    state = WorkflowState.load_or_create(config)
    queue_path = tmp_path / "queue.csv"
    _write_adjudication_queue(
        queue_path,
        outcomes=("PROTOCOL_CLARIFICATION_REQUIRED", "ACCEPT_B"),
    )
    with pytest.raises(ValueError, match="non-final outcomes"):
        finalize_adjudication(config, state, queue_path)

    _write_adjudication_queue(
        queue_path,
        outcomes=("CREATE_CONSENSUS", "ACCEPT_B"),
    )
    with pytest.raises(ValueError, match="requires consensus_mask_file"):
        finalize_adjudication(config, state, queue_path)

    _write_adjudication_queue(queue_path)
    state.grader_state("grader_01")["revision_requests"].append(
        {
            "request_id": "REV-00002",
            "image_id": "I1",
            "status": "open",
        }
    )
    with pytest.raises(ValueError, match="unresolved revision requests"):
        finalize_adjudication(config, state, queue_path)


def test_final_adjudication_is_idempotent_but_not_overwritable(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    state = WorkflowState.load_or_create(config)
    queue_path = tmp_path / "queue.csv"
    _write_adjudication_queue(queue_path)

    first = finalize_adjudication(config, state, queue_path)
    second = finalize_adjudication(config, state, queue_path)
    assert first["idempotent"] is False
    assert second["idempotent"] is True
    assert first["sha256"] == second["sha256"]

    _write_adjudication_queue(queue_path, comment="changed")
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        finalize_adjudication(config, state, queue_path)


def test_consensus_materialization_uses_per_image_revision_history(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    state = WorkflowState.load_or_create(config)
    base = np.zeros((8, 8), dtype=np.uint8)
    base[1:7, 1:7] = 2
    base[3:5, 3:5] = 1
    revised = base.copy()
    revised[0, 0] = 3
    _add_array_submission(
        state,
        tmp_path,
        "grader_01",
        1,
        {"I1": base, "I2": base},
    )
    _add_array_submission(
        state,
        tmp_path,
        "grader_01",
        2,
        {"I1": revised},
    )
    _add_array_submission(
        state,
        tmp_path,
        "grader_02",
        1,
        {"I1": base, "I2": base},
    )
    queue_path = tmp_path / "queue.csv"
    pd.DataFrame(
        [
            {
                "image_id": image_id,
                "image_file": f"{image_id}.jpg",
                "senior_outcome": "ACCEPT_A",
                "senior_comment": "reviewed",
                "adjudication_status": "resolved",
                "consensus_mask_file": "",
            }
            for image_id in ["I1", "I2"]
        ]
    ).to_csv(queue_path, index=False)
    frozen = finalize_adjudication(config, state, queue_path)

    result = materialize_consensus_dataset(
        config,
        _ai_config(tmp_path),
        state,
        Path(frozen["final_adjudication_path"]),
    )

    consensus_dir = Path(result["masks_dir"])
    assert np.asarray(Image.open(consensus_dir / "I1_mask.png"))[0, 0] == 3
    assert np.asarray(Image.open(consensus_dir / "I2_mask.png"))[0, 0] == 0
