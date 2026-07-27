from __future__ import annotations

from pathlib import Path

from openslit.workflow.config import (
    AdjudicatorConfig,
    CvatWorkflowConfig,
    DriveWorkflowConfig,
    GraderConfig,
    WorkflowConfig,
)
from openslit.workflow.state import WorkflowState


def _config(tmp_path: Path) -> WorkflowConfig:
    grader_a = GraderConfig("grader_01", "a@x.org", "a", tmp_path / "a.xlsx")
    grader_b = GraderConfig("grader_02", "b@x.org", "b", tmp_path / "b.xlsx")
    return WorkflowConfig(
        config_path=tmp_path / "config.json",
        workflow_id="pilot",
        pilot_dir=tmp_path,
        image_dir=tmp_path,
        image_index_path=tmp_path / "index.csv",
        mask_manifest_path=tmp_path / "masks.csv",
        schema_path=tmp_path / "schema.json",
        state_path=tmp_path / "state.json",
        graders=(grader_a, grader_b),
        adjudicator=AdjudicatorConfig("senior", "senior@x.org", "senior"),
        drive=DriveWorkflowConfig("root", "Pilot"),
        cvat=CvatWorkflowConfig(
            "http://localhost:8080",
            "Pilot - {grader_id}",
            "Task - {grader_id} - v{version}",
        ),
    )


def test_segmentation_unlocks_only_after_both_graders_are_frozen(tmp_path: Path) -> None:
    state = WorkflowState.load_or_create(_config(tmp_path))
    state.set_grading_status("grader_01", "SUBMITTED", "grader_01")
    state.set_grading_status("grader_01", "FROZEN", "custodian")
    assert state.grader_state("grader_01")["segmentation_status"] == "LOCKED"

    state.set_grading_status("grader_02", "SUBMITTED", "grader_02")
    state.set_grading_status("grader_02", "FROZEN", "custodian")
    assert state.grader_state("grader_01")["segmentation_status"] == "ASSIGNED"
    assert state.grader_state("grader_02")["segmentation_status"] == "ASSIGNED"


def test_adjudication_unlocks_after_both_segmentations_are_frozen(tmp_path: Path) -> None:
    state = WorkflowState.load_or_create(_config(tmp_path))
    for grader_id in ["grader_01", "grader_02"]:
        state.set_grading_status(grader_id, "SUBMITTED", grader_id)
        state.set_grading_status(grader_id, "FROZEN", "custodian")
    for grader_id in ["grader_01", "grader_02"]:
        state.set_segmentation_status(grader_id, "SUBMITTED", grader_id)
        state.set_segmentation_status(grader_id, "FROZEN", "custodian")
    assert state.data["adjudication"]["status"] == "READY"
