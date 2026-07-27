"""Versioned workflow state and guarded status transitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import WorkflowConfig


GRADING_STATES = {
    "NOT_STARTED",
    "IN_PROGRESS",
    "SUBMITTED",
    "FROZEN",
    "REVISION_REQUESTED",
}
SEGMENTATION_STATES = {
    "LOCKED",
    "ASSIGNED",
    "IN_PROGRESS",
    "SUBMITTED",
    "FROZEN",
    "REVISION_REQUESTED",
}
ADJUDICATION_STATES = {
    "LOCKED",
    "READY",
    "IN_PROGRESS",
    "REVISION_REQUESTED",
    "FINALIZED",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initial_state(config: WorkflowConfig) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "workflow_id": config.workflow_id,
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "drive": {
            "status": "NOT_CONFIGURED",
            "folders": {},
            "images": {},
            "sheets": {},
            "adjudication_sheet_id": None,
        },
        "graders": {
            grader.grader_id: {
                "grading_status": "NOT_STARTED",
                "grading_version": 0,
                "grading_submissions": [],
                "segmentation_status": "LOCKED",
                "segmentation_version": 0,
                "segmentation_submissions": [],
                "cvat": {
                    "project_id": None,
                    "project_name": None,
                    "task_id": None,
                    "task_name": None,
                },
                "revision_requests": [],
            }
            for grader in config.graders
        },
        "adjudication": {
            "status": "LOCKED",
            "quality_queue_path": None,
            "mask_queue_path": None,
            "package_dir": None,
            "revision_requests_path": None,
            "final_consensus_path": None,
        },
        "events": [],
    }


@dataclass
class WorkflowState:
    """Mutable workflow state persisted atomically to JSON."""

    path: Path
    data: dict[str, Any]

    @classmethod
    def load_or_create(cls, config: WorkflowConfig) -> "WorkflowState":
        path = config.state_path
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("workflow_id") != config.workflow_id:
                raise ValueError(
                    "State workflow_id does not match the selected configuration"
                )
            state = cls(path=path, data=data)
            state.validate(config)
            return state
        state = cls(path=path, data=initial_state(config))
        state.save()
        return state

    def validate(self, config: WorkflowConfig) -> None:
        if set(self.data.get("graders", {})) != {
            grader.grader_id for grader in config.graders
        }:
            raise ValueError("State grader identities do not match configuration")
        for grader_id, grader_state in self.data["graders"].items():
            if grader_state.get("grading_status") not in GRADING_STATES:
                raise ValueError(f"Invalid grading status for {grader_id}")
            if grader_state.get("segmentation_status") not in SEGMENTATION_STATES:
                raise ValueError(f"Invalid segmentation status for {grader_id}")
        if self.data.get("adjudication", {}).get("status") not in ADJUDICATION_STATES:
            raise ValueError("Invalid adjudication status")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_utc"] = utc_now()
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def record_event(
        self,
        event: str,
        actor: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.data.setdefault("events", []).append(
            {
                "timestamp_utc": utc_now(),
                "event": event,
                "actor": actor,
                "details": details or {},
            }
        )

    def grader_state(self, grader_id: str) -> dict[str, Any]:
        try:
            return self.data["graders"][grader_id]
        except KeyError as exc:
            raise KeyError(f"Unknown grader_id: {grader_id}") from exc

    def set_grading_status(
        self,
        grader_id: str,
        new_status: str,
        actor: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        if new_status not in GRADING_STATES:
            raise ValueError(f"Invalid grading status: {new_status}")
        grader = self.grader_state(grader_id)
        current = grader["grading_status"]
        allowed = {
            "NOT_STARTED": {"IN_PROGRESS", "SUBMITTED"},
            "IN_PROGRESS": {"SUBMITTED"},
            "SUBMITTED": {"FROZEN"},
            "FROZEN": {"REVISION_REQUESTED"},
            "REVISION_REQUESTED": {"IN_PROGRESS", "SUBMITTED"},
        }
        if new_status != current and new_status not in allowed[current]:
            raise ValueError(
                f"Invalid grading transition for {grader_id}: {current} -> {new_status}"
            )
        grader["grading_status"] = new_status
        self.record_event(
            "grading_status_changed",
            actor,
            {
                "grader_id": grader_id,
                "from": current,
                "to": new_status,
                **(details or {}),
            },
        )
        self._refresh_gates()
        self.save()

    def set_segmentation_status(
        self,
        grader_id: str,
        new_status: str,
        actor: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        if new_status not in SEGMENTATION_STATES:
            raise ValueError(f"Invalid segmentation status: {new_status}")
        grader = self.grader_state(grader_id)
        current = grader["segmentation_status"]
        allowed = {
            "LOCKED": {"ASSIGNED"},
            "ASSIGNED": {"IN_PROGRESS", "SUBMITTED"},
            "IN_PROGRESS": {"SUBMITTED"},
            "SUBMITTED": {"FROZEN"},
            "FROZEN": {"REVISION_REQUESTED"},
            "REVISION_REQUESTED": {"ASSIGNED", "IN_PROGRESS", "SUBMITTED"},
        }
        if new_status != current and new_status not in allowed[current]:
            raise ValueError(
                f"Invalid segmentation transition for {grader_id}: "
                f"{current} -> {new_status}"
            )
        grader["segmentation_status"] = new_status
        self.record_event(
            "segmentation_status_changed",
            actor,
            {
                "grader_id": grader_id,
                "from": current,
                "to": new_status,
                **(details or {}),
            },
        )
        self._refresh_gates()
        self.save()

    def require_both_grading_frozen(self) -> None:
        statuses = {
            grader_id: value["grading_status"]
            for grader_id, value in self.data["graders"].items()
        }
        if any(status != "FROZEN" for status in statuses.values()):
            raise RuntimeError(
                f"Both grading submissions must be FROZEN before CVAT setup: {statuses}"
            )

    def require_both_segmentations_frozen(self) -> None:
        statuses = {
            grader_id: value["segmentation_status"]
            for grader_id, value in self.data["graders"].items()
        }
        if any(status != "FROZEN" for status in statuses.values()):
            raise RuntimeError(
                "Both segmentation submissions must be FROZEN before adjudication: "
                f"{statuses}"
            )

    def _refresh_gates(self) -> None:
        graders = list(self.data["graders"].values())
        if graders and all(item["grading_status"] == "FROZEN" for item in graders):
            for item in graders:
                if item["segmentation_status"] == "LOCKED":
                    item["segmentation_status"] = "ASSIGNED"
        adjudication = self.data["adjudication"]
        if graders and all(
            item["segmentation_status"] == "FROZEN" for item in graders
        ):
            if adjudication["status"] == "LOCKED":
                adjudication["status"] = "READY"

    def summary(self) -> dict[str, Any]:
        return {
            "workflow_id": self.data["workflow_id"],
            "drive_status": self.data["drive"]["status"],
            "graders": {
                grader_id: {
                    "grading_status": item["grading_status"],
                    "grading_version": item["grading_version"],
                    "segmentation_status": item["segmentation_status"],
                    "segmentation_version": item["segmentation_version"],
                    "cvat": item["cvat"],
                }
                for grader_id, item in self.data["graders"].items()
            },
            "adjudication_status": self.data["adjudication"]["status"],
            "updated_utc": self.data["updated_utc"],
        }
