"""Revision-request state reconciliation and durable audit-log synchronization."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from .config import WorkflowConfig
from .state import WorkflowState, utc_now


def unresolved_revision_requests(state: WorkflowState) -> list[dict[str, Any]]:
    """Return unresolved requests with the affected grader attached."""

    unresolved: list[dict[str, Any]] = []
    for grader_id, grader_state in state.data.get("graders", {}).items():
        for request in grader_state.get("revision_requests", []):
            if request.get("status") != "resolved":
                unresolved.append({"grader_id": grader_id, **request})
    return unresolved


def sync_revision_request_log(
    config: WorkflowConfig,
    state: WorkflowState,
) -> Path:
    """Rewrite the revision audit CSV from authoritative workflow state."""

    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for grader_id, grader_state in state.data.get("graders", {}).items():
        for request in grader_state.get("revision_requests", []):
            request_id = str(request["request_id"])
            grouped.setdefault(request_id, []).append((grader_id, request))

    rows: list[dict[str, Any]] = []
    for request_id, copies in sorted(grouped.items()):
        first = copies[0][1]
        statuses = {
            grader_id: str(item.get("status", "open")) for grader_id, item in copies
        }
        if statuses and all(value == "resolved" for value in statuses.values()):
            aggregate_status = "resolved"
        elif any(value == "assigned" for value in statuses.values()):
            aggregate_status = "assigned"
        else:
            aggregate_status = "open"
        resolved_times = [
            str(item.get("resolved_utc", ""))
            for _, item in copies
            if item.get("resolved_utc")
        ]
        rows.append(
            {
                "request_id": request_id,
                "created_utc": first.get("created_utc", ""),
                "image_id": first.get("image_id", ""),
                "requested_from": first.get("requested_from", ""),
                "reason": first.get("reason", ""),
                "protocol_reference": first.get("protocol_reference", ""),
                "status": aggregate_status,
                "grader_statuses": json.dumps(statuses, sort_keys=True),
                "resolved_utc": max(resolved_times) if resolved_times else "",
            }
        )

    output_path = config.state_path.parent / "adjudication" / "revision_requests.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=[
            "request_id",
            "created_utc",
            "image_id",
            "requested_from",
            "reason",
            "protocol_reference",
            "status",
            "grader_statuses",
            "resolved_utc",
        ],
    ).to_csv(output_path, index=False)
    state.data["adjudication"]["revision_requests_path"] = str(output_path)
    return output_path


def mark_revision_requests_resolved(
    config: WorkflowConfig,
    state: WorkflowState,
    grader_id: str,
    submission_version: int,
    submitted_image_ids: Iterable[str],
) -> int:
    """Resolve requests satisfied by a successfully frozen revision submission."""

    image_ids = {str(value) for value in submitted_image_ids}
    requests = state.grader_state(grader_id).get("revision_requests", [])
    if not requests:
        return 0
    resolved = 0
    for request in requests:
        request_version = request.get("revision_version")
        if request_version is None or int(request_version) != int(submission_version):
            continue
        if str(request.get("image_id")) not in image_ids:
            continue
        if request.get("status") == "resolved":
            continue
        request["status"] = "resolved"
        request["resolved_utc"] = utc_now()
        request["resolved_by_submission_version"] = int(submission_version)
        resolved += 1

    sync_revision_request_log(config, state)
    if not unresolved_revision_requests(state):
        adjudication = state.data.get("adjudication", {})
        if adjudication.get("status") == "REVISION_REQUESTED":
            adjudication["status"] = "IN_PROGRESS"
    return resolved
