"""Upload versioned feature tables and reports to the controlled Drive workspace."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

from openslit.workflow.config import load_workflow_config
from openslit.workflow.google_drive import (
    build_drive_service,
    ensure_folder,
    set_user_role,
    upload_file,
)
from openslit.workflow.state import WorkflowState

from .config import FeatureExtractionConfig


def _latest_run(state: WorkflowState, run_id: str | None) -> dict[str, Any]:
    runs = state.data.get("features", {}).get("runs", [])
    if not runs:
        raise RuntimeError("No feature extraction run is recorded")
    if run_id is None:
        return runs[-1]
    matches = [item for item in runs if item.get("run_id") == run_id]
    if len(matches) != 1:
        raise KeyError(f"Unknown or duplicated feature run_id: {run_id}")
    return matches[0]


def upload_feature_run(
    config: FeatureExtractionConfig,
    run_id: str | None = None,
    service: Any | None = None,
) -> dict[str, Any]:
    """Upload derived tables, reports, manifests, and previews.

    Source images and masks remain in their existing controlled folders and are
    not duplicated into the feature-results folder.
    """

    if not config.drive.upload_enabled:
        raise RuntimeError("Drive upload is disabled in the feature configuration")
    workflow_config = load_workflow_config(config.workflow_config_path)
    state = WorkflowState.load_or_create(workflow_config)
    if state.data.get("drive", {}).get("status") != "READY":
        raise RuntimeError("Bootstrap the Google Drive workspace first")
    run = _latest_run(state, run_id)
    run_dir = Path(run["run_dir"])
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    service = service or build_drive_service(workflow_config.drive.impersonated_user)
    root_id = state.data["drive"].get("root_folder_id")
    if not root_id:
        raise RuntimeError("Drive root folder ID is missing from workflow state")
    feature_root = ensure_folder(service, root_id, config.drive.folder_name)
    run_folder = ensure_folder(service, feature_root["id"], str(run["run_id"]))

    for grader in workflow_config.graders:
        set_user_role(
            service,
            feature_root["id"],
            grader.email,
            config.drive.grader_role,
            workflow_config.drive.notify_users,
        )
    set_user_role(
        service,
        feature_root["id"],
        workflow_config.adjudicator.email,
        config.drive.adjudicator_role,
        workflow_config.drive.notify_users,
    )

    uploaded: list[dict[str, Any]] = []
    allowed_suffixes = {
        ".csv",
        ".xlsx",
        ".json",
        ".html",
        ".jpg",
        ".jpeg",
        ".png",
    }
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
            continue
        relative = path.relative_to(run_dir)
        parent = run_folder
        if len(relative.parts) > 1:
            current_parent = run_folder
            for folder_name in relative.parts[:-1]:
                current_parent = ensure_folder(
                    service,
                    current_parent["id"],
                    folder_name,
                )
            parent = current_parent
        remote = upload_file(
            service,
            path,
            parent["id"],
            name=relative.name,
            mime_type=mimetypes.guess_type(path.name)[0],
            app_properties={
                "openslit_workflow_id": workflow_config.workflow_id,
                "openslit_stage": "iris_feature_extraction",
                "openslit_feature_version": config.feature_version,
                "openslit_feature_run_id": str(run["run_id"]),
            },
        )
        uploaded.append(
            {
                "relative_path": str(relative),
                "file_id": remote["id"],
                "web_view_link": remote.get("webViewLink"),
            }
        )

    drive_record = {
        "folder_id": run_folder["id"],
        "folder_web_view_link": run_folder.get("webViewLink"),
        "files": uploaded,
    }
    run["drive"] = drive_record
    state.data.setdefault("features", {})["status"] = "UPLOADED"
    state.data["drive"].setdefault("folders", {})["features"] = feature_root["id"]
    state.record_event(
        "iris_feature_run_uploaded",
        "data_custodian",
        {
            "run_id": run["run_id"],
            "folder_id": run_folder["id"],
            "files": len(uploaded),
        },
    )
    state.save()
    local_manifest = run_dir / "drive_upload_manifest.json"
    local_manifest.write_text(
        json.dumps(drive_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_remote = upload_file(
        service,
        local_manifest,
        run_folder["id"],
        app_properties={
            "openslit_workflow_id": workflow_config.workflow_id,
            "openslit_stage": "iris_feature_extraction_drive_manifest",
            "openslit_feature_version": config.feature_version,
            "openslit_feature_run_id": str(run["run_id"]),
        },
    )
    return {
        "run_id": run["run_id"],
        "folder_id": run_folder["id"],
        "folder_web_view_link": run_folder.get("webViewLink"),
        "files": len(uploaded) + 1,
        "manifest_path": str(local_manifest),
        "manifest_drive_file_id": manifest_remote["id"],
    }
