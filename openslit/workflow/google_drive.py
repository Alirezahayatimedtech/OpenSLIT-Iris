"""Google Drive orchestration for blinded grader workspaces."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from openslit.collaboration.validation import validate_submission
from openslit.collaboration.workbook import apply_drive_links, apply_drive_links_to_csv

from .config import GraderConfig, WorkflowConfig
from .state import WorkflowState, utc_now


DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
FOLDER_MIME = "application/vnd.google-apps.folder"
SHEET_MIME = "application/vnd.google-apps.spreadsheet"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _require_google_api() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import google.auth
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive integration is not installed. Run: "
            "python -m pip install -e '.[google]'"
        ) from exc
    return google.auth, service_account, build, MediaFileUpload, MediaIoBaseDownload


def build_drive_service(impersonated_user: str | None = None) -> Any:
    """Create a Drive v3 service from Application Default Credentials."""

    google_auth, service_account, build, _, _ = _require_google_api()
    credentials, _ = google_auth.default(scopes=[DRIVE_SCOPE])
    if impersonated_user:
        if not isinstance(credentials, service_account.Credentials):
            raise RuntimeError(
                "drive.impersonated_user requires service-account credentials"
            )
        credentials = credentials.with_subject(impersonated_user)
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _escape_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _list_children(service: Any, parent_id: str, name: str) -> list[dict[str, Any]]:
    query = (
        f"'{_escape_query(parent_id)}' in parents and "
        f"name = '{_escape_query(name)}' and trashed = false"
    )
    response = (
        service.files()
        .list(
            q=query,
            spaces="drive",
            fields="files(id,name,mimeType,webViewLink,appProperties)",
            pageSize=100,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    return list(response.get("files", []))


def ensure_folder(service: Any, parent_id: str, name: str) -> dict[str, Any]:
    matches = [
        item
        for item in _list_children(service, parent_id, name)
        if item.get("mimeType") == FOLDER_MIME
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple Drive folders named {name!r} exist")
    if matches:
        return matches[0]
    return (
        service.files()
        .create(
            body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]},
            fields="id,name,mimeType,webViewLink",
            supportsAllDrives=True,
        )
        .execute()
    )


def _permissions(service: Any, file_id: str) -> list[dict[str, Any]]:
    response = (
        service.permissions()
        .list(
            fileId=file_id,
            fields=(
                "permissions(id,type,role,emailAddress,displayName,permissionDetails)"
            ),
            supportsAllDrives=True,
        )
        .execute()
    )
    return list(response.get("permissions", []))


def set_user_role(
    service: Any,
    file_id: str,
    email: str,
    role: str,
    notify: bool = False,
) -> str:
    """Create or update a direct user permission."""

    normalized = email.strip().lower()
    matches = [
        permission
        for permission in _permissions(service, file_id)
        if str(permission.get("emailAddress", "")).strip().lower() == normalized
    ]

    def inherited(permission: dict[str, Any]) -> bool:
        return any(
            bool(item.get("inherited"))
            for item in (permission.get("permissionDetails") or [])
        )

    direct = [permission for permission in matches if not inherited(permission)]
    if len(direct) > 1:
        raise RuntimeError(f"Multiple direct permissions found for {email}")
    if direct:
        permission = direct[0]
        if permission.get("role") != role:
            permission = (
                service.permissions()
                .update(
                    fileId=file_id,
                    permissionId=permission["id"],
                    body={"role": role},
                    fields="id,role,emailAddress",
                    supportsAllDrives=True,
                )
                .execute()
            )
        return str(permission["id"])
    if matches and any(permission.get("role") == role for permission in matches):
        return str(matches[0]["id"])
    permission = (
        service.permissions()
        .create(
            fileId=file_id,
            body={"type": "user", "role": role, "emailAddress": email},
            sendNotificationEmail=notify,
            fields="id,role,emailAddress",
            supportsAllDrives=True,
        )
        .execute()
    )
    return str(permission["id"])


def upload_file(
    service: Any,
    local_path: Path,
    parent_id: str,
    *,
    name: str | None = None,
    mime_type: str | None = None,
    app_properties: dict[str, str] | None = None,
    convert_to_google_sheet: bool = False,
) -> dict[str, Any]:
    """Upload a file once, identified by name under one parent."""

    filename = name or local_path.name
    matches = _list_children(service, parent_id, filename)
    if len(matches) > 1:
        raise RuntimeError(f"Multiple Drive files named {filename!r} exist")
    if matches:
        return matches[0]
    _, _, _, MediaFileUpload, _ = _require_google_api()
    body: dict[str, Any] = {
        "name": filename,
        "parents": [parent_id],
        "appProperties": app_properties or {},
    }
    if convert_to_google_sheet:
        body["mimeType"] = SHEET_MIME
    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
    return (
        service.files()
        .create(
            body=body,
            media_body=media,
            fields="id,name,mimeType,webViewLink,webContentLink,appProperties",
            supportsAllDrives=True,
        )
        .execute()
    )


def export_google_sheet_xlsx(service: Any, file_id: str, output: Path) -> Path:
    _, _, _, _, MediaIoBaseDownload = _require_google_api()
    output.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().export_media(fileId=file_id, mimeType=XLSX_MIME)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    output.write_bytes(buffer.getvalue())
    return output


def _folder_layout(config: WorkflowConfig) -> dict[str, str]:
    return {
        "images": "01_Aliased_Images",
        config.graders[0].grader_id: f"02_{config.graders[0].grader_id}",
        config.graders[1].grader_id: f"03_{config.graders[1].grader_id}",
        "adjudication": "04_Adjudication",
        "consensus": "05_Final_Consensus",
    }


def bootstrap_drive(
    config: WorkflowConfig,
    state: WorkflowState,
    service: Any | None = None,
) -> dict[str, Any]:
    """Create blinded Drive folders, images, and private grader Sheets."""

    config.validate(require_runtime_files=True)
    service = service or build_drive_service(config.drive.impersonated_user)
    root = ensure_folder(
        service,
        config.drive.parent_folder_id,
        config.drive.root_folder_name,
    )
    folders: dict[str, dict[str, Any]] = {"root": root}
    for key, name in _folder_layout(config).items():
        folders[key] = ensure_folder(service, root["id"], name)

    for email in [
        config.graders[0].email,
        config.graders[1].email,
        config.adjudicator.email,
    ]:
        set_user_role(
            service,
            folders["images"]["id"],
            email,
            "reader",
            config.drive.notify_users,
        )

    index = pd.read_csv(config.image_index_path, dtype=str, keep_default_na=False)
    required = {"blinded_image_id", "image_file"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"Image index is missing columns: {sorted(missing)}")

    uploaded_images: dict[str, dict[str, Any]] = {}
    link_rows: list[dict[str, str]] = []
    for row in index.itertuples(index=False):
        image_id = str(getattr(row, "blinded_image_id"))
        image_file = str(getattr(row, "image_file"))
        path = config.image_dir / image_file
        if not path.is_file():
            raise FileNotFoundError(path)
        remote = upload_file(
            service,
            path,
            folders["images"]["id"],
            app_properties={
                "openslit_workflow_id": config.workflow_id,
                "openslit_image_id": image_id,
                "sha256": sha256_file(path),
            },
        )
        uploaded_images[image_id] = remote
        link_rows.append(
            {
                "blinded_image_id": image_id,
                "image_file": image_file,
                "drive_url": remote.get(
                    "webViewLink",
                    f"https://drive.google.com/open?id={remote['id']}",
                ),
            }
        )

    links = pd.DataFrame(link_rows)
    links_path = config.pilot_dir / "shared" / "drive_links.csv"
    links.to_csv(links_path, index=False)
    apply_drive_links_to_csv(config.image_index_path, links)

    sheets: dict[str, dict[str, Any]] = {}
    for grader in config.graders:
        apply_drive_links(grader.workbook_path, links)
        folder = folders[grader.grader_id]
        set_user_role(
            service,
            folder["id"],
            grader.email,
            "reader",
            config.drive.notify_users,
        )
        remote_sheet = upload_file(
            service,
            grader.workbook_path,
            folder["id"],
            name=f"{config.workflow_id}_{grader.grader_id}_quality_grading",
            mime_type=XLSX_MIME,
            app_properties={
                "openslit_workflow_id": config.workflow_id,
                "openslit_grader_id": grader.grader_id,
                "openslit_stage": "quality_grading",
            },
            convert_to_google_sheet=config.drive.convert_workbooks_to_google_sheets,
        )
        permission_id = set_user_role(
            service,
            remote_sheet["id"],
            grader.email,
            "writer",
            config.drive.notify_users,
        )
        remote_sheet["grader_permission_id"] = permission_id
        sheets[grader.grader_id] = remote_sheet
        state.grader_state(grader.grader_id)["grading_status"] = "IN_PROGRESS"

    state.data["drive"] = {
        "status": "READY",
        "root_folder_id": root["id"],
        "folders": {key: value["id"] for key, value in folders.items()},
        "images": {
            image_id: {
                "file_id": remote["id"],
                "web_view_link": remote.get("webViewLink"),
            }
            for image_id, remote in uploaded_images.items()
        },
        "sheets": {
            grader_id: {
                "file_id": remote["id"],
                "web_view_link": remote.get("webViewLink"),
                "permission_id": remote["grader_permission_id"],
            }
            for grader_id, remote in sheets.items()
        },
        "adjudication_sheet_id": None,
    }
    state.record_event(
        "drive_workspace_bootstrapped",
        "data_custodian",
        {
            "root_folder_id": root["id"],
            "images": len(uploaded_images),
            "graders": list(sheets),
        },
    )
    state.save()

    manifest_path = config.state_path.parent / "drive_workspace_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(state.data["drive"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "root_folder_id": root["id"],
        "root_web_view_link": root.get("webViewLink"),
        "images": len(uploaded_images),
        "grader_sheets": {
            grader_id: remote.get("webViewLink")
            for grader_id, remote in sheets.items()
        },
        "manifest_path": str(manifest_path),
    }


def _upload_frozen_snapshot(
    service: Any,
    config: WorkflowConfig,
    state: WorkflowState,
    local_path: Path,
    grader_id: str,
    version: int,
) -> dict[str, Any]:
    return upload_file(
        service,
        local_path,
        state.data["drive"]["folders"]["adjudication"],
        name=f"{grader_id}_quality_grading_v{version}.xlsx",
        mime_type=XLSX_MIME,
        app_properties={
            "openslit_workflow_id": config.workflow_id,
            "openslit_grader_id": grader_id,
            "openslit_stage": "frozen_quality_grading",
            "openslit_version": str(version),
        },
    )


def freeze_grading_submission(
    config: WorkflowConfig,
    state: WorkflowState,
    grader_id: str,
    service: Any | None = None,
) -> dict[str, Any]:
    """Export, validate, hash, and lock one grader's Google Sheet."""

    grader: GraderConfig = config.grader(grader_id)
    grader_state = state.grader_state(grader_id)
    if state.data["drive"]["status"] != "READY":
        raise RuntimeError("Bootstrap the Google Drive workspace first")
    if grader_state["grading_status"] not in {
        "IN_PROGRESS",
        "SUBMITTED",
        "REVISION_REQUESTED",
    }:
        raise RuntimeError(
            f"Cannot freeze {grader_id} from {grader_state['grading_status']}"
        )

    service = service or build_drive_service(config.drive.impersonated_user)
    sheet = state.data["drive"]["sheets"][grader_id]
    version = int(grader_state["grading_version"]) + 1
    output = (
        config.state_path.parent
        / "submissions"
        / grader_id
        / f"quality_grading_v{version}.xlsx"
    )
    export_google_sheet_xlsx(service, sheet["file_id"], output)
    _, errors = validate_submission(
        output,
        config.image_index_path,
        require_complete=True,
    )
    if errors:
        raise ValueError(
            "Quality-grading submission failed validation:\n" + "\n".join(errors)
        )

    digest = sha256_file(output)
    set_user_role(service, sheet["file_id"], grader.email, "reader", notify=False)
    snapshot = _upload_frozen_snapshot(
        service,
        config,
        state,
        output,
        grader_id,
        version,
    )
    grader_state["grading_version"] = version
    grader_state["grading_submissions"].append(
        {
            "version": version,
            "submitted_utc": utc_now(),
            "local_path": str(output),
            "sha256": digest,
            "drive_file_id": snapshot["id"],
        }
    )
    if grader_state["grading_status"] != "SUBMITTED":
        state.set_grading_status(
            grader_id,
            "SUBMITTED",
            actor=grader_id,
            details={"version": version},
        )
    state.set_grading_status(
        grader_id,
        "FROZEN",
        actor="data_custodian",
        details={"version": version, "sha256": digest},
    )

    both_frozen = all(
        item["grading_status"] == "FROZEN"
        for item in state.data["graders"].values()
    )
    if both_frozen:
        set_user_role(
            service,
            state.data["drive"]["folders"]["adjudication"],
            config.adjudicator.email,
            "writer",
            config.drive.notify_users,
        )
        state.record_event(
            "adjudication_drive_unlocked",
            "data_custodian",
            {"adjudicator": config.adjudicator.email},
        )
        state.save()

    return {
        "grader_id": grader_id,
        "version": version,
        "sha256": digest,
        "local_path": str(output),
        "drive_file_id": snapshot["id"],
        "both_grading_frozen": both_frozen,
    }


def upload_adjudication_package(
    config: WorkflowConfig,
    state: WorkflowState,
    package_dir: Path,
    queue_csv: Path,
    service: Any | None = None,
) -> dict[str, Any]:
    """Upload reports and create an editable senior adjudication Sheet."""

    service = service or build_drive_service(config.drive.impersonated_user)
    folder_id = state.data["drive"]["folders"]["adjudication"]
    uploaded: list[dict[str, Any]] = []
    for path in sorted(package_dir.rglob("*")):
        if path.is_file() and path != queue_csv:
            uploaded.append(
                upload_file(
                    service,
                    path,
                    folder_id,
                    name=path.name,
                    app_properties={
                        "openslit_workflow_id": config.workflow_id,
                        "openslit_stage": "adjudication_package",
                    },
                )
            )
    queue = upload_file(
        service,
        queue_csv,
        folder_id,
        name=f"{config.workflow_id}_senior_adjudication_queue",
        mime_type="text/csv",
        app_properties={
            "openslit_workflow_id": config.workflow_id,
            "openslit_stage": "adjudication_queue",
        },
        convert_to_google_sheet=True,
    )
    set_user_role(
        service,
        queue["id"],
        config.adjudicator.email,
        "writer",
        config.drive.notify_users,
    )
    state.data["drive"]["adjudication_sheet_id"] = queue["id"]
    state.data["adjudication"]["status"] = "IN_PROGRESS"
    state.record_event(
        "adjudication_package_uploaded",
        "data_custodian",
        {"queue_file_id": queue["id"], "files": len(uploaded) + 1},
    )
    state.save()
    return {
        "queue_sheet_id": queue["id"],
        "queue_web_view_link": queue.get("webViewLink"),
        "uploaded_files": len(uploaded) + 1,
    }
