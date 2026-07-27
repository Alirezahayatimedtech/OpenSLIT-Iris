"""CVAT SDK integration for creating the OpenSLIT-Iris pilot workspace."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from openslit.annotation.schema import AnnotationSchema

from .config import CvatSetupConfig, CvatTaskPlan


def _require_sdk() -> tuple[Any, Any, Any]:
    try:
        from cvat_sdk import make_client, models
        from cvat_sdk.core.proxies.tasks import ResourceType
    except ImportError as exc:
        raise RuntimeError(
            "CVAT integration is not installed. Run: python -m pip install -e '.[cvat]'"
        ) from exc
    return make_client, models, ResourceType


def _color_hex(color_rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{channel:02x}" for channel in color_rgb)


def build_cvat_labels(schema: AnnotationSchema, models: Any) -> list[Any]:
    """Convert the OpenSLIT schema into CVAT project labels.

    Background is implicit in CVAT and is therefore not created as a drawable label.
    """

    return [
        models.PatchedLabelRequest(
            name=item.name,
            color=_color_hex(item.color_rgb),
            attributes=[],
        )
        for item in schema.classes
        if item.id != 0
    ]


@contextmanager
def authenticated_client(host: str) -> Iterator[Any]:
    """Create an authenticated high-level CVAT client.

    A Personal Access Token is preferred. Username/password remains available for
    a local pilot. Credentials are read only from environment variables.
    """

    make_client, _, _ = _require_sdk()
    token = os.getenv("CVAT_ACCESS_TOKEN", "").strip()
    username = os.getenv("CVAT_USERNAME", "").strip()
    password = os.getenv("CVAT_PASSWORD", "")

    if token:
        client_context = make_client(host=host)
    elif username and password:
        client_context = make_client(host=host, credentials=(username, password))
    else:
        raise RuntimeError(
            "Set CVAT_ACCESS_TOKEN, or both CVAT_USERNAME and CVAT_PASSWORD."
        )

    with client_context as client:
        if token:
            client.api_client.set_default_header("Authorization", f"Bearer {token}")
        if hasattr(client, "check_server_version"):
            client.check_server_version(fail_if_unsupported=True)
        yield client


def _find_unique_by_name(resources: list[Any], name: str, kind: str) -> Any | None:
    matches = [
        resource for resource in resources if getattr(resource, "name", None) == name
    ]
    if len(matches) > 1:
        ids = [getattr(resource, "id", None) for resource in matches]
        raise RuntimeError(f"Multiple {kind} resources named {name!r} exist: {ids}")
    return matches[0] if matches else None


def _resolve_user_id(client: Any, username: str | None) -> int | None:
    if not username:
        return None

    data, _ = client.api_client.users_api.list(username=username, page_size=100)
    results = list(getattr(data, "results", []) or [])
    exact = [item for item in results if getattr(item, "username", None) == username]
    if len(exact) != 1:
        raise RuntimeError(
            f"Expected exactly one CVAT user named {username!r}; found {len(exact)}. "
            "Create the account in CVAT before running setup."
        )
    return int(exact[0].id)


def _validate_existing_project_labels(
    client: Any, project: Any, schema: AnnotationSchema
) -> None:
    data, _ = client.api_client.labels_api.list(
        project_id=int(project.id), page_size=100
    )
    labels = list(getattr(data, "results", []) or [])
    observed = {getattr(label, "name", "") for label in labels}
    expected = {item.name for item in schema.classes if item.id != 0}
    if observed != expected:
        raise RuntimeError(
            "Existing CVAT project labels do not match Annotation Protocol v1. "
            f"Expected {sorted(expected)}, found {sorted(observed)}."
        )


def _create_or_reuse_project(
    client: Any,
    config: CvatSetupConfig,
    schema: AnnotationSchema,
    models: Any,
    allow_existing: bool,
) -> tuple[Any, bool]:
    existing = _find_unique_by_name(
        list(client.projects.list()), config.project_name, "project"
    )
    if existing is not None:
        if not allow_existing:
            raise RuntimeError(
                f"CVAT project {config.project_name!r} already exists. "
                "Use --allow-existing to reuse it after label validation."
            )
        _validate_existing_project_labels(client, existing, schema)
        return existing, False

    project = client.projects.create(
        spec=models.ProjectWriteRequest(
            name=config.project_name,
            labels=build_cvat_labels(schema, models),
        )
    )
    return project, True


def _task_exists(client: Any, project_id: int, name: str) -> Any | None:
    tasks = [
        task
        for task in client.tasks.list()
        if getattr(task, "project_id", None) == project_id
    ]
    return _find_unique_by_name(tasks, name, "task")


def _create_task(
    client: Any,
    project_id: int,
    task_plan: CvatTaskPlan,
    resources: list[str],
    segment_size: int,
    models: Any,
    resource_type: Any,
    allow_existing: bool,
) -> tuple[Any, bool]:
    existing = _task_exists(client, project_id, task_plan.name)
    if existing is not None:
        if allow_existing:
            return existing, False
        raise RuntimeError(
            f"CVAT task {task_plan.name!r} already exists in project {project_id}."
        )

    assignee_id = _resolve_user_id(client, task_plan.assignee_username)
    task = client.tasks.create_from_data(
        spec=models.TaskWriteRequest(
            name=task_plan.name,
            project_id=project_id,
            assignee_id=assignee_id,
            segment_size=segment_size,
            subset="pilot",
        ),
        resource_type=resource_type.LOCAL,
        resources=resources,
    )
    return task, True


def setup_cvat_workspace(
    config: CvatSetupConfig,
    host: str,
    allow_existing: bool = False,
) -> dict[str, Any]:
    """Create or validate the project and independent annotator tasks."""

    local_summary = config.validate()
    schema = config.load_schema()
    image_paths = [str(path) for path in config.image_paths()]
    _, models, resource_type = _require_sdk()

    with authenticated_client(host) as client:
        project, project_created = _create_or_reuse_project(
            client, config, schema, models, allow_existing
        )
        task_results = []
        for plan in config.tasks:
            task, created = _create_task(
                client=client,
                project_id=int(project.id),
                task_plan=plan,
                resources=image_paths,
                segment_size=config.segment_size,
                models=models,
                resource_type=resource_type,
                allow_existing=allow_existing,
            )
            task_results.append(
                {
                    "id": int(task.id),
                    "name": task.name,
                    "created": created,
                    "assignee_username": plan.assignee_username,
                    "images": len(image_paths),
                }
            )

    return {
        **local_summary,
        "host": host,
        "project_id": int(project.id),
        "project_created": project_created,
        "task_results": task_results,
    }
