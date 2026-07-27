"""Configuration for the end-to-end OpenSLIT-Iris grader workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from openslit.annotation.schema import AnnotationSchema, load_annotation_schema


@dataclass(frozen=True)
class GraderConfig:
    """Identity and local resources for one independent ophthalmologist grader."""

    grader_id: str
    email: str
    cvat_username: str
    workbook_path: Path


@dataclass(frozen=True)
class AdjudicatorConfig:
    """Identity for the senior ophthalmologist who reviews disagreements."""

    adjudicator_id: str
    email: str
    cvat_username: str | None = None


@dataclass(frozen=True)
class DriveWorkflowConfig:
    """Google Drive folder layout and authentication options."""

    parent_folder_id: str
    root_folder_name: str
    impersonated_user: str | None = None
    notify_users: bool = False
    convert_workbooks_to_google_sheets: bool = True


@dataclass(frozen=True)
class CvatWorkflowConfig:
    """CVAT project naming and export settings."""

    host: str
    project_name_template: str
    task_name_template: str
    export_format: str = "Segmentation mask 1.0"
    segment_size: int = 0

    def project_name(self, grader: GraderConfig) -> str:
        return self.project_name_template.format(
            grader_id=grader.grader_id,
            cvat_username=grader.cvat_username,
        )

    def task_name(self, grader: GraderConfig, version: int = 1) -> str:
        return self.task_name_template.format(
            grader_id=grader.grader_id,
            cvat_username=grader.cvat_username,
            version=version,
        )


@dataclass(frozen=True)
class WorkflowConfig:
    """Resolved configuration for one blinded two-grader pilot."""

    config_path: Path
    workflow_id: str
    pilot_dir: Path
    image_dir: Path
    image_index_path: Path
    mask_manifest_path: Path
    schema_path: Path
    state_path: Path
    graders: tuple[GraderConfig, GraderConfig]
    adjudicator: AdjudicatorConfig
    drive: DriveWorkflowConfig
    cvat: CvatWorkflowConfig

    def load_schema(self) -> AnnotationSchema:
        return load_annotation_schema(self.schema_path)

    def selected_mask_manifest(self) -> pd.DataFrame:
        manifest = pd.read_csv(
            self.mask_manifest_path,
            dtype=str,
            keep_default_na=False,
        )
        required = {
            "blinded_image_id",
            "image_file",
            "independent_double_annotation",
        }
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(
                f"Mask manifest is missing required columns: {sorted(missing)}"
            )
        selected = manifest[
            manifest["independent_double_annotation"]
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        ].copy()
        if selected.empty:
            raise ValueError("No independent double-annotation images are selected")
        if selected["blinded_image_id"].duplicated().any():
            duplicate_ids = sorted(
                selected.loc[
                    selected["blinded_image_id"].duplicated(), "blinded_image_id"
                ].unique()
            )
            raise ValueError(f"Duplicate blinded image IDs: {duplicate_ids}")
        return selected.sort_values("blinded_image_id").reset_index(drop=True)

    def selected_image_paths(self) -> tuple[Path, ...]:
        selected = self.selected_mask_manifest()
        paths = tuple(self.image_dir / value for value in selected["image_file"])
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            preview = missing[:10]
            suffix = "" if len(missing) <= 10 else f" and {len(missing) - 10} more"
            raise FileNotFoundError(
                f"Missing selected pilot images: {preview}{suffix}"
            )
        return paths

    def grader(self, grader_id: str) -> GraderConfig:
        matches = [grader for grader in self.graders if grader.grader_id == grader_id]
        if len(matches) != 1:
            raise KeyError(f"Unknown grader_id: {grader_id}")
        return matches[0]

    def validate(self, require_runtime_files: bool = True) -> dict[str, Any]:
        if len(self.graders) != 2:
            raise ValueError("Exactly two independent graders are required")

        grader_ids = [grader.grader_id for grader in self.graders]
        grader_emails = [grader.email.lower() for grader in self.graders]
        cvat_users = [grader.cvat_username for grader in self.graders]
        for values, label in [
            (grader_ids, "grader IDs"),
            (grader_emails, "grader emails"),
            (cvat_users, "CVAT usernames"),
        ]:
            if len(values) != len(set(values)):
                raise ValueError(f"The two {label} must be unique")

        if self.adjudicator.email.lower() in grader_emails:
            raise ValueError("The adjudicator must be different from both graders")
        if self.cvat.segment_size < 0:
            raise ValueError("CVAT segment_size cannot be negative")
        if "{grader_id}" not in self.cvat.project_name_template:
            raise ValueError("CVAT project_name_template must contain {grader_id}")
        if "{grader_id}" not in self.cvat.task_name_template:
            raise ValueError("CVAT task_name_template must contain {grader_id}")

        schema = self.load_schema()
        if schema.class_ids != frozenset(range(8)):
            raise ValueError(
                "Protocol v1 must use class IDs 0-7. Resolve schema drift before grading."
            )

        warnings: list[str] = []
        identity_values = [
            *grader_emails,
            *cvat_users,
            self.adjudicator.email.lower(),
            self.drive.parent_folder_id,
        ]
        if any(
            value.startswith("replace_") or "example.com" in value
            for value in identity_values
        ):
            warnings.append(
                "Replace placeholder emails, CVAT usernames, and Drive folder ID before deployment."
            )

        selected_images: int | None = None
        if require_runtime_files:
            if not self.pilot_dir.is_dir():
                raise FileNotFoundError(self.pilot_dir)
            for path in [self.image_index_path, self.mask_manifest_path]:
                if not path.is_file():
                    raise FileNotFoundError(path)
            for grader in self.graders:
                if not grader.workbook_path.is_file():
                    raise FileNotFoundError(grader.workbook_path)
            selected_images = len(self.selected_image_paths())

        return {
            "workflow_id": self.workflow_id,
            "protocol_version": schema.protocol_version,
            "graders": [
                {
                    "grader_id": grader.grader_id,
                    "email": grader.email,
                    "cvat_username": grader.cvat_username,
                    "workbook_path": str(grader.workbook_path),
                }
                for grader in self.graders
            ],
            "adjudicator": {
                "adjudicator_id": self.adjudicator.adjudicator_id,
                "email": self.adjudicator.email,
                "cvat_username": self.adjudicator.cvat_username,
            },
            "selected_images": selected_images,
            "state_path": str(self.state_path),
            "drive_root_name": self.drive.root_folder_name,
            "cvat_host": self.cvat.host,
            "warnings": warnings,
        }


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def load_workflow_config(path: str | Path) -> WorkflowConfig:
    """Load the end-to-end workflow JSON and resolve relative paths."""

    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent

    pilot_dir = _resolve(base, str(raw["pilot_dir"]))
    graders_raw = raw.get("graders", [])
    if len(graders_raw) != 2:
        raise ValueError("Workflow configuration must define exactly two graders")

    graders = tuple(
        GraderConfig(
            grader_id=str(item["grader_id"]).strip(),
            email=str(item["email"]).strip(),
            cvat_username=str(item["cvat_username"]).strip(),
            workbook_path=_resolve(base, str(item["workbook_path"])),
        )
        for item in graders_raw
    )

    adjudicator_raw = raw["adjudicator"]
    drive_raw = raw["drive"]
    cvat_raw = raw["cvat"]

    return WorkflowConfig(
        config_path=config_path,
        workflow_id=str(raw["workflow_id"]).strip(),
        pilot_dir=pilot_dir,
        image_dir=_resolve(base, str(raw["image_dir"])),
        image_index_path=_resolve(base, str(raw["image_index_path"])),
        mask_manifest_path=_resolve(base, str(raw["mask_manifest_path"])),
        schema_path=_resolve(base, str(raw["schema_path"])),
        state_path=_resolve(base, str(raw["state_path"])),
        graders=graders,  # type: ignore[arg-type]
        adjudicator=AdjudicatorConfig(
            adjudicator_id=str(adjudicator_raw["adjudicator_id"]).strip(),
            email=str(adjudicator_raw["email"]).strip(),
            cvat_username=(
                str(adjudicator_raw.get("cvat_username", "")).strip() or None
            ),
        ),
        drive=DriveWorkflowConfig(
            parent_folder_id=str(drive_raw["parent_folder_id"]).strip(),
            root_folder_name=str(drive_raw["root_folder_name"]).strip(),
            impersonated_user=(
                str(drive_raw.get("impersonated_user", "")).strip() or None
            ),
            notify_users=bool(drive_raw.get("notify_users", False)),
            convert_workbooks_to_google_sheets=bool(
                drive_raw.get("convert_workbooks_to_google_sheets", True)
            ),
        ),
        cvat=CvatWorkflowConfig(
            host=str(cvat_raw.get("host", "http://localhost:8080")).rstrip("/"),
            project_name_template=str(cvat_raw["project_name_template"]),
            task_name_template=str(cvat_raw["task_name_template"]),
            export_format=str(
                cvat_raw.get("export_format", "Segmentation mask 1.0")
            ),
            segment_size=int(cvat_raw.get("segment_size", 0)),
        ),
    )
