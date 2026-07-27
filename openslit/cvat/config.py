"""Configuration and local validation for the OpenSLIT-Iris CVAT pilot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from openslit.annotation.schema import AnnotationSchema, load_annotation_schema


@dataclass(frozen=True)
class CvatTaskPlan:
    """One independent CVAT task and its optional assignee."""

    name: str
    assignee_username: str | None = None


@dataclass(frozen=True)
class CvatSetupConfig:
    """Resolved local paths and project settings for a CVAT pilot."""

    config_path: Path
    project_name: str
    schema_path: Path
    image_dir: Path
    manifest_path: Path
    image_column: str
    selection_column: str | None
    selection_values: tuple[str, ...]
    segment_size: int
    tasks: tuple[CvatTaskPlan, ...]

    def load_schema(self) -> AnnotationSchema:
        return load_annotation_schema(self.schema_path)

    def selected_manifest(self) -> pd.DataFrame:
        manifest = pd.read_csv(self.manifest_path, dtype=str, keep_default_na=False)
        if self.image_column not in manifest.columns:
            raise ValueError(
                f"Manifest is missing configured image column {self.image_column!r}"
            )

        selected = manifest.copy()
        if self.selection_column:
            if self.selection_column not in selected.columns:
                raise ValueError(
                    f"Manifest is missing selection column {self.selection_column!r}"
                )
            accepted = {value.strip().lower() for value in self.selection_values}
            selected = selected[
                selected[self.selection_column].str.strip().str.lower().isin(accepted)
            ]

        if selected.empty:
            raise ValueError("No images remain after applying the CVAT selection rule")
        if selected[self.image_column].duplicated().any():
            duplicates = sorted(
                selected.loc[selected[self.image_column].duplicated(), self.image_column]
                .astype(str)
                .unique()
            )
            raise ValueError(f"Selected manifest contains duplicate image files: {duplicates}")

        return selected.reset_index(drop=True)

    def image_paths(self) -> tuple[Path, ...]:
        selected = self.selected_manifest()
        paths = tuple(self.image_dir / value for value in selected[self.image_column])
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            preview = missing[:10]
            suffix = "" if len(missing) <= 10 else f" and {len(missing) - 10} more"
            raise FileNotFoundError(f"Missing CVAT input images: {preview}{suffix}")
        return paths

    def validate(self) -> dict[str, Any]:
        schema = self.load_schema()
        selected = self.selected_manifest()
        images = self.image_paths()

        if not self.tasks:
            raise ValueError("At least one CVAT task must be configured")
        names = [task.name for task in self.tasks]
        if len(names) != len(set(names)):
            raise ValueError("CVAT task names must be unique")
        if self.segment_size < 0:
            raise ValueError("segment_size cannot be negative")

        return {
            "project_name": self.project_name,
            "protocol_version": schema.protocol_version,
            "labels": [item.name for item in schema.classes if item.id != 0],
            "selected_images": len(selected),
            "image_dir": str(self.image_dir),
            "tasks": [
                {
                    "name": task.name,
                    "assignee_username": task.assignee_username,
                    "images": len(images),
                }
                for task in self.tasks
            ],
        }


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def load_cvat_setup_config(path: str | Path) -> CvatSetupConfig:
    """Load a CVAT pilot JSON configuration and resolve relative paths."""

    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent

    raw_tasks = raw.get("tasks", [])
    tasks = tuple(
        CvatTaskPlan(
            name=str(item["name"]).strip(),
            assignee_username=(
                str(item.get("assignee_username", "")).strip() or None
            ),
        )
        for item in raw_tasks
    )

    selection_values = tuple(
        str(value) for value in raw.get("selection_values", ["true", "1", "yes"])
    )

    return CvatSetupConfig(
        config_path=config_path,
        project_name=str(raw["project_name"]).strip(),
        schema_path=_resolve(base, str(raw["schema_path"])),
        image_dir=_resolve(base, str(raw["image_dir"])),
        manifest_path=_resolve(base, str(raw["manifest_path"])),
        image_column=str(raw.get("image_column", "image_file")),
        selection_column=(
            str(raw.get("selection_column", "")).strip() or None
        ),
        selection_values=selection_values,
        segment_size=int(raw.get("segment_size", 0)),
        tasks=tasks,
    )
