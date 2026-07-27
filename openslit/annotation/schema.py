"""Machine-readable OpenSLIT-Iris annotation schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnnotationClass:
    """One semantic-segmentation class."""

    id: int
    name: str
    display_name: str
    color_rgb: tuple[int, int, int]
    required_per_gradable_image: bool
    description: str


@dataclass(frozen=True)
class AnnotationSchema:
    """Validated annotation schema used by mask tooling."""

    protocol_name: str
    protocol_version: str
    task_type: str
    classes: tuple[AnnotationClass, ...]
    class_precedence_high_to_low: tuple[str, ...]
    required_manifest_columns: tuple[str, ...]
    optional_manifest_columns: tuple[str, ...]
    forbidden_shared_fields: tuple[str, ...]

    @property
    def class_ids(self) -> frozenset[int]:
        return frozenset(item.id for item in self.classes)

    @property
    def class_names(self) -> frozenset[str]:
        return frozenset(item.name for item in self.classes)

    @property
    def required_class_ids(self) -> frozenset[int]:
        return frozenset(
            item.id for item in self.classes if item.required_per_gradable_image
        )

    @property
    def class_by_id(self) -> dict[int, AnnotationClass]:
        return {item.id: item for item in self.classes}

    @property
    def class_by_name(self) -> dict[str, AnnotationClass]:
        return {item.name: item for item in self.classes}


def _parse_color(value: Any, class_name: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Class {class_name!r} must define a three-value color_rgb")
    color = tuple(int(channel) for channel in value)
    if any(channel < 0 or channel > 255 for channel in color):
        raise ValueError(f"Class {class_name!r} has an invalid RGB color: {color}")
    return color


def load_annotation_schema(path: str | Path) -> AnnotationSchema:
    """Load and strictly validate an annotation-schema JSON file."""

    schema_path = Path(path)
    raw = json.loads(schema_path.read_text(encoding="utf-8"))

    raw_classes = raw.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise ValueError("Annotation schema must contain a non-empty classes list")

    classes: list[AnnotationClass] = []
    for raw_class in raw_classes:
        name = str(raw_class["name"]).strip()
        if not name:
            raise ValueError("Annotation class names cannot be empty")
        classes.append(
            AnnotationClass(
                id=int(raw_class["id"]),
                name=name,
                display_name=str(raw_class["display_name"]).strip(),
                color_rgb=_parse_color(raw_class["color_rgb"], name),
                required_per_gradable_image=bool(
                    raw_class.get("required_per_gradable_image", False)
                ),
                description=str(raw_class["description"]).strip(),
            )
        )

    ids = [item.id for item in classes]
    names = [item.name for item in classes]
    colors = [item.color_rgb for item in classes]

    if len(ids) != len(set(ids)):
        raise ValueError("Annotation class IDs must be unique")
    if len(names) != len(set(names)):
        raise ValueError("Annotation class names must be unique")
    if len(colors) != len(set(colors)):
        raise ValueError("Annotation class colors must be unique")
    if 0 not in ids:
        raise ValueError("Annotation schema must contain class ID 0 for background")

    precedence = tuple(str(value) for value in raw["class_precedence_high_to_low"])
    if set(precedence) != set(names) or len(precedence) != len(names):
        raise ValueError(
            "class_precedence_high_to_low must list every class name exactly once"
        )

    required_columns = tuple(str(value) for value in raw["required_manifest_columns"])
    optional_columns = tuple(str(value) for value in raw.get("optional_manifest_columns", []))
    forbidden_fields = tuple(str(value) for value in raw.get("forbidden_shared_fields", []))

    if len(required_columns) != len(set(required_columns)):
        raise ValueError("Required manifest columns must be unique")
    if set(required_columns).intersection(optional_columns):
        raise ValueError("Manifest columns cannot be both required and optional")

    return AnnotationSchema(
        protocol_name=str(raw["protocol_name"]),
        protocol_version=str(raw["protocol_version"]),
        task_type=str(raw["task_type"]),
        classes=tuple(classes),
        class_precedence_high_to_low=precedence,
        required_manifest_columns=required_columns,
        optional_manifest_columns=optional_columns,
        forbidden_shared_fields=forbidden_fields,
    )
