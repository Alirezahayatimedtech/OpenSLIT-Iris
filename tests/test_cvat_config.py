from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

from openslit.cvat.config import load_cvat_setup_config


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "configs" / "annotation_schema_v1.json"


def _write_config(tmp_path: Path, tasks: list[dict[str, str]]) -> Path:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (12, 12)).save(images / "PILOT-I001.jpg")
    Image.new("RGB", (12, 12)).save(images / "PILOT-I002.jpg")

    manifest = pd.DataFrame(
        [
            {
                "image_file": "PILOT-I001.jpg",
                "independent_double_annotation": "true",
            },
            {
                "image_file": "PILOT-I002.jpg",
                "independent_double_annotation": "false",
            },
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    raw = {
        "project_name": "Pilot",
        "schema_path": str(SCHEMA_PATH),
        "image_dir": str(images),
        "manifest_path": str(manifest_path),
        "image_column": "image_file",
        "selection_column": "independent_double_annotation",
        "selection_values": ["true"],
        "tasks": tasks,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    return config_path


def test_cvat_config_selects_double_annotation_images(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        [
            {"name": "Task A", "assignee_username": "grader_a"},
            {"name": "Task B", "assignee_username": "grader_b"},
        ],
    )
    config = load_cvat_setup_config(config_path)
    summary = config.validate()

    assert summary["selected_images"] == 1
    assert [path.name for path in config.image_paths()] == ["PILOT-I001.jpg"]
    assert summary["labels"] == [
        "pupil",
        "iris",
        "reflection",
        "slit_beam",
        "eyelid",
        "eyelash",
        "uncertain",
    ]
    assert len(summary["tasks"]) == 2


def test_cvat_config_rejects_duplicate_task_names(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        [
            {"name": "Same task", "assignee_username": "grader_a"},
            {"name": "Same task", "assignee_username": "grader_b"},
        ],
    )
    config = load_cvat_setup_config(config_path)

    try:
        config.validate()
    except ValueError as exc:
        assert "task names must be unique" in str(exc)
    else:
        raise AssertionError("Duplicate task names should fail validation")


def test_cvat_config_reports_missing_images(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        [{"name": "Task A", "assignee_username": "grader_a"}],
    )
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["image_dir"] = str(tmp_path / "missing")
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    config = load_cvat_setup_config(config_path)

    try:
        config.image_paths()
    except FileNotFoundError as exc:
        assert "Missing CVAT input images" in str(exc)
    else:
        raise AssertionError("Missing image files should fail validation")
