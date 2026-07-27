from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

from openslit.workflow.config import load_workflow_config


def _schema(path: Path) -> None:
    classes = []
    for class_id, name in enumerate(
        [
            "background",
            "pupil",
            "iris",
            "reflection",
            "slit_beam",
            "eyelid",
            "eyelash",
            "uncertain",
        ]
    ):
        classes.append(
            {
                "id": class_id,
                "name": name,
                "display_name": name,
                "color_rgb": [class_id, class_id + 1, class_id + 2],
                "required_per_gradable_image": name in {"pupil", "iris"},
                "description": name,
            }
        )
    path.write_text(
        json.dumps(
            {
                "protocol_name": "test",
                "protocol_version": "1.0.0",
                "task_type": "single-label semantic segmentation",
                "class_precedence_high_to_low": [
                    "uncertain",
                    "eyelash",
                    "eyelid",
                    "reflection",
                    "slit_beam",
                    "pupil",
                    "iris",
                    "background",
                ],
                "classes": classes,
                "required_manifest_columns": [
                    "image_id",
                    "image_file",
                    "mask_file",
                    "annotator_id",
                    "protocol_version",
                ],
                "optional_manifest_columns": ["gradable"],
                "forbidden_shared_fields": ["patient_name"],
            }
        ),
        encoding="utf-8",
    )


def _workflow(tmp_path: Path) -> Path:
    pilot = tmp_path / "pilot"
    images = pilot / "drive_upload" / "images"
    shared = pilot / "shared"
    images.mkdir(parents=True)
    shared.mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(images / "PILOT-I001.jpg")
    pd.DataFrame(
        [
            {
                "blinded_image_id": "PILOT-I001",
                "blinded_patient_id": "PILOT-P001",
                "image_file": "PILOT-I001.jpg",
                "drive_url": "",
            }
        ]
    ).to_csv(shared / "pilot_image_index.csv", index=False)
    pd.DataFrame(
        [
            {
                "blinded_image_id": "PILOT-I001",
                "image_file": "PILOT-I001.jpg",
                "independent_double_annotation": "true",
            }
        ]
    ).to_csv(shared / "mask_task_manifest.csv", index=False)
    for grader in ["grader_01", "grader_02"]:
        (shared / f"{grader}_quality_grading.xlsx").write_bytes(b"placeholder")
    _schema(tmp_path / "schema.json")
    config = {
        "workflow_id": "test_v1",
        "pilot_dir": "pilot",
        "image_dir": "pilot/drive_upload/images",
        "image_index_path": "pilot/shared/pilot_image_index.csv",
        "mask_manifest_path": "pilot/shared/mask_task_manifest.csv",
        "schema_path": "schema.json",
        "state_path": "pilot/workflow/state.json",
        "graders": [
            {
                "grader_id": "grader_01",
                "email": "a@hospital.org",
                "cvat_username": "grader_a",
                "workbook_path": "pilot/shared/grader_01_quality_grading.xlsx",
            },
            {
                "grader_id": "grader_02",
                "email": "b@hospital.org",
                "cvat_username": "grader_b",
                "workbook_path": "pilot/shared/grader_02_quality_grading.xlsx",
            },
        ],
        "adjudicator": {
            "adjudicator_id": "senior",
            "email": "senior@hospital.org",
            "cvat_username": "senior",
        },
        "drive": {
            "parent_folder_id": "folder-id",
            "root_folder_name": "Pilot",
        },
        "cvat": {
            "host": "http://localhost:8080",
            "project_name_template": "Pilot - {grader_id}",
            "task_name_template": "Pilot - {grader_id} - v{version}",
        },
    }
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_workflow_configuration_validates(tmp_path: Path) -> None:
    config = load_workflow_config(_workflow(tmp_path))
    summary = config.validate()
    assert summary["selected_images"] == 1
    assert config.cvat.project_name(config.graders[0]) == "Pilot - grader_01"
    assert config.selected_image_paths()[0].name == "PILOT-I001.jpg"
