from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from openslit.annotation.schema import load_annotation_schema
from openslit.annotation.validate_masks import validate_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "configs" / "annotation_schema_v1.json"


def test_annotation_schema_loads() -> None:
    schema = load_annotation_schema(SCHEMA_PATH)
    assert schema.protocol_version == "1.0.0"
    assert schema.class_ids == frozenset(range(8))
    assert schema.class_by_name["pupil"].id == 1
    assert schema.class_by_name["iris"].id == 2
    assert schema.required_class_ids == frozenset({1, 2})


def test_schema_rejects_duplicate_class_ids(tmp_path: Path) -> None:
    raw = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    raw["classes"][1]["id"] = raw["classes"][0]["id"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    try:
        load_annotation_schema(path)
    except ValueError as exc:
        assert "IDs must be unique" in str(exc)
    else:
        raise AssertionError("Duplicate class IDs should fail validation")


def test_valid_mask_dataset(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[4:16, 4:16] = 2
    mask[8:12, 8:12] = 1

    Image.fromarray(image).save(images / "image.png")
    Image.fromarray(mask, mode="L").save(masks / "mask.png")

    manifest = pd.DataFrame(
        [
            {
                "image_id": "PILOT-I001",
                "image_file": "image.png",
                "mask_file": "mask.png",
                "annotator_id": "grader_01",
                "protocol_version": "1.0.0",
                "gradable": "true",
            }
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    report_path = tmp_path / "report.csv"
    manifest.to_csv(manifest_path, index=False)

    summary = validate_dataset(
        SCHEMA_PATH, manifest_path, images, masks, report_path
    )

    assert summary == {"images": 1, "errors": 0, "warnings": 0, "valid_images": 1}
    assert report_path.is_file()


def test_invalid_mask_class_is_reported(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:9, 1:9] = 2
    mask[3:7, 3:7] = 99

    Image.fromarray(image).save(images / "image.png")
    Image.fromarray(mask, mode="L").save(masks / "mask.png")

    manifest = pd.DataFrame(
        [
            {
                "image_id": "PILOT-I001",
                "image_file": "image.png",
                "mask_file": "mask.png",
                "annotator_id": "grader_01",
                "protocol_version": "1.0.0",
                "gradable": "true",
            }
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    report_path = tmp_path / "report.csv"
    manifest.to_csv(manifest_path, index=False)

    summary = validate_dataset(
        SCHEMA_PATH, manifest_path, images, masks, report_path
    )
    report = pd.read_csv(report_path)

    assert summary["errors"] >= 1
    assert "invalid_class_ids" in set(report["code"])
