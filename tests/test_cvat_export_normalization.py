from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from openslit.annotation.schema import load_annotation_schema
from openslit.workflow.cvat_bridge import normalize_segmentation_export


def _schema(path: Path) -> None:
    names = [
        "background",
        "pupil",
        "iris",
        "reflection",
        "slit_beam",
        "eyelid",
        "eyelash",
        "uncertain",
    ]
    colors = [
        [0, 0, 0],
        [32, 76, 255],
        [35, 190, 95],
        [255, 220, 30],
        [255, 65, 65],
        [160, 80, 220],
        [255, 140, 25],
        [145, 145, 145],
    ]
    path.write_text(
        json.dumps(
            {
                "protocol_name": "test",
                "protocol_version": "1.0.0",
                "task_type": "single-label semantic segmentation",
                "class_precedence_high_to_low": list(reversed(names)),
                "classes": [
                    {
                        "id": index,
                        "name": name,
                        "display_name": name,
                        "color_rgb": colors[index],
                        "required_per_gradable_image": name in {"pupil", "iris"},
                        "description": name,
                    }
                    for index, name in enumerate(names)
                ],
                "required_manifest_columns": [
                    "image_id",
                    "image_file",
                    "mask_file",
                    "annotator_id",
                    "protocol_version",
                ],
                "optional_manifest_columns": ["gradable"],
                "forbidden_shared_fields": [],
            }
        ),
        encoding="utf-8",
    )


def test_rgb_cvat_export_is_converted_to_protocol_ids(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    _schema(schema_path)
    schema = load_annotation_schema(schema_path)
    export_root = tmp_path / "export"
    segmentation = export_root / "SegmentationClass"
    segmentation.mkdir(parents=True)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[1:7, 1:7] = [35, 190, 95]
    rgb[3:5, 3:5] = [32, 76, 255]
    Image.fromarray(rgb).save(segmentation / "PILOT-I001.png")
    (export_root / "labelmap.txt").write_text(
        "background:0,0,0::\npupil:32,76,255::\niris:35,190,95::\n",
        encoding="utf-8",
    )
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in export_root.rglob("*"):
            if path.is_file():
                zipped.write(path, path.relative_to(export_root))

    selected = pd.DataFrame(
        [{"blinded_image_id": "PILOT-I001", "image_file": "PILOT-I001.jpg"}]
    )
    masks_dir, manifest_path = normalize_segmentation_export(
        archive,
        tmp_path / "normalized",
        selected,
        schema,
        "grader_01",
    )
    with Image.open(masks_dir / "PILOT-I001_mask.png") as mask_image:
        mask = np.asarray(mask_image)
    assert set(np.unique(mask)) == {0, 1, 2}
    manifest = pd.read_csv(manifest_path)
    assert manifest.loc[0, "annotator_id"] == "grader_01"
