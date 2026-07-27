import json

import numpy as np

from openslit.ai.quality import assess_mask_quality
from openslit.annotation.schema import load_annotation_schema


def _schema(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(
        json.dumps(
            {
                "protocol_name": "test",
                "protocol_version": "1.0",
                "task_type": "single-label semantic segmentation",
                "supported_input": "test",
                "mask_format": {
                    "type": "indexed_png",
                    "background_value": 0,
                    "one_class_per_pixel": True,
                },
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
                "classes": [
                    {"id": 0, "name": "background", "display_name": "Background", "color_rgb": [0, 0, 0], "required_per_gradable_image": False, "description": ""},
                    {"id": 1, "name": "pupil", "display_name": "Pupil", "color_rgb": [1, 1, 1], "required_per_gradable_image": True, "description": ""},
                    {"id": 2, "name": "iris", "display_name": "Iris", "color_rgb": [2, 2, 2], "required_per_gradable_image": True, "description": ""},
                    {"id": 3, "name": "reflection", "display_name": "Reflection", "color_rgb": [3, 3, 3], "required_per_gradable_image": False, "description": ""},
                    {"id": 4, "name": "slit_beam", "display_name": "Slit", "color_rgb": [4, 4, 4], "required_per_gradable_image": False, "description": ""},
                    {"id": 5, "name": "eyelid", "display_name": "Eyelid", "color_rgb": [5, 5, 5], "required_per_gradable_image": False, "description": ""},
                    {"id": 6, "name": "eyelash", "display_name": "Eyelash", "color_rgb": [6, 6, 6], "required_per_gradable_image": False, "description": ""},
                    {"id": 7, "name": "uncertain", "display_name": "Uncertain", "color_rgb": [7, 7, 7], "required_per_gradable_image": False, "description": ""},
                ],
                "required_manifest_columns": [],
                "optional_manifest_columns": [],
                "forbidden_shared_fields": [],
            }
        ),
        encoding="utf-8",
    )
    return load_annotation_schema(path)


def test_quality_gate_accepts_simple_plausible_mask(tmp_path):
    schema = _schema(tmp_path)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[3:17, 3:17] = 2
    mask[8:12, 8:12] = 1
    result = assess_mask_quality(mask, schema)
    assert result.accepted
    assert result.flags == ()


def test_quality_gate_rejects_missing_pupil(tmp_path):
    schema = _schema(tmp_path)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[3:17, 3:17] = 2
    result = assess_mask_quality(mask, schema)
    assert not result.accepted
    assert "MISSING_PUPIL" in result.flags
