from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openslit.annotation.schema import load_annotation_schema
from openslit.features.color import extract_color_features, gray_world_normalize
from openslit.features.config import load_feature_config
from openslit.features.extract import extract_feature_table
from openslit.features.geometry import extract_geometry_features
from openslit.features.normalization import normalize_iris
from openslit.features.quality import assess_feature_eligibility
from openslit.features.repeatability import analyze_repeatability
from openslit.features.texture import extract_texture_features


def write_schema(path: Path) -> Path:
    classes = [
        (0, "background"),
        (1, "pupil"),
        (2, "iris"),
        (3, "reflection"),
        (4, "slit_beam"),
        (5, "eyelid"),
        (6, "eyelash"),
        (7, "uncertain"),
    ]
    path.write_text(
        json.dumps(
            {
                "protocol_name": "test",
                "protocol_version": "1.0",
                "task_type": "single-label semantic segmentation",
                "class_precedence_high_to_low": [name for _, name in reversed(classes)],
                "classes": [
                    {
                        "id": class_id,
                        "name": name,
                        "display_name": name,
                        "color_rgb": [
                            class_id * 20,
                            class_id * 20 + 1,
                            class_id * 20 + 2,
                        ],
                        "required_per_gradable_image": name in {"pupil", "iris"},
                        "description": name,
                    }
                    for class_id, name in classes
                ],
                "required_manifest_columns": [],
                "optional_manifest_columns": [],
                "forbidden_shared_fields": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def synthetic_eye(size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices((size, size))
    center = (size - 1) / 2
    radius = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[(radius >= 18) & (radius <= 48)] = 2
    mask[radius < 18] = 1
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 0] = np.clip(80 + radius, 0, 255)
    image[..., 1] = np.clip(120 + 0.5 * radius, 0, 255)
    image[..., 2] = np.clip(150 - 0.4 * radius, 0, 255)
    return image, mask


def test_geometry_and_polar_normalization(tmp_path: Path) -> None:
    schema = load_annotation_schema(write_schema(tmp_path / "schema.json"))
    image, mask = synthetic_eye()
    geometry = extract_geometry_features(mask, schema)
    assert geometry["pupil_area_px"] > 0
    assert geometry["iris_visible_area_px"] > geometry["pupil_area_px"]
    assert geometry["pupil_circularity"] > 0.5
    polar = normalize_iris(
        image,
        mask,
        schema,
        angular_samples=180,
        radial_samples=32,
    )
    assert polar.image.shape == (32, 180, 3)
    assert polar.valid_angle_fraction > 0.95
    assert polar.valid_pixel_fraction > 0.8


def test_color_and_texture_features(tmp_path: Path) -> None:
    schema = load_annotation_schema(write_schema(tmp_path / "schema.json"))
    image, mask = synthetic_eye()
    polar = normalize_iris(
        image,
        mask,
        schema,
        angular_samples=180,
        radial_samples=32,
    )
    color = extract_color_features(image, mask == 2, polar)
    texture = extract_texture_features(polar)
    assert color["iris_color_pixel_count"] > 0
    assert color["iris_normalized_lab_l_mean"] is not None
    assert texture["texture_valid_pixels"] > 0
    assert texture["glcm_contrast_mean"] is not None


def test_gray_world_balances_channel_means() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[..., 0] = 30
    image[..., 1] = 60
    image[..., 2] = 120
    normalized = gray_world_normalize(
        image,
        np.ones((10, 10), dtype=bool),
    )
    means = normalized.reshape(-1, 3).mean(axis=0)
    assert float(means.max() - means.min()) <= 1.0


def test_quality_gate(tmp_path: Path) -> None:
    schema = load_annotation_schema(write_schema(tmp_path / "schema.json"))
    image, mask = synthetic_eye()
    polar = normalize_iris(image, mask, schema)
    config_path = tmp_path / "feature.json"
    config_path.write_text(
        json.dumps(
            {
                "workflow_config_path": "workflow.json",
                "feature_version": "1.0",
                "image_dir": "images",
                "manifest_path": "manifest.csv",
                "masks_dir": "masks",
                "schema_path": "schema.json",
                "output_dir": "output",
                "source_requirements": {
                    "allowed_review_status": ["senior_consensus"],
                    "require_gradable": True,
                    "require_mask_sha256": True,
                },
                "normalization": {},
                "color": {},
                "texture": {},
                "quality": {
                    "minimum_visible_iris_pixels": 100,
                    "minimum_valid_polar_fraction": 0.2,
                    "maximum_uncertain_fraction": 0.2,
                    "maximum_artifact_fraction": 0.4,
                    "minimum_laplacian_variance": 0,
                },
                "preview": {},
                "drive": {},
            }
        ),
        encoding="utf-8",
    )
    config = load_feature_config(config_path)
    result = assess_feature_eligibility(
        image,
        mask,
        schema,
        polar,
        {
            "review_status": "senior_consensus",
            "gradable": "true",
            "mask_sha256": "abc",
        },
        config.source_requirements,
        config.quality,
    )
    assert result.accepted


def test_repeatability_outputs(tmp_path: Path) -> None:
    rows = []
    for group in range(5):
        rows.extend(
            [
                {
                    "image_id": f"I{group}a",
                    "repeat_group_id": f"P{group}",
                    "feature_gate_passed": "true",
                    "pupil_area_px": 100 + group,
                },
                {
                    "image_id": f"I{group}b",
                    "repeat_group_id": f"P{group}",
                    "feature_gate_passed": "true",
                    "pupil_area_px": 101 + group,
                },
            ]
        )
    path = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = analyze_repeatability(
        path,
        "repeat_group_id",
        tmp_path / "repeat",
    )
    summary = pd.read_csv(result["summary_path"])
    assert "icc_2_1" in summary.columns
    assert len(summary) == 1


def test_feature_run_cannot_overwrite_existing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_path = write_schema(tmp_path / "schema.json")
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "image_id": "I1",
                "image_file": "I1.jpg",
                "mask_file": "I1.png",
                "review_status": "senior_consensus",
                "gradable": "true",
            }
        ]
    ).to_csv(manifest_path, index=False)
    output_dir = tmp_path / "output"
    run_dir = output_dir / "fixed_run"
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    config = SimpleNamespace(
        validate=lambda require_runtime_files: {},
        schema_path=schema_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        feature_version="1.0",
        source_requirements=SimpleNamespace(require_mask_sha256=False),
    )
    monkeypatch.setattr(
        "openslit.features.extract._feature_state",
        lambda selected: (SimpleNamespace(workflow_id="pilot"), object()),
    )

    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        extract_feature_table(config, run_id="fixed_run")
    assert sentinel.read_text(encoding="utf-8") == "preserve"
