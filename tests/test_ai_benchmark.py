from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from openslit.ai.benchmark import compare_source_to_consensus
from openslit.ai.config import load_ai_workflow_config
from openslit.ai.cvat_assist import approve_model_for_assistance
from openslit.workflow.state import WorkflowState


def _write_schema(path: Path) -> None:
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
    path.write_text(
        json.dumps(
            {
                "protocol_name": "test",
                "protocol_version": "1.0.0",
                "task_type": "single-label semantic segmentation",
                "class_precedence_high_to_low": list(reversed(names)),
                "classes": [
                    {
                        "id": class_id,
                        "name": name,
                        "display_name": name,
                        "color_rgb": [class_id, class_id, class_id],
                        "required_per_gradable_image": name in {"pupil", "iris"},
                        "description": name,
                    }
                    for class_id, name in enumerate(names)
                ],
                "required_manifest_columns": [],
                "optional_manifest_columns": [],
                "forbidden_shared_fields": [],
            }
        ),
        encoding="utf-8",
    )


def _ai_config(tmp_path: Path):
    _write_schema(tmp_path / "schema.json")
    path = tmp_path / "ai.json"
    path.write_text(
        json.dumps(
            {
                "workflow_config_path": "workflow.json",
                "consensus_manifest_path": "consensus.csv",
                "image_dir": "images",
                "output_dir": "output",
                "schema_path": "schema.json",
                "random_seed": 7,
                "split": {
                    "group_column": "patient_id",
                    "train_fraction": 0.6,
                    "validation_fraction": 0.2,
                    "test_fraction": 0.2,
                },
                "models": [
                    {
                        "model_id": "unet",
                        "family": "segmentation_models_pytorch",
                        "role": "baseline",
                        "enabled": True,
                        "parameters": {},
                    }
                ],
                "active_learning": {
                    "batch_size": 4,
                    "uncertainty_fraction": 0.4,
                    "model_disagreement_fraction": 0.25,
                    "diversity_fraction": 0.2,
                    "random_fraction": 0.15,
                    "minimum_random_images": 1,
                    "exclude_test_set": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return load_ai_workflow_config(path)


def test_benchmark_accepts_test_only_source_manifest(tmp_path: Path) -> None:
    config = _ai_config(tmp_path)
    consensus_masks = tmp_path / "consensus_masks"
    source_masks = tmp_path / "source_masks"
    consensus_masks.mkdir()
    source_masks.mkdir()
    consensus = pd.DataFrame(
        {
            "image_id": ["I1", "I2", "I3"],
            "image_file": ["I1.jpg", "I2.jpg", "I3.jpg"],
            "mask_file": ["I1.png", "I2.png", "I3.png"],
        }
    )
    consensus.to_csv(config.consensus_manifest_path, index=False)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    mask[1:7, 1:7][mask[1:7, 1:7] == 0] = 2
    for image_id in consensus["image_id"]:
        Image.fromarray(mask).save(consensus_masks / f"{image_id}.png")
    Image.fromarray(mask).save(source_masks / "I2_prediction.png")
    source_manifest = tmp_path / "test_predictions.csv"
    pd.DataFrame(
        [
            {
                "image_id": "I2",
                "image_file": "I2.jpg",
                "mask_file": "I2_prediction.png",
                "split": "test",
            }
        ]
    ).to_csv(source_manifest, index=False)

    summary = compare_source_to_consensus(
        config,
        source_name="unet",
        source_manifest_path=source_manifest,
        source_masks_dir=source_masks,
        consensus_masks_dir=consensus_masks,
    )

    assert summary["images"] == 1
    assert summary["split"] == "test"
    assert summary["split_counts"] == {"test": 1}
    assert summary["macro_foreground_dice_mean"] == pytest.approx(1.0)


def test_model_approval_requires_test_split_benchmark(tmp_path: Path) -> None:
    state = WorkflowState(path=tmp_path / "state.json", data={"events": []})
    summary_path = tmp_path / "summary.json"
    base_summary = {
        "source": "unet",
        "reference": "senior_consensus",
        "images": 3,
        "macro_foreground_dice_mean": 0.9,
    }
    summary_path.write_text(
        json.dumps({**base_summary, "split": "validation"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="untouched test split"):
        approve_model_for_assistance(
            state,
            "unet",
            summary_path,
            "senior",
        )

    summary_path.write_text(
        json.dumps({**base_summary, "split": "test"}),
        encoding="utf-8",
    )
    result = approve_model_for_assistance(
        state,
        "unet",
        summary_path,
        "senior",
    )
    assert result["approved_for_assistance"] is True
