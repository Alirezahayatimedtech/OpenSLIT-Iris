import json
from pathlib import Path

import pytest

from openslit.ai.config import load_ai_workflow_config


def _write_config(path: Path, random_fraction: float = 0.15) -> None:
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
                    "train_fraction": 0.7,
                    "validation_fraction": 0.15,
                    "test_fraction": 0.15,
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
                "training": {},
                "uncertainty": {},
                "evaluation": {},
                "ai_assisted_cvat": {},
                "active_learning": {
                    "batch_size": 20,
                    "uncertainty_fraction": 0.4,
                    "model_disagreement_fraction": 0.25,
                    "diversity_fraction": 0.2,
                    "random_fraction": random_fraction,
                    "minimum_random_images": 2,
                    "exclude_test_set": True,
                },
            }
        ),
        encoding="utf-8",
    )


def test_ai_configuration_only_validation(tmp_path):
    path = tmp_path / "ai.json"
    _write_config(path)
    config = load_ai_workflow_config(path)
    summary = config.validate(require_runtime_files=False)
    assert summary["enabled_models"] == ["unet"]
    assert summary["split"]["test"] == 0.15


def test_active_learning_fractions_must_sum_to_one(tmp_path):
    path = tmp_path / "ai.json"
    _write_config(path, random_fraction=0.25)
    config = load_ai_workflow_config(path)
    with pytest.raises(ValueError, match="fractions must sum to 1"):
        config.validate(require_runtime_files=False)
