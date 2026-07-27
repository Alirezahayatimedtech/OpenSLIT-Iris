import json
from pathlib import Path

import pandas as pd

from openslit.ai.active_learning import select_active_learning_batch
from openslit.ai.config import load_ai_workflow_config


def _config(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "workflow_config_path": "workflow.json",
                "consensus_manifest_path": "consensus.csv",
                "image_dir": "images",
                "output_dir": "output",
                "schema_path": "schema.json",
                "random_seed": 17,
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
                    "batch_size": 8,
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
    return path


def test_active_learning_excludes_test_and_labelled_images(tmp_path):
    config = load_ai_workflow_config(_config(tmp_path / "ai.json"))
    candidates = pd.DataFrame(
        {
            "image_id": [f"I{i:02d}" for i in range(20)],
            "split": ["test" if i < 3 else "train" for i in range(20)],
            "labelled": ["true" if i == 3 else "false" for i in range(20)],
            "uncertainty_score": [str(i / 20) for i in range(20)],
            "model_disagreement_score": [str((20 - i) / 20) for i in range(20)],
        }
    )
    candidate_path = tmp_path / "candidates.csv"
    output_path = tmp_path / "batch.csv"
    candidates.to_csv(candidate_path, index=False)
    result = select_active_learning_batch(config, candidate_path, output_path=output_path)
    selected = pd.read_csv(output_path, dtype=str)
    assert result["batch_size"] == 8
    assert not set(selected["image_id"]).intersection({"I00", "I01", "I02", "I03"})
    assert "random_control" in set(selected["selection_reason"])
