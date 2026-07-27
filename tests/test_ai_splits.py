import json

import pandas as pd

from openslit.ai.config import load_ai_workflow_config
from openslit.ai.splits import create_grouped_splits, verify_split_manifest


def test_grouped_split_has_no_patient_leakage(tmp_path):
    consensus = pd.DataFrame(
        {
            "image_id": [f"I{i}" for i in range(12)],
            "image_file": [f"I{i}.jpg" for i in range(12)],
            "mask_file": [f"I{i}_mask.png" for i in range(12)],
            "patient_id": [f"P{i // 2}" for i in range(12)],
        }
    )
    consensus_path = tmp_path / "consensus.csv"
    consensus.to_csv(consensus_path, index=False)
    config_path = tmp_path / "ai.json"
    config_path.write_text(
        json.dumps(
            {
                "workflow_config_path": "workflow.json",
                "consensus_manifest_path": "consensus.csv",
                "image_dir": "images",
                "output_dir": "output",
                "schema_path": "schema.json",
                "random_seed": 19,
                "split": {
                    "group_column": "patient_id",
                    "train_fraction": 0.5,
                    "validation_fraction": 0.25,
                    "test_fraction": 0.25,
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
    config = load_ai_workflow_config(config_path)
    output = tmp_path / "splits.csv"
    create_grouped_splits(config, output)
    summary = verify_split_manifest(output, "patient_id")
    assert summary["leakage"] is False
    split_table = pd.read_csv(output, dtype=str)
    assert split_table.groupby("patient_id")["split"].nunique().max() == 1
