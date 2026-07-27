import json

import pandas as pd

from openslit.ai.config import load_ai_workflow_config
from openslit.ai.productivity import create_crossover_plan, summarize_crossover_results


def _config(tmp_path):
    path = tmp_path / "ai.json"
    path.write_text(
        json.dumps(
            {
                "workflow_config_path": "workflow.json",
                "consensus_manifest_path": "consensus.csv",
                "image_dir": "images",
                "output_dir": "output",
                "schema_path": "schema.json",
                "random_seed": 11,
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


def test_crossover_assigns_each_image_to_both_arms(tmp_path):
    config = _config(tmp_path)
    batch = pd.DataFrame(
        {
            "image_id": ["I1", "I2", "I3", "I4"],
            "image_file": ["1.jpg", "2.jpg", "3.jpg", "4.jpg"],
        }
    )
    batch_path = tmp_path / "batch.csv"
    plan_path = tmp_path / "plan.csv"
    batch.to_csv(batch_path, index=False)
    create_crossover_plan(config, batch_path, ("g1", "g2"), plan_path)
    plan = pd.read_csv(plan_path, dtype=str)
    assert len(plan) == 8
    for _, group in plan.groupby("image_id"):
        assert set(group["arm"]) == {"MANUAL_BLANK", "AI_ASSISTED"}


def test_productivity_summary(tmp_path):
    completed = pd.DataFrame(
        {
            "image_id": ["I1", "I1", "I2", "I2"],
            "grader_id": ["g1", "g2", "g1", "g2"],
            "arm": ["AI_ASSISTED", "MANUAL_BLANK", "MANUAL_BLANK", "AI_ASSISTED"],
            "active_annotation_seconds": ["20", "40", "50", "25"],
            "correction_category": [
                "MINOR_CORRECTION",
                "ACCEPTED_WITHOUT_CHANGE",
                "MAJOR_CORRECTION",
                "MINOR_CORRECTION",
            ],
            "sent_to_senior": ["false", "false", "true", "false"],
        }
    )
    path = tmp_path / "completed.csv"
    completed.to_csv(path, index=False)
    summary = summarize_crossover_results(path, tmp_path / "summary")
    assert summary["assignments"] == 4
    assert len(summary["arm_summary"]) == 2
