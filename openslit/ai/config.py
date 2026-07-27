"""Configuration for AI benchmarking, CVAT assistance, and active learning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    role: str
    enabled: bool
    parameters: dict[str, Any]


@dataclass(frozen=True)
class SplitSpec:
    group_column: str
    train_fraction: float
    validation_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class ActiveLearningSpec:
    batch_size: int
    uncertainty_fraction: float
    model_disagreement_fraction: float
    diversity_fraction: float
    random_fraction: float
    minimum_random_images: int
    exclude_test_set: bool


@dataclass(frozen=True)
class AIWorkflowConfig:
    config_path: Path
    workflow_config_path: Path
    consensus_manifest_path: Path
    image_dir: Path
    output_dir: Path
    schema_path: Path
    random_seed: int
    split: SplitSpec
    models: tuple[ModelSpec, ...]
    training: dict[str, Any]
    uncertainty: dict[str, Any]
    evaluation: dict[str, Any]
    ai_assisted_cvat: dict[str, Any]
    active_learning: ActiveLearningSpec

    @property
    def enabled_models(self) -> tuple[ModelSpec, ...]:
        return tuple(model for model in self.models if model.enabled)

    def model(self, model_id: str) -> ModelSpec:
        matches = [item for item in self.models if item.model_id == model_id]
        if len(matches) != 1:
            raise KeyError(f"Unknown or duplicated model_id: {model_id}")
        return matches[0]

    def validate(self, require_runtime_files: bool = True) -> dict[str, Any]:
        fractions = (
            self.split.train_fraction,
            self.split.validation_fraction,
            self.split.test_fraction,
        )
        if any(value <= 0 or value >= 1 for value in fractions):
            raise ValueError("All split fractions must be between 0 and 1")
        if abs(sum(fractions) - 1.0) > 1e-8:
            raise ValueError("Train, validation, and test fractions must sum to 1")
        if not self.models:
            raise ValueError("At least one AI model must be configured")
        model_ids = [item.model_id for item in self.models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("AI model IDs must be unique")
        if not self.enabled_models:
            raise ValueError("At least one AI model must be enabled")

        weights = (
            self.active_learning.uncertainty_fraction,
            self.active_learning.model_disagreement_fraction,
            self.active_learning.diversity_fraction,
            self.active_learning.random_fraction,
        )
        if any(value < 0 or value > 1 for value in weights):
            raise ValueError("Active-learning fractions must be within 0 and 1")
        if abs(sum(weights) - 1.0) > 1e-8:
            raise ValueError("Active-learning fractions must sum to 1")
        if self.active_learning.batch_size < 1:
            raise ValueError("Active-learning batch size must be positive")
        if self.active_learning.minimum_random_images < 0:
            raise ValueError("minimum_random_images cannot be negative")

        missing: list[str] = []
        if require_runtime_files:
            for path in [
                self.workflow_config_path,
                self.consensus_manifest_path,
                self.schema_path,
            ]:
                if not path.is_file():
                    missing.append(str(path))
            if not self.image_dir.is_dir():
                missing.append(str(self.image_dir))
            if missing:
                raise FileNotFoundError(f"Missing AI workflow inputs: {missing}")

        return {
            "enabled_models": [item.model_id for item in self.enabled_models],
            "model_roles": {item.model_id: item.role for item in self.models},
            "split": {
                "group_column": self.split.group_column,
                "train": self.split.train_fraction,
                "validation": self.split.validation_fraction,
                "test": self.split.test_fraction,
            },
            "active_learning_batch_size": self.active_learning.batch_size,
            "consensus_manifest_path": str(self.consensus_manifest_path),
            "output_dir": str(self.output_dir),
            "runtime_files_checked": require_runtime_files,
        }


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_ai_workflow_config(path: str | Path) -> AIWorkflowConfig:
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    split_raw = raw["split"]
    active_raw = raw["active_learning"]
    models = tuple(
        ModelSpec(
            model_id=str(item["model_id"]).strip(),
            family=str(item["family"]).strip(),
            role=str(item["role"]).strip(),
            enabled=bool(item.get("enabled", True)),
            parameters=dict(item.get("parameters", {})),
        )
        for item in raw.get("models", [])
    )
    return AIWorkflowConfig(
        config_path=config_path,
        workflow_config_path=_resolve(base, str(raw["workflow_config_path"])),
        consensus_manifest_path=_resolve(base, str(raw["consensus_manifest_path"])),
        image_dir=_resolve(base, str(raw["image_dir"])),
        output_dir=_resolve(base, str(raw["output_dir"])),
        schema_path=_resolve(base, str(raw["schema_path"])),
        random_seed=int(raw.get("random_seed", 0)),
        split=SplitSpec(
            group_column=str(split_raw.get("group_column", "blinded_patient_id")),
            train_fraction=float(split_raw["train_fraction"]),
            validation_fraction=float(split_raw["validation_fraction"]),
            test_fraction=float(split_raw["test_fraction"]),
        ),
        models=models,
        training=dict(raw.get("training", {})),
        uncertainty=dict(raw.get("uncertainty", {})),
        evaluation=dict(raw.get("evaluation", {})),
        ai_assisted_cvat=dict(raw.get("ai_assisted_cvat", {})),
        active_learning=ActiveLearningSpec(
            batch_size=int(active_raw["batch_size"]),
            uncertainty_fraction=float(active_raw["uncertainty_fraction"]),
            model_disagreement_fraction=float(
                active_raw["model_disagreement_fraction"]
            ),
            diversity_fraction=float(active_raw["diversity_fraction"]),
            random_fraction=float(active_raw["random_fraction"]),
            minimum_random_images=int(active_raw.get("minimum_random_images", 0)),
            exclude_test_set=bool(active_raw.get("exclude_test_set", True)),
        ),
    )
