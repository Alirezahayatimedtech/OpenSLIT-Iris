"""Config-driven training for the two primary OpenSLIT segmentation baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from openslit.annotation.schema import load_annotation_schema

from .config import AIWorkflowConfig
from .data import SegmentationDataset, load_split_table
from .registry import build_model, extract_logits


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Install AI dependencies with: python -m pip install -e '.[ai]'"
        ) from exc
    return torch, functional, DataLoader


def _soft_dice_loss(logits: Any, target: Any, num_classes: int, functional: Any) -> Any:
    probabilities = functional.softmax(logits, dim=1)
    one_hot = functional.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)
    one_hot = one_hot.to(dtype=probabilities.dtype)
    dimensions = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + one_hot.sum(dim=dimensions)
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return 1.0 - dice.mean()


def _forward_logits(model: Any, images: Any, target_size: tuple[int, int], functional: Any) -> Any:
    logits = extract_logits(model(images))
    if tuple(logits.shape[-2:]) != tuple(target_size):
        logits = functional.interpolate(
            logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    return logits


def train_model(
    config: AIWorkflowConfig,
    model_id: str,
    split_manifest_path: Path,
    consensus_masks_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Train one baseline and save the best validation checkpoint.

    The untouched test set is never loaded by this function.
    """

    torch, functional, DataLoader = _require_torch()
    spec = config.model(model_id)
    if not spec.enabled:
        raise ValueError(f"Model {model_id!r} is disabled in configuration")
    if spec.family not in {"segmentation_models_pytorch", "huggingface_segformer"}:
        raise ValueError(f"Model {model_id!r} is not trainable by this command")

    schema = load_annotation_schema(config.schema_path)
    num_classes = len(schema.class_ids)
    input_size = int(spec.parameters.get("input_size", 512))
    train_table = load_split_table(split_manifest_path, "train")
    validation_table = load_split_table(split_manifest_path, "validation")
    train_dataset = SegmentationDataset(
        train_table,
        config.image_dir,
        consensus_masks_dir,
        input_size,
    )
    validation_dataset = SegmentationDataset(
        validation_table,
        config.image_dir,
        consensus_masks_dir,
        input_size,
    )

    batch_size = int(config.training.get("batch_size", 4))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(spec, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.get("learning_rate", 1e-4)),
        weight_decay=float(config.training.get("weight_decay", 1e-5)),
    )
    use_amp = bool(config.training.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    epochs = int(config.training.get("epochs", 100))
    patience = int(config.training.get("early_stopping_patience", 15))
    output_dir = output_dir or (config.output_dir / "models" / model_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_checkpoint.pt"
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = _forward_logits(
                    model,
                    images,
                    tuple(targets.shape[-2:]),
                    functional,
                )
                cross_entropy = functional.cross_entropy(logits, targets)
                dice_loss = _soft_dice_loss(logits, targets, num_classes, functional)
                loss = cross_entropy + dice_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for batch in validation_loader:
                images = batch["image"].to(device)
                targets = batch["mask"].to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = _forward_logits(
                        model,
                        images,
                        tuple(targets.shape[-2:]),
                        functional,
                    )
                    cross_entropy = functional.cross_entropy(logits, targets)
                    dice_loss = _soft_dice_loss(logits, targets, num_classes, functional)
                    loss = cross_entropy + dice_loss
                validation_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_id": model_id,
                    "model_family": spec.family,
                    "model_parameters": spec.parameters,
                    "protocol_version": schema.protocol_version,
                    "class_ids": sorted(schema.class_ids),
                    "input_size": input_size,
                    "state_dict": model.state_dict(),
                    "best_validation_loss": best_validation,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    summary = {
        "model_id": model_id,
        "device": str(device),
        "epochs_completed": len(history),
        "best_validation_loss": best_validation,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "test_set_used": False,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
