"""Inference, probability export, uncertainty maps, and quality gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from openslit.annotation.schema import load_annotation_schema

from .config import AIWorkflowConfig
from .data import IMAGENET_MEAN, IMAGENET_STD
from .metrics import predictive_entropy
from .quality import assess_mask_quality
from .registry import build_model, extract_logits


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Install AI dependencies with: python -m pip install -e '.[ai]'"
        ) from exc
    return torch, functional


def _prepare_image(path: Path, input_size: int, torch: Any) -> tuple[Any, tuple[int, int]]:
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        original_size = (rgb.height, rgb.width)
        resized = rgb.resize((input_size, input_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, original_size


def run_inference(
    config: AIWorkflowConfig,
    model_id: str,
    checkpoint_path: Path,
    split_manifest_path: Path,
    split: str = "test",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run frozen-model inference and save masks, uncertainty, and reject flags."""

    torch, functional = _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("model_id") != model_id:
        raise ValueError("Checkpoint model_id does not match requested model")
    spec = config.model(model_id)
    schema = load_annotation_schema(config.schema_path)
    model = build_model(spec, num_classes=len(schema.class_ids))
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    input_size = int(checkpoint.get("input_size", spec.parameters.get("input_size", 512)))

    table = pd.read_csv(split_manifest_path, dtype=str, keep_default_na=False)
    required = {"image_id", "image_file", "split"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    table = table[table["split"] == split].copy()
    if table.empty:
        raise ValueError(f"No images found for split {split!r}")

    output_dir = output_dir or (config.output_dir / "predictions" / model_id / split)
    masks_dir = output_dir / "masks"
    probabilities_dir = output_dir / "probabilities"
    entropy_dir = output_dir / "entropy"
    masks_dir.mkdir(parents=True, exist_ok=True)
    probabilities_dir.mkdir(parents=True, exist_ok=True)
    entropy_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for row in table.itertuples(index=False):
            image_id = str(row.image_id)
            image_file = str(row.image_file)
            image_path = config.image_dir / image_file
            tensor, original_size = _prepare_image(image_path, input_size, torch)
            tensor = tensor.to(device)
            output = extract_logits(model(tensor))
            logits = functional.interpolate(
                output,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
            probabilities = functional.softmax(logits, dim=1)[0].cpu().numpy()
            prediction = probabilities.argmax(axis=0).astype(np.uint8)
            entropy = predictive_entropy(probabilities, axis=0).astype(np.float32)
            confidence = probabilities.max(axis=0)
            quality = assess_mask_quality(prediction, schema, entropy=entropy)

            mask_file = f"{image_id}_mask.png"
            probability_file = f"{image_id}_probabilities.npz"
            entropy_file = f"{image_id}_entropy.npy"
            Image.fromarray(prediction, mode="L").save(masks_dir / mask_file)
            np.savez_compressed(
                probabilities_dir / probability_file,
                probabilities=probabilities,
            )
            np.save(entropy_dir / entropy_file, entropy)
            manifest_rows.append(
                {
                    "image_id": image_id,
                    "image_file": image_file,
                    "mask_file": mask_file,
                    "probability_file": probability_file,
                    "entropy_file": entropy_file,
                    "model_id": model_id,
                    "split": split,
                    "mean_confidence": float(confidence.mean()),
                    "mean_entropy": float(entropy.mean()),
                    "high_uncertainty_fraction": float((entropy >= 0.5).mean()),
                    "quality_gate_accepted": quality.accepted,
                    "quality_gate_flags": "|".join(quality.flags),
                    **quality.measurements,
                }
            )

    manifest_path = output_dir / "prediction_manifest.csv"
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "model_id": model_id,
        "split": split,
        "images": len(manifest_rows),
        "quality_gate_rejected": int((~manifest["quality_gate_accepted"]).sum()),
        "checkpoint_path": str(checkpoint_path),
        "manifest_path": str(manifest_path),
        "masks_dir": str(masks_dir),
        "probabilities_dir": str(probabilities_dir),
        "entropy_dir": str(entropy_dir),
    }
    (output_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
