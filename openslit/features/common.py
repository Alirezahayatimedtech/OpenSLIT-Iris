"""Shared utilities for iris feature extraction."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from openslit.annotation.schema import AnnotationSchema


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rgb_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_indexed_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        mask = np.asarray(image)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2-D indexed mask at {path}; got {mask.shape}")
    return mask.astype(np.uint8, copy=False)


def validate_image_mask_pair(
    image: np.ndarray,
    mask: np.ndarray,
    schema: AnnotationSchema,
) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image; got {image.shape}")
    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Image and mask dimensions differ: {image.shape[:2]} vs {mask.shape}"
        )
    unknown = sorted(set(np.unique(mask).tolist()) - set(schema.class_ids))
    if unknown:
        raise ValueError(f"Mask contains unknown class IDs: {unknown}")


def tissue_masks(mask: np.ndarray, schema: AnnotationSchema) -> dict[str, np.ndarray]:
    output = {item.name: mask == item.id for item in schema.classes}
    artifact_names = ["reflection", "slit_beam", "eyelid", "eyelash"]
    output["artifact"] = np.logical_or.reduce(
        [output[name] for name in artifact_names]
    )
    output["valid_iris"] = output["iris"]
    output["foreground"] = mask != schema.class_by_name["background"].id
    return output


def fraction(numerator: np.ndarray, denominator: np.ndarray | int) -> float:
    numerator_count = int(np.asarray(numerator, dtype=bool).sum())
    if isinstance(denominator, np.ndarray):
        denominator_count = int(np.asarray(denominator, dtype=bool).sum())
    else:
        denominator_count = int(denominator)
    return 0.0 if denominator_count <= 0 else float(numerator_count / denominator_count)


def finite_or_none(value: float | np.floating[Any]) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return finite_or_none(value)
    if isinstance(value, Path):
        return str(value)
    return value
