"""Anatomical plausibility and uncertainty gates for AI-generated masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from openslit.annotation.schema import AnnotationSchema


@dataclass(frozen=True)
class MaskQualityResult:
    accepted: bool
    flags: tuple[str, ...]
    measurements: dict[str, float | int | bool]


def _largest_component_fraction(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    total = int(mask.sum())
    if total == 0:
        return 0.0
    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Install AI dependencies with: python -m pip install -e '.[ai]'"
        ) from exc
    labels, count = ndimage.label(mask)
    if count == 0:
        return 0.0
    component_sizes = np.bincount(labels.reshape(-1))[1:]
    return float(component_sizes.max() / total)


def assess_mask_quality(
    mask: np.ndarray,
    schema: AnnotationSchema,
    entropy: np.ndarray | None = None,
    max_uncertain_fraction: float = 0.20,
    max_artifact_fraction: float = 0.45,
    max_mean_entropy: float = 0.35,
    min_pupil_iris_ratio: float = 0.005,
    max_pupil_iris_ratio: float = 0.80,
) -> MaskQualityResult:
    """Apply conservative reject/flag rules before clinical feature extraction."""

    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D indexed mask; got {mask.shape}")
    unknown = sorted(set(np.unique(mask).tolist()) - set(schema.class_ids))
    if unknown:
        return MaskQualityResult(
            accepted=False,
            flags=(f"UNKNOWN_CLASS_IDS:{unknown}",),
            measurements={"unknown_class_count": len(unknown)},
        )

    pupil_id = schema.class_by_name["pupil"].id
    iris_id = schema.class_by_name["iris"].id
    uncertain_id = schema.class_by_name["uncertain"].id
    artifact_names = ["reflection", "slit_beam", "eyelid", "eyelash"]
    pupil = mask == pupil_id
    iris = mask == iris_id
    uncertain = mask == uncertain_id
    artifact = np.isin(
        mask,
        [schema.class_by_name[name].id for name in artifact_names],
    )
    foreground = mask != schema.background_value
    pupil_pixels = int(pupil.sum())
    iris_pixels = int(iris.sum())
    foreground_pixels = int(foreground.sum())
    denominator = max(1, pupil_pixels + iris_pixels)
    pupil_iris_ratio = float(pupil_pixels / denominator)
    uncertain_fraction = float(uncertain.sum() / max(1, foreground_pixels))
    artifact_fraction = float(artifact.sum() / max(1, foreground_pixels))
    pupil_component_fraction = _largest_component_fraction(pupil)
    iris_component_fraction = _largest_component_fraction(iris)
    pupil_touches_edge = bool(
        pupil[0, :].any()
        or pupil[-1, :].any()
        or pupil[:, 0].any()
        or pupil[:, -1].any()
    )
    mean_entropy = None
    high_entropy_fraction = None
    if entropy is not None:
        entropy = np.asarray(entropy, dtype=np.float64)
        if entropy.shape != mask.shape:
            raise ValueError("Entropy map and mask must have identical dimensions")
        mean_entropy = float(entropy.mean())
        high_entropy_fraction = float((entropy >= 0.5).mean())

    flags: list[str] = []
    if pupil_pixels == 0:
        flags.append("MISSING_PUPIL")
    if iris_pixels == 0:
        flags.append("MISSING_IRIS")
    if pupil_pixels and pupil_component_fraction < 0.95:
        flags.append("FRAGMENTED_PUPIL")
    if iris_pixels and iris_component_fraction < 0.80:
        flags.append("FRAGMENTED_IRIS")
    if pupil_touches_edge:
        flags.append("PUPIL_TOUCHES_IMAGE_EDGE")
    if pupil_pixels and iris_pixels and not (
        min_pupil_iris_ratio <= pupil_iris_ratio <= max_pupil_iris_ratio
    ):
        flags.append("IMPLAUSIBLE_PUPIL_IRIS_RATIO")
    if uncertain_fraction > max_uncertain_fraction:
        flags.append("EXCESSIVE_UNCERTAIN_AREA")
    if artifact_fraction > max_artifact_fraction:
        flags.append("EXCESSIVE_ARTIFACT_AREA")
    if mean_entropy is not None and mean_entropy > max_mean_entropy:
        flags.append("HIGH_MODEL_UNCERTAINTY")

    measurements: dict[str, float | int | bool] = {
        "pupil_pixels": pupil_pixels,
        "iris_pixels": iris_pixels,
        "foreground_pixels": foreground_pixels,
        "pupil_iris_ratio": pupil_iris_ratio,
        "uncertain_fraction": uncertain_fraction,
        "artifact_fraction": artifact_fraction,
        "pupil_largest_component_fraction": pupil_component_fraction,
        "iris_largest_component_fraction": iris_component_fraction,
        "pupil_touches_edge": pupil_touches_edge,
    }
    if mean_entropy is not None:
        measurements["mean_entropy"] = mean_entropy
        measurements["high_entropy_fraction"] = float(high_entropy_fraction)
    return MaskQualityResult(
        accepted=not flags,
        flags=tuple(flags),
        measurements=measurements,
    )


def quality_result_to_row(image_id: str, result: MaskQualityResult) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "accepted": result.accepted,
        "flags": "|".join(result.flags),
        **result.measurements,
    }
