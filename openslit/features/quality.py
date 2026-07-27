"""Image, segmentation, and polar-normalization quality features and gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from openslit.annotation.schema import AnnotationSchema

from .common import fraction, tissue_masks
from .config import QualityConfig, SourceRequirements
from .normalization import PolarIris


@dataclass(frozen=True)
class FeatureEligibility:
    accepted: bool
    flags: tuple[str, ...]
    measurements: dict[str, Any]


def _gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    return 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]


def _laplacian_variance(gray: np.ndarray) -> float:
    gray = np.asarray(gray, dtype=np.float64)
    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return float(laplacian.var())


def _illumination_uniformity(
    gray: np.ndarray,
    mask: np.ndarray,
    grid: int = 4,
) -> float | None:
    gray = np.asarray(gray, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    means: list[float] = []
    y_edges = np.linspace(0, gray.shape[0], grid + 1, dtype=int)
    x_edges = np.linspace(0, gray.shape[1], grid + 1, dtype=int)
    for y_start, y_stop in zip(y_edges[:-1], y_edges[1:]):
        for x_start, x_stop in zip(x_edges[:-1], x_edges[1:]):
            region_mask = mask[y_start:y_stop, x_start:x_stop]
            values = gray[y_start:y_stop, x_start:x_stop][region_mask]
            if values.size:
                means.append(float(values.mean()))
    if len(means) < 2:
        return None
    mean = float(np.mean(means))
    return None if mean == 0 else float(np.std(means) / mean)


def image_quality_features(
    image: np.ndarray,
    iris_mask: np.ndarray,
) -> dict[str, Any]:
    gray = _gray(image)
    iris_values = gray[np.asarray(iris_mask, dtype=bool)]
    values = iris_values if iris_values.size else gray.reshape(-1)
    return {
        "image_gray_mean": float(values.mean()),
        "image_gray_median": float(np.median(values)),
        "image_gray_std": float(values.std()),
        "image_dark_clip_fraction": float((values <= 5).mean()),
        "image_bright_clip_fraction": float((values >= 250).mean()),
        "image_laplacian_variance": _laplacian_variance(gray),
        "image_illumination_cv": _illumination_uniformity(gray, iris_mask),
    }


def assess_feature_eligibility(
    image: np.ndarray,
    mask: np.ndarray,
    schema: AnnotationSchema,
    polar: PolarIris | None,
    manifest_row: dict[str, str],
    source_requirements: SourceRequirements,
    quality: QualityConfig,
) -> FeatureEligibility:
    masks = tissue_masks(mask, schema)
    ocular_region = np.logical_or.reduce(
        [masks["pupil"], masks["iris"], masks["artifact"], masks["uncertain"]]
    )
    image_metrics = image_quality_features(image, masks["iris"])
    iris_pixels = int(masks["iris"].sum())
    uncertain_fraction = fraction(masks["uncertain"], ocular_region)
    artifact_fraction = fraction(masks["artifact"], ocular_region)
    polar_valid_fraction = 0.0 if polar is None else polar.valid_pixel_fraction
    valid_angle_fraction = 0.0 if polar is None else polar.valid_angle_fraction
    review_status = str(manifest_row.get("review_status", "")).strip()
    gradable = str(manifest_row.get("gradable", "")).strip().lower() in {
        "true",
        "1",
        "yes",
    }
    mask_hash_present = bool(str(manifest_row.get("mask_sha256", "")).strip())

    flags: list[str] = []
    if source_requirements.allowed_review_status and review_status not in set(
        source_requirements.allowed_review_status
    ):
        flags.append("UNAPPROVED_REVIEW_STATUS")
    if source_requirements.require_gradable and not gradable:
        flags.append("NOT_GRADABLE")
    if source_requirements.require_mask_sha256 and not mask_hash_present:
        flags.append("MISSING_MASK_SHA256")
    if iris_pixels < quality.minimum_visible_iris_pixels:
        flags.append("INSUFFICIENT_VISIBLE_IRIS")
    if polar_valid_fraction < quality.minimum_valid_polar_fraction:
        flags.append("INSUFFICIENT_POLAR_COVERAGE")
    if uncertain_fraction > quality.maximum_uncertain_fraction:
        flags.append("EXCESSIVE_UNCERTAIN_AREA")
    if artifact_fraction > quality.maximum_artifact_fraction:
        flags.append("EXCESSIVE_ARTIFACT_AREA")
    if image_metrics["image_laplacian_variance"] < quality.minimum_laplacian_variance:
        flags.append("LOW_SHARPNESS")

    measurements = {
        **image_metrics,
        "visible_iris_pixels": iris_pixels,
        "feature_uncertain_fraction": uncertain_fraction,
        "feature_artifact_fraction": artifact_fraction,
        "polar_valid_fraction": polar_valid_fraction,
        "polar_valid_angle_fraction": valid_angle_fraction,
        "source_review_status": review_status,
        "source_gradable": gradable,
        "source_mask_hash_present": mask_hash_present,
    }
    return FeatureEligibility(
        accepted=not flags,
        flags=tuple(flags),
        measurements=measurements,
    )
