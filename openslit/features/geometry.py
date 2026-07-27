"""Interpretable pupil, iris, and occlusion geometry features."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from openslit.annotation.schema import AnnotationSchema

from .common import fraction, tissue_masks


def _centroid(binary: np.ndarray) -> tuple[float, float] | None:
    points = np.argwhere(np.asarray(binary, dtype=bool))
    if len(points) == 0:
        return None
    y, x = points.mean(axis=0)
    return float(x), float(y)


def _boundary(binary: np.ndarray) -> np.ndarray:
    binary = np.asarray(binary, dtype=bool)
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    eroded = (
        center
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return center & ~eroded


def _perimeter(binary: np.ndarray) -> float:
    binary = np.asarray(binary, dtype=bool)
    if not binary.any():
        return 0.0
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    edges = (
        (center & ~padded[:-2, 1:-1]).sum()
        + (center & ~padded[2:, 1:-1]).sum()
        + (center & ~padded[1:-1, :-2]).sum()
        + (center & ~padded[1:-1, 2:]).sum()
    )
    return float(edges)


def _orientation_degrees(x: np.ndarray, y: np.ndarray) -> float:
    centered = np.stack([x - x.mean(), y - y.mean()])
    covariance = centered @ centered.T / max(1, centered.shape[1])
    values, vectors = np.linalg.eigh(covariance)
    vector = vectors[:, int(np.argmax(values))]
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


def _shape_features(binary: np.ndarray, prefix: str) -> dict[str, Any]:
    binary = np.asarray(binary, dtype=bool)
    area = int(binary.sum())
    centroid = _centroid(binary)
    perimeter = _perimeter(binary)
    result: dict[str, Any] = {
        f"{prefix}_area_px": area,
        f"{prefix}_perimeter_px": perimeter,
        f"{prefix}_equivalent_diameter_px": (
            float(math.sqrt(4.0 * area / math.pi)) if area else None
        ),
        f"{prefix}_circularity": (
            float(4.0 * math.pi * area / (perimeter**2))
            if area and perimeter
            else None
        ),
        f"{prefix}_centroid_x_px": None if centroid is None else centroid[0],
        f"{prefix}_centroid_y_px": None if centroid is None else centroid[1],
    }
    points = np.argwhere(binary)
    if len(points) >= 2:
        y = points[:, 0].astype(np.float64)
        x = points[:, 1].astype(np.float64)
        covariance = np.cov(np.stack([x, y]), bias=True)
        eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
        major = float(max(eigenvalues[0], 0.0))
        minor = float(max(eigenvalues[1], 0.0))
        result[f"{prefix}_eccentricity"] = (
            float(math.sqrt(max(0.0, 1.0 - minor / major))) if major > 0 else 0.0
        )
        result[f"{prefix}_orientation_deg"] = _orientation_degrees(x, y)
        result[f"{prefix}_bbox_width_px"] = int(x.max() - x.min() + 1)
        result[f"{prefix}_bbox_height_px"] = int(y.max() - y.min() + 1)
    else:
        result.update(
            {
                f"{prefix}_eccentricity": None,
                f"{prefix}_orientation_deg": None,
                f"{prefix}_bbox_width_px": 0,
                f"{prefix}_bbox_height_px": 0,
            }
        )
    result[f"{prefix}_boundary_pixels"] = int(_boundary(binary).sum())
    return result


def _center_distance(
    first: tuple[float, float] | None,
    second: tuple[float, float] | None,
) -> float | None:
    if first is None or second is None:
        return None
    return float(math.dist(first, second))


def _boundary_visibility_fraction(
    boundary: np.ndarray,
    obscured: np.ndarray,
) -> float | None:
    count = int(boundary.sum())
    if count == 0:
        return None
    visible = boundary & ~np.asarray(obscured, dtype=bool)
    return float(visible.sum() / count)


def extract_geometry_features(
    mask: np.ndarray,
    schema: AnnotationSchema,
) -> dict[str, Any]:
    masks = tissue_masks(mask, schema)
    pupil = masks["pupil"]
    iris = masks["iris"]
    pupil_features = _shape_features(pupil, "pupil")
    iris_features = _shape_features(iris, "iris_visible")
    pupil_center = _centroid(pupil)
    iris_center = _centroid(iris)
    image_pixels = int(mask.size)
    iris_pixels = int(iris.sum())
    pupil_pixels = int(pupil.sum())
    ocular_region = np.logical_or.reduce(
        [pupil, iris, masks["artifact"], masks["uncertain"]]
    )
    result: dict[str, Any] = {
        **pupil_features,
        **iris_features,
        "image_height_px": int(mask.shape[0]),
        "image_width_px": int(mask.shape[1]),
        "pupil_iris_area_ratio": (
            float(pupil_pixels / iris_pixels) if iris_pixels else None
        ),
        "pupil_iris_center_distance_px": _center_distance(
            pupil_center,
            iris_center,
        ),
        "visible_iris_fraction_of_frame": float(iris_pixels / image_pixels),
        "ocular_region_fraction_of_frame": fraction(ocular_region, image_pixels),
        "reflection_fraction_of_ocular_region": fraction(
            masks["reflection"], ocular_region
        ),
        "slit_beam_fraction_of_ocular_region": fraction(
            masks["slit_beam"], ocular_region
        ),
        "eyelid_fraction_of_ocular_region": fraction(masks["eyelid"], ocular_region),
        "eyelash_fraction_of_ocular_region": fraction(
            masks["eyelash"], ocular_region
        ),
        "artifact_fraction_of_ocular_region": fraction(
            masks["artifact"], ocular_region
        ),
        "uncertain_fraction_of_ocular_region": fraction(
            masks["uncertain"], ocular_region
        ),
    }
    pupil_boundary = _boundary(pupil)
    iris_boundary = _boundary(iris)
    result["pupil_boundary_visible_fraction"] = _boundary_visibility_fraction(
        pupil_boundary,
        masks["artifact"] | masks["uncertain"],
    )
    result["iris_boundary_visible_fraction"] = _boundary_visibility_fraction(
        iris_boundary,
        masks["artifact"] | masks["uncertain"],
    )
    return result
