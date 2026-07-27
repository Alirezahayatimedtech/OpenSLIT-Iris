"""Rubber-sheet style normalization of visible iris tissue."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openslit.annotation.schema import AnnotationSchema


@dataclass(frozen=True)
class PolarIris:
    image: np.ndarray
    labels: np.ndarray
    valid: np.ndarray
    angle_valid: np.ndarray
    inner_radius: np.ndarray
    outer_radius: np.ndarray
    center_x: float
    center_y: float

    @property
    def valid_angle_fraction(self) -> float:
        return float(self.angle_valid.mean()) if len(self.angle_valid) else 0.0

    @property
    def valid_pixel_fraction(self) -> float:
        return float(self.valid.mean()) if self.valid.size else 0.0


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    coordinates = np.argwhere(np.asarray(mask, dtype=bool))
    if len(coordinates) == 0:
        return None
    y, x = coordinates.mean(axis=0)
    return float(x), float(y)


def _sample_nearest(array: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_index = np.rint(x).astype(int)
    y_index = np.rint(y).astype(int)
    x_index = np.clip(x_index, 0, array.shape[1] - 1)
    y_index = np.clip(y_index, 0, array.shape[0] - 1)
    return array[y_index, x_index]


def _sample_bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    dx = (x - x0)[..., None]
    dy = (y - y0)[..., None]
    top = image[y0, x0] * (1.0 - dx) + image[y0, x1] * dx
    bottom = image[y1, x0] * (1.0 - dx) + image[y1, x1] * dx
    return top * (1.0 - dy) + bottom * dy


def _maximum_radius(center_x: float, center_y: float, width: int, height: int) -> int:
    distances = [
        np.hypot(center_x, center_y),
        np.hypot(width - 1 - center_x, center_y),
        np.hypot(center_x, height - 1 - center_y),
        np.hypot(width - 1 - center_x, height - 1 - center_y),
    ]
    return max(2, int(np.ceil(max(distances))))


def normalize_iris(
    image: np.ndarray,
    mask: np.ndarray,
    schema: AnnotationSchema,
    angular_samples: int = 360,
    radial_samples: int = 64,
) -> PolarIris:
    """Map visible iris tissue to a fixed polar strip.

    For each angle, the inner radius is the last pupil pixel and the outer radius
    is the farthest visible iris pixel on that ray. Missing angles remain invalid.
    Occlusion and artefact labels are preserved in the polar label map.
    """

    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask dimensions must match")
    pupil_id = schema.class_by_name["pupil"].id
    iris_id = schema.class_by_name["iris"].id
    pupil = mask == pupil_id
    iris = mask == iris_id
    center = _centroid(pupil) or _centroid(iris)
    if center is None:
        raise ValueError("Cannot normalize an image without pupil or iris pixels")
    center_x, center_y = center
    height, width = mask.shape
    max_radius = _maximum_radius(center_x, center_y, width, height)
    ray_radii = np.arange(max_radius + 1, dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, angular_samples, endpoint=False)
    cosines = np.cos(angles)
    sines = np.sin(angles)

    inner = np.full(angular_samples, np.nan, dtype=np.float64)
    outer = np.full(angular_samples, np.nan, dtype=np.float64)
    angle_valid = np.zeros(angular_samples, dtype=bool)
    for index, (cosine, sine) in enumerate(zip(cosines, sines)):
        x = center_x + ray_radii * cosine
        y = center_y + ray_radii * sine
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not inside.any():
            continue
        labels = _sample_nearest(mask, x[inside], y[inside])
        radii = ray_radii[inside]
        pupil_radii = radii[labels == pupil_id]
        iris_radii = radii[labels == iris_id]
        if len(pupil_radii) == 0 or len(iris_radii) == 0:
            continue
        inner_radius = float(pupil_radii.max())
        outer_candidates = iris_radii[iris_radii > inner_radius]
        if len(outer_candidates) == 0:
            continue
        outer_radius = float(outer_candidates.max())
        if outer_radius - inner_radius < 2.0:
            continue
        inner[index] = inner_radius
        outer[index] = outer_radius
        angle_valid[index] = True

    radial_fraction = np.linspace(0.0, 1.0, radial_samples, endpoint=True)[:, None]
    inner_filled = np.where(angle_valid, inner, 0.0)[None, :]
    outer_filled = np.where(angle_valid, outer, 0.0)[None, :]
    radius_grid = inner_filled + radial_fraction * (outer_filled - inner_filled)
    x_grid = center_x + radius_grid * cosines[None, :]
    y_grid = center_y + radius_grid * sines[None, :]
    polar_image = _sample_bilinear(image.astype(np.float64), x_grid, y_grid)
    polar_labels = _sample_nearest(mask, x_grid, y_grid)
    polar_image[:, ~angle_valid, :] = 0.0
    polar_labels[:, ~angle_valid] = schema.class_by_name["background"].id
    valid = (polar_labels == iris_id) & angle_valid[None, :]
    return PolarIris(
        image=np.clip(polar_image, 0, 255).astype(np.uint8),
        labels=polar_labels.astype(np.uint8),
        valid=valid,
        angle_valid=angle_valid,
        inner_radius=inner,
        outer_radius=outer,
        center_x=center_x,
        center_y=center_y,
    )


def radial_profile(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.shape[:2] != valid.shape:
        raise ValueError("Values and valid mask dimensions differ")
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    for row in range(values.shape[0]):
        selected = values[row][valid[row]]
        if selected.size:
            result[row] = float(selected.mean())
    return result


def angular_coverage(valid: np.ndarray) -> np.ndarray:
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("Polar valid mask must be two-dimensional")
    return valid.mean(axis=0)
