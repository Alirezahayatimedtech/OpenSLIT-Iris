"""Raw and illumination-normalized iris color features."""

from __future__ import annotations

from typing import Any

import numpy as np

from .normalization import PolarIris


def gray_world_normalize(image: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    reference_mask = np.asarray(reference_mask, dtype=bool)
    if image.shape[:2] != reference_mask.shape:
        raise ValueError("Image and reference mask dimensions differ")
    pixels = image[reference_mask]
    if len(pixels) == 0:
        return image.astype(np.uint8)
    channel_means = pixels.mean(axis=0)
    target = float(channel_means.mean())
    scale = np.divide(
        target,
        channel_means,
        out=np.ones_like(channel_means),
        where=channel_means > 1e-8,
    )
    normalized = image * scale[None, None, :]
    return np.clip(normalized, 0, 255).astype(np.uint8)


def rgb_to_hsv_array(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    delta = maximum - minimum
    saturation = np.divide(
        delta,
        maximum,
        out=np.zeros_like(delta),
        where=maximum > 0,
    )
    hue = np.zeros_like(maximum)
    nonzero = delta > 0
    red = nonzero & (maximum == rgb[..., 0])
    green = nonzero & (maximum == rgb[..., 1])
    blue = nonzero & (maximum == rgb[..., 2])
    hue[red] = ((rgb[..., 1][red] - rgb[..., 2][red]) / delta[red]) % 6.0
    hue[green] = (rgb[..., 2][green] - rgb[..., 0][green]) / delta[green] + 2.0
    hue[blue] = (rgb[..., 0][blue] - rgb[..., 1][blue]) / delta[blue] + 4.0
    hue = (hue / 6.0) % 1.0
    return np.stack([hue, saturation, maximum], axis=-1)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def rgb_to_lab_array(rgb: np.ndarray) -> np.ndarray:
    linear = _srgb_to_linear(rgb)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = linear @ matrix.T
    reference = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
    xyz = xyz / reference
    delta = 6.0 / 29.0
    threshold = delta**3
    f = np.where(
        xyz > threshold,
        np.cbrt(xyz),
        xyz / (3 * delta**2) + 4.0 / 29.0,
    )
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def _channel_statistics(
    values: np.ndarray,
    names: tuple[str, ...],
    prefix: str,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    result: dict[str, Any] = {}
    if values.size == 0:
        for name in names:
            for statistic in ["mean", "median", "std", "p10", "p90"]:
                result[f"{prefix}_{name}_{statistic}"] = None
        return result
    for index, name in enumerate(names):
        channel = values[:, index]
        result[f"{prefix}_{name}_mean"] = float(channel.mean())
        result[f"{prefix}_{name}_median"] = float(np.median(channel))
        result[f"{prefix}_{name}_std"] = float(channel.std())
        result[f"{prefix}_{name}_p10"] = float(np.percentile(channel, 10))
        result[f"{prefix}_{name}_p90"] = float(np.percentile(channel, 90))
    return result


def _entropy(
    values: np.ndarray,
    bins: int,
    value_range: tuple[float, float],
) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    histogram, _ = np.histogram(values, bins=bins, range=value_range)
    probability = histogram.astype(np.float64)
    total = probability.sum()
    if total == 0:
        return None
    probability /= total
    probability = probability[probability > 0]
    return float(-(probability * np.log2(probability)).sum())


def _delta_e(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
        )
    )


def _polar_region_means(
    lab: np.ndarray,
    valid: np.ndarray,
    radial_zones: int,
    angular_sectors: int,
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    radial_means: list[np.ndarray | None] = []
    sector_means: list[np.ndarray | None] = []
    radial_edges = np.linspace(0, lab.shape[0], radial_zones + 1, dtype=int)
    for start, stop in zip(radial_edges[:-1], radial_edges[1:]):
        region = valid[start:stop]
        values = lab[start:stop][region]
        radial_means.append(values.mean(axis=0) if len(values) else None)
    sector_edges = np.linspace(0, lab.shape[1], angular_sectors + 1, dtype=int)
    for start, stop in zip(sector_edges[:-1], sector_edges[1:]):
        region = valid[:, start:stop]
        values = lab[:, start:stop][region]
        sector_means.append(values.mean(axis=0) if len(values) else None)
    return radial_means, sector_means


def extract_color_features(
    image: np.ndarray,
    iris_mask: np.ndarray,
    polar: PolarIris | None,
    normalization: str = "gray_world",
    radial_zones: int = 3,
    angular_sectors: int = 8,
) -> dict[str, Any]:
    image = np.asarray(image, dtype=np.uint8)
    iris_mask = np.asarray(iris_mask, dtype=bool)
    if image.shape[:2] != iris_mask.shape:
        raise ValueError("Image and iris mask dimensions differ")
    normalized = (
        gray_world_normalize(image, iris_mask)
        if normalization == "gray_world"
        else image.copy()
    )
    result: dict[str, Any] = {
        "color_normalization": normalization,
        "iris_color_pixel_count": int(iris_mask.sum()),
    }
    for prefix, current in [("raw", image), ("normalized", normalized)]:
        rgb_values = current[iris_mask].astype(np.float64)
        hsv_values = rgb_to_hsv_array(current)[iris_mask]
        lab_values = rgb_to_lab_array(current)[iris_mask]
        result.update(
            _channel_statistics(rgb_values, ("r", "g", "b"), f"iris_{prefix}_rgb")
        )
        result.update(
            _channel_statistics(hsv_values, ("h", "s", "v"), f"iris_{prefix}_hsv")
        )
        result.update(
            _channel_statistics(lab_values, ("l", "a", "b"), f"iris_{prefix}_lab")
        )
        if len(lab_values):
            chroma = np.sqrt(lab_values[:, 1] ** 2 + lab_values[:, 2] ** 2)
            result[f"iris_{prefix}_lab_chroma_mean"] = float(chroma.mean())
            result[f"iris_{prefix}_lab_color_spread"] = float(
                np.linalg.norm(lab_values.std(axis=0))
            )
            result[f"iris_{prefix}_lab_l_entropy"] = _entropy(
                lab_values[:, 0], bins=32, value_range=(0.0, 100.0)
            )
            result[f"iris_{prefix}_hue_entropy"] = _entropy(
                hsv_values[:, 0], bins=36, value_range=(0.0, 1.0)
            )
        else:
            result[f"iris_{prefix}_lab_chroma_mean"] = None
            result[f"iris_{prefix}_lab_color_spread"] = None
            result[f"iris_{prefix}_lab_l_entropy"] = None
            result[f"iris_{prefix}_hue_entropy"] = None

    if polar is None or not polar.valid.any():
        result.update(
            {
                "polar_radial_inner_outer_delta_e": None,
                "polar_opposite_sector_delta_e_mean": None,
                "polar_sector_delta_e_max": None,
                "polar_valid_color_fraction": 0.0,
            }
        )
        return result

    polar_image = polar.image
    if normalization == "gray_world":
        polar_image = gray_world_normalize(polar_image, polar.valid)
    polar_lab = rgb_to_lab_array(polar_image)
    radial_means, sector_means = _polar_region_means(
        polar_lab,
        polar.valid,
        radial_zones=radial_zones,
        angular_sectors=angular_sectors,
    )
    result["polar_valid_color_fraction"] = polar.valid_pixel_fraction
    if radial_means and radial_means[0] is not None and radial_means[-1] is not None:
        result["polar_radial_inner_outer_delta_e"] = _delta_e(
            radial_means[0], radial_means[-1]
        )
    else:
        result["polar_radial_inner_outer_delta_e"] = None
    opposite: list[float] = []
    half = angular_sectors // 2
    for index in range(half):
        first = sector_means[index]
        second = sector_means[index + half]
        if first is not None and second is not None:
            opposite.append(_delta_e(first, second))
    result["polar_opposite_sector_delta_e_mean"] = (
        float(np.mean(opposite)) if opposite else None
    )
    result["polar_sector_delta_e_max"] = float(max(opposite)) if opposite else None
    for index, mean in enumerate(radial_means, start=1):
        for channel_index, channel in enumerate(("l", "a", "b")):
            result[f"polar_zone_{index}_lab_{channel}_mean"] = (
                None if mean is None else float(mean[channel_index])
            )
    return result
