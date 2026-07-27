"""Texture features from a normalized polar iris strip."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .normalization import PolarIris


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    return 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]


def _entropy(values: np.ndarray, bins: int = 32) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    histogram, _ = np.histogram(values, bins=bins)
    probability = histogram.astype(np.float64)
    if probability.sum() == 0:
        return None
    probability /= probability.sum()
    probability = probability[probability > 0]
    return float(-(probability * np.log2(probability)).sum())


def _lbp_codes(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.float64)
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    center = padded[1:-1, 1:-1]
    neighbours = [
        padded[:-2, :-2],
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, 2:],
        padded[2:, 2:],
        padded[2:, 1:-1],
        padded[2:, :-2],
        padded[1:-1, :-2],
    ]
    codes = np.zeros(center.shape, dtype=np.uint8)
    for bit, neighbour in enumerate(neighbours):
        codes |= ((neighbour >= center).astype(np.uint8) << bit)
    return codes


def _uniform_lbp_fraction(codes: np.ndarray, valid: np.ndarray) -> float | None:
    selected = codes[np.asarray(valid, dtype=bool)]
    if selected.size == 0:
        return None
    bits = ((selected[:, None] >> np.arange(8)) & 1).astype(np.uint8)
    wrapped = np.concatenate([bits, bits[:, :1]], axis=1)
    transitions = np.abs(np.diff(wrapped, axis=1)).sum(axis=1)
    return float((transitions <= 2).mean())


def _quantize(gray: np.ndarray, levels: int) -> np.ndarray:
    clipped = np.clip(np.asarray(gray, dtype=np.float64), 0, 255)
    return np.minimum((clipped * levels / 256.0).astype(np.int16), levels - 1)


def _offset(angle_degrees: int, distance: int) -> tuple[int, int]:
    radians = math.radians(angle_degrees)
    dx = int(round(math.cos(radians) * distance))
    dy = int(round(math.sin(radians) * distance))
    return dy, dx


def _paired_regions(
    array: np.ndarray,
    valid: np.ndarray,
    dy: int,
    dx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = array.shape
    y0_start = max(0, -dy)
    y0_stop = min(height, height - dy)
    x0_start = max(0, -dx)
    x0_stop = min(width, width - dx)
    source = array[y0_start:y0_stop, x0_start:x0_stop]
    target = array[y0_start + dy : y0_stop + dy, x0_start + dx : x0_stop + dx]
    valid_pairs = (
        valid[y0_start:y0_stop, x0_start:x0_stop]
        & valid[y0_start + dy : y0_stop + dy, x0_start + dx : x0_stop + dx]
    )
    return source, target, valid_pairs


def _glcm(
    quantized: np.ndarray,
    valid: np.ndarray,
    levels: int,
    dy: int,
    dx: int,
) -> np.ndarray | None:
    source, target, valid_pairs = _paired_regions(quantized, valid, dy, dx)
    first = source[valid_pairs]
    second = target[valid_pairs]
    if first.size == 0:
        return None
    matrix = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(matrix, (first, second), 1)
    np.add.at(matrix, (second, first), 1)
    matrix_sum = matrix.sum()
    return matrix / matrix_sum if matrix_sum else None


def _glcm_features(matrix: np.ndarray) -> dict[str, float]:
    levels = matrix.shape[0]
    i, j = np.indices((levels, levels))
    difference = i - j
    contrast = float((matrix * difference**2).sum())
    dissimilarity = float((matrix * np.abs(difference)).sum())
    homogeneity = float((matrix / (1.0 + difference**2)).sum())
    asm = float((matrix**2).sum())
    energy = float(math.sqrt(asm))
    nonzero = matrix[matrix > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    p_i = matrix.sum(axis=1)
    p_j = matrix.sum(axis=0)
    mean_i = float((np.arange(levels) * p_i).sum())
    mean_j = float((np.arange(levels) * p_j).sum())
    std_i = float(np.sqrt(((np.arange(levels) - mean_i) ** 2 * p_i).sum()))
    std_j = float(np.sqrt(((np.arange(levels) - mean_j) ** 2 * p_j).sum()))
    correlation = 1.0
    if std_i > 0 and std_j > 0:
        correlation = float(
            (matrix * (i - mean_i) * (j - mean_j)).sum() / (std_i * std_j)
        )
    return {
        "contrast": contrast,
        "dissimilarity": dissimilarity,
        "homogeneity": homogeneity,
        "asm": asm,
        "energy": energy,
        "entropy": entropy,
        "correlation": correlation,
    }


def _haar_energies(gray: np.ndarray, valid: np.ndarray, levels: int) -> dict[str, Any]:
    values = np.asarray(gray, dtype=np.float64).copy()
    valid = np.asarray(valid, dtype=bool)
    selected = values[valid]
    fill = float(selected.mean()) if selected.size else 0.0
    values[~valid] = fill
    result: dict[str, Any] = {}
    for level in range(1, levels + 1):
        height = values.shape[0] - values.shape[0] % 2
        width = values.shape[1] - values.shape[1] % 2
        if height < 2 or width < 2:
            break
        values = values[:height, :width]
        a = values[0::2, 0::2]
        b = values[0::2, 1::2]
        c = values[1::2, 0::2]
        d = values[1::2, 1::2]
        approximation = (a + b + c + d) / 4.0
        horizontal = (a + b - c - d) / 4.0
        vertical = (a - b + c - d) / 4.0
        diagonal = (a - b - c + d) / 4.0
        result[f"haar_level_{level}_horizontal_energy"] = float(
            np.mean(horizontal**2)
        )
        result[f"haar_level_{level}_vertical_energy"] = float(
            np.mean(vertical**2)
        )
        result[f"haar_level_{level}_diagonal_energy"] = float(
            np.mean(diagonal**2)
        )
        values = approximation
    return result


def extract_texture_features(
    polar: PolarIris | None,
    gray_levels: int = 16,
    glcm_distances: tuple[int, ...] = (1, 2),
    glcm_angles_degrees: tuple[int, ...] = (0, 45, 90, 135),
    haar_levels: int = 2,
) -> dict[str, Any]:
    if polar is None or not polar.valid.any():
        return {
            "texture_valid_pixels": 0,
            "texture_gray_mean": None,
            "texture_gray_std": None,
            "texture_gray_entropy": None,
            "lbp_entropy": None,
            "lbp_uniform_fraction": None,
            "lbp_dominant_pattern_fraction": None,
        }
    gray = rgb_to_gray(polar.image)
    valid = polar.valid
    selected = gray[valid]
    result: dict[str, Any] = {
        "texture_valid_pixels": int(valid.sum()),
        "texture_gray_mean": float(selected.mean()),
        "texture_gray_std": float(selected.std()),
        "texture_gray_entropy": _entropy(selected, bins=32),
    }
    codes = _lbp_codes(gray)
    selected_codes = codes[valid]
    histogram = np.bincount(selected_codes, minlength=256).astype(np.float64)
    probability = histogram / histogram.sum() if histogram.sum() else histogram
    nonzero = probability[probability > 0]
    result["lbp_entropy"] = (
        float(-(nonzero * np.log2(nonzero)).sum()) if nonzero.size else None
    )
    result["lbp_uniform_fraction"] = _uniform_lbp_fraction(codes, valid)
    result["lbp_dominant_pattern_fraction"] = (
        float(probability.max()) if probability.size else None
    )

    quantized = _quantize(gray, gray_levels)
    glcm_rows: list[dict[str, float]] = []
    for distance in glcm_distances:
        for angle in glcm_angles_degrees:
            dy, dx = _offset(angle, distance)
            if dy == 0 and dx == 0:
                continue
            matrix = _glcm(quantized, valid, gray_levels, dy, dx)
            if matrix is not None:
                glcm_rows.append(_glcm_features(matrix))
    for feature in [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "asm",
        "energy",
        "entropy",
        "correlation",
    ]:
        values = [row[feature] for row in glcm_rows]
        result[f"glcm_{feature}_mean"] = float(np.mean(values)) if values else None
        result[f"glcm_{feature}_std"] = float(np.std(values)) if values else None
    result.update(_haar_energies(gray, valid, haar_levels))

    midpoint = gray.shape[0] // 2
    inner = gray[:midpoint][valid[:midpoint]]
    outer = gray[midpoint:][valid[midpoint:]]
    inner_entropy = _entropy(inner, bins=24)
    outer_entropy = _entropy(outer, bins=24)
    result["texture_inner_entropy"] = inner_entropy
    result["texture_outer_entropy"] = outer_entropy
    result["texture_inner_outer_entropy_difference"] = (
        None
        if inner_entropy is None or outer_entropy is None
        else float(outer_entropy - inner_entropy)
    )
    return result
