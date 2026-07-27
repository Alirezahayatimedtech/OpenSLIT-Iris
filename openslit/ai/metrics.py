"""Metrics used for human-human and human-AI segmentation comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    dice: float
    iou: float
    precision: float
    recall: float
    reference_pixels: int
    prediction_pixels: int
    false_positive_pixels: int
    false_negative_pixels: int


def _safe_ratio(numerator: float, denominator: float, empty_value: float = 1.0) -> float:
    return empty_value if denominator == 0 else float(numerator / denominator)


def binary_metrics(reference: np.ndarray, prediction: np.ndarray) -> BinaryMetrics:
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if reference.shape != prediction.shape:
        raise ValueError(f"Mask shapes differ: {reference.shape} vs {prediction.shape}")
    intersection = int(np.logical_and(reference, prediction).sum())
    ref_pixels = int(reference.sum())
    pred_pixels = int(prediction.sum())
    union = int(np.logical_or(reference, prediction).sum())
    false_positive = int(np.logical_and(~reference, prediction).sum())
    false_negative = int(np.logical_and(reference, ~prediction).sum())
    return BinaryMetrics(
        dice=_safe_ratio(2 * intersection, ref_pixels + pred_pixels),
        iou=_safe_ratio(intersection, union),
        precision=_safe_ratio(intersection, pred_pixels),
        recall=_safe_ratio(intersection, ref_pixels),
        reference_pixels=ref_pixels,
        prediction_pixels=pred_pixels,
        false_positive_pixels=false_positive,
        false_negative_pixels=false_negative,
    )


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    points = np.argwhere(np.asarray(mask, dtype=bool))
    if len(points) == 0:
        return None
    y, x = points.mean(axis=0)
    return float(x), float(y)


def centroid_distance(reference: np.ndarray, prediction: np.ndarray) -> float | None:
    ref = mask_centroid(reference)
    pred = mask_centroid(prediction)
    if ref is None or pred is None:
        return None
    return float(np.hypot(ref[0] - pred[0], ref[1] - pred[1]))


def area_relative_error(reference: np.ndarray, prediction: np.ndarray) -> float | None:
    ref_pixels = int(np.asarray(reference, dtype=bool).sum())
    pred_pixels = int(np.asarray(prediction, dtype=bool).sum())
    if ref_pixels == 0:
        return None
    return float((pred_pixels - ref_pixels) / ref_pixels)


def multiclass_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    class_ids: list[int] | tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    reference = np.asarray(reference)
    prediction = np.asarray(prediction)
    if reference.shape != prediction.shape:
        raise ValueError(f"Mask shapes differ: {reference.shape} vs {prediction.shape}")
    output: dict[int, dict[str, Any]] = {}
    for class_id in class_ids:
        metrics = binary_metrics(reference == class_id, prediction == class_id)
        output[int(class_id)] = {
            **metrics.__dict__,
            "centroid_distance_pixels": centroid_distance(
                reference == class_id,
                prediction == class_id,
            ),
            "area_relative_error": area_relative_error(
                reference == class_id,
                prediction == class_id,
            ),
        }
    return output


def predictive_entropy(probabilities: np.ndarray, axis: int = 0) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim < 2:
        raise ValueError("Probability array must contain a class dimension")
    clipped = np.clip(probabilities, 1e-8, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=axis)
    class_count = probabilities.shape[axis]
    return entropy / np.log(class_count) if class_count > 1 else entropy


def probability_margin(probabilities: np.ndarray, axis: int = 0) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    sorted_probabilities = np.sort(probabilities, axis=axis)
    best = np.take(sorted_probabilities, -1, axis=axis)
    second = np.take(sorted_probabilities, -2, axis=axis)
    return best - second


def ensemble_vote_disagreement(predictions: np.ndarray, axis: int = 0) -> np.ndarray:
    predictions = np.asarray(predictions)
    if predictions.shape[axis] < 2:
        return np.zeros(np.delete(predictions.shape, axis), dtype=np.float64)
    moved = np.moveaxis(predictions, axis, 0)
    result = np.zeros(moved.shape[1:], dtype=np.float64)
    flat = moved.reshape(moved.shape[0], -1)
    out = result.reshape(-1)
    for index in range(flat.shape[1]):
        _, counts = np.unique(flat[:, index], return_counts=True)
        out[index] = 1.0 - float(counts.max() / moved.shape[0])
    return result


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    bins: int = 10,
) -> float:
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    if confidences.shape != correct.shape:
        raise ValueError("confidences and correct must have identical shape")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidences)
    if total == 0:
        return float("nan")
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        include_upper = upper == 1.0
        mask = (confidences >= lower) & (
            (confidences <= upper) if include_upper else (confidences < upper)
        )
        if not mask.any():
            continue
        error += float(mask.mean()) * abs(
            float(correct[mask].mean()) - float(confidences[mask].mean())
        )
    return error
