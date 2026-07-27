import numpy as np

from openslit.ai.metrics import (
    binary_metrics,
    ensemble_vote_disagreement,
    expected_calibration_error,
    predictive_entropy,
)


def test_binary_metrics_perfect_and_partial():
    reference = np.array([[1, 1], [0, 0]], dtype=bool)
    perfect = binary_metrics(reference, reference)
    assert perfect.dice == 1.0
    assert perfect.iou == 1.0

    prediction = np.array([[1, 0], [1, 0]], dtype=bool)
    partial = binary_metrics(reference, prediction)
    assert partial.dice == 0.5
    assert partial.iou == 1 / 3
    assert partial.false_positive_pixels == 1
    assert partial.false_negative_pixels == 1


def test_entropy_is_low_for_certain_and_high_for_uniform():
    certain = np.array([[[0.99]], [[0.01]]])
    uniform = np.array([[[0.5]], [[0.5]]])
    assert float(predictive_entropy(certain)[0, 0]) < 0.1
    assert np.isclose(float(predictive_entropy(uniform)[0, 0]), 1.0)


def test_ensemble_vote_disagreement():
    predictions = np.array(
        [
            [[1, 1], [2, 2]],
            [[1, 2], [2, 2]],
            [[1, 2], [3, 2]],
        ]
    )
    disagreement = ensemble_vote_disagreement(predictions)
    assert disagreement[0, 0] == 0.0
    assert np.isclose(disagreement[0, 1], 1 / 3)
    assert np.isclose(disagreement[1, 0], 1 / 3)


def test_expected_calibration_error_perfect():
    confidences = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([1, 1, 0, 0])
    error = expected_calibration_error(confidences, correct, bins=2)
    assert 0 <= error <= 1
