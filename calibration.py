from __future__ import annotations

import numpy as np


def normalize_probability_rows(
    probabilities: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=float), 0.0, None)
    row_sum = values.sum(axis=-1, keepdims=True)
    zero_rows = row_sum <= epsilon
    if np.any(zero_rows):
        values = np.where(
            zero_rows,
            1.0 / values.shape[-1],
            values,
        )
        row_sum = values.sum(axis=-1, keepdims=True)
    return values / row_sum


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    probabilities = normalize_probability_rows(probabilities)
    confidence = np.max(probabilities, axis=1)
    prediction = np.argmax(probabilities, axis=1)
    correct = prediction == np.asarray(y_true, dtype=int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if np.any(mask):
            error += float(np.mean(mask)) * abs(
                float(np.mean(correct[mask]))
                - float(np.mean(confidence[mask]))
            )
    return float(error)


def classwise_oof_calibration_audit(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = normalize_probability_rows(probabilities)
    rows: dict[str, dict[str, float]] = {}
    for class_index in range(probabilities.shape[1]):
        truth = (y_true == class_index).astype(float)
        probability = probabilities[:, class_index]
        rows[str(class_index)] = {
            "one_vs_rest_brier": float(
                np.mean((probability - truth) ** 2)
            ),
        }
    return {
        "source": "training-partition out-of-fold calibrated probabilities",
        "used_as_fusion_weight": False,
        "n_samples": int(len(y_true)),
        "classwise": rows,
        f"top_label_ece_{n_bins}": expected_calibration_error(
            y_true,
            probabilities,
            n_bins=n_bins,
        ),
    }

