from __future__ import annotations

import numpy as np
from scipy.stats import t
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from calibration import (
    expected_calibration_error,
    normalize_probability_rows,
)


METRIC_NAMES = (
    "overall_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_f1",
    "brier_score",
)


def multiclass_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_classes: int,
) -> float:
    one_hot = np.eye(n_classes)[np.asarray(y_true, dtype=int)]
    probabilities = normalize_probability_rows(probabilities)
    return float(
        np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
    )


def probability_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = normalize_probability_rows(probabilities)
    prediction = np.argmax(probabilities, axis=1)
    return {
        "overall_accuracy": float(
            accuracy_score(y_true, prediction)
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                prediction,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                prediction,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                prediction,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_precision": float(
            precision_score(
                y_true,
                prediction,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                prediction,
                average="weighted",
                zero_division=0,
            )
        ),
        "brier_score": multiclass_brier_score(
            y_true,
            probabilities,
            n_classes,
        ),
        "ece_10": expected_calibration_error(
            y_true,
            probabilities,
            n_bins=10,
        ),
    }


def selection_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_classes: int,
    complexity: float,
) -> tuple[float, float, float, float]:
    metrics = probability_metrics(y_true, probabilities, n_classes)
    return (
        metrics["overall_accuracy"],
        metrics["macro_f1"],
        -metrics["brier_score"],
        -float(complexity),
    )


def summarize_internal(
    fold_metrics: list[dict[str, dict[str, float]]],
    method_names: tuple[str, ...],
) -> dict[str, object]:
    fold_count = len(fold_metrics)
    if fold_count < 2:
        raise ValueError("internal summary requires at least two folds")
    critical = float(t.ppf(0.975, df=fold_count - 1))
    summary: dict[str, object] = {}
    for method in method_names:
        summary[method] = {}
        for metric in METRIC_NAMES:
            values = np.asarray(
                [fold[method][metric] for fold in fold_metrics],
                dtype=float,
            )
            mean = float(np.mean(values))
            standard_deviation = float(np.std(values, ddof=1))
            half_width = critical * standard_deviation / np.sqrt(fold_count)
            summary[method][metric] = {
                "mean": mean,
                "sample_sd": standard_deviation,
                "ci95": [
                    float(mean - half_width),
                    float(mean + half_width),
                ],
                "fold_values": values.tolist(),
                "fold_count": fold_count,
                "sd_definition": "outer-fold sample SD (ddof=1)",
            }
    return summary
