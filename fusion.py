from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from calibration import normalize_probability_rows

def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=axis, keepdims=True)

@dataclass
class ProbabilityAdjustmentResult:
    probabilities: np.ndarray
    agreement_confidence: np.ndarray
    global_agreement: np.ndarray
    local_agreement: np.ndarray
    mean_jousselme_distance: np.ndarray


def adjust_probabilities(
    probabilities: np.ndarray,
    beta: float,
    epsilon: float = 1e-9,
) -> ProbabilityAdjustmentResult:

    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    probabilities = normalize_probability_rows(probabilities, epsilon)
    n_samples, n_models, _ = probabilities.shape
    mean_distance = np.zeros((n_samples, n_models), dtype=float)
    local_agreement = np.zeros_like(probabilities)
    for left in range(n_models):
        distances = []
        compatibilities = []
        for right in range(n_models):
            if left == right:
                continue
            difference = (
                probabilities[:, left, :]
                - probabilities[:, right, :]
            )
            distances.append(
                np.sqrt(0.5 * np.sum(difference**2, axis=1))
            )
            numerator = (
                2.0
                * probabilities[:, left, :]
                * probabilities[:, right, :]
            )
            denominator = (
                probabilities[:, left, :] ** 2
                + probabilities[:, right, :] ** 2
                + epsilon
            )
            compatibilities.append(numerator / denominator)
        mean_distance[:, left] = np.mean(distances, axis=0)
        local_agreement[:, left, :] = np.mean(
            compatibilities,
            axis=0,
        )
    global_agreement = _softmax(-beta * mean_distance, axis=1)
    if not np.allclose(
        global_agreement.sum(axis=1),
        1.0,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise RuntimeError("Wg failed strict per-sample normalization")
    combined = (
        global_agreement[:, :, None]
        * np.clip(local_agreement, epsilon, 1.0)
    )
    class_multiplier = combined / np.mean(
        combined,
        axis=2,
        keepdims=True,
    )
    adjusted = normalize_probability_rows(
        probabilities * np.clip(class_multiplier, epsilon, None),
        epsilon,
    )
    agreement_confidence = combined / np.mean(
        combined,
        axis=1,
        keepdims=True,
    )
    return ProbabilityAdjustmentResult(
        probabilities=adjusted,
        agreement_confidence=np.clip(
            agreement_confidence,
            0.10,
            10.0,
        ),
        global_agreement=global_agreement,
        local_agreement=local_agreement,
        mean_jousselme_distance=mean_distance,
    )


@dataclass
class KalmanFusionAudit:
    alpha_min: float
    alpha_max: float
    covariance_min_eigenvalue: float
    covariance_max_asymmetry: float
    classifier_order: tuple[int, ...]
    state_definition: str = "latent diagnostic posterior probability vector"
    sequence_definition: str = "classifier-fusion index (not physical time)"
    covariance_adaptation: str = "bounded covariance scaling, not estimation"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _project_probability(
    vector: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    vector = np.clip(np.asarray(vector, dtype=float), epsilon, None)
    return vector / vector.sum()


def _repair_covariance(
    covariance: np.ndarray,
    eigen_floor: float,
) -> tuple[np.ndarray, float, float]:
    asymmetry = float(np.max(np.abs(covariance - covariance.T)))
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    repaired_values = np.maximum(eigenvalues, eigen_floor)
    repaired = (eigenvectors * repaired_values) @ eigenvectors.T
    return repaired, float(np.min(repaired_values)), asymmetry


def sequential_kalman_fusion(
    probabilities: np.ndarray,
    order: tuple[int, ...],
    *,
    initial_covariance: float,
    process_covariance: float,
    measurement_covariance: float,
    forgetting_factor: float,
    max_covariance_change: float,
    adaptive: bool,
    eigen_floor: float,
    agreement_confidence: np.ndarray | None = None,
    epsilon: float = 1e-9,
) -> tuple[np.ndarray, KalmanFusionAudit]:
    probabilities = normalize_probability_rows(probabilities, epsilon)
    n_samples, n_models, n_classes = probabilities.shape
    if sorted(order) != list(range(n_models)):
        raise ValueError("order must be a permutation of model indices")
    if agreement_confidence is not None:
        agreement_confidence = np.asarray(
            agreement_confidence,
            dtype=float,
        )
        if agreement_confidence.shape not in {
            (n_samples, n_models),
            (n_samples, n_models, n_classes),
        }:
            raise ValueError("agreement_confidence has an invalid shape")
    identity = np.eye(n_classes)
    fused = np.empty((n_samples, n_classes), dtype=float)
    alphas: list[float] = []
    minimum_eigenvalue = np.inf
    maximum_asymmetry = 0.0
    for sample in range(n_samples):
        state = np.full(n_classes, 1.0 / n_classes)
        covariance = identity * float(initial_covariance)
        smoothed_innovation: float | None = None
        for model_index in order:
            observation = probabilities[sample, model_index]
            innovation_norm = float(np.linalg.norm(observation - state))
            if smoothed_innovation is None:
                smoothed_innovation = innovation_norm
                raw_ratio = 1.0
            else:
                smoothed_innovation = (
                    forgetting_factor * smoothed_innovation
                    + (1.0 - forgetting_factor) * innovation_norm
                )
                raw_ratio = innovation_norm / (
                    smoothed_innovation + epsilon
                )
            bounded_ratio = float(
                np.clip(
                    raw_ratio,
                    1.0 - max_covariance_change,
                    1.0 + max_covariance_change,
                )
            )
            alpha = bounded_ratio if adaptive else 1.0
            alphas.append(alpha)
            process = identity * (process_covariance / alpha)
            predicted_covariance = covariance + process
            if agreement_confidence is None:
                agreement_scale = np.ones(n_classes)
            elif agreement_confidence.ndim == 2:
                agreement_scale = np.full(
                    n_classes,
                    np.clip(
                        agreement_confidence[sample, model_index],
                        0.10,
                        10.0,
                    ),
                )
            else:
                agreement_scale = np.clip(
                    agreement_confidence[sample, model_index],
                    0.10,
                    10.0,
                )
            measurement = np.diag(
                measurement_covariance * alpha / agreement_scale
            )
            innovation_covariance = (
                predicted_covariance + measurement
            )
            gain = (
                predicted_covariance
                @ np.linalg.inv(innovation_covariance)
            )
            state = state + gain @ (observation - state)
            state = _project_probability(state, epsilon)
            residual_operator = identity - gain
            covariance = (
                residual_operator
                @ predicted_covariance
                @ residual_operator.T
                + gain @ measurement @ gain.T
            )
            covariance, min_eigenvalue, asymmetry = _repair_covariance(
                covariance,
                eigen_floor,
            )
            minimum_eigenvalue = min(
                minimum_eigenvalue,
                min_eigenvalue,
            )
            maximum_asymmetry = max(maximum_asymmetry, asymmetry)
        fused[sample] = state
    audit = KalmanFusionAudit(
        alpha_min=float(np.min(alphas)),
        alpha_max=float(np.max(alphas)),
        covariance_min_eigenvalue=float(minimum_eigenvalue),
        covariance_max_asymmetry=float(maximum_asymmetry),
        classifier_order=tuple(order),
    )
    return normalize_probability_rows(fused, epsilon), audit


def batch_diagonal_kalman(
    probabilities: np.ndarray,
    *,
    orders: np.ndarray,
    initial_covariance: np.ndarray,
    process_covariance: np.ndarray,
    measurement_covariance: np.ndarray,
    forgetting_factor: np.ndarray,
    max_covariance_change: np.ndarray,
    adaptive: bool,
    eigen_floor: float,
    agreement_confidence: np.ndarray | None = None,
    epsilon: float = 1e-9,
) -> np.ndarray:

    probabilities = normalize_probability_rows(probabilities, epsilon)
    orders = np.asarray(orders, dtype=int)
    candidate_count = len(orders)
    n_samples, n_models, n_classes = probabilities.shape
    if orders.shape != (candidate_count, n_models):
        raise ValueError("orders shape is incompatible")
    state = np.full(
        (candidate_count, n_samples, n_classes),
        1.0 / n_classes,
        dtype=float,
    )
    covariance = np.broadcast_to(
        np.asarray(initial_covariance, dtype=float)[:, None, None],
        state.shape,
    ).copy()
    smoothed = np.zeros((candidate_count, n_samples), dtype=float)
    for step in range(n_models):
        model_indices = orders[:, step]
        observation = np.transpose(
            probabilities[:, model_indices, :],
            (1, 0, 2),
        )
        innovation_norm = np.linalg.norm(observation - state, axis=2)
        if step == 0:
            smoothed = innovation_norm
            raw_ratio = np.ones_like(smoothed)
        else:
            forgetting = np.asarray(
                forgetting_factor,
                dtype=float,
            )[:, None]
            smoothed = (
                forgetting * smoothed
                + (1.0 - forgetting) * innovation_norm
            )
            raw_ratio = innovation_norm / (smoothed + epsilon)
        maximum = np.asarray(
            max_covariance_change,
            dtype=float,
        )[:, None]
        bounded = np.clip(
            raw_ratio,
            1.0 - maximum,
            1.0 + maximum,
        )
        alpha = bounded if adaptive else np.ones_like(bounded)
        predicted = covariance + (
            np.asarray(process_covariance, dtype=float)[:, None, None]
            / alpha[:, :, None]
        )
        if agreement_confidence is None:
            agreement_scale = 1.0
        else:
            agreement_scale = np.transpose(
                agreement_confidence[:, model_indices, :],
                (1, 0, 2),
            )
            agreement_scale = np.clip(
                agreement_scale,
                0.10,
                10.0,
            )
        measurement = (
            np.asarray(measurement_covariance, dtype=float)[:, None, None]
            * alpha[:, :, None]
            / agreement_scale
        )
        gain = predicted / (predicted + measurement)
        state = state + gain * (observation - state)
        state = np.clip(state, epsilon, None)
        state /= state.sum(axis=2, keepdims=True)
        covariance = (
            (1.0 - gain) ** 2 * predicted
            + gain**2 * measurement
        )
        covariance = np.maximum(covariance, eigen_floor)
    return normalize_probability_rows(state, epsilon)
