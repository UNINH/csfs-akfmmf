from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from calibration import (
    classwise_oof_calibration_audit,
    normalize_probability_rows,
)
from config import (
    AGGREGATION,
    BETA_CANDIDATES,
    BIN_CANDIDATES,
    CALIBRATION_SPLITS,
    CLASSIFIER_ORDER_CANDIDATES,
    COVARIANCE_EIGEN_FLOOR,
    FEATURE_COUNT_CANDIDATES,
    FORGETTING_FACTOR_CANDIDATES,
    INITIAL_COVARIANCE_CANDIDATES,
    INNER_SPLITS,
    LAPLACE_CANDIDATES,
    MAX_COVARIANCE_CHANGE_CANDIDATES,
    MEASUREMENT_COVARIANCE_CANDIDATES,
    MODEL_NAMES,
    MODEL_PARAMETER_GRID,
    PROCESS_COVARIANCE_CANDIDATES,
    PROBABILITY_EPSILON,
    REFERENCE_BINS,
    REFERENCE_FEATURE_COUNTS,
    REFERENCE_FUSION,
    REFERENCE_LAPLACE,
)
from csfs import CSFSSelector
from data import TrainingMedianPreprocessor
from evaluation import probability_metrics, selection_score
from fusion import (
    adjust_probabilities,
    batch_diagonal_kalman,
    sequential_kalman_fusion,
)
from models import ClassSpecificProbabilisticModel


ProgressCallback = Callable[[str], None] | None


@dataclass
class PreparedInnerFold:
    fold: int
    train_positions: np.ndarray
    validation_positions: np.ndarray
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    y_train: np.ndarray
    train_record_ids: np.ndarray
    validation_record_ids: np.ndarray
    preprocessing_audit: dict[str, object]
    selector_cache: dict[tuple[int, float], CSFSSelector] = field(
        default_factory=dict
    )

    def selector(self, bins: int, laplace: float) -> CSFSSelector:
        key = (int(bins), float(laplace))
        if key not in self.selector_cache:
            self.selector_cache[key] = CSFSSelector(
                max_features=max(FEATURE_COUNT_CANDIDATES),
                n_bins=int(bins),
                laplace=float(laplace),
            ).fit(self.X_train, self.y_train)
        return self.selector_cache[key]


def _prepare_inner_folds(
    raw_gases: pd.DataFrame,
    y: np.ndarray,
    record_ids: np.ndarray,
    *,
    seed: int,
) -> list[PreparedInnerFold]:
    splitter = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    prepared: list[PreparedInnerFold] = []
    for fold, (train_positions, validation_positions) in enumerate(
        splitter.split(raw_gases, y)
    ):
        if len(np.intersect1d(train_positions, validation_positions)):
            raise RuntimeError("inner train/validation indices overlap")
        preprocessor = TrainingMedianPreprocessor()
        X_train = preprocessor.fit_transform(
            raw_gases.iloc[train_positions],
            record_ids[train_positions],
        )
        X_validation = preprocessor.transform(
            raw_gases.iloc[validation_positions]
        )
        prepared.append(
            PreparedInnerFold(
                fold=fold,
                train_positions=train_positions,
                validation_positions=validation_positions,
                X_train=X_train,
                X_validation=X_validation,
                y_train=y[train_positions],
                train_record_ids=record_ids[train_positions],
                validation_record_ids=record_ids[validation_positions],
                preprocessing_audit=preprocessor.audit_metadata(),
            )
        )
    return prepared


def _candidate_oof(
    prepared_folds: list[PreparedInnerFold],
    y: np.ndarray,
    *,
    family: str,
    parameters: dict[str, float | int | str],
    feature_count: int,
    bins: int,
    laplace: float,
    seed: int,
    record_audit: bool = False,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    n_classes = len(np.unique(y))
    oof = np.zeros((len(y), n_classes), dtype=float)
    audits: list[dict[str, object]] = []
    for prepared in prepared_folds:
        selector = prepared.selector(bins, laplace)
        feature_subsets = selector.subsets(feature_count)
        model = ClassSpecificProbabilisticModel(
            family=family,
            feature_subsets=feature_subsets,
            parameters=parameters,
            calibration_splits=CALIBRATION_SPLITS,
            aggregation=AGGREGATION,
            seed=seed + 10007 * (prepared.fold + 1),
        ).fit(
            prepared.X_train,
            prepared.y_train,
            prepared.train_record_ids,
        )
        oof[prepared.validation_positions] = model.predict_proba(
            prepared.X_validation
        )
        if record_audit:
            audits.append(
                {
                    "fold": prepared.fold,
                    "train_positions": prepared.train_positions.tolist(),
                    "validation_positions": (
                        prepared.validation_positions.tolist()
                    ),
                    "train_record_ids": [
                        str(value)
                        for value in prepared.train_record_ids
                    ],
                    "validation_record_ids": [
                        str(value)
                        for value in prepared.validation_record_ids
                    ],
                    "overlap_count": int(
                        len(
                            set(prepared.train_record_ids)
                            & set(prepared.validation_record_ids)
                        )
                    ),
                    "preprocessing": prepared.preprocessing_audit,
                    "csfs": selector.audit_metadata(),
                    "feature_subsets": feature_subsets,
                    "calibration": (
                        model.calibration_audit_metadata()
                    ),
                }
            )
    return normalize_probability_rows(oof), audits


def _result_row(
    y: np.ndarray,
    probability: np.ndarray,
    n_classes: int,
    *,
    candidate: dict[str, object],
    complexity: float,
) -> dict[str, object]:
    metrics = probability_metrics(y, probability, n_classes)
    return {
        "candidate": candidate,
        "macro_f1": metrics["macro_f1"],
        "overall_accuracy": metrics["overall_accuracy"],
        "brier_score": metrics["brier_score"],
        "complexity_tiebreak": float(complexity),
        "score": list(
            selection_score(
                y,
                probability,
                n_classes,
                complexity,
            )
        ),
    }


def _best_row(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise RuntimeError("selection stage evaluated no candidates")
    return max(rows, key=lambda row: tuple(row["score"]))


def tune_base_models(
    raw_gases: pd.DataFrame,
    y: np.ndarray,
    record_ids: np.ndarray,
    *,
    seed: int,
    progress: ProgressCallback = None,
) -> tuple[dict[str, dict[str, object]], np.ndarray, dict[str, object]]:

    y = np.asarray(y, dtype=int)
    record_ids = np.asarray(record_ids, dtype=object)
    prepared = _prepare_inner_folds(
        raw_gases,
        y,
        record_ids,
        seed=seed + 17,
    )
    selected: dict[str, dict[str, object]] = {}
    selected_oof: list[np.ndarray] = []
    family_audits: dict[str, object] = {}
    n_classes = len(np.unique(y))
    for family_index, family in enumerate(MODEL_NAMES):
        family_seed = seed + family_index * 1009
        if progress:
            progress(f"    base tuning {family}: model parameters")
        model_rows: list[dict[str, object]] = []
        for candidate_index, parameters in enumerate(
            MODEL_PARAMETER_GRID[family]
        ):
            probability, _ = _candidate_oof(
                prepared,
                y,
                family=family,
                parameters=parameters,
                feature_count=REFERENCE_FEATURE_COUNTS[family],
                bins=REFERENCE_BINS,
                laplace=REFERENCE_LAPLACE,
                seed=family_seed,
            )
            model_rows.append(
                _result_row(
                    y,
                    probability,
                    n_classes,
                    candidate={"parameters": parameters},
                    complexity=float(candidate_index),
                )
            )
        selected_model = _best_row(model_rows)
        parameters = dict(
            selected_model["candidate"]["parameters"]
        )

        if progress:
            progress(f"    base tuning {family}: bins + CSFS smoothing")
        bin_rows: list[dict[str, object]] = []
        for bins, laplace in itertools.product(
            BIN_CANDIDATES,
            LAPLACE_CANDIDATES,
        ):
            probability, _ = _candidate_oof(
                prepared,
                y,
                family=family,
                parameters=parameters,
                feature_count=REFERENCE_FEATURE_COUNTS[family],
                bins=bins,
                laplace=laplace,
                seed=family_seed,
            )
            complexity = abs(math.log(bins / REFERENCE_BINS))
            complexity += abs(
                math.log(laplace / REFERENCE_LAPLACE)
            )
            bin_rows.append(
                _result_row(
                    y,
                    probability,
                    n_classes,
                    candidate={
                        "bins": int(bins),
                        "laplace": float(laplace),
                    },
                    complexity=complexity,
                )
            )
        selected_bins = _best_row(bin_rows)
        bins = int(selected_bins["candidate"]["bins"])
        laplace = float(
            selected_bins["candidate"]["laplace"]
        )

        if progress:
            progress(f"    base tuning {family}: feature subset size")
        feature_rows: list[dict[str, object]] = []
        for feature_count in FEATURE_COUNT_CANDIDATES:
            probability, _ = _candidate_oof(
                prepared,
                y,
                family=family,
                parameters=parameters,
                feature_count=feature_count,
                bins=bins,
                laplace=laplace,
                seed=family_seed,
            )
            feature_rows.append(
                _result_row(
                    y,
                    probability,
                    n_classes,
                    candidate={
                        "feature_count": int(feature_count)
                    },
                    complexity=abs(
                        feature_count
                        - REFERENCE_FEATURE_COUNTS[family]
                    ),
                )
            )
        selected_features = _best_row(feature_rows)
        feature_count = int(
            selected_features["candidate"]["feature_count"]
        )
        probability, selected_inner_audit = _candidate_oof(
            prepared,
            y,
            family=family,
            parameters=parameters,
            feature_count=feature_count,
            bins=bins,
            laplace=laplace,
            seed=family_seed,
            record_audit=True,
        )
        selected[family] = {
            "parameters": parameters,
            "bins": bins,
            "laplace": laplace,
            "feature_count": feature_count,
        }
        selected_oof.append(probability)
        family_audits[family] = {
            "selected": selected[family],
            "model_parameter_stage": {
                "candidate_count": len(model_rows),
                "candidate_results": model_rows,
                "selected": selected_model,
            },
            "bins_laplace_stage": {
                "candidate_count": len(bin_rows),
                "candidate_results": bin_rows,
                "selected": selected_bins,
            },
            "feature_count_stage": {
                "candidate_count": len(feature_rows),
                "candidate_results": feature_rows,
                "selected": selected_features,
            },
            "selected_inner_fit_audit": selected_inner_audit,
            "oof_calibration_quality": (
                classwise_oof_calibration_audit(y, probability)
            ),
        }
    oof_stack = np.stack(selected_oof, axis=1)
    return selected, oof_stack, {
        "selection_data": "training-partition OOF predictions only",
        "selector_signature_excludes_inference_data_and_labels": True,
        "random_seed_search": False,
        "shared_inner_splits_across_candidates": True,
        "families": family_audits,
    }


def fit_selected_base_models(
    train_gases: pd.DataFrame,
    y_train: np.ndarray,
    train_record_ids: np.ndarray,
    inference_gases: pd.DataFrame,
    inference_record_ids: np.ndarray,
    selected: dict[str, dict[str, object]],
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    preprocessor = TrainingMedianPreprocessor()
    X_train = preprocessor.fit_transform(
        train_gases,
        train_record_ids,
    )
    X_inference = preprocessor.transform(inference_gases)
    probabilities: list[np.ndarray] = []
    audit: dict[str, object] = {
        "fit_predict_signature_excludes_inference_labels": True,
        "training_record_ids": [
            str(value) for value in train_record_ids
        ],
        "inference_record_ids": [
            str(value) for value in inference_record_ids
        ],
        "record_id_overlap_count": int(
            len(set(train_record_ids) & set(inference_record_ids))
        ),
        "preprocessing": preprocessor.audit_metadata(),
        "families": {},
    }
    for family_index, family in enumerate(MODEL_NAMES):
        configuration = selected[family]
        selector = CSFSSelector(
            max_features=int(configuration["feature_count"]),
            n_bins=int(configuration["bins"]),
            laplace=float(configuration["laplace"]),
        ).fit(X_train, y_train)
        feature_subsets = selector.subsets(
            int(configuration["feature_count"])
        )
        model = ClassSpecificProbabilisticModel(
            family=family,
            feature_subsets=feature_subsets,
            parameters=dict(configuration["parameters"]),
            calibration_splits=CALIBRATION_SPLITS,
            aggregation=AGGREGATION,
            seed=seed + family_index * 1009,
        ).fit(X_train, y_train, train_record_ids)
        probabilities.append(model.predict_proba(X_inference))
        audit["families"][family] = {
            "selected_configuration": configuration,
            "csfs": selector.audit_metadata(),
            "feature_subsets": feature_subsets,
            "calibration": model.calibration_audit_metadata(),
        }
    return np.stack(probabilities, axis=1), audit


def _candidate_metric_arrays(
    y: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    y = np.asarray(y, dtype=int)
    prediction = np.argmax(probabilities, axis=2)
    accuracy = np.mean(prediction == y[None, :], axis=1)
    class_f1 = []
    for class_index in range(probabilities.shape[2]):
        truth = y == class_index
        predicted = prediction == class_index
        true_positive = np.sum(predicted & truth[None, :], axis=1)
        false_positive = np.sum(
            predicted & ~truth[None, :],
            axis=1,
        )
        false_negative = np.sum(
            ~predicted & truth[None, :],
            axis=1,
        )
        denominator = (
            2 * true_positive + false_positive + false_negative
        )
        class_f1.append(
            np.divide(
                2 * true_positive,
                denominator,
                out=np.zeros_like(true_positive, dtype=float),
                where=denominator > 0,
            )
        )
    macro_f1 = np.mean(np.stack(class_f1, axis=1), axis=1)
    one_hot = np.eye(probabilities.shape[2])[y]
    brier = np.mean(
        np.sum(
            (probabilities - one_hot[None, :, :]) ** 2,
            axis=2,
        ),
        axis=1,
    )
    return macro_f1, accuracy, brier


def _order_names(order: tuple[int, ...]) -> list[str]:
    return [MODEL_NAMES[index] for index in order]


def _fusion_distance(candidate: dict[str, object], adaptive: bool) -> float:
    distance = sum(
        abs(
            math.log(
                float(candidate[key])
                / float(REFERENCE_FUSION[key])
            )
        )
        for key in (
            "initial_covariance",
            "process_covariance",
            "measurement_covariance",
        )
    )
    if adaptive:
        distance += abs(
            float(candidate["beta"])
            - float(REFERENCE_FUSION["beta"])
        )
        distance += 4.0 * abs(
            float(candidate["forgetting_factor"])
            - float(REFERENCE_FUSION["forgetting_factor"])
        )
        distance += 2.0 * abs(
            float(candidate["max_covariance_change"])
            - float(
                REFERENCE_FUSION["max_covariance_change"]
            )
        )
    if tuple(candidate["classifier_order"]) != tuple(
        REFERENCE_FUSION["classifier_order"]
    ):
        distance += 0.25
    return float(distance)


def _select_candidate_rows(
    candidates: list[dict[str, object]],
    probabilities: np.ndarray,
    y: np.ndarray,
    *,
    adaptive: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    macro_f1, accuracy, brier = _candidate_metric_arrays(
        y,
        probabilities,
    )
    rows: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        distance = _fusion_distance(candidate, adaptive)
        rows.append(
            {
                "candidate": candidate,
                "macro_f1": float(macro_f1[index]),
                "overall_accuracy": float(accuracy[index]),
                "brier_score": float(brier[index]),
                "change_distance": distance,
                "score": [
                    float(accuracy[index]),
                    float(macro_f1[index]),
                    float(-brier[index]),
                    float(-distance),
                ],
            }
        )
    selected = max(rows, key=lambda row: tuple(row["score"]))
    leaderboard = sorted(
        rows,
        key=lambda row: tuple(row["score"]),
        reverse=True,
    )[:20]
    return selected, leaderboard


def select_fusion_parameters(
    y: np.ndarray,
    base_oof_probabilities: np.ndarray,
    *,
    progress: ProgressCallback = None,
) -> tuple[dict[str, object], dict[str, object]]:
    if progress:
        candidate_count = (
            len(BETA_CANDIDATES)
            * len(INITIAL_COVARIANCE_CANDIDATES)
            * len(PROCESS_COVARIANCE_CANDIDATES)
            * len(MEASUREMENT_COVARIANCE_CANDIDATES)
            * len(FORGETTING_FACTOR_CANDIDATES)
            * len(MAX_COVARIANCE_CHANGE_CANDIDATES)
            * len(CLASSIFIER_ORDER_CANDIDATES)
        )
        progress(
            f"    fusion tuning AKFMMF: {candidate_count} full-grid candidates"
        )
    all_akf_rows: list[dict[str, object]] = []
    for beta in BETA_CANDIDATES:
        adjustment = adjust_probabilities(
            base_oof_probabilities,
            beta=float(beta),
            epsilon=PROBABILITY_EPSILON,
        )
        candidates = [
            {
                "beta": float(beta),
                "initial_covariance": float(initial),
                "process_covariance": float(process),
                "measurement_covariance": float(measurement),
                "forgetting_factor": float(forgetting),
                "max_covariance_change": float(maximum_change),
                "classifier_order": tuple(order),
            }
            for (
                initial,
                process,
                measurement,
                forgetting,
                maximum_change,
                order,
            ) in itertools.product(
                INITIAL_COVARIANCE_CANDIDATES,
                PROCESS_COVARIANCE_CANDIDATES,
                MEASUREMENT_COVARIANCE_CANDIDATES,
                FORGETTING_FACTOR_CANDIDATES,
                MAX_COVARIANCE_CHANGE_CANDIDATES,
                CLASSIFIER_ORDER_CANDIDATES,
            )
        ]
        beta_rows: list[dict[str, object]] = []
        for start in range(0, len(candidates), 256):
            chunk = candidates[start:start + 256]
            chunk_probability = batch_diagonal_kalman(
                adjustment.probabilities,
                orders=np.asarray(
                    [row["classifier_order"] for row in chunk],
                    dtype=int,
                ),
                initial_covariance=np.asarray(
                    [row["initial_covariance"] for row in chunk]
                ),
                process_covariance=np.asarray(
                    [row["process_covariance"] for row in chunk]
                ),
                measurement_covariance=np.asarray(
                    [row["measurement_covariance"] for row in chunk]
                ),
                forgetting_factor=np.asarray(
                    [row["forgetting_factor"] for row in chunk]
                ),
                max_covariance_change=np.asarray(
                    [row["max_covariance_change"] for row in chunk]
                ),
                adaptive=True,
                eigen_floor=COVARIANCE_EIGEN_FLOOR,
                agreement_confidence=adjustment.agreement_confidence,
            )
            _, rows = _select_candidate_rows(
                chunk,
                chunk_probability,
                y,
                adaptive=True,
            )
            beta_rows.extend(rows)
        all_akf_rows.extend(
            sorted(
                beta_rows,
                key=lambda row: tuple(row["score"]),
                reverse=True,
            )[:20]
        )
    selected_akf_row = max(
        all_akf_rows,
        key=lambda row: tuple(row["score"]),
    )
    akf_leaderboard = sorted(
        all_akf_rows,
        key=lambda row: tuple(row["score"]),
        reverse=True,
    )[:20]
    selected_akf = dict(selected_akf_row["candidate"])
    audit = {
        "selection_data": "training-partition OOF probabilities only",
        "target_or_outer_labels_available_to_selector": False,
        "selection_objective": (
            "accuracy, macro-F1, negative multiclass Brier, "
            "negative predeclared change distance"
        ),
        "AKFMMF": {
            "candidate_count": (
                len(BETA_CANDIDATES)
                * len(INITIAL_COVARIANCE_CANDIDATES)
                * len(PROCESS_COVARIANCE_CANDIDATES)
                * len(MEASUREMENT_COVARIANCE_CANDIDATES)
                * len(FORGETTING_FACTOR_CANDIDATES)
                * len(MAX_COVARIANCE_CHANGE_CANDIDATES)
                * len(CLASSIFIER_ORDER_CANDIDATES)
            ),
            "selected": selected_akf_row,
            "top20": akf_leaderboard,
        },
    }
    return selected_akf, audit


def fuse_selected_probabilities(
    base_probabilities: np.ndarray,
    selected_akf: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    adjustment = adjust_probabilities(
        base_probabilities,
        beta=float(selected_akf["beta"]),
        epsilon=PROBABILITY_EPSILON,
    )
    akf_probability, akf_audit = sequential_kalman_fusion(
        adjustment.probabilities,
        tuple(selected_akf["classifier_order"]),
        initial_covariance=float(
            selected_akf["initial_covariance"]
        ),
        process_covariance=float(
            selected_akf["process_covariance"]
        ),
        measurement_covariance=float(
            selected_akf["measurement_covariance"]
        ),
        forgetting_factor=float(
            selected_akf["forgetting_factor"]
        ),
        max_covariance_change=float(
            selected_akf["max_covariance_change"]
        ),
        adaptive=True,
        eigen_floor=COVARIANCE_EIGEN_FLOOR,
        agreement_confidence=adjustment.agreement_confidence,
    )
    akf_fast = batch_diagonal_kalman(
        adjustment.probabilities,
        orders=np.asarray(
            [selected_akf["classifier_order"]],
            dtype=int,
        ),
        initial_covariance=np.asarray(
            [selected_akf["initial_covariance"]]
        ),
        process_covariance=np.asarray(
            [selected_akf["process_covariance"]]
        ),
        measurement_covariance=np.asarray(
            [selected_akf["measurement_covariance"]]
        ),
        forgetting_factor=np.asarray(
            [selected_akf["forgetting_factor"]]
        ),
        max_covariance_change=np.asarray(
            [selected_akf["max_covariance_change"]]
        ),
        adaptive=True,
        eigen_floor=COVARIANCE_EIGEN_FLOOR,
        agreement_confidence=adjustment.agreement_confidence,
    )[0]
    if not np.allclose(
        akf_fast,
        akf_probability,
        atol=1e-10,
        rtol=1e-10,
    ):
        raise RuntimeError("AKFMMF search/production paths disagree")
    audit = {
        "AKFMMF": {
            **akf_audit.to_dict(),
            "beta": float(selected_akf["beta"]),
            "Wg_formula": "exp(-beta*dbar_i) / sum_k exp(-beta*dbar_k)",
            "Wg_sum_min": float(
                adjustment.global_agreement.sum(axis=1).min()
            ),
            "Wg_sum_max": float(
                adjustment.global_agreement.sum(axis=1).max()
            ),
            "mean_jousselme_distance_min": float(
                adjustment.mean_jousselme_distance.min()
            ),
            "mean_jousselme_distance_max": float(
                adjustment.mean_jousselme_distance.max()
            ),
        },
        "search_and_production_paths_match": True,
    }
    return akf_probability, audit


def fit_select_predict(
    train_gases: pd.DataFrame,
    y_train: np.ndarray,
    train_record_ids: np.ndarray,
    inference_gases: pd.DataFrame,
    inference_record_ids: np.ndarray,
    *,
    seed: int,
    progress: ProgressCallback = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    selected_base, base_oof, base_selection_audit = tune_base_models(
        train_gases,
        y_train,
        train_record_ids,
        seed=seed,
        progress=progress,
    )
    selected_akf, fusion_selection_audit = (
        select_fusion_parameters(
            y_train,
            base_oof,
            progress=progress,
        )
    )
    inference_base, final_fit_audit = fit_selected_base_models(
        train_gases,
        y_train,
        train_record_ids,
        inference_gases,
        inference_record_ids,
        selected_base,
        seed=seed,
    )
    probability, fusion_audit = fuse_selected_probabilities(
        inference_base,
        selected_akf,
    )
    return {"AKFMMF": probability}, {
        "fit_select_predict_signature_excludes_inference_labels": True,
        "selection_boundary": "training records and labels only",
        "selected_parameters": {
            "base_models": selected_base,
            "AKFMMF": {
                **selected_akf,
                "classifier_order": _order_names(
                    tuple(selected_akf["classifier_order"])
                ),
            },
        },
        "base_selection": base_selection_audit,
        "fusion_selection": fusion_selection_audit,
        "final_fit": final_fit_audit,
        "fusion": fusion_audit,
    }
