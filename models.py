from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from calibration import normalize_probability_rows
from config import AGGREGATION, CALIBRATION_METHOD, CALIBRATION_SPLITS


def make_base_classifier(
    family: str,
    seed: int,
    parameters: dict[str, float | int | str],
):
    if family == "SVM":
        return SVC(
            kernel="rbf",
            C=float(parameters["C"]),
            gamma=parameters["gamma"],
            probability=False,
            decision_function_shape="ovo",
            random_state=seed,
        )
    if family == "Bagging":
        max_depth = int(parameters["max_depth"])
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=None if max_depth == 0 else max_depth,
                random_state=seed,
            ),
            n_estimators=int(parameters["n_estimators"]),
            bootstrap=True,
            max_samples=float(parameters["max_samples"]),
            random_state=seed,
            n_jobs=-1,
        )
    if family == "AdaBoost":
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=int(parameters["max_depth"]),
                random_state=seed,
            ),
            n_estimators=int(parameters["n_estimators"]),
            learning_rate=float(parameters["learning_rate"]),
            random_state=seed,
        )
    raise ValueError(f"unknown model family: {family}")


def safe_calibration_splits(y: np.ndarray, requested: int) -> int:
    smallest_class = int(
        np.min(np.bincount(np.asarray(y, dtype=int)))
    )
    splits = min(int(requested), smallest_class)
    if splits < 2:
        raise ValueError(
            "at least two samples per class are required for calibration"
        )
    return splits


@dataclass
class ClassSpecificProbabilisticModel:
    family: str
    feature_subsets: dict[int, list[str]]
    parameters: dict[str, float | int | str]
    calibration_splits: int = CALIBRATION_SPLITS
    calibration_method: str = CALIBRATION_METHOD
    aggregation: str = AGGREGATION
    seed: int = 42

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        record_ids: np.ndarray | None = None,
    ) -> "ClassSpecificProbabilisticModel":
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        expected = {int(value) for value in self.classes_}
        if set(self.feature_subsets) != expected:
            raise ValueError(
                "feature_subsets must contain one subset for every class"
            )
        if self.calibration_method != "sigmoid":
            raise ValueError("formal protocol requires sigmoid calibration")
        splits = safe_calibration_splits(y, self.calibration_splits)
        ids = (
            np.asarray(record_ids, dtype=object)
            if record_ids is not None
            else np.asarray(
                [f"local_{index}" for index in range(len(y))],
                dtype=object,
            )
        )
        if len(ids) != len(y):
            raise ValueError("record_ids length does not match training rows")

        self.models_: dict[int, CalibratedClassifierCV] = {}
        self.calibration_audit_: dict[str, object] = {
            "method": self.calibration_method,
            "display_name": "Platt-style sigmoid calibration",
            "family": self.family,
            "requested_splits": int(self.calibration_splits),
            "actual_splits_by_submodel": {},
            "fit_record_ids": [str(value) for value in ids],
            "inference_labels_available_to_calibrator": False,
            "training_partition_only": True,
            "submodels": {},
        }
        for offset, target_class in enumerate(self.classes_):
            target_class = int(target_class)
            submodel_seed = self.seed + 101 * offset
            splitter = StratifiedKFold(
                n_splits=splits,
                shuffle=True,
                random_state=self.seed + 1009 * (offset + 1),
            )
            cv_splits = [
                (train.copy(), validation.copy())
                for train, validation in splitter.split(X, y)
            ]
            base = make_base_classifier(
                self.family,
                submodel_seed,
                self.parameters,
            )
            calibrated = CalibratedClassifierCV(
                estimator=base,
                method=self.calibration_method,
                cv=cv_splits,
                n_jobs=-1,
                ensemble=True,
            )
            calibrated.fit(
                X.loc[:, self.feature_subsets[target_class]],
                y,
            )
            self.models_[target_class] = calibrated
            self.calibration_audit_[
                "actual_splits_by_submodel"
            ][str(target_class)] = int(splits)
            self.calibration_audit_["submodels"][str(target_class)] = {
                "features": self.feature_subsets[target_class],
                "splits": [
                    {
                        "train_record_ids": [
                            str(value) for value in ids[train]
                        ],
                        "validation_record_ids": [
                            str(value) for value in ids[validation]
                        ],
                        "overlap_count": int(
                            len(np.intersect1d(train, validation))
                        ),
                    }
                    for train, validation in cv_splits
                ],
            }
        return self

    def _aligned_probability(
        self,
        model: CalibratedClassifierCV,
        X: pd.DataFrame,
        features: list[str],
    ) -> np.ndarray:
        raw = model.predict_proba(X.loc[:, features])
        aligned = np.zeros(
            (len(X), len(self.classes_)),
            dtype=float,
        )
        for source_column, class_value in enumerate(model.classes_):
            destination = int(
                np.flatnonzero(self.classes_ == class_value)[0]
            )
            aligned[:, destination] = raw[:, source_column]
        return normalize_probability_rows(aligned)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "models_"):
            raise RuntimeError("model must be fitted before prediction")
        probabilities = {
            target_class: self._aligned_probability(
                model,
                X,
                self.feature_subsets[target_class],
            )
            for target_class, model in self.models_.items()
        }
        if self.aggregation == "mean_vector":
            return normalize_probability_rows(
                np.mean(list(probabilities.values()), axis=0)
            )
        if self.aggregation != "target_probability":
            raise ValueError(f"unsupported aggregation: {self.aggregation}")
        combined = np.zeros(
            (len(X), len(self.classes_)),
            dtype=float,
        )
        for target_class, probability in probabilities.items():
            destination = int(
                np.flatnonzero(self.classes_ == target_class)[0]
            )
            combined[:, destination] = probability[:, destination]
        return normalize_probability_rows(combined)

    def calibration_audit_metadata(self) -> dict[str, object]:
        if not hasattr(self, "calibration_audit_"):
            raise RuntimeError("model must be fitted before audit")
        return self.calibration_audit_
