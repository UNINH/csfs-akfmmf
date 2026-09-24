from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _normalized_counts(
    shape: tuple[int, ...],
    indices: tuple[np.ndarray, ...],
    alpha: float,
) -> np.ndarray:
    counts = np.full(shape, float(alpha), dtype=float)
    np.add.at(counts, indices, 1.0)
    return counts / counts.sum()


def class_specific_information_bits(
    x: np.ndarray,
    y_binary: np.ndarray,
    x_states: int,
    alpha: float,
) -> float:
    joint = _normalized_counts((x_states, 2), (x, y_binary), alpha)
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    contribution = joint[:, 1] * np.log2(
        joint[:, 1] / (px * py[1])
    )
    return float(np.sum(contribution))


def class_specific_conditional_information_bits(
    x: np.ndarray,
    y_binary: np.ndarray,
    z: np.ndarray,
    x_states: int,
    z_states: int,
    alpha: float,
) -> float:
    joint = _normalized_counts(
        (x_states, 2, z_states),
        (x, y_binary, z),
        alpha,
    )
    pz = joint.sum(axis=(0, 1))
    pxz = joint.sum(axis=1)
    pyz = joint.sum(axis=0)
    p_x1z = joint[:, 1, :]
    ratio = (p_x1z * pz[None, :]) / (
        pxz * pyz[1, :][None, :]
    )
    return float(np.sum(p_x1z * np.log2(ratio)))


def class_specific_redundancy_bits(
    x: np.ndarray,
    z: np.ndarray,
    y_binary: np.ndarray,
    x_states: int,
    z_states: int,
    alpha: float,
) -> float:
    joint = _normalized_counts(
        (x_states, 2, z_states),
        (x, y_binary, z),
        alpha,
    )
    p_y = joint.sum(axis=(0, 2))
    p_xy = joint.sum(axis=2)
    p_yz = joint.sum(axis=0)
    p_x1z = joint[:, 1, :]
    ratio = (p_x1z * p_y[1]) / (
        p_xy[:, 1][:, None] * p_yz[1, :][None, :]
    )
    return float(max(np.sum(p_x1z * np.log2(ratio)), 0.0))


def class_specific_interaction_information_bits(
    x: np.ndarray,
    z: np.ndarray,
    y_binary: np.ndarray,
    x_states: int,
    z_states: int,
    alpha: float,
) -> float:
    joint = _normalized_counts(
        (x_states, 2, z_states),
        (x, y_binary, z),
        alpha,
    )
    p_xz = joint.sum(axis=1)
    p_xy = joint.sum(axis=2)
    p_yz = joint.sum(axis=0)
    p_x = joint.sum(axis=(1, 2))
    p_y = joint.sum(axis=(0, 2))
    p_z = joint.sum(axis=(0, 1))
    p_x1z = joint[:, 1, :]
    ratio = (
        p_xz
        * p_xy[:, 1][:, None]
        * p_yz[1, :][None, :]
    ) / (
        p_x[:, None]
        * p_z[None, :]
        * p_y[1]
        * p_x1z
    )
    return float(np.sum(p_x1z * np.log2(ratio)))


@dataclass
class QuantileDiscretizer:
    n_bins: int

    def fit(self, X: pd.DataFrame) -> "QuantileDiscretizer":
        self.columns_ = tuple(X.columns)
        probabilities = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        self.edges_: dict[str, np.ndarray] = {}
        for column in self.columns_:
            values = X[column].to_numpy(dtype=float)
            self.edges_[column] = np.unique(
                np.quantile(values, probabilities)
            )
        return self

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = X.loc[:, self.columns_]
        columns: list[np.ndarray] = []
        cardinalities: list[int] = []
        for column in self.columns_:
            edges = self.edges_[column]
            columns.append(
                np.digitize(
                    X[column].to_numpy(dtype=float),
                    edges,
                    right=False,
                )
            )
            cardinalities.append(len(edges) + 1)
        return (
            np.column_stack(columns).astype(int),
            np.asarray(cardinalities, dtype=int),
        )

    def fit_transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(X).transform(X)


@dataclass
class CSFSSelector:
    max_features: int
    n_bins: int
    laplace: float

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "CSFSSelector":
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        self.feature_names_ = tuple(X.columns)
        if self.max_features > len(self.feature_names_):
            raise ValueError("max_features exceeds the available features")
        self.discretizer_ = QuantileDiscretizer(self.n_bins)
        discrete, cardinalities = self.discretizer_.fit_transform(X)
        self.rankings_: dict[int, list[str]] = {}
        self.score_history_: dict[
            int,
            list[dict[str, float | str]],
        ] = {}
        for target_class in self.classes_:
            ranking, history = self._rank_for_class(
                discrete,
                cardinalities,
                y,
                int(target_class),
            )
            self.rankings_[int(target_class)] = ranking
            self.score_history_[int(target_class)] = history
        return self

    def _rank_for_class(
        self,
        Xd: np.ndarray,
        cardinalities: np.ndarray,
        y: np.ndarray,
        target_class: int,
    ) -> tuple[list[str], list[dict[str, float | str]]]:
        y_binary = (y == target_class).astype(int)
        candidates = list(range(Xd.shape[1]))
        first_scores = [
            class_specific_information_bits(
                Xd[:, feature],
                y_binary,
                int(cardinalities[feature]),
                self.laplace,
            )
            for feature in candidates
        ]
        first = max(
            candidates,
            key=lambda feature: (first_scores[feature], -feature),
        )
        selected = [first]
        candidates.remove(first)
        history: list[dict[str, float | str]] = [
            {
                "feature": self.feature_names_[first],
                "score": float(first_scores[first]),
                "relevance": float(first_scores[first]),
                "redundancy": 0.0,
                "complementarity": 0.0,
            }
        ]
        while len(selected) < self.max_features:
            best_feature: int | None = None
            best_detail: dict[str, float | str] | None = None
            for feature in candidates:
                relevance_values = []
                redundancy_values = []
                interaction_values = []
                for chosen in selected:
                    relevance_values.append(
                        class_specific_conditional_information_bits(
                            Xd[:, feature],
                            y_binary,
                            Xd[:, chosen],
                            int(cardinalities[feature]),
                            int(cardinalities[chosen]),
                            self.laplace,
                        )
                    )
                    redundancy_values.append(
                        class_specific_redundancy_bits(
                            Xd[:, feature],
                            Xd[:, chosen],
                            y_binary,
                            int(cardinalities[feature]),
                            int(cardinalities[chosen]),
                            self.laplace,
                        )
                    )
                    interaction_values.append(
                        class_specific_interaction_information_bits(
                            Xd[:, feature],
                            Xd[:, chosen],
                            y_binary,
                            int(cardinalities[feature]),
                            int(cardinalities[chosen]),
                            self.laplace,
                        )
                    )
                relevance = min(relevance_values)
                redundancy = float(np.mean(redundancy_values))
                complementarity = -min(interaction_values)
                score = relevance - redundancy + complementarity
                detail: dict[str, float | str] = {
                    "feature": self.feature_names_[feature],
                    "score": float(score),
                    "relevance": float(relevance),
                    "redundancy": float(redundancy),
                    "complementarity": float(complementarity),
                }
                if (
                    best_detail is None
                    or score > float(best_detail["score"]) + 1e-15
                    or (
                        abs(score - float(best_detail["score"])) <= 1e-15
                        and feature < int(best_feature)
                    )
                ):
                    best_feature = feature
                    best_detail = detail
            if best_feature is None or best_detail is None:
                raise RuntimeError("CSFS failed to select a feature")
            selected.append(best_feature)
            candidates.remove(best_feature)
            history.append(best_detail)
        return [self.feature_names_[index] for index in selected], history

    def subsets(self, count: int) -> dict[int, list[str]]:
        if not hasattr(self, "rankings_"):
            raise RuntimeError("selector must be fitted before subsets")
        if count < 1 or count > self.max_features:
            raise ValueError("feature count is outside the fitted ranking")
        return {
            target_class: ranking[:count]
            for target_class, ranking in self.rankings_.items()
        }

    def audit_metadata(self) -> dict[str, object]:
        if not hasattr(self, "rankings_"):
            raise RuntimeError("selector must be fitted before audit")
        return {
            "method": "class-specific relevance-redundancy-complementarity",
            "n_bins": int(self.n_bins),
            "laplace_alpha": float(self.laplace),
            "fit_boundary": "training partition only",
            "binning": "equal-frequency quantiles; duplicate edges removed",
            "bin_edges": {
                feature: edges.tolist()
                for feature, edges in self.discretizer_.edges_.items()
            },
            "logarithm_base": 2,
            "rankings": self.rankings_,
            "score_history": self.score_history_,
        }

