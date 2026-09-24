from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import (
    CLASS_NAMES,
    DATA_DIR,
    EXPECTED_SOURCE_COUNTS,
    GAS_COLUMNS,
    PROBABILITY_EPSILON,
)
from features import FEATURE_NAMES, expand_dga_features


@dataclass(frozen=True)
class SourceDataset:
    name: str
    gases: pd.DataFrame
    y: np.ndarray
    labels: np.ndarray
    record_ids: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_key(row: pd.Series, label: str) -> str:
    parts = []
    for column in GAS_COLUMNS:
        value = row[column]
        parts.append("MISSING" if pd.isna(value) else f"{float(value):.14g}")
    return "|".join((*parts, str(label)))


def load_snapshot(path: Path, name: str) -> SourceDataset:
    table = pd.read_csv(path)
    required = ("record_id", *GAS_COLUMNS, "Fault_Type")
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}")
    gases = table.loc[:, GAS_COLUMNS].apply(pd.to_numeric, errors="coerce")
    labels = table["Fault_Type"].astype(str).str.strip().to_numpy(dtype=object)
    unknown = sorted(set(labels) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"{path.name} contains unknown labels: {unknown}")
    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    y = np.asarray([class_to_index[str(label)] for label in labels], dtype=int)
    return SourceDataset(
        name=name,
        gases=gases.reset_index(drop=True),
        y=y,
        labels=labels,
        record_ids=table["record_id"].astype(str).to_numpy(dtype=object),
    )


def load_dataset(
    data_dir: Path = DATA_DIR,
) -> tuple[SourceDataset, dict[str, object]]:
    manifest_path = data_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = data_dir / "iec_167.csv"
    expected_hash = manifest["snapshot_sha256"][path.name]
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise RuntimeError(f"frozen IEC snapshot hash mismatch: {observed_hash}")
    dataset = load_snapshot(path, "IEC")
    audit: dict[str, object] = {
        "manifest": manifest,
        "sources": {},
        "duplicate_policy": "retain all source records; no deduplication",
    }
    counts = Counter(str(value) for value in dataset.labels)
    expected = EXPECTED_SOURCE_COUNTS["IEC"]
    observed = {"samples": len(dataset.y), **{name: int(counts.get(name, 0)) for name in CLASS_NAMES}}
    if observed != expected:
        raise RuntimeError(f"IEC boundary/count mismatch: {observed} != {expected}")
    audit["sources"]["IEC"] = {
        **observed,
        "missing_by_gas": {column: int(dataset.gases[column].isna().sum()) for column in GAS_COLUMNS},
        "exact_duplicates_beyond_first": int(pd.DataFrame({**{column: dataset.gases[column] for column in GAS_COLUMNS}, "Fault_Type": dataset.labels}).duplicated(keep="first").sum()),
        "record_id_unique": bool(len(set(dataset.record_ids)) == len(dataset.record_ids)),
    }
    audit["scope"] = "IEC 167 records only; no self-constructed or cross-dataset data"
    return dataset, audit


class TrainingMedianPreprocessor:

    def __init__(self, epsilon: float = PROBABILITY_EPSILON):
        self.epsilon = float(epsilon)
        self.scaler = MinMaxScaler(clip=False)
        self._fitted = False

    def fit(
        self,
        gases: pd.DataFrame,
        record_ids: np.ndarray | None = None,
    ) -> "TrainingMedianPreprocessor":
        numeric = gases.loc[:, GAS_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )
        medians = numeric.median(axis=0, skipna=True)
        if medians.isna().any():
            missing = medians[medians.isna()].index.tolist()
            raise ValueError(f"training partition has all-missing gases: {missing}")
        self.medians_ = medians.astype(float)
        imputed = numeric.fillna(self.medians_)
        expanded = expand_dga_features(
            imputed.to_numpy(dtype=float),
            epsilon=self.epsilon,
        )
        transformed = np.arctan(expanded.to_numpy(dtype=float)) * (2.0 / np.pi)
        self.scaler.fit(transformed)
        self.fit_record_ids_ = (
            [] if record_ids is None else [str(value) for value in record_ids]
        )
        self._fitted = True
        return self

    def transform(self, gases: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("preprocessor must be fitted before transform")
        numeric = gases.loc[:, GAS_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )
        imputed = numeric.fillna(self.medians_)
        if imputed.isna().any().any():
            raise RuntimeError("median imputation left missing values")
        expanded = expand_dga_features(
            imputed.to_numpy(dtype=float),
            epsilon=self.epsilon,
        )
        transformed = np.arctan(expanded.to_numpy(dtype=float)) * (2.0 / np.pi)
        scaled = self.scaler.transform(transformed)
        result = pd.DataFrame(
            scaled,
            columns=FEATURE_NAMES,
            index=gases.index,
        )
        if not np.isfinite(result.to_numpy()).all():
            raise FloatingPointError("preprocessing produced non-finite values")
        return result

    def fit_transform(
        self,
        gases: pd.DataFrame,
        record_ids: np.ndarray | None = None,
    ) -> pd.DataFrame:
        return self.fit(gases, record_ids).transform(gases)

    def audit_metadata(self) -> dict[str, object]:
        if not self._fitted:
            raise RuntimeError("preprocessor must be fitted before audit")
        return {
            "imputer": "per-gas training-partition median",
            "fit_record_ids": self.fit_record_ids_,
            "gas_medians": {
                name: float(self.medians_[name])
                for name in GAS_COLUMNS
            },
            "feature_minima": {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, self.scaler.data_min_)
            },
            "feature_maxima": {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, self.scaler.data_max_)
            },
            "median_before_feature_expansion": True,
            "ratio_epsilon": self.epsilon,
            "arctangent_transform": "atan(x) * 2/pi",
        }
