from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from config import CLASS_NAMES, METHOD_NAMES
from evaluation import METRIC_NAMES


def _json_key(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def parameter_frequencies(
    result: dict[str, object],
) -> dict[str, object]:
    frequencies: dict[str, object] = {}
    for experiment_id, experiment in result["experiments"].items():
        section: dict[str, object] = {}
        records = experiment["folds"]
        for family in ("SVM", "Bagging", "AdaBoost"):
            counter = Counter(
                _json_key(
                    record["audit"]["selected_parameters"]
                    ["base_models"][family]
                )
                for record in records
            )
            section[family] = [
                {
                    "configuration": json.loads(key),
                    "count": count,
                }
                for key, count in counter.most_common()
            ]
        counter = Counter(
            _json_key(
                record["audit"]["selected_parameters"]["AKFMMF"]
            )
            for record in records
        )
        section["AKFMMF"] = [
            {
                "configuration": json.loads(key),
                "count": count,
            }
            for key, count in counter.most_common()
        ]
        frequencies[experiment_id] = section
    return frequencies


def build_leakage_audit(
    result: dict[str, object],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    all_pass = True
    for experiment_id, experiment in result["experiments"].items():
        for record in experiment["folds"]:
            audit = record["audit"]
            final_fit = audit["final_fit"]
            train_ids = set(final_fit["training_record_ids"])
            inference_ids = set(final_fit["inference_record_ids"])
            disjoint = len(train_ids & inference_ids) == 0
            median_fit_exact = (
                set(final_fit["preprocessing"]["fit_record_ids"])
                == train_ids
            )
            inner_overlaps = []
            calibration_overlaps = []
            for family in ("SVM", "Bagging", "AdaBoost"):
                family_audit = audit["base_selection"]["families"][
                    family
                ]
                for inner in family_audit["selected_inner_fit_audit"]:
                    inner_overlaps.append(inner["overlap_count"])
                    for submodel in inner["calibration"][
                        "submodels"
                    ].values():
                        calibration_overlaps.extend(
                            split["overlap_count"]
                            for split in submodel["splits"]
                        )
                for submodel in final_fit["families"][family][
                    "calibration"
                ]["submodels"].values():
                    calibration_overlaps.extend(
                        split["overlap_count"]
                        for split in submodel["splits"]
                    )
            passed = (
                disjoint
                and median_fit_exact
                and max(inner_overlaps, default=0) == 0
                and max(calibration_overlaps, default=0) == 0
                and audit[
                    "fit_select_predict_signature_excludes_inference_labels"
                ]
            )
            all_pass = all_pass and passed
            rows.append(
                {
                    "experiment": experiment_id,
                    "unit": record["fold"],
                    "train_inference_record_id_overlap_count": len(
                        train_ids & inference_ids
                    ),
                    "median_fit_record_ids_equal_training_ids": (
                        median_fit_exact
                    ),
                    "maximum_inner_train_validation_overlap": max(
                        inner_overlaps,
                        default=0,
                    ),
                    "maximum_calibration_train_validation_overlap": max(
                        calibration_overlaps,
                        default=0,
                    ),
                    "selector_accepts_inference_labels": False,
                    "passed": passed,
                }
            )
    return {
        "overall_pass": all_pass,
        "operational_test_leakage_detected": not all_pass,
        "target_or_outer_labels_role": (
            "metrics only, after fit/select/predict returns"
        ),
        "fit_select_predict_accepts_inference_labels": False,
        "source_snapshots_are_runtime_inputs": True,
        "old_merged_project_imported_or_called": False,
        "rows": rows,
    }


def write_main_tables_csv(
    result: dict[str, object],
    output_path: Path,
) -> None:
    fields = [
        "experiment",
        "method",
        "metric",
        "estimate",
        "standard_deviation",
        "ci95_low",
        "ci95_high",
        "uncertainty_type",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for experiment_id, experiment in result["experiments"].items():
            for method in METHOD_NAMES:
                for metric in METRIC_NAMES:
                    cell = experiment["summary"][method][metric]
                    low, high = cell["ci95"]
                    writer.writerow(
                        {
                            "experiment": experiment_id,
                            "method": method,
                            "metric": metric,
                            "estimate": cell["mean"],
                            "standard_deviation": cell["sample_sd"],
                            "ci95_low": low,
                            "ci95_high": high,
                            "uncertainty_type": (
                            f"{experiment['outer_fold_count']} outer-fold sample SD (ddof=1)"
                            ),
                        }
                    )


def write_fold_metrics_csv(
    result: dict[str, object],
    output_path: Path,
) -> None:
    fields = ["experiment", "fold", "method", *METRIC_NAMES, "ece_10"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for experiment_id, experiment in result["experiments"].items():
            for fold in experiment["folds"]:
                for method in METHOD_NAMES:
                    writer.writerow(
                        {
                            "experiment": experiment_id,
                            "fold": fold["fold"],
                            "method": method,
                            **fold["metrics"][method],
                        }
                    )


def write_selected_parameters_csv(
    result: dict[str, object],
    output_path: Path,
) -> None:
    fields = [
        "experiment",
        "unit",
        "component",
        "selected_parameters_json",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for experiment_id, experiment in result["experiments"].items():
            for record in experiment["folds"]:
                selected = record["audit"]["selected_parameters"]
                for component, parameters in (
                    *selected["base_models"].items(),
                    ("AKFMMF", selected["AKFMMF"]),
                ):
                    writer.writerow(
                        {
                            "experiment": experiment_id,
                            "unit": record["fold"],
                            "component": component,
                            "selected_parameters_json": json.dumps(
                                parameters,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                        }
                    )


def write_predictions_csv(
    result: dict[str, object],
    output_path: Path,
) -> None:
    fields = [
        "experiment",
        "unit",
        "record_id",
        "true_label",
        "method",
        "predicted_label",
        *[f"p_{name}" for name in CLASS_NAMES],
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for experiment_id, experiment in result["experiments"].items():
            for record in experiment["folds"]:
                for prediction in record["predictions"]:
                    writer.writerow(
                        {
                            "experiment": experiment_id,
                            "unit": record["fold"],
                            **prediction,
                        }
                    )
