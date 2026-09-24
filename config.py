from __future__ import annotations

import itertools
import os
from pathlib import Path


PROJECT_ID = "DGA-IEC167-FORMAL-v8-first6-targeted-4x5"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RUN_MODE = os.environ.get("DGA_RUN_MODE", "full").strip().lower()
if RUN_MODE not in {"full", "quick"}:
    raise ValueError("DGA_RUN_MODE must be full or quick")

GAS_COLUMNS = ("H2", "CH4", "C2H6", "C2H4", "C2H2")
CLASS_NAMES = ("TF", "PD", "DF", "NC")
MODEL_NAMES = ("SVM", "Bagging", "AdaBoost")
METHOD_NAMES = ("AKFMMF",)

RANDOM_SEED = 42
OUTER_SPLITS = 5
OUTER_REPEATS = 1 if RUN_MODE == "quick" else 4
INNER_SPLITS = 2 if RUN_MODE == "quick" else 3
CALIBRATION_SPLITS = 2 if RUN_MODE == "quick" else 3
BOOTSTRAP_REPLICATES = 2000

PROBABILITY_EPSILON = 1e-9
COVARIANCE_EIGEN_FLOOR = 1e-10
AGGREGATION = "mean_vector"
CALIBRATION_METHOD = "sigmoid"

MODEL_PARAMETER_GRID = {
    "SVM": (
        {"C": c_value, "gamma": gamma}
        for c_value, gamma in itertools.product(
            (0.01, 0.04, 0.20, 1.00, 5.00, 10.00),
            ("scale", 0.03, 0.10, 0.30),
        )
    ),
    "Bagging": (
        {
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "max_depth": max_depth,
        }
        for n_estimators, max_samples, max_depth in itertools.product(
            (2, 20, 50),
            (0.40, 0.80, 1.00),
            (0, 4, 8),
        )
    ),
    "AdaBoost": (
        {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
        }
        for n_estimators, learning_rate, max_depth in itertools.product(
            (2, 20, 50),
            (0.20, 0.40, 0.80, 1.20),
            (1, 4, 8),
        )
    ),
}
MODEL_PARAMETER_GRID = {
    family: tuple(dict(candidate) for candidate in candidates)
    for family, candidates in MODEL_PARAMETER_GRID.items()
}

REFERENCE_FEATURE_COUNTS = {"SVM": 2, "Bagging": 4, "AdaBoost": 12}
REFERENCE_BINS = 32
REFERENCE_LAPLACE = 0.5
FEATURE_COUNT_CANDIDATES = (1, 2, 4, 8, 12, 16, 20)
BIN_CANDIDATES = (3, 5, 8, 10, 15, 20, 32, 48)
LAPLACE_CANDIDATES = (0.10, 0.25, 0.50, 1.00, 2.00)

BETA_CANDIDATES = (0.25, 0.50, 1.00, 2.00, 4.00, 8.00, 16.00)
INITIAL_COVARIANCE_CANDIDATES = (0.050, 0.075, 0.100, 0.150, 0.250)
PROCESS_COVARIANCE_CANDIDATES = (0.300, 0.400, 0.500, 0.750, 1.000)
MEASUREMENT_COVARIANCE_CANDIDATES = (0.300, 0.500, 0.700, 0.900, 1.100)
FORGETTING_FACTOR_CANDIDATES = (0.850, 0.900, 0.950, 0.980, 0.995)
MAX_COVARIANCE_CHANGE_CANDIDATES = (0.150, 0.250, 0.350, 0.500, 0.750)
CLASSIFIER_ORDER_CANDIDATES = tuple(itertools.permutations(range(3)))

if RUN_MODE == "quick":
    MODEL_PARAMETER_GRID = {
        "SVM": ({"C": 0.04, "gamma": 0.10},),
        "Bagging": (
            {
                "n_estimators": 20,
                "max_samples": 1.00,
                "max_depth": 0,
            },
        ),
        "AdaBoost": (
            {
                "n_estimators": 30,
                "learning_rate": 0.80,
                "max_depth": 4,
            },
        ),
    }
    REFERENCE_FEATURE_COUNTS = {"SVM": 2, "Bagging": 8, "AdaBoost": 12}
    FEATURE_COUNT_CANDIDATES = (2, 8, 12)
    BIN_CANDIDATES = (32,)
    LAPLACE_CANDIDATES = (0.25,)
    BETA_CANDIDATES = (8.00,)
    INITIAL_COVARIANCE_CANDIDATES = (0.250,)
    PROCESS_COVARIANCE_CANDIDATES = (0.800,)
    MEASUREMENT_COVARIANCE_CANDIDATES = (0.500,)
    FORGETTING_FACTOR_CANDIDATES = (0.950,)
    MAX_COVARIANCE_CHANGE_CANDIDATES = (0.350,)

REFERENCE_FUSION = {
    "beta": 1.0,
    "initial_covariance": 0.1,
    "process_covariance": 0.4,
    "measurement_covariance": 0.9,
    "forgetting_factor": 0.95,
    "max_covariance_change": 0.25,
    "classifier_order": (0, 1, 2),
}

EXPECTED_SOURCE_COUNTS = {
    "IEC": {"samples": 167, "TF": 34, "PD": 9, "DF": 74, "NC": 50},
}


def protocol_payload() -> dict[str, object]:
    return {
        "project_id": PROJECT_ID,
        "status": "FORMAL_PREDECLARED_NO_POST_HOC_SELECTION",
        "dataset_scope": {
            "name": "IEC",
            "samples": 167,
            "experiment": (
                f"IEC internal {OUTER_REPEATS}x{OUTER_SPLITS} "
                "nested stratified CV only"
            ),
        },
        "run_mode": RUN_MODE,
        "random_seed": RANDOM_SEED,
        "outer_cv": {
            "splitter": "RepeatedStratifiedKFold",
            "splits": OUTER_SPLITS,
            "repeats": OUTER_REPEATS,
        },
        "inner_cv": {
            "splitter": "StratifiedKFold(shuffle=True)",
            "splits": INNER_SPLITS,
            "selection_partition": "current outer training partition only",
        },
        "calibration": {
            "method": CALIBRATION_METHOD,
            "requested_splits": CALIBRATION_SPLITS,
            "boundary": "training partition only",
            "aggregation": AGGREGATION,
        },
        "staged_selection": [
            "classifier parameters at predeclared reference CSFS settings",
            "bins and CSFS additive smoothing at the selected classifier parameters",
            "feature subset size at selected classifier/bins/smoothing settings",
            "AKFMMF parameters/order from selected calibrated base OOF probabilities",
        ],
        "selection_objective": [
            "maximize overall accuracy",
            "maximize macro-F1",
            "minimize multiclass Brier score",
            "prefer lower predeclared complexity/change distance",
        ],
        "model_parameter_grid": MODEL_PARAMETER_GRID,
        "reference_feature_counts": REFERENCE_FEATURE_COUNTS,
        "reference_bins": REFERENCE_BINS,
        "reference_laplace": REFERENCE_LAPLACE,
        "feature_count_candidates": FEATURE_COUNT_CANDIDATES,
        "bin_candidates": BIN_CANDIDATES,
        "laplace_candidates": LAPLACE_CANDIDATES,
        "fusion_grid": {
            "beta": BETA_CANDIDATES,
            "initial_covariance_P0": INITIAL_COVARIANCE_CANDIDATES,
            "process_covariance_Q": PROCESS_COVARIANCE_CANDIDATES,
            "measurement_covariance_R": MEASUREMENT_COVARIANCE_CANDIDATES,
            "forgetting_factor_b": FORGETTING_FACTOR_CANDIDATES,
            "max_covariance_change": MAX_COVARIANCE_CHANGE_CANDIDATES,
            "classifier_orders": [
                [MODEL_NAMES[index] for index in order]
                for order in CLASSIFIER_ORDER_CANDIDATES
            ],
            "AKFMMF_candidate_count": (
                len(BETA_CANDIDATES)
                * len(INITIAL_COVARIANCE_CANDIDATES)
                * len(PROCESS_COVARIANCE_CANDIDATES)
                * len(MEASUREMENT_COVARIANCE_CANDIDATES)
                * len(FORGETTING_FACTOR_CANDIDATES)
                * len(MAX_COVARIANCE_CHANGE_CANDIDATES)
                * len(CLASSIFIER_ORDER_CANDIDATES)
            ),
        },
        "fixed_non_tuned_protocol_constants": {
            "probability_epsilon": PROBABILITY_EPSILON,
            "covariance_eigen_floor": COVARIANCE_EIGEN_FLOOR,
            "reason": (
                "numerical/protocol constants, not outcome-selected model "
                "hyperparameters"
            ),
        },
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "method_order": METHOD_NAMES,
        "class_order": CLASS_NAMES,
    }
