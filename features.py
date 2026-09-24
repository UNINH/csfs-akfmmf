from __future__ import annotations

import numpy as np
import pandas as pd

from config import GAS_COLUMNS


FEATURE_NAMES = (
    "H2",
    "CH4",
    "C2H6",
    "C2H4",
    "C2H2",
    "THC",
    "TCE",
    "TCG",
    "CH4/H2",
    "C2H4/CH4",
    "C2H2/H2",
    "C2H2/CH4",
    "C2H2/C2H4",
    "C2H4/H2",
    "C2H4/C2H6",
    "C2H6/H2",
    "C2H6/CH4",
    "C2H6/C2H2",
    "H2/(H2+CH4+C2H4+C2H2+C2H6)",
    "CH4/(CH4+C2H4+C2H2+C2H6)",
    "C2H2/(CH4+C2H4+C2H2+C2H6)",
    "C2H4/(CH4+C2H4+C2H2+C2H6)",
    "C2H6/(CH4+C2H4+C2H2+C2H6)",
    "(CH4+C2H4)/(CH4+C2H4+C2H2+C2H6)",
    "CH4/(CH4+C2H4+C2H2)",
    "C2H2/(CH4+C2H4+C2H2)",
    "C2H4/(CH4+C2H4+C2H2)",
    "H2/(CH4+C2H4+C2H2+C2H6)",
    "max(key gas)",
    "10/(C2H4/C2H2)",
    "C2H2/TCE/0.21",
    "C2H6/TCE/0.23",
)


def safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    epsilon: float = 1e-9,
) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    near_zero = np.abs(denominator) < epsilon
    signed_epsilon = np.where(denominator < 0.0, -epsilon, epsilon)
    safe_denominator = np.where(near_zero, signed_epsilon, denominator)
    result = numerator / safe_denominator
    result = np.where(near_zero & (np.abs(numerator) < epsilon), 0.0, result)
    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0 / epsilon,
        neginf=-1.0 / epsilon,
    )


def expand_dga_features(
    gases: np.ndarray,
    epsilon: float = 1e-9,
) -> pd.DataFrame:
    gases = np.asarray(gases, dtype=float)
    if gases.ndim != 2 or gases.shape[1] != len(GAS_COLUMNS):
        raise ValueError(f"expected an (n, {len(GAS_COLUMNS)}) gas matrix")

    h2, ch4, c2h6, c2h4, c2h2 = gases.T
    tch = ch4 + c2h4 + c2h2 + c2h6
    tch2 = ch4 + c2h4 + c2h2
    th = h2 + tch
    values = {
        "H2": h2,
        "CH4": ch4,
        "C2H6": c2h6,
        "C2H4": c2h4,
        "C2H2": c2h2,
        "THC": tch,
        "TCE": tch2,
        "TCG": th,
        "CH4/H2": safe_ratio(ch4, h2, epsilon),
        "C2H4/CH4": safe_ratio(c2h4, ch4, epsilon),
        "C2H2/H2": safe_ratio(c2h2, h2, epsilon),
        "C2H2/CH4": safe_ratio(c2h2, ch4, epsilon),
        "C2H2/C2H4": safe_ratio(c2h2, c2h4, epsilon),
        "C2H4/H2": safe_ratio(c2h4, h2, epsilon),
        "C2H4/C2H6": safe_ratio(c2h4, c2h6, epsilon),
        "C2H6/H2": safe_ratio(c2h6, h2, epsilon),
        "C2H6/CH4": safe_ratio(c2h6, ch4, epsilon),
        "C2H6/C2H2": safe_ratio(c2h6, c2h2, epsilon),
        "H2/(H2+CH4+C2H4+C2H2+C2H6)": safe_ratio(h2, th, epsilon),
        "CH4/(CH4+C2H4+C2H2+C2H6)": safe_ratio(ch4, tch, epsilon),
        "C2H2/(CH4+C2H4+C2H2+C2H6)": safe_ratio(c2h2, tch, epsilon),
        "C2H4/(CH4+C2H4+C2H2+C2H6)": safe_ratio(c2h4, tch, epsilon),
        "C2H6/(CH4+C2H4+C2H2+C2H6)": safe_ratio(c2h6, tch, epsilon),
        "(CH4+C2H4)/(CH4+C2H4+C2H2+C2H6)": safe_ratio(
            ch4 + c2h4,
            tch,
            epsilon,
        ),
        "CH4/(CH4+C2H4+C2H2)": safe_ratio(ch4, tch2, epsilon),
        "C2H2/(CH4+C2H4+C2H2)": safe_ratio(c2h2, tch2, epsilon),
        "C2H4/(CH4+C2H4+C2H2)": safe_ratio(c2h4, tch2, epsilon),
        "H2/(CH4+C2H4+C2H2+C2H6)": safe_ratio(h2, tch, epsilon),
        "max(key gas)": np.max(gases, axis=1),
        "10/(C2H4/C2H2)": safe_ratio(10.0 * c2h2, c2h4, epsilon),
        "C2H2/TCE/0.21": safe_ratio(c2h2, 0.21 * tch2, epsilon),
        "C2H6/TCE/0.23": safe_ratio(c2h6, 0.23 * tch2, epsilon),
    }
    expanded = pd.DataFrame(values, columns=FEATURE_NAMES)
    if expanded.shape[1] != 32:
        raise AssertionError("DGA feature expansion must produce 32 features")
    return expanded
