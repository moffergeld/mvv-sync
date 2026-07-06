from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

from roles import cookie_mgr


ACWR_MODE_STANDARD = "standard_4w_mean"
ACWR_MODE_LOG_6W = "log_6w_weighted"
DEFAULT_ACWR_MODE = ACWR_MODE_STANDARD
ACWR_MODE_COOKIE = "mvv_acwr_mode"
ACWR_MODE_MAX_AGE = 60 * 60 * 24 * 180

ACWR_MODE_META: Dict[str, Dict[str, Any]] = {
    ACWR_MODE_STANDARD: {
        "label": "Standaard 4 weken",
        "short_label": "4w gemiddeld",
        "description": "Huidige week gedeeld door het gemiddelde van de vorige 4 weken.",
        "window": 4,
        "min_periods": 2,
    },
    ACWR_MODE_LOG_6W: {
        "label": "Logaritmisch 6 weken",
        "short_label": "log 6w",
        "description": "Huidige week gedeeld door een logaritmisch gewogen gemiddelde van de vorige 6 weken, met meer gewicht voor recente weken.",
        "window": 6,
        "min_periods": 3,
    },
}


def normalize_acwr_mode(value: Any) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in ACWR_MODE_META else DEFAULT_ACWR_MODE


def get_acwr_mode() -> str:
    if "acwr_mode" in st.session_state:
        return normalize_acwr_mode(st.session_state.get("acwr_mode"))

    raw_value = None
    try:
        raw_value = cookie_mgr().get(ACWR_MODE_COOKIE)
    except Exception:
        raw_value = None

    mode = normalize_acwr_mode(raw_value)
    st.session_state["acwr_mode"] = mode
    return mode


def set_acwr_mode(mode: str) -> str:
    normalized = normalize_acwr_mode(mode)
    st.session_state["acwr_mode"] = normalized
    try:
        cookie_mgr().set(
            ACWR_MODE_COOKIE,
            normalized,
            max_age=ACWR_MODE_MAX_AGE,
            key="set_mvv_acwr_mode",
        )
    except Exception:
        pass
    return normalized


def get_acwr_mode_meta(mode: str | None = None) -> Dict[str, Any]:
    resolved_mode = normalize_acwr_mode(mode or get_acwr_mode())
    meta = ACWR_MODE_META[resolved_mode].copy()
    meta["mode"] = resolved_mode
    return meta


def _log_weighted_average(values: np.ndarray, window: int) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")

    weights = np.log1p(np.arange(1, window + 1, dtype=float))
    window_weights = weights[-len(arr):]
    mask = ~np.isnan(arr)
    if not mask.any():
        return float("nan")

    arr = arr[mask]
    window_weights = window_weights[mask]
    total_weight = float(window_weights.sum())
    if total_weight == 0:
        return float("nan")
    return float(np.dot(arr, window_weights) / total_weight)


def compute_chronic_series(series: pd.Series, mode: str | None = None) -> pd.Series:
    meta = get_acwr_mode_meta(mode)
    numeric_series = pd.to_numeric(series, errors="coerce")
    shifted = numeric_series.shift(1)

    if meta["mode"] == ACWR_MODE_STANDARD:
        return shifted.rolling(window=meta["window"], min_periods=meta["min_periods"]).mean()

    return shifted.rolling(window=meta["window"], min_periods=meta["min_periods"]).apply(
        lambda values: _log_weighted_average(values, int(meta["window"])),
        raw=True,
    )


def compute_chronic_reference_value(series: pd.Series, mode: str | None = None) -> float | None:
    meta = get_acwr_mode_meta(mode)
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        return None

    recent_values = numeric_series.tail(int(meta["window"]))
    if len(recent_values) < int(meta["min_periods"]):
        return None

    if meta["mode"] == ACWR_MODE_STANDARD:
        return float(recent_values.mean())

    value = _log_weighted_average(recent_values.to_numpy(dtype=float), int(meta["window"]))
    return None if pd.isna(value) else float(value)
