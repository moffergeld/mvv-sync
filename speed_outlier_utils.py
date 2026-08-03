from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def sanitize_progressive_max_speed(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    speed_col: str = "max_speed",
    date_col: str = "datum",
    order_cols: Sequence[str] | None = None,
    multiplier: float = 1.30,
) -> pd.Series:
    if df.empty or speed_col not in df.columns:
        return pd.Series(index=df.index, dtype="float64")

    active_group_cols = [column for column in group_cols if column in df.columns]
    active_order_cols = [column for column in (order_cols or ()) if column in df.columns]

    working_df = df.copy()
    working_df[speed_col] = pd.to_numeric(working_df[speed_col], errors="coerce")
    if date_col in working_df.columns:
        working_df[date_col] = pd.to_datetime(working_df[date_col], errors="coerce")

    sort_cols: list[str] = []
    sort_cols.extend(active_group_cols)
    if date_col in working_df.columns:
        sort_cols.append(date_col)
    for column in active_order_cols:
        if column not in sort_cols:
            sort_cols.append(column)

    if sort_cols:
        working_df = working_df.sort_values(sort_cols, kind="mergesort")

    sanitized_values: dict[object, float] = {}
    grouped_frames = (
        working_df.groupby(active_group_cols, dropna=False, sort=False)
        if active_group_cols
        else [(None, working_df)]
    )

    for _, group_df in grouped_frames:
        accepted_max: float | None = None
        for idx, raw_value in group_df[speed_col].items():
            if pd.isna(raw_value) or not np.isfinite(raw_value) or float(raw_value) <= 0:
                sanitized_values[idx] = np.nan
                continue

            value = float(raw_value)
            if accepted_max is None:
                accepted_max = value
                sanitized_values[idx] = value
                continue

            if value > accepted_max * float(multiplier):
                sanitized_values[idx] = np.nan
                continue

            sanitized_values[idx] = value
            if value > accepted_max:
                accepted_max = value

    return pd.Series(sanitized_values, index=working_df.index, dtype="float64").reindex(df.index)
