from __future__ import annotations

import math
import re

import pandas as pd


def duration_seconds(value: object) -> float | None:
    """Convert numeric and interval-like STATSports durations to seconds."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "nat", "none", "<na>"}:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", raw):
        return float(raw)

    match = re.fullmatch(
        r"(?:(\d+)\s+days?\s+)?(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?",
        raw,
        re.IGNORECASE,
    )
    if not match:
        return None

    days = int(match.group(1) or 0)
    first = int(match.group(2))
    second = int(match.group(3))
    third = float(match.group(4)) if match.group(4) is not None else None
    clock_seconds = first * 60 + second if third is None else first * 3600 + second * 60 + third
    return days * 86400 + clock_seconds


def normalize_hr_zone_seconds(
    frame: pd.DataFrame,
    absolute_columns: list[str],
    legacy_minute_columns: list[str],
) -> pd.DataFrame:
    """Normalize new absolute durations and historical minute fields to seconds."""
    normalized = frame.copy()
    for zone_index, column in enumerate(absolute_columns):
        absolute_seconds = (
            normalized[column].map(duration_seconds)
            if column in normalized.columns
            else pd.Series(float("nan"), index=normalized.index, dtype=float)
        )
        if zone_index < len(legacy_minute_columns):
            legacy_column = legacy_minute_columns[zone_index]
            if legacy_column in normalized.columns:
                legacy_seconds = pd.to_numeric(normalized[legacy_column], errors="coerce").mul(60)
                absolute_seconds = absolute_seconds.where(absolute_seconds.notna(), legacy_seconds)
        normalized[column] = absolute_seconds.fillna(0.0)
    return normalized
