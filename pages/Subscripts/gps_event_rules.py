from __future__ import annotations

import pandas as pd


MATCH_FULL = "Full match"
MATCH_FIRST = "First Half"
MATCH_SECOND = "Second Half"

MATCH_PHASE_EVENT_KEYS = {
    MATCH_FIRST: {"firsthalf", "matchhalvesfirsthalf", "matchhalfesfirsthalf"},
    MATCH_SECOND: {"secondhalf", "matchhalvessecondhalf", "matchhalfessecondhalf"},
}


def select_match_phase_rows(
    rows: pd.DataFrame,
    phase: str,
    *,
    event_column: str = "event",
) -> pd.DataFrame:
    """Select match halves without ever using an entire-session load row."""

    if rows.empty or event_column not in rows.columns:
        return rows.copy()

    selected = rows.copy()
    selected["event_norm"] = (
        selected[event_column]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )

    if phase == MATCH_FIRST:
        allowed = MATCH_PHASE_EVENT_KEYS[MATCH_FIRST]
    elif phase == MATCH_SECOND:
        allowed = MATCH_PHASE_EVENT_KEYS[MATCH_SECOND]
    else:
        allowed = MATCH_PHASE_EVENT_KEYS[MATCH_FIRST] | MATCH_PHASE_EVENT_KEYS[MATCH_SECOND]

    return selected[selected["event_norm"].isin(allowed)].copy()
