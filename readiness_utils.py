from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

import pandas as pd

WELLNESS_PHYSICAL_COLS = ["muscle_soreness", "fatigue"]
WELLNESS_MENTAL_COLS = ["sleep_quality", "stress", "mood"]
WELLNESS_ALL_COLS = WELLNESS_PHYSICAL_COLS + WELLNESS_MENTAL_COLS


def _optional_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


def enrich_wellness_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    for col in WELLNESS_ALL_COLS:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["wellness_physical"] = out[WELLNESS_PHYSICAL_COLS].mean(axis=1)
    out["wellness_mental"] = out[WELLNESS_MENTAL_COLS].mean(axis=1)
    out["wellness_avg"] = out[WELLNESS_ALL_COLS].mean(axis=1)
    out["readiness_score"] = out[["wellness_physical", "wellness_mental"]].max(axis=1)
    no_component_scores = out[["wellness_physical", "wellness_mental"]].isna().all(axis=1)
    out.loc[no_component_scores, "readiness_score"] = pd.NA
    return out


def build_wellness_snapshot_lookup(
    df: pd.DataFrame,
    date_col: str = "entry_date",
    today_value: Optional[date] = None,
) -> Dict[str, Dict[str, Any]]:
    if df.empty:
        return {}

    today_value = today_value or date.today()
    out: Dict[str, Dict[str, Any]] = {}
    for player_id, grp in df.groupby("player_id"):
        grp = grp.sort_values(date_col)
        latest = grp.tail(1).iloc[0]
        today_mask = grp[date_col] == today_value
        row = grp[today_mask].tail(1).iloc[0] if today_mask.any() else latest
        out[str(player_id)] = {
            "overall": _optional_float(row.get("wellness_avg")),
            "physical": _optional_float(row.get("wellness_physical")),
            "mental": _optional_float(row.get("wellness_mental")),
            "readiness_score": _optional_float(row.get("readiness_score")),
            "date": row[date_col],
            "is_today": bool(row[date_col] == today_value),
        }
    return out
