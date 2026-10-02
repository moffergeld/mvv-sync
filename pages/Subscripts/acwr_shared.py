from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from acwr_settings import compute_chronic_series, get_acwr_mode_meta
from pages.Subscripts.gps_hybrid_source import load_hybrid_gps
from pages.Subscripts.week_session_labels import source_value


ACWR_LOW = 0.80
ACWR_HIGH = 1.50
ACWR_METRICS: dict[str, dict[str, str]] = {
    "total_distance": {"label": "Total Distance", "short": "TD", "color": "#526986"},
    "total_distance_zone_5": {"label": "Zone 5", "short": "Z5", "color": "#EDB45B"},
    "total_distance_zone_6": {"label": "Zone 6", "short": "Z6", "color": "#E84664"},
    "high_metabolic_load_distance": {"label": "HMLD", "short": "HMLD", "color": "#69D5CB"},
}
ACWR_COLUMNS = [
    "player_id",
    "player_name",
    "datum",
    "event",
    "extra_metrics",
    *ACWR_METRICS.keys(),
]


@st.cache_data(show_spinner=False, ttl=180)
def load_acwr_gps_cached(access_token: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    return load_hybrid_gps(
        access_token,
        ACWR_COLUMNS,
        start=start_iso,
        end=end_iso,
        event="Summary",
    )


@st.cache_data(show_spinner=False, ttl=300)
def fetch_player_positions_cached(_sb, access_scope: str) -> dict[str, str]:
    del access_scope
    rows: list[dict[str, Any]] = []
    for select_clause in ("player_id,position", 'player_id,"Position"', "player_id"):
        try:
            rows = _sb.table("players").select(select_clause).execute().data or []
            break
        except Exception:
            rows = []
    return {
        str(row.get("player_id")): str(row.get("Position") or row.get("position") or "").strip()
        for row in rows
        if str(row.get("player_id") or "").strip()
    }


def _is_goalkeeper(value: object) -> bool:
    text = str(value or "").strip().lower()
    normalized = re.sub(r"[^a-z]", "", text)
    return bool(
        re.search(r"(?:^|[^a-z])gk(?:$|[^a-z])", text)
        or any(label in normalized for label in ("goalkeeper", "keeper", "doelman", "goalie"))
    )


def exclude_goalkeepers(df: pd.DataFrame, positions: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    result = df.copy()
    mapped = result.get("player_id", pd.Series("", index=result.index)).fillna("").astype(str).map(positions).fillna("")
    extra_column = "extra_metrics" if "extra_metrics" in result.columns else "Extra Metrics" if "Extra Metrics" in result.columns else None
    if extra_column:
        fallback = result[extra_column].map(
            lambda value: source_value(value, "Player Position", "playerPrimaryPosition", "primaryPosition")
        )
        mapped = mapped.where(mapped.astype(str).str.strip().ne(""), fallback)
    return result.loc[~mapped.map(_is_goalkeeper)].copy()


def current_week_start(today: date | pd.Timestamp | None = None) -> pd.Timestamp:
    stamp = pd.Timestamp(today or date.today()).normalize()
    return stamp - pd.Timedelta(days=int(stamp.weekday()))


def build_weekly_loads(gps_df: pd.DataFrame, *, through_week: pd.Timestamp | None = None) -> pd.DataFrame:
    columns = ["week_start", "week_label", "player_id", "player_name", *ACWR_METRICS.keys()]
    if gps_df is None or gps_df.empty:
        return pd.DataFrame(columns=columns)
    work = gps_df.copy()
    work["datum"] = pd.to_datetime(work["datum"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["datum", "player_id"]).copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    work["player_id"] = work["player_id"].astype(str)
    work["player_name"] = work["player_name"].fillna(work["player_id"]).astype(str)
    for metric in ACWR_METRICS:
        work[metric] = pd.to_numeric(work.get(metric), errors="coerce")
    work["week_start"] = work["datum"] - pd.to_timedelta(work["datum"].dt.weekday, unit="D")
    grouped = (
        work.groupby(["week_start", "player_id", "player_name"], as_index=False)[list(ACWR_METRICS)]
        .sum(min_count=1)
        .sort_values(["player_name", "week_start"])
    )
    calendar_end = (
        pd.Timestamp(through_week).normalize()
        if through_week is not None
        else pd.Timestamp(grouped["week_start"].max()).normalize()
    )
    calendar_rows: list[pd.DataFrame] = []
    for player_id, player_df in grouped.groupby("player_id", sort=False):
        player = player_df.sort_values("week_start").copy()
        first_week = pd.Timestamp(player["week_start"].min()).normalize()
        if first_week > calendar_end:
            continue
        player_name = str(player["player_name"].dropna().iloc[-1])
        player = player.set_index("week_start").reindex(pd.date_range(first_week, calendar_end, freq="7D"))
        player.index.name = "week_start"
        player["player_id"] = str(player_id)
        player["player_name"] = player_name
        calendar_rows.append(player.reset_index())
    grouped = pd.concat(calendar_rows, ignore_index=True) if calendar_rows else grouped.iloc[0:0].copy()
    grouped["week_label"] = grouped["week_start"].map(
        lambda value: f"{pd.Timestamp(value).isocalendar().year} | Week {int(pd.Timestamp(value).isocalendar().week)}"
    )
    return grouped[columns].sort_values(["player_name", "week_start"]).reset_index(drop=True)


def add_acwr_columns(weekly_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if weekly_df is None or weekly_df.empty:
        return pd.DataFrame(columns=list(weekly_df.columns) if weekly_df is not None else [])
    output: list[pd.DataFrame] = []
    for _, player_df in weekly_df.groupby("player_id", sort=False):
        player = player_df.sort_values("week_start").copy()
        for metric in ACWR_METRICS:
            chronic = compute_chronic_series(player[metric], mode)
            player[f"{metric}_chronic"] = chronic
            player[f"{metric}_acwr"] = player[metric].div(chronic.where(chronic > 0))
        output.append(player)
    return pd.concat(output, ignore_index=True) if output else weekly_df.copy()


def build_player_monitor(
    weekly_df: pd.DataFrame,
    *,
    week_start: pd.Timestamp,
    mode: str,
) -> pd.DataFrame:
    metric_columns = list(ACWR_METRICS)
    output_columns = ["player_id", "player_name", "week_start", "week_label"]
    for metric in metric_columns:
        output_columns.extend(
            [metric, f"{metric}_chronic", f"{metric}_acwr", f"{metric}_to_low", f"{metric}_to_high"]
        )
    if weekly_df is None or weekly_df.empty:
        return pd.DataFrame(columns=output_columns)

    target_week = pd.Timestamp(week_start).normalize()
    players = weekly_df[["player_id", "player_name"]].drop_duplicates("player_id")
    rows: list[dict[str, Any]] = []
    meta = get_acwr_mode_meta(mode)
    for player in players.itertuples(index=False):
        history = weekly_df[
            (weekly_df["player_id"].astype(str) == str(player.player_id))
            & (pd.to_datetime(weekly_df["week_start"]) < target_week)
        ].sort_values("week_start")
        current = weekly_df[
            (weekly_df["player_id"].astype(str) == str(player.player_id))
            & (pd.to_datetime(weekly_df["week_start"]) == target_week)
        ]
        row: dict[str, Any] = {
            "player_id": str(player.player_id),
            "player_name": str(player.player_name),
            "week_start": target_week,
            "week_label": f"{target_week.isocalendar().year} | Week {int(target_week.isocalendar().week)}",
            "reference_weeks": int(meta["window"]),
        }
        for metric in metric_columns:
            acute = float(pd.to_numeric(current[metric], errors="coerce").sum(min_count=1)) if not current.empty else 0.0
            if pd.isna(acute):
                acute = 0.0
            history_values = pd.to_numeric(history[metric], errors="coerce")
            chronic_series = compute_chronic_series(
                pd.concat([history_values.reset_index(drop=True), pd.Series([acute])], ignore_index=True),
                mode,
            )
            chronic = chronic_series.iloc[-1] if not chronic_series.empty else np.nan
            chronic_value = float(chronic) if pd.notna(chronic) and float(chronic) > 0 else np.nan
            ratio = acute / chronic_value if pd.notna(chronic_value) else np.nan
            row[metric] = acute
            row[f"{metric}_chronic"] = chronic_value
            row[f"{metric}_acwr"] = ratio
            row[f"{metric}_to_low"] = max(0.0, ACWR_LOW * chronic_value - acute) if pd.notna(chronic_value) else np.nan
            row[f"{metric}_to_high"] = max(0.0, ACWR_HIGH * chronic_value - acute) if pd.notna(chronic_value) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("player_name").reset_index(drop=True)


def default_history_start(today: date | None = None) -> date:
    return (today or date.today()) - timedelta(days=420)
