from __future__ import annotations

from datetime import date
from typing import Iterable, Optional

import pandas as pd
import streamlit as st

from readiness_utils import WELLNESS_ALL_COLS, enrich_wellness_scores


WELLNESS_PARAMETER_SPECS: list[tuple[str, str]] = [
    ("muscle_soreness", "Muscle Soreness"),
    ("fatigue", "Fatigue"),
    ("sleep_quality", "Sleep Quality"),
    ("stress", "Stress"),
    ("mood", "Mood"),
]
WELLNESS_PARAMETER_COLUMNS = [column for column, _ in WELLNESS_PARAMETER_SPECS]

MONITORING_NUMERIC_COLUMNS = [
    *WELLNESS_PARAMETER_COLUMNS,
    "wellness_physical",
    "wellness_mental",
    "wellness_avg",
    "readiness_score",
    "avg_rpe",
]

MONITORING_COLUMNS = [
    "entry_date",
    "player_id",
    "player_name",
    *MONITORING_NUMERIC_COLUMNS,
]


def _empty_monitoring_df() -> pd.DataFrame:
    return pd.DataFrame(columns=MONITORING_COLUMNS)


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _coerce_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def _empty_grouped_summary() -> pd.DataFrame:
    base_columns = [
        "bucket",
        "label",
        "wellness_players",
        "rpe_players",
        *WELLNESS_PARAMETER_COLUMNS,
        *(f"{column}_std" for column in WELLNESS_PARAMETER_COLUMNS),
        "wellness_avg",
        "readiness_score",
        "avg_rpe",
        "avg_rpe_std",
    ]
    return pd.DataFrame(columns=base_columns)


def _empty_player_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "player_id",
            "player_name",
            "wellness_days",
            "rpe_days",
            *WELLNESS_PARAMETER_COLUMNS,
            "wellness_avg",
            "readiness_score",
            "avg_rpe",
        ]
    )


@st.cache_data(show_spinner=False, ttl=120)
def fetch_report_wellness_range_cached(sb_url_key: str, _sb, d0_iso: str, d1_iso: str) -> pd.DataFrame:
    rows = (
        _sb.table("asrm_entries")
        .select("player_id,entry_date,muscle_soreness,fatigue,sleep_quality,stress,mood")
        .gte("entry_date", d0_iso)
        .lte("entry_date", d1_iso)
        .execute()
        .data
        or []
    )
    df = _df(rows)
    if df.empty:
        return df
    df["player_id"] = df["player_id"].astype(str)
    df["entry_date"] = _coerce_date(df["entry_date"])
    return enrich_wellness_scores(df)


@st.cache_data(show_spinner=False, ttl=120)
def fetch_report_rpe_headers_range_cached(sb_url_key: str, _sb, d0_iso: str, d1_iso: str) -> pd.DataFrame:
    rows = (
        _sb.table("rpe_entries")
        .select("id,player_id,entry_date")
        .gte("entry_date", d0_iso)
        .lte("entry_date", d1_iso)
        .execute()
        .data
        or []
    )
    df = _df(rows)
    if df.empty:
        return df
    df["id"] = df["id"].astype(str)
    df["player_id"] = df["player_id"].astype(str)
    df["entry_date"] = _coerce_date(df["entry_date"])
    return df


@st.cache_data(show_spinner=False, ttl=120)
def fetch_report_rpe_sessions_for_ids_cached(
    sb_url_key: str,
    _sb,
    entry_ids_tuple: tuple[str, ...],
) -> pd.DataFrame:
    entry_ids = list(entry_ids_tuple)
    if not entry_ids:
        return pd.DataFrame()

    rows = (
        _sb.table("rpe_sessions")
        .select("rpe_entry_id,session_index,rpe")
        .in_("rpe_entry_id", entry_ids)
        .execute()
        .data
        or []
    )
    df = _df(rows)
    if df.empty:
        return df

    df["rpe_entry_id"] = df["rpe_entry_id"].astype(str)
    df["rpe"] = pd.to_numeric(df["rpe"], errors="coerce").fillna(0.0)
    return df


def fetch_report_rpe_daily_cached(sb_url_key: str, sb, d0_iso: str, d1_iso: str) -> pd.DataFrame:
    headers_df = fetch_report_rpe_headers_range_cached(sb_url_key, sb, d0_iso, d1_iso)
    if headers_df.empty:
        return pd.DataFrame(columns=["entry_date", "player_id", "avg_rpe"])

    entry_ids = headers_df["id"].astype(str).tolist()
    sess = fetch_report_rpe_sessions_for_ids_cached(sb_url_key, sb, tuple(entry_ids))
    if sess.empty:
        return pd.DataFrame(columns=["entry_date", "player_id", "avg_rpe"])

    merged = sess.merge(
        headers_df[["id", "player_id", "entry_date"]],
        left_on="rpe_entry_id",
        right_on="id",
        how="left",
    ).dropna(subset=["player_id", "entry_date"])

    out = (
        merged.groupby(["entry_date", "player_id"], as_index=False)
        .agg(avg_rpe=("rpe", "mean"))
        .reset_index(drop=True)
    )
    out["entry_date"] = _coerce_date(out["entry_date"])
    out["player_id"] = out["player_id"].astype(str)
    return out[["entry_date", "player_id", "avg_rpe"]]


def build_monitoring_dataset(
    sb_url_key: str,
    sb,
    start_date: date,
    end_date: date,
    *,
    player_ids: Optional[Iterable[str]] = None,
    player_lookup: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    wellness_df = fetch_report_wellness_range_cached(sb_url_key, sb, start_date.isoformat(), end_date.isoformat())
    rpe_df = fetch_report_rpe_daily_cached(sb_url_key, sb, start_date.isoformat(), end_date.isoformat())

    if player_ids is not None:
        valid_ids = {str(player_id) for player_id in player_ids if str(player_id).strip()}
        if not wellness_df.empty:
            wellness_df = wellness_df[wellness_df["player_id"].astype(str).isin(valid_ids)].copy()
        if not rpe_df.empty:
            rpe_df = rpe_df[rpe_df["player_id"].astype(str).isin(valid_ids)].copy()

    wellness_cols = [
        "entry_date",
        "player_id",
        *WELLNESS_PARAMETER_COLUMNS,
        "wellness_physical",
        "wellness_mental",
        "wellness_avg",
        "readiness_score",
    ]
    rpe_cols = ["entry_date", "player_id", "avg_rpe"]

    if wellness_df.empty and rpe_df.empty:
        return _empty_monitoring_df()

    merged = pd.merge(
        wellness_df[wellness_cols] if not wellness_df.empty else pd.DataFrame(columns=wellness_cols),
        rpe_df[rpe_cols] if not rpe_df.empty else pd.DataFrame(columns=rpe_cols),
        on=["entry_date", "player_id"],
        how="outer",
    )

    merged["entry_date"] = _coerce_date(merged["entry_date"])
    merged["player_id"] = merged["player_id"].astype(str)

    lookup = {str(key): str(value) for key, value in (player_lookup or {}).items() if str(key).strip()}
    merged["player_name"] = merged["player_id"].map(lookup).fillna(merged["player_id"])

    for numeric_col in MONITORING_NUMERIC_COLUMNS:
        if numeric_col not in merged.columns:
            merged[numeric_col] = pd.NA
        merged[numeric_col] = pd.to_numeric(merged[numeric_col], errors="coerce")

    merged = merged.sort_values(["entry_date", "player_name"]).reset_index(drop=True)
    return merged[MONITORING_COLUMNS]


def summarize_monitoring_dataset(df: pd.DataFrame) -> dict[str, object]:
    empty_summary: dict[str, object] = {
        "wellness_players": 0,
        "rpe_players": 0,
        "wellness_entries": 0,
        "rpe_entries": 0,
        "wellness_physical": float("nan"),
        "wellness_mental": float("nan"),
        "wellness_avg": float("nan"),
        "readiness_avg": float("nan"),
        "avg_rpe": float("nan"),
    }
    for column in WELLNESS_PARAMETER_COLUMNS:
        empty_summary[column] = float("nan")

    if df is None or df.empty:
        return empty_summary

    wellness_mask = df["wellness_avg"].notna()
    rpe_mask = df["avg_rpe"].notna()
    avg_rpe = float(pd.to_numeric(df.loc[rpe_mask, "avg_rpe"], errors="coerce").mean()) if rpe_mask.any() else float("nan")

    summary = {
        "wellness_players": int(df.loc[wellness_mask, "player_id"].nunique()),
        "rpe_players": int(df.loc[rpe_mask, "player_id"].nunique()),
        "wellness_entries": int(wellness_mask.sum()),
        "rpe_entries": int(rpe_mask.sum()),
        "wellness_physical": float(pd.to_numeric(df["wellness_physical"], errors="coerce").mean()) if df["wellness_physical"].notna().any() else float("nan"),
        "wellness_mental": float(pd.to_numeric(df["wellness_mental"], errors="coerce").mean()) if df["wellness_mental"].notna().any() else float("nan"),
        "wellness_avg": float(pd.to_numeric(df["wellness_avg"], errors="coerce").mean()) if df["wellness_avg"].notna().any() else float("nan"),
        "readiness_avg": float(pd.to_numeric(df["readiness_score"], errors="coerce").mean()) if df["readiness_score"].notna().any() else float("nan"),
        "avg_rpe": avg_rpe,
    }
    for column in WELLNESS_PARAMETER_COLUMNS:
        summary[column] = float(pd.to_numeric(df[column], errors="coerce").mean()) if df[column].notna().any() else float("nan")
    return summary


def build_monitoring_grouped_summary(df: pd.DataFrame, period: str = "day") -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_grouped_summary()

    tmp = df.copy()
    ts = pd.to_datetime(tmp["entry_date"], errors="coerce")

    if period == "week":
        tmp["bucket"] = (ts - pd.to_timedelta(ts.dt.weekday, unit="D")).dt.normalize()
        tmp["label"] = tmp["bucket"].apply(
            lambda value: f"W{int(pd.Timestamp(value).isocalendar().week):02d} | {pd.Timestamp(value):%d/%m}"
        )
    elif period == "month":
        tmp["bucket"] = ts.dt.to_period("M").dt.to_timestamp()
        tmp["label"] = tmp["bucket"].dt.strftime("%Y-%m")
    else:
        tmp["bucket"] = ts.dt.normalize()
        tmp["label"] = tmp["bucket"].dt.strftime("%d/%m")

    tmp["wellness_present"] = tmp["wellness_avg"].notna()
    tmp["rpe_present"] = tmp["avg_rpe"].notna()

    aggregation: dict[str, tuple[str, str]] = {
        "wellness_avg": ("wellness_avg", "mean"),
        "readiness_score": ("readiness_score", "mean"),
        "avg_rpe": ("avg_rpe", "mean"),
        "avg_rpe_std": ("avg_rpe", "std"),
    }
    for column in WELLNESS_PARAMETER_COLUMNS:
        aggregation[column] = (column, "mean")
        aggregation[f"{column}_std"] = (column, "std")

    grouped = (
        tmp.groupby(["bucket", "label"], as_index=False)
        .agg(**aggregation)
        .sort_values("bucket")
        .reset_index(drop=True)
    )

    wellness_players = (
        tmp.loc[tmp["wellness_present"]]
        .groupby("bucket")["player_id"]
        .nunique()
        .rename("wellness_players")
        .reset_index()
    )
    rpe_players = (
        tmp.loc[tmp["rpe_present"]]
        .groupby("bucket")["player_id"]
        .nunique()
        .rename("rpe_players")
        .reset_index()
    )

    grouped = grouped.merge(wellness_players, on="bucket", how="left").merge(rpe_players, on="bucket", how="left")
    grouped["wellness_players"] = grouped["wellness_players"].fillna(0).astype(int)
    grouped["rpe_players"] = grouped["rpe_players"].fillna(0).astype(int)
    return grouped


def build_monitoring_player_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_player_summary()

    tmp = df.copy()
    tmp["wellness_present"] = tmp["wellness_avg"].notna()
    tmp["rpe_present"] = tmp["avg_rpe"].notna()

    aggregation: dict[str, tuple[str, str]] = {
        "wellness_days": ("wellness_present", "sum"),
        "rpe_days": ("rpe_present", "sum"),
        "wellness_avg": ("wellness_avg", "mean"),
        "readiness_score": ("readiness_score", "mean"),
        "avg_rpe": ("avg_rpe", "mean"),
    }
    for column in WELLNESS_PARAMETER_COLUMNS:
        aggregation[column] = (column, "mean")

    grouped = (
        tmp.groupby(["player_id", "player_name"], as_index=False)
        .agg(**aggregation)
        .reset_index(drop=True)
    )
    grouped = grouped.sort_values(["rpe_days", "wellness_days", "avg_rpe", "player_name"], ascending=[False, False, False, True]).reset_index(drop=True)
    return grouped
