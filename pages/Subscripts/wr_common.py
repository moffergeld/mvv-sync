from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Config / constants
# -----------------------------
CHART_H = 420

ASRM_PARAMS = [
    ("Muscle soreness", "muscle_soreness"),
    ("Fatigue", "fatigue"),
    ("Sleep quality", "sleep_quality"),
    ("Stress", "stress"),
    ("Mood", "mood"),
]

RPE_PARAMS = [
    ("RPE (weighted avg)", "avg_rpe"),          # 0–10
    ("RPE Load (sum dur*rpe)", "rpe_load"),     # arbitrary
    ("Total duration (min)", "duration_min"),   # minutes
]

# Zones (1 = best, 10 = worst) voor 0–10 schalen
ZONE_GREEN_MAX = 4.5
ZONE_ORANGE_MAX = 7.5

# “Rood” detectie voor notice
ASRM_RED_THRESHOLD = 7.5
RPE_RED_THRESHOLD = 7.5  # voor avg_rpe


# -----------------------------
# Small helpers
# -----------------------------
def _df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _coerce_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def iso_week_start_end(year: int, week: int) -> Tuple[date, date]:
    # ISO week start = maandag
    d0 = date.fromisocalendar(year, week, 1)
    d1 = d0 + timedelta(days=6)
    return d0, d1


def add_zone_background(fig: go.Figure, y_min: float = 0, y_max: float = 10) -> None:
    zones = [
        (0, ZONE_GREEN_MAX, "rgba(0, 200, 0, 0.12)"),
        (ZONE_GREEN_MAX, ZONE_ORANGE_MAX, "rgba(255, 165, 0, 0.14)"),
        (ZONE_ORANGE_MAX, 10, "rgba(255, 0, 0, 0.14)"),
    ]
    for y0, y1, color in zones:
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="y",
            x0=0,
            x1=1,
            y0=y0,
            y1=y1,
            fillcolor=color,
            line=dict(width=0),
            layer="below",
        )
    fig.update_yaxes(range=[y_min, y_max], tick0=0, dtick=1)


# -----------------------------
# Caching layer (common calls)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60)
def fetch_active_players_cached(sb_url_key: str, _sb, ttl_salt: str = "") -> pd.DataFrame:
    # sb_url_key + ttl_salt om cache te scheiden per omgeving
    rows = (
        _sb.table("players")
        .select("player_id,full_name,is_active")
        .eq("is_active", True)
        .order("full_name")
        .execute()
        .data
        or []
    )
    df = _df(rows)
    if not df.empty:
        df["player_id"] = df["player_id"].astype(str)
        df["full_name"] = df["full_name"].astype(str)
    return df


@st.cache_data(show_spinner=False, ttl=60)
def fetch_asrm_range_cached(sb_url_key: str, _sb, d0_iso: str, d1_iso: str) -> pd.DataFrame:
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
    return df


@st.cache_data(show_spinner=False, ttl=60)
def fetch_asrm_date_cached(sb_url_key: str, _sb, d_iso: str) -> pd.DataFrame:
    rows = (
        _sb.table("asrm_entries")
        .select("player_id,entry_date,muscle_soreness,fatigue,sleep_quality,stress,mood")
        .eq("entry_date", d_iso)
        .execute()
        .data
        or []
    )
    df = _df(rows)
    if df.empty:
        return df
    df["player_id"] = df["player_id"].astype(str)
    df["entry_date"] = _coerce_date(df["entry_date"])
    return df


@st.cache_data(show_spinner=False, ttl=60)
def fetch_rpe_headers_range_cached(sb_url_key: str, _sb, d0_iso: str, d1_iso: str) -> pd.DataFrame:
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


@st.cache_data(show_spinner=False, ttl=60)
def fetch_rpe_headers_date_cached(sb_url_key: str, _sb, d_iso: str) -> pd.DataFrame:
    rows = (
        _sb.table("rpe_entries")
        .select("id,player_id,entry_date")
        .eq("entry_date", d_iso)
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


@st.cache_data(show_spinner=False, ttl=60)
def fetch_rpe_sessions_for_ids_cached(sb_url_key: str, _sb, entry_ids_tuple: Tuple[str, ...]) -> pd.DataFrame:
    entry_ids = list(entry_ids_tuple)
    if not entry_ids:
        return pd.DataFrame()

    rows = (
        _sb.table("rpe_sessions")
        .select("rpe_entry_id,session_index,duration_min,rpe")
        .in_("rpe_entry_id", entry_ids)
        .execute()
        .data
        or []
    )
    df = _df(rows)
    if df.empty:
        return df
    df["rpe_entry_id"] = df["rpe_entry_id"].astype(str)
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce").fillna(0.0)
    df["rpe"] = pd.to_numeric(df["rpe"], errors="coerce").fillna(0.0)
    df["load"] = df["duration_min"] * df["rpe"]
    return df


# -----------------------------
# RPE computations
# -----------------------------
def build_rpe_player_daily(sb_url_key: str, sb, headers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Output columns:
    - entry_date
    - player_id
    - avg_rpe (weighted by duration; fallback mean if duration==0)
    - rpe_load (sum duration*rpe)
    - duration_min (sum)
    """
    if headers_df is None or headers_df.empty:
        return pd.DataFrame(columns=["entry_date", "player_id", "avg_rpe", "rpe_load", "duration_min"])

    entry_ids = headers_df["id"].astype(str).tolist()
    sess = fetch_rpe_sessions_for_ids_cached(sb_url_key, sb, tuple(entry_ids))
    if sess.empty:
        return pd.DataFrame(columns=["entry_date", "player_id", "avg_rpe", "rpe_load", "duration_min"])

    merged = sess.merge(
        headers_df[["id", "player_id", "entry_date"]],
        left_on="rpe_entry_id",
        right_on="id",
        how="left",
    ).dropna(subset=["player_id", "entry_date"])

    def _weighted_avg(g: pd.DataFrame) -> float:
        dur = float(g["duration_min"].sum())
        if dur <= 0:
            return float(g["rpe"].mean()) if len(g) else float("nan")
        return float(g["load"].sum() / dur)

    out = (
        merged.groupby(["entry_date", "player_id"], as_index=False)
        .apply(lambda g: pd.Series({
            "avg_rpe": _weighted_avg(g),
            "rpe_load": float(g["load"].sum()),
            "duration_min": float(g["duration_min"].sum()),
        }))
        .reset_index(drop=True)
    )
    out["entry_date"] = _coerce_date(out["entry_date"])
    out["player_id"] = out["player_id"].astype(str)
    return out


# -----------------------------
# Aggregations for WEEK (per player mean + std)
# -----------------------------
def agg_week_player_mean_std(df_long: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Input rows: per player per day.
    Output: per player mean + std + n
    """
    if df_long is None or df_long.empty:
        return pd.DataFrame(columns=["player_id", "mean", "std", "n"])

    tmp = df_long.copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=["player_id", value_col])

    out = (
        tmp.groupby("player_id", as_index=False)
        .agg(mean=(value_col, "mean"), std=(value_col, "std"), n=(value_col, "count"))
    )
    out["std"] = out["std"].fillna(0.0)
    out["player_id"] = out["player_id"].astype(str)
    return out


# -----------------------------
# Plotting
# -----------------------------
def plot_day_bars(df: pd.DataFrame, x_col: str, y_col: str, y_title: str, zone_0_10: bool) -> None:
    if df.empty:
        st.info("Geen data voor deze selectie.")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[x_col].astype(str),
        y=pd.to_numeric(df[y_col], errors="coerce"),
        opacity=0.92,
    ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=CHART_H,
        showlegend=False,
    )
    fig.update_xaxes(type="category", tickangle=90)
    fig.update_yaxes(title_text=y_title)

    if zone_0_10:
        add_zone_background(fig, 0, 10)

    st.plotly_chart(fig, use_container_width=True)


def plot_week_player_mean_std_bars(
    df_stats: pd.DataFrame,
    player_name_col: str,
    mean_col: str = "mean",
    std_col: str = "std",
    y_title: str = "",
    zone_0_10: bool = False,
) -> None:
    if df_stats.empty:
        st.info("Geen data voor deze week/selectie.")
        return

    x = df_stats[player_name_col].astype(str).tolist()
    y = pd.to_numeric(df_stats[mean_col], errors="coerce")
    err = pd.to_numeric(df_stats[std_col], errors="coerce").fillna(0.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x,
        y=y,
        opacity=0.92,
        error_y=dict(type="data", array=err, thickness=1.5, width=6),
    ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=CHART_H,
        showlegend=False,
    )
    fig.update_xaxes(type="category", tickangle=90)
    fig.update_yaxes(title_text=y_title)

    if zone_0_10:
        add_zone_background(fig, 0, 10)

    st.plotly_chart(fig, use_container_width=True)
