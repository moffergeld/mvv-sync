# pages/Subscripts/wr_common.py
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
    ("RPE (weighted avg)", "avg_rpe"),
    ("RPE Load (sum dur*rpe)", "rpe_load"),
    ("Total duration (min)", "duration_min"),
]

ZONE_GREEN_MAX = 4.5
ZONE_ORANGE_MAX = 7.5

ASRM_RED_THRESHOLD = 7.5
RPE_RED_THRESHOLD = 7.5

# MVV Design System Colors
MVV_COLORS = {
    'primary': '#C8102E',
    'light': '#E8213F',
    'dark': '#8B0A1F',
    'background': '#0D0E13',
    'card': 'rgba(255, 255, 255, 0.04)',
    'text_primary': '#F0F0F0',
    'text_muted': 'rgba(240, 240, 240, 0.45)',
    'grid': 'rgba(255, 255, 255, 0.09)',
    'zone_green': 'rgba(0, 200, 0, 0.12)',
    'zone_orange': 'rgba(255, 165, 0, 0.14)',
    'zone_red': 'rgba(255, 0, 0, 0.14)'
}


def _df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _coerce_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def iso_week_start_end(year: int, week: int) -> Tuple[date, date]:
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
# MVV Styled Charts
# -----------------------------
def create_mvv_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str = "", 
                        show_zones: bool = False, y_range: tuple = (0, 10)) -> go.Figure:
    """Create a styled bar chart according to MVV design system"""
    
    fig = go.Figure()
    
    # Add bars with error bars if available
    if 'std' in df.columns:
        fig.add_trace(go.Bar(
            x=df[x_col],
            y=df[y_col],
            error_y=dict(
                type='data',
                array=df['std'] if 'std' in df.columns else None,
                color=MVV_COLORS['light'],
                thickness=2,
                width=6
            ),
            marker=dict(
                color=MVV_COLORS['primary'],
                line=dict(color=MVV_COLORS['light'], width=2)
            ),
            opacity=0.9
        ))
    else:
        fig.add_trace(go.Bar(
            x=df[x_col],
            y=df[y_col],
            marker=dict(
                color=MVV_COLORS['primary'],
                line=dict(color=MVV_COLORS['light'], width=2)
            ),
            opacity=0.9
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="DM Sans", size=16, color=MVV_COLORS['text_primary']),
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=MVV_COLORS['background'],
        font=dict(family="DM Sans", size=12, color=MVV_COLORS['text_primary']),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color=MVV_COLORS['text_primary']),
            title_font=dict(color=MVV_COLORS['text_primary'])
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=MVV_COLORS['grid'],
            tickfont=dict(color=MVV_COLORS['text_primary']),
            title_font=dict(color=MVV_COLORS['text_primary']),
            range=y_range
        ),
        bargap=0.3,
        margin=dict(l=50, r=30, t=50, b=50),
        showlegend=False
    )
    
    # Add zones if requested
    if show_zones:
        zones = [
            (0, ZONE_GREEN_MAX, MVV_COLORS['zone_green']),
            (ZONE_GREEN_MAX, ZONE_ORANGE_MAX, MVV_COLORS['zone_orange']),
            (ZONE_ORANGE_MAX, y_range[1], MVV_COLORS['zone_red']),
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
    
    return fig


# -----------------------------
# Caching layer (common calls)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60)
def fetch_active_players_cached(sb_url_key: str, _sb, ttl_salt: str = "") -> pd.DataFrame:
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
# NEW: Injury fetch (from rpe_entries)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60)
def fetch_rpe_injuries_range_cached(sb_url_key: str, _sb, d0_iso: str, d1_iso: str) -> pd.DataFrame:
    """
    Pull injuries directly from rpe_entries (no sessions needed).
    Uses your index: (player_id, entry_date desc).
    """
    rows = (
        _sb.table("rpe_entries")
        .select("id,player_id,entry_date,injury,injury_type,injury_pain,attachment_url,notes,created_at,updated_at")
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
    df["injury"] = df["injury"].astype(bool)
    df["injury_pain"] = pd.to_numeric(df["injury_pain"], errors="coerce")
    return df


# -----------------------------
# RPE computations
# -----------------------------
def build_rpe_player_daily(sb_url_key: str, sb, headers_df: pd.DataFrame) -> pd.DataFrame:
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


def agg_week_player_mean_std(df_long: pd.DataFrame, value_col: str) -> pd.DataFrame:
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
# Plotting (deprecation fix: width="stretch") - UPDATED WITH MVV STYLING
# -----------------------------
def plot_day_bars(df: pd.DataFrame, x_col: str, y_col: str, y_title: str, zone_0_10: bool) -> None:
    if df.empty:
        st.info("Geen data voor deze selectie.")
        return

    # Sort by value descending for better visualization
    df = df.sort_values(y_col, ascending=False)
    
    # Use the new MVV styled chart
    fig = create_mvv_bar_chart(
        df=df,
        x_col=x_col,
        y_col=y_col,
        title=y_title,
        show_zones=zone_0_10,
        y_range=(0, 10) if zone_0_10 else None
    )
    
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

    # Sort by mean value descending for better visualization
    df_stats = df_stats.sort_values(mean_col, ascending=False)
    
    # Use the new MVV styled chart
    fig = create_mvv_bar_chart(
        df=df_stats,
        x_col=player_name_col,
        y_col=mean_col,
        title=y_title,
        show_zones=zone_0_10,
        y_range=(0, 10) if zone_0_10 else None
    )
    
    # Add error bars if std column exists
    if std_col in df_stats.columns and not df_stats[std_col].isna().all():
        fig.data[0].error_y = dict(
            type='data',
            array=df_stats[std_col],
            color=MVV_COLORS['light'],
            thickness=2,
            width=6
        )
    
    st.plotly_chart(fig, use_container_width=True)
