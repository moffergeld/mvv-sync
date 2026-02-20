# Subscripts/player_tab_data.py
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


CHART_H = 340

GPS_TABLE = "v_gps_summary"
GPS_METRICS = [
    ("Total Distance (m)", "total_distance"),
    ("14.4–19.7 km/h", "running"),
    ("19.8–25.1 km/h", "sprint"),
    (">25,1 km/h", "high_sprint"),
    ("Max Speed (km/u)", "max_speed"),
]
GPS_TABLE_COLS_RAW = ["datum", "type", "total_distance", "running", "sprint", "high_sprint", "max_speed"]
GPS_RENAME = {
    "datum": "Date",
    "type": "Type",
    "total_distance": "Total Distance (m)",
    "running": "14.4–19.7 km/h",
    "sprint": "19.8–25.1 km/h",
    "high_sprint": ">25,1 km/h",
    "max_speed": "Max Speed (km/u)",
}

ASRM_COLS = [
    ("Muscle soreness", "muscle_soreness"),
    ("Fatigue", "fatigue"),
    ("Sleep quality", "sleep_quality"),
    ("Stress", "stress"),
    ("Mood", "mood"),
]


def _df_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _coerce_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _strip_titles(fig: go.Figure):
    fig.update_layout(title_text="", xaxis_title=None)
    return fig


def _add_zone_background(fig: go.Figure, y_min: float = 0, y_max: float = 10):
    zones = [
        (0, 4.5, "rgba(0, 200, 0, 0.12)"),
        (4.5, 7.5, "rgba(255, 165, 0, 0.14)"),
        (7.5, 10, "rgba(255, 0, 0, 0.14)"),
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
# GPS
# -----------------------------
def fetch_gps_summary_recent_raw(sb, player_id: str, limit: int = 400) -> pd.DataFrame:
    try:
        rows = (
            sb.table(GPS_TABLE)
            .select(",".join(set(GPS_TABLE_COLS_RAW + ["player_id"])))
            .eq("player_id", player_id)
            .order("datum", desc=True)
            .limit(limit)
            .execute()
            .data
            or []
        )
        df = _df_from_rows(rows)
        if df.empty:
            return df
        df["datum"] = _coerce_date_series(df["datum"])
        return df
    except Exception:
        return pd.DataFrame()


def gps_table_pretty(df_raw: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if df_raw.empty:
        return df_raw
    show = df_raw.sort_values("datum", ascending=False).head(n).copy()
    cols = [c for c in GPS_TABLE_COLS_RAW if c in show.columns]
    return show[cols].rename(columns=GPS_RENAME)


def gps_timeseries_summed_for_plot(df_raw: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    if df_raw.empty:
        return df_raw
    df = df_raw.copy()
    df["datum"] = _coerce_date_series(df["datum"])
    df[metric_key] = pd.to_numeric(df[metric_key], errors="coerce").fillna(0.0)
    out = df.groupby("datum", as_index=False)[metric_key].sum().sort_values("datum").reset_index(drop=True)
    return out.tail(10)


def plot_gps_over_time(df_summed: pd.DataFrame, metric_label: str, metric_key: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_summed["datum"],
            y=pd.to_numeric(df_summed[metric_key], errors="coerce"),
            mode="lines+markers",
            line=dict(color="#FF0033", width=3, shape="spline", smoothing=1.2),
            marker=dict(size=6),
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=CHART_H, showlegend=False)
    fig.update_xaxes(type="date", tickformat="%d-%m-%Y", title_text=None)
    fig.update_yaxes(title_text=metric_label)
    _strip_titles(fig)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Wellness (ASRM)
# -----------------------------
def load_asrm(sb, player_id: str, entry_date: date) -> Optional[Dict[str, Any]]:
    try:
        resp = (
            sb.table("asrm_entries")
            .select("*")
            .eq("player_id", player_id)
            .eq("entry_date", entry_date.isoformat())
            .maybe_single()
            .execute()
        )
        return resp.data
    except Exception:
        return None


def fetch_asrm_over_time(sb, player_id: str, limit: int = 180) -> pd.DataFrame:
    try:
        rows = (
            sb.table("asrm_entries")
            .select("entry_date,muscle_soreness,fatigue,sleep_quality,stress,mood")
            .eq("player_id", player_id)
            .order("entry_date", desc=True)
            .limit(limit)
            .execute()
            .data
            or []
        )
        df = _df_from_rows(rows)
        if df.empty:
            return df
        df["entry_date"] = _coerce_date_series(df["entry_date"])
        return df.sort_values("entry_date")
    except Exception:
        return pd.DataFrame()


def plot_asrm_over_time(df: pd.DataFrame, param_key: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["entry_date"],
            y=pd.to_numeric(df[param_key], errors="coerce"),
            mode="lines+markers",
            line=dict(color="#FF0033", width=3, shape="spline", smoothing=1.2),
            marker=dict(size=6),
        )
    )
    _add_zone_background(fig, 0, 10)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=CHART_H, showlegend=False)
    fig.update_xaxes(type="date", tickformat="%d-%m-%Y", title_text=None)
    fig.update_yaxes(title_text="Score (0–10)")
    _strip_titles(fig)
    st.plotly_chart(fig, use_container_width=True)


def plot_asrm_session(row: Dict[str, Any]):
    labels = [x[0] for x in ASRM_COLS]
    keys = [x[1] for x in ASRM_COLS]
    values = [int(row.get(k, 0) or 0) for k in keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values, marker=dict(color="rgba(135,206,250,1)")))
    _add_zone_background(fig, 0, 10)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=CHART_H, showlegend=False)
    fig.update_xaxes(title_text=None)
    fig.update_yaxes(title_text="Score (0–10)")
    _strip_titles(fig)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# RPE (plots)
# -----------------------------
def fetch_rpe_for_date(sb, player_id: str, d: date) -> pd.DataFrame:
    try:
        header = (
            sb.table("rpe_entries")
            .select("id")
            .eq("player_id", player_id)
            .eq("entry_date", d.isoformat())
            .maybe_single()
            .execute()
            .data
        )
    except Exception:
        header = None

    if not header or not header.get("id"):
        return pd.DataFrame()

    try:
        rows = (
            sb.table("rpe_sessions")
            .select("session_index,rpe")
            .eq("rpe_entry_id", header["id"])
            .order("session_index")
            .execute()
            .data
            or []
        )
        df = _df_from_rows(rows)
        if df.empty:
            return df
        df["session_index"] = pd.to_numeric(df["session_index"], errors="coerce").fillna(0).astype(int)
        df["rpe"] = pd.to_numeric(df["rpe"], errors="coerce").fillna(0)
        return df
    except Exception:
        return pd.DataFrame()


def plot_rpe_session(sessions_df: pd.DataFrame):
    if sessions_df is None or sessions_df.empty:
        st.info("Geen RPE sessions gevonden voor deze datum.")
        return

    df = sessions_df.copy()
    df["session_index"] = pd.to_numeric(df["session_index"], errors="coerce").fillna(0).astype(int)
    df["rpe"] = pd.to_numeric(df["rpe"], errors="coerce").fillna(0)
    df = df[df["session_index"].isin([1, 2])].sort_values("session_index")
    if df.empty:
        st.info("Geen RPE sessions (1/2) gevonden voor deze datum.")
        return

    x = df["session_index"].astype(str).tolist()
    y = df["rpe"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y, marker=dict(color="#FF0033"), opacity=0.9))
    _add_zone_background(fig, 0, 10)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=CHART_H, showlegend=False)
    fig.update_xaxes(type="category", tickmode="array", tickvals=["1", "2"], ticktext=["1", "2"], title_text=None)
    fig.update_yaxes(title_text="RPE (0–10)", tick0=0, dtick=1)
    _strip_titles(fig)
    st.plotly_chart(fig, use_container_width=True)


def render_data_tab(sb, target_player_id: str):
    left, right = st.columns(2)

    with left:
        st.subheader("GPS – Recent (laatste 20)")
        gps_raw = fetch_gps_summary_recent_raw(sb, target_player_id, limit=400)
        if gps_raw.empty:
            st.info("Geen GPS summary data gevonden (v_gps_summary).")
        else:
            st.dataframe(gps_table_pretty(gps_raw, n=20), use_container_width=True, hide_index=True)

    with right:
        st.subheader("GPS – Over time")
        gps_raw = fetch_gps_summary_recent_raw(sb, target_player_id, limit=400)
        if gps_raw.empty:
            st.info("Geen GPS summary data gevonden (v_gps_summary).")
        else:
            metric_label = st.selectbox("Parameter", options=[m[0] for m in GPS_METRICS], index=0, key="gps_metric_sel")
            metric_key = dict(GPS_METRICS)[metric_label]
            summed = gps_timeseries_summed_for_plot(gps_raw, metric_key=metric_key)
            if summed.empty:
                st.info("Onvoldoende data voor grafiek.")
            else:
                plot_gps_over_time(summed, metric_label, metric_key)

    st.divider()
    well_col, rpe_col = st.columns(2)

    with well_col:
        st.subheader("Wellness")
        mode_w = st.radio("Weergave", ["Session", "Over time"], horizontal=True, key="well_mode")
        if mode_w == "Session":
            d = st.date_input("Datum (Wellness)", value=date.today(), key="well_date")
            row = load_asrm(sb, target_player_id, d)
            if not row:
                st.info("Geen Wellness entry voor deze datum.")
            else:
                plot_asrm_session(row)
        else:
            df = fetch_asrm_over_time(sb, target_player_id, limit=180)
            if df.empty:
                st.info("Geen Wellness entries gevonden.")
            else:
                param_label = st.selectbox("Parameter", options=[x[0] for x in ASRM_COLS], index=0, key="well_param")
                param_key = dict(ASRM_COLS)[param_label]
                plot_asrm_over_time(df, param_key)

    with rpe_col:
        st.subheader("RPE")
        mode_r = st.radio("Weergave", ["Session", "Over time"], horizontal=True, key="rpe_mode")
        if mode_r == "Session":
            d = st.date_input("Datum (RPE)", value=date.today(), key="rpe_date")
            sess = fetch_rpe_for_date(sb, target_player_id, d)
            if sess.empty:
                st.info("Geen RPE entry voor deze datum.")
            else:
                plot_rpe_session(sess)
        else:
            st.info("RPE 'Over time' staat nu in Forms-tab (dag-status + invullen).")

