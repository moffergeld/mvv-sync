# pages/Subscripts/player_tab_data.py
# ============================================================
# Player Page - Data Tab
#
# Doel
# - Snelle "Data" tab met:
#   - GPS (laatste 14 dagen): tabel + over time grafiek
#   - Wellness:
#       * Session view (1 datum)
#       * Over time (laatste 14 dagen)
#   - RPE:
#       * Session view (1 datum)
#       * Over time (laatste 7 dagen)
#
# Performance strategie
# 1) Server-side filtering:
#    - .eq("player_id", ...) + datum range (gte/lte) in Supabase query
#    -> laadt NIET de hele tabel, alleen relevante rijen.
# 2) Caching:
#    - @st.cache_data(ttl=120) per speler/datum
#    -> snelle player-switch (staff) en minder herhaalde calls.
# 3) Static Plotly charts:
#    - config={"staticPlot": True}
#    -> minder "gevoelig" op telefoon (geen zoom/pan per ongeluk).
#
# Aggregatie GPS per dag
# - default: som per dag (total_distance/running/sprint/high_sprint)
# - max_speed: MAX per dag (niet som!)
# ============================================================

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

CHART_H = 340

GPS_TABLE = "v_gps_summary"

GPS_METRICS: List[Tuple[str, str]] = [
    ("Total Distance (m)", "total_distance"),
    ("14.4–19.7 km/h", "running"),
    ("19.8–25.1 km/h", "sprint"),
    (">25,1 km/h", "high_sprint"),
    ("Max Speed (km/u)", "max_speed"),
]

GPS_TABLE_COLS_RAW = [
    "datum",
    "type",
    "total_distance",
    "running",
    "sprint",
    "high_sprint",
    "max_speed",
]

GPS_RENAME = {
    "datum": "Date",
    "type": "Type",
    "total_distance": "Total Distance (m)",
    "running": "14.4–19.7 km/h",
    "sprint": "19.8–25.1 km/h",
    "high_sprint": ">25,1 km/h",
    "max_speed": "Max Speed (km/u)",
}

ASRM_COLS: List[Tuple[str, str]] = [
    ("Muscle soreness", "muscle_soreness"),
    ("Fatigue", "fatigue"),
    ("Sleep quality", "sleep_quality"),
    ("Stress", "stress"),
    ("Mood", "mood"),
]


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _to_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _plotly_config_static() -> Dict[str, Any]:
    """
    Static plot:
    - Geen zoom/pan/hover gedrag op mobiel
    - Sneller en minder "gevoelig"
    """
    return {
        "staticPlot": True,
        "displayModeBar": False,
        "scrollZoom": False,
        "doubleClick": False,
        "responsive": True,
    }


def _strip_titles(fig: go.Figure) -> None:
    fig.update_layout(
        title_text="",
        xaxis_title=None,
        margin=dict(l=10, r=10, t=10, b=10),
        height=CHART_H,
        showlegend=False,
    )


def _add_zone_background(fig: go.Figure, y_min: float = 0, y_max: float = 10) -> None:
    """Kleurzones (groen/oranje/rood) voor wellness/RPE (0–10)."""
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


# ============================================================
# GPS (Last 14 days) - CACHED TTL 120s, SERVER-SIDE FILTER
# ============================================================

@st.cache_data(show_spinner=False, ttl=120)
def fetch_gps_14d_cached(player_id: str, start_iso: str, end_iso: str, limit: int = 1000) -> pd.DataFrame:
    """
    Haalt GPS summary rijen op voor 1 speler binnen datumbereik.
    Server-side filter: eq(player_id) + gte/lte(datum).
    """
    sb = st.session_state.get("_sb_for_cache")
    if sb is None:
        return pd.DataFrame()

    try:
        rows = (
            sb.table(GPS_TABLE)
            .select(",".join(GPS_TABLE_COLS_RAW + ["player_id"]))
            .eq("player_id", player_id)
            .gte("datum", start_iso)
            .lte("datum", end_iso)
            .order("datum", desc=True)
            .limit(limit)
            .execute()
            .data
            or []
        )
        df = _df(rows)
        if df.empty:
            return df
        df["datum"] = _to_date_series(df["datum"])
        return df
    except Exception:
        return pd.DataFrame()


def fetch_gps_14d(sb, player_id: str, days: int = 14) -> pd.DataFrame:
    """
    Wrapper die:
    - sb in session_state zet voor cached functies
    - standaard laatste 14 dagen pakt
    """
    st.session_state["_sb_for_cache"] = sb
    end_d = date.today()
    start_d = end_d - timedelta(days=days - 1)
    return fetch_gps_14d_cached(str(player_id), start_d.isoformat(), end_d.isoformat())


def gps_table_pretty(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Tabel view (laatste 14 dagen, desc op datum)."""
    if df_raw.empty:
        return df_raw
    show = df_raw.sort_values("datum", ascending=False).copy()
    cols = [c for c in GPS_TABLE_COLS_RAW if c in show.columns]
    return show[cols].rename(columns=GPS_RENAME)


def gps_daily_aggregate(df_raw: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    """
    Aggregatie per dag:
    - alle metrics: som per dag
    - max_speed: MAX per dag (belangrijk!)
    """
    if df_raw.empty:
        return df_raw

    df = df_raw.copy()
    df["datum"] = _to_date_series(df["datum"])
    df[metric_key] = pd.to_numeric(df[metric_key], errors="coerce")

    if metric_key == "max_speed":
        out = df.groupby("datum", as_index=False)[metric_key].max()
    else:
        df[metric_key] = df[metric_key].fillna(0.0)
        out = df.groupby("datum", as_index=False)[metric_key].sum()

    return out.sort_values("datum").reset_index(drop=True)


def plot_gps_over_time(df_daily: pd.DataFrame, metric_label: str, metric_key: str) -> None:
    """GPS line chart (static)."""
    if df_daily.empty:
        st.info("Onvoldoende data voor grafiek.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_daily["datum"],
            y=pd.to_numeric(df_daily[metric_key], errors="coerce"),
            mode="lines+markers",
            line=dict(color="#FF0033", width=3, shape="spline", smoothing=1.2),
            marker=dict(size=6),
        )
    )
    _strip_titles(fig)
    fig.update_xaxes(type="date", tickformat="%d-%m", title_text=None)
    fig.update_yaxes(title_text=metric_label)
    st.plotly_chart(fig, use_container_width=True, config=_plotly_config_static())


# ============================================================
# WELLNESS (Last 14 days) - CACHED TTL 120s
# ============================================================

@st.cache_data(show_spinner=False, ttl=120)
def fetch_asrm_14d_cached(player_id: str, start_iso: str, end_iso: str, limit: int = 200) -> pd.DataFrame:
    """Haalt wellness entries voor 14 dagen (server-side date range)."""
    sb = st.session_state.get("_sb_for_cache")
    if sb is None:
        return pd.DataFrame()

    try:
        rows = (
            sb.table("asrm_entries")
            .select("entry_date,muscle_soreness,fatigue,sleep_quality,stress,mood,player_id")
            .eq("player_id", player_id)
            .gte("entry_date", start_iso)
            .lte("entry_date", end_iso)
            .order("entry_date", desc=True)
            .limit(limit)
            .execute()
            .data
            or []
        )
        df = _df(rows)
        if df.empty:
            return df
        df["entry_date"] = _to_date_series(df["entry_date"])
        return df.sort_values("entry_date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def fetch_asrm_14d(sb, player_id: str, days: int = 14) -> pd.DataFrame:
    st.session_state["_sb_for_cache"] = sb
    end_d = date.today()
    start_d = end_d - timedelta(days=days - 1)
    return fetch_asrm_14d_cached(str(player_id), start_d.isoformat(), end_d.isoformat())


@st.cache_data(show_spinner=False, ttl=120)
def load_asrm_cached(player_id: str, entry_date_iso: str) -> Optional[Dict[str, Any]]:
    """1 wellness entry op 1 dag (cached)."""
    sb = st.session_state.get("_sb_for_cache")
    if sb is None:
        return None
    try:
        resp = (
            sb.table("asrm_entries")
            .select("*")
            .eq("player_id", player_id)
            .eq("entry_date", entry_date_iso)
            .maybe_single()
            .execute()
        )
        return resp.data
    except Exception:
        return None


def load_asrm(sb, player_id: str, entry_date: date) -> Optional[Dict[str, Any]]:
    st.session_state["_sb_for_cache"] = sb
    return load_asrm_cached(str(player_id), entry_date.isoformat())


def plot_asrm_over_time(df: pd.DataFrame, param_key: str) -> None:
    """Wellness over time (static)."""
    if df.empty or param_key not in df.columns:
        st.info("Geen wellness data voor deze periode.")
        return

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
    _strip_titles(fig)
    fig.update_xaxes(type="date", tickformat="%d-%m", title_text=None)
    fig.update_yaxes(title_text="Score (0–10)")
    st.plotly_chart(fig, use_container_width=True, config=_plotly_config_static())


def plot_asrm_session(row: Dict[str, Any]) -> None:
    """Wellness per sessie (bar chart, static)."""
    labels = [x[0] for x in ASRM_COLS]
    keys = [x[1] for x in ASRM_COLS]
    values = [int(row.get(k, 0) or 0) for k in keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values, marker=dict(color="#87CEFA")))
    _add_zone_background(fig, 0, 10)
    _strip_titles(fig)
    fig.update_yaxes(title_text="Score (0–10)")
    st.plotly_chart(fig, use_container_width=True, config=_plotly_config_static())


# ============================================================
# RPE (Session + Over time last 7 days)
# ============================================================

@st.cache_data(show_spinner=False, ttl=120)
def fetch_rpe_for_date_cached(player_id: str, entry_date_iso: str) -> pd.DataFrame:
    """
    RPE session view:
    - pakt rpe_entries header (id) voor player+date
    - pakt rpe_sessions voor die header (session_index, rpe)
    """
    sb = st.session_state.get("_sb_for_cache")
    if sb is None:
        return pd.DataFrame()

    try:
        header = (
            sb.table("rpe_entries")
            .select("id")
            .eq("player_id", player_id)
            .eq("entry_date", entry_date_iso)
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
        df = _df(rows)
        if df.empty:
            return df
        df["session_index"] = pd.to_numeric(df["session_index"], errors="coerce").fillna(0).astype(int)
        df["rpe"] = pd.to_numeric(df["rpe"], errors="coerce").fillna(0)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_rpe_for_date(sb, player_id: str, d: date) -> pd.DataFrame:
    st.session_state["_sb_for_cache"] = sb
    return fetch_rpe_for_date_cached(str(player_id), d.isoformat())


@st.cache_data(show_spinner=False, ttl=120)
def fetch_rpe_over_time_7d_cached(player_id: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    """
    RPE over time (laatste 7 dagen):
    - haalt rpe_entries in dat range
    - haalt sessions per entry (max 7 entries)
    - geeft gemiddelde RPE per dag terug
    """
    sb = st.session_state.get("_sb_for_cache")
    if sb is None:
        return pd.DataFrame()

    try:
        entries = (
            sb.table("rpe_entries")
            .select("id,entry_date")
            .eq("player_id", player_id)
            .gte("entry_date", start_iso)
            .lte("entry_date", end_iso)
            .order("entry_date")
            .execute()
            .data
            or []
        )
    except Exception:
        entries = []

    if not entries:
        return pd.DataFrame()

    rows_all: List[Dict[str, Any]] = []
    for e in entries:
        eid = e.get("id")
        ed = e.get("entry_date")
        if not eid or not ed:
            continue
        try:
            sess = (
                sb.table("rpe_sessions")
                .select("rpe")
                .eq("rpe_entry_id", eid)
                .execute()
                .data
                or []
            )
            for s in sess:
                rows_all.append({"entry_date": ed, "rpe": s.get("rpe")})
        except Exception:
            continue

    df = _df(rows_all)
    if df.empty:
        return df

    df["entry_date"] = _to_date_series(df["entry_date"])
    df["rpe"] = pd.to_numeric(df["rpe"], errors="coerce")
    out = df.groupby("entry_date", as_index=False)["rpe"].mean().sort_values("entry_date").reset_index(drop=True)
    return out


def fetch_rpe_over_time_7d(sb, player_id: str) -> pd.DataFrame:
    st.session_state["_sb_for_cache"] = sb
    end_d = date.today()
    start_d = end_d - timedelta(days=6)
    return fetch_rpe_over_time_7d_cached(str(player_id), start_d.isoformat(), end_d.isoformat())


def plot_rpe_session(sessions_df: pd.DataFrame) -> None:
    """RPE session bar chart (static)."""
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
    _strip_titles(fig)
    fig.update_xaxes(type="category", tickmode="array", tickvals=["1", "2"], ticktext=["1", "2"], title_text=None)
    fig.update_yaxes(title_text="RPE (0–10)", tick0=0, dtick=1)
    st.plotly_chart(fig, use_container_width=True, config=_plotly_config_static())


def plot_rpe_over_time(df_7d: pd.DataFrame) -> None:
    """RPE over time line chart (static)."""
    if df_7d.empty:
        st.info("Geen RPE data in de laatste 7 dagen.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_7d["entry_date"],
            y=pd.to_numeric(df_7d["rpe"], errors="coerce"),
            mode="lines+markers",
            line=dict(color="#FF0033", width=3, shape="spline", smoothing=1.2),
            marker=dict(size=6),
        )
    )
    _add_zone_background(fig, 0, 10)
    _strip_titles(fig)
    fig.update_xaxes(type="date", tickformat="%d-%m", title_text=None)
    fig.update_yaxes(title_text="RPE (gemiddelde)")
    st.plotly_chart(fig, use_container_width=True, config=_plotly_config_static())


# ============================================================
# RENDER
# ============================================================

def render_data_tab(sb, target_player_id: str) -> None:
    """
    Main renderer voor Player Page -> Data tab.
    """
    st.session_state["_sb_for_cache"] = sb

    st.header("Data")
    st.caption(
        "GPS & Wellness: laatste 14 dagen | RPE over time: laatste 7 dagen | Grafieken: static (mobiel vriendelijk)"
    )

    # ---- GPS last 14d (server-side) ----
    gps_raw = fetch_gps_14d(sb, target_player_id, days=14)

    left, right = st.columns(2)

    with left:
        st.subheader("GPS – Laatste 14 dagen")
        st.caption("Tabel bevat sessions binnen de laatste 14 dagen (server-side gefilterd).")
        if gps_raw.empty:
            st.info("Geen GPS summary data gevonden (v_gps_summary).")
        else:
            st.dataframe(gps_table_pretty(gps_raw), use_container_width=True, hide_index=True)

    with right:
        st.subheader("GPS – Over time")
        st.caption("Per dag geaggregeerd: som (behalve Max Speed = max).")
        if gps_raw.empty:
            st.info("Geen GPS summary data gevonden (v_gps_summary).")
        else:
            metric_label = st.selectbox(
                "Parameter",
                options=[m[0] for m in GPS_METRICS],
                index=0,
                key="gps_metric_sel",
                help="Kies metric voor de grafiek. Aggregatie is per dag.",
            )
            metric_key = dict(GPS_METRICS)[metric_label]
            daily = gps_daily_aggregate(gps_raw, metric_key=metric_key)
            plot_gps_over_time(daily, metric_label, metric_key)

    st.divider()

    well_col, rpe_col = st.columns(2)

    # ---- Wellness ----
    with well_col:
        st.subheader("Wellness")
        mode_w = st.radio(
            "Weergave",
            ["Session", "Over time"],
            horizontal=True,
            key="well_mode",
            help="Session: 1 dag | Over time: laatste 14 dagen",
        )

        if mode_w == "Session":
            d = st.date_input("Datum (Wellness)", value=date.today(), key="well_date")
            row = load_asrm(sb, target_player_id, d)
            if not row:
                st.info("Geen Wellness entry voor deze datum.")
            else:
                plot_asrm_session(row)
        else:
            dfw = fetch_asrm_14d(sb, target_player_id, days=14)
            if dfw.empty:
                st.info("Geen Wellness entries in laatste 14 dagen.")
            else:
                param_label = st.selectbox(
                    "Parameter",
                    options=[x[0] for x in ASRM_COLS],
                    index=0,
                    key="well_param",
                    help="Kies wellness parameter voor de grafiek (0–10).",
                )
                param_key = dict(ASRM_COLS)[param_label]
                plot_asrm_over_time(dfw, param_key)

    # ---- RPE ----
    with rpe_col:
        st.subheader("RPE")
        mode_r = st.radio(
            "Weergave",
            ["Session", "Over time"],
            horizontal=True,
            key="rpe_mode",
            help="Session: 1 dag | Over time: laatste 7 dagen",
        )

        if mode_r == "Session":
            d = st.date_input("Datum (RPE)", value=date.today(), key="rpe_date")
            sess = fetch_rpe_for_date(sb, target_player_id, d)
            if sess.empty:
                st.info("Geen RPE entry voor deze datum.")
            else:
                plot_rpe_session(sess)
        else:
            dfr = fetch_rpe_over_time_7d(sb, target_player_id)
            plot_rpe_over_time(dfr)
