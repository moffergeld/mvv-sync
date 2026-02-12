# player_pages.py
# ============================================================
# Update (requested):
# - Wellness + RPE charts:
#   - Geen chart-titels (bovenin)
#   - Geen x-as titel ("Date" / "Session #")
#   - Zelfde hoogte voor Wellness en RPE (session + over time)
# - Fix: _add_zone_background accepteert nu y_min/y_max zodat TypeError weg is
#
# Verder:
# - GPS tabel blijft “alles apart”
# - GPS over-time blijft per dag gesommeerd + smooth + rood
# - RPE over-time: per datum 1 punt (gewogen avg) + errorbars (min/max)
# ============================================================

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from roles import get_sb, require_auth, get_profile, pick_target_player


CHART_HEIGHT = 340  # overal dezelfde hoogte


# -----------------------------
# Utils
# -----------------------------
def _df_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _coerce_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _add_zone_background(fig: go.Figure, y_min: float = 0, y_max: float = 10, dtick: int = 1):
    zones = [
        (0, 4, "rgba(0, 200, 0, 0.12)"),
        (5, 7, "rgba(255, 165, 0, 0.14)"),
        (8, 10, "rgba(255, 0, 0, 0.14)"),
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
    fig.update_yaxes(range=[y_min, y_max], tick0=y_min, dtick=dtick)


# -----------------------------
# GPS (v_gps_summary)
# -----------------------------
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


def fetch_gps_summary_recent_raw(sb, player_id: str, limit: int = 200) -> pd.DataFrame:
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
    show = show[cols].rename(columns=GPS_RENAME)
    return show


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
            name=metric_label,
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=CHART_HEIGHT,
        xaxis_title="Date",
        yaxis_title=metric_label,
        showlegend=False,
    )
    fig.update_xaxes(type="date", tickformat="%d-%m-%Y")
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# ASRM (Wellness)
# -----------------------------
ASRM_COLS = [
    ("Muscle soreness", "muscle_soreness"),
    ("Fatigue", "fatigue"),
    ("Sleep quality", "sleep_quality"),
    ("Stress", "stress"),
    ("Mood", "mood"),
]


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


def fetch_asrm_for_date(sb, player_id: str, d: date) -> Optional[Dict[str, Any]]:
    return load_asrm(sb, player_id, d)


def save_asrm(
    sb,
    player_id: str,
    entry_date: date,
    muscle_soreness: int,
    fatigue: int,
    sleep_quality: int,
    stress: int,
    mood: int,
):
    payload = {
        "player_id": player_id,
        "entry_date": entry_date.isoformat(),
        "muscle_soreness": int(muscle_soreness),
        "fatigue": int(fatigue),
        "sleep_quality": int(sleep_quality),
        "stress": int(stress),
        "mood": int(mood),
    }
    sb.table("asrm_entries").upsert(payload, on_conflict="player_id,entry_date").execute()


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


def plot_asrm_over_time(df: pd.DataFrame, param_label: str, param_key: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["entry_date"],
            y=pd.to_numeric(df[param_key], errors="coerce"),
            mode="lines+markers",
            line=dict(color="#FF0033", width=3, shape="spline", smoothing=1.2),
            marker=dict(size=6),
            name=param_label,
        )
    )
    _add_zone_background(fig, y_min=0, y_max=10, dtick=1)
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=CHART_HEIGHT,
        showlegend=False,
        xaxis_title=None,   # geen x-as titel
        title=None,         # geen titel
    )
    fig.update_xaxes(type="date", tickformat="%d-%m-%Y")
    fig.update_yaxes(title_text="Score (0–10)")
    st.plotly_chart(fig, use_container_width=True)


def plot_asrm_session(row: Dict[str, Any]):
    labels = [x[0] for x in ASRM_COLS]
    keys = [x[1] for x in ASRM_COLS]
    values = [int(row.get(k, 0) or 0) for k in keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values))
    _add_zone_background(fig, y_min=0, y_max=10, dtick=1)
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=CHART_HEIGHT,
        showlegend=False,
        xaxis_title=None,   # geen x-as titel
        title=None,         # geen titel
    )
    fig.update_yaxes(title_text="Score (0–10)")
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# RPE
# -----------------------------
def load_rpe(sb, player_id: str, entry_date: date) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    header = None
    sessions: List[Dict[str, Any]] = []
    try:
        r = (
            sb.table("rpe_entries")
            .select("*")
            .eq("player_id", player_id)
            .eq("entry_date", entry_date.isoformat())
            .maybe_single()
            .execute()
        )
        header = r.data
        if header and header.get("id"):
            s = (
                sb.table("rpe_sessions")
                .select("*")
                .eq("rpe_entry_id", header["id"])
                .order("session_index")
                .execute()
            )
            sessions = s.data or []
    except Exception:
        pass
    return header, sessions


def _get_rpe_entry_id(sb, player_id: str, entry_date: date) -> Optional[str]:
    try:
        r = (
            sb.table("rpe_entries")
            .select("id")
            .eq("player_id", player_id)
            .eq("entry_date", entry_date.isoformat())
            .maybe_single()
            .execute()
        )
        if r.data and r.data.get("id"):
            return str(r.data["id"])
    except Exception:
        pass
    return None


def save_rpe(
    sb,
    player_id: str,
    entry_date: date,
    injury: bool,
    injury_type: Optional[str],
    injury_pain: Optional[int],
    notes: str,
    sessions: List[Dict[str, int]],
):
    header_payload = {
        "player_id": player_id,
        "entry_date": entry_date.isoformat(),
        "injury": bool(injury),
        "injury_type": injury_type if injury else None,
        "injury_pain": int(injury_pain) if (injury and injury_pain is not None) else None,
        "notes": notes.strip() if notes else None,
    }

    sb.table("rpe_entries").upsert(header_payload, on_conflict="player_id,entry_date").execute()

    rpe_entry_id = _get_rpe_entry_id(sb, player_id, entry_date)
    if not rpe_entry_id:
        raise RuntimeError("Kon rpe_entry_id niet ophalen na opslaan.")

    payload: List[Dict[str, Any]] = []
    for s in sessions:
        payload.append(
            {
                "rpe_entry_id": rpe_entry_id,
                "session_index": int(s["session_index"]),
                "duration_min": int(s["duration_min"]),
                "rpe": int(s["rpe"]),
            }
        )
    if payload:
        sb.table("rpe_sessions").upsert(payload, on_conflict="rpe_entry_id,session_index").execute()


def fetch_rpe_header_over_time(sb, player_id: str, limit: int = 180) -> pd.DataFrame:
    try:
        rows = (
            sb.table("rpe_entries")
            .select("id,entry_date")
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


def fetch_rpe_sessions_for_entry_ids(sb, entry_ids: List[str]) -> pd.DataFrame:
    if not entry_ids:
        return pd.DataFrame()
    try:
        rows = (
            sb.table("rpe_sessions")
            .select("rpe_entry_id,session_index,duration_min,rpe")
            .in_("rpe_entry_id", entry_ids)
            .execute()
            .data
            or []
        )
        return _df_from_rows(rows)
    except Exception:
        return pd.DataFrame()


def build_rpe_timeseries_daily(sb, player_id: str, limit: int = 180) -> pd.DataFrame:
    h = fetch_rpe_header_over_time(sb, player_id, limit=limit)
    if h.empty:
        return h

    entry_ids = h["id"].astype(str).tolist()
    s = fetch_rpe_sessions_for_entry_ids(sb, entry_ids)
    if s.empty:
        return pd.DataFrame(columns=["entry_date", "avg_rpe", "min_rpe", "max_rpe"])

    s["duration_min"] = pd.to_numeric(s["duration_min"], errors="coerce").fillna(0.0)
    s["rpe"] = pd.to_numeric(s["rpe"], errors="coerce").fillna(0.0)
    s["load"] = s["duration_min"] * s["rpe"]

    m = s.merge(h[["id", "entry_date"]], left_on="rpe_entry_id", right_on="id", how="left").drop(columns=["id"])
    m = m.dropna(subset=["entry_date"])
    m["entry_date"] = _coerce_date_series(m["entry_date"])

    def _avg_weighted(g: pd.DataFrame) -> float:
        dur = float(g["duration_min"].sum())
        if dur <= 0:
            return float(g["rpe"].mean()) if len(g) else float("nan")
        return float(g["load"].sum() / dur)

    daily = (
        m.groupby("entry_date", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "avg_rpe": _avg_weighted(g),
                    "min_rpe": float(g["rpe"].min()) if len(g) else None,
                    "max_rpe": float(g["rpe"].max()) if len(g) else None,
                }
            )
        )
        .reset_index(drop=True)
        .sort_values("entry_date")
    )

    return daily.tail(30)


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
        df["rpe"] = pd.to_numeric(df["rpe"], errors="coerce").fillna(0.0)
        return df
    except Exception:
        return pd.DataFrame()


def plot_rpe_over_time_daily(df_daily: pd.DataFrame):
    if df_daily is None or df_daily.empty:
        st.info("Geen RPE entries gevonden.")
        return

    dff = df_daily.copy()
    dff["entry_date"] = pd.to_datetime(dff["entry_date"], errors="coerce").dt.date
    dff = dff.dropna(subset=["entry_date"])

    dff = dff.groupby("entry_date", as_index=False).agg(
        avg_rpe=("avg_rpe", "mean"),
        min_rpe=("min_rpe", "min"),
        max_rpe=("max_rpe", "max"),
    ).sort_values("entry_date")

    y = pd.to_numeric(dff["avg_rpe"], errors="coerce")
    ymin = pd.to_numeric(dff["min_rpe"], errors="coerce")
    ymax = pd.to_numeric(dff["max_rpe"], errors="coerce")

    y_up = (ymax - y).fillna(0)
    y_dn = (y - ymin).fillna(0)

    x_vals = dff["entry_date"].tolist()
    x_text = [pd.to_datetime(d).strftime("%d-%m-%Y") for d in x_vals]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y,
            mode="lines+markers",
            line=dict(color="#FF0033", width=3, shape="spline", smoothing=1.2),
            marker=dict(size=7),
            error_y=dict(
                type="data",
                symmetric=False,
                array=y_up,
                arrayminus=y_dn,
                thickness=1.5,
                width=6,
            ),
            name="RPE",
        )
    )

    _add_zone_background(fig, y_min=0, y_max=10, dtick=1)

    fig.update_xaxes(
        type="category",
        tickmode="array",
        tickvals=x_vals,
        ticktext=x_text,
        title_text=None,  # geen x-as titel
    )
    fig.update_yaxes(title_text="RPE (0–10)")

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=CHART_HEIGHT,
        showlegend=False,
        title=None,  # geen titel
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_rpe_session_bars(df_sessions: pd.DataFrame):
    """
    RPE session chart:
    - Bars (zoals Wellness)
    - Zones (groen/oranje/rood)
    - Y: 0–10 dtick=1
    - X: alleen 1 en 2
    - Geen titel, geen x-as titel
    """
    if df_sessions is None or df_sessions.empty:
        st.info("Geen RPE sessions gevonden voor deze datum.")
        return

    df = df_sessions.copy()
    df["session_index"] = pd.to_numeric(df["session_index"], errors="coerce").fillna(0).astype(int)
    df["rpe"] = pd.to_numeric(df["rpe"], errors="coerce").fillna(0.0)

    df = df[df["session_index"].isin([1, 2])].sort_values("session_index")
    if df.empty:
        st.info("Geen RPE sessions (1/2) gevonden voor deze datum.")
        return

    x = df["session_index"].astype(str).tolist()
    y = df["rpe"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y, opacity=0.9))

    _add_zone_background(fig, y_min=0, y_max=10, dtick=1)

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=CHART_HEIGHT,
        showlegend=False,
        title=None,        # geen titel
        xaxis_title=None,  # geen x-as titel
        yaxis_title="RPE (0–10)",
    )

    fig.update_xaxes(
        type="category",
        tickmode="array",
        tickvals=["1", "2"],
        ticktext=["1", "2"],
    )

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# UI
# -----------------------------
def player_pages_main():
    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    profile = get_profile(sb)
    target_player_id, target_player_name, _ = pick_target_player(sb, profile, label="Speler", key="pp_player_select")
    if not target_player_id:
        st.error("Geen speler beschikbaar.")
        st.stop()

    st.title(f"Player: {target_player_name}")
    tab_data, tab_forms = st.tabs(["Data", "Forms"])

    # -------------------------
    # DATA TAB
    # -------------------------
    with tab_data:
        left, right = st.columns(2)

        with left:
            st.subheader("GPS – Recent (laatste 20)")
            gps_raw = fetch_gps_summary_recent_raw(sb, target_player_id, limit=200)
            if gps_raw.empty:
                st.info("Geen GPS summary data gevonden (v_gps_summary).")
            else:
                st.dataframe(gps_table_pretty(gps_raw, n=20), use_container_width=True, hide_index=True)

        with right:
            st.subheader("GPS – Over time")
            gps_raw = fetch_gps_summary_recent_raw(sb, target_player_id, limit=200)
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
                row = fetch_asrm_for_date(sb, target_player_id, d)
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
                    plot_asrm_over_time(df, param_label, param_key)

        with rpe_col:
            st.subheader("RPE")
            mode_r = st.radio("Weergave", ["Session", "Over time"], horizontal=True, key="rpe_mode")

            if mode_r == "Session":
                d = st.date_input("Datum (RPE)", value=date.today(), key="rpe_date")
                sess = fetch_rpe_for_date(sb, target_player_id, d)
                if sess.empty:
                    st.info("Geen RPE entry voor deze datum.")
                else:
                    plot_rpe_session_bars(sess)
            else:
                daily = build_rpe_timeseries_daily(sb, target_player_id, limit=180)
                if daily.empty or daily["avg_rpe"].isna().all():
                    st.info("Geen RPE entries gevonden.")
                else:
                    plot_rpe_over_time_daily(daily)

    # -------------------------
    # FORMS TAB
    # -------------------------
    with tab_forms:
        st.header("Forms")
        entry_date = st.date_input("Datum", value=date.today(), key="form_date")

        col_asrm, col_rpe = st.columns(2)

        with col_asrm:
            st.subheader("ASRM (Wellbeing)")
            existing = load_asrm(sb, target_player_id, entry_date) or {}
            ms = st.slider("Muscle soreness (1–10)", 1, 10, value=int(existing.get("muscle_soreness", 5)), key="asrm_ms")
            fat = st.slider("Fatigue (1–10)", 1, 10, value=int(existing.get("fatigue", 5)), key="asrm_fat")
            sleep = st.slider("Sleep quality (1–10)", 1, 10, value=int(existing.get("sleep_quality", 5)), key="asrm_sleep")
            stress = st.slider("Stress (1–10)", 1, 10, value=int(existing.get("stress", 5)), key="asrm_stress")
            mood = st.slider("Mood (1–10)", 1, 10, value=int(existing.get("mood", 5)), key="asrm_mood")

            if st.button("ASRM opslaan", use_container_width=True, key="asrm_save"):
                try:
                    save_asrm(sb, target_player_id, entry_date, ms, fat, sleep, stress, mood)
                    st.success("ASRM opgeslagen.")
                except Exception as e:
                    st.error(f"Opslaan faalde: {e}")

        with col_rpe:
            st.subheader("RPE (Session)")
            header, sessions = load_rpe(sb, target_player_id, entry_date)
            header = header or {}
            sessions = sessions or []

            has_s2 = any(int(s.get("session_index", 0) or 0) == 2 for s in sessions)
            enable_s2 = st.toggle("2e sessie invullen?", value=has_s2, key="rpe_enable_s2")

            injury_default = bool(header.get("injury", False))
            injury = st.toggle("Injury?", value=injury_default, key="rpe_injury")

            injury_type = st.text_input("Injury type", value=str(header.get("injury_type") or ""), disabled=not injury, key="rpe_injury_type")
            injury_pain = st.slider("Pain (0–10)", 0, 10, value=int(header.get("injury_pain", 0) or 0), disabled=not injury, key="rpe_pain")
            notes = st.text_area("Notes (optioneel)", value=str(header.get("notes") or ""), key="rpe_notes")

            st.markdown("### Sessions")

            def _sess(idx: int, key: str, default: int) -> int:
                hit = next((s for s in sessions if int(s.get("session_index", 0) or 0) == idx), None)
                if not hit:
                    return default
                v = hit.get(key)
                return int(v) if v is not None else default

            s1_dur = st.number_input("[1] Duration (min)", 0, 600, value=_sess(1, "duration_min", 0), key="rpe_s1_dur")
            s1_rpe = st.slider("[1] RPE (1–10)", 1, 10, value=_sess(1, "rpe", 5), key="rpe_s1_rpe")

            if enable_s2:
                s2_dur = st.number_input("[2] Duration (min)", 0, 600, value=_sess(2, "duration_min", 0), key="rpe_s2_dur")
                s2_rpe = st.slider("[2] RPE (1–10)", 1, 10, value=_sess(2, "rpe", 5), key="rpe_s2_rpe")
            else:
                s2_dur, s2_rpe = 0, 0

            sessions_payload: List[Dict[str, int]] = []
            if s1_dur > 0:
                sessions_payload.append({"session_index": 1, "duration_min": int(s1_dur), "rpe": int(s1_rpe)})
            if enable_s2 and s2_dur > 0:
                sessions_payload.append({"session_index": 2, "duration_min": int(s2_dur), "rpe": int(s2_rpe)})

            if st.button("RPE opslaan", use_container_width=True, key="rpe_save"):
                try:
                    save_rpe(
                        sb,
                        player_id=target_player_id,
                        entry_date=entry_date,
                        injury=injury,
                        injury_type=injury_type.strip() or None,
                        injury_pain=int(injury_pain) if injury else None,
                        notes=notes,
                        sessions=sessions_payload,
                    )
                    st.success("RPE opgeslagen.")
                except Exception as e:
                    st.error(f"Opslaan faalde: {e}")


if __name__ == "__main__":
    player_pages_main()
