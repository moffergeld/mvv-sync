# player_pages.py
# ============================================================
# Player pagina: tabs Data + Forms
#
# FIX RPE save error:
#   "SyncQueryRequestBuilder object has no attribute 'select'"
# -> in supabase-py v2 werkt .select() niet op upsert builder zoals in sommige voorbeelden.
# -> Oplossing: upsert header, daarna opnieuw SELECT header.id (via maybe_single).
#
# DATA tab:
#   2x2 layout:
#     [GPS Session Table]    [GPS Over time graph]
#     [Wellness switch]      [RPE switch]
#
# GPS data uit: public.v_gps_summary
# ============================================================

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from roles import get_sb, require_auth, get_profile, pick_target_player


# -----------------------------
# Utils
# -----------------------------
def _df_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _add_zone_background(fig: go.Figure, y_min: float = 1, y_max: float = 10):
    zones = [
        (1, 4, "rgba(0, 200, 0, 0.12)"),
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
    fig.update_yaxes(range=[y_min, y_max])


# -----------------------------
# GPS (v_gps_summary)
# -----------------------------
GPS_TABLE = "v_gps_summary"

GPS_METRICS = [
    ("Total Distance", "total_distance"),
    ("Running", "running"),
    ("Sprint", "sprint"),
    ("High Sprint", "high_sprint"),
    ("Max Speed", "max_speed"),
    ("PlayerLoad2D", "playerload2d"),
]

GPS_RECENT_COLS = [
    "datum",
    "type",
    "total_distance",
    "running",
    "sprint",
    "high_sprint",
    "max_speed",
    "playerload2d",
]


def fetch_gps_summary_recent(sb, player_id: str, limit: int = 30) -> pd.DataFrame:
    try:
        rows = (
            sb.table(GPS_TABLE)
            .select(",".join(set(GPS_RECENT_COLS + ["player_id"])))
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
        df["datum"] = pd.to_datetime(df["datum"]).dt.date
        return df
    except Exception:
        return pd.DataFrame()


def fetch_gps_summary_for_date(sb, player_id: str, d: date) -> Optional[Dict[str, Any]]:
    try:
        row = (
            sb.table(GPS_TABLE)
            .select(",".join(set(GPS_RECENT_COLS + ["player_id"])))
            .eq("player_id", player_id)
            .eq("datum", d.isoformat())
            .maybe_single()
            .execute()
            .data
        )
        return row
    except Exception:
        return None


def plot_gps_over_time(df: pd.DataFrame, metric_label: str, metric_key: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["datum"],
            y=pd.to_numeric(df[metric_key], errors="coerce"),
            mode="lines+markers",
            name=metric_label,
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=340,
        xaxis_title="Datum",
        yaxis_title=metric_label,
        showlegend=False,
    )
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


def fetch_asrm_over_time(sb, player_id: str, limit: int = 60) -> pd.DataFrame:
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
        df["entry_date"] = pd.to_datetime(df["entry_date"]).dt.date
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
            name=param_label,
        )
    )
    _add_zone_background(fig)
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=340,
        xaxis_title="Datum",
        yaxis_title="Score (1-10)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_asrm_session(row: Dict[str, Any], title: str):
    labels = [x[0] for x in ASRM_COLS]
    keys = [x[1] for x in ASRM_COLS]
    values = [int(row.get(k, 0) or 0) for k in keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values))
    _add_zone_background(fig)
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=340,
        yaxis_title="Score (1-10)",
        title=title,
        showlegend=False,
    )
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
    """
    Na upsert: haal de id op met een SELECT.
    Dit voorkomt de .select() op upsert builder bug.
    """
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

    # 1) upsert header (geen .select() hier!)
    sb.table("rpe_entries").upsert(header_payload, on_conflict="player_id,entry_date").execute()

    # 2) fetch id
    rpe_entry_id = _get_rpe_entry_id(sb, player_id, entry_date)
    if not rpe_entry_id:
        raise RuntimeError("Kon rpe_entry_id niet ophalen na opslaan.")

    # 3) upsert sessions
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


def fetch_rpe_header_over_time(sb, player_id: str, limit: int = 60) -> pd.DataFrame:
    try:
        rows = (
            sb.table("rpe_entries")
            .select("id,entry_date,injury,injury_pain")
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
        df["entry_date"] = pd.to_datetime(df["entry_date"]).dt.date
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


def build_rpe_timeseries(sb, player_id: str, limit: int = 60) -> pd.DataFrame:
    h = fetch_rpe_header_over_time(sb, player_id, limit=limit)
    if h.empty:
        return h

    entry_ids = h["id"].astype(str).tolist()
    s = fetch_rpe_sessions_for_entry_ids(sb, entry_ids)
    if s.empty:
        h["duration_sum"] = 0
        h["session_load_sum"] = 0
        h["rpe_avg"] = None
        return h

    s["duration_min"] = pd.to_numeric(s["duration_min"], errors="coerce").fillna(0.0)
    s["rpe"] = pd.to_numeric(s["rpe"], errors="coerce").fillna(0.0)
    s["session_load"] = s["duration_min"] * s["rpe"]

    agg = (
        s.groupby("rpe_entry_id", as_index=False)
        .agg(duration_sum=("duration_min", "sum"), session_load_sum=("session_load", "sum"))
    )

    w = s.groupby("rpe_entry_id", as_index=False).apply(
        lambda g: pd.Series(
            {"rpe_avg": (g["session_load"].sum() / g["duration_min"].sum()) if g["duration_min"].sum() > 0 else None}
        )
    )
    w = w.reset_index(drop=True)

    out = h.merge(agg, left_on="id", right_on="rpe_entry_id", how="left").drop(columns=["rpe_entry_id"])
    out = out.merge(w, left_on="id", right_on="rpe_entry_id", how="left").drop(columns=["rpe_entry_id"])
    return out.sort_values("entry_date")


def fetch_rpe_for_date(sb, player_id: str, d: date) -> Tuple[Optional[Dict[str, Any]], pd.DataFrame]:
    try:
        header = (
            sb.table("rpe_entries")
            .select("*")
            .eq("player_id", player_id)
            .eq("entry_date", d.isoformat())
            .maybe_single()
            .execute()
            .data
        )
    except Exception:
        header = None

    if not header or not header.get("id"):
        return header, pd.DataFrame()

    try:
        rows = (
            sb.table("rpe_sessions")
            .select("session_index,duration_min,rpe")
            .eq("rpe_entry_id", header["id"])
            .order("session_index")
            .execute()
            .data
            or []
        )
        df = _df_from_rows(rows)
        if not df.empty:
            df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce").fillna(0)
            df["rpe"] = pd.to_numeric(df["rpe"], errors="coerce").fillna(0)
            df["session_load"] = df["duration_min"] * df["rpe"]
        return header, df
    except Exception:
        return header, pd.DataFrame()


def plot_rpe_over_time(df: pd.DataFrame, metric_key: str, metric_label: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["entry_date"],
            y=pd.to_numeric(df[metric_key], errors="coerce"),
            mode="lines+markers",
            name=metric_label,
        )
    )
    if metric_key == "rpe_avg":
        _add_zone_background(fig)
        ytitle = "Score (1-10)"
    else:
        ytitle = metric_label

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=340,
        xaxis_title="Datum",
        yaxis_title=ytitle,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_rpe_session(sessions_df: pd.DataFrame, title: str):
    if sessions_df.empty:
        st.info("Geen RPE sessions gevonden voor deze datum.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sessions_df["session_index"].astype(str),
            y=sessions_df["session_load"],
            name="Session load (dur*rpe)",
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sessions_df["session_index"].astype(str),
            y=sessions_df["rpe"],
            mode="lines+markers",
            name="RPE",
            yaxis="y2",
        )
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=340,
        title=title,
        xaxis_title="Session #",
        yaxis_title="Session load",
        yaxis2=dict(
            title="RPE (1-10)",
            overlaying="y",
            side="right",
            range=[1, 10],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )

    for y, col in [(5, "rgba(255,165,0,0.35)"), (8, "rgba(255,0,0,0.35)")]:
        fig.add_shape(
            type="line",
            xref="paper",
            yref="y2",
            x0=0,
            x1=1,
            y0=y,
            y1=y,
            line=dict(width=2, color=col),
            layer="above",
        )

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# UI
# -----------------------------
def player_pages_main():
    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar. Controleer secrets + supabase package.")
        st.stop()

    profile = get_profile(sb)
    target_player_id, target_player_name, _ = pick_target_player(sb, profile, label="Speler", key="pp_player_select")

    if not target_player_id:
        st.error("Geen speler beschikbaar (players leeg of geen toegang).")
        st.stop()

    st.title(f"Player: {target_player_name}")

    tab_data, tab_forms = st.tabs(["Data", "Forms"])

    # ============================
    # DATA TAB
    # ============================
    with tab_data:
        gps_left, gps_right = st.columns(2)

        with gps_left:
            st.subheader("GPS – Session")

            gps_df_recent = fetch_gps_summary_recent(sb, target_player_id, limit=60)
            if gps_df_recent.empty:
                st.info("Geen GPS summary data gevonden (v_gps_summary).")
            else:
                available_dates = sorted(gps_df_recent["datum"].dropna().unique().tolist(), reverse=True)

                sel_date = st.selectbox(
                    "Datum",
                    options=available_dates,
                    index=0,
                    key="gps_date_sel",
                    format_func=lambda d: d.isoformat(),
                )

                row = fetch_gps_summary_for_date(sb, target_player_id, sel_date)
                if not row:
                    st.info("Geen GPS entry voor deze datum.")
                else:
                    show = {k: row.get(k) for k in GPS_RECENT_COLS if k in row}
                    st.dataframe(pd.DataFrame([show]), use_container_width=True, hide_index=True)

                st.markdown("**Recent sessions (laatste 10)**")
                show_df = gps_df_recent.sort_values("datum", ascending=False).head(10)
                cols = [c for c in GPS_RECENT_COLS if c in show_df.columns]
                st.dataframe(show_df[cols], use_container_width=True, hide_index=True)

        with gps_right:
            st.subheader("GPS – Over time")

            gps_df_recent = fetch_gps_summary_recent(sb, target_player_id, limit=60)
            if gps_df_recent.empty:
                st.info("Geen GPS summary data gevonden (v_gps_summary).")
            else:
                metric_label = st.selectbox(
                    "Parameter",
                    options=[m[0] for m in GPS_METRICS],
                    index=0,
                    key="gps_metric_sel",
                )
                metric_key = dict(GPS_METRICS)[metric_label]
                df_plot = gps_df_recent.sort_values("datum").tail(10)
                plot_gps_over_time(df_plot, metric_label, metric_key)

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
                    plot_asrm_session(row, title=f"Wellness (Session) — {d.isoformat()}")
            else:
                df = fetch_asrm_over_time(sb, target_player_id, limit=90)
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
                header, sessions_df = fetch_rpe_for_date(sb, target_player_id, d)
                if not header:
                    st.info("Geen RPE entry voor deze datum.")
                else:
                    plot_rpe_session(sessions_df, title=f"RPE (Session) — {d.isoformat()}")
            else:
                df = build_rpe_timeseries(sb, target_player_id, limit=90)
                if df.empty:
                    st.info("Geen RPE entries gevonden.")
                else:
                    options = [("RPE (avg)", "rpe_avg"), ("Duration (sum)", "duration_sum"), ("Session Load (sum)", "session_load_sum")]
                    metric_label = st.selectbox("Parameter", options=[o[0] for o in options], index=0, key="rpe_param")
                    metric_key = dict(options)[metric_label]
                    plot_rpe_over_time(df, metric_key, metric_label)

    # ============================
    # FORMS TAB
    # ============================
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

            injury_default = bool(header.get("injury", False))
            injury = st.toggle("Injury?", value=injury_default, key="rpe_injury")

            injury_type = st.text_input(
                "Injury type",
                value=str(header.get("injury_type") or ""),
                disabled=not injury,
                key="rpe_injury_type",
            )
            injury_pain = st.slider(
                "Pain (0–10)",
                0,
                10,
                value=int(header.get("injury_pain", 0) or 0),
                disabled=not injury,
                key="rpe_pain",
            )

            notes = st.text_area("Notes (optioneel)", value=str(header.get("notes") or ""), key="rpe_notes")

            st.markdown("### Sessions")

            def _sess(idx: int, key: str, default: int) -> int:
                hit = next((s for s in sessions if int(s.get("session_index", 0)) == idx), None)
                if not hit:
                    return default
                v = hit.get(key)
                return int(v) if v is not None else default

            s1_dur = st.number_input("[1] Duration (min)", 0, 600, value=_sess(1, "duration_min", 0), key="rpe_s1_dur")
            s1_rpe = st.slider("[1] RPE (1–10)", 1, 10, value=_sess(1, "rpe", 5), key="rpe_s1_rpe")

            s2_dur = st.number_input("[2] Duration (min)", 0, 600, value=_sess(2, "duration_min", 0), key="rpe_s2_dur")
            s2_rpe = st.slider("[2] RPE (1–10)", 1, 10, value=_sess(2, "rpe", 5), key="rpe_s2_rpe")

            sessions_payload: List[Dict[str, int]] = []
            if s1_dur > 0:
                sessions_payload.append({"session_index": 1, "duration_min": int(s1_dur), "rpe": int(s1_rpe)})
            if s2_dur > 0:
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
