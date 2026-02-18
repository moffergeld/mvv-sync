# session_load_pages.py
# ============================================================
# Session Load (Streamlit)
# ✅ Optie C: FullCalendar maand view + direct Session Load
# - Maand kalender (dayGridMonth) met kleuren:
#     Match/Practice Match = rood
#     Practice/data        = blauw
# - Klik op datum of event => selected_day en direct session load plots
# - Werkt beter op verschillende schermgroottes en mobiel
#
# Vereist: streamlit-calendar  -> voeg toe aan requirements.txt
# ============================================================

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_calendar import calendar as st_calendar

COL_DATE = "Datum"
COL_PLAYER = "Speler"
COL_EVENT = "Event"
COL_TYPE = "Type"

COL_TD = "Total Distance"
COL_SPRINT = "Sprint"
COL_HS = "High Sprint"
COL_ACC_TOT = "Total Accelerations"
COL_ACC_HI = "High Accelerations"
COL_DEC_TOT = "Total Decelerations"
COL_DEC_HI = "High Decelerations"

HR_COLS = ["HRzone1", "HRzone2", "HRzone3", "HRzone4", "HRzone5"]
TRIMP_CANDIDATES = ["HRTrimp", "HR Trimp", "HRtrimp", "Trimp", "TRIMP"]

MVV_RED = "#FF0033"
PRACTICE_BLUE = "#4AA3FF"


def _normalize_event(e: str) -> str:
    s = str(e).strip().lower()
    return "summary" if s == "summary" else s


def _is_match_type(t: str) -> bool:
    s = str(t).strip().lower()
    return "match" in s


def _prepare_gps(df_gps: pd.DataFrame) -> pd.DataFrame:
    df = df_gps.copy()

    if COL_DATE not in df.columns or COL_PLAYER not in df.columns:
        return df.iloc[0:0].copy()

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE, COL_PLAYER]).copy()

    # ✅ alleen Summary
    if COL_EVENT in df.columns:
        df["_event_norm"] = df[COL_EVENT].map(_normalize_event)
        df = df[df["_event_norm"] == "summary"].copy()

    # TRIMP kolom zoeken
    trimp_col = None
    for c in TRIMP_CANDIDATES:
        if c in df.columns:
            trimp_col = c
            break
    df["TRIMP"] = pd.to_numeric(df[trimp_col], errors="coerce").fillna(0.0) if trimp_col else 0.0

    numeric_cols = [
        COL_TD,
        COL_SPRINT,
        COL_HS,
        COL_ACC_TOT,
        COL_ACC_HI,
        COL_DEC_TOT,
        COL_DEC_HI,
        *HR_COLS,
        "TRIMP",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in [COL_PLAYER, COL_TYPE]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


def _compute_day_sets(df: pd.DataFrame) -> tuple[set[date], set[date]]:
    if df.empty:
        return set(), set()

    d = df.copy()
    d["_day"] = d[COL_DATE].dt.date
    days_with_data = set(d["_day"].dropna().tolist())

    match_days: set[date] = set()
    if COL_TYPE in d.columns:
        mask = d[COL_TYPE].map(_is_match_type)
        match_days = set(d.loc[mask, "_day"].dropna().tolist())

    return days_with_data, match_days


def _build_calendar_events(df: pd.DataFrame) -> list[dict]:
    days_with_data, match_days = _compute_day_sets(df)

    events: list[dict] = []
    for d in sorted(days_with_data):
        is_match = d in match_days
        events.append(
            {
                "title": "Match" if is_match else "Training",
                "start": d.isoformat(),
                "allDay": True,
                "color": MVV_RED if is_match else PRACTICE_BLUE,
                "textColor": "#ffffff",
                "extendedProps": {"day": d.isoformat()},
            }
        )
    return events


def calendar_day_picker_fullcalendar(df: pd.DataFrame, key_prefix: str = "sl") -> date:
    days_with_data, _ = _compute_day_sets(df)
    min_day = min(days_with_data) if days_with_data else date.today()
    max_day = max(days_with_data) if days_with_data else date.today()

    if f"{key_prefix}_selected" not in st.session_state:
        st.session_state[f"{key_prefix}_selected"] = max_day

    selected: date = st.session_state[f"{key_prefix}_selected"]

    events = _build_calendar_events(df)

    options = {
        "initialView": "dayGridMonth",
        "initialDate": selected.isoformat(),
        "height": "auto",
        "firstDay": 1,  # maandag
        "headerToolbar": {
            "left": "prev,next today",
            "center": "title",
            "right": "",
        },
        "fixedWeekCount": False,
        "dayMaxEvents": True,
        "eventDisplay": "block",
    }

    result = st_calendar(events=events, options=options, key=f"{key_prefix}_fc")

    if result:
        # Klik op lege dag
        if result.get("dateClick"):
            ds = result["dateClick"].get("dateStr")
            if ds:
                st.session_state[f"{key_prefix}_selected"] = date.fromisoformat(ds[:10])
                st.rerun()

        # Klik op event
        if result.get("eventClick"):
            ev = result["eventClick"].get("event", {})
            ds = ev.get("start")
            if ds:
                st.session_state[f"{key_prefix}_selected"] = date.fromisoformat(ds[:10])
                st.rerun()

    st.caption(f"Bereik: {min_day.strftime('%d-%m-%Y')} – {max_day.strftime('%d-%m-%Y')}")
    return st.session_state[f"{key_prefix}_selected"]


def _agg_by_player(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metric_cols = [
        COL_TD,
        COL_SPRINT,
        COL_HS,
        COL_ACC_TOT,
        COL_ACC_HI,
        COL_DEC_TOT,
        COL_DEC_HI,
        *HR_COLS,
        "TRIMP",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    return df.groupby(COL_PLAYER, as_index=False)[metric_cols].sum()


def _get_day_session_subset(df: pd.DataFrame, day: date, session_mode: str) -> pd.DataFrame:
    df_day = df[df[COL_DATE].dt.date == day].copy()
    if df_day.empty or COL_TYPE not in df_day.columns:
        return df_day

    types_day = sorted(df_day[COL_TYPE].dropna().astype(str).unique().tolist())
    if "Practice (1)" in types_day and "Practice (2)" in types_day:
        if session_mode == "Practice (1)":
            return df_day[df_day[COL_TYPE] == "Practice (1)"].copy()
        if session_mode == "Practice (2)":
            return df_day[df_day[COL_TYPE] == "Practice (2)"].copy()
        return df_day[df_day[COL_TYPE].isin(["Practice (1)", "Practice (2)"])].copy()

    return df_day


def _plot_total_distance(df_agg: pd.DataFrame):
    if COL_TD not in df_agg.columns:
        st.info("Kolom 'Total Distance' niet gevonden.")
        return

    data = df_agg.sort_values(COL_TD, ascending=False).reset_index(drop=True)
    players = data[COL_PLAYER].astype(str).tolist()
    vals = data[COL_TD].to_numpy()

    fig = go.Figure()
    fig.add_bar(
        x=players,
        y=vals,
        marker_color="rgba(255,150,150,0.9)",
        text=[f"{v:,.0f}".replace(",", " ") for v in vals],
        textposition="inside",
        insidetextanchor="middle",
        name="Total Distance",
    )
    mean_val = float(np.nanmean(vals)) if len(vals) else 0.0
    fig.add_hline(
        y=mean_val,
        line_dash="dot",
        line_color="black",
        annotation_text=f"Gem.: {mean_val:,.0f} m".replace(",", " "),
        annotation_position="top left",
        annotation_font_size=10,
    )
    fig.update_layout(
        title="Total Distance",
        yaxis_title="Total Distance (m)",
        xaxis_title=None,
        margin=dict(l=10, r=10, t=40, b=80),
    )
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


def _plot_sprint_hs(df_agg: pd.DataFrame):
    if COL_SPRINT not in df_agg.columns or COL_HS not in df_agg.columns:
        st.info("Sprint / High Sprint kolommen niet compleet.")
        return

    data = df_agg.sort_values(COL_SPRINT, ascending=False).reset_index(drop=True)
    players = data[COL_PLAYER].astype(str).tolist()
    sprint_vals = data[COL_SPRINT].to_numpy()
    hs_vals = data[COL_HS].to_numpy()

    x = np.arange(len(players))
    fig = go.Figure()
    fig.add_bar(
        x=x - 0.2,
        y=sprint_vals,
        width=0.4,
        name="Sprint",
        marker_color="rgba(255,180,180,0.9)",
        text=[f"{v:,.0f}".replace(",", " ") for v in sprint_vals],
        textposition="outside",
    )
    fig.add_bar(
        x=x + 0.2,
        y=hs_vals,
        width=0.4,
        name="High Sprint",
        marker_color="rgba(150,0,0,0.9)",
        text=[f"{v:,.0f}".replace(",", " ") for v in hs_vals],
        textposition="outside",
    )
    fig.update_layout(
        title="Sprint & High Sprint Distance",
        yaxis_title="Distance (m)",
        xaxis_title=None,
        barmode="group",
        margin=dict(l=10, r=10, t=40, b=80),
    )
    fig.update_xaxes(tickvals=x, ticktext=players, tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


def _plot_acc_dec(df_agg: pd.DataFrame):
    have_cols = [c for c in [COL_ACC_TOT, COL_ACC_HI, COL_DEC_TOT, COL_DEC_HI] if c in df_agg.columns]
    if not have_cols:
        st.info("Geen Acceleration/Deceleration kolommen gevonden.")
        return

    sort_col = COL_ACC_TOT if COL_ACC_TOT in df_agg.columns else have_cols[0]
    data = df_agg.sort_values(sort_col, ascending=False).reset_index(drop=True)
    players = data[COL_PLAYER].astype(str).tolist()
    x = np.arange(len(players))
    width = 0.18

    fig = go.Figure()
    if COL_ACC_TOT in data.columns:
        fig.add_bar(
            x=x - 1.5 * width,
            y=data[COL_ACC_TOT],
            width=width,
            name="Total Accelerations",
            marker_color="rgba(255,180,180,0.9)",
        )
    if COL_ACC_HI in data.columns:
        fig.add_bar(
            x=x - 0.5 * width,
            y=data[COL_ACC_HI],
            width=width,
            name="High Accelerations",
            marker_color="rgba(200,0,0,0.9)",
        )
    if COL_DEC_TOT in data.columns:
        fig.add_bar(
            x=x + 0.5 * width,
            y=data[COL_DEC_TOT],
            width=width,
            name="Total Decelerations",
            marker_color="rgba(180,210,255,0.9)",
        )
    if COL_DEC_HI in data.columns:
        fig.add_bar(
            x=x + 1.5 * width,
            y=data[COL_DEC_HI],
            width=width,
            name="High Decelerations",
            marker_color="rgba(0,60,180,0.9)",
        )

    fig.update_layout(
        title="Accelerations / Decelerations",
        yaxis_title="Aantal (N)",
        xaxis_title=None,
        barmode="group",
        margin=dict(l=10, r=10, t=40, b=80),
    )
    fig.update_xaxes(tickvals=x, ticktext=players, tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


def _plot_hr_trimp(df_agg: pd.DataFrame):
    have_hr = [c for c in HR_COLS if c in df_agg.columns]
    has_trimp = "TRIMP" in df_agg.columns
    if not have_hr and not has_trimp:
        st.info("Geen HR-zone kolommen of TRIMP gevonden.")
        return

    players = df_agg[COL_PLAYER].astype(str).tolist()
    base_x = np.arange(len(players))

    fig = make_subplots(specs=[[{"secondary_y": has_trimp}]])
    color_map = {
        "HRzone1": "rgba(180,180,180,0.9)",
        "HRzone2": "rgba(150,200,255,0.9)",
        "HRzone3": "rgba(0,150,0,0.9)",
        "HRzone4": "rgba(220,220,50,0.9)",
        "HRzone5": "rgba(255,0,0,0.9)",
    }

    if have_hr:
        n = len(have_hr)
        group_w = 0.80
        bar_w = group_w / max(n, 1)
        start = -group_w / 2 + bar_w / 2
        for idx, z in enumerate(have_hr):
            x = base_x + (start + idx * bar_w)
            fig.add_bar(
                x=x,
                y=df_agg[z],
                name=z,
                marker_color=color_map.get(z, "gray"),
                width=bar_w * 0.95,
                secondary_y=False,
            )

    if has_trimp:
        fig.add_trace(
            go.Scatter(
                x=base_x,
                y=df_agg["TRIMP"],
                mode="lines+markers",
                name="HR Trimp",
                line=dict(color="rgba(0,255,100,1.0)", width=3, shape="spline"),
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title="Time in HR zone",
        xaxis_title=None,
        barmode="group",
        bargap=0.15,
        margin=dict(l=10, r=10, t=40, b=80),
    )
    fig.update_xaxes(tickvals=base_x, ticktext=players, tickangle=90)
    fig.update_yaxes(title_text="Time in HR zone (min)", secondary_y=False)
    if has_trimp:
        fig.update_yaxes(title_text="HR Trimp", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)


def session_load_pages_main(df_gps: pd.DataFrame):
    st.header("Session Load")

    missing = [c for c in [COL_DATE, COL_PLAYER] if c not in df_gps.columns]
    if missing:
        st.error(f"Ontbrekende kolommen in GPS-data: {missing}")
        return

    df = _prepare_gps(df_gps)
    if df.empty:
        st.warning("Geen bruikbare GPS-data gevonden (controleer Event='Summary' en Datum/Speler).")
        return

    # ✅ FullCalendar maand view (mobielvriendelijk)
    selected_day = calendar_day_picker_fullcalendar(df, key_prefix="sl")

    df_day_all = df[df[COL_DATE].dt.date == selected_day].copy()
    if df_day_all.empty:
        st.info(f"Geen data op {selected_day.strftime('%d-%m-%Y')}.")
        return

    types_day = (
        sorted(df_day_all[COL_TYPE].dropna().astype(str).unique().tolist())
        if COL_TYPE in df_day_all.columns
        else []
    )

    session_mode = "Beide (1+2)"
    if "Practice (1)" in types_day and "Practice (2)" in types_day:
        session_mode = st.radio(
            "Sessie",
            options=["Practice (1)", "Practice (2)", "Beide (1+2)"],
            index=2,
            key="session_load_session_mode",
        )

    df_day = _get_day_session_subset(df, selected_day, session_mode)
    if df_day.empty:
        st.warning("Geen data gevonden voor deze selectie (dag + sessie).")
        return

    st.caption("Beschikbare sessie op deze dag: " + (", ".join(types_day) if types_day else "—"))

    df_agg = _agg_by_player(df_day)
    if df_agg.empty:
        st.warning("Geen data om te aggregeren per speler.")
        return

    col_top1, col_top2 = st.columns(2)
    with col_top1:
        _plot_total_distance(df_agg)
    with col_top2:
        _plot_sprint_hs(df_agg)

    col_bot1, col_bot2 = st.columns(2)
    with col_bot1:
        _plot_acc_dec(df_agg)
    with col_bot2:
        _plot_hr_trimp(df_agg)
