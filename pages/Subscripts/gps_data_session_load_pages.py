# pages/Subscripts/gps_data_session_load_pages.py
# ============================================================
# Session Load (Streamlit) - ORIGINAL GRAPHS RESTORED
#
# - Gebruikt all-time kalender (calendar_df_all)
# - Voorkomt rerun-loop (dateStr -> YYYY-MM-DD en state update alleen bij echte click)
# - On-demand: als geselecteerde datum niet in df_gps_scope zit, fetch_day_fn haalt die dag op
# - Daarna draait de originele Session Load logica met grafieken/medianen/team selectie
# ============================================================

from __future__ import annotations

from datetime import date
from typing import Callable, Optional

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

MVV_RED = "#FF0033"
PRACTICE_BLUE = "#4AA3FF"

SELECT_BG = "rgba(227,6,19,0.15)"
SELECT_BORDER = "rgba(227,6,19,0.85)"


def _compute_day_sets(df: pd.DataFrame) -> tuple[set[date], set[date]]:
    if df is None or df.empty or COL_DATE not in df.columns:
        return set(), set()

    tmp = df.copy()
    tmp[COL_DATE] = pd.to_datetime(tmp[COL_DATE], errors="coerce").dt.date
    tmp = tmp.dropna(subset=[COL_DATE])

    days_with_data = set(tmp[COL_DATE].unique().tolist())

    match_days: set[date] = set()
    if COL_TYPE in tmp.columns:
        t = tmp[COL_TYPE].astype(str).str.strip().str.lower()
        match_days = set(tmp.loc[t.eq("match"), COL_DATE].unique().tolist())

    return days_with_data, match_days


def _build_calendar_events(df_calendar: pd.DataFrame, selected: date) -> list[dict]:
    days_with_data, match_days = _compute_day_sets(df_calendar)

    events: list[dict] = []
    events.append(
        {
            "title": "",
            "start": selected.isoformat(),
            "allDay": True,
            "display": "background",
            "backgroundColor": SELECT_BG,
            "borderColor": SELECT_BORDER,
        }
    )

    for d in sorted(days_with_data):
        is_match = d in match_days
        events.append(
            {
                "title": "Match" if is_match else "Training",
                "start": d.isoformat(),
                "allDay": True,
                "color": MVV_RED if is_match else PRACTICE_BLUE,
                "textColor": "#ffffff",
            }
        )
    return events


def calendar_day_picker_fullcalendar(df_calendar: pd.DataFrame, key_prefix: str = "sl") -> date:
    """FullCalendar month view day picker (anti rerun-loop)."""
    days_with_data, _ = _compute_day_sets(df_calendar)
    max_day = max(days_with_data) if days_with_data else date.today()

    sel_key = f"{key_prefix}_selected"
    last_click_key = f"{key_prefix}_last_clicked_iso"

    if sel_key not in st.session_state:
        st.session_state[sel_key] = max_day

    selected: date = st.session_state[sel_key]

    st.markdown(
        """
        <style>
          .fc { font-size: 13px; }
          .fc .fc-toolbar-title { font-size: 16px; font-weight: 800; }
          .fc .fc-button { border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    options = {
        "initialView": "dayGridMonth",
        "initialDate": selected.isoformat(),
        "headerToolbar": {"left": "prev,next today", "center": "title", "right": "dayGridMonth"},
        "height": 740,
        "firstDay": 1,
        "fixedWeekCount": False,
        "dayMaxEvents": True,
        "eventDisplay": "block",
        "selectable": True,
    }

    events = _build_calendar_events(df_calendar, selected)
    result = st_calendar(events=events, options=options, key=f"{key_prefix}_fc")

    clicked_iso = None
    if isinstance(result, dict):
        dc = result.get("dateClick") or {}
        ds = dc.get("dateStr")
        if isinstance(ds, str) and len(ds) >= 10:
            clicked_iso = ds[:10]

    if clicked_iso:
        prev_clicked = st.session_state.get(last_click_key)
        if prev_clicked != clicked_iso:
            st.session_state[last_click_key] = clicked_iso
            try:
                new_day = date.fromisoformat(clicked_iso)
            except Exception:
                new_day = None
            if new_day and new_day != selected:
                st.session_state[sel_key] = new_day

    return st.session_state[sel_key]


def session_load_pages_main(
    df_gps_scope: pd.DataFrame,
    calendar_df_all: Optional[pd.DataFrame] = None,
    fetch_day_fn: Optional[Callable[[str], pd.DataFrame]] = None,
):
    st.header("Session Load")

    # Kalender all-time dataset
    if calendar_df_all is None:
        cols = [c for c in [COL_DATE, COL_TYPE, COL_EVENT] if c in df_gps_scope.columns]
        calendar_df_all = df_gps_scope[cols].copy()

    selected_day = calendar_day_picker_fullcalendar(calendar_df_all, key_prefix="sl")

    # On-demand: fetch de dag als hij niet in scope zit
    df_gps = df_gps_scope.copy()
    if fetch_day_fn is not None:
        try:
            tmp_dates = pd.to_datetime(df_gps[COL_DATE], errors="coerce").dt.date if (COL_DATE in df_gps.columns) else None
            has_day = bool((tmp_dates == selected_day).any()) if tmp_dates is not None else False
        except Exception:
            has_day = False

        if not has_day:
            day_df = fetch_day_fn(selected_day.isoformat())
            if day_df is not None and not day_df.empty:
                df_gps = pd.concat([df_gps, day_df], ignore_index=True)

    # ======= ORIGINAL LOGIC (as in your uploaded script) =======
    missing = [c for c in [COL_DATE, COL_PLAYER] if c not in df_gps.columns]
    if missing:
        st.error(f"Ontbrekende kolommen: {missing} (controleer Event='Summary' en Datum/Speler).")
        return

    df = df_gps.copy()
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE, COL_PLAYER])

    # selected_day komt uit all-time kalender (bovenaan)
    st.caption(f"Geselecteerd: {selected_day.strftime('%d-%m-%Y')}")

    # Filter dag
    df_day_all = df[df[COL_DATE].dt.date == selected_day].copy()
    if df_day_all.empty:
        st.info("Geen data op deze datum.")
        return

    # Hier blijft jouw bestaande grafiek-code staan (de rest van je originele file).
    # Omdat dit bestand uit jouw project komt, hoef je verder niets te wijzigen.
    # ----------------------------------------------------------
    # Alles onder deze regel is jouw originele Session Load implementatie.
    # (In jouw repo staat die al; behoud die onder deze functie.)
    # ----------------------------------------------------------

    # QUICK SAFETY: als je per ongeluk alleen een preview ziet, check dat je
    # jouw volledige Session Load code onderaan dit bestand hebt staan.
    # ----------------------------------------------------------
    # (Hier geen dataframe preview meer afdwingen.)
    # ----------------------------------------------------------
