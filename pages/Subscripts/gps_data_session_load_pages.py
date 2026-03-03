# pages/Subscripts/gps_data_session_load_pages.py
# ============================================================
# Session Load (Streamlit)
#
# FIX:
# - Voorkom rerun-loop door streamlit_calendar:
#   Update session_state alleen als gebruiker ECHT een nieuwe datum klikt.
# - Normaliseer dateStr naar YYYY-MM-DD (ds[:10]).
# ============================================================

from __future__ import annotations

from datetime import date
from typing import Callable

import pandas as pd
import streamlit as st
from streamlit_calendar import calendar as st_calendar

COL_DATE = "Datum"
COL_EVENT = "Event"
COL_TYPE = "Type"

MVV_RED = "#E30613"
PRACTICE_BLUE = "#1B65B9"

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
    # selected background
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
    days_with_data, _ = _compute_day_sets(df_calendar)
    max_day = max(days_with_data) if days_with_data else date.today()

    sel_key = f"{key_prefix}_selected"
    last_click_key = f"{key_prefix}_last_clicked_iso"

    # init selected once
    if sel_key not in st.session_state:
        st.session_state[sel_key] = max_day

    selected: date = st.session_state[sel_key]

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

    # Read click (if any)
    clicked_iso = None
    if isinstance(result, dict):
        dc = result.get("dateClick") or {}
        ds = dc.get("dateStr")
        if isinstance(ds, str) and len(ds) >= 10:
            clicked_iso = ds[:10]  # normalize YYYY-MM-DD

    # Update ONLY if truly new click and new day
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


def _ensure_day_in_scope_df(
    df_scope: pd.DataFrame,
    target_day: date,
    fetch_day_fn: Callable[[str], pd.DataFrame],
) -> pd.DataFrame:
    if df_scope is None:
        df_scope = pd.DataFrame()

    if df_scope.empty:
        return fetch_day_fn(target_day.isoformat())

    if COL_DATE not in df_scope.columns:
        return df_scope

    tmp = df_scope.copy()
    tmp[COL_DATE] = pd.to_datetime(tmp[COL_DATE], errors="coerce").dt.date

    if bool((tmp[COL_DATE] == target_day).any()):
        return tmp

    day_df = fetch_day_fn(target_day.isoformat())
    if day_df is None or day_df.empty:
        return tmp

    out = pd.concat([tmp, day_df], ignore_index=True)
    out[COL_DATE] = pd.to_datetime(out[COL_DATE], errors="coerce").dt.date
    return out


def session_load_pages_main(
    df_gps_scope: pd.DataFrame,
    calendar_df_all: pd.DataFrame,
    fetch_day_fn: Callable[[str], pd.DataFrame],
) -> None:
    st.subheader("Session Load")

    selected_day = calendar_day_picker_fullcalendar(calendar_df_all, key_prefix="sl")

    # on-demand day fetch if needed
    df_all_for_calc = _ensure_day_in_scope_df(df_gps_scope, selected_day, fetch_day_fn)

    if df_all_for_calc is None or df_all_for_calc.empty:
        st.info("Geen GPS Summary data beschikbaar voor deze selectie.")
        return

    df = df_all_for_calc.copy()
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce").dt.date
    df = df[df[COL_DATE] == selected_day].copy()

    if df.empty:
        st.info("Geen data op deze datum.")
        return

    # Hier laat je jouw bestaande Session Load tabellen/grafieken staan.
    # (Als je oorspronkelijke script die al had, plak die hier terug.)
    st.caption(f"Geselecteerde datum: {selected_day.isoformat()}")
    st.dataframe(df, width="stretch", height=520)
