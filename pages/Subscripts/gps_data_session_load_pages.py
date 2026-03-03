# pages/Subscripts/gps_data_session_load_pages.py
from __future__ import annotations

from datetime import date, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    from streamlit_calendar import calendar as st_calendar
except Exception:
    st_calendar = None  # fallback naar date_input

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

SELECT_BG = "rgba(255, 215, 0, 0.55)"
SELECT_BORDER = "rgba(255, 215, 0, 1.0)"

SELECT_ALL_OPT = "— Select all —"


def _normalize_event(e: str) -> str:
    return "summary" if str(e).strip().lower() == "summary" else str(e).strip().lower()


def _is_match_type(t: str) -> bool:
    return "match" in str(t).strip().lower()


def _prepare_gps(df_gps: pd.DataFrame) -> pd.DataFrame:
    df = df_gps.copy()

    if COL_DATE not in df.columns or COL_PLAYER not in df.columns:
        return df.iloc[0:0].copy()

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE, COL_PLAYER]).copy()

    if COL_EVENT in df.columns:
        df["_event_norm"] = df[COL_EVENT].map(_normalize_event)
        df = df[df["_event_norm"] == "summary"].copy()

    trimp_col = None
    for c in TRIMP_CANDIDATES:
        if c in df.columns:
            trimp_col = c
            break
    df["TRIMP"] = pd.to_numeric(df[trimp_col], errors="coerce").fillna(0.0) if trimp_col else 0.0

    numeric_cols = [
        COL_TD, COL_SPRINT, COL_HS,
        COL_ACC_TOT, COL_ACC_HI, COL_DEC_TOT, COL_DEC_HI,
        *HR_COLS, "TRIMP",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in [COL_PLAYER, COL_TYPE]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


def _compute_day_sets(df: pd.DataFrame) -> tuple[set[date], set[date]]:
    if df is None or df.empty or COL_DATE not in df.columns:
        return set(), set()

    d = df.copy()
    d[COL_DATE] = pd.to_datetime(d[COL_DATE], errors="coerce").dt.date
    d = d.dropna(subset=[COL_DATE])

    days_with_data = set(d[COL_DATE].unique().tolist())

    match_days: set[date] = set()
    if COL_TYPE in d.columns:
        mask = d[COL_TYPE].map(_is_match_type)
        match_days = set(d.loc[mask, COL_DATE].unique().tolist())

    return days_with_data, match_days


def _build_calendar_events(df_calendar: pd.DataFrame, selected: date, window_days: int = 180) -> list[dict]:
    """
    Belangrijk: beperk events, anders kan streamlit_calendar leeg/instabiel worden.
    """
    days_with_data, match_days = _compute_day_sets(df_calendar)

    # window
    start_win = selected - timedelta(days=window_days)
    end_win = selected + timedelta(days=7)

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
        if d < start_win or d > end_win:
            continue
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


def _calendar_picker(df_calendar: pd.DataFrame, key_prefix: str = "sl") -> date:
    """
    Robust picker:
    - use streamlit_calendar if available
    - otherwise fallback to st.date_input
    - no rerun loops
    """
    days_with_data, _ = _compute_day_sets(df_calendar)
    max_day = max(days_with_data) if days_with_data else date.today()

    sel_key = f"{key_prefix}_selected"
    last_evt_key = f"{key_prefix}_last_evt"

    if sel_key not in st.session_state:
        st.session_state[sel_key] = max_day
    if last_evt_key not in st.session_state:
        st.session_state[last_evt_key] = None

    selected: date = st.session_state[sel_key]

    # Fallback if component missing
    if st_calendar is None:
        return st.date_input("Datum", value=selected, key=f"{key_prefix}_date_input")

    st.markdown(
        """
        <style>
          .fc { font-size: 13px; }
          .fc .fc-toolbar-title { font-weight: 800; }
          .fc .fc-button { border-radius: 8px; }
          .fc .fc-event { border-radius: 6px; padding: 2px 6px; font-weight: 800; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    options = {
        "initialView": "dayGridMonth",
        "initialDate": selected.isoformat(),
        "height": 740,
        "firstDay": 1,
        "headerToolbar": {"left": "prev,next today", "center": "title", "right": ""},
        "fixedWeekCount": False,
        "dayMaxEvents": True,
        "eventDisplay": "block",
        "selectable": True,
    }

    # Limit events window to keep component stable
    events = _build_calendar_events(df_calendar, selected, window_days=180)

    result = st_calendar(events=events, options=options, key=f"{key_prefix}_fc")

    clicked_iso = None
    fingerprint = None

    if isinstance(result, dict):
        dc = result.get("dateClick") or {}
        ds = dc.get("dateStr")
        if isinstance(ds, str) and len(ds) >= 10:
            clicked_iso = ds[:10]
            fingerprint = f"date:{clicked_iso}"

        ec = result.get("eventClick")
        if isinstance(ec, dict):
            ev = ec.get("event", {}) or {}
            ds2 = ev.get("start")
            if isinstance(ds2, str) and len(ds2) >= 10:
                clicked_iso = ds2[:10]
                fingerprint = f"event:{clicked_iso}:{ev.get('title','')}"

    if clicked_iso and fingerprint:
        if st.session_state[last_evt_key] != fingerprint:
            st.session_state[last_evt_key] = fingerprint
            try:
                new_day = date.fromisoformat(clicked_iso)
            except Exception:
                new_day = None
            if new_day and new_day != selected:
                st.session_state[sel_key] = new_day

    return st.session_state[sel_key]


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


def _agg_by_player(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metric_cols = [
        COL_TD, COL_SPRINT, COL_HS,
        COL_ACC_TOT, COL_ACC_HI, COL_DEC_TOT, COL_DEC_HI,
        *HR_COLS, "TRIMP",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    return df.groupby(COL_PLAYER, as_index=False)[metric_cols].sum()


def _legend_directly_under_title(fig: go.Figure, *, n_items: int) -> None:
    top_margin = 84 if n_items <= 2 else 96 if n_items <= 4 else 112 if n_items <= 6 else 128
    fig.update_layout(
        margin=dict(l=10, r=10, t=top_margin, b=80),
        legend=dict(
            orientation="h",
            xanchor="left",
            x=0.0,
            yanchor="top",
            y=1.085,
            tracegroupgap=8,
        ),
    )


def _median_safe(a: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return None if a.size == 0 else float(np.median(a))


def _median_for_players(df_agg: pd.DataFrame, players: list[str], col: str) -> float | None:
    if df_agg.empty or col not in df_agg.columns:
        return None
    sub = df_agg[df_agg[COL_PLAYER].astype(str).isin(players)]
    if sub.empty:
        return None
    return _median_safe(sub[col].to_numpy())


def _add_median_line(fig: go.Figure, y: float, label: str) -> None:
    fig.add_hline(
        y=y,
        line_dash="dot",
        line_width=2,
        line_color="rgba(255,255,255,0.55)",
        annotation_text=label,
        annotation_position="top left",
        annotation_font_size=10,
    )


def _resolve_select_all(selected: list[str], players_all: list[str]) -> list[str]:
    if any(s == SELECT_ALL_OPT for s in selected):
        return players_all
    return [p for p in selected if p in players_all]


def _team_selection_ui_inline(players_all: list[str]) -> tuple[bool, list[str], list[str]]:
    st.markdown("### Team selectie")

    if "sl_team_sel_on" not in st.session_state:
        st.session_state["sl_team_sel_on"] = False
    if "sl_starters_raw" not in st.session_state:
        st.session_state["sl_starters_raw"] = []
    if "sl_subs_raw" not in st.session_state:
        st.session_state["sl_subs_raw"] = []

    opt_all = [SELECT_ALL_OPT] + players_all

    c1, c2, c3 = st.columns([0.9, 3.0, 3.0], vertical_alignment="center")
    with c1:
        enabled = st.toggle("Aan", value=st.session_state["sl_team_sel_on"], key="sl_team_sel_on")

    if not enabled:
        with c2:
            st.multiselect("Vaste selectie", options=opt_all, default=[], key="sl_starters_raw_disabled")
        with c3:
            st.multiselect("Wisselspelers", options=opt_all, default=[], key="sl_subs_raw_disabled")
        return False, [], []

    with c2:
        starters_raw = st.multiselect(
            "Vaste selectie",
            options=opt_all,
            default=[p for p in st.session_state["sl_starters_raw"] if p in opt_all],
            key="sl_starters_raw",
        )

    starters_resolved = _resolve_select_all(starters_raw, players_all)
    subs_pool = [p for p in players_all if p not in set(starters_resolved)]
    opt_all_subs = [SELECT_ALL_OPT] + subs_pool

    with c3:
        subs_raw = st.multiselect(
            "Wisselspelers",
            options=opt_all_subs,
            default=[p for p in st.session_state["sl_subs_raw"] if p in opt_all_subs],
            key="sl_subs_raw",
        )

    subs_resolved = _resolve_select_all(subs_raw, subs_pool)

    starters_final = starters_resolved
    subs_final = [p for p in subs_resolved if p not in set(starters_final)]

    return True, starters_final, subs_final


def _plot_total_distance(df_agg: pd.DataFrame, *, groups: dict[str, list[str]] | None):
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
        showlegend=True,
    )

    med_team = _median_safe(vals)
    if med_team is not None:
        _add_median_line(fig, med_team, f"Mediaan (team): {med_team:,.0f} m".replace(",", " "))

    if groups:
        for gname, gplayers in groups.items():
            med = _median_for_players(df_agg, gplayers, COL_TD)
            if med is not None:
                _add_median_line(fig, med, f"Mediaan ({gname}): {med:,.0f} m".replace(",", " "))

    fig.update_layout(title="Total Distance", yaxis_title="Total Distance (m)", xaxis_title=None)
    fig.update_xaxes(tickangle=90)
    _legend_directly_under_title(fig, n_items=1)
    st.plotly_chart(fig, width="stretch")


def _plot_sprint_hs(df_agg: pd.DataFrame, *, groups: dict[str, list[str]] | None):
    if COL_SPRINT not in df_agg.columns or COL_HS not in df_agg.columns:
        st.info("Sprint / High Sprint kolommen niet compleet.")
        return

    data = df_agg.sort_values(COL_SPRINT, ascending=False).reset_index(drop=True)
    players = data[COL_PLAYER].astype(str).tolist()
    sprint_vals = data[COL_SPRINT].to_numpy()
    hs_vals = data[COL_HS].to_numpy()

    x = np.arange(len(players))
    fig = go.Figure()
    fig.add_bar(x=x - 0.2, y=sprint_vals, width=0.4, name="Sprint", marker_color="rgba(255,180,180,0.9)")
    fig.add_bar(x=x + 0.2, y=hs_vals, width=0.4, name="High Sprint", marker_color="rgba(150,0,0,0.9)")

    med_s_team = _median_safe(sprint_vals)
    med_h_team = _median_safe(hs_vals)
    if med_s_team is not None:
        _add_median_line(fig, med_s_team, f"Mediaan Sprint (team): {med_s_team:,.0f} m".replace(",", " "))
    if med_h_team is not None:
        _add_median_line(fig, med_h_team, f"Mediaan High Sprint (team): {med_h_team:,.0f} m".replace(",", " "))

    if groups:
        for gname, gplayers in groups.items():
            ms = _median_for_players(df_agg, gplayers, COL_SPRINT)
            mh = _median_for_players(df_agg, gplayers, COL_HS)
            if ms is not None:
                _add_median_line(fig, ms, f"Mediaan Sprint ({gname}): {ms:,.0f} m".replace(",", " "))
            if mh is not None:
                _add_median_line(fig, mh, f"Mediaan High Sprint ({gname}): {mh:,.0f} m".replace(",", " "))

    fig.update_layout(title="Sprint & High Sprint Distance", yaxis_title="Distance (m)", xaxis_title=None, barmode="group")
    fig.update_xaxes(tickvals=x, ticktext=players, tickangle=90)
    _legend_directly_under_title(fig, n_items=2)
    st.plotly_chart(fig, width="stretch")


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
        fig.add_bar(x=x - 1.5 * width, y=data[COL_ACC_TOT], width=width, name="Total Accelerations")
    if COL_ACC_HI in data.columns:
        fig.add_bar(x=x - 0.5 * width, y=data[COL_ACC_HI], width=width, name="High Accelerations")
    if COL_DEC_TOT in data.columns:
        fig.add_bar(x=x + 0.5 * width, y=data[COL_DEC_TOT], width=width, name="Total Decelerations")
    if COL_DEC_HI in data.columns:
        fig.add_bar(x=x + 1.5 * width, y=data[COL_DEC_HI], width=width, name="High Decelerations")

    fig.update_layout(title="Accelerations / Decelerations", yaxis_title="Aantal (N)", xaxis_title=None, barmode="group")
    fig.update_xaxes(tickvals=x, ticktext=players, tickangle=90)
    _legend_directly_under_title(fig, n_items=4)
    st.plotly_chart(fig, width="stretch")


def _plot_hr_trimp(df_agg: pd.DataFrame):
    have_hr = [c for c in HR_COLS if c in df_agg.columns]
    has_trimp = "TRIMP" in df_agg.columns
    if not have_hr and not has_trimp:
        return

    players = df_agg[COL_PLAYER].astype(str).tolist()
    base_x = np.arange(len(players))

    fig = make_subplots(specs=[[{"secondary_y": has_trimp}]])
    if have_hr:
        for z in have_hr:
            fig.add_bar(x=base_x, y=df_agg[z], name=z, secondary_y=False)
    if has_trimp:
        fig.add_trace(go.Scatter(x=base_x, y=df_agg["TRIMP"], mode="lines+markers", name="HR Trimp"), secondary_y=True)

    fig.update_layout(title="Time in HR zone", xaxis_title=None, barmode="group")
    fig.update_xaxes(tickvals=base_x, ticktext=players, tickangle=90)
    fig.update_yaxes(title_text="Time in HR zone (min)", secondary_y=False)
    if has_trimp:
        fig.update_yaxes(title_text="HR Trimp", secondary_y=True)

    _legend_directly_under_title(fig, n_items=(len(have_hr) + (1 if has_trimp else 0)))
    st.plotly_chart(fig, width="stretch")


def session_load_pages_main(
    df_gps_scope: pd.DataFrame,
    calendar_df_all: Optional[pd.DataFrame] = None,
    fetch_day_fn: Optional[Callable[[str], pd.DataFrame]] = None,
):
    st.header("Session Load")

    # kalender dataset
    cal_df = calendar_df_all if calendar_df_all is not None else df_gps_scope

    with st.expander("📅 Kalender", expanded=True):
        selected_day = _calendar_picker(cal_df, key_prefix="sl")

    st.caption(f"Geselecteerd: {selected_day.strftime('%d-%m-%Y')}")

    # on-demand dag ophalen
    df_work = df_gps_scope.copy()
    if fetch_day_fn is not None:
        try:
            tmp = df_work.copy()
            tmp[COL_DATE] = pd.to_datetime(tmp[COL_DATE], errors="coerce")
            has_day = bool((tmp[COL_DATE].dt.date == selected_day).any())
        except Exception:
            has_day = False

        if not has_day:
            day_df = fetch_day_fn(selected_day.isoformat())
            if day_df is not None and not day_df.empty:
                df_work = pd.concat([df_work, day_df], ignore_index=True)

    df = _prepare_gps(df_work)
    if df.empty:
        st.warning("Geen bruikbare GPS-data gevonden.")
        return

    df_day_all = df[df[COL_DATE].dt.date == selected_day].copy()
    if df_day_all.empty:
        st.info("Geen data op deze datum.")
        return

    types_day = sorted(df_day_all[COL_TYPE].dropna().astype(str).unique().tolist()) if COL_TYPE in df_day_all.columns else []
    session_mode = "Beide (1+2)"
    if "Practice (1)" in types_day and "Practice (2)" in types_day:
        session_mode = st.radio("Sessie", ["Practice (1)", "Practice (2)", "Beide (1+2)"], index=2, key="session_load_session_mode")

    df_day = _get_day_session_subset(df, selected_day, session_mode)
    if df_day.empty:
        st.warning("Geen data gevonden voor deze selectie.")
        return

    st.caption("Beschikbare sessie op deze dag: " + (", ".join(types_day) if types_day else "—"))

    df_agg = _agg_by_player(df_day)
    if df_agg.empty:
        st.warning("Geen data om te aggregeren.")
        return

    players_all = sorted(df_agg[COL_PLAYER].astype(str).unique().tolist())
    team_on, starters, subs = _team_selection_ui_inline(players_all)

    groups: dict[str, list[str]] | None = None
    if team_on:
        groups = {}
        if starters:
            groups["Vaste selectie"] = starters
        if subs:
            groups["Wissels"] = subs

    col_top1, col_top2 = st.columns(2)
    with col_top1:
        _plot_total_distance(df_agg, groups=groups)
    with col_top2:
        _plot_sprint_hs(df_agg, groups=groups)

    col_bot1, col_bot2 = st.columns(2)
    with col_bot1:
        _plot_acc_dec(df_agg)
    with col_bot2:
        _plot_hr_trimp(df_agg)
