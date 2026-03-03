# pages/Subscripts/gps_data_session_load_pages.py
from __future__ import annotations

from datetime import date
from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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

# --- kleuren (terug zoals "vroeger" style) ---
MVV_RED = "#FF0033"          # accent / high
LIGHT_RED = "#FF9AA2"        # hoofd-balken (licht rood)
DARK_RED = "#8B0000"         # high sprint / high acc
LIGHT_BLUE = "#BFD9FF"       # optioneel (decel tot)
BLUE = "#1E6BFF"
GRAY_LINE = "rgba(255,255,255,0.55)"

SELECT_ALL_OPT = "— Select all —"


def _normalize_event(e: str) -> str:
    return "summary" if str(e).strip().lower() == "summary" else str(e).strip().lower()


def _prepare_gps(df_gps: pd.DataFrame) -> pd.DataFrame:
    df = df_gps.copy()
    if COL_DATE not in df.columns or COL_PLAYER not in df.columns:
        return df.iloc[0:0].copy()

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE, COL_PLAYER]).copy()

    # Alleen Summary
    if COL_EVENT in df.columns:
        df["_event_norm"] = df[COL_EVENT].map(_normalize_event)
        df = df[df["_event_norm"] == "summary"].copy()

    # TRIMP fallback
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


def _available_days(df_calendar: pd.DataFrame) -> list[date]:
    if df_calendar is None or df_calendar.empty or COL_DATE not in df_calendar.columns:
        return []
    d = df_calendar.copy()
    d[COL_DATE] = pd.to_datetime(d[COL_DATE], errors="coerce").dt.date
    d = d.dropna(subset=[COL_DATE])
    return sorted(set(d[COL_DATE].tolist()))


def _pick_day_dateinput(df_calendar: pd.DataFrame, key_prefix: str = "sl") -> date:
    days = _available_days(df_calendar)
    if not days:
        return st.date_input("Datum", value=date.today(), key=f"{key_prefix}_date_input_empty")

    sel_key = f"{key_prefix}_selected"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = days[-1]  # laatste beschikbare dag

    picked = st.date_input(
        "Datum",
        value=st.session_state[sel_key],
        min_value=days[0],
        max_value=days[-1],
        key=f"{key_prefix}_date_input",
    )
    st.session_state[sel_key] = picked
    return picked


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
        line_color=GRAY_LINE,
        annotation_text=label,
        annotation_position="top left",
        annotation_font_size=10,
    )


def _resolve_select_all(selected: list[str], players_all: list[str]) -> list[str]:
    if any(s == SELECT_ALL_OPT for s in selected):
        return players_all
    return [p for p in selected if p in players_all]


def _team_selection_ui_inline(players_all: list[str]) -> tuple[bool, list[str], list[str]]:
    # Team selectie in expander zodat de pagina minder "geclusterd" is
    with st.expander("Team selectie", expanded=False):
        if "sl_team_sel_on" not in st.session_state:
            st.session_state["sl_team_sel_on"] = False
        if "sl_starters_raw" not in st.session_state:
            st.session_state["sl_starters_raw"] = []
        if "sl_subs_raw" not in st.session_state:
            st.session_state["sl_subs_raw"] = []

        enabled = st.toggle("Team selectie aan", value=st.session_state["sl_team_sel_on"], key="sl_team_sel_on")

        if not enabled:
            st.info("Zet aan om vaste selectie en wissels te definiëren.")
            return False, [], []

        opt_all = [SELECT_ALL_OPT] + players_all

        st.caption("Kies eerst Vaste selectie. Wisselspelers toont daarna alleen de resterende spelers.")
        c1, c2 = st.columns(2, vertical_alignment="top")

        with c1:
            starters_raw = st.multiselect(
                "Vaste selectie",
                options=opt_all,
                default=[p for p in st.session_state["sl_starters_raw"] if p in opt_all],
                key="sl_starters_raw",
            )

        starters_resolved = _resolve_select_all(starters_raw, players_all)
        subs_pool = [p for p in players_all if p not in set(starters_resolved)]
        opt_all_subs = [SELECT_ALL_OPT] + subs_pool

        with c2:
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

def _plot_total_distance(df_agg: pd.DataFrame, groups: dict[str, list[str]] | None):
    if COL_TD not in df_agg.columns:
        st.info("Kolom 'Total Distance' niet gevonden.")
        return

    data = df_agg.sort_values(COL_TD, ascending=False).reset_index(drop=True)
    players = data[COL_PLAYER].astype(str).tolist()
    vals = data[COL_TD].to_numpy()

    fig = go.Figure()
    fig.add_bar(x=players, y=vals, name="Total Distance", marker_color=MVV_RED)

    med_team = _median_safe(vals)
    if med_team is not None:
        _add_median_line(fig, med_team, f"Mediaan (team): {med_team:,.0f} m".replace(",", " "))

    if groups:
        for gname, gplayers in groups.items():
            med = _median_for_players(df_agg, gplayers, COL_TD)
            if med is not None:
                _add_median_line(fig, med, f"Mediaan ({gname}): {med:,.0f} m".replace(",", " "))

    fig.update_layout(title="Total Distance", yaxis_title="m", xaxis_title=None, showlegend=False)
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, width="stretch")


def _plot_sprint_hs(df_agg: pd.DataFrame, groups: dict[str, list[str]] | None):
    if COL_SPRINT not in df_agg.columns or COL_HS not in df_agg.columns:
        st.info("Sprint / High Sprint kolommen niet compleet.")
        return

    data = df_agg.sort_values(COL_SPRINT, ascending=False).reset_index(drop=True)
    players = data[COL_PLAYER].astype(str).tolist()
    sprint_vals = data[COL_SPRINT].to_numpy()
    hs_vals = data[COL_HS].to_numpy()

    x = np.arange(len(players))
    fig = go.Figure()
    fig.add_bar(x=x - 0.2, y=sprint_vals, width=0.4, name="Sprint", marker_color=MVV_RED)
    fig.add_bar(x=x + 0.2, y=hs_vals, width=0.4, name="High Sprint", marker_color=DARK_RED)

    ms = _median_safe(sprint_vals)
    mh = _median_safe(hs_vals)
    if ms is not None:
        _add_median_line(fig, ms, f"Mediaan Sprint (team): {ms:,.0f} m".replace(",", " "))
    if mh is not None:
        _add_median_line(fig, mh, f"Mediaan High Sprint (team): {mh:,.0f} m".replace(",", " "))

    if groups:
        for gname, gplayers in groups.items():
            m1 = _median_for_players(df_agg, gplayers, COL_SPRINT)
            m2 = _median_for_players(df_agg, gplayers, COL_HS)
            if m1 is not None:
                _add_median_line(fig, m1, f"Mediaan Sprint ({gname}): {m1:,.0f} m".replace(",", " "))
            if m2 is not None:
                _add_median_line(fig, m2, f"Mediaan High Sprint ({gname}): {m2:,.0f} m".replace(",", " "))

    fig.update_layout(title="Sprint & High Sprint Distance", yaxis_title="m", xaxis_title=None, barmode="group")
    fig.update_xaxes(tickvals=x, ticktext=players, tickangle=90)
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
    w = 0.18

    fig = go.Figure()
    if COL_ACC_TOT in data.columns:
        fig.add_bar(x=x - 1.5 * w, y=data[COL_ACC_TOT], width=w, name="Total Acc", marker_color=MVV_RED)
    if COL_ACC_HI in data.columns:
        fig.add_bar(x=x - 0.5 * w, y=data[COL_ACC_HI], width=w, name="High Acc", marker_color=LIGHT_RED)
    if COL_DEC_TOT in data.columns:
        fig.add_bar(x=x + 0.5 * w, y=data[COL_DEC_TOT], width=w, name="Total Dec", marker_color=BLUE)
    if COL_DEC_HI in data.columns:
        fig.add_bar(x=x + 1.5 * w, y=data[COL_DEC_HI], width=w, name="High Dec", marker_color=LIGHT_BLUE)

    fig.update_layout(title="Accelerations / Decelerations", yaxis_title="Aantal (N)", xaxis_title=None, barmode="group")
    fig.update_xaxes(tickvals=x, ticktext=players, tickangle=90)
    st.plotly_chart(fig, width="stretch")


def _plot_hr_trimp(df_agg: pd.DataFrame):
    have_hr = [c for c in HR_COLS if c in df_agg.columns]
    has_trimp = "TRIMP" in df_agg.columns
    if not have_hr and not has_trimp:
        st.info("Geen HR zone/TRIMP data gevonden.")
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

    fig.update_layout(title="Time in HR zone", xaxis_title=None, barmode="group", bargap=0.15)
    fig.update_xaxes(tickvals=base_x, ticktext=players, tickangle=90)
    fig.update_yaxes(title_text="Time in HR zone (min)", secondary_y=False)
    if has_trimp:
        fig.update_yaxes(title_text="HR Trimp", secondary_y=True)

    st.plotly_chart(fig, width="stretch")


def session_load_pages_main(
    df_gps_scope: pd.DataFrame,
    calendar_df_all: Optional[pd.DataFrame] = None,
    fetch_day_fn: Optional[Callable[[str], pd.DataFrame]] = None,
):
    st.header("Session Load")

    cal_df = calendar_df_all if calendar_df_all is not None else df_gps_scope

    with st.expander("📅 Kalender", expanded=True):
        selected_day = _pick_day_dateinput(cal_df, key_prefix="sl")

    st.caption(f"Geselecteerd: {selected_day.strftime('%d-%m-%Y')}")

    # On-demand dag ophalen als datum niet in scope zit
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

    df_day = df[df[COL_DATE].dt.date == selected_day].copy()
    if df_day.empty:
        st.info("Geen data op deze datum.")
        return

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
        _plot_total_distance(df_agg, groups)
    with col_top2:
        _plot_sprint_hs(df_agg, groups)

    col_bot1, col_bot2 = st.columns(2)
    with col_bot1:
        _plot_acc_dec(df_agg)
    with col_bot2:
        _plot_hr_trimp(df_agg)


