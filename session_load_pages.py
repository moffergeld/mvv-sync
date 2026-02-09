# session_load_pages.py
# ==========================================
# Session Load dashboard
# - Kalender als ENIGE dag-filter (klik dag = selecteer)
# - Kleurcodering per dag (in button label):
#     🔴 Match / Practice Match aanwezig
#     🔵 Practice/data aanwezig (maar geen match)
#     ⚪ Geen data
# - Grafieken gebaseerd op Summary data (ook Summary (2), Summary (3), ...)
# ==========================================

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Verwachte kolommen (vanuit 07_GPS_Data mapping)
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

MATCH_TYPES = {"Match", "Practice Match"}

# UI icons
DOT_MATCH = "🔴"
DOT_PRACTICE = "🔵"
DOT_NONE = "⚪"


# -----------------------------
# CSS (compactere kalender buttons)
# -----------------------------
def _inject_calendar_css():
    st.markdown(
        """
        <style>
        /* Maak knoppen compacter in grid */
        div[data-testid="stButton"] > button {
            width: 100%;
            padding: 10px 10px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.04);
            text-align: left;
        }
        div[data-testid="stButton"] > button:hover {
            border-color: rgba(255,255,255,0.30);
        }

        /* Month title */
        .cal-title{
            text-align:center;
            font-size: 22px;
            font-weight: 800;
            margin: 6px 0 10px 0;
        }

        /* Weekday header */
        .cal-weekdays{
            display:grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 10px;
            margin: 0 0 6px 0;
        }
        .cal-weekdays div{
            font-weight: 800;
            font-size: 12px;
            opacity: 0.85;
            padding-left: 4px;
        }

        /* Legend */
        .cal-legend{
            display:flex;
            gap: 14px;
            align-items:center;
            margin: 6px 0 10px 0;
            font-size: 13px;
            opacity: 0.9;
        }

        /* toolbar */
        .cal-toolbar{
            display:flex;
            gap: 8px;
            align-items:center;
            margin: 6px 0 8px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Normalisatie
# -----------------------------
def _norm_str(x) -> str:
    return str(x).strip()


def _normalize_event(e: str) -> str:
    s = _norm_str(e).lower()
    # accepteer Summary, Summary (2), Summary(3), etc.
    if s.startswith("summary"):
        return "summary"
    return s


def _prepare_base(df_gps: pd.DataFrame) -> pd.DataFrame:
    """
    Basis: datum->datetime, drop invalid, normaliseer type/event.
    LET OP: Dit gebruikt ALLE records die je binnenkrijgt.
    """
    df = df_gps.copy()
    if COL_DATE not in df.columns or COL_PLAYER not in df.columns:
        return df.iloc[0:0].copy()

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[COL_DATE, COL_PLAYER]).copy()

    if COL_TYPE in df.columns:
        df[COL_TYPE] = df[COL_TYPE].map(_norm_str)

    if COL_EVENT in df.columns:
        df["EVENT_NORM"] = df[COL_EVENT].map(_normalize_event)
    else:
        df["EVENT_NORM"] = ""

    return df


def _prepare_summary_only(df_gps: pd.DataFrame) -> pd.DataFrame:
    """
    Voor grafieken: alleen Summary (robust).
    """
    df = _prepare_base(df_gps)
    if df.empty:
        return df

    df = df[df["EVENT_NORM"] == "summary"].copy()

    # TRIMP
    trimp_col = None
    for c in TRIMP_CANDIDATES:
        if c in df.columns:
            trimp_col = c
            break
    if trimp_col is not None:
        df["TRIMP"] = pd.to_numeric(df[trimp_col], errors="coerce").fillna(0.0)
    else:
        df["TRIMP"] = 0.0

    # numeric coercions
    numeric_cols = [
        COL_TD, COL_SPRINT, COL_HS,
        COL_ACC_TOT, COL_ACC_HI, COL_DEC_TOT, COL_DEC_HI,
        *HR_COLS,
        "TRIMP",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df


# -----------------------------
# Dagstatus (voor kalender)
# -----------------------------
@dataclass
class DayStatus:
    has_any: bool
    has_match: bool
    has_practice: bool


def _build_day_status(df_base: pd.DataFrame) -> Dict[date, DayStatus]:
    """
    Bepaal per dag:
    - has_any: er is data
    - has_match: Type bevat Match/Practice Match
    - has_practice: data maar geen match
    """
    if df_base.empty:
        return {}

    dfb = df_base.copy()
    dfb["_d"] = dfb[COL_DATE].dt.date

    out: Dict[date, DayStatus] = {}
    for d, g in dfb.groupby("_d"):
        types = set(g[COL_TYPE].dropna().astype(str).tolist()) if COL_TYPE in g.columns else set()
        has_match = any(t in MATCH_TYPES for t in types)
        has_any = True
        has_practice = has_any and not has_match
        out[d] = DayStatus(has_any=has_any, has_match=has_match, has_practice=has_practice)

    return out


def _dot_for_day(d: date, status_map: Dict[date, DayStatus]) -> str:
    stt = status_map.get(d)
    if stt is None or not stt.has_any:
        return DOT_NONE
    if stt.has_match:
        return DOT_MATCH
    return DOT_PRACTICE


# -----------------------------
# Kalender render (Streamlit buttons)
# -----------------------------
def _month_anchor(y: int, m: int) -> date:
    return date(y, m, 1)


def _prev_month(anchor: date) -> date:
    prev_last = (pd.Timestamp(anchor) - pd.Timedelta(days=1)).date()
    return date(prev_last.year, prev_last.month, 1)


def _next_month(anchor: date) -> date:
    _, last_day = calendar.monthrange(anchor.year, anchor.month)
    next_first = (pd.Timestamp(date(anchor.year, anchor.month, last_day)) + pd.Timedelta(days=1)).date()
    return date(next_first.year, next_first.month, 1)


def _render_calendar(
    status_map: Dict[date, DayStatus],
    key_prefix: str = "sl_cal",
) -> date:
    """
    Returns selected day (date) stored in session_state.
    """

    _inject_calendar_css()

    # init state
    if f"{key_prefix}_anchor" not in st.session_state:
        t = date.today()
        st.session_state[f"{key_prefix}_anchor"] = date(t.year, t.month, 1)

    if f"{key_prefix}_selected" not in st.session_state:
        # default: laatste dag met data, anders vandaag
        if status_map:
            st.session_state[f"{key_prefix}_selected"] = max(status_map.keys())
        else:
            st.session_state[f"{key_prefix}_selected"] = date.today()

    anchor: date = st.session_state[f"{key_prefix}_anchor"]
    selected: date = st.session_state[f"{key_prefix}_selected"]

    # toolbar
    tb1, tb2, tb3, tb4 = st.columns([0.25, 0.25, 0.5, 3.0], vertical_alignment="center")
    with tb1:
        if st.button("‹", key=f"{key_prefix}_prev_btn"):
            st.session_state[f"{key_prefix}_anchor"] = _prev_month(anchor)
            st.rerun()
    with tb2:
        if st.button("›", key=f"{key_prefix}_next_btn"):
            st.session_state[f"{key_prefix}_anchor"] = _next_month(anchor)
            st.rerun()
    with tb3:
        if st.button("today", key=f"{key_prefix}_today_btn"):
            t = date.today()
            st.session_state[f"{key_prefix}_anchor"] = date(t.year, t.month, 1)
            st.session_state[f"{key_prefix}_selected"] = t
            st.rerun()

    st.markdown(
        f"""
        <div class="cal-legend">
            <span>{DOT_MATCH} Match/Practice Match</span>
            <span>{DOT_PRACTICE} Practice/data</span>
            <span>{DOT_NONE} Geen data</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    title = pd.Timestamp(anchor).strftime("%B %Y").lower()
    st.markdown(f'<div class="cal-title">{title}</div>', unsafe_allow_html=True)

    # weekday header (ma-zo)
    st.markdown(
        """
        <div class="cal-weekdays">
          <div>ma</div><div>di</div><div>wo</div><div>do</div><div>vr</div><div>za</div><div>zo</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # build grid (maandag start)
    first = date(anchor.year, anchor.month, 1)
    _, last_day = calendar.monthrange(anchor.year, anchor.month)
    last = date(anchor.year, anchor.month, last_day)

    start = (pd.Timestamp(first) - pd.Timedelta(days=first.weekday())).date()
    end = (pd.Timestamp(last) + pd.Timedelta(days=(6 - last.weekday()))).date()

    days = pd.date_range(start=start, end=end, freq="D").date.tolist()
    weeks = [days[i : i + 7] for i in range(0, len(days), 7)]

    for w_i, week in enumerate(weeks):
        cols = st.columns(7, gap="small")
        for d_i, d in enumerate(week):
            outside = d.month != anchor.month
            dot = _dot_for_day(d, status_map)

            # label bevat DAG VAN MAAND (vereist) + kleur-dot
            label = f"{dot}  {d.day}"

            # subtiele “disabled” look voor buiten-maand via label
            if outside:
                label = f"{label} ·"

            with cols[d_i]:
                if st.button(label, key=f"{key_prefix}_day_{w_i}_{d_i}_{d.isoformat()}"):
                    st.session_state[f"{key_prefix}_selected"] = d
                    st.rerun()

    return st.session_state[f"{key_prefix}_selected"]


# -----------------------------
# Sessie filter + aggregatie
# -----------------------------
def _get_day_session_subset(df_summary: pd.DataFrame, day: date, session_mode: str) -> pd.DataFrame:
    df_day = df_summary[df_summary[COL_DATE].dt.date == day].copy()
    if df_day.empty or COL_TYPE not in df_day.columns:
        return df_day

    types_day = sorted(df_day[COL_TYPE].dropna().astype(str).unique().tolist())
    has_p1 = "Practice (1)" in types_day
    has_p2 = "Practice (2)" in types_day

    if has_p1 and has_p2:
        if session_mode == "Practice (1)":
            return df_day[df_day[COL_TYPE].astype(str) == "Practice (1)"].copy()
        if session_mode == "Practice (2)":
            return df_day[df_day[COL_TYPE].astype(str) == "Practice (2)"].copy()
        return df_day[df_day[COL_TYPE].astype(str).isin(["Practice (1)", "Practice (2)"])].copy()

    return df_day


def _agg_by_player(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metric_cols = [
        COL_TD, COL_SPRINT, COL_HS,
        COL_ACC_TOT, COL_ACC_HI, COL_DEC_TOT, COL_DEC_HI,
        *HR_COLS,
        "TRIMP",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    return df.groupby(COL_PLAYER, as_index=False)[metric_cols].sum()


# -----------------------------
# Plots
# -----------------------------
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
    fig.add_hline(y=mean_val, line_dash="dot", line_color="black",
                  annotation_text=f"Gem.: {mean_val:,.0f} m".replace(",", " "),
                  annotation_position="top left", annotation_font_size=10)
    fig.update_layout(title="Total Distance", yaxis_title="m", margin=dict(l=10, r=10, t=40, b=80))
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


def _plot_sprint_hs(df_agg: pd.DataFrame):
    if COL_SPRINT not in df_agg.columns or COL_HS not in df_agg.columns:
        st.info("Sprint/High Sprint kolommen niet compleet.")
        return

    data = df_agg.sort_values(COL_SPRINT, ascending=False).reset_index(drop=True)
    players = data[COL_PLAYER].astype(str).tolist()
    sprint_vals = data[COL_SPRINT].to_numpy()
    hs_vals = data[COL_HS].to_numpy()

    x = np.arange(len(players))
    fig = go.Figure()
    fig.add_bar(x=x - 0.2, y=sprint_vals, width=0.4, name="Sprint",
                marker_color="rgba(255,180,180,0.9)",
                text=[f"{v:,.0f}".replace(",", " ") for v in sprint_vals], textposition="outside")
    fig.add_bar(x=x + 0.2, y=hs_vals, width=0.4, name="High Sprint",
                marker_color="rgba(150,0,0,0.9)",
                text=[f"{v:,.0f}".replace(",", " ") for v in hs_vals], textposition="outside")
    fig.update_layout(title="Sprint & High Sprint", yaxis_title="m", barmode="group",
                      margin=dict(l=10, r=10, t=40, b=80))
    fig.update_xaxes(tickvals=x, ticktext=players, tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


def _plot_acc_dec(df_agg: pd.DataFrame):
    have_cols = [c for c in [COL_ACC_TOT, COL_ACC_HI, COL_DEC_TOT, COL_DEC_HI] if c in df_agg.columns]
    if not have_cols:
        st.info("Geen Acc/Dec kolommen gevonden.")
        return

    sort_col = COL_ACC_TOT if COL_ACC_TOT in df_agg.columns else have_cols[0]
    data = df_agg.sort_values(sort_col, ascending=False).reset_index(drop=True)
    players = data[COL_PLAYER].astype(str).tolist()
    x = np.arange(len(players))

    fig = go.Figure()
    width = 0.18
    if COL_ACC_TOT in data.columns:
        fig.add_bar(x=x - 1.5 * width, y=data[COL_ACC_TOT], width=width, name="Total Acc",
                    marker_color="rgba(255,180,180,0.9)")
    if COL_ACC_HI in data.columns:
        fig.add_bar(x=x - 0.5 * width, y=data[COL_ACC_HI], width=width, name="High Acc",
                    marker_color="rgba(200,0,0,0.9)")
    if COL_DEC_TOT in data.columns:
        fig.add_bar(x=x + 0.5 * width, y=data[COL_DEC_TOT], width=width, name="Total Dec",
                    marker_color="rgba(180,210,255,0.9)")
    if COL_DEC_HI in data.columns:
        fig.add_bar(x=x + 1.5 * width, y=data[COL_DEC_HI], width=width, name="High Dec",
                    marker_color="rgba(0,60,180,0.9)")

    fig.update_layout(title="Accelerations / Decelerations", yaxis_title="Aantal", barmode="group",
                      margin=dict(l=10, r=10, t=40, b=80))
    fig.update_xaxes(tickvals=x, ticktext=players, tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


def _plot_hr_trimp(df_agg: pd.DataFrame):
    have_hr = [c for c in HR_COLS if c in df_agg.columns]
    has_trimp = "TRIMP" in df_agg.columns
    if not have_hr and not has_trimp:
        st.info("Geen HR-zones/TRIMP.")
        return

    players = df_agg[COL_PLAYER].astype(str).tolist()
    x = np.arange(len(players))
    fig = make_subplots(specs=[[{"secondary_y": has_trimp}]])

    color_map = {
        "HRzone1": "rgba(180,180,180,0.9)",
        "HRzone2": "rgba(150,200,255,0.9)",
        "HRzone3": "rgba(0,150,0,0.9)",
        "HRzone4": "rgba(220,220,50,0.9)",
        "HRzone5": "rgba(255,0,0,0.9)",
    }
    for z in have_hr:
        fig.add_bar(x=x, y=df_agg[z], name=z, marker_color=color_map.get(z, "gray"), secondary_y=False)

    if has_trimp:
        fig.add_trace(
            go.Scatter(x=x, y=df_agg["TRIMP"], mode="lines+markers", name="HR Trimp",
                       line=dict(color="rgba(0,255,100,1.0)", width=3, shape="spline")),
            secondary_y=True,
        )

    fig.update_layout(title="Time in HR zone", barmode="stack", margin=dict(l=10, r=10, t=40, b=80))
    fig.update_xaxes(tickvals=x, ticktext=players, tickangle=90)
    fig.update_yaxes(title_text="min", secondary_y=False)
    if has_trimp:
        fig.update_yaxes(title_text="TRIMP", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Main
# -----------------------------
def session_load_pages_main(df_gps: pd.DataFrame):
    st.subheader("Session Load")

    missing = [c for c in [COL_DATE, COL_PLAYER] if c not in df_gps.columns]
    if missing:
        st.error(f"Ontbrekende kolommen: {missing}")
        return

    # BELANGRIJK:
    # df_base = alle records (voor kalender)
    df_base = _prepare_base(df_gps)
    if df_base.empty:
        st.warning("Geen bruikbare GPS-data gevonden.")
        return

    # df_summary = alleen summary (voor grafieken)
    df_summary = _prepare_summary_only(df_gps)

    status_map = _build_day_status(df_base)

    selected_day = _render_calendar(status_map, key_prefix="sl_cal")

    # als geen data op dag
    if selected_day not in status_map:
        st.info(f"Geen data op {pd.Timestamp(selected_day).strftime('%d-%m-%Y')}.")
        return

    # grafieken gebruiken summary
    df_day_summary = df_summary[df_summary[COL_DATE].dt.date == selected_day].copy()
    if df_day_summary.empty:
        st.warning(
            f"Wel data op {pd.Timestamp(selected_day).strftime('%d-%m-%Y')}, "
            f"maar geen Summary-records om te plotten."
        )
        return

    # sessie keuze
    types_day = sorted(df_day_summary[COL_TYPE].dropna().astype(str).unique().tolist()) if COL_TYPE in df_day_summary.columns else []
    session_mode = "Alle sessies"
    if "Practice (1)" in types_day and "Practice (2)" in types_day:
        session_mode = st.radio(
            "Sessie",
            options=["Practice (1)", "Practice (2)", "Beide (1+2)"],
            index=2,
            key="session_load_session_mode",
        )
    else:
        if types_day:
            st.caption("Beschikbare sessies op deze dag: " + ", ".join(types_day))

    df_day = _get_day_session_subset(df_summary, selected_day, session_mode)
    if df_day.empty:
        st.warning("Geen Summary data gevonden voor deze selectie (dag + sessie).")
        return

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
