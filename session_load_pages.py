# session_load_pages.py
# ==========================================
# Session Load dashboard
# - Kalender als enige dag-filter
# - Kleurcodering per dag:
#     * Rood  = Match / Practice Match aanwezig
#     * Blauw = Practice/data aanwezig (maar geen match)
#     * Grijs = geen data
# - Klik dag = selecteer dag
# - 4 grafieken per speler (op Summary data)
# ==========================================

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Kolomnamen (verwacht vanuit pages/07_GPS_Data.py mapping)
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

# Type mapping
MATCH_TYPES = {"Match", "Practice Match"}

# Kleuren (pas aan als je wil)
MVV_RED = "#FF0033"
LIGHT_BLUE = "#2F80ED"
NO_DATA_GRAY = "rgba(255,255,255,0.10)"


# -----------------------------
# Helpers: normalisatie
# -----------------------------
def _norm_str(x) -> str:
    return str(x).strip()


def _normalize_type(t: str) -> str:
    return _norm_str(t)


def _normalize_event(e: str) -> str:
    """
    Belangrijk: accepteer ook 'Summary (2)' etc.
    """
    s = _norm_str(e).lower()
    if s.startswith("summary"):
        return "summary"
    return s


def _prepare_base(df_gps: pd.DataFrame) -> pd.DataFrame:
    """
    Basis prep:
    - Datum naar datetime
    - Drop rows zonder Datum/Speler
    - Type/Event normaliseren
    """
    df = df_gps.copy()

    if COL_DATE not in df.columns or COL_PLAYER not in df.columns:
        return df.iloc[0:0].copy()

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[COL_DATE, COL_PLAYER]).copy()

    if COL_TYPE in df.columns:
        df[COL_TYPE] = df[COL_TYPE].map(_normalize_type)

    if COL_EVENT in df.columns:
        df["EVENT_NORM"] = df[COL_EVENT].map(_normalize_event)
    else:
        df["EVENT_NORM"] = ""

    return df


def _prepare_summary_only(df_gps: pd.DataFrame) -> pd.DataFrame:
    """
    Voor grafieken: alleen Summary (maar robuust: startswith('summary'))
    + numeric coercions
    """
    df = _prepare_base(df_gps)
    if df.empty:
        return df

    df = df[df["EVENT_NORM"] == "summary"].copy()

    # TRIMP alias → 'TRIMP'
    trimp_col = None
    for c in TRIMP_CANDIDATES:
        if c in df.columns:
            trimp_col = c
            break
    if trimp_col is not None:
        df["TRIMP"] = pd.to_numeric(df[trimp_col], errors="coerce").fillna(0.0)
    else:
        df["TRIMP"] = 0.0

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
# Kalender status per dag
# -----------------------------
@dataclass
class DayStatus:
    has_any: bool
    has_match: bool
    has_practice: bool  # alles wat geen match is maar wel data


def _build_day_status(df_base: pd.DataFrame) -> Dict[date, DayStatus]:
    """
    Status wordt bepaald op ALLE records (dus niet alleen Summary).
    """
    if df_base.empty:
        return {}

    dfb = df_base.copy()
    dfb["_d"] = dfb[COL_DATE].dt.date

    out: Dict[date, DayStatus] = {}

    for d, g in dfb.groupby("_d"):
        types = set()
        if COL_TYPE in g.columns:
            types = set(g[COL_TYPE].dropna().astype(str).tolist())

        has_match = any(t in MATCH_TYPES for t in types)
        has_any = True
        has_practice = (not has_match)  # als je later extra type-logica wil: pas dit aan

        out[d] = DayStatus(
            has_any=has_any,
            has_match=has_match,
            has_practice=has_practice,
        )

    return out


# -----------------------------
# Kalender UI (maand grid)
# -----------------------------
def _inject_calendar_css():
    st.markdown(
        f"""
        <style>
        .cal-toolbar {{
            display:flex; align-items:center; gap:8px; margin: 6px 0 10px 0;
        }}
        .cal-legend {{
            display:flex; gap:14px; align-items:center; margin: 6px 0 10px 0;
            font-size: 13px; opacity: 0.9;
        }}
        .cal-dot {{
            width:10px; height:10px; border-radius:3px; display:inline-block; margin-right:6px;
        }}
        .cal-title {{
            text-align:center; font-size: 24px; font-weight: 700; margin: 6px 0 10px 0;
        }}
        .cal-grid {{
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 12px;
            overflow: hidden;
        }}
        .cal-row {{
            display:grid;
            grid-template-columns: repeat(7, 1fr);
        }}
        .cal-head {{
            background: rgba(255,255,255,0.04);
            font-weight: 700;
            font-size: 13px;
        }}
        .cal-cell {{
            position: relative;
            border-top: 1px solid rgba(255,255,255,0.12);
            border-right: 1px solid rgba(255,255,255,0.12);
            min-height: 56px;
            padding: 8px;
        }}
        .cal-cell:last-child {{ border-right: none; }}
        .cal-daynum {{
            position:absolute; top:6px; right:8px;
            font-size: 14px; font-weight: 700;
            opacity: 0.95;
        }}
        .cal-outside {{
            opacity: 0.35;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _calendar_month_grid(
    month_anchor: date,
    day_status: Dict[date, DayStatus],
    selected_day: date,
    key_prefix: str = "cal",
) -> Tuple[date, date]:
    """
    Render 1 maand (ma-zo).
    Buttons in cell = klik dag.
    Returns: (new_month_anchor, new_selected_day)
    """
    _inject_calendar_css()

    y, m = month_anchor.year, month_anchor.month
    first_of_month = date(y, m, 1)
    _, last_day = calendar.monthrange(y, m)
    last_of_month = date(y, m, last_day)

    # toolbar
    c1, c2, c3, c4, c5 = st.columns([0.35, 0.35, 0.6, 1.8, 0.9], vertical_alignment="center")
    with c1:
        if st.button("‹", key=f"{key_prefix}_prev"):
            # vorige maand
            prev_m = (first_of_month.replace(day=1) - pd.Timedelta(days=1)).date()
            return date(prev_m.year, prev_m.month, 1), selected_day
    with c2:
        if st.button("›", key=f"{key_prefix}_next"):
            # volgende maand
            next_m = (last_of_month + pd.Timedelta(days=1)).date()
            return date(next_m.year, next_m.month, 1), selected_day
    with c3:
        if st.button("today", key=f"{key_prefix}_today"):
            today = date.today()
            return date(today.year, today.month, 1), today

    # legend
    st.markdown(
        f"""
        <div class="cal-legend">
            <span><span class="cal-dot" style="background:{MVV_RED};"></span>Match/Practice Match</span>
            <span><span class="cal-dot" style="background:{LIGHT_BLUE};"></span>Practice/data</span>
            <span><span class="cal-dot" style="background:{NO_DATA_GRAY}; border:1px solid rgba(255,255,255,0.18);"></span>Geen data</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # title
    month_name = pd.Timestamp(first_of_month).strftime("%B %Y").lower()
    st.markdown(f'<div class="cal-title">{month_name}</div>', unsafe_allow_html=True)

    # build grid range (start on Monday)
    start = first_of_month - pd.Timedelta(days=(first_of_month.weekday())).to_pytimedelta()
    end = last_of_month + pd.Timedelta(days=(6 - last_of_month.weekday())).to_pytimedelta()

    days = pd.date_range(start=start, end=end, freq="D").to_pydatetime().tolist()
    weeks = [days[i : i + 7] for i in range(0, len(days), 7)]

    # header
    st.markdown('<div class="cal-grid">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="cal-row cal-head">
            <div class="cal-cell">ma</div>
            <div class="cal-cell">di</div>
            <div class="cal-cell">wo</div>
            <div class="cal-cell">do</div>
            <div class="cal-cell">vr</div>
            <div class="cal-cell">za</div>
            <div class="cal-cell">zo</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # weeks
    for w_i, week in enumerate(weeks):
        cols = st.columns(7, gap="small")
        for d_i, dt in enumerate(week):
            d = dt.date()
            outside = (d.month != m)

            stt = day_status.get(d, DayStatus(False, False, False))
            if not stt.has_any:
                bg = NO_DATA_GRAY
            elif stt.has_match:
                bg = MVV_RED
            else:
                bg = LIGHT_BLUE

            is_selected = (d == selected_day)
            border = "2px solid rgba(255,255,255,0.85)" if is_selected else "1px solid rgba(255,255,255,0.12)"
            opacity = "0.45" if outside else "1.0"

            with cols[d_i]:
                # clickable day (button)
                label = str(d.day)  # DAG VAN DE MAAND in cel
                if st.button(
                    label,
                    key=f"{key_prefix}_d_{w_i}_{d_i}_{d.isoformat()}",
                    help=d.isoformat(),
                    use_container_width=True,
                ):
                    selected_day = d

                # style button container via markdown overlay (pragmatisch)
                st.markdown(
                    f"""
                    <div style="
                        margin-top:-44px;
                        height:44px;
                        border-radius:10px;
                        background:{bg};
                        border:{border};
                        opacity:{opacity};
                        pointer-events:none;
                    "></div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)
    return month_anchor, selected_day


# -----------------------------
# Data helpers voor plots
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
    st.header("Session Load")

    missing = [c for c in [COL_DATE, COL_PLAYER] if c not in df_gps.columns]
    if missing:
        st.error(f"Ontbrekende kolommen: {missing}")
        return

    # 1) Basis (ALLE records) voor kalenderstatus
    df_base = _prepare_base(df_gps)
    if df_base.empty:
        st.warning("Geen bruikbare GPS-data gevonden.")
        return

    # 2) Summary-only voor grafieken (maar robuust)
    df_summary = _prepare_summary_only(df_gps)

    # kalender state init
    if "sl_month_anchor" not in st.session_state:
        today = date.today()
        st.session_state["sl_month_anchor"] = date(today.year, today.month, 1)
    if "sl_selected_day" not in st.session_state:
        # default: laatste dag met data
        last_day = df_base[COL_DATE].dt.date.max()
        st.session_state["sl_selected_day"] = last_day if isinstance(last_day, date) else date.today()

    day_status = _build_day_status(df_base)

    # render kalender
    new_anchor, new_selected = _calendar_month_grid(
        month_anchor=st.session_state["sl_month_anchor"],
        day_status=day_status,
        selected_day=st.session_state["sl_selected_day"],
        key_prefix="sl_cal",
    )
    st.session_state["sl_month_anchor"] = new_anchor
    st.session_state["sl_selected_day"] = new_selected

    sel_day = st.session_state["sl_selected_day"]

    # melding als geen data (op basis van ALLE records)
    if sel_day not in day_status:
        st.info(f"Geen data op {pd.Timestamp(sel_day).strftime('%d-%m-%Y')}.")
        return

    # Grafieken: als er geen Summary is op die dag, zeg dat expliciet
    df_day_summary = df_summary[df_summary[COL_DATE].dt.date == sel_day].copy()
    if df_day_summary.empty:
        st.warning(f"Wel data op {pd.Timestamp(sel_day).strftime('%d-%m-%Y')}, maar geen Summary-records om te plotten.")
        return

    # sessie keuze (alleen relevant als Type bestaat en Practice(1)/(2) voorkomt)
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

    df_day = _get_day_session_subset(df_summary, sel_day, session_mode)
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
