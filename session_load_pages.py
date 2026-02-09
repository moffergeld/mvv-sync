# session_load_pages.py
# ==========================================
# Session Load dashboard
# - Echte kalender (ma-zo grid) als ENIGE dag-filter
# - Kleuren per dag:
#     * rood  = Match / Practice Match (als er data is)
#     * blauw = Practice/data (als er data is)
#     * grijs = geen data
# - Klik dag in kalender = selecteer dag
# - Navigatie: vorige/volgende maand + today
# - Data-selectie per dag:
#     * Als er "Summary" bestaat op die dag -> gebruik alleen Summary (voorkomt dubbel tellen)
#     * Anders -> gebruik alle events op die dag
# - Kies sessie: Practice (1) / Practice (2) / beide (alleen als beide bestaan op die dag)
# - 4 grafieken per speler:
#   * Total Distance
#   * Sprint & High Sprint
#   * Accelerations / Decelerations
#   * Time in HR zones + HR Trimp
# ==========================================

from __future__ import annotations

import calendar
from datetime import date

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

MATCH_TYPES = {"Match", "Practice Match"}
MVV_RED = "#FF0033"


# -----------------------------
# Helpers (data prep)
# -----------------------------
def _normalize_event(e: str) -> str:
    return str(e).strip().lower()


def _prepare_gps(df_gps: pd.DataFrame) -> pd.DataFrame:
    """
    - Datum naar datetime (tz-naive)
    - Alleen rijen met datum + speler
    - Event normaliseren (maar NIET meer hard filteren op Summary)
    - TRIMP-naam normaliseren naar kolom 'TRIMP'
    - Numerieke kolommen coerces
    """
    df = df_gps.copy()
    if COL_DATE not in df.columns or COL_PLAYER not in df.columns:
        return df.iloc[0:0].copy()

    # datum -> datetime (en tz-naive)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce", dayfirst=True)
    try:
        df[COL_DATE] = df[COL_DATE].dt.tz_localize(None)
    except Exception:
        pass

    df = df.dropna(subset=[COL_DATE, COL_PLAYER]).copy()

    if COL_EVENT in df.columns:
        df["EVENT_NORM"] = df[COL_EVENT].map(_normalize_event)
    else:
        df["EVENT_NORM"] = ""

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

    # Type opschonen
    if COL_TYPE in df.columns:
        df[COL_TYPE] = df[COL_TYPE].astype(str).str.strip()

    return df


def _get_day_session_subset(df_day: pd.DataFrame, session_mode: str) -> pd.DataFrame:
    """Filter binnen een dag op Practice(1)/(2) logica."""
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
# Calendar UI (click day)
# -----------------------------
def _calendar_css() -> None:
    st.markdown(
        f"""
        <style>
        .cal-toolbar {{
            display:flex; align-items:center; gap:10px; margin: 6px 0 10px 0;
        }}
        .cal-legend {{
            display:flex; gap:16px; align-items:center; margin: 6px 0 10px 0;
            font-size: 13px; opacity: 0.9;
        }}
        .cal-dot {{
            width:10px; height:10px; border-radius:3px; display:inline-block; margin-right:6px;
        }}

        /* Button sizing for day-cells */
        .cal-day button {{
            width: 100% !important;
            height: 54px !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            font-weight: 700 !important;
        }}
        .cal-day button:hover {{
            border-color: rgba(255,255,255,0.30) !important;
        }}

        /* Selected day outline */
        .cal-selected button {{
            outline: 2px solid {MVV_RED} !important;
            outline-offset: 0px !important;
        }}

        /* Weekday header */
        .cal-wd {{
            text-align:center;
            font-weight:700;
            opacity:0.9;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 8px;
        }}

        /* Make buttons look cleaner */
        div[data-testid="stButton"] > button {{
            box-shadow: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _add_months(d: date, delta: int) -> date:
    y = d.year + (d.month - 1 + delta) // 12
    m = (d.month - 1 + delta) % 12 + 1
    return date(y, m, 1)


def calendar_day_picker(
    df: pd.DataFrame,
    key_prefix: str = "sl_cal",
) -> date:
    """
    Kalender toont altijd de "view month". Klik op een dag selecteert die dag.
    Kleuren worden bepaald op basis van data in df.
    """
    _calendar_css()

    # beschikbare datums in df
    dts = pd.to_datetime(df[COL_DATE], errors="coerce").dropna()
    if dts.empty:
        raise ValueError("Geen geldige datums in de data.")

    available_days = sorted(pd.Series(dts.dt.date.unique()).tolist())
    min_day, max_day = available_days[0], available_days[-1]

    # init state
    view_key = f"{key_prefix}_view_month"
    sel_key = f"{key_prefix}_selected_day"

    if view_key not in st.session_state:
        st.session_state[view_key] = _month_start(max_day)
    if sel_key not in st.session_state:
        st.session_state[sel_key] = max_day

    view_month: date = st.session_state[view_key]
    selected_day: date = st.session_state[sel_key]

    # toolbar
    cols = st.columns([0.6, 0.6, 1.2, 3.6])
    with cols[0]:
        if st.button("‹", key=f"{key_prefix}_prev", help="Vorige maand"):
            st.session_state[view_key] = _add_months(view_month, -1)
            st.rerun()
    with cols[1]:
        if st.button("›", key=f"{key_prefix}_next", help="Volgende maand"):
            st.session_state[view_key] = _add_months(view_month, +1)
            st.rerun()
    with cols[2]:
        if st.button("today", key=f"{key_prefix}_today", help="Ga naar huidige maand/dag"):
            today = date.today()
            st.session_state[view_key] = _month_start(today)
            st.session_state[sel_key] = today
            st.rerun()

    # legend + title
    st.markdown(
        f"""
        <div class="cal-legend">
            <span><span class="cal-dot" style="background:{MVV_RED}"></span>Match/Practice Match</span>
            <span><span class="cal-dot" style="background:#3B82F6"></span>Practice/data</span>
            <span><span class="cal-dot" style="background:#6B7280"></span>Geen data</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    month_title = pd.Timestamp(view_month).strftime("%B %Y")
    st.markdown(f"<h3 style='text-align:center; margin: 6px 0 12px 0;'>{month_title}</h3>", unsafe_allow_html=True)

    # build status per day (within whole df)
    # status: "match" | "data" | "none"
    day_type = {}
    if COL_TYPE in df.columns:
        g = df.groupby(df[COL_DATE].dt.date)[COL_TYPE].apply(lambda s: set(map(str, s.dropna().tolist())))
        for d0, types in g.items():
            if any(t in MATCH_TYPES for t in types):
                day_type[d0] = "match"
            else:
                day_type[d0] = "data"
    else:
        for d0 in available_days:
            day_type[d0] = "data"

    # generate 6-week calendar grid (Mon..Sun)
    cal = calendar.Calendar(firstweekday=0)  # Monday
    month_days = list(cal.itermonthdates(view_month.year, view_month.month))

    # weekday header
    wd_cols = st.columns(7)
    labels = ["ma", "di", "wo", "do", "vr", "za", "zo"]
    for i, lab in enumerate(labels):
        with wd_cols[i]:
            st.markdown(f"<div class='cal-wd'>{lab}</div>", unsafe_allow_html=True)

    # per-day button styling via key classes
    style_lines = []
    for d0 in month_days:
        status = day_type.get(d0, "none")
        if status == "match":
            bg = "rgba(255,0,51,0.35)"
            bd = "rgba(255,0,51,0.55)"
        elif status == "data":
            bg = "rgba(59,130,246,0.25)"
            bd = "rgba(59,130,246,0.45)"
        else:
            bg = "rgba(107,114,128,0.18)"
            bd = "rgba(255,255,255,0.08)"

        k = f"{key_prefix}_day_{d0.isoformat()}"
        style_lines.append(
            f""".st-key-{k} .cal-day button{{background:{bg} !important; border-color:{bd} !important;}}"""
        )

    st.markdown("<style>" + "\n".join(style_lines) + "</style>", unsafe_allow_html=True)

    # render weeks
    for week_i in range(0, len(month_days), 7):
        week = month_days[week_i : week_i + 7]
        cols7 = st.columns(7)
        for col_i, d0 in enumerate(week):
            k = f"{key_prefix}_day_{d0.isoformat()}"
            in_month = (d0.month == view_month.month)

            # disable clicks outside month OR outside available range (optional)
            disabled = (not in_month)

            with cols7[col_i]:
                outer_class = "cal-selected" if d0 == selected_day else ""
                st.markdown(f"<div class='cal-day {outer_class}'>", unsafe_allow_html=True)
                if st.button(
                    str(d0.day),  # dag van de maand zichtbaar
                    key=k,
                    disabled=disabled,
                ):
                    st.session_state[sel_key] = d0
                    # als je op een dag in deze maand klikt: view month blijft gelijk
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    # clamp selected_day binnen min/max (als needed)
    selected_day = st.session_state[sel_key]
    if selected_day < min_day:
        selected_day = min_day
        st.session_state[sel_key] = selected_day
    if selected_day > max_day:
        selected_day = max_day
        st.session_state[sel_key] = selected_day

    return selected_day


# -----------------------------
# Plot helpers (4 grafieken)
# -----------------------------
def _plot_total_distance(df_agg: pd.DataFrame):
    if COL_TD not in df_agg.columns:
        st.info("Kolom 'Total Distance' niet gevonden in de data.")
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

    mean_val = float(np.nanmean(vals)) if len(vals) > 0 else 0.0
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
        st.info("Sprint / High Sprint kolommen niet compleet in de data.")
        return

    data = df_agg.sort_values(COL_SPRINT, ascending=False).reset_index(drop=True)
    players = data[COL_PLAYER].astype(str).tolist()
    sprint_vals = data[COL_SPRINT].to_numpy()
    hs_vals = data[COL_HS].to_numpy()

    fig = go.Figure()
    x = np.arange(len(players))

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
    if len(have_cols) == 0:
        st.info("Geen Acceleration/Deceleration kolommen gevonden.")
        return

    sort_col = COL_ACC_TOT if COL_ACC_TOT in df_agg.columns else have_cols[0]
    data = df_agg.sort_values(sort_col, ascending=False).reset_index(drop=True)
    players = data[COL_PLAYER].astype(str).tolist()
    x = np.arange(len(players))

    fig = go.Figure()
    width = 0.18

    if COL_ACC_TOT in data.columns:
        fig.add_bar(x=x - 1.5 * width, y=data[COL_ACC_TOT], width=width, name="Total Accelerations",
                    marker_color="rgba(255,180,180,0.9)")
    if COL_ACC_HI in data.columns:
        fig.add_bar(x=x - 0.5 * width, y=data[COL_ACC_HI], width=width, name="High Accelerations",
                    marker_color="rgba(200,0,0,0.9)")
    if COL_DEC_TOT in data.columns:
        fig.add_bar(x=x + 0.5 * width, y=data[COL_DEC_TOT], width=width, name="Total Decelerations",
                    marker_color="rgba(180,210,255,0.9)")
    if COL_DEC_HI in data.columns:
        fig.add_bar(x=x + 1.5 * width, y=data[COL_DEC_HI], width=width, name="High Decelerations",
                    marker_color="rgba(0,60,180,0.9)")

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
        st.info("Geen HR-zone kolommen of TRIMP-kolom gevonden.")
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
            go.Scatter(
                x=x,
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
        barmode="stack",
        margin=dict(l=10, r=10, t=40, b=80),
    )
    fig.update_xaxes(tickvals=x, ticktext=players, tickangle=90)
    fig.update_yaxes(title_text="Time in HR zone (min)", secondary_y=False)
    if has_trimp:
        fig.update_yaxes(title_text="HR Trimp", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Hoofd entrypoint
# -----------------------------
def session_load_pages_main(df_gps: pd.DataFrame):
    st.header("Session Load")

    missing = [c for c in [COL_DATE, COL_PLAYER] if c not in df_gps.columns]
    if missing:
        st.error(f"Ontbrekende kolommen in GPS-data: {missing}")
        return

    df = _prepare_gps(df_gps)
    if df.empty:
        st.warning("Geen bruikbare GPS-data gevonden (controleer Datum / Speler).")
        return

    # kalender = enige dagfilter
    try:
        selected_day = calendar_day_picker(df, key_prefix="sl_cal")
    except Exception as e:
        st.error(f"Kon kalender niet bouwen: {e}")
        return

    # filter op geselecteerde dag (stabiel)
    sel_ts = pd.Timestamp(selected_day)
    df_day_all = df[df[COL_DATE].dt.normalize() == sel_ts.normalize()].copy()

    if df_day_all.empty:
        st.info(f"Geen data op {sel_ts.strftime('%d-%m-%Y')}.")
        return

    # Als Summary bestaat op die dag -> gebruik alleen Summary, anders alle events
    if (df_day_all["EVENT_NORM"] == "summary").any():
        df_day_all = df_day_all[df_day_all["EVENT_NORM"] == "summary"].copy()

    # sessie-keuze (alleen als Practice (1) & Practice (2) beide bestaan)
    types_day = (
        sorted(df_day_all[COL_TYPE].dropna().astype(str).unique().tolist())
        if COL_TYPE in df_day_all.columns
        else []
    )

    session_mode = "Alle sessies"
    if "Practice (1)" in types_day and "Practice (2)" in types_day:
        session_mode = st.radio(
            "Sessie",
            options=["Practice (1)", "Practice (2)", "Beide (1+2)"],
            index=2,
            key="session_load_session_mode",
            help="Kies welke training op deze dag je wilt tonen.",
        )
    else:
        if types_day:
            st.caption("Beschikbare sessies op deze dag: " + ", ".join(types_day))

    df_day = _get_day_session_subset(df_day_all, session_mode)
    if df_day.empty:
        st.info("Geen data na sessie-filter.")
        return

    df_agg = _agg_by_player(df_day)
    if df_agg.empty:
        st.info("Geen data om te aggregeren per speler.")
        return

    # 4 grafieken in 2×2 grid
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
