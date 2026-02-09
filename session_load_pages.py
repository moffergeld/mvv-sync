# session_load_pages.py
# ==========================================
# Session Load dashboard (KALENDER als enige dag-filter)
# - Geen sliders/extra filters
# - 1 echte kalender (FullCalendar) met kleur-codering per dag:
#     * Rood  = Match / Practice Match (als die dag zo'n Type bevat)
#     * Lichtblauw = Data/Practice (wel data maar geen match-type)
#     * Grijs = Geen data
# - Klik op dag in kalender = selecteer dag
# - Daarna (optioneel) sessiekeuze: Practice (1) / Practice (2) / beide (alleen als beide bestaan op die dag)
# - 4 grafieken per speler:
#   * Total Distance
#   * Sprint & High Sprint
#   * Accelerations / Decelerations
#   * Time in HR zones + HR Trimp
#
# Vereist in requirements.txt:
#   streamlit-calendar
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from streamlit_calendar import calendar  # pip install streamlit-calendar


# -----------------------------
# Config
# -----------------------------
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
LIGHT_BLUE = "#4EA3FF"
GREY = "#8A8A8A"


# -----------------------------
# Helpers (data prep)
# -----------------------------
def _normalize_event(e: str) -> str:
    s = str(e).strip().lower()
    if s == "summary":
        return "summary"
    return s


def _prepare_gps(df_gps: pd.DataFrame) -> pd.DataFrame:
    """
    - Datum naar datetime
    - Alleen rijen met datum + speler
    - Event normaliseren en filteren op Summary
    - TRIMP-naam normaliseren naar kolom 'TRIMP'
    - Metric kolommen numeriek
    """
    df = df_gps.copy()

    if COL_DATE not in df.columns or COL_PLAYER not in df.columns:
        return df.iloc[0:0].copy()

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[COL_DATE, COL_PLAYER]).copy()

    if COL_EVENT in df.columns:
        df["EVENT_NORM"] = df[COL_EVENT].map(_normalize_event)
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

    if COL_TYPE in df.columns:
        df[COL_TYPE] = df[COL_TYPE].astype(str).str.strip()

    return df


def _get_day_session_subset(df: pd.DataFrame, day: pd.Timestamp, session_mode: str) -> pd.DataFrame:
    df_day = df[df[COL_DATE].dt.date == day.date()].copy()
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
# Calendar (kleur per dag) + click = select day
# -----------------------------
def _day_color_map(df: pd.DataFrame) -> dict[str, str]:
    """
    Returns {YYYY-MM-DD: hexcolor}
    - red if match/practice match
    - light blue if any data (but no match type)
    - grey if no data (filled later by range)
    """
    df0 = df.copy()
    df0["d"] = df0[COL_DATE].dt.date
    day_to_types = (
        df0.groupby("d")[COL_TYPE]
        .apply(lambda s: set(map(str, s.dropna().tolist())))
        .to_dict()
    )

    out: dict[str, str] = {}
    for d, types in day_to_types.items():
        if any(t in MATCH_TYPES for t in types):
            out[pd.Timestamp(d).strftime("%Y-%m-%d")] = MVV_RED
        else:
            out[pd.Timestamp(d).strftime("%Y-%m-%d")] = LIGHT_BLUE
    return out


def _calendar_css() -> None:
    # FullCalendar styling + legenda blokjes
    st.markdown(
        f"""
        <style>
        /* Calendar container look */
        .fc {{
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 10px 12px;
            background: rgba(255,255,255,0.02);
        }}
        .fc .fc-toolbar-title {{
            font-size: 16px !important;
            font-weight: 700 !important;
            letter-spacing: 0.2px;
        }}
        .fc .fc-button {{
            border-radius: 10px !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            background: rgba(255,255,255,0.04) !important;
        }}
        .fc .fc-button:hover {{
            background: rgba(255,255,255,0.07) !important;
        }}
        /* day numbers visible & clean */
        .fc .fc-daygrid-day-number {{
            font-weight: 700;
            opacity: 0.95;
        }}
        /* background events should not cover the number */
        .fc .fc-bg-event {{
            opacity: 0.28;
        }}

        /* Legend */
        .sl-legend {{
            display:flex;
            gap: 14px;
            flex-wrap: wrap;
            align-items:center;
            margin: 10px 0 0 0;
            opacity: 0.95;
        }}
        .sl-legend-item {{
            display:flex;
            gap: 8px;
            align-items:center;
            font-size: 12.5px;
        }}
        .sl-dot {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
            display:inline-block;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sl-legend">
          <div class="sl-legend-item"><span class="sl-dot" style="background:{MVV_RED};"></span>Match/Practice Match</div>
          <div class="sl-legend-item"><span class="sl-dot" style="background:{LIGHT_BLUE};"></span>Practice/data</div>
          <div class="sl-legend-item"><span class="sl-dot" style="background:{GREY};"></span>Geen data</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _calendar_select_day(df: pd.DataFrame, key: str = "sl_cal") -> pd.Timestamp:
    """
    Renders calendar; click date sets st.session_state[f"{key}_selected"].
    Returns selected day (Timestamp).
    """
    _calendar_css()

    dts = pd.to_datetime(df[COL_DATE], errors="coerce").dropna()
    all_days = sorted(pd.Series(dts.dt.date.unique()).tolist())
    if not all_days:
        raise ValueError("Geen datums gevonden in de data.")

    min_d = all_days[0]
    max_d = all_days[-1]

    # default selection
    ss_key = f"{key}_selected"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = pd.Timestamp(max_d).strftime("%Y-%m-%d")

    # build colors for days with data
    color_by_day = _day_color_map(df)

    # fill full range with grey for "no data"
    full_range = pd.date_range(pd.Timestamp(min_d), pd.Timestamp(max_d), freq="D")
    events = []
    for d in full_range:
        iso = d.strftime("%Y-%m-%d")
        col = color_by_day.get(iso, GREY)
        # background event fills the cell
        events.append(
            {
                "title": "",
                "start": iso,
                "end": iso,  # same day
                "display": "background",
                "backgroundColor": col,
            }
        )

    # initial date = selected
    initial_date = st.session_state[ss_key]

    cal_options = {
        "initialView": "dayGridMonth",
        "height": 520,
        "locale": "nl",
        "firstDay": 1,  # Monday
        "headerToolbar": {
            "left": "prev,next today",
            "center": "title",
            "right": "",
        },
        "navLinks": True,
        "selectable": True,
        "dayMaxEvents": False,
        "initialDate": initial_date,
    }

    cal_state = calendar(
        events=events,
        options=cal_options,
        key=key,
    )

    # streamlit-calendar returns callbacks state; handle date click/select
    picked_iso = None

    # dateClick
    if isinstance(cal_state, dict):
        if cal_state.get("dateClick"):
            picked_iso = cal_state["dateClick"].get("date")
        # select (range) – if user drags, take start
        if not picked_iso and cal_state.get("select"):
            picked_iso = cal_state["select"].get("start")

    if picked_iso:
        # normalize to YYYY-MM-DD
        picked_iso = str(picked_iso)[:10]
        st.session_state[ss_key] = picked_iso

    return pd.Timestamp(st.session_state[ss_key])


# -----------------------------
# Plots (4 grafieken)
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
        st.warning("Geen bruikbare GPS-data gevonden (controleer Datum / Event='Summary').")
        return

    # --- Kalender = enige filter
    try:
        selected_day = _calendar_select_day(df, key="sl_cal")
    except Exception as e:
        st.error(f"Kon kalender niet maken: {e}")
        return

    # Dagfilter
    df_day_all = df[df[COL_DATE].dt.date == selected_day.date()].copy()

    if df_day_all.empty:
        st.info(f"Geen data op {selected_day.strftime('%d-%m-%Y')}.")
        return

    # Sessiekeuze alleen als Practice (1) en (2) beide bestaan
    types_day = (
        sorted(df_day_all[COL_TYPE].dropna().astype(str).unique().tolist())
        if (COL_TYPE in df_day_all.columns)
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

    df_day = _get_day_session_subset(df, selected_day, session_mode)
    if df_day.empty:
        st.warning("Geen data gevonden voor deze selectie (dag + sessie).")
        return

    df_agg = _agg_by_player(df_day)
    if df_agg.empty:
        st.warning("Geen data om te aggregeren per speler.")
        return

    st.caption(f"Geselecteerde dag: {selected_day.strftime('%d-%m-%Y')}")

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
