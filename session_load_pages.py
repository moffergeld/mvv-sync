# session_load_pages.py
# ==========================================
# Session Load dashboard
# - ENIGE dag-filter = kalender (geen sliders)
# - Kalender:
#     * Maanden header + dagen als klikbare "chips"
#     * Kleurcodering:
#         - Rood: Match / Practice Match (op basis van Type)
#         - Lichtblauw: Practice/data (alle overige Types met data)
#         - Grijs: geen data
# - Klik dag in kalender = selecteer dag
# - Kies sessie: Practice (1) / Practice (2) / beide (alleen als beide bestaan op die dag)
# - 4 grafieken per speler:
#   * Total Distance
#   * Sprint & High Sprint
#   * Accelerations / Decelerations
#   * Time in HR zones + HR Trimp
# ==========================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as dt_date
from datetime import timedelta
import calendar as pycal

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -----------------------------
# Kolomnamen (verwacht in df_all vanuit 07_GPS_Data.py)
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

# Type-categorieën voor kalenderkleuren
MATCH_TYPES = {"Match", "Practice Match"}

# MVV kleuren
MVV_RED = "#FF0033"
MVV_BLUE = "#5DAEFF"  # lichtblauw accent voor Practice/data
MVV_GREY = "#6B7280"  # grijs "geen data"


# -----------------------------
# CSS
# -----------------------------
def _inject_calendar_css() -> None:
    st.markdown(
        f"""
        <style>
        .sl-wrap {{
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 14px 14px 10px 14px;
            background: rgba(255,255,255,0.02);
            margin: 12px 0 14px 0;
        }}
        .sl-top {{
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap: 10px;
            margin-bottom: 8px;
        }}
        .sl-title {{
            font-size: 28px;
            font-weight: 800;
            letter-spacing: 0.2px;
        }}
        .sl-legend {{
            display:flex;
            align-items:center;
            gap: 14px;
            margin: 4px 0 10px 0;
            font-size: 12.5px;
            opacity: 0.9;
        }}
        .sl-dot {{
            display:inline-block;
            width: 10px;
            height: 10px;
            border-radius: 2px;
            margin-right: 6px;
        }}

        /* Buttons - force compact */
        div[data-testid="stButton"] > button {{
            border-radius: 10px !important;
            padding: 6px 10px !important;
            height: 40px !important;
            width: 100% !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            background: rgba(255,255,255,0.02) !important;
        }}

        /* Make the month title centered */
        .sl-month {{
            text-align:center;
            font-weight: 800;
            font-size: 22px;
            margin: 8px 0 10px 0;
        }}

        /* Weekday header */
        .sl-weekdays {{
            display:grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 8px;
            margin: 0 0 6px 0;
            font-weight: 700;
            font-size: 12.5px;
            opacity: 0.9;
            text-align:center;
        }}
        .sl-grid {{
            display:grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 8px;
        }}
        .sl-chip {{
            display:flex;
            align-items:center;
            justify-content:center;
            gap: 8px;
            width: 100%;
            height: 40px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.02);
        }}
        .sl-daynum {{
            font-weight: 800;
            font-size: 14px;
        }}
        .sl-ind {{
            width: 10px;
            height: 10px;
            border-radius: 3px;
            opacity: 0.95;
        }}

        /* Selected day outline */
        .sl-selected {{
            outline: 2px solid {MVV_RED};
            outline-offset: 1px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Helpers
# -----------------------------
def _normalize_event(e: str) -> str:
    s = str(e).strip().lower()
    if s == "summary":
        return "summary"
    return s


def _ensure_datetime_series(s: pd.Series) -> pd.Series:
    # If already datetime64, keep it. If object/string, parse with dayfirst True as fallback.
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def _prepare_gps(df_gps: pd.DataFrame) -> pd.DataFrame:
    """
    - Datum -> datetime
    - Filter: datum + speler aanwezig
    - Filter: Event == Summary (normalised)
    - TRIMP alias -> 'TRIMP'
    - Metrics -> numeric
    """
    df = df_gps.copy()

    if COL_DATE not in df.columns or COL_PLAYER not in df.columns:
        return df.iloc[0:0].copy()

    df[COL_DATE] = _ensure_datetime_series(df[COL_DATE])
    df = df.dropna(subset=[COL_DATE, COL_PLAYER]).copy()

    if COL_EVENT in df.columns:
        df["EVENT_NORM"] = df[COL_EVENT].map(_normalize_event)
        df = df[df["EVENT_NORM"] == "summary"].copy()
    else:
        # Zonder Event-kolom kan Session Load niet betrouwbaar werken
        return df.iloc[0:0].copy()

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

    # Normalize Type strings
    if COL_TYPE in df.columns:
        df[COL_TYPE] = df[COL_TYPE].astype(str).str.strip()

    return df


def _get_day_session_subset(df: pd.DataFrame, day: pd.Timestamp, session_mode: str) -> pd.DataFrame:
    df_day = df[df[COL_DATE].dt.date == dayday_to_date(day)].copy()
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


def day_to_timestamp(d: dt_date) -> pd.Timestamp:
    return pd.Timestamp(d)


def T_to_date(ts: pd.Timestamp) -> dt_date:
    return ts.date()


def Rng_month_anchor(d: dt_date) -> dt_date:
    return dt_date(d.year, d.month, 1)


def add_months(anchor: dt_date, delta: int) -> dt_date:
    y = anchor.year + (anchor.month - 1 + delta) // 12
    m = (anchor.month - 1 + delta) % 12 + 1
    return dt_date(y, m, 1)


def month_label(anchor: dt_date) -> str:
    # NL labels
    months_nl = [
        "januari", "februari", "maart", "april", "mei", "juni",
        "juli", "augustus", "september", "oktober", "november", "december"
    ]
    return f"{months_nl[anchor.month - 1]} {anchor.year}"


def weekday_headers_nl() -> list[str]:
    # maandag .. zondag
    return ["ma", "di", "wo", "do", "vr", "za", "zo"]


def month_grid_dates(anchor: dt_date) -> list[dt_date]:
    """
    Returns a 6-week grid (42 dates) starting on Monday.
    Includes leading/trailing days from adjacent months.
    """
    first = dt_date(anchor.year, anchor.month, 1)
    # Python weekday: Monday=0 ... Sunday=6
    start = first - timedelta(days=first.weekday())
    return [start + timedelta(days=i) for i in range(42)]


def classify_day(df: pd.DataFrame, d: dt_date) -> str:
    """
    Classification based on prepared df (Summary-only).
    Returns: "match" | "practice" | "none"
    """
    if df.empty:
        return "none"
    day_mask = df[COL_DATE].dt.date == d
    if not day_mask.any():
        return "none"

    if COL_TYPE in df.columns:
        types = set(df.loc[day_mask, COL_TYPE].dropna().astype(str).str.strip().tolist())
        if any(t in MATCH_TYPES for t in types):
            return "match"

    return "practice"


def indicator_color(kind: str) -> str:
    if kind == "match":
        return MVV_RED
    if kind == "practice":
        return MVV_BLUE
    return MVV_GREY


def is_in_month(d: dt_date, anchor: dt_date) -> bool:
    return d.month == anchor.month and d.year == anchor.year


def _set_selected_day(d: dt_date) -> None:
    st.session_state["sl_selected_day"] = d.isoformat()


def _get_selected_day(default: dt_date) -> dt_date:
    v = st.session_state.get("sl_selected_day")
    if not v:
        return default
    try:
        return dt_date.fromisoformat(str(v))
    except Exception:
        return default


def _set_month_anchor(anchor: dt_date) -> None:
    st.session_state["sl_month_anchor"] = anchor.isoformat()


def _get_month_anchor(default: dt_date) -> dt_date:
    v = st.session_state.get("sl_month_anchor")
    if not v:
        return default
    try:
        return dt_date.fromisoformat(str(v))
    except Exception:
        return default


# -----------------------------
# Plots
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
# Kalender UI
# -----------------------------
def _calendar_ui(df_prepared: pd.DataFrame, min_day: dt_date, max_day: dt_date) -> dt_date:
    _inject_calendar_css()

    # Default anchor: month of max_day (meest recente data)
    default_anchor = Rng_month_anchor(max_day)
    anchor = _get_month_anchor(default_anchor)

    # Clamp anchor between min and max months
    if anchor < Rng_month_anchor(min_day):
        anchor = Rng_month_anchor(min_day)
        _set_month_anchor(anchor)
    if anchor > Rng_month_anchor(max_day):
        anchor = Rng_month_anchor(max_day)
        _set_month_anchor(anchor)

    # Default selected day: max_day
    selected = _get_selected_day(max_day)
    if selected < min_day or selected > max_day:
        selected = max_day
        _set_selected_day(selected)

    # Top navigation
    cprev, cnext, ctoday, _spacer = st.columns([0.6, 0.6, 0.9, 6.0])
    with cprev:
        if st.button("‹", key="sl_cal_prev"):
            _set_month_anchor(add_months(anchor, -1))
            st.rerun()
    with cnext:
        if st.button("›", key="sl_cal_next"):
            _set_month_anchor(add_months(anchor, +1))
            st.rerun()
    with ctoday:
        if st.button("today", key="sl_cal_today"):
            _set_month_anchor(Rng_month_anchor(dt_date.today()))
            _set_selected_day(dt_date.today())
            st.rerun()

    st.markdown(f'<div class="sl-month">{month_label(anchor)}</div>', unsafe_allow_html=True)

    # Legend
    st.markdown(
        f"""
        <div class="sl-legend">
          <span><span class="sl-dot" style="background:{MVV_RED}"></span>Match/Practice Match</span>
          <span><span class="sl-dot" style="background:{MVV_BLUE}"></span>Practice/data</span>
          <span><span class="sl-dot" style="background:{MVV_GREY}"></span>Geen data</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Weekday header
    wh = weekday_headers_nl()
    st.markdown(
        '<div class="sl-weekdays">' + "".join([f"<div>{w}</div>" for w in wh]) + "</div>",
        unsafe_allow_html=True,
    )

    # Render 6-week grid (42 days)
    grid_days = month_grid_dates(anchor)

    # We create 7 columns repeatedly
    for week in range(6):
        cols = st.columns(7, gap="small")
        for i in range(7):
            d = grid_days[week * 7 + i]
            in_month = is_in_month(d, anchor)
            within_data_bounds = (min_day <= d <= max_day)

            kind = classify_day(df_prepared, d) if within_data_bounds else "none"
            ind_col = indicator_color(kind)

            # Disable days outside the displayed month OR outside data bounds
            disabled = (not in_month) or (not within_data_bounds)

            # Visual label: day number only (must be in)
            label = str(d.day)

            # Button key must be unique per cell
            bkey = f"sl_daybtn_{d.isoformat()}"

            # We simulate chip with button; selection is stored in session_state
            with cols[i]:
                # add a tiny colored indicator via markdown above button
                sel_class = "sl-selected" if (d == selected and not disabled) else ""
                st.markdown(
                    f"""
                    <div class="sl-chip {sel_class}">
                      <div class="sl-daynum">{label}</div>
                      <div class="sl-ind" style="background:{ind_col}"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # invisible click area: actual button
                if st.button(" ", key=bkey, disabled=disabled):
                    _set_selected_day(d)
                    st.rerun()

    return _get_selected_day(selected)


def day_to_date(ts: pd.Timestamp) -> dt_date:
    return ts.date()


def Rng_to_date(d: dt_date) -> dt_date:
    return d


def month_grid_dates(anchor: dt_date) -> list[dt_date]:
    first = dt_date(anchor.year, anchor.month, 1)
    start = first - timedelta(days=first.weekday())
    return [start + timedelta(days=i) for i in range(42)]


def is_in_month(d: dt_date, anchor: dt_date) -> bool:
    return d.month == anchor.month and d.year == anchor.year


def day_to_date(ts: pd.Timestamp) -> dt_date:
    return ts.date()


def Tstamp(d: dt_date) -> pd.Timestamp:
    return pd.Timestamp(d)


def day_to_timestamp(d: dt_date) -> pd.Timestamp:
    return pd.Timestamp(d)


def T_to_date(ts: pd.Timestamp) -> dt_date:
    return ts.date()


def Rng_month_anchor(d: dt_date) -> dt_date:
    return dt_date(d.year, d.month, 1)


def add_months(anchor: dt_date, delta: int) -> dt_date:
    y = anchor.year + (anchor.month - 1 + delta) // 12
    m = (anchor.month - 1 + delta) % 12 + 1
    return dt_date(y, m, 1)


def month_label(anchor: dt_date) -> str:
    months_nl = [
        "januari", "februari", "maart", "april", "mei", "juni",
        "juli", "augustus", "september", "oktober", "november", "december"
    ]
    return f"{months_nl[anchor.month - 1]} {anchor.year}"


def weekday_headers_nl() -> list[str]:
    return ["ma", "di", "wo", "do", "vr", "za", "zo"]


def classify_day(df: pd.DataFrame, d: dt_date) -> str:
    if df.empty:
        return "none"
    mask = df[COL_DATE].dt.date == d
    if not mask.any():
        return "none"
    if COL_TYPE in df.columns:
        types = set(df.loc[mask, COL_TYPE].dropna().astype(str).str.strip().tolist())
        if any(t in MATCH_TYPES for t in types):
            return "match"
    return "practice"


def indicator_color(kind: str) -> str:
    if kind == "match":
        return MVV_RED
    if kind == "practice":
        return MVV_BLUE
    return MVV_GREY


def _set_selected_day(d: dt_date) -> None:
    st.session_state["sl_selected_day"] = d.isoformat()


def _get_selected_day(default: dt_date) -> dt_date:
    v = st.session_state.get("sl_selected_day")
    if not v:
        return default
    try:
        return dt_date.fromisoformat(str(v))
    except Exception:
        return default


def _set_month_anchor(anchor: dt_date) -> None:
    st.session_state["sl_month_anchor"] = anchor.isoformat()


def _get_month_anchor(default: dt_date) -> dt_date:
    v = st.session_state.get("sl_month_anchor")
    if not v:
        return default
    try:
        return dt_date.fromisoformat(str(v))
    except Exception:
        return default


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


# -----------------------------
# Main entrypoint
# -----------------------------
def session_load_pages_main(df_gps: pd.DataFrame):
    st.header("Session Load")

    missing = [c for c in [COL_DATE, COL_PLAYER, COL_EVENT] if c not in df_gps.columns]
    if missing:
        st.error(f"Ontbrekende kolommen in GPS-data: {missing}")
        return

    df = _prepare_gps(df_gps)
    if df.empty:
        st.warning("Geen bruikbare GPS-data gevonden (controleer: Event='Summary', Datum, Speler).")
        return

    # min/max dag op basis van beschikbare data
    all_days = sorted(df[COL_DATE].dt.date.unique().tolist())
    if not all_days:
        st.warning("Geen datums gevonden in de data.")
        return
    min_day = all_days[0]
    max_day = all_days[-1]

    # Kalender = enige filter
    selected_day = _calendar_ui(df, min_day=min_day, max_day=max_day)
    selected_ts = pd.Timestamp(selected_day)

    # Dagfilter
    df_day_all = df[df[COL_DATE].dt.date == selected_day].copy()

    if df_day_all.empty:
        st.info(f"Geen data op {selected_ts.strftime('%d-%m-%Y')}.")
        return

    # Beschikbare types op deze dag
    types_day = sorted(df_day_all[COL_TYPE].dropna().astype(str).unique().tolist()) if COL_TYPE in df_day_all.columns else []

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
        else:
            st.caption("Geen sessie-types gevonden op deze dag.")

    df_day = _get_day_session_subset(df, pd.Timestamp(selected_day), session_mode)
    if df_day.empty:
        st.info(f"Geen data na sessiefilter op {selected_ts.strftime('%d-%m-%Y')}.")
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
