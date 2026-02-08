# session_load_pages.py
# ==========================================
# Session Load dashboard
# - Power BI-achtige Date slicer (D/W/M/Q/Y + range + dagselectie binnen range)
# - Kies sessie: Practice (1) / Practice (2) / beide
# - 4 grafieken per speler:
#   * Total Distance
#   * Sprint & High Sprint
#   * Accelerations / Decelerations
#   * Time in HR zones + HR Trimp
# ==========================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Kolomnamen
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


# -----------------------------
# Power BI-achtige date slicer
# -----------------------------
@dataclass
class TimeSlicerResult:
    granularity: str
    start: date
    end: date
    selected_day: date


def _fmt_month(d: date) -> str:
    return pd.Timestamp(d).strftime("%b %Y")  # Jan 2026


def _fmt_range_label(granularity: str, start: date, end: date) -> str:
    if granularity == "Month":
        return f"{_fmt_month(start)} - {_fmt_month(end)}"
    if granularity == "Quarter":
        s = pd.Period(pd.Timestamp(start), freq="Q")
        e = pd.Period(pd.Timestamp(end), freq="Q")
        return f"{s} - {e}"
    if granularity == "Year":
        return f"{start.year} - {end.year}"
    return f"{pd.Timestamp(start).strftime('%d-%m-%Y')} - {pd.Timestamp(end).strftime('%d-%m-%Y')}"


def _powerbi_slicer_css(accent: str = MVV_RED) -> None:
    st.markdown(
        f"""
        <style>
        .pbi-slicer-wrap {{
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 12px 14px;
            background: rgba(255,255,255,0.02);
            margin-bottom: 10px;
        }}
        .pbi-slicer-top {{
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap: 12px;
            margin-bottom: 8px;
        }}
        .pbi-slicer-title {{
            font-size: 14px;
            opacity: 0.9;
            font-weight: 600;
        }}
        .pbi-slicer-range {{
            font-size: 13px;
            opacity: 0.75;
            white-space: nowrap;
        }}

        /* Segmented control styling */
        div[data-testid="stSegmentedControl"] > div {{
            background: rgba(255,255,255,0.03) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            padding: 4px !important;
        }}
        div[data-testid="stSegmentedControl"] button {{
            border-radius: 10px !important;
            font-size: 12px !important;
            padding: 6px 10px !important;
        }}

        /* Slider styling */
        div[data-testid="stSlider"] {{
            padding-top: 0.15rem;
        }}
        div[data-testid="stSlider"] [data-baseweb="slider"] > div {{
            height: 14px !important;
        }}
        div[data-testid="stSlider"] [data-baseweb="slider"] [data-baseweb="thumb"] {{
            width: 18px !important;
            height: 18px !important;
            background: {accent} !important;
            border: 2px solid rgba(255,255,255,0.85) !important;
        }}
        div[data-testid="stSlider"] [data-baseweb="slider"] [data-baseweb="track"] {{
            background: rgba(255,255,255,0.10) !important;
        }}
        div[data-testid="stSlider"] [data-baseweb="slider"] [data-baseweb="track"] > div {{
            background: {accent} !important;
        }}

        /* Select slider styling (day picker) */
        div[data-testid="stSelectSlider"] [data-baseweb="slider"] > div {{
            height: 14px !important;
        }}
        div[data-testid="stSelectSlider"] [data-baseweb="thumb"] {{
            width: 18px !important;
            height: 18px !important;
            background: {accent} !important;
            border: 2px solid rgba(255,255,255,0.85) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def powerbi_time_slicer(
    df: pd.DataFrame,
    date_col: str,
    key_prefix: str = "pbi",
    accent: str = MVV_RED,
    default_granularity: str = "Month",
) -> TimeSlicerResult:
    """
    Power BI-achtige tijd slicer:
    - Granularity: Day / Week / Month / Quarter / Year
    - Range slider: start–end (bucket-based)
    - Extra single-day selection binnen range (voor jouw dag-dashboard)
    """
    _powerbi_slicer_css(accent=accent)

    dts = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dts.empty:
        raise ValueError("Geen geldige datums in de data.")

    # Granularity buttons
    gran = st.segmented_control(
        "Granularity",
        options=["Day", "Week", "Month", "Quarter", "Year"],
        default=default_granularity,
        key=f"{key_prefix}_gran",
    )

    # Bucket representative dates
    if gran == "Day":
        rep = sorted(pd.Series(dts.dt.date.unique()).tolist())
    else:
        ts = pd.to_datetime(dts.dt.date.astype(str))
        if gran == "Week":
            rep = sorted((ts - pd.to_timedelta(ts.dt.weekday, unit="D")).dt.date.unique().tolist())
        elif gran == "Month":
            rep = sorted(ts.dt.to_period("M").dt.to_timestamp().dt.date.unique().tolist())
        elif gran == "Quarter":
            rep = sorted(ts.dt.to_period("Q").dt.to_timestamp().dt.date.unique().tolist())
        else:  # Year
            rep = sorted(ts.dt.to_period("Y").dt.to_timestamp().dt.date.unique().tolist())

    if not rep:
        raise ValueError("Geen buckets konden worden gemaakt op basis van de datums.")

    # Default range (laatste 12 buckets als mogelijk)
    default_start = rep[-12] if len(rep) > 12 else rep[0]
    default_end = rep[-1]

    # Header met range label
    st.markdown(
        f"""
        <div class="pbi-slicer-wrap">
          <div class="pbi-slicer-top">
            <div class="pbi-slicer-title">Date</div>
            <div class="pbi-slicer-range">{_fmt_range_label(gran, default_start, default_end)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Range slider
    start, end = st.slider(
        " ",
        min_value=rep[0],
        max_value=rep[-1],
        value=(default_start, default_end),
        format="DD-MM-YYYY",
        key=f"{key_prefix}_range",
        label_visibility="collapsed",
    )

    # Compute actual available days inside range
    all_days = sorted(pd.Series(dts.dt.date.unique()).tolist())

    if gran == "Day":
        days_in_range = [d for d in all_days if start <= d <= end]
    elif gran == "Week":
        def week_monday(d: date) -> date:
            dt0 = pd.Timestamp(d)
            return (dt0 - pd.Timedelta(days=dt0.weekday())).date()
        days_in_range = [d for d in all_days if start <= week_monday(d) <= end]
    elif gran == "Month":
        def month_start(d: date) -> date:
            return pd.Timestamp(d).to_period("M").to_timestamp().date()
        days_in_range = [d for d in all_days if start <= month_start(d) <= end]
    elif gran == "Quarter":
        def q_start(d: date) -> date:
            return pd.Timestamp(d).to_period("Q").to_timestamp().date()
        days_in_range = [d for d in all_days if start <= q_start(d) <= end]
    else:  # Year
        def y_start(d: date) -> date:
            return pd.Timestamp(d).to_period("Y").to_timestamp().date()
        days_in_range = [d for d in all_days if start <= y_start(d) <= end]

    if not days_in_range:
        days_in_range = [all_days[-1]]

    # Single-day slider binnen range (voor jouw bestaande dag-view)
    selected_day = st.select_slider(
        "Selecteer dag",
        options=days_in_range,
        value=days_in_range[-1],
        format_func=lambda d: pd.Timestamp(d).strftime("%d-%m-%Y"),
        key=f"{key_prefix}_day",
    )

    return TimeSlicerResult(granularity=gran, start=start, end=end, selected_day=selected_day)


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

    # Zorg dat alle metrische kolommen numeriek zijn
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


def _get_day_session_subset(df: pd.DataFrame, day: pd.Timestamp, session_mode: str) -> pd.DataFrame:
    """Filter op gekozen datum + sessie-keuze."""
    df_day = df[df[COL_DATE].dt.date == day.date()].copy()
    if df_day.empty or COL_TYPE not in df_day.columns:
        return df_day

    types_day = sorted(df_day[COL_TYPE].dropna().astype(str).unique().tolist())

    has_p1 = "Practice (1)" in types_day
    has_p2 = "Practice (2)" in types_day

    if has_p1 and has_p2:
        if session_mode == "Practice (1)":
            return df_day[df_day[COL_TYPE].astype(str) == "Practice (1)"].copy()
        elif session_mode == "Practice (2)":
            return df_day[df_day[COL_TYPE].astype(str) == "Practice (2)"].copy()
        else:  # Beide (1+2)
            return df_day[df_day[COL_TYPE].astype(str).isin(["Practice (1)", "Practice (2)"])].copy()
    else:
        return df_day


def _agg_by_player(df: pd.DataFrame) -> pd.DataFrame:
    """Sommeer alle load-variabelen per speler."""
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
        st.warning("Geen bruikbare GPS-data gevonden (controleer Datum / Event='Summary').")
        return

    # Power BI-achtige slicer
    try:
        slicer = powerbi_time_slicer(
            df=df,
            date_col=COL_DATE,
            key_prefix="sl",
            accent=MVV_RED,
            default_granularity="Month",
        )
    except Exception as e:
        st.error(f"Kon date slicer niet maken: {e}")
        return

    selected_date = slicer.selected_day
    st.caption(f"Geselecteerde datum: {pd.Timestamp(selected_date).strftime('%d-%m-%Y')}")

    # Beschikbare types op deze dag
    df_day_all = df[df[COL_DATE].dt.date == selected_date].copy()
    types_day = (
        sorted(df_day_all[COL_TYPE].dropna().astype(str).unique().tolist())
        if (not df_day_all.empty and COL_TYPE in df_day_all.columns)
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
        else:
            st.caption("Geen kolom 'Type' of sessies op deze dag gevonden.")

    df_day = _get_day_session_subset(df, pd.to_datetime(selected_date), session_mode)
    if df_day.empty:
        st.warning("Geen data gevonden voor deze selectie (datum + sessie).")
        return

    df_agg = _agg_by_player(df_day)
    if df_agg.empty:
        st.warning("Geen data om te aggregeren per speler.")
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
