# session_load_pages.py
# ==========================================
# Session Load dashboard
# - ECHTE kalenderbalk:
#     * 1 rij met maanden
#     * daaronder alle dagen (dd) als klikbare “chips”
#     * kleurstatus:
#         🟥 Match / Practice Match
#         🟦 Practice / data
#         ⬜ geen data
#     * klik op dag => selecteer dag (stuurt ook de dag-slider)
# - Power BI-achtige date slicer (2 sliders) blijft bestaan:
#     1) Periode (range slider)
#     2) Selecteer dag (single slider)
# - Kies sessie: Practice (1) / Practice (2) / beide
# - 4 grafieken per speler
# ==========================================

from __future__ import annotations

import calendar
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
MVV_BLUE = "#5AA7FF"
MVV_GREY = "#9AA0A6"

MATCH_TYPES = {"Match", "Practice Match"}  # voor rode dagen


# -----------------------------
# Power BI-achtige slicer CSS + kalender chips
# -----------------------------
def _ui_css(accent: str = MVV_RED) -> None:
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

        /* Slider styling */
        div[data-testid="stSlider"] [data-baseweb="thumb"],
        div[data-testid="stSelectSlider"] [data-baseweb="thumb"] {{
            background: {accent} !important;
            border: 2px solid rgba(255,255,255,0.85) !important;
        }}
        div[data-testid="stSlider"] [data-baseweb="track"] > div,
        div[data-testid="stSelectSlider"] [data-baseweb="track"] > div {{
            background: {accent} !important;
        }}

        /* Calendar bar */
        .calbar {{
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 10px 12px;
            background: rgba(255,255,255,0.02);
            margin: 8px 0 12px 0;
        }}
        .cal-month-row {{
            display:flex;
            gap: 10px;
            flex-wrap: nowrap;
            overflow-x: auto;
            padding-bottom: 6px;
            margin-bottom: 6px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }}
        .cal-month {{
            font-size: 12px;
            opacity: 0.8;
            font-weight: 700;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.03);
            white-space: nowrap;
        }}
        .cal-days {{
            display:flex;
            gap: 6px;
            flex-wrap: wrap;
        }}

        /* Make st.button look like a chip */
        div[data-testid="stButton"] > button {{
            border-radius: 999px !important;
            padding: 6px 10px !important;
            min-height: 32px !important;
            font-size: 12px !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
        }}

        /* Legends */
        .cal-legend {{
            font-size: 12px;
            opacity: 0.8;
            margin-top: 6px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# 2 sliders (range + day)
# -----------------------------
def powerbi_two_sliders(
    df: pd.DataFrame,
    date_col: str,
    key_prefix: str = "sl",
    accent: str = MVV_RED,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    _ui_css(accent=accent)

    dts = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dts.empty:
        raise ValueError("Geen geldige datums in de data.")

    all_days = sorted(pd.Series(dts.dt.date.unique()).tolist())
    min_d = all_days[0]
    max_d = all_days[-1]

    default_start = all_days[-30] if len(all_days) > 30 else min_d
    default_end = max_d

    st.markdown(
        f"""
        <div class="pbi-slicer-wrap">
          <div class="pbi-slicer-top">
            <div class="pbi-slicer-title">Date</div>
            <div class="pbi-slicer-range">{pd.Timestamp(default_start).strftime('%d-%m-%Y')} - {pd.Timestamp(default_end).strftime('%d-%m-%Y')}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    start_d, end_d = st.slider(
        "Periode",
        min_value=min_d,
        max_value=max_d,
        value=(default_start, default_end),
        format="DD-MM-YYYY",
        key=f"{key_prefix}_range",
    )

    days_in_range = [d for d in all_days if start_d <= d <= end_d]
    if not days_in_range:
        days_in_range = [end_d]

    day_key = f"{key_prefix}_day"
    if day_key in st.session_state:
        try:
            cur = pd.to_datetime(st.session_state[day_key]).date()
        except Exception:
            cur = None
        default_day = cur if cur in days_in_range else days_in_range[-1]
    else:
        default_day = days_in_range[-1]
        st.session_state[day_key] = default_day

    selected_d = st.select_slider(
        "Selecteer dag",
        options=days_in_range,
        value=default_day,
        format_func=lambda d: pd.Timestamp(d).strftime("%d-%m-%Y"),
        key=day_key,
    )

    return pd.Timestamp(start_d), pd.Timestamp(end_d), pd.Timestamp(selected_d)


# -----------------------------
# Kalenderbalk: 1 rij maanden + alle dagen chips
# -----------------------------
def _month_spans(days: list[date]) -> list[tuple[int, int, int]]:
    """
    Returns list of (year, month, count_days_in_period_for_this_month)
    """
    if not days:
        return []
    spans: list[tuple[int, int, int]] = []
    cur_y, cur_m = days[0].year, days[0].month
    cnt = 0
    for d in days:
        if d.year == cur_y and d.month == cur_m:
            cnt += 1
        else:
            spans.append((cur_y, cur_m, cnt))
            cur_y, cur_m = d.year, d.month
            cnt = 1
    spans.append((cur_y, cur_m, cnt))
    return spans


def _day_status_map(df: pd.DataFrame) -> dict[date, str]:
    """
    date -> "match" | "data"
    (missing => geen data)
    """
    out: dict[date, str] = {}
    if df.empty or COL_DATE not in df.columns:
        return out

    dts = pd.to_datetime(df[COL_DATE], errors="coerce")
    tmp = pd.DataFrame(
        {
            "d": dts.dt.date,
            "t": df[COL_TYPE].astype(str).str.strip() if COL_TYPE in df.columns else "",
        }
    ).dropna(subset=["d"])
    if tmp.empty:
        return out

    for d, g in tmp.groupby("d"):
        ts = set(g["t"].tolist())
        out[d] = "match" if any(t in MATCH_TYPES for t in ts) else "data"
    return out


def calendar_bar_click_day(
    df: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    selected_day: pd.Timestamp,
    key_prefix: str = "sl",
) -> pd.Timestamp:
    d0 = period_start.date()
    d1 = period_end.date()
    if d0 > d1:
        d0, d1 = d1, d0

    all_days = [d0 + pd.Timedelta(days=i) for i in range((d1 - d0).days + 1)]
    all_days = [pd.Timestamp(d).date() for d in all_days]

    if not all_days:
        return selected_day

    status = _day_status_map(df)
    sel = selected_day.date()

    spans = _month_spans(all_days)
    month_labels = [
        f"{calendar.month_abbr[m].upper()} {y}" if i == 0 or spans[i - 1][0] != y else f"{calendar.month_abbr[m].upper()}"
        for i, (y, m, _) in enumerate(spans)
    ]

    st.markdown('<div class="calbar">', unsafe_allow_html=True)

    # 1 rij maanden
    st.markdown('<div class="cal-month-row">', unsafe_allow_html=True)
    for lab in month_labels:
        st.markdown(f'<div class="cal-month">{lab}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # dagen chips (dd)
    # kleur: 🟥 match, 🟦 data, ⬜ none; selected krijgt "▶"
    st.markdown('<div class="cal-days">', unsafe_allow_html=True)

    cols = st.columns(14)  # wrap in nette “flow”
    col_idx = 0

    for d in all_days:
        stt = status.get(d)
        if stt == "match":
            prefix = "🟥"
        elif stt == "data":
            prefix = "🟦"
        else:
            prefix = "⬜"

        label = f"{prefix} {d.day:02d}"
        if d == sel:
            label = f"▶ {label}"

        with cols[col_idx]:
            if st.button(label, key=f"{key_prefix}_calbar_{d.isoformat()}"):
                st.session_state[f"{key_prefix}_day"] = d
                st.rerun()

        col_idx = (col_idx + 1) % len(cols)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='cal-legend'>🟥 Match/Practice Match &nbsp;&nbsp; 🟦 Practice/data &nbsp;&nbsp; ⬜ geen data</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    return pd.Timestamp(st.session_state.get(f"{key_prefix}_day", sel))


# -----------------------------
# Helpers (data prep)
# -----------------------------
def _normalize_event(e: str) -> str:
    s = str(e).strip().lower()
    return "summary" if s == "summary" else s


def _prepare_gps(df_gps: pd.DataFrame) -> pd.DataFrame:
    df = df_gps.copy()

    if COL_DATE not in df.columns or COL_PLAYER not in df.columns:
        return df.iloc[0:0].copy()

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[COL_DATE, COL_PLAYER]).copy()

    if COL_EVENT in df.columns:
        df["EVENT_NORM"] = df[COL_EVENT].map(_normalize_event)
        df = df[df["EVENT_NORM"] == "summary"].copy()

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
        *HR_COLS, "TRIMP",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    return df.groupby(COL_PLAYER, as_index=False)[metric_cols].sum()


# -----------------------------
# Plot helpers
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

    # 2 sliders (periode + dag)
    try:
        period_start, period_end, selected_day = powerbi_two_sliders(
            df=df,
            date_col=COL_DATE,
            key_prefix="sl",
            accent=MVV_RED,
        )
    except Exception as e:
        st.error(f"Kon date slicer niet maken: {e}")
        return

    # Kalenderbalk (1 rij maanden + alle dagen met kleur + dagnummer)
    selected_day = calendar_bar_click_day(
        df=df,
        period_start=period_start,
        period_end=period_end,
        selected_day=selected_day,
        key_prefix="sl",
    )

    st.caption(
        f"Periode: {period_start.strftime('%d-%m-%Y')} – {period_end.strftime('%d-%m-%Y')} | "
        f"Dag: {selected_day.strftime('%d-%m-%Y')}"
    )

    # Dagfilter
    df_day_all = df[df[COL_DATE].dt.date == selected_day.date()].copy()

    # Beschikbare types op deze dag
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

    df_day = _get_day_session_subset(df, selected_day, session_mode)
    if df_day.empty:
        st.warning("Geen data gevonden voor deze selectie (dag + sessie).")
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
