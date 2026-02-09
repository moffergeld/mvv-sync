# session_load_pages.py
# ==========================================
# Session Load dashboard
# - Power BI-achtige Date slicer (2 sliders):
#     1) Periode (range slider: start–eind)
#     2) Selecteer dag binnen periode (single slider)
# - Kalender (klik op dag = selecteer dag)
#     * 🟥 = Match / Practice Match
#     * 🟦 = Practice / overige data
#     * ⬜ = geen data
# - Kies sessie: Practice (1) / Practice (2) / beide
# - 4 grafieken per speler:
#   * Total Distance
#   * Sprint & High Sprint
#   * Accelerations / Decelerations
#   * Time in HR zones + HR Trimp
# ==========================================

from __future__ import annotations

from datetime import date, timedelta

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
MATCH_TYPES = {"Match", "Practice Match"}  # kalender status


# -----------------------------
# Power BI-achtige slicer CSS
# -----------------------------
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

        /* Range slider */
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

        /* Single-day select slider */
        div[data-testid="stSelectSlider"] [data-baseweb="slider"] > div {{
            height: 14px !important;
        }}
        div[data-testid="stSelectSlider"] [data-baseweb="thumb"] {{
            width: 18px !important;
            height: 18px !important;
            background: {accent} !important;
            border: 2px solid rgba(255,255,255,0.85) !important;
        }}
        div[data-testid="stSelectSlider"] [data-baseweb="track"] > div {{
            background: {accent} !important;
        }}

        /* Calendar grid */
        .cal-wrap {{
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 12px 14px;
            background: rgba(255,255,255,0.02);
            margin-top: 10px;
        }}
        .cal-top {{
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap: 10px;
            margin-bottom: 8px;
        }}
        .cal-legend {{
            font-size: 12px;
            opacity: 0.8;
            white-space: nowrap;
        }}
        .cal-dow {{
            font-size: 12px;
            opacity: 0.7;
            text-align: center;
            padding: 6px 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Power BI-achtige slicer (2 sliders) + session_state friendly
# -----------------------------
def powerbi_two_sliders(
    df: pd.DataFrame,
    date_col: str,
    key_prefix: str = "sl",
    accent: str = MVV_RED,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    Returns:
      period_start (Timestamp),
      period_end (Timestamp),
      selected_day (Timestamp)  # always within range
    """
    _powerbi_slicer_css(accent=accent)

    dts = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dts.empty:
        raise ValueError("Geen geldige datums in de data.")

    all_days = sorted(pd.Series(dts.dt.date.unique()).tolist())
    min_d = all_days[0]
    max_d = all_days[-1]

    # Default periode = laatste 30 dagen (als beschikbaar)
    if len(all_days) > 30:
        default_start = all_days[-30]
    else:
        default_start = min_d
    default_end = max_d

    # Toon header (range tekst wordt hieronder ook getoond via caption)
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

    # 1) Periode slider (persists via key)
    start_d, end_d = st.slider(
        "Periode",
        min_value=min_d,
        max_value=max_d,
        value=(default_start, default_end),
        format="DD-MM-YYYY",
        key=f"{key_prefix}_range",
    )

    # Dagen binnen periode
    days_in_range = [d for d in all_days if start_d <= d <= end_d]
    if not days_in_range:
        days_in_range = [end_d]

    # Default selected day = laatste dag in range, tenzij state al gezet is (bijv. via kalenderklik)
    state_day_key = f"{key_prefix}_day"
    if state_day_key in st.session_state:
        existing = st.session_state.get(state_day_key)
        try:
            existing_d = pd.to_datetime(existing).date()
        except Exception:
            existing_d = None
        if existing_d in days_in_range:
            default_day = existing_d
        else:
            default_day = days_in_range[-1]
            st.session_state[state_day_key] = default_day
    else:
        default_day = days_in_range[-1]
        st.session_state[state_day_key] = default_day

    # 2) Dag slider binnen periode
    selected_d = st.select_slider(
        "Selecteer dag",
        options=days_in_range,
        value=default_day,
        format_func=lambda d: pd.Timestamp(d).strftime("%d-%m-%Y"),
        key=state_day_key,
    )

    return pd.Timestamp(start_d), pd.Timestamp(end_d), pd.Timestamp(selected_d)


# -----------------------------
# Kalender (klik dag = selecteer dag)
# -----------------------------
def _month_range(d0: date, d1: date) -> list[tuple[int, int]]:
    """Alle (year, month) in [d0, d1]."""
    y, m = d0.year, d0.month
    out = []
    while (y < d1.year) or (y == d1.year and m <= d1.month):
        out.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out


def _days_in_month(y: int, m: int) -> int:
    if m == 12:
        nxt = date(y + 1, 1, 1)
    else:
        nxt = date(y, m + 1, 1)
    return (nxt - date(y, m, 1)).days


def _build_day_status_map(df: pd.DataFrame) -> dict[date, str]:
    """
    Returns map date -> status:
      "match" (🟥) if any Type in MATCH_TYPES
      "data"  (🟦) else if any rows on that date
    """
    out: dict[date, str] = {}
    if df.empty or COL_DATE not in df.columns:
        return out

    d = pd.to_datetime(df[COL_DATE], errors="coerce")
    dd = d.dt.date
    types = df[COL_TYPE].astype(str).str.strip() if COL_TYPE in df.columns else pd.Series([""] * len(df))

    tmp = pd.DataFrame({"d": dd, "t": types})
    tmp = tmp.dropna(subset=["d"])
    if tmp.empty:
        return out

    for day, g in tmp.groupby("d"):
        ts = set(g["t"].tolist())
        if any(t in MATCH_TYPES for t in ts):
            out[day] = "match"
        else:
            out[day] = "data"
    return out


def calendar_click_select_day(
    df: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    selected_day: pd.Timestamp,
    key_prefix: str = "sl",
) -> pd.Timestamp:
    """
    Kalender binnen de gekozen periode.
    Klik op een dag => zet st.session_state[f"{key_prefix}_day"] en rerun.
    """
    d0 = period_start.date()
    d1 = period_end.date()
    if d0 > d1:
        d0, d1 = d1, d0

    months = _month_range(d0, d1)
    if not months:
        return selected_day

    status_map = _build_day_status_map(df)

    # default month = month van selected_day (als in range), anders laatste maand
    sel = selected_day.date()
    default_month = (sel.year, sel.month) if (sel.year, sel.month) in months else months[-1]

    st.markdown('<div class="cal-wrap">', unsafe_allow_html=True)
    top_left, top_right = st.columns([1.4, 2.6], vertical_alignment="center")

    with top_left:
        labels = [f"{y}-{m:02d}" for (y, m) in months]
        idx = labels.index(f"{default_month[0]}-{default_month[1]:02d}") if f"{default_month[0]}-{default_month[1]:02d}" in labels else len(labels) - 1
        picked = st.selectbox("Kalender", options=labels, index=idx, key=f"{key_prefix}_cal_month")
        y = int(picked.split("-")[0])
        m = int(picked.split("-")[1])

    with top_right:
        st.markdown(
            "<div class='cal-legend'>🟥 Match/Practice Match &nbsp;&nbsp; 🟦 Practice/data &nbsp;&nbsp; ⬜ geen data</div>",
            unsafe_allow_html=True,
        )

    # DOW header (ma–zo)
    dows = ["Ma", "Di", "Wo", "Do", "Vr", "Za", "Zo"]
    cols = st.columns(7)
    for i, lab in enumerate(dows):
        with cols[i]:
            st.markdown(f"<div class='cal-dow'>{lab}</div>", unsafe_allow_html=True)

    first = date(y, m, 1)
    dim = _days_in_month(y, m)
    # Python weekday: Mon=0 .. Sun=6
    offset = first.weekday()  # how many empty cells before day 1

    # grid cells = offset + dim, round up to weeks
    total_cells = offset + dim
    weeks = (total_cells + 6) // 7

    def _day_label(d: date) -> str:
        if d < d0 or d > d1:
            return "  "
        stt = status_map.get(d)
        if stt == "match":
            return f"🟥 {d.day:02d}"
        if stt == "data":
            return f"🟦 {d.day:02d}"
        return f"⬜ {d.day:02d}"

    # render weeks
    day_num = 1
    for w in range(weeks):
        row = st.columns(7)
        for c in range(7):
            cell_idx = w * 7 + c
            with row[c]:
                if cell_idx < offset or day_num > dim:
                    st.button(" ", key=f"{key_prefix}_cal_empty_{y}_{m}_{w}_{c}", disabled=True)
                    continue

                d = date(y, m, day_num)
                day_num += 1

                # buiten periode: toon disabled
                if d < d0 or d > d1:
                    st.button(" ", key=f"{key_prefix}_cal_out_{d.isoformat()}", disabled=True)
                    continue

                lab = _day_label(d)

                # markeer selected_day visueel met "▶"
                if d == sel:
                    lab = f"▶ {lab}"

                if st.button(lab, key=f"{key_prefix}_cal_{d.isoformat()}"):
                    # klik = zet slider dag
                    st.session_state[f"{key_prefix}_day"] = d
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    return pd.Timestamp(st.session_state.get(f"{key_prefix}_day", sel))


# -----------------------------
# Helpers (data prep)
# -----------------------------
def _normalize_event(e: str) -> str:
    s = str(e).strip().lower()
    return "summary" if s == "summary" else s


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
        if session_mode == "Practice (2)":
            return df_day[df_day[COL_TYPE].astype(str) == "Practice (2)"].copy()
        return df_day[df_day[COL_TYPE].astype(str).isin(["Practice (1)", "Practice (2)"])].copy()

    return df_day


def _agg_by_player(df: pd.DataFrame) -> pd.DataFrame:
    """Sommeer alle load-variabelen per speler."""
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
        fig.add_bar(
            x=x - 1.5 * width,
            y=data[COL_ACC_TOT],
            width=width,
            name="Total Accelerations",
            marker_color="rgba(255,180,180,0.9)",
        )
    if COL_ACC_HI in data.columns:
        fig.add_bar(
            x=x - 0.5 * width,
            y=data[COL_ACC_HI],
            width=width,
            name="High Accelerations",
            marker_color="rgba(200,0,0,0.9)",
        )
    if COL_DEC_TOT in data.columns:
        fig.add_bar(
            x=x + 0.5 * width,
            y=data[COL_DEC_TOT],
            width=width,
            name="Total Decelerations",
            marker_color="rgba(180,210,255,0.9)",
        )
    if COL_DEC_HI in data.columns:
        fig.add_bar(
            x=x + 1.5 * width,
            y=data[COL_DEC_HI],
            width=width,
            name="High Decelerations",
            marker_color="rgba(0,60,180,0.9)",
        )

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

    # Kalender: klik dag => set selected_day
    selected_day = calendar_click_select_day(
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

    # Dagfilter (dashboard blijft per dag)
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
