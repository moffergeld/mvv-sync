# session_load_pages.py
# ==========================================
# Session Load dashboard
# - ENIGE dag-filter = kalender (maand-grid)
# - Kleuren:
#     * Rood: Match / Practice Match
#     * Lichtblauw: Practice / overige data
#     * Grijs: geen data
# - Klik op dag = selecteer dag
# - 4 grafieken per speler (dag-totalen):
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

# Kolomnamen (dashboard DF)
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
TRIMP_CANDIDATES = ["HRTrimp", "HR Trimp", "HRtrimp", "Trimp", "TRIMP", "TRIMP ", "TRIMP_"]

MATCH_TYPES = {"Match", "Practice Match"}

MVV_RED = "#FF0033"
PRACTICE_BLUE = "#4DA3FF"
NO_DATA_GREY = "#8A8F98"


# -----------------------------
# Helpers (data prep)
# -----------------------------
def _prepare_gps(df_gps: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal prep:
    - Datum -> datetime
    - drop rows without date/player
    - numeric columns -> numeric
    - TRIMP -> column 'TRIMP'
    """
    df = df_gps.copy()

    if COL_DATE not in df.columns or COL_PLAYER not in df.columns:
        return df.iloc[0:0].copy()

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE, COL_PLAYER]).copy()

    # TRIMP alias -> TRIMP
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

    if COL_EVENT in df.columns:
        df[COL_EVENT] = df[COL_EVENT].astype(str).str.strip()

    return df


def _get_day_session_subset(df: pd.DataFrame, day: date, session_mode: str) -> pd.DataFrame:
    """Filter op gekozen datum + sessie-keuze (Practice (1)/(2) samen)."""
    df_day = df[df[COL_DATE].dt.date == day].copy()
    if df_day.empty or COL_TYPE not in df_day.columns:
        return df_day

    types_day = sorted(df_day[COL_TYPE].dropna().astype(str).unique().tolist())

    has_p1 = "Practice (1)" in types_day
    has_p2 = "Practice (2)" in types_day

    if has_p1 and has_p2:
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
        *HR_COLS,
        "TRIMP",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    return df.groupby(COL_PLAYER, as_index=False)[metric_cols].sum()


# -----------------------------
# Calendar UI
# -----------------------------
def _calendar_css() -> None:
    st.markdown(
        f"""
        <style>
        .sl-cal-wrap {{
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 14px;
            background: rgba(255,255,255,0.02);
            margin: 8px 0 14px 0;
        }}
        .sl-cal-top {{
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap: 10px;
            margin-bottom: 10px;
        }}
        .sl-cal-title {{
            font-size: 22px;
            font-weight: 700;
            opacity: 0.95;
            text-transform: lowercase;
        }}
        .sl-cal-legend {{
            display:flex;
            align-items:center;
            gap: 14px;
            flex-wrap: wrap;
            font-size: 13px;
            opacity: 0.9;
            margin: 6px 0 10px 0;
        }}
        .sl-dot {{
            width: 10px; height: 10px; border-radius: 3px; display:inline-block;
            margin-right: 6px;
        }}
        .sl-cal-dow {{
            font-size: 13px;
            opacity: 0.75;
            font-weight: 600;
            text-transform: lowercase;
            padding: 6px 0 6px 0;
            text-align: left;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _month_name_nl(month: int) -> str:
    nl = {
        1: "januari", 2: "februari", 3: "maart", 4: "april",
        5: "mei", 6: "juni", 7: "juli", 8: "augustus",
        9: "september", 10: "oktober", 11: "november", 12: "december",
    }
    return nl.get(month, str(month))


def _day_type_map(df: pd.DataFrame) -> dict[date, str]:
    """
    Returns map: day -> "match" | "practice" | "none"
    """
    out: dict[date, str] = {}
    if df.empty or COL_DATE not in df.columns:
        return out

    tmp = df[[COL_DATE] + ([COL_TYPE] if COL_TYPE in df.columns else [])].copy()
    tmp["day"] = tmp[COL_DATE].dt.date

    if COL_TYPE in tmp.columns:
        g = tmp.groupby("day")[COL_TYPE].agg(lambda s: set([str(x).strip() for x in s.dropna().tolist()]))
        for d, types in g.items():
            if any(t in MATCH_TYPES for t in types):
                out[d] = "match"
            else:
                out[d] = "practice"
    else:
        # if no type column, any data -> practice
        for d in tmp["day"].unique().tolist():
            out[d] = "practice"

    return out


def calendar_day_picker(df: pd.DataFrame, key_prefix: str = "slcal") -> date:
    """
    Renders a month calendar and returns selected date (python date).
    Selected date persists in session_state.
    """
    _calendar_css()

    dts = pd.to_datetime(df[COL_DATE], errors="coerce").dropna()
    if dts.empty:
        raise ValueError("Geen geldige datums in de data.")

    min_day = dts.dt.date.min()
    max_day = dts.dt.date.max()

    # state: current view month & selected day
    if f"{key_prefix}_sel" not in st.session_state:
        st.session_state[f"{key_prefix}_sel"] = max_day
    if f"{key_prefix}_ym" not in st.session_state:
        st.session_state[f"{key_prefix}_ym"] = (max_day.year, max_day.month)

    sel: date = st.session_state[f"{key_prefix}_sel"]
    view_y, view_m = st.session_state[f"{key_prefix}_ym"]

    # Day status
    day_status = _day_type_map(df)

    # Header controls
    col_a, col_b, col_c, col_d, col_e = st.columns([0.5, 0.5, 0.9, 4.5, 1.2], vertical_alignment="center")
    with col_a:
        if st.button("‹", key=f"{key_prefix}_prev", use_container_width=True):
            y, m = view_y, view_m
            if m == 1:
                y -= 1
                m = 12
            else:
                m -= 1
            st.session_state[f"{key_prefix}_ym"] = (y, m)
            st.rerun()

    with col_b:
        if st.button("›", key=f"{key_prefix}_next", use_container_width=True):
            y, m = view_y, view_m
            if m == 12:
                y += 1
                m = 1
            else:
                m += 1
            st.session_state[f"{key_prefix}_ym"] = (y, m)
            st.rerun()

    with col_c:
        if st.button("today", key=f"{key_prefix}_today"):
            today = date.today()
            st.session_state[f"{key_prefix}_sel"] = today
            st.session_state[f"{key_prefix}_ym"] = (today.year, today.month)
            st.rerun()

    with col_d:
        st.markdown(
            f"<div class='sl-cal-title'>{_month_name_nl(view_m)} {view_y}</div>",
            unsafe_allow_html=True,
        )

    with col_e:
        # quick info
        st.caption(f"Bereik: {min_day.strftime('%d-%m-%Y')} – {max_day.strftime('%d-%m-%Y')}")

    st.markdown(
        f"""
        <div class="sl-cal-legend">
            <span><span class="sl-dot" style="background:{MVV_RED};"></span>Match/Practice Match</span>
            <span><span class="sl-dot" style="background:{PRACTICE_BLUE};"></span>Practice/data</span>
            <span><span class="sl-dot" style="background:{NO_DATA_GREY};"></span>geen data</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Calendar grid (Mon..Sun)
    cal = calendar.Calendar(firstweekday=0)  # Monday
    weeks = cal.monthdatescalendar(view_y, view_m)

    dow = ["ma", "di", "wo", "do", "vr", "za", "zo"]
    cols = st.columns(7)
    for i, name in enumerate(dow):
        with cols[i]:
            st.markdown(f"<div class='sl-cal-dow'>{name}</div>", unsafe_allow_html=True)

    # render weeks
    for w in weeks:
        row = st.columns(7, vertical_alignment="top")
        for i, d in enumerate(w):
            in_month = (d.month == view_m)

            status = day_status.get(d, "none")
            if status == "match":
                dot_color = MVV_RED
            elif status == "practice":
                dot_color = PRACTICE_BLUE
            else:
                dot_color = NO_DATA_GREY

            # Disable days outside month for cleaner view (still show number faded)
            disabled = (not in_month)

            # Keep unique key per date
            k = f"{key_prefix}_day_{d.isoformat()}"

            label = f"{d.day}"
            # selected styling via button type + small indicator
            is_sel = (d == sel)
            btn_type = "primary" if is_sel else "secondary"

            with row[i]:
                # little dot + number (dot rendered by markdown above button)
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>"
                    f"<span class='sl-dot' style='background:{dot_color};'></span>"
                    f"<span style='opacity:{'1.0' if in_month else '0.35'};font-weight:700;'>{label}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if st.button(
                    "select",
                    key=k,
                    type=btn_type,
                    disabled=disabled,
                    use_container_width=True,
                ):
                    st.session_state[f"{key_prefix}_sel"] = d
                    st.session_state[f"{key_prefix}_ym"] = (d.year, d.month)
                    st.rerun()

    return st.session_state[f"{key_prefix}_sel"]


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

    mean_val = float(np.nanmean(vals)) if len(vals) else 0.0
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
    if not have_cols:
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
        st.warning("Geen bruikbare GPS-data gevonden.")
        return

    # Kalender = enige dagfilter
    try:
        selected_day = calendar_day_picker(df, key_prefix="slcal")
    except Exception as e:
        st.error(f"Kon kalender niet maken: {e}")
        return

    # Beschikbare types op deze dag
    df_day_all = df[df[COL_DATE].dt.date == selected_day].copy()
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
            st.info(f"Geen data op {selected_day.strftime('%d-%m-%Y')}.")

    df_day = _get_day_session_subset(df, selected_day, session_mode)
    if df_day.empty:
        st.stop()

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
