# session_load_pages.py
# ==========================================
# Session Load dashboard
# - Dag-selectie via kalender grid (geen sliders)
# - Kleuren:
#     * Rood: Match/Practice Match (Type bevat 'match')
#     * Blauw: Practice/data (wel data, geen match)
#     * Grijs: geen data
# - Klik dag = selecteer dag
# - Optioneel sessie-keuze als meerdere Type's op die dag
# - 4 grafieken per speler (ongewijzigd)
# ==========================================

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime

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
PRACTICE_BLUE = "#4AA3FF"
NO_DATA_GRAY = "rgba(180,180,180,0.28)"


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
    - (BELANGRIJK) In session load gebruiken we Summary-level voor load (als aanwezig)
    - TRIMP alias → 'TRIMP'
    """
    df = df_gps.copy()

    if COL_DATE not in df.columns or COL_PLAYER not in df.columns:
        return df.iloc[0:0].copy()

    # Datum kan al datetime zijn; errors=coerce maakt consistent
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE, COL_PLAYER]).copy()

    # Gebruik Summary voor session load (wat jij wil)
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

    # Numeriek maken
    numeric_cols = [
        COL_TD, COL_SPRINT, COL_HS,
        COL_ACC_TOT, COL_ACC_HI, COL_DEC_TOT, COL_DEC_HI,
        *HR_COLS, "TRIMP",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Type normaliseren
    if COL_TYPE in df.columns:
        df[COL_TYPE] = df[COL_TYPE].astype(str).str.strip()

    return df


def _is_match_type(type_val: str) -> bool:
    s = str(type_val).strip().lower()
    return "match" in s  # match + practice match


def _day_status(df: pd.DataFrame, day: date) -> str:
    """
    Returns: 'match' | 'practice' | 'none'
    """
    sub = df[df[COL_DATE].dt.date == day]
    if sub.empty:
        return "none"
    if COL_TYPE in sub.columns and sub[COL_TYPE].map(_is_match_type).any():
        return "match"
    return "practice"


def _get_day_session_subset(df: pd.DataFrame, day: date, session_mode: str) -> pd.DataFrame:
    """
    session_mode:
      - "Alle sessies"
      - specifieke Type-waarde
    """
    df_day = df[df[COL_DATE].dt.date == day].copy()
    if df_day.empty or COL_TYPE not in df_day.columns:
        return df_day

    if session_mode == "Alle sessies":
        return df_day

    return df_day[df_day[COL_TYPE].astype(str) == str(session_mode)].copy()


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
# Kalender UI
# -----------------------------
MONTHS_NL = [
    "januari", "februari", "maart", "april", "mei", "juni",
    "juli", "augustus", "september", "oktober", "november", "december",
]


def _calendar_css():
    st.markdown(
        """
        <style>
        .sl-cal-top {
            display:flex; align-items:center; justify-content:space-between;
            margin: 4px 0 8px 0;
        }
        .sl-cal-title {
            font-size: 22px;
            font-weight: 700;
            text-align:center;
            width: 100%;
        }
        .sl-legend { display:flex; gap:18px; align-items:center; margin: 6px 0 8px 0; }
        .sl-dot { width:10px; height:10px; border-radius: 50%; display:inline-block; margin-right:8px; }
        .sl-dot.match { background:#FF0033; }
        .sl-dot.practice { background:#4AA3FF; }
        .sl-dot.none { background: rgba(180,180,180,0.45); }

        /* buttons compacter */
        div[data-testid="stButton"] button {
            width: 100%;
            border-radius: 10px;
            padding: 10px 10px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.04);
        }
        div[data-testid="stButton"] button:hover {
            border: 1px solid rgba(255,255,255,0.22);
            background: rgba(255,255,255,0.06);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _month_add(year: int, month: int, delta: int) -> tuple[int, int]:
    m = month + delta
    y = year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    return y, m


def calendar_day_picker(df: pd.DataFrame, key_prefix: str = "slcal") -> date:
    """
    Render maand-kalender met klikbare dagen (buttons).
    Retourneert geselecteerde dag.
    """
    _calendar_css()

    min_dt = df[COL_DATE].min()
    max_dt = df[COL_DATE].max()
    min_day = min_dt.date()
    max_day = max_dt.date()

    # init state
    if f"{key_prefix}_selected" not in st.session_state:
        st.session_state[f"{key_prefix}_selected"] = max_day

    if f"{key_prefix}_ym" not in st.session_state:
        st.session_state[f"{key_prefix}_ym"] = (st.session_state[f"{key_prefix}_selected"].year,
                                                st.session_state[f"{key_prefix}_selected"].month)

    sel: date = st.session_state[f"{key_prefix}_selected"]
    y, m = st.session_state[f"{key_prefix}_ym"]

    # top nav
    c1, c2, c3 = st.columns([0.08, 0.08, 0.12])
    with c1:
        if st.button("‹", key=f"{key_prefix}_prev"):
            y, m = _month_add(y, m, -1)
            st.session_state[f"{key_prefix}_ym"] = (y, m)
    with c2:
        if st.button("›", key=f"{key_prefix}_next"):
            y, m = _month_add(y, m, +1)
            st.session_state[f"{key_prefix}_ym"] = (y, m)
    with c3:
        if st.button("today", key=f"{key_prefix}_today"):
            st.session_state[f"{key_prefix}_selected"] = date.today()
            st.session_state[f"{key_prefix}_ym"] = (date.today().year, date.today().month)
            sel = st.session_state[f"{key_prefix}_selected"]
            y, m = st.session_state[f"{key_prefix}_ym"]

    month_title = f"{MONTHS_NL[m-1]} {y}"
    st.markdown(
        f"""
        <div class="sl-cal-top">
          <div class="sl-cal-title">{month_title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='opacity:.75; font-size:12px; text-align:right;'>Bereik: {min_day.strftime('%d-%m-%Y')} – {max_day.strftime('%d-%m-%Y')}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sl-legend">
          <div><span class="sl-dot match"></span>Match/Practice Match</div>
          <div><span class="sl-dot practice"></span>Practice/data</div>
          <div><span class="sl-dot none"></span>geen data</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # build month grid (Mon..Sun)
    cal = calendar.Calendar(firstweekday=0)  # 0=ma
    weeks = cal.monthdatescalendar(y, m)

    day_names = ["ma", "di", "wo", "do", "vr", "za", "zo"]
    header_cols = st.columns(7)
    for i, dn in enumerate(day_names):
        header_cols[i].markdown(f"<div style='text-align:left; opacity:.85; font-weight:600;'>{dn}</div>", unsafe_allow_html=True)

    for wi, week in enumerate(weeks):
        cols = st.columns(7)
        for di, d in enumerate(week):
            status = _day_status(df, d)

            dot_color = {
                "match": MVV_RED,
                "practice": PRACTICE_BLUE,
                "none": "rgba(180,180,180,0.45)",
            }[status]

            in_month = (d.month == m)
            disabled = not in_month

            # label met dag van de maand
            label = f"{d.day}"

            # visueel geselecteerd
            is_selected = (d == st.session_state[f"{key_prefix}_selected"])
            border = f"2px solid {MVV_RED}" if is_selected else "1px solid rgba(255,255,255,0.12)"
            bg = "rgba(255,255,255,0.06)" if is_selected else "rgba(255,255,255,0.03)"
            opacity = "1.0" if in_month else "0.35"

            # per dag unieke key
            bkey = f"{key_prefix}_d_{d.isoformat()}"

            with cols[di]:
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px; opacity:{opacity};">
                      <span style="width:8px;height:8px;border-radius:50%;background:{dot_color};display:inline-block;"></span>
                      <span style="font-weight:700;">{label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                clicked = st.button(
                    "select",
                    key=bkey,
                    disabled=disabled,
                    use_container_width=True,
                )
                # custom border/background via extra wrapper is niet direct mogelijk per button,
                # dus we tonen selected vooral via de rode rand in de Streamlit focus-state (en dot+caption).
                if clicked and not disabled:
                    st.session_state[f"{key_prefix}_selected"] = d
                    st.session_state[f"{key_prefix}_ym"] = (d.year, d.month)
                    sel = d

    return st.session_state[f"{key_prefix}_selected"]


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

    # Kalender (enige dag-filter)
    selected_day = calendar_day_picker(df=df, key_prefix="slcal")

    # Dag subset
    df_day_all = df[df[COL_DATE].dt.date == selected_day].copy()

    if df_day_all.empty:
        st.info(f"Geen data op {selected_day.strftime('%d-%m-%Y')}.")
        return

    # Sessies op dag
    session_mode = "Alle sessies"
    if COL_TYPE in df_day_all.columns:
        types_day = sorted(df_day_all[COL_TYPE].dropna().astype(str).unique().tolist())
        if len(types_day) > 1:
            session_mode = st.selectbox(
                "Sessie",
                options=["Alle sessies", *types_day],
                index=0,
                key="session_load_session_mode",
            )
        else:
            st.caption("Beschikbare sessie op deze dag: " + ", ".join(types_day))

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
