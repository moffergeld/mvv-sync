# session_load_pages.py
# ==========================================
# Session Load dashboard
# - Dag-selectie via kalender grid (geen sliders)
# - In elk vak: kleur + dagnummer (geen "select" tekst)
# - Kleuren:
#     * 🔴 Match/Practice Match (Type bevat 'match')
#     * 🔵 Practice/data (wel data, geen match)
#     * ⚪ geen data
# - Navigatie: <  [Maand Jaar]  >   (title tussen pijlen) + today knop
# - Compactere kalender (kleiner)
# - Sneller gevoel: caching voor dag-aggregatie + geen onnodige herberekeningen
#
# Let op (Streamlit): elke klik veroorzaakt altijd een rerun van de pagina.
# We maken het “instant” door:
#   - alleen dag-aggregaties te cachen
#   - kalender rendering compact/licht te houden
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
PRACTICE_BLUE = "#4AA3FF"


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
    - Event='Summary' (session load werkt op summary-level)
    - TRIMP alias → 'TRIMP'
    """
    df = df_gps.copy()

    if COL_DATE not in df.columns or COL_PLAYER not in df.columns:
        return df.iloc[0:0].copy()

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE, COL_PLAYER]).copy()

    if COL_EVENT in df.columns:
        df["EVENT_NORM"] = df[COL_EVENT].map(_normalize_event)
        df = df[df["EVENT_NORM"] == "summary"].copy()

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
        *HR_COLS, "TRIMP",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if COL_TYPE in df.columns:
        df[COL_TYPE] = df[COL_TYPE].astype(str).str.strip()

    return df


def _is_match_type(type_val: str) -> bool:
    s = str(type_val).strip().lower()
    return "match" in s  # match + practice match


def _month_add(year: int, month: int, delta: int) -> tuple[int, int]:
    m = month + delta
    y = year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    return y, m


def _day_status(df: pd.DataFrame, day: date) -> str:
    sub = df[df[COL_DATE].dt.date == day]
    if sub.empty:
        return "none"
    if COL_TYPE in sub.columns and sub[COL_TYPE].map(_is_match_type).any():
        return "match"
    return "practice"


def _get_day_session_subset(df: pd.DataFrame, day: date, session_mode: str) -> pd.DataFrame:
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
# Cached day aggregation (sneller gevoel bij klikken)
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def _cached_day_agg(df: pd.DataFrame, day_iso: str, session_mode: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Cache key wordt bepaald door (day_iso + session_mode + df contents hash in streamlit).
    Retour:
      df_agg, types_day
    """
    day = date.fromisoformat(day_iso)
    df_day_all = df[df[COL_DATE].dt.date == day].copy()

    types_day: list[str] = []
    if not df_day_all.empty and COL_TYPE in df_day_all.columns:
        types_day = sorted(df_day_all[COL_TYPE].dropna().astype(str).unique().tolist())

    df_day = _get_day_session_subset(df, day, session_mode)
    df_agg = _agg_by_player(df_day) if not df_day.empty else df_day
    return df_agg, types_day


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
# Kalender UI (compact)
# -----------------------------
MONTHS_NL_CAP = [
    "Januari", "Februari", "Maart", "April", "Mei", "Juni",
    "Juli", "Augustus", "September", "Oktober", "November", "December",
]


def _calendar_css_compact():
    st.markdown(
        """
        <style>
        .sl-legend { display:flex; gap:16px; align-items:center; margin: 6px 0 10px 0; font-size: 12px; opacity:.9; }
        .sl-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:7px; }
        .sl-dot.match { background:#FF0033; }
        .sl-dot.practice { background:#4AA3FF; }
        .sl-dot.none { background: rgba(180,180,180,0.45); }

        /* Kalender knoppen compacter */
        div[data-testid="stButton"] button {
            width: 100%;
            border-radius: 10px;
            padding: 6px 8px !important;
            min-height: 34px !important;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.03);
            font-weight: 700;
            font-size: 13px;
        }
        div[data-testid="stButton"] button:hover {
            border: 1px solid rgba(255,255,255,0.22);
            background: rgba(255,255,255,0.05);
        }

        /* Maak headers compacter */
        .sl-dow { font-size: 12px; font-weight: 700; opacity: .85; margin-bottom: 4px; }

        /* Compacte spacing tussen rijen */
        .block-container { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def calendar_day_picker(df: pd.DataFrame, key_prefix: str = "slcal") -> date:
    _calendar_css_compact()

    min_dt = df[COL_DATE].min()
    max_dt = df[COL_DATE].max()
    min_day = min_dt.date()
    max_day = max_dt.date()

    if f"{key_prefix}_selected" not in st.session_state:
        st.session_state[f"{key_prefix}_selected"] = max_day

    if f"{key_prefix}_ym" not in st.session_state:
        st.session_state[f"{key_prefix}_ym"] = (
            st.session_state[f"{key_prefix}_selected"].year,
            st.session_state[f"{key_prefix}_selected"].month,
        )

    y, m = st.session_state[f"{key_prefix}_ym"]

    # NAV: <  [Maand Jaar]  >   + today
    nav1, nav2, nav3, nav4 = st.columns([0.08, 0.72, 0.08, 0.12])
    with nav1:
        if st.button("‹", key=f"{key_prefix}_prev"):
            y, m = _month_add(y, m, -1)
            st.session_state[f"{key_prefix}_ym"] = (y, m)
    with nav3:
        if st.button("›", key=f"{key_prefix}_next"):
            y, m = _month_add(y, m, +1)
            st.session_state[f"{key_prefix}_ym"] = (y, m)
    with nav4:
        if st.button("today", key=f"{key_prefix}_today"):
            st.session_state[f"{key_prefix}_selected"] = date.today()
            st.session_state[f"{key_prefix}_ym"] = (date.today().year, date.today().month)
            y, m = st.session_state[f"{key_prefix}_ym"]

    with nav2:
        st.markdown(
            f"<div style='text-align:center; font-size:20px; font-weight:800; margin-top:6px;'>{MONTHS_NL_CAP[m-1]} {y}</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div style='opacity:.75; font-size:11px; text-align:right; margin-top:-2px;'>Bereik: {min_day.strftime('%d-%m-%Y')} – {max_day.strftime('%d-%m-%Y')}</div>",
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

    cal = calendar.Calendar(firstweekday=0)  # ma
    weeks = cal.monthdatescalendar(y, m)

    day_names = ["ma", "di", "wo", "do", "vr", "za", "zo"]
    header_cols = st.columns(7)
    for i, dn in enumerate(day_names):
        header_cols[i].markdown(f"<div class='sl-dow'>{dn}</div>", unsafe_allow_html=True)

    selected = st.session_state[f"{key_prefix}_selected"]

    # Snelle status lookup: per dag in zichtbare maand (scheelt veel .dt.date filters)
    # We bouwen een set van datums met data + set met match-dagen.
    df_dates = df[COL_DATE].dt.date
    days_with_data = set(df_dates.unique())

    match_days: set[date] = set()
    if COL_TYPE in df.columns:
        tmp = df[df[COL_TYPE].map(_is_match_type)]
        match_days = set(tmp[COL_DATE].dt.date.unique())

    for week in weeks:
        cols = st.columns(7)
        for i, d in enumerate(week):
            in_month = (d.month == m)
            disabled = not in_month

            if d in match_days:
                prefix = "🔴"
            elif d in days_with_data:
                prefix = "🔵"
            else:
                prefix = "⚪"

            label = f"{prefix} {d.day}"

            # highlight selected (via label + hint, Streamlit knoppen niet per-knop te stylen)
            if d == selected:
                label = f"✅ {label}"

            bkey = f"{key_prefix}_d_{d.isoformat()}"
            with cols[i]:
                if st.button(label, key=bkey, disabled=disabled, use_container_width=True):
                    st.session_state[f"{key_prefix}_selected"] = d
                    st.session_state[f"{key_prefix}_ym"] = (d.year, d.month)

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

    selected_day = calendar_day_picker(df=df, key_prefix="slcal")

    # Sessie select (alleen als meerdere types op die dag)
    df_day_all = df[df[COL_DATE].dt.date == selected_day].copy()
    if df_day_all.empty:
        st.info(f"Geen data op {selected_day.strftime('%d-%m-%Y')}.")
        return

    session_mode = "Alle sessies"
    types_day = []
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

    # Cached dag-aggregatie (sneller)
    df_agg, _types_day_cached = _cached_day_agg(df=df, day_iso=selected_day.isoformat(), session_mode=session_mode)

    if df_agg.empty:
        st.warning("Geen data gevonden voor deze selectie (dag + sessie).")
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
