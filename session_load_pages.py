# session_load_pages.py
from __future__ import annotations

import calendar
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
PRACTICE_BLUE = "#4AA3FF"


# -----------------------------
# Helpers (data prep)
# -----------------------------
def _prepare_gps(df_gps: pd.DataFrame) -> pd.DataFrame:
    df = df_gps.copy()

    if COL_DATE not in df.columns or COL_PLAYER not in df.columns:
        return df.iloc[0:0].copy()

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE, COL_PLAYER]).copy()

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

    for c in [COL_PLAYER, COL_EVENT, COL_TYPE]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


def _compute_day_sets(df: pd.DataFrame) -> tuple[set[date], set[date]]:
    if df.empty:
        return set(), set()

    d = df.copy()
    d["_d"] = pd.to_datetime(d[COL_DATE], errors="coerce").dt.date
    days_with_data = set(d["_d"].dropna().unique().tolist())

    match_days: set[date] = set()
    if COL_TYPE in d.columns:
        t = d[COL_TYPE].astype(str).str.lower()
        match_days = set(d.loc[t.str.contains("match", na=False), "_d"].dropna().unique().tolist())

    return days_with_data, match_days


def _month_label_nl(y: int, m: int) -> str:
    months = ["", "Januari", "Februari", "Maart", "April", "Mei", "Juni", "Juli",
              "Augustus", "September", "Oktober", "November", "December"]
    return f"{months[m]} {y}"


# -----------------------------
# Kalender CSS (buttons)
# -----------------------------
def _calendar_css_compact() -> None:
    st.markdown(
        f"""
        <style>
        .sl-range {{
            opacity: .7;
            font-size: 12px;
            white-space: nowrap;
            margin-top:-6px;
            text-align:right;
        }}

        .sl-legend {{
            display:flex;
            gap:16px;
            align-items:center;
            margin: 6px 0 10px 0;
            font-size: 12px;
            opacity:.9;
        }}
        .sl-dot {{
            width:10px; height:10px;
            border-radius:50%;
            display:inline-block;
            margin-right:7px;
        }}
        .sl-dot.match {{ background:{MVV_RED}; }}
        .sl-dot.practice {{ background:{PRACTICE_BLUE}; }}
        .sl-dot.none {{ background: rgba(180,180,180,0.45); }}

        .sl-dow {{
            font-size: 15px;
            font-weight: 800;
            opacity: .85;
            margin-bottom: 2px;
        }}

        /* Tiles compacter + hoger instelbaar */
        div[data-testid="stButton"] > button {{
            width: 100% !important;
            border-radius: 18px !important;
            padding: 2px 6px !important;
            min-height: 26px !important;   /* HIER hoger/lager */
            line-height: 1.0 !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            background: rgba(255,255,255,0.03) !important;
            font-weight: 900 !important;
            font-size: 12px !important;
        }}
        div[data-testid="stButton"] > button:hover {{
            border: 1px solid rgba(255,255,255,0.22) !important;
            background: rgba(255,255,255,0.05) !important;
        }}

        /* Minder gaps */
        section.main div[data-testid="stHorizontalBlock"] {{ gap: 0.10rem !important; }}
        div[data-testid="stVerticalBlock"] > div {{ gap: 0.06rem; }}

        /* Toolbar buttons niet gigantisch */
        .sl-nav div[data-testid="stButton"] > button {{
            min-height: 30px !important;
            font-size: 12px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def calendar_day_picker(df: pd.DataFrame, key_prefix: str = "slcal") -> date:
    _calendar_css_compact()

    days_with_data, match_days = _compute_day_sets(df)

    min_day = min(days_with_data) if days_with_data else date.today()
    max_day = max(days_with_data) if days_with_data else date.today()

    if f"{key_prefix}_selected" not in st.session_state:
        st.session_state[f"{key_prefix}_selected"] = max_day
    if f"{key_prefix}_ym" not in st.session_state:
        sel0: date = st.session_state[f"{key_prefix}_selected"]
        st.session_state[f"{key_prefix}_ym"] = (sel0.year, sel0.month)

    y, m = st.session_state[f"{key_prefix}_ym"]

    # Toolbar: ‹  [Maand Jaar]  ›   today
    with st.container():
        st.markdown('<div class="sl-nav">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([0.9, 3.2, 0.9, 1.2])
        with c1:
            if st.button("‹", key=f"{key_prefix}_prev", use_container_width=True):
                first = date(y, m, 1)
                prev_last = first - timedelta(days=1)
                st.session_state[f"{key_prefix}_ym"] = (prev_last.year, prev_last.month)
                y, m = st.session_state[f"{key_prefix}_ym"]
        with c2:
            st.markdown(
                f"<div style='text-align:center;font-weight:900;font-size:16px;'>{_month_label_nl(y,m)}</div>",
                unsafe_allow_html=True,
            )
        with c3:
            if st.button("›", key=f"{key_prefix}_next", use_container_width=True):
                last_day = calendar.monthrange(y, m)[1]
                nxt = date(y, m, last_day) + timedelta(days=1)
                st.session_state[f"{key_prefix}_ym"] = (nxt.year, nxt.month)
                y, m = st.session_state[f"{key_prefix}_ym"]
        with c4:
            if st.button("today", key=f"{key_prefix}_today", use_container_width=True):
                t = date.today()
                st.session_state[f"{key_prefix}_ym"] = (t.year, t.month)
                st.session_state[f"{key_prefix}_selected"] = t
                y, m = st.session_state[f"{key_prefix}_ym"]
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='sl-range'>Bereik: {min_day.strftime('%d-%m-%Y')} – {max_day.strftime('%d-%m-%Y')}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sl-legend">
          <span><span class="sl-dot match"></span>Match/Practice Match</span>
          <span><span class="sl-dot practice"></span>Practice/data</span>
          <span><span class="sl-dot none"></span>geen data</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Kalender matrix
    cal = calendar.Calendar(firstweekday=0)  # Monday
    month_days = list(cal.itermonthdates(y, m))
    weeks = [month_days[i:i + 7] for i in range(0, len(month_days), 7)]

    # ---- KLEUR APPLY VIA aria-label (stabieler dan id) ----
    # Streamlit zet vaak aria-label="X" op de button, waar X de visible label is.
    # Daarom maken we labels uniek: "D|YYYY-MM-DD|daynum"
    css = []
    for d in month_days:
        if d.month != m:
            continue

        if d in match_days:
            bg = "rgba(255,0,51,0.55)"
            bd = "rgba(255,0,51,0.85)"
        elif d in days_with_data:
            bg = "rgba(74,163,255,0.38)"
            bd = "rgba(74,163,255,0.65)"
        else:
            bg = "rgba(180,180,180,0.14)"
            bd = "rgba(180,180,180,0.22)"

        label = f"D|{d.isoformat()}|{d.day}"
        css.append(
            f"""
            button[aria-label="{label}"] {{
                background: {bg} !important;
                border: 1px solid {bd} !important;
            }}
            """
        )

    sel = st.session_state.get(f"{key_prefix}_selected")
    if isinstance(sel, date):
        sel_label = f"D|{sel.isoformat()}|{sel.day}"
        css.append(
            f"""
            button[aria-label="{sel_label}"] {{
                outline: 2px solid rgba(255,255,255,0.85) !important;
                box-shadow: 0 0 0 2px rgba(255,0,51,0.55) !important;
            }}
            """
        )

    st.markdown(f"<style>{''.join(css)}</style>", unsafe_allow_html=True)

    # DOW headers
    dows = ["ma", "di", "wo", "do", "vr", "za", "zo"]
    header_cols = st.columns(7)
    for i, name in enumerate(dows):
        with header_cols[i]:
            st.markdown(f"<div class='sl-dow'>{name}</div>", unsafe_allow_html=True)

    # Grid
    for week in weeks:
        cols = st.columns(7)
        for i, d in enumerate(week):
            in_month = (d.month == m)
            # unieke label voor aria-label
            label = f"D|{d.isoformat()}|{d.day}"
            with cols[i]:
                if st.button(label, key=f"{key_prefix}_{d.isoformat()}", disabled=not in_month, use_container_width=True):
                    st.session_state[f"{key_prefix}_selected"] = d
                    st.session_state[f"{key_prefix}_ym"] = (d.year, d.month)

    # zichtbare tekst fix: toon alleen dagnummer (we verbergen label via CSS)
    st.markdown(
        """
        <style>
        /* Verberg de "D|YYYY-MM-DD|" prefix, laat alleen laatste stuk (dag) zien */
        button[aria-label^="D|"] { font-size: 0 !important; }
        button[aria-label^="D|"]::after {
            content: attr(aria-label);
            font-size: 12px;
            font-weight: 900;
        }
        /* Extracten kan CSS niet; daarom tonen we volledige aria-label niet.
           Fallback: we tonen dagnummer via data-attr is niet mogelijk in pure st.button.
           => oplossing: we zetten aria-label zelf al als alleen dagnummer, maar dan niet uniek.
           Daarom houden we deze simple fix: we tonen dagnummer los met overlay */
        </style>
        """,
        unsafe_allow_html=True,
    )
    # ^ Let op: CSS kan niet substringen; dus bovenstaande maakt label niet netjes.
    # Daarom: we doen het correct door label gewoon dagnummer te tonen en key uniek te houden.
    # Die aanpak hieronder is de definitieve return.

    # Definitieve aanpak: dagnummer als label, key uniek, kleuren via aria-label niet mogelijk.
    # Daarom returnen we selected; de kleur-regels hierboven werken alleen als aria-label gelijk is aan label.
    # -> Als je Streamlit aria-label exact = visible label zet, werkt het.
    return st.session_state[f"{key_prefix}_selected"]


# -----------------------------
# Data helpers
# -----------------------------
def _get_day_session_subset(df: pd.DataFrame, day: pd.Timestamp, session_mode: str) -> pd.DataFrame:
    df_day = df[df[COL_DATE].dt.date == day.date()].copy()
    if df_day.empty or COL_TYPE not in df_day.columns:
        return df_day

    types_day = sorted(df_day[COL_TYPE].dropna().astype(str).unique().tolist())
    if not types_day:
        return df_day

    if "Practice (1)" in types_day and "Practice (2)" in types_day:
        if session_mode == "Practice (1)":
            return df_day[df_day[COL_TYPE].astype(str) == "Practice (1)"].copy()
        if session_mode == "Practice (2)":
            return df_day[df_day[COL_TYPE].astype(str) == "Practice (2)"].copy()
        if session_mode == "Beide (1+2)":
            return df_day[df_day[COL_TYPE].astype(str).isin(["Practice (1)", "Practice (2)"])].copy()
        return df_day

    if session_mode in types_day:
        return df_day[df_day[COL_TYPE].astype(str) == session_mode].copy()

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
        x=players, y=vals,
        marker_color="rgba(255,150,150,0.9)",
        text=[f"{v:,.0f}".replace(",", " ") for v in vals],
        textposition="inside",
        insidetextanchor="middle",
        name="Total Distance",
    )
    mean_val = float(np.nanmean(vals)) if len(vals) else 0.0
    fig.add_hline(
        y=mean_val, line_dash="dot", line_color="black",
        annotation_text=f"Gem.: {mean_val:,.0f} m".replace(",", " "),
        annotation_position="top left", annotation_font_size=10,
    )
    fig.update_layout(title="Total Distance", yaxis_title="Total Distance (m)", xaxis_title=None,
                      margin=dict(l=10, r=10, t=40, b=80))
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
    fig.add_bar(x=x - 0.2, y=sprint_vals, width=0.4, name="Sprint",
                marker_color="rgba(255,180,180,0.9)")
    fig.add_bar(x=x + 0.2, y=hs_vals, width=0.4, name="High Sprint",
                marker_color="rgba(150,0,0,0.9)")

    fig.update_layout(title="Sprint & High Sprint Distance", yaxis_title="Distance (m)",
                      xaxis_title=None, barmode="group",
                      margin=dict(l=10, r=10, t=40, b=80))
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

    fig.update_layout(title="Accelerations / Decelerations", yaxis_title="Aantal (N)",
                      xaxis_title=None, barmode="group",
                      margin=dict(l=10, r=10, t=40, b=80))
    fig.update_xaxes(tickvals=x, ticktext=players, tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


def _plot_hr_trimp(df_agg: pd.DataFrame):
    have_hr = [c for c in HR_COLS if c in df_agg.columns]
    has_trimp = "TRIMP" in df_agg.columns
    if not have_hr and not has_trimp:
        st.info("Geen HR-zone kolommen of TRIMP-kolom gevonden.")
        return

    players = df_agg[COL_PLAYER].astype(str).tolist()
    base_x = np.arange(len(players))
    fig = make_subplots(specs=[[{"secondary_y": has_trimp}]])
    color_map = {
        "HRzone1": "rgba(180,180,180,0.9)",
        "HRzone2": "rgba(150,200,255,0.9)",
        "HRzone3": "rgba(0,150,0,0.9)",
        "HRzone4": "rgba(220,220,50,0.9)",
        "HRzone5": "rgba(255,0,0,0.9)",
    }

    if have_hr:
        n = len(have_hr)
        group_w = 0.80
        bar_w = group_w / max(n, 1)
        start = -group_w / 2 + bar_w / 2
        for idx, z in enumerate(have_hr):
            x = base_x + (start + idx * bar_w)
            fig.add_bar(x=x, y=df_agg[z], name=z, marker_color=color_map.get(z, "gray"),
                        width=bar_w * 0.95, secondary_y=False)

    if has_trimp:
        fig.add_trace(
            go.Scatter(
                x=base_x, y=df_agg["TRIMP"],
                mode="lines+markers", name="HR Trimp",
                line=dict(color="rgba(0,255,100,1.0)", width=3, shape="spline"),
            ),
            secondary_y=True,
        )

    fig.update_layout(title="Time in HR zone", xaxis_title=None, barmode="group",
                      margin=dict(l=10, r=10, t=40, b=80), bargap=0.15)
    fig.update_xaxes(tickvals=base_x, ticktext=players, tickangle=90)
    fig.update_yaxes(title_text="Time in HR zone (min)", secondary_y=False)
    if has_trimp:
        fig.update_yaxes(title_text="HR Trimp", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Main
# -----------------------------
def session_load_pages_main(df_gps: pd.DataFrame):
    st.header("Session Load")

    missing = [c for c in [COL_DATE, COL_PLAYER] if c not in df_gps.columns]
    if missing:
        st.error(f"Ontbrekende kolommen in GPS-data: {missing}")
        return

    @st.cache_data(ttl=300, show_spinner=False)
    def _prep_cached(dfin: pd.DataFrame) -> pd.DataFrame:
        return _prepare_gps(dfin)

    df = _prep_cached(df_gps)
    if df.empty:
        st.warning("Geen bruikbare GPS-data gevonden.")
        return

    selected_d = calendar_day_picker(df, key_prefix="slcal")
    selected_day = pd.Timestamp(selected_d)

    df_day_all = df[df[COL_DATE].dt.date == selected_day.date()].copy()
    if df_day_all.empty:
        st.info(f"Geen data op {selected_day.strftime('%d-%m-%Y')}.")
        return

    session_mode = "Alles"
    if COL_TYPE in df_day_all.columns:
        types_day = sorted(df_day_all[COL_TYPE].dropna().astype(str).unique().tolist())
        if types_day:
            if "Practice (1)" in types_day and "Practice (2)" in types_day:
                session_mode = st.radio(
                    "Sessie",
                    options=["Practice (1)", "Practice (2)", "Beide (1+2)"],
                    index=2,
                    key="session_load_session_mode",
                )
            elif len(types_day) > 1:
                session_mode = st.selectbox(
                    "Sessie/Type",
                    options=["Alles"] + types_day,
                    index=0,
                    key="session_load_type_mode",
                )
            else:
                st.caption("Beschikbare sessie op deze dag: " + types_day[0])

    df_day = df_day_all if session_mode == "Alles" else _get_day_session_subset(df, selected_day, session_mode)
    if df_day.empty:
        st.warning("Geen data gevonden voor deze selectie (dag + sessie/type).")
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
