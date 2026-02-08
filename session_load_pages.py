# session_load_pages.py
# ==========================================
# Session Load dashboard
# - Selecteer dag via slider
# - Kies sessie: Practice (1) / Practice (2) / beide
# - 4 grafieken per speler:
#   * Total Distance
#   * Sprint & High Sprint
#   * Accelerations / Decelerations
#   * Time in HR zones + HR Trimp
# ==========================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Kolomnamen (pas aan als nodig)
COL_DATE   = "Datum"
COL_PLAYER = "Speler"
COL_EVENT  = "Event"
COL_TYPE   = "Type"

COL_TD      = "Total Distance"
COL_SPRINT  = "Sprint"
COL_HS      = "High Sprint"
COL_ACC_TOT = "Total Accelerations"
COL_ACC_HI  = "High Accelerations"
COL_DEC_TOT = "Total Decelerations"
COL_DEC_HI  = "High Decelerations"

HR_COLS = ["HRzone1", "HRzone2", "HRzone3", "HRzone4", "HRzone5"]
TRIMP_CANDIDATES = ["HRTrimp", "HR Trimp", "HRtrimp", "Trimp", "TRIMP"]

# -----------------------------
# Helpers
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

    # Speciale logica voor Practice (1)/(2)
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
        # Anders: alle sessies samen
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

    agg = (
        df.groupby(COL_PLAYER, as_index=False)[metric_cols]
        .sum()
    )
    return agg

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
        textangle=0,
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

    st.plotly_chart(fig, width='stretch')


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
        yaxis_title="Sprint (m)",
        xaxis_title=None,
        barmode="group",
        margin=dict(l=10, r=10, t=40, b=80),
    )
    fig.update_xaxes(
        tickvals=x,
        ticktext=players,
        tickangle=90,
    )

    st.plotly_chart(fig, width='stretch')


def _plot_acc_dec(df_agg: pd.DataFrame):
    have_cols = [c for c in [COL_ACC_TOT, COL_ACC_HI, COL_DEC_TOT, COL_DEC_HI] if c in df_agg.columns]
    if len(have_cols) == 0:
        st.info("Geen Acceleration/Deceleration kolommen gevonden.")
        return

    data = df_agg.sort_values(COL_ACC_TOT if COL_ACC_TOT in df_agg.columns else have_cols[0],
                              ascending=False).reset_index(drop=True)
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
    fig.update_xaxes(
        tickvals=x,
        ticktext=players,
        tickangle=90,
    )

    st.plotly_chart(fig, width='stretch')


def _plot_hr_trimp(df_agg: pd.DataFrame):
    have_hr = [c for c in HR_COLS if c in df_agg.columns]
    has_trimp = "TRIMP" in df_agg.columns

    if not have_hr and not has_trimp:
        st.info("Geen HR-zone kolommen of TRIMP-kolom gevonden.")
        return

    players = df_agg[COL_PLAYER].astype(str).tolist()
    x = np.arange(len(players))

    fig = make_subplots(specs=[[{"secondary_y": has_trimp}]])

    # HR zones gestapelde bars
    color_map = {
        "HRzone1": "rgba(180,180,180,0.9)",
        "HRzone2": "rgba(150,200,255,0.9)",
        "HRzone3": "rgba(0,150,0,0.9)",
        "HRzone4": "rgba(220,220,50,0.9)",
        "HRzone5": "rgba(255,0,0,0.9)",
    }

    for z in have_hr:
        fig.add_bar(
            x=x,
            y=df_agg[z],
            name=z,
            marker_color=color_map.get(z, "gray"),
            secondary_y=False,
        )

    # TRIMP-lijn
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
    fig.update_xaxes(
        tickvals=x,
        ticktext=players,
        tickangle=90,
    )

    fig.update_yaxes(title_text="Time in HR zone (min)", secondary_y=False)
    if has_trimp:
        fig.update_yaxes(title_text="HR Trimp", secondary_y=True)

    st.plotly_chart(fig, width='stretch')


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

    # Beschikbare datums
    dates = sorted(df[COL_DATE].dt.date.unique().tolist())
    if not dates:
        st.warning("Geen datums gevonden in de data.")
        return

    min_date = dates[0]
    max_date = dates[-1]

    # UI: datum slider
    selected_date = st.slider(
        "Selecteer dag",
        min_value=min_date,
        max_value=max_date,
        value=max_date,
        format="DD-MM-YYYY",
    )
    st.markdown(f"**Geselecteerde datum:** {selected_date.strftime('%d-%m-%Y')}")

    # Beschikbare types op deze dag
    df_day_all = df[df[COL_DATE].dt.date == selected_date].copy()
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
            st.markdown(
                "Beschikbare sessies op deze dag: " +
                ", ".join(types_day)
            )
        else:
            st.markdown("_Geen kolom 'Type' of sessies op deze dag gevonden._")

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
