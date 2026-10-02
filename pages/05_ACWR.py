from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from acwr_settings import ACWR_MODE_LOG_6W, ACWR_MODE_META, ACWR_MODE_STANDARD
from auth_session import ensure_auth_restored, get_sb_client
from pages.Subscripts.acwr_shared import (
    ACWR_METRICS,
    add_acwr_columns,
    build_weekly_loads,
    current_week_start,
    default_history_start,
    exclude_goalkeepers,
    fetch_player_positions_cached,
    load_acwr_gps_cached,
)
from pages.Subscripts.mvv_branding import TEAM_HERO_BG, TEAM_LOGO, build_data_uri
from roles import get_profile, is_staff_user, render_sidebar_footer, render_sidebar_navigation
from utils.streamlit_ui import apply_dashboard_polish, apply_streamlit_chrome


st.set_page_config(page_title="ACWR", layout="wide", initial_sidebar_state="expanded")
apply_streamlit_chrome()

BG_URI = build_data_uri(TEAM_HERO_BG)
LOGO_URI = build_data_uri(TEAM_LOGO)


def render_css() -> None:
    background = (
        f"linear-gradient(180deg, rgba(6,10,20,.84), rgba(6,10,20,.82)), url('{BG_URI}')"
        if BG_URI
        else "linear-gradient(180deg,#070c18,#0a1020)"
    )
    st.markdown(
        """
        <style>
        .stApp{background:__BG__;background-size:cover;background-position:center top;background-attachment:fixed}
        .block-container{max-width:1380px;padding-top:1.25rem;padding-bottom:2.4rem}
        .acwr-head{display:flex;align-items:center;justify-content:center;gap:1rem;margin-bottom:1.1rem}
        .acwr-head img{width:64px;height:64px;object-fit:contain}
        .acwr-head h1{margin:0;color:#fff;font-size:2.25rem}.acwr-head p{margin:.25rem 0 0;color:rgba(255,255,255,.68);font-weight:700}
        div[data-testid="stVerticalBlock"]:has(.acwr-panel-anchor){padding:1rem 1.05rem;border:1px solid rgba(255,255,255,.08);border-radius:10px;background:linear-gradient(135deg,rgba(18,25,42,.92),rgba(10,15,27,.88));margin-bottom:1rem}
        </style>
        """.replace("__BG__", background),
        unsafe_allow_html=True,
    )


def metric_chart(player_df: pd.DataFrame, metric: str) -> go.Figure:
    spec = ACWR_METRICS[metric]
    ratio_col = f"{metric}_acwr"
    data = player_df.dropna(subset=[ratio_col]).sort_values("week_start").tail(14)
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=.8, fillcolor="#E84664", opacity=.12, line_width=0)
    fig.add_hrect(y0=.8, y1=1.3, fillcolor="#3FC58A", opacity=.14, line_width=0)
    fig.add_hrect(y0=1.3, y1=1.5, fillcolor="#EDB45B", opacity=.14, line_width=0)
    fig.add_hrect(y0=1.5, y1=max(1.8, float(data[ratio_col].max()) * 1.12 if not data.empty else 1.8), fillcolor="#E84664", opacity=.12, line_width=0)
    fig.add_trace(
        go.Scatter(
            x=data["week_label"],
            y=data[ratio_col],
            mode="lines+markers",
            name=spec["short"],
            line=dict(color=spec["color"], width=3),
            marker=dict(size=8, line=dict(color="#101827", width=1.5)),
            customdata=np.column_stack([data[metric], data[f"{metric}_chronic"]]) if not data.empty else None,
            hovertemplate="%{x}<br>ACWR %{y:.2f}<br>Acute %{customdata[0]:,.0f} m<br>Chronisch %{customdata[1]:,.0f} m<extra></extra>",
        )
    )
    fig.add_hline(y=.8, line_dash="dot", line_color="#F8FAFC", opacity=.7)
    fig.add_hline(y=1.5, line_dash="dot", line_color="#F8FAFC", opacity=.7)
    fig.update_layout(
        height=350,
        title=dict(text=spec["label"], x=.02, font=dict(size=18, color="#F8FAFC")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.012)",
        font=dict(color="#F8FAFC"),
        margin=dict(l=30, r=20, t=55, b=55),
        showlegend=False,
        hovermode="x unified",
        yaxis=dict(title="ACWR", range=[0, max(1.8, float(data[ratio_col].max()) * 1.12 if not data.empty else 1.8)], gridcolor="rgba(255,255,255,.10)"),
        xaxis=dict(tickangle=-25, showgrid=False),
    )
    return fig


def main() -> None:
    render_css()
    apply_dashboard_polish()
    sb = get_sb_client()
    ok, token = ensure_auth_restored(sb)
    if not ok or not token:
        st.switch_page("app.py")
        st.stop()
    profile = get_profile(sb)
    if not is_staff_user(profile):
        st.error("Geen toegang: deze pagina is alleen voor staff.")
        st.stop()
    render_sidebar_navigation(profile)

    logo = f'<img src="{LOGO_URI}" alt="MVV" />' if LOGO_URI else ""
    st.markdown(f'<div class="acwr-head">{logo}<div><h1>ACWR</h1><p>TD · Zone 5 · Zone 6 · HMLD</p></div></div>', unsafe_allow_html=True)

    mode = st.selectbox(
        "Rekenmethode",
        options=[ACWR_MODE_LOG_6W, ACWR_MODE_STANDARD],
        format_func=lambda value: ACWR_MODE_META[value]["label"],
        key="acwr_page_mode",
    )
    today = date.today()
    try:
        with st.spinner("ACWR-historie laden..."):
            raw = load_acwr_gps_cached(str(token), default_history_start(today).isoformat(), today.isoformat())
            positions = fetch_player_positions_cached(sb, str(token))
            weekly = build_weekly_loads(exclude_goalkeepers(raw, positions), through_week=current_week_start(today))
            history = add_acwr_columns(weekly, mode)
    except Exception as exc:
        st.error(f"ACWR-data kon niet worden geladen: {exc}")
        st.stop()
    if history.empty:
        st.info("Nog geen bruikbare Entire Session - Live-data voor ACWR.")
        st.stop()

    players = sorted(history["player_name"].dropna().astype(str).unique(), key=str.lower)
    selected = st.selectbox("Speler", players, key="acwr_page_player")
    player_df = history[history["player_name"].astype(str) == str(selected)].copy()
    current = player_df[player_df["week_start"] == current_week_start(today)]
    current_row = current.iloc[-1] if not current.empty else None
    cards = st.columns(4)
    for container, (metric, spec) in zip(cards, ACWR_METRICS.items()):
        value = current_row.get(f"{metric}_acwr") if current_row is not None else np.nan
        container.metric(spec["short"], "–" if pd.isna(value) else f"{float(value):.2f}")

    rows = list(ACWR_METRICS)
    for offset in range(0, len(rows), 2):
        cols = st.columns(2, gap="large")
        for container, metric in zip(cols, rows[offset:offset + 2]):
            with container:
                st.markdown('<div class="acwr-panel-anchor"></div>', unsafe_allow_html=True)
                st.plotly_chart(metric_chart(player_df, metric), width="stretch", config={"displayModeBar": False})
    st.caption(f"{ACWR_MODE_META[mode]['description']} De lopende kalenderweek is de acute belasting en is uitgesloten van de chronische referentie.")
    render_sidebar_footer(profile)


if __name__ == "__main__":
    main()

