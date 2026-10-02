from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from acwr_settings import ACWR_MODE_LOG_6W, ACWR_MODE_META, ACWR_MODE_STANDARD
from auth_session import ensure_auth_restored, get_sb_client
from pages.Subscripts.acwr_shared import (
    ACWR_HIGH,
    ACWR_LOW,
    ACWR_METRICS,
    build_player_monitor,
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


st.set_page_config(page_title="Speler Monitor", layout="wide", initial_sidebar_state="expanded")
apply_streamlit_chrome()
BG_URI = build_data_uri(TEAM_HERO_BG)
LOGO_URI = build_data_uri(TEAM_LOGO)


def render_css() -> None:
    background = f"linear-gradient(180deg,rgba(6,10,20,.84),rgba(6,10,20,.82)),url('{BG_URI}')" if BG_URI else "linear-gradient(180deg,#070c18,#0a1020)"
    st.markdown(
        """
        <style>
        .stApp{background:__BG__;background-size:cover;background-position:center top;background-attachment:fixed}
        .block-container{max-width:1380px;padding-top:1.25rem;padding-bottom:2.4rem}
        .monitor-head{display:flex;align-items:center;justify-content:center;gap:1rem;margin-bottom:1rem}.monitor-head img{width:64px;height:64px;object-fit:contain}.monitor-head h1{margin:0;color:#fff;font-size:2.2rem}.monitor-head p{margin:.25rem 0 0;color:rgba(255,255,255,.68);font-weight:700}
        div[data-testid="stVerticalBlock"]:has(.monitor-panel-anchor){padding:1rem;border:1px solid rgba(255,255,255,.08);border-radius:10px;background:linear-gradient(135deg,rgba(18,25,42,.92),rgba(10,15,27,.88));margin-bottom:1rem}
        </style>
        """.replace("__BG__", background),
        unsafe_allow_html=True,
    )


def remaining_chart(monitor: pd.DataFrame, metric: str) -> go.Figure:
    spec = ACWR_METRICS[metric]
    low_col, high_col, ratio_col = f"{metric}_to_low", f"{metric}_to_high", f"{metric}_acwr"
    data = monitor.dropna(subset=[f"{metric}_chronic"]).sort_values(low_col, ascending=False).copy()
    colors = np.where(data[ratio_col] < ACWR_LOW, "#EDB45B", np.where(data[ratio_col] <= ACWR_HIGH, "#3FC58A", "#E84664"))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data["player_name"], y=data[low_col], name="Nog nodig tot 0,80", marker_color=colors))
    fig.add_trace(go.Bar(x=data["player_name"], y=data[high_col], name="Ruimte tot 1,50", marker_color="#526986"))
    fig.update_layout(
        height=410,
        title=dict(text=spec["label"], x=.02, font=dict(size=18, color="#F8FAFC")),
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.012)",
        font=dict(color="#F8FAFC"),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=30, r=20, t=60, b=100),
        yaxis=dict(title="Resterende meters", gridcolor="rgba(255,255,255,.10)"),
        xaxis=dict(tickangle=-55, showgrid=False),
        hovermode="x unified",
    )
    return fig


def metric_table(monitor: pd.DataFrame, metric: str) -> pd.DataFrame:
    spec = ACWR_METRICS[metric]
    show = monitor[["player_name", metric, f"{metric}_chronic", f"{metric}_acwr", f"{metric}_to_low", f"{metric}_to_high"]].copy()
    show.columns = ["Speler", "Huidige week", "Chronisch", "ACWR", "Nog tot 0,80", "Ruimte tot 1,50"]
    for column in ("Huidige week", "Chronisch", "Nog tot 0,80", "Ruimte tot 1,50"):
        show[column] = show[column].map(lambda value: "–" if pd.isna(value) else f"{float(value):,.0f} m".replace(",", "."))
    show["ACWR"] = show["ACWR"].map(lambda value: "–" if pd.isna(value) else f"{float(value):.2f}")
    show.attrs["metric_label"] = spec["label"]
    return show


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
    st.markdown(f'<div class="monitor-head">{logo}<div><h1>Speler Monitor</h1><p>Resterende ACWR-load in de huidige week</p></div></div>', unsafe_allow_html=True)

    mode = st.selectbox(
        "Rekenmethode",
        options=[ACWR_MODE_LOG_6W, ACWR_MODE_STANDARD],
        format_func=lambda value: ACWR_MODE_META[value]["label"],
        key="player_monitor_mode",
    )
    today = date.today()
    week_start = current_week_start(today)
    try:
        with st.spinner("Spelersmonitor laden..."):
            raw = load_acwr_gps_cached(str(token), default_history_start(today).isoformat(), today.isoformat())
            positions = fetch_player_positions_cached(sb, str(token))
            weekly = build_weekly_loads(exclude_goalkeepers(raw, positions), through_week=week_start)
            monitor = build_player_monitor(weekly, week_start=week_start, mode=mode)
    except Exception as exc:
        st.error(f"Speler Monitor kon niet worden geladen: {exc}")
        st.stop()
    if monitor.empty:
        st.info("Nog onvoldoende GPS-historie voor de Speler Monitor.")
        st.stop()

    st.info("Ondergrens 0,80: wat een speler minimaal nog moet halen. Bovengrens 1,50: hoeveel ruimte er maximaal nog over is.")
    tabs = st.tabs([spec["short"] for spec in ACWR_METRICS.values()])
    for tab, metric in zip(tabs, ACWR_METRICS):
        with tab:
            st.markdown('<div class="monitor-panel-anchor"></div>', unsafe_allow_html=True)
            st.plotly_chart(remaining_chart(monitor, metric), width="stretch", config={"displayModeBar": False})
            st.dataframe(metric_table(monitor, metric), width="stretch", hide_index=True)
    render_sidebar_footer(profile)


if __name__ == "__main__":
    main()

