# pages/Subscripts/wr_tab_day.py (aangepast met MVV styling)
from __future__ import annotations

from datetime import date
from typing import Dict

import pandas as pd
import streamlit as st

from pages.Subscripts.wr_common import (
    ASRM_PARAMS,
    RPE_PARAMS,
    ASRM_RED_THRESHOLD,
    RPE_RED_THRESHOLD,
    fetch_asrm_date_cached,
    fetch_rpe_headers_date_cached,
    build_rpe_player_daily,
    plot_day_bars,
    create_mvv_bar_chart
)

def _notice_asrm(df: pd.DataFrame, param_label: str, param_key: str) -> None:
    red = df.loc[df[param_key] >= ASRM_RED_THRESHOLD, ["Player", param_key]].dropna()
    if red.empty:
        return
    red = red.sort_values(param_key, ascending=False)
    names = ", ".join([f"{r['Player']} ({int(r[param_key])})" for _, r in red.iterrows()])
    st.error(f"🔴 Rood ({param_label} ≥ {ASRM_RED_THRESHOLD}): {names}")

def _notice_rpe_avg(df: pd.DataFrame) -> None:
    red = df.loc[df["avg_rpe"] >= RPE_RED_THRESHOLD, ["Player", "avg_rpe"]].dropna()
    if red.empty:
        return
    red = red.sort_values("avg_rpe", ascending=False)
    names = ", ".join([f"{r['Player']} ({r['avg_rpe']:.1f})" for _, r in red.iterrows()])
    st.error(f"🔴 Rood (avg RPE ≥ {RPE_RED_THRESHOLD}): {names}")

def render_wellness_rpe_tab_day(sb, sb_url_key: str, pid_to_name: Dict[str, str]) -> None:
    d = st.date_input("Datum", value=date.today(), key="wr_day_date")

    cat = st.radio("Categorie", ["Wellness (ASRM)", "RPE"], horizontal=True, key="wr_day_cat")

    # Wrap content in glass card
    st.markdown('<div class="mvv-card">', unsafe_allow_html=True)

    # -------------------------
    # WELLNESS (AS
