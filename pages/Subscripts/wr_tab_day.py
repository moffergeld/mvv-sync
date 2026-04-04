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
    # WELLNESS (ASRM) — ALL PARAMS
    # -------------------------
    if cat == "Wellness (ASRM)":
        asrm = fetch_asrm_date_cached(sb_url_key, sb, d.isoformat())
        if asrm.empty:
            st.info("Geen ASRM entries voor deze datum.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        asrm["Player"] = asrm["player_id"].map(pid_to_name).fillna(asrm["player_id"])

        for param_label, param_key in ASRM_PARAMS:
            st.markdown(f'<div class="mvv-section-label">{param_label}</div>', unsafe_allow_html=True)

            asrm[param_key] = pd.to_numeric(asrm[param_key], errors="coerce")
            dfp = asrm.dropna(subset=[param_key]).copy()

            _notice_asrm(dfp, param_label, param_key)

            # Create styled chart
            df_sorted = dfp.sort_values(param_key, ascending=False)
            fig = create_mvv_bar_chart(
                df=df_sorted,
                x_col="Player",
                y_col=param_key,
                title=f"{param_label} (0–10)",
                show_zones=True,
                y_range=(0, 10)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")

    # -------------------------
    # RPE — ALL PARAMS
    # -------------------------
    else:  # RPE
        headers = fetch_rpe_headers_date_cached(sb_url_key, sb, d.isoformat())
        daily = build_rpe_player_daily(sb_url_key, sb, headers)
        if daily.empty:
            st.info("Geen RPE entries voor deze datum.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        daily["Player"] = daily["player_id"].map(pid_to_name).fillna(daily["player_id"])

        # notice alleen op avg_rpe
        _notice_rpe_avg(daily)

        for param_label, param_key in RPE_PARAMS:
            st.markdown(f'<div class="mvv-section-label">{param_label}</div>', unsafe_allow_html=True)

            daily[param_key] = pd.to_numeric(daily[param_key], errors="coerce")
            dfp = daily.dropna(subset=[param_key]).copy()

            # Create styled chart
            df_sorted = dfp.sort_values(param_key, ascending=False)
            fig = create_mvv_bar_chart(
                df=df_sorted,
                x_col="Player",
                y_col=param_key,
                title=param_label,
                show_zones=(param_key == "avg_rpe"),
                y_range=(0, 10) if (param_key == "avg_rpe") else None
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")

    st.markdown('</div>', unsafe_allow_html=True)
