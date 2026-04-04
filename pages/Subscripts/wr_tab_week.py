# pages/Subscripts/wr_tab_week.py (aangepast met MVV styling)
from __future__ import annotations

from datetime import date
from typing import Dict

import pandas as pd
import streamlit as st

from pages.Subscripts.wr_common import (
    ASRM_PARAMS,
    RPE_PARAMS,
    iso_week_start_end,
    fetch_asrm_range_cached,
    fetch_rpe_headers_range_cached,
    build_rpe_player_daily,
    agg_week_player_mean_std,
    create_mvv_bar_chart,
)

def render_wellness_rpe_tab_week(sb, sb_url_key: str, pid_to_name: Dict[str, str]) -> None:
    today = date.today()
    iso_year, iso_week, _ = today.isocalendar()

    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Jaar", min_value=2020, max_value=2100, value=int(iso_year), step=1, key="wr_week_year")
    with c2:
        week = st.number_input("Week (ISO)", min_value=1, max_value=53, value=int(iso_week), step=1, key="wr_week_week")

    d0, d1 = iso_week_start_end(int(year), int(week))
    st.caption(f"Periode: {d0.isoformat()} t/m {d1.isoformat()}")

    cat = st.radio("Categorie", ["Wellness (ASRM)", "RPE"], horizontal=True, key="wr_week_cat")

    # Wrap content in glass card
    st.markdown('<div class="mvv-card">', unsafe_allow_html=True)

    if cat == "Wellness (ASRM)":
        param_label = st.selectbox("Parameter", [x[0] for x in ASRM_PARAMS], key="wr_week_asrm_param")
        param_key = dict(ASRM_PARAMS)[param_label]

        asrm = fetch_asrm_range_cached(sb_url_key, sb, d0.isoformat(), d1.isoformat())
        if asrm.empty:
            st.info("Geen ASRM entries in deze week.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        asrm[param_key] = pd.to_numeric(asrm[param_key], errors="coerce")
        asrm = asrm.dropna(subset=["player_id", "entry_date", param_key])

        stats = agg_week_player_mean_std(asrm, value_col=param_key)
        if stats.empty:
            st.info("Geen bruikbare ASRM data in deze week.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        stats["Player"] = stats["player_id"].map(pid_to_name).fillna(stats["player_id"])
        stats = stats.sort_values("mean", ascending=False)

        # Create styled chart
        fig = create_mvv_bar_chart(
            df=stats,
            x_col="Player",
            y_col="mean",
            title=f"{param_label} — Gemiddelde ± Standaardafwijking (Week {week})",
            show_zones=True,
            y_range=(0, 10)
        )
        
        # Add standard deviation information to the chart
        if 'std' in stats.columns:
            fig.data[0].error_y = dict(
                type='data',
                array=stats['std'],
                color='#E8213F',
                thickness=2,
                width=6
            )
        
        st_
