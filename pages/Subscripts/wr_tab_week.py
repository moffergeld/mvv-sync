from __future__ import annotations

from datetime import date
from typing import Dict

import pandas as pd
import streamlit as st

from subscripts.wr_common import (
    ASRM_PARAMS,
    RPE_PARAMS,
    iso_week_start_end,
    fetch_asrm_range_cached,
    fetch_rpe_headers_range_cached,
    build_rpe_player_daily,
    agg_week_player_mean_std,
    plot_week_player_mean_std_bars,
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

    if cat == "Wellness (ASRM)":
        param_label = st.selectbox("Parameter", [x[0] for x in ASRM_PARAMS], key="wr_week_asrm_param")
        param_key = dict(ASRM_PARAMS)[param_label]

        asrm = fetch_asrm_range_cached(sb_url_key, sb, d0.isoformat(), d1.isoformat())
        if asrm.empty:
            st.info("Geen ASRM entries in deze week.")
            return

        asrm[param_key] = pd.to_numeric(asrm[param_key], errors="coerce")
        asrm = asrm.dropna(subset=["player_id", "entry_date", param_key])

        stats = agg_week_player_mean_std(asrm, value_col=param_key)
        if stats.empty:
            st.info("Geen bruikbare ASRM data in deze week.")
            return

        stats["Player"] = stats["player_id"].map(pid_to_name).fillna(stats["player_id"])
        stats = stats.sort_values("mean", ascending=False)

        plot_week_player_mean_std_bars(
            df_stats=stats,
            player_name_col="Player",
            y_title=f"{param_label} — mean ± std (week)",
            zone_0_10=True,
        )
        return

    # RPE
    param_label = st.selectbox("Parameter", [x[0] for x in RPE_PARAMS], key="wr_week_rpe_param")
    param_key = dict(RPE_PARAMS)[param_label]

    headers = fetch_rpe_headers_range_cached(sb_url_key, sb, d0.isoformat(), d1.isoformat())
    daily = build_rpe_player_daily(sb_url_key, sb, headers)
    if daily.empty:
        st.info("Geen RPE entries in deze week.")
        return

    daily[param_key] = pd.to_numeric(daily[param_key], errors="coerce")
    daily = daily.dropna(subset=["player_id", "entry_date", param_key])

    stats = agg_week_player_mean_std(daily, value_col=param_key)
    if stats.empty:
        st.info("Geen bruikbare RPE data in deze week.")
        return

    stats["Player"] = stats["player_id"].map(pid_to_name).fillna(stats["player_id"])
    stats = stats.sort_values("mean", ascending=False)

    plot_week_player_mean_std_bars(
        df_stats=stats,
        player_name_col="Player",
        y_title=f"{param_label} — mean ± std (week)",
        zone_0_10=(param_key == "avg_rpe"),
    )
