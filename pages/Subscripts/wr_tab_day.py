from __future__ import annotations

from datetime import date
from typing import Dict

import pandas as pd
import streamlit as st

from subscripts.wr_common import (
    ASRM_PARAMS,
    RPE_PARAMS,
    ASRM_RED_THRESHOLD,
    RPE_RED_THRESHOLD,
    fetch_asrm_date_cached,
    fetch_rpe_headers_date_cached,
    build_rpe_player_daily,
    plot_day_bars,
)


def render_wellness_rpe_tab_day(sb, sb_url_key: str, pid_to_name: Dict[str, str]) -> None:
    d = st.date_input("Datum", value=date.today(), key="wr_day_date")

    cat = st.radio("Categorie", ["Wellness (ASRM)", "RPE"], horizontal=True, key="wr_day_cat")

    if cat == "Wellness (ASRM)":
        param_label = st.selectbox("Parameter", [x[0] for x in ASRM_PARAMS], key="wr_day_asrm_param")
        param_key = dict(ASRM_PARAMS)[param_label]

        asrm = fetch_asrm_date_cached(sb_url_key, sb, d.isoformat())
        if asrm.empty:
            st.info("Geen ASRM entries voor deze datum.")
            return

        asrm["Player"] = asrm["player_id"].map(pid_to_name).fillna(asrm["player_id"])
        asrm[param_key] = pd.to_numeric(asrm[param_key], errors="coerce")

        red = asrm.loc[asrm[param_key] >= ASRM_RED_THRESHOLD, ["Player", param_key]].dropna()
        if not red.empty:
            red = red.sort_values(param_key, ascending=False)
            names = ", ".join([f"{r['Player']} ({int(r[param_key])})" for _, r in red.iterrows()])
            st.error(f"Rood ({param_label} ≥ {ASRM_RED_THRESHOLD}): {names}")

        plot_day_bars(
            asrm.sort_values(param_key, ascending=False),
            x_col="Player",
            y_col=param_key,
            y_title=f"{param_label} (0–10)",
            zone_0_10=True,
        )
        return

    # RPE
    param_label = st.selectbox("Parameter", [x[0] for x in RPE_PARAMS], key="wr_day_rpe_param")
    param_key = dict(RPE_PARAMS)[param_label]

    headers = fetch_rpe_headers_date_cached(sb_url_key, sb, d.isoformat())
    daily = build_rpe_player_daily(sb_url_key, sb, headers)
    if daily.empty:
        st.info("Geen RPE entries voor deze datum.")
        return

    daily["Player"] = daily["player_id"].map(pid_to_name).fillna(daily["player_id"])
    daily[param_key] = pd.to_numeric(daily[param_key], errors="coerce")

    if param_key == "avg_rpe":
        red = daily.loc[daily["avg_rpe"] >= RPE_RED_THRESHOLD, ["Player", "avg_rpe"]].dropna()
        if not red.empty:
            red = red.sort_values("avg_rpe", ascending=False)
            names = ", ".join([f"{r['Player']} ({r['avg_rpe']:.1f})" for _, r in red.iterrows()])
            st.error(f"Rood (avg RPE ≥ {RPE_RED_THRESHOLD}): {names}")

    plot_day_bars(
        daily.sort_values(param_key, ascending=False),
        x_col="Player",
        y_col=param_key,
        y_title=param_label,
        zone_0_10=(param_key == "avg_rpe"),
    )
