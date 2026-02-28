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
)


def _notice_asrm(df: pd.DataFrame, param_label: str, param_key: str) -> None:
    red = df.loc[df[param_key] >= ASRM_RED_THRESHOLD, ["Player", param_key]].dropna()
    if red.empty:
        return
    red = red.sort_values(param_key, ascending=False)
    names = ", ".join([f"{r['Player']} ({int(r[param_key])})" for _, r in red.iterrows()])
    st.error(f"Rood ({param_label} ≥ {ASRM_RED_THRESHOLD}): {names}")


def _notice_rpe_avg(df: pd.DataFrame) -> None:
    red = df.loc[df["avg_rpe"] >= RPE_RED_THRESHOLD, ["Player", "avg_rpe"]].dropna()
    if red.empty:
        return
    red = red.sort_values("avg_rpe", ascending=False)
    names = ", ".join([f"{r['Player']} ({r['avg_rpe']:.1f})" for _, r in red.iterrows()])
    st.error(f"Rood (avg RPE ≥ {RPE_RED_THRESHOLD}): {names}")


def render_wellness_rpe_tab_day(sb, sb_url_key: str, pid_to_name: Dict[str, str]) -> None:
    d = st.date_input("Datum", value=date.today(), key="wr_day_date")

    cat = st.radio("Categorie", ["Wellness (ASRM)", "RPE"], horizontal=True, key="wr_day_cat")

    # -------------------------
    # WELLNESS (ASRM) — ALL PARAMS
    # -------------------------
    if cat == "Wellness (ASRM)":
        asrm = fetch_asrm_date_cached(sb_url_key, sb, d.isoformat())
        if asrm.empty:
            st.info("Geen ASRM entries voor deze datum.")
            return

        asrm["Player"] = asrm["player_id"].map(pid_to_name).fillna(asrm["player_id"])

        for param_label, param_key in ASRM_PARAMS:
            st.subheader(param_label)

            asrm[param_key] = pd.to_numeric(asrm[param_key], errors="coerce")
            dfp = asrm.dropna(subset=[param_key]).copy()

            _notice_asrm(dfp, param_label, param_key)

            plot_day_bars(
                dfp.sort_values(param_key, ascending=False),
                x_col="Player",
                y_col=param_key,
                y_title=f"{param_label} (0–10)",
                zone_0_10=True,
            )
        return

    # -------------------------
    # RPE — ALL PARAMS
    # -------------------------
    headers = fetch_rpe_headers_date_cached(sb_url_key, sb, d.isoformat())
    daily = build_rpe_player_daily(sb_url_key, sb, headers)
    if daily.empty:
        st.info("Geen RPE entries voor deze datum.")
        return

    daily["Player"] = daily["player_id"].map(pid_to_name).fillna(daily["player_id"])

    # notice alleen op avg_rpe
    _notice_rpe_avg(daily)

    for param_label, param_key in RPE_PARAMS:
        st.subheader(param_label)

        daily[param_key] = pd.to_numeric(daily[param_key], errors="coerce")
        dfp = daily.dropna(subset=[param_key]).copy()

        plot_day_bars(
            dfp.sort_values(param_key, ascending=False),
            x_col="Player",
            y_col=param_key,
            y_title=param_label,
            zone_0_10=(param_key == "avg_rpe"),
        )
