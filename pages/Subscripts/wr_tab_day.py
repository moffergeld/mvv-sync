from __future__ import annotations

from datetime import date
from typing import Dict

import pandas as pd
import streamlit as st

from pages.Subscripts.wr_common import (
    ASRM_PARAMS,
    ASRM_RED_THRESHOLD,
    RPE_PARAMS,
    RPE_RED_THRESHOLD,
    build_rpe_player_daily,
    create_mvv_bar_chart,
    fetch_asrm_date_cached,
    fetch_rpe_headers_date_cached,
)

DAY_CHART_HEIGHT = 520
DAY_CHART_TICK_ANGLE = -90


def _notice_asrm(df: pd.DataFrame, param_label: str, param_key: str) -> None:
    red = df.loc[df[param_key] >= ASRM_RED_THRESHOLD, ["Player", param_key]].dropna()
    if red.empty:
        return
    red = red.sort_values(param_key, ascending=False)
    names = ", ".join([f"{r['Player']} ({int(r[param_key])})" for _, r in red.iterrows()])
    st.error(f"Rood ({param_label} >= {ASRM_RED_THRESHOLD}): {names}")


def _notice_rpe_avg(df: pd.DataFrame) -> None:
    red = df.loc[df["avg_rpe"] >= RPE_RED_THRESHOLD, ["Player", "avg_rpe"]].dropna()
    if red.empty:
        return
    red = red.sort_values("avg_rpe", ascending=False)
    names = ", ".join([f"{r['Player']} ({r['avg_rpe']:.1f})" for _, r in red.iterrows()])
    st.error(f"Rood (avg RPE >= {RPE_RED_THRESHOLD}): {names}")


def _render_asrm_metric_chart(asrm: pd.DataFrame, param_label: str, param_key: str) -> None:
    st.markdown(f'<div class="mvv-section-label">{param_label}</div>', unsafe_allow_html=True)

    metric_df = asrm.copy()
    metric_df[param_key] = pd.to_numeric(metric_df[param_key], errors="coerce")
    metric_df = metric_df.dropna(subset=[param_key]).copy()

    if metric_df.empty:
        st.info(f"Geen data voor {param_label}.")
        return

    _notice_asrm(metric_df, param_label, param_key)

    df_sorted = metric_df.sort_values(param_key, ascending=False)
    fig = create_mvv_bar_chart(
        df=df_sorted,
        x_col="Player",
        y_col=param_key,
        title="",
        show_zones=True,
        y_range=(0, 10),
        height_override=DAY_CHART_HEIGHT,
        x_tick_angle=DAY_CHART_TICK_ANGLE,
    )
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )


def render_wellness_rpe_tab_day(sb, sb_url_key: str, pid_to_name: Dict[str, str]) -> None:
    d = st.date_input("Datum", value=date.today(), key="wr_day_date")
    cat = st.radio("Categorie", ["Wellness (ASRM)", "RPE"], horizontal=True, key="wr_day_cat")

    st.markdown('<div class="mvv-card">', unsafe_allow_html=True)

    if cat == "Wellness (ASRM)":
        asrm = fetch_asrm_date_cached(sb_url_key, sb, d.isoformat())
        if asrm.empty:
            st.info("Geen ASRM entries voor deze datum.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        asrm["Player"] = asrm["player_id"].map(pid_to_name).fillna(asrm["player_id"])

        physical_params = [item for item in ASRM_PARAMS if item[1] in {"muscle_soreness", "fatigue"}]
        mental_params = [item for item in ASRM_PARAMS if item[1] in {"sleep_quality", "stress", "mood"}]

        st.markdown('<div class="mvv-section-label">Physical</div>', unsafe_allow_html=True)
        physical_cols = st.columns(2, gap="large")
        for col, (param_label, param_key) in zip(physical_cols, physical_params):
            with col:
                _render_asrm_metric_chart(asrm, param_label, param_key)

        st.markdown('<div class="mvv-section-label">Mental</div>', unsafe_allow_html=True)
        mental_cols = st.columns(3, gap="small")
        for col, (param_label, param_key) in zip(mental_cols, mental_params):
            with col:
                _render_asrm_metric_chart(asrm, param_label, param_key)

    else:
        try:
            headers = fetch_rpe_headers_date_cached(sb_url_key, sb, d.isoformat())
            daily = build_rpe_player_daily(sb_url_key, sb, headers)
        except Exception as exc:
            st.error(f"RPE-data kon niet geladen worden: {exc}")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        if daily.empty:
            st.info("Geen RPE entries voor deze datum.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        daily["Player"] = daily["player_id"].map(pid_to_name).fillna(daily["player_id"])
        _notice_rpe_avg(daily)

        for param_label, param_key in RPE_PARAMS:
            st.markdown(f'<div class="mvv-section-label">{param_label}</div>', unsafe_allow_html=True)

            daily[param_key] = pd.to_numeric(daily[param_key], errors="coerce")
            dfp = daily.dropna(subset=[param_key]).copy()

            if dfp.empty:
                st.info(f"Geen data voor {param_label}.")
                st.markdown("---")
                continue

            df_sorted = dfp.sort_values(param_key, ascending=False)
            fig = create_mvv_bar_chart(
                df=df_sorted,
                x_col="Player",
                y_col=param_key,
                title="",
                show_zones=(param_key == "avg_rpe"),
                y_range=(0, 10) if param_key == "avg_rpe" else None,
                range_min_col="min_rpe" if param_key == "avg_rpe" else None,
                range_max_col="max_rpe" if param_key == "avg_rpe" else None,
                height_override=DAY_CHART_HEIGHT,
                x_tick_angle=DAY_CHART_TICK_ANGLE,
            )
            st.plotly_chart(
                fig,
                width="stretch",
                config={"displayModeBar": False, "responsive": True},
            )
            st.markdown("---")

    st.markdown("</div>", unsafe_allow_html=True)
