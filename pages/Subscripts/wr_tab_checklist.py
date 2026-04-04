from __future__ import annotations

from datetime import date
from typing import Dict

import pandas as pd
import streamlit as st

from pages.Subscripts.wr_common import (
    fetch_asrm_range_cached,
    fetch_rpe_headers_range_cached,
)


def _today_local() -> date:
    return date.today()


def _sort_missing_then_alpha(df: pd.DataFrame, missing_col: str, name_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["_missing_sort"] = tmp[missing_col].astype(bool).map({True: 0, False: 1})
    tmp = tmp.sort_values(["_missing_sort", name_col], ascending=[True, True]).drop(columns=["_missing_sort"])
    return tmp


def render_wellness_rpe_tab_checklist(sb, sb_url_key: str, pid_to_name: Dict[str, str]) -> None:
    st.subheader("Checklist (Team)")
    d = st.date_input("Datum", value=_today_local(), key="wr_chk_date")

    players_df = pd.DataFrame({
        "player_id": list(pid_to_name.keys()),
        "Player": list(pid_to_name.values()),
    })

    if players_df.empty:
        st.info("Geen spelers.")
        return

    asrm = fetch_asrm_range_cached(sb_url_key, sb, d.isoformat(), d.isoformat())
    asrm_pids = set(asrm["player_id"].astype(str).tolist()) if not asrm.empty else set()

    wellness_tbl = players_df.copy()
    wellness_tbl["Filled"] = wellness_tbl["player_id"].astype(str).isin(asrm_pids)
    wellness_tbl["Missing"] = ~wellness_tbl["Filled"]
    wellness_tbl = _sort_missing_then_alpha(wellness_tbl, missing_col="Missing", name_col="Player")

    wellness_show = wellness_tbl[["Player", "Filled"]].copy()
    wellness_show["Filled"] = wellness_show["Filled"].map({True: "✅", False: "❌"})

    st.markdown("### Wellness (ASRM)")
    st.dataframe(wellness_show, use_container_width=True, hide_index=True)

    headers = fetch_rpe_headers_range_cached(sb_url_key, sb, d.isoformat(), d.isoformat())
    rpe_pids = set(headers["player_id"].astype(str).tolist()) if not headers.empty else set()

    rpe_tbl = players_df.copy()
    rpe_tbl["Filled"] = rpe_tbl["player_id"].astype(str).isin(rpe_pids)
    rpe_tbl["Missing"] = ~rpe_tbl["Filled"]
    rpe_tbl = _sort_missing_then_alpha(rpe_tbl, missing_col="Missing", name_col="Player")

    rpe_show = rpe_tbl[["Player", "Filled"]].copy()
    rpe_show["Filled"] = rpe_show["Filled"].map({True: "✅", False: "❌"})

    st.markdown("### RPE")
    st.dataframe(rpe_show, use_container_width=True, hide_index=True)

    miss_w = int(wellness_tbl["Missing"].sum())
    miss_r = int(rpe_tbl["Missing"].sum())
    st.caption(f"Ontbrekend — Wellness: {miss_w} | RPE: {miss_r}")
