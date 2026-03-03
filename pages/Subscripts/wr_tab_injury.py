# pages/Subscripts/wr_tab_injury.py
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict

import pandas as pd
import streamlit as st

from pages.Subscripts.wr_common import fetch_rpe_injuries_range_cached


def render_wellness_rpe_tab_injury(sb, sb_url_key: str, pid_to_name: Dict[str, str]) -> None:
    """
    Injury tab (staff):
    - toont alle rpe_entries met injury=True in een datumbereik
    - highlight op basis van pain (0–10) via sortering + emoji labels
    """

    st.subheader("Injuries (RPE)")

    c1, c2, c3 = st.columns([1.1, 1.1, 1.0])
    with c1:
        days_back = st.number_input("Periode (dagen terug)", min_value=1, max_value=120, value=14, step=1, key="inj_days")
    with c2:
        min_pain = st.slider("Min. pain", 0, 10, value=0, step=1, key="inj_min_pain")
    with c3:
        only_open = st.toggle("Alleen injury=True", value=True, key="inj_only_true")

    d1 = date.today()
    d0 = d1 - timedelta(days=int(days_back) - 1)
    st.caption(f"Periode: {d0.isoformat()} t/m {d1.isoformat()}")

    df = fetch_rpe_injuries_range_cached(sb_url_key, sb, d0.isoformat(), d1.isoformat())

    if df.empty:
        st.info("Geen injuries gevonden in deze periode.")
        return

    # map player name
    df["Player"] = df["player_id"].map(pid_to_name).fillna(df["player_id"])

    # filters
    df["injury_pain"] = pd.to_numeric(df["injury_pain"], errors="coerce")
    if only_open:
        df = df[df["injury"] == True]  # noqa: E712
    if int(min_pain) > 0:
        df = df[df["injury_pain"].fillna(0) >= int(min_pain)]

    if df.empty:
        st.info("Geen injuries voor deze selectie.")
        return

    # severity label
    def _sev(p):
        try:
            p = float(p)
        except Exception:
            return ""
        if p >= 8:
            return "🔴"
        if p >= 5:
            return "🟠"
        if p > 0:
            return "🟢"
        return ""

    df["Severity"] = df["injury_pain"].apply(_sev)

    # sort: pain desc, date desc, player
    df = df.sort_values(["injury_pain", "entry_date", "Player"], ascending=[False, False, True])

    # show table
    show_cols = [
        "Severity",
        "entry_date",
        "Player",
        "injury_type",
        "injury_pain",
        "notes",
        "attachment_url",
    ]
    show = df[show_cols].copy()
    show = show.rename(
        columns={
            "entry_date": "Date",
            "injury_type": "Type",
            "injury_pain": "Pain",
            "notes": "Notes",
            "attachment_url": "Attachment",
        }
    )

    st.dataframe(show, width="stretch", hide_index=True)

    # quick summary
    n = len(df)
    n_red = int((df["injury_pain"].fillna(-1) >= 8).sum())
    n_orange = int(((df["injury_pain"].fillna(-1) >= 5) & (df["injury_pain"].fillna(-1) < 8)).sum())
    st.caption(f"Totaal: {n} | 🔴 pain≥8: {n_red} | 🟠 pain 5–7: {n_orange}")
