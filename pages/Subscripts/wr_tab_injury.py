# pages/Subscripts/wr_tab_injury.py
# ============================================================
# Injury tab (Team / Staff)
#
# Doel
# - Toon alle relevante "injury signals" vanuit public.rpe_entries
# - Relevantie = injury=True OF notes gevuld OF pain>0
# - Entries die volledig leeg zijn (injury=False AND notes leeg AND pain<=0) worden altijd weggelaten
#
# UI
# - Periode (dagen terug)
# - Min. pain filter
# - Toggle: Alleen injury=True  (uitgebreid: injury OR notes OR pain)
#
# Output
# - Tabel met Severity (emoji), Date, Player, Type, Pain, Notes, Attachment
# - Samenvatting onderaan
# ============================================================

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict

import pandas as pd
import streamlit as st

from pages.Subscripts.wr_common import fetch_rpe_injuries_range_cached


def render_wellness_rpe_tab_injury(sb, sb_url_key: str, pid_to_name: Dict[str, str]) -> None:
    st.subheader("Injuries (RPE)")

    c1, c2, c3 = st.columns([1.1, 1.1, 1.0])
    with c1:
        days_back = st.number_input(
            "Periode (dagen terug)",
            min_value=1,
            max_value=120,
            value=4,
            step=1,
            key="inj_days",
        )
    with c2:
        min_pain = st.slider("Min. pain", 0, 10, value=0, step=1, key="inj_min_pain")
    with c3:
        only_open = st.toggle("Alleen injury=True", value=True, key="inj_only_true")

    d1 = date.today()
    d0 = d1 - timedelta(days=int(days_back) - 1)
    st.caption(f"Periode: {d0.isoformat()} t/m {d1.isoformat()}")

    df = fetch_rpe_injuries_range_cached(sb_url_key, sb, d0.isoformat(), d1.isoformat())
    if df.empty:
        st.info("Geen entries gevonden in deze periode.")
        return

    # --------------------------------------------------------
    # Normaliseren
    # --------------------------------------------------------
    df["Player"] = df["player_id"].map(pid_to_name).fillna(df["player_id"]).astype(str)

    df["injury"] = df["injury"].astype(bool)
    df["injury_pain"] = pd.to_numeric(df["injury_pain"], errors="coerce").fillna(0)
    df["notes"] = df["notes"].fillna("").astype(str)
    df["_has_notes"] = df["notes"].str.strip().ne("")

    # --------------------------------------------------------
    # Basisregel: drop "lege" rijen
    # leeg = injury=False AND notes leeg AND pain<=0
    # --------------------------------------------------------
    df = df[~((df["injury"] == False) & (~df["_has_notes"]) & (df["injury_pain"] <= 0))]  # noqa: E712

    # --------------------------------------------------------
    # Toggle logica: "Alleen injury=True" betekent:
    # toon injury=True OF notes gevuld OF pain>0
    # --------------------------------------------------------
    if only_open:
        df = df[(df["injury"] == True) | (df["_has_notes"]) | (df["injury_pain"] > 0)]  # noqa: E712

    # --------------------------------------------------------
    # Min pain filter (na bovenstaande)
    # --------------------------------------------------------
    if int(min_pain) > 0:
        df = df[df["injury_pain"] >= int(min_pain)]

    if df.empty:
        st.info("Geen injuries voor deze selectie.")
        return

    # --------------------------------------------------------
    # Severity label
    # --------------------------------------------------------
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
        # pain==0 maar notes/injury kan nog relevant zijn
        return "⚪"

    df["Severity"] = df["injury_pain"].apply(_sev)

    # sort: pain desc, date desc, player
    df = df.sort_values(["injury_pain", "entry_date", "Player"], ascending=[False, False, True])

    # --------------------------------------------------------
    # Tabel output
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    pain = df["injury_pain"].fillna(0)
    n = int(len(df))
    n_red = int((pain >= 8).sum())
    n_orange = int(((pain >= 5) & (pain < 8)).sum())
    n_green = int(((pain > 0) & (pain < 5)).sum())
    n_notes_only = int(((pain <= 0) & (df["_has_notes"])).sum())
    n_injury_flag = int((df["injury"] == True).sum())  # noqa: E712

    st.caption(
        f"Totaal: {n} | injury=True: {n_injury_flag} | 🔴≥8: {n_red} | 🟠5–7: {n_orange} | 🟢1–4: {n_green} | ⚪ notes/pain0: {n_notes_only}"
    )

    # cleanup (not shown, but keeps df clean if reused)
    # (no effect after this function ends, but harmless)
    df.drop(columns=["_has_notes"], inplace=True, errors="ignore")
