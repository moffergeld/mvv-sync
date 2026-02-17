# gps_import_tab_export.py
# ============================================================
# Subtab: Export
# ============================================================

from __future__ import annotations

import io
from datetime import date

import pandas as pd
import requests
import streamlit as st

from gps_import_common import (
    GPS_COLS,
    df_to_excel_bytes_single,
    fetch_all_gps_records,
    rest_get,
    safe_sheet_name,
    toast_err,
    toast_ok,
)


def tab_export_main(access_token: str, player_options: list[str]) -> None:
    st.subheader("Export gps_records → Excel")

    st.markdown("### 1) Export alles")
    c1, c2 = st.columns([1.4, 2.6])
    with c1:
        limit = st.number_input(
            "Max rows (veiligheidslimiet)",
            min_value=1,
            max_value=500000,
            value=200000,
            step=10000,
            key="exp_all_limit",
        )
    with c2:
        st.caption("Klik op generate, daarna verschijnt de downloadknop.")

    if st.button("Generate export (ALL)", key="exp_all_btn"):
        try:
            df = fetch_all_gps_records(access_token, limit=int(limit))
            if df.empty:
                st.warning("Geen data gevonden.")
            else:
                ordered = [c for c in GPS_COLS if c in df.columns]
                df = df[ordered]
                xbytes = df_to_excel_bytes_single(df, sheet_name="gps_records")
                st.session_state["export_all_bytes"] = xbytes
                toast_ok(f"Export bevestigd: {len(df)} rijen klaar voor download.")
        except Exception as e:
            toast_err(str(e))

    if st.session_state.get("export_all_bytes"):
        st.download_button(
            "Download gps_records_ALL.xlsx",
            data=st.session_state["export_all_bytes"],
            file_name="gps_records_ALL.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="exp_all_dl",
        )

    st.divider()
    st.markdown("### 2) Export geselecteerde speler(s) (elk tabblad = speler)")

    export_players = st.multiselect(
        "Selecteer speler(s)",
        options=player_options if player_options else [],
        key="exp_sel_players",
    )

    date_col1, date_col2 = st.columns([1, 1])
    with date_col1:
        exp_from = st.date_input("Van datum", value=date.today().replace(day=1), key="exp_sel_from")
    with date_col2:
        exp_to = st.date_input("Tot datum", value=date.today(), key="exp_sel_to")

    if st.button("Generate export (selected players)", key="exp_sel_btn"):
        try:
            if not export_players:
                toast_err("Selecteer minimaal 1 speler.")
                st.stop()

            bio = io.BytesIO()
            used = set()
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                for p in export_players:
                    pname = requests.utils.quote(str(p), safe="")
                    q = (
                        f"select={','.join(GPS_COLS)}"
                        f"&player_name=eq.{pname}"
                        f"&datum=gte.{exp_from.isoformat()}"
                        f"&datum=lte.{exp_to.isoformat()}"
                        f"&order=datum.asc"
                        f"&limit=200000"
                    )
                    dfp = rest_get(access_token, "gps_records", q)
                    ordered = [c for c in GPS_COLS if c in dfp.columns]
                    dfp = dfp[ordered] if not dfp.empty else pd.DataFrame(columns=GPS_COLS)
                    sheet = safe_sheet_name(p, used)
                    dfp.to_excel(writer, index=False, sheet_name=sheet)

            st.session_state["export_sel_bytes"] = bio.getvalue()
            toast_ok("Export bevestigd: bestand klaar voor download.")
        except Exception as e:
            toast_err(str(e))

    if st.session_state.get("export_sel_bytes"):
        st.download_button(
            "Download gps_records_SELECTED_PLAYERS.xlsx",
            data=st.session_state["export_sel_bytes"],
            file_name="gps_records_SELECTED_PLAYERS.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="exp_sel_dl",
        )
