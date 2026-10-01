"""Streamlit tab for direct STATSports API -> Supabase synchronisation."""

from __future__ import annotations

import os
from datetime import date, timedelta

import streamlit as st

from pages.Subscripts.gps_import_common import (
    apply_auto_match_ids_to_rows,
    df_to_db_rows,
    rest_upsert,
    toast_err,
    toast_ok,
)
from pages.Subscripts.statsports_api import (
    STATSPORTS_FIRST_DATE,
    fetch_statsports_range,
    sessions_to_dataframe,
    validate_api_key,
)


def _configured_api_key() -> str:
    return str(
        st.secrets.get("STATSPORTS_API_KEY", "")
        or st.secrets.get("STATSPORTS_API_ID", "")
        or st.secrets.get("STATSport", "")
        or os.getenv("STATSPORTS_API_KEY", "")
        or os.getenv("STATSPORTS_API_ID", "")
        or os.getenv("STATSport", "")
    ).strip()


def _preview_signature(start: date, end: date) -> str:
    return f"{start.isoformat()}:{end.isoformat()}"


def tab_import_statsports_main(access_token: str, name_to_id: dict) -> None:
    st.subheader("STATSports API → Supabase")
    st.caption(
        "Haalt sessies rechtstreeks op bij STATSports en schrijft ze zonder lokale tussenopslag naar "
        "Supabase. De eerste volledige import start standaard op 24 augustus 2026."
    )

    configured_key = _configured_api_key()
    if configured_key:
        st.success("STATSports API-ID is via de beveiligde app-configuratie gekoppeld.")
        api_key = configured_key
    else:
        st.warning(
            "STATSPORTS_API_KEY ontbreekt in de Streamlit-secrets. Een tijdelijk ingevulde API-ID "
            "blijft alleen in deze browsersessie en wordt niet opgeslagen."
        )
        api_key = st.text_input(
            "STATSports API-ID",
            type="password",
            key="statsports_api_key_session",
        ).strip()

    mode = st.radio(
        "Synchronisatie",
        options=["Eerste volledige import", "Periode synchroniseren"],
        horizontal=True,
        key="statsports_sync_mode",
    )
    today = date.today()
    if mode == "Eerste volledige import":
        start = STATSPORTS_FIRST_DATE
        end = today
        st.info(f"Volledige importperiode: {start:%d-%m-%Y} t/m {end:%d-%m-%Y}.")
    else:
        left, right = st.columns(2)
        with left:
            start = st.date_input(
                "Vanaf",
                value=max(STATSPORTS_FIRST_DATE, today - timedelta(days=7)),
                min_value=STATSPORTS_FIRST_DATE,
                max_value=today,
                key="statsports_start_date",
            )
        with right:
            end = st.date_input(
                "Tot en met",
                value=today,
                min_value=STATSPORTS_FIRST_DATE,
                max_value=today,
                key="statsports_end_date",
            )

    signature = _preview_signature(start, end)
    if st.session_state.get("statsports_preview_signature") != signature:
        st.session_state.pop("statsports_api_preview", None)
        st.session_state["statsports_preview_signature"] = signature

    if st.button("Data ophalen en controleren", type="secondary", key="statsports_fetch_button"):
        progress = st.progress(0.0, text="Verbinding maken met STATSports…")

        def update_progress(index, total, chunk_start, chunk_end, sessions):
            progress.progress(
                index / total,
                text=(
                    f"{index}/{total}: {chunk_start:%d-%m-%Y} t/m {chunk_end:%d-%m-%Y} "
                    f"· {len(sessions)} sessies"
                ),
            )

        try:
            validate_api_key(api_key)
            sessions = fetch_statsports_range(api_key, start, end, on_chunk=update_progress)
            frame = sessions_to_dataframe(sessions)
            st.session_state["statsports_api_preview"] = {
                "signature": signature,
                "sessions": len(sessions),
                "frame": frame,
            }
            progress.empty()
            toast_ok(f"STATSports opgehaald: {len(sessions)} sessies en {len(frame)} meetregels.")
        except Exception as exc:
            progress.empty()
            st.session_state.pop("statsports_api_preview", None)
            toast_err(str(exc))

    preview = st.session_state.get("statsports_api_preview")
    if not preview or preview.get("signature") != signature:
        return

    frame = preview["frame"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("API-sessies", preview["sessions"])
    c2.metric("Meetregels", len(frame))
    c3.metric("Spelers", frame["Speler"].nunique(dropna=True) if not frame.empty else 0)
    c4.metric("Dagen", frame["Datum"].nunique(dropna=True) if not frame.empty else 0)

    if frame.empty:
        st.info("STATSports heeft voor deze periode geen bruikbare gelabelde sessies teruggegeven.")
        return

    preview_columns = [
        column
        for column in ("Datum", "Speler", "Type", "Event", "Session Title", "Session Type", "totalTime", "totalDistance", "maxSpeed")
        if column in frame.columns
    ]
    st.dataframe(frame[preview_columns].head(150), width="stretch", hide_index=True)
    st.caption(
        "Bij import worden alle STATSports-velden bewaard: bekende meetwaarden in vaste Supabase-kolommen "
        "en de volledige ruwe API-context in extra_metrics."
    )

    if st.button("Importeer rechtstreeks naar Supabase", type="primary", key="statsports_import_button"):
        try:
            rows, unmapped = df_to_db_rows(frame, source_file="STATSports API", name_to_id=name_to_id)
            destination_keys = [
                (row.get("player_name"), row.get("datum"), row.get("type"), row.get("event"))
                for row in rows
            ]
            if len(destination_keys) != len(set(destination_keys)):
                raise ValueError("Dubbele STATSports-meetregels gevonden; er is niets opgeslagen.")

            rows = apply_auto_match_ids_to_rows(access_token, rows, ui_key_prefix="statsports_api_apply")
            rest_upsert(
                access_token,
                "gps_records",
                rows,
                on_conflict="player_name,datum,type,event",
            )
            if unmapped:
                st.warning(
                    "Niet-gematchte spelers zijn wel opgeslagen met player_id = NULL: "
                    + ", ".join(unmapped[:30])
                )
            st.session_state.pop("statsports_api_preview", None)
            toast_ok(f"STATSports-synchronisatie voltooid: {len(rows)} regels opgeslagen in Supabase.")
        except Exception as exc:
            toast_err(f"STATSports-import mislukt: {exc}")
