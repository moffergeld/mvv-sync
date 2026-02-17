# gps_import_tab_matches.py
# ============================================================
# Main Tab: Matches
# ============================================================

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from Subscripts.gps_import_common import (
    HOME_AWAY_OPTIONS,
    MATCH_TYPE_OPTIONS,
    TEAM_NAME_MATCHES,
    build_fixture,
    build_result,
    default_season_today,
    fetch_matches_range,
    matches_df_to_rows,
    parse_matches_csv,
    rest_delete,
    rest_patch,
    rest_upsert,
    season_options,
    toast_err,
    toast_ok,
)


def tab_matches_main(access_token: str) -> None:
    st.subheader("Matches")

    st.markdown("### 1) Import Matches.csv → Supabase")
    matches_file = st.file_uploader("Upload Matches.csv", type=["csv"], key="matches_csv")
    if matches_file:
        b = matches_file.getvalue()
        fname = matches_file.name

        if st.button("Preview matches", key="m_prev_btn"):
            try:
                dfm = parse_matches_csv(b)
                st.session_state["matches_preview"] = dfm
                toast_ok(f"Preview bevestigd: {len(dfm)} rijen.")
                st.dataframe(dfm, use_container_width=True)
            except Exception as e:
                toast_err(str(e))

        dfm = st.session_state.get("matches_preview")
        if dfm is not None and not dfm.empty:
            if st.button("Import matches (upsert)", type="primary", key="m_import_btn"):
                try:
                    rows = matches_df_to_rows(dfm, source_file=fname)
                    rest_upsert(access_token, "matches", rows, on_conflict="match_date,fixture,season")
                    st.session_state["matches_preview"] = None
                    toast_ok(f"Import bevestigd: matches = {len(rows)}")
                except Exception as e:
                    toast_err(f"Import fout: {e}")

    st.divider()
    st.markdown("### 2) Selecteer & pas score aan")

    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        d_from = st.date_input("Van", value=date.today().replace(month=1, day=1), key="m_edit_from")
    with c2:
        d_to = st.date_input("Tot", value=date.today(), key="m_edit_to")
    with c3:
        season_filter = st.selectbox(
            "Seizoen filter (optioneel)",
            options=["(alles)"] + season_options(start_year=2020, years_ahead=6),
            index=0,
            key="m_edit_season",
        )

    try:
        df_list = fetch_matches_range(access_token, d_from, d_to, "" if season_filter == "(alles)" else season_filter)
    except Exception as e:
        toast_err(f"Kon matches niet ophalen: {e}")
        df_list = pd.DataFrame()

    if df_list.empty:
        st.info("Geen matches gevonden in deze periode.")
    else:
        df_list = df_list.copy()
        df_list["label"] = df_list.apply(
            lambda r: f"{r.get('match_date')} | {(r.get('fixture') or '').strip()} | {build_result(r.get('goals_for'), r.get('goals_against'))}",
            axis=1,
        )

        pick = st.selectbox("Kies wedstrijd", options=df_list["label"].tolist(), key="m_pick_match")
        match_id = int(df_list.loc[df_list["label"] == pick, "match_id"].iloc[0])
        row = df_list[df_list["match_id"] == match_id].iloc[0].to_dict()

        st.markdown("**Aanpassen:** (alleen score + match type + season)")
        e1, e2, e3 = st.columns([1, 1, 1.2])
        with e1:
            goals_for = st.number_input("Goals for", min_value=0, step=1, value=int(row.get("goals_for") or 0), key="m_gf")
        with e2:
            goals_against = st.number_input("Goals against", min_value=0, step=1, value=int(row.get("goals_against") or 0), key="m_ga")
        with e3:
            match_type = st.selectbox(
                "Match type",
                options=MATCH_TYPE_OPTIONS,
                index=(MATCH_TYPE_OPTIONS.index(row.get("match_type")) if row.get("match_type") in MATCH_TYPE_OPTIONS else 0),
                key="m_mt",
            )

        seasons = season_options(start_year=2020, years_ahead=6)
        current_season = str(row.get("season") or "").strip()
        if current_season and current_season not in seasons:
            seasons = [current_season] + seasons

        season = st.selectbox(
            "Season",
            options=seasons,
            index=(seasons.index(current_season) if current_season in seasons else 0),
            key="m_season",
        )

        csave, cdel = st.columns([1, 1])
        with csave:
            if st.button("Save changes", type="primary", key="m_save_btn"):
                try:
                    payload = {
                        "goals_for": int(goals_for),
                        "goals_against": int(goals_against),
                        "result": build_result(goals_for, goals_against),
                        "match_type": match_type,
                        "season": season if season else None,
                    }
                    rest_patch(access_token, "matches", f"match_id=eq.{match_id}", payload)
                    toast_ok("Save bevestigd.")
                except Exception as e:
                    toast_err(f"Opslaan mislukt: {e}")

        with cdel:
            if st.button("Delete match", type="secondary", key="m_del_btn"):
                st.session_state["confirm_delete_match_id"] = match_id

        if st.session_state.get("confirm_delete_match_id") == match_id:
            st.warning("Bevestig verwijderen van deze wedstrijd. Dit kan niet ongedaan gemaakt worden.")
            y, n = st.columns([1, 1])
            with y:
                if st.button("Ja, verwijderen", type="primary", key="m_del_yes"):
                    try:
                        rest_delete(access_token, "matches", f"match_id=eq.{match_id}")
                        st.session_state["confirm_delete_match_id"] = None
                        toast_ok("Verwijderen bevestigd.")
                    except Exception as e:
                        toast_err(f"Verwijderen mislukt: {e}")
            with n:
                if st.button("Nee, annuleren", key="m_del_no"):
                    st.session_state["confirm_delete_match_id"] = None
                    toast_ok("Verwijderen geannuleerd.")

    st.divider()
    st.markdown("### 3) Nieuwe wedstrijd handmatig toevoegen")

    a1, a2, a3 = st.columns([1, 1.4, 1])
    with a1:
        new_date = st.date_input("Match date", value=date.today(), key="m_new_date")
    with a2:
        new_opponent = st.text_input("Opponent", value="", key="m_new_opp")
    with a3:
        new_home_away = st.selectbox("Home/Away", options=HOME_AWAY_OPTIONS, key="m_new_ha")

    b1, b2 = st.columns([1, 1])
    with b1:
        new_match_type = st.selectbox("Match type", options=MATCH_TYPE_OPTIONS, key="m_new_type")
    with b2:
        seasons_new = season_options(start_year=2020, years_ahead=6)
        ds = default_season_today()
        idx = seasons_new.index(ds) if ds in seasons_new else 0
        new_season = st.selectbox("Season", options=seasons_new, index=idx, key="m_new_season")

    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        new_gf = st.number_input("Goals for", min_value=0, step=1, value=0, key="m_new_gf")
    with c2:
        new_ga = st.number_input("Goals against", min_value=0, step=1, value=0, key="m_new_ga")
    with c3:
        new_result = st.text_input("Result (optioneel override)", value="", key="m_new_result")

    auto_fixture = build_fixture(TEAM_NAME_MATCHES, new_home_away, new_opponent)
    st.text_input("Fixture (auto)", value=auto_fixture, disabled=True, key="m_new_fix")

    if st.button("Add match", type="primary", key="m_add_btn"):
        try:
            if not new_opponent.strip():
                toast_err("Opponent is verplicht.")
                st.stop()

            payload = [
                {
                    "match_date": new_date.isoformat(),
                    "home_away": new_home_away,
                    "opponent": new_opponent.strip(),
                    "fixture": auto_fixture,
                    "match_type": new_match_type,
                    "season": new_season.strip(),
                    "goals_for": int(new_gf),
                    "goals_against": int(new_ga),
                    "result": (new_result.strip() if new_result.strip() else build_result(new_gf, new_ga)),
                    "source_file": "manual",
                }
            ]

            rest_upsert(access_token, "matches", payload, on_conflict="match_date,fixture,season")
            toast_ok("Toevoegen bevestigd.")
        except Exception as e:
            toast_err(f"Toevoegen mislukt: {e}")
