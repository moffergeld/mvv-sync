# pages/06_GPS_Import.py
# ============================================================
# MAIN PAGE: GPS Import (router)
# - Tab: Import GPS -> subtabs: Import (Excel), Manual add, Export
# - Tab: Matches
# NOTE: profiles.team column removed -> code must NOT select it.
# ============================================================
from __future__ import annotations
import streamlit as st
from pages.Subscripts.gps_import_common import (
    ALLOWED_IMPORT,
    get_access_token,
    get_players_map,
    get_profile_role,
)
from pages.Subscripts.gps_import_tab_excel import tab_import_excel_main
from pages.Subscripts.gps_import_tab_export import tab_export_main
from pages.Subscripts.gps_import_tab_manual import tab_manual_add_main
from pages.Subscripts.gps_import_tab_matches import tab_matches_main

st.set_page_config(page_title="GPS Import", layout="wide")
st.title("GPS Import")

access_token = get_access_token()
if not access_token:
    st.error("Niet ingelogd (access_token ontbreekt).")
    st.stop()

try:
    user_id, email, role, team = get_profile_role(access_token)
except Exception as e:
    st.error(f"Kon profiel/role niet ophalen: {e}")
    st.stop()

role_ui = (role or st.session_state.get("role") or "").strip().lower()
if role_ui == "player":
    st.error("Geen toegang.")
    st.stop()

st.session_state["role"] = role_ui

with st.expander("Debug (auth/role)", expanded=False):
    st.write({"email": email, "user_id": user_id, "role": role_ui, "team": team})

if role_ui not in ALLOWED_IMPORT:
    st.error("Geen rechten voor GPS import/export.")
    st.stop()

name_to_id, player_options = get_players_map(access_token)

main_page = st.radio(
    "Navigatie",
    options=["Import GPS", "Matches"],
    horizontal=True,
    key="nav_main",
    label_visibility="collapsed",
)

# -------------------------
# PAGE: Import GPS
# -------------------------
if main_page == "Import GPS":
    sub_page = st.radio(
        "Sub",
        options=["Import (Excel)", "Manual add", "Export"],
        horizontal=True,
        key="nav_gps_sub",
        label_visibility="collapsed",
    )
    st.divider()

    if sub_page == "Import (Excel)":
        tab_import_excel_main(access_token=access_token, name_to_id=name_to_id)

    elif sub_page == "Manual add":
        tab_manual_add_main(access_token=access_token, name_to_id=name_to_id, player_options=player_options)

    elif sub_page == "Export":
        tab_export_main(access_token=access_token, player_options=player_options)

# -------------------------
# PAGE: Matches
# -------------------------
elif main_page == "Matches":
    st.divider()
    tab_matches_main(access_token=access_token)
