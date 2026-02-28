from __future__ import annotations

import streamlit as st

from roles import require_auth, get_sb, get_profile
from pages.Subscripts.wr_common import fetch_active_players_cached
from pages.Subscripts.wr_tab_day import render_wellness_rpe_tab_day
from pages.Subscripts.wr_tab_week import render_wellness_rpe_tab_week


def render_staff_wellness_rpe_page():
    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    profile = get_profile(sb) or {}
    role = str(profile.get("role") or "").lower()

    # Staff-only
    if role == "player":
        st.error("Geen toegang: deze pagina is alleen voor staff.")
        st.stop()

    st.title("Wellness / RPE (Team)")

    # cache key per project/env (zet hier iets dat bij jou altijd bestaat, bv st.secrets["SUPABASE_URL"])
    sb_url_key = str(st.secrets.get("SUPABASE_URL", "sb"))

    players = fetch_active_players_cached(sb_url_key, sb)
    if players.empty:
        st.warning("Geen actieve spelers gevonden.")
        st.stop()

    pid_to_name = dict(zip(players["player_id"], players["full_name"]))

    tab_day, tab_week = st.tabs(["Dag", "Week"])
    with tab_day:
        render_wellness_rpe_tab_day(sb, sb_url_key, pid_to_name)
    with tab_week:
        render_wellness_rpe_tab_week(sb, sb_url_key, pid_to_name)


if __name__ == "__main__":
    render_staff_wellness_rpe_page()
