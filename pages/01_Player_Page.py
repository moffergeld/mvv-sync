# pages/09_Player_Page.py
from __future__ import annotations

import streamlit as st

from roles import get_sb, require_auth, get_profile

from pages.Subscripts.player_tab_data import render_data_tab
from pages.Subscripts.player_tab_forms import render_forms_tab
from pages.Subscripts.player_tab_checklist import render_checklist_tab


def _fetch_player_name(sb, player_id: str) -> str:
    try:
        p = (
            sb.table("players")
            .select("full_name")
            .eq("player_id", player_id)
            .maybe_single()
            .execute()
            .data
        )
        return (p or {}).get("full_name") or "Player"
    except Exception:
        return "Player"


def fetch_active_players(sb):
    try:
        rows = (
            sb.table("players")
            .select("player_id,full_name,is_active")
            .eq("is_active", True)
            .order("full_name")
            .execute()
            .data
            or []
        )
    except Exception:
        rows = []
    return rows


def _pick_active_player_dropdown(sb, key: str = "pp_player_select"):
    rows = fetch_active_players(sb)
    if not rows:
        return None, None

    names = [str(r.get("full_name") or "").strip() for r in rows]
    ids = [str(r.get("player_id") or "").strip() for r in rows]
    pairs = [(n, i) for n, i in zip(names, ids) if n and i]

    if not pairs:
        return None, None

    options = [p[0] for p in pairs]
    name = st.selectbox("Speler", options=options, key=key)
    pid = dict(pairs).get(name)
    return pid, name


def main():
    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    profile = get_profile(sb) or {}
    role = str(profile.get("role") or "").lower()
    my_player_id = profile.get("player_id")

    # Player: forceer eigen speler, geen dropdown
    if role == "player":
        if not my_player_id:
            st.error("Je profiel is niet gekoppeld aan een speler (player_id ontbreekt).")
            st.stop()
        target_player_id = str(my_player_id)
        target_player_name = _fetch_player_name(sb, target_player_id)

    # Staff/Admin: dropdown met alleen actieve spelers
    else:
        target_player_id, target_player_name = _pick_active_player_dropdown(sb, key="pp_player_select")
        if not target_player_id:
            st.error("Geen speler beschikbaar.")
            st.stop()

    st.title(f"Player: {target_player_name}")

    tab_names = ["Data", "Forms"] + (["Checklist"] if role != "player" else [])
    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_data_tab(sb, target_player_id)

    with tabs[1]:
        render_forms_tab(sb, target_player_id)

    if "Checklist" in tab_names:
        with tabs[tab_names.index("Checklist")]:
            render_checklist_tab(sb)


if __name__ == "__main__":
    main()
