# pages/01_Player_Page.py
# ============================================================
# Player Page (Streamlit)
#
# Doel
# - Centrale pagina voor speler-specifieke modules (Data + Forms)
# - Snelle laadtijd door:
#   - caching van spelerslijst (staff) en player name (player)
#   - tabs zonder Checklist (Checklist staat nu op andere pagina)
#
# Data-bronnen (Supabase)
# - public.players          (player_id, full_name, is_active)
# - public.profiles         (role, player_id) via roles.get_profile()
#
# Rollen
# - role == "player": geen dropdown, altijd eigen player_id (UI)
# - staff: dropdown met actieve spelers
#
# Belangrijk (UI vs security)
# - Dit script filtert UI op basis van role/player_id, maar dit is geen security.
#   Als je ooit security nodig hebt: Supabase RLS policies gebruiken.
# ============================================================

from __future__ import annotations

import streamlit as st

from roles import get_sb, require_auth, get_profile
from pages.Subscripts.player_tab_data import render_data_tab
from pages.Subscripts.player_tab_forms import render_forms_tab


# ============================================================
# CACHING HELPERS
# ============================================================

@st.cache_data(show_spinner=False, ttl=300)
def fetch_active_players_cached(sb):
    """
    Staff dropdown: actieve spelers.

    TTL=300s:
    - spelerlijst verandert zelden
    - maakt player-switch sneller en voorkomt extra queries bij reruns
    """
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
        return rows
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=300)
def fetch_player_name_cached(sb, player_id: str) -> str:
    """
    Player role: toon titel met naam zonder elke rerun een query.

    TTL=300s:
    - naam verandert vrijwel nooit
    - voorkomt kleine maar onnodige calls
    """
    try:
        row = (
            sb.table("players")
            .select("full_name")
            .eq("player_id", player_id)
            .maybe_single()
            .execute()
            .data
        )
        return (row or {}).get("full_name") or "Player"
    except Exception:
        return "Player"


# ============================================================
# UI HELPERS
# ============================================================

def pick_active_player_dropdown(sb, key: str = "pp_player_select"):
    """
    Staff only: dropdown met actieve spelers.

    Returns:
      (player_id, player_name)
    """
    rows = fetch_active_players_cached(sb)
    if not rows:
        return None, None

    pairs = []
    for r in rows:
        pid = r.get("player_id")
        name = (r.get("full_name") or "").strip()
        if pid and name:
            pairs.append((name, str(pid)))

    if not pairs:
        return None, None

    options = [p[0] for p in pairs]
    sel_name = st.selectbox("Speler", options=options, key=key)
    name_to_id = dict(pairs)
    return name_to_id.get(sel_name), sel_name


# ============================================================
# MAIN
# ============================================================

def main():
    # --- Auth gate ---
    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    # --- Profile: bepaalt role + player_id ---
    profile = get_profile(sb) or {}
    role = str(profile.get("role") or "").lower()
    my_player_id = profile.get("player_id")

    # --- Select target player (UI) ---
    if role == "player":
        # Player: forceer eigen speler (UI)
        if not my_player_id:
            st.error("Je profiel is niet gekoppeld aan een speler (player_id ontbreekt).")
            st.stop()

        target_player_id = str(my_player_id)
        target_player_name = fetch_player_name_cached(sb, target_player_id)

        # Optionele UI tekst (klein + duidelijk)
        st.caption("Je ziet hier alleen jouw eigen data in de Player Page.")

    else:
        # Staff: kies speler
        st.subheader("Selecteer speler")
        target_player_id, target_player_name = pick_active_player_dropdown(sb, key="pp_player_select")
        if not target_player_id:
            st.error("Geen speler beschikbaar.")
            st.stop()

    # --- Title ---
    st.title(f"Player: {target_player_name}")

    # Extra info (handig voor users)
    with st.expander("ℹ️ Info (Data-tab)", expanded=False):
        st.write(
            "- **GPS & Wellness**: laatste **14 dagen**\n"
            "- **RPE Over time**: laatste **7 dagen**\n"
            "- Grafieken zijn **static** (minder gevoelig op telefoon)\n"
            "- **Max Speed** wordt per dag als **max** genomen (niet som)\n"
        )

    # --- Tabs (Checklist is weggehaald) ---
    tabs = st.tabs(["Data", "Forms"])

    with tabs[0]:
        render_data_tab(sb, target_player_id)

    with tabs[1]:
        render_forms_tab(sb, target_player_id)


if __name__ == "__main__":
    main()
