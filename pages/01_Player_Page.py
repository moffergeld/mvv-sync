# pages/01_Player_Page.py
# ============================================================
# Player Page (Streamlit)
#
# FIXES:
# - Robuuste imports (werkt ook als packages niet goed staan)
# - Checklist tab verwijderd (staat op andere pagina)
# - Cache: Supabase client is unhashable -> underscore param _sb
# ============================================================

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ------------------------------------------------------------
# Import path fix (works even if __init__.py is missing/wrong)
# ------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent          # .../pages
ROOT_DIR = THIS_DIR.parent                          # project root

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))              # so "Subscripts.*" works
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))              # so "roles" works


from roles import get_sb, require_auth, get_profile  # noqa: E402
from Subscripts.player_tab_data import render_data_tab  # noqa: E402
from Subscripts.player_tab_forms import render_forms_tab  # noqa: E402


# ============================================================
# CACHING HELPERS (sb is unhashable -> use _sb)
# ============================================================

@st.cache_data(show_spinner=False, ttl=300)
def fetch_active_players_cached(_sb):
    """Staff dropdown: actieve spelers (cached)."""
    try:
        rows = (
            _sb.table("players")
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
def fetch_player_name_cached(_sb, player_id: str) -> str:
    """Player role: titel met naam (cached)."""
    try:
        row = (
            _sb.table("players")
            .select("full_name")
            .eq("player_id", player_id)
            .maybe_single()
            .execute()
            .data
        )
        return (row or {}).get("full_name") or "Player"
    except Exception:
        return "Player"


def pick_active_player_dropdown(sb, key: str = "pp_player_select"):
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


def main():
    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    profile = get_profile(sb) or {}
    role = str(profile.get("role") or "").lower()
    my_player_id = profile.get("player_id")

    if role == "player":
        if not my_player_id:
            st.error("Je profiel is niet gekoppeld aan een speler (player_id ontbreekt).")
            st.stop()

        target_player_id = str(my_player_id)
        target_player_name = fetch_player_name_cached(sb, target_player_id)
        st.caption("Je ziet hier alleen jouw eigen data in de Player Page.")
    else:
        st.subheader("Selecteer speler")
        target_player_id, target_player_name = pick_active_player_dropdown(sb, key="pp_player_select")
        if not target_player_id:
            st.error("Geen speler beschikbaar.")
            st.stop()

    st.title(f"Player: {target_player_name}")

    with st.expander("ℹ️ Info (Data-tab)", expanded=False):
        st.write(
            "- **GPS & Wellness**: laatste **14 dagen**\n"
            "- **RPE Over time**: laatste **7 dagen**\n"
            "- Grafieken zijn **static** (minder gevoelig op telefoon)\n"
            "- **Max Speed**: per dag **max** (niet som)\n"
        )

    tabs = st.tabs(["Data", "Forms"])

    with tabs[0]:
        render_data_tab(sb, target_player_id)

    with tabs[1]:
        render_forms_tab(sb, target_player_id)


if __name__ == "__main__":
    main()
