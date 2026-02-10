# roles.py
# ============================================================
# Centrale role/auth helpers voor alle pagina's
# FIX: uid ophalen via sb.auth.get_user(token) i.p.v. sb.auth.get_user()
# ============================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    from supabase import create_client  # type: ignore
except Exception:
    create_client = None


# Zet op True om UI-logica open te zetten (iedereen = staff in de UI)
OPEN_MODE = True

STAFF_ROLES = {
    "staff",
    "performance_coach",
    "data_scientist",
    "technical_director",
    "physio",
}


@st.cache_resource(show_spinner=False)
def get_sb():
    if create_client is None:
        return None
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])


def require_auth():
    if "access_token" not in st.session_state or not st.session_state.get("access_token"):
        st.error("Niet ingelogd.")
        st.stop()


def get_access_token() -> str:
    require_auth()
    return str(st.session_state["access_token"])


def sb_postgrest_auth(sb):
    """
    Zet Authorization header voor table() calls (PostgREST).
    """
    token = get_access_token()
    try:
        sb.postgrest.auth(token)
    except Exception:
        pass


def get_auth_uid(sb) -> Optional[str]:
    """
    FIX: GoTrue kent de token niet automatisch.
    Gebruik get_user(token) zodat uid altijd klopt.
    """
    token = get_access_token()
    try:
        u = sb.auth.get_user(token)
        return u.user.id
    except Exception:
        return None


def get_profile(sb) -> Optional[Dict[str, Any]]:
    """
    Leest public.profiles voor huidige user.
    """
    sb_postgrest_auth(sb)
    uid = get_auth_uid(sb)
    if not uid:
        return None
    try:
        resp = (
            sb.table("profiles")
            .select("user_id,role,team,player_id")
            .eq("user_id", uid)
            .maybe_single()
            .execute()
        )
        return resp.data
    except Exception:
        return None


def is_staff_user(profile: Optional[Dict[str, Any]]) -> bool:
    if OPEN_MODE:
        return True
    if not profile:
        return False
    role = str(profile.get("role", "player"))
    return role in STAFF_ROLES


def list_players(sb) -> List[Dict[str, Any]]:
    sb_postgrest_auth(sb)
    try:
        resp = sb.table("players").select("player_id,full_name,is_active").order("full_name").execute()
        rows = resp.data or []
        out: List[Dict[str, Any]] = []
        for r in rows:
            if "is_active" in r and r["is_active"] is False:
                continue
            out.append(r)
        return out
    except Exception:
        return []


def get_player_name(sb, player_id: str) -> str:
    sb_postgrest_auth(sb)
    try:
        resp = sb.table("players").select("full_name").eq("player_id", player_id).single().execute()
        return resp.data["full_name"]
    except Exception:
        return "Onbekend"


def pick_target_player(
    sb,
    profile: Optional[Dict[str, Any]],
    label: str = "Speler",
    key: str = "target_player_select",
) -> Tuple[Optional[str], Optional[str], bool]:
    """
    staff/open-mode: dropdown met alle spelers
    player: alleen eigen player_id uit profile
    """
    staff = is_staff_user(profile)

    if staff:
        players = list_players(sb)
        if not players:
            return None, None, True
        name_to_id = {p["full_name"]: p["player_id"] for p in players}
        sel_name = st.selectbox(label, options=list(name_to_id.keys()), key=key)
        pid = name_to_id[sel_name]
        return pid, sel_name, True

    if not profile or not profile.get("player_id"):
        return None, None, False

    pid = profile["player_id"]
    return pid, get_player_name(sb, pid), False
