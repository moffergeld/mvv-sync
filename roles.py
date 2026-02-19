# roles.py
# ============================================================
# Centrale role/auth helpers voor alle pagina's
#
# Belangrijk:
# - profiles.team is verwijderd -> NIET selecteren.
# - uid ophalen via sb.auth.get_user(token) zodat uid klopt.
# - Role altijd normalizen (enum/qualified strings).
# - OPEN_MODE standaard UIT (False), anders wordt iedereen "staff" in de UI.
# ============================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    from supabase import create_client  # type: ignore
except Exception:
    create_client = None


# Zet op True om UI-gates te omzeilen (iedereen = staff in UI).
# Laat dit in productie op False.
OPEN_MODE = False

STAFF_ROLES = {
    "staff",
    "performance_coach",
    "data_scientist",
    "technical_director",
    "physio",
    "admin",
}


def normalize_role(v: Any) -> str:
    """
    Normaliseert role-waardes zoals:
    - "player"
    - "public.user_role_v2.player"
    - "player::something"
    """
    if v is None:
        return "player"
    s = str(v).strip().lower()
    if "." in s:
        s = s.split(".")[-1]
    if "::" in s:
        s = s.split("::")[0]
    s = s.strip()
    return s or "player"


@st.cache_resource(show_spinner=False)
def get_sb():
    if create_client is None:
        return None
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"])


def require_auth() -> None:
    if not st.session_state.get("access_token"):
        st.error("Niet ingelogd.")
        st.stop()


def get_access_token() -> str:
    require_auth()
    return str(st.session_state["access_token"])


def sb_postgrest_auth(sb) -> None:
    """
    Zet Authorization header voor PostgREST table() calls.
    """
    token = get_access_token()
    try:
        sb.postgrest.auth(token)
    except Exception:
        # Sommige supabase-py versies gooien hier soms een error; dan werkt het alsnog vaak.
        pass


def get_auth_uid(sb) -> Optional[str]:
    """
    Haalt user_id op via GoTrue met token.
    """
    token = get_access_token()
    try:
        u = sb.auth.get_user(token)
        return u.user.id
    except Exception:
        return None


def get_profile(sb) -> Optional[Dict[str, Any]]:
    """
    Leest public.profiles voor de huidige user.

    Verwacht kolommen:
    - user_id (uuid)
    - role (user_role_v2 / text)
    - player_id (uuid, nullable)
    """
    sb_postgrest_auth(sb)
    uid = get_auth_uid(sb)
    if not uid:
        return None

    try:
        resp = (
            sb.table("profiles")
            .select("user_id,role,player_id")
            .eq("user_id", uid)
            .maybe_single()
            .execute()
        )
        prof = resp.data or None
        if prof is None:
            return None

        # Normaliseer role direct en zet ook in session_state (handig voor UI gates)
        prof["role"] = normalize_role(prof.get("role"))
        st.session_state["role"] = prof["role"]
        if prof.get("player_id"):
            st.session_state["player_id"] = str(prof["player_id"])
        return prof
    except Exception:
        return None


def is_staff_user(profile: Optional[Dict[str, Any]]) -> bool:
    if OPEN_MODE:
        return True
    if not profile:
        return False
    role = normalize_role(profile.get("role"))
    return role in STAFF_ROLES


def list_players(sb) -> List[Dict[str, Any]]:
    """
    Returns actieve spelers (player_id, full_name).
    """
    sb_postgrest_auth(sb)
    try:
        resp = (
            sb.table("players")
            .select("player_id,full_name,is_active")
            .eq("is_active", True)
            .order("full_name")
            .execute()
        )
        rows = resp.data or []
        out: List[Dict[str, Any]] = []
        for r in rows:
            # extra guard
            if r.get("is_active") is False:
                continue
            out.append(r)
        return out
    except Exception:
        return []


def get_player_name(sb, player_id: str) -> str:
    sb_postgrest_auth(sb)
    try:
        resp = sb.table("players").select("full_name").eq("player_id", player_id).maybe_single().execute()
        return (resp.data or {}).get("full_name") or "Onbekend"
    except Exception:
        return "Onbekend"


def pick_target_player(
    sb,
    profile: Optional[Dict[str, Any]],
    label: str = "Speler",
    key: str = "target_player_select",
) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Bepaalt target speler voor Player Page:
    - staff/open-mode: dropdown met alle actieve spelers
    - player: altijd eigen player_id uit profile (geen dropdown)

    Returns: (player_id, player_name, is_staff_mode)
    """
    staff_mode = is_staff_user(profile)

    if staff_mode:
        players = list_players(sb)
        if not players:
            return None, None, True

        name_to_id = {str(p.get("full_name", "")).strip(): p.get("player_id") for p in players if p.get("player_id")}
        options = [n for n in name_to_id.keys() if n]
        if not options:
            return None, None, True

        sel_name = st.selectbox(label, options=options, key=key)
        pid = name_to_id.get(sel_name)
        return (str(pid) if pid else None), sel_name, True

    # Player: force eigen speler
    if not profile or not profile.get("player_id"):
        return None, None, False

    pid = str(profile["player_id"])
    return pid, get_player_name(sb, pid), False
