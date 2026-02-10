# roles.py
# ============================================================
# Centrale role/auth helpers voor alle pagina's
# - 1 plek om Supabase client + auth + profile/role logic te beheren
# - "OPEN_MODE" kan je aan/uit zetten (debug: iedereen ziet alles)
#
# Gebruik in pagina's:
#   from roles import get_sb, sb_auth, require_auth, get_profile, is_staff_user, pick_target_player
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    from supabase import create_client  # type: ignore
except Exception:
    create_client = None


# ================
# CONFIG
# ================
# Zet op True om tijdelijk alle roles/RLS checks te omzeilen in je UI-logica
# (Let op: RLS in Supabase blijft dan nog steeds leidend!)
OPEN_MODE = False

# Als je nog meerdere staff-rollen wil supporten: voeg hier toe.
# Als je later weer strict wilt, gebruik je deze set.
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


def sb_auth(sb):
    """
    Koppel Supabase PostgREST aan de access token uit session_state.
    Vereist: st.session_state["access_token"]
    """
    token = st.session_state.get("access_token")
    if sb is not None and token:
        try:
            sb.postgrest.auth(token)
        except Exception:
            pass


def require_auth():
    if "access_token" not in st.session_state:
        st.error("Niet ingelogd.")
        st.stop()


def get_auth_uid(sb) -> Optional[str]:
    sb_auth(sb)
    try:
        return sb.auth.get_user().user.id
    except Exception:
        return None


def get_profile(sb) -> Optional[Dict[str, Any]]:
    """
    Leest public.profiles voor huidige user.
    Verwacht kolommen: user_id, role, team, player_id
    """
    sb_auth(sb)
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


def ensure_profile_exists_minimal(sb, role: str = "player", team: str = "MVV") -> bool:
    """
    Maakt een minimale profile-rij aan als die ontbreekt.
    Handig tijdens setup/debug.
    """
    sb_auth(sb)
    uid = get_auth_uid(sb)
    if not uid:
        return False
    try:
        # best-effort insert
        sb.table("profiles").upsert(
            {"user_id": uid, "role": role, "team": team, "player_id": None},
            on_conflict="user_id",
        ).execute()
        return True
    except Exception:
        return False


def is_staff_user(profile: Optional[Dict[str, Any]]) -> bool:
    """
    UI-logica: bepaalt of user staff is.
    OPEN_MODE = True => altijd staff.
    """
    if OPEN_MODE:
        return True
    if not profile:
        return False
    role = str(profile.get("role", "player"))
    return role in STAFF_ROLES


def list_players(sb) -> List[Dict[str, Any]]:
    sb_auth(sb)
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
    sb_auth(sb)
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
    Bepaalt welke speler je bekijkt:
    - staff: dropdown met alle spelers
    - player: alleen profile.player_id
    Returns: (player_id, player_name, is_staff)
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

    # player
    if not profile or not profile.get("player_id"):
        return None, None, False
    pid = profile["player_id"]
    return pid, get_player_name(sb, pid), False
