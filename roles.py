# roles.py
# ============================================================
# Centrale role/auth helpers voor alle pagina's
#
# Belangrijk:
# - profiles.team is verwijderd -> NIET selecteren.
# - uid ophalen via sb.auth.get_user(token) zodat uid klopt.
# - Role altijd normalizen (enum/qualified strings).
# - OPEN_MODE standaard UIT (False), anders wordt iedereen "staff" in de UI.
# - Herstelt auth automatisch vanuit cookies bij Streamlit session reset
#   (mobiel/tab switch/reconnect), mits app.py cookies zet.
# ============================================================

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import extra_streamlit_components as stx

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


# ----------------------------
# Cookie/session restore helpers
# ----------------------------
def _cookie_mgr():
    return stx.CookieManager()


def _set_postgrest_auth_safely(sb, token: Optional[str]) -> None:
    if not sb or not token:
        return
    try:
        sb.postgrest.auth(token)
    except Exception:
        # Sommige supabase-py versies gooien hier soms een error; dan werkt het alsnog vaak.
        pass


def _set_tokens_in_cookie(access_token: str, refresh_token: str, email: str | None = None) -> None:
    cm = _cookie_mgr()
    cm.set("sb_access", access_token or "", max_age=60 * 60)               # 1 uur
    cm.set("sb_refresh", refresh_token or "", max_age=60 * 60 * 24 * 30)   # 30 dagen
    if email:
        cm.set("sb_email", email, max_age=60 * 60 * 24 * 30)


def _clear_tokens_in_cookie() -> None:
    cm = _cookie_mgr()
    # delete bestaat niet in alle versies
    cm.set("sb_access", "", max_age=1)
    cm.set("sb_refresh", "", max_age=1)
    cm.set("sb_email", "", max_age=1)


def _get_access_token_from_state() -> Optional[str]:
    tok = st.session_state.get("access_token")
    if tok:
        return str(tok)

    sess = st.session_state.get("sb_session")
    if sess is not None:
        token = getattr(sess, "access_token", None)
        if token:
            return str(token)
    return None


def _try_restore_or_refresh_session(sb=None) -> bool:
    """
    Herstelt sessie uit cookies wanneer session_state leeg is geraakt.
    Vereist dat app.py bij login tokens in cookies opslaat.
    """
    if sb is None:
        sb = get_sb()
    if sb is None:
        st.session_state["auth_err"] = "Supabase client niet beschikbaar"
        return False

    cm = _cookie_mgr()
    access = cm.get("sb_access")
    refresh = cm.get("sb_refresh")
    email = cm.get("sb_email")

    if email and not st.session_state.get("user_email"):
        st.session_state["user_email"] = email

    if not refresh:
        return False

    last_err = None

    # Soft retries bij korte netwerkdip
    for _ in range(2):
        try:
            if access:
                try:
                    sb.auth.set_session(access, refresh)
                except Exception:
                    pass

            refreshed = sb.auth.refresh_session(refresh)
            sess = getattr(refreshed, "session", None)

            if sess and getattr(sess, "access_token", None) and getattr(sess, "refresh_token", None):
                st.session_state["access_token"] = sess.access_token
                st.session_state["sb_session"] = sess
                _set_postgrest_auth_safely(sb, sess.access_token)

                _set_tokens_in_cookie(
                    sess.access_token,
                    sess.refresh_token,
                    st.session_state.get("user_email"),
                )
                return True

        except Exception as e:
            last_err = e
            time.sleep(0.35)

    st.session_state["auth_err"] = str(last_err) if last_err else "Unknown auth restore error"
    return False


def require_auth() -> None:
    """
    Zorgt dat er een geldige access_token aanwezig is.
    Probeert eerst cookie-restore als session_state leeg is.
    """
    token = _get_access_token_from_state()
    if token:
        st.session_state["access_token"] = token
        return

    sb = get_sb()
    ok = _try_restore_or_refresh_session(sb)
    if ok and _get_access_token_from_state():
        return

    st.error("Niet ingelogd.")
    st.stop()


def get_access_token() -> str:
    require_auth()
    tok = _get_access_token_from_state()
    if not tok:
        st.error("Geen access token beschikbaar.")
        st.stop()
    return str(tok)


def sb_postgrest_auth(sb) -> None:
    """
    Zet Authorization header voor PostgREST table() calls.
    """
    token = get_access_token()
    _set_postgrest_auth_safely(sb, token)


def get_auth_uid(sb) -> Optional[str]:
    """
    Haalt user_id op via GoTrue met token.
    Probeert 1x silent refresh bij verlopen token.
    """
    token = get_access_token()
    try:
        u = sb.auth.get_user(token)
        return u.user.id
    except Exception:
        # mogelijk verlopen token -> probeer restore/refresh
        st.session_state.pop("access_token", None)
        if _try_restore_or_refresh_session(sb):
            token = _get_access_token_from_state()
            if token:
                try:
                    u = sb.auth.get_user(token)
                    return u.user.id
                except Exception:
                    return None
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
