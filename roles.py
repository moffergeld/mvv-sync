# roles.py
# ============================================================
# Single Source of Truth for Auth + Roles (Optie A)
#
# Functies die je overal gebruikt:
# - get_sb()
# - cookie_mgr()
# - try_restore_or_refresh_session()
# - require_auth()
# - get_access_token()
# - get_profile()
# - pick_target_player()
#
# Belangrijk:
# - 30 dagen refresh cookie
# - Supabase client is cached (st.cache_resource)
# - Profile wordt gecached in session_state (_profile_cache)
# - time.sleep geminimaliseerd (0.10) voor snellere UX
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


# Laat dit in productie op False
OPEN_MODE = False

STAFF_ROLES = {
    "staff",
    "performance_coach",
    "data_scientist",
    "technical_director",
    "physio",
    "admin",
}

ACCESS_COOKIE_SECONDS = 60 * 60
REFRESH_COOKIE_DAYS = 30
REFRESH_COOKIE_SECONDS = 60 * 60 * 24 * REFRESH_COOKIE_DAYS

# Kleine settle voor cookie refresh (zet naar 0 als je wil testen)
COOKIE_SETTLE_SECONDS = 0.10


def normalize_role(v: Any) -> str:
    if v is None:
        return "player"
    s = str(v).strip().lower()
    if "." in s:
        s = s.split(".")[-1]
    if "::" in s:
        s = s.split("::")[0]
    s = s.strip()
    return s or "player"


# ============================================================
# SUPABASE CLIENT
# ============================================================

@st.cache_resource(show_spinner=False)
def get_sb():
    if create_client is None:
        return None
    url = st.secrets.get("SUPABASE_URL", "").strip()
    key = st.secrets.get("SUPABASE_ANON_KEY", "").strip()
    if not url or not key:
        return None
    return create_client(url, key)


# ============================================================
# COOKIES
# ============================================================

def cookie_mgr():
    if "_cookie_mgr_instance" not in st.session_state:
        st.session_state["_cookie_mgr_instance"] = stx.CookieManager(key="mvv_cookie_mgr")
    return st.session_state["_cookie_mgr_instance"]


def _set_postgrest_auth_safely(sb, token: Optional[str]) -> None:
    if not sb or not token:
        return
    try:
        sb.postgrest.auth(token)
    except Exception:
        pass


def set_tokens_in_cookie(access_token: str, refresh_token: str, email: str | None = None) -> None:
    cm = cookie_mgr()
    cm.set("sb_access", str(access_token or ""), max_age=ACCESS_COOKIE_SECONDS, key="set_sb_access")
    cm.set("sb_refresh", str(refresh_token or ""), max_age=REFRESH_COOKIE_SECONDS, key="set_sb_refresh")
    if email:
        cm.set("sb_email", str(email), max_age=REFRESH_COOKIE_SECONDS, key="set_sb_email")


def clear_tokens_in_cookie() -> None:
    cm = cookie_mgr()
    cm.set("sb_access", "", max_age=1, key="clear_sb_access")
    cm.set("sb_refresh", "", max_age=1, key="clear_sb_refresh")
    cm.set("sb_email", "", max_age=1, key="clear_sb_email")


# ============================================================
# SESSION STATE HELPERS
# ============================================================

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


# ============================================================
# RESTORE / REFRESH SESSION
# ============================================================

def try_restore_or_refresh_session(sb=None) -> bool:
    if sb is None:
        sb = get_sb()
    if sb is None:
        st.session_state["auth_err"] = "Supabase client niet beschikbaar"
        return False

    cm = cookie_mgr()
    access = cm.get("sb_access")
    refresh = cm.get("sb_refresh")
    email = cm.get("sb_email")

    if email and not st.session_state.get("user_email"):
        st.session_state["user_email"] = email

    if not refresh:
        return False

    last_err = None
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
                set_tokens_in_cookie(sess.access_token, sess.refresh_token, st.session_state.get("user_email"))

                if COOKIE_SETTLE_SECONDS > 0:
                    time.sleep(COOKIE_SETTLE_SECONDS)

                return True

        except Exception as e:
            last_err = e
            if COOKIE_SETTLE_SECONDS > 0:
                time.sleep(COOKIE_SETTLE_SECONDS)

    st.session_state["auth_err"] = str(last_err) if last_err else "Unknown auth restore error"
    return False


# ============================================================
# AUTH GATE
# ============================================================

def require_auth() -> None:
    token = _get_access_token_from_state()
    if token:
        st.session_state["access_token"] = token
        return

    sb = get_sb()
    ok = try_restore_or_refresh_session(sb)
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
    token = get_access_token()
    _set_postgrest_auth_safely(sb, token)


def get_auth_uid(sb) -> Optional[str]:
    token = get_access_token()
    try:
        u = sb.auth.get_user(token)
        return u.user.id
    except Exception:
        st.session_state.pop("access_token", None)
        if try_restore_or_refresh_session(sb):
            token = _get_access_token_from_state()
            if token:
                try:
                    u = sb.auth.get_user(token)
                    return u.user.id
                except Exception:
                    return None
        return None


# ============================================================
# PROFILE / ROLES
# ============================================================

def get_profile(sb) -> Optional[Dict[str, Any]]:
    cached = st.session_state.get("_profile_cache")
    if isinstance(cached, dict) and cached.get("user_id") and cached.get("role"):
        return cached

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

        prof["role"] = normalize_role(prof.get("role"))
        st.session_state["role"] = prof["role"]
        if prof.get("player_id"):
            st.session_state["player_id"] = str(prof["player_id"])

        st.session_state["_profile_cache"] = prof
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


# ============================================================
# PLAYER HELPERS (staff dropdown)
# ============================================================

@st.cache_data(show_spinner=False, ttl=300)
def _list_players_cached(_cache_buster: str = "v1") -> List[Dict[str, Any]]:
    sb = get_sb()
    if sb is None:
        return []
    sb_postgrest_auth(sb)
    try:
        resp = (
            sb.table("players")
            .select("player_id,full_name,is_active")
            .eq("is_active", True)
            .order("full_name")
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


def list_players(sb) -> List[Dict[str, Any]]:
    return _list_players_cached("v1")


@st.cache_data(show_spinner=False, ttl=300)
def _get_player_name_cached(player_id: str, _cache_buster: str = "v1") -> str:
    sb = get_sb()
    if sb is None:
        return "Onbekend"
    sb_postgrest_auth(sb)
    try:
        resp = sb.table("players").select("full_name").eq("player_id", player_id).maybe_single().execute()
        return (resp.data or {}).get("full_name") or "Onbekend"
    except Exception:
        return "Onbekend"


def get_player_name(sb, player_id: str) -> str:
    return _get_player_name_cached(str(player_id), "v1")


def pick_target_player(
    sb,
    profile: Optional[Dict[str, Any]],
    label: str = "Speler",
    key: str = "target_player_select",
) -> Tuple[Optional[str], Optional[str], bool]:
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

    if not profile or not profile.get("player_id"):
        return None, None, False

    pid = str(profile["player_id"])
    return pid, get_player_name(sb, pid), False
