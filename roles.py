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
# - sb_postgrest_auth()
# - get_profile()
# - pick_target_player()
#
# Belangrijk (fix voor mobile/RLS issues):
# - Supabase client NIET meer global cachen (geen st.cache_resource)
# - Supabase client per Streamlit session (st.session_state["_sb_client"])
# - PostgREST auth header wordt bij elke get_sb() call gezet op basis van access_token
#
# Overig:
# - 30 dagen refresh cookie
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

SIDEBAR_PAGE_LINKS = [
    ("app.py", "Dashboard"),
    ("pages/02_Match_Reports.py", "Match Reports"),
    ("pages/10_Data_Page_Beta.py", "Data"),
    ("pages/09_Management.py", "Management"),
]

SIDEBAR_BETA_PAGE_LINKS = [
    ("pages/07_Player_Page_Beta.py", "Player Page Beta"),
    ("pages/08_Team_Page_Beta.py", "Team Page Beta"),
]
LOGIN_PAGE_PATH = "app.py"


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


def _format_role_label(v: Any) -> str:
    role = normalize_role(v)
    return role.replace("_", " ").title()


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
    # (optioneel) storage auth; kan geen kwaad
    try:
        sb.storage.auth(token)
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


def clear_auth_state(clear_cookies: bool = False) -> None:
    for key in (
        "access_token",
        "sb_session",
        "_profile_cache",
        "role",
        "player_id",
        "user_id",
        "_sb_client",
        "auth_err",
    ):
        st.session_state.pop(key, None)
    if clear_cookies:
        clear_tokens_in_cookie()


def redirect_to_login(message: str = "Sessie verlopen. Log opnieuw in.", clear_cookies: bool = False) -> None:
    clear_auth_state(clear_cookies=clear_cookies)
    if message:
        st.session_state["_login_notice"] = message
    try:
        st.switch_page(LOGIN_PAGE_PATH)
    except Exception:
        if message:
            st.error(message)
    st.stop()


def consume_login_notice() -> Optional[str]:
    notice = st.session_state.pop("_login_notice", None)
    if notice is None:
        return None
    return str(notice)


def _sidebar_logout_action() -> None:
    try:
        sb = get_sb()
        if sb is not None:
            sb.auth.sign_out()
    except Exception:
        pass
    redirect_to_login("Je bent uitgelogd.", clear_cookies=True)


def _render_sidebar_css() -> None:
    st.sidebar.markdown(
        """
        <style>
        [data-testid="stSidebarUserContent"] > div {
          min-height: 100%;
        }

        [data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
        }

        [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > div:has(.mvv-sidebar-footer-anchor) {
          margin-top: auto;
          padding-top: 0;
          padding-bottom: 0.35rem;
          background: linear-gradient(180deg, rgba(9,13,23,0) 0%, rgba(9,13,23,0.90) 16%, rgba(9,13,23,0.98) 100%);
          position: sticky;
          bottom: 0;
          z-index: 5;
        }

        [data-testid="stSidebar"] .mvv-sidebar-nav-label {
          color: rgba(255,255,255,0.62);
          font-size: 0.74rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          margin: 0.1rem 0 0.55rem 0;
        }

        [data-testid="stSidebar"] div[data-testid="stPageLink"] {
          margin-bottom: 0.15rem;
        }

        [data-testid="stSidebar"] div[data-testid="stPageLink"] a,
        [data-testid="stSidebar"] div[data-testid="stPageLink"] span {
          border-radius: 8px !important;
        }

        [data-testid="stSidebar"] .mvv-sidebar-account-copy {
          color: rgba(255,255,255,0.76);
          font-size: 0.84rem;
          line-height: 1.45;
        }

        [data-testid="stSidebar"] .mvv-sidebar-divider {
          height: 1px;
          margin: 0.9rem 0 0.95rem 0;
          background: rgba(255,255,255,0.08);
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_navigation(profile: Optional[Dict[str, Any]] = None) -> None:
    _render_sidebar_css()
    with st.sidebar:
        st.markdown('<div class="mvv-sidebar-nav-label">Navigatie</div>', unsafe_allow_html=True)
        for page_path, label in SIDEBAR_PAGE_LINKS:
            st.page_link(page_path, label=label)


def render_sidebar_footer(profile: Optional[Dict[str, Any]] = None, show_debug: bool = False) -> None:
    resolved_profile = profile or {}
    email = str(
        st.session_state.get("user_email")
        or resolved_profile.get("email")
        or ""
    ).strip() or "--"
    role_label = _format_role_label(resolved_profile.get("role") or st.session_state.get("role"))

    with st.sidebar:
        st.markdown('<div class="mvv-sidebar-footer-anchor"></div>', unsafe_allow_html=True)
        if SIDEBAR_BETA_PAGE_LINKS:
            with st.expander("Beta pagina's", expanded=False):
                for page_path, label in SIDEBAR_BETA_PAGE_LINKS:
                    st.page_link(page_path, label=label)

        st.markdown('<div class="mvv-sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="mvv-sidebar-nav-label">Account</div>', unsafe_allow_html=True)
        with st.expander("Account info", expanded=False):
            st.markdown(
                f"""
                <div class="mvv-sidebar-account-copy"><strong>Email</strong><br>{email}</div>
                <div style="height:0.7rem;"></div>
                <div class="mvv-sidebar-account-copy"><strong>Rol</strong><br>{role_label}</div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("Logout", use_container_width=True, key="sidebar_logout_btn"):
            _sidebar_logout_action()

        if show_debug:
            with st.expander("Auth debug", expanded=True):
                cm = cookie_mgr()
                st.write("session access:", bool(st.session_state.get("access_token")))
                st.write("cookie access:", bool(cm.get("sb_access")))
                st.write("cookie refresh:", bool(cm.get("sb_refresh")))
                st.write("auth_err:", st.session_state.get("auth_err"))


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
# SUPABASE CLIENT  (FIX: per-session, geen global cache)
# ============================================================

def get_sb():
    """
    Supabase client per Streamlit session.

    Waarom:
    - st.cache_resource deelt objecten tussen users/sessies.
    - postgrest auth header kan dan door andere users overschreven worden.
    - op mobiel (meer reruns) resulteert dat in RLS errors bij writes.
    """
    if create_client is None:
        return None

    url = st.secrets.get("SUPABASE_URL", "").strip()
    key = st.secrets.get("SUPABASE_ANON_KEY", "").strip()
    if not url or not key:
        return None

    # per-session client
    if "_sb_client" not in st.session_state or st.session_state.get("_sb_client") is None:
        st.session_state["_sb_client"] = create_client(url, key)

    sb = st.session_state["_sb_client"]

    # zet ALTIJD postgrest auth voor deze sessie (als token er al is)
    tok = _get_access_token_from_state()
    _set_postgrest_auth_safely(sb, tok)

    return sb


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


def ensure_valid_session(sb=None) -> bool:
    if sb is None:
        sb = get_sb()
    if sb is None:
        st.session_state["auth_err"] = "Supabase client niet beschikbaar"
        return False

    token = _get_access_token_from_state()
    if token:
        try:
            _set_postgrest_auth_safely(sb, token)
            user_resp = sb.auth.get_user(token)
            user = getattr(user_resp, "user", None)
            if user is not None and getattr(user, "id", None):
                st.session_state["access_token"] = token
                st.session_state["user_id"] = str(user.id)
                return True
        except Exception as exc:
            st.session_state["auth_err"] = str(exc)
        clear_auth_state(clear_cookies=False)

    ok = try_restore_or_refresh_session(sb)
    return bool(ok and _get_access_token_from_state())


# ============================================================
# AUTH GATE
# ============================================================

def require_auth() -> None:
    sb = get_sb()
    if ensure_valid_session(sb):
        return
    redirect_to_login("Sessie verlopen. Log opnieuw in.", clear_cookies=True)


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

def _player_cache_scope() -> str:
    profile = st.session_state.get("_profile_cache")
    if not isinstance(profile, dict):
        profile = {}

    user_id = str(profile.get("user_id") or st.session_state.get("user_id") or "anon")
    role = normalize_role(profile.get("role") or st.session_state.get("role"))
    return f"{user_id}:{role}"

@st.cache_data(show_spinner=False, ttl=300)
def _list_players_cached(access_scope: str, _cache_buster: str = "v2") -> List[Dict[str, Any]]:
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
    return _list_players_cached(_player_cache_scope(), "v2")


@st.cache_data(show_spinner=False, ttl=300)
def _get_player_name_cached(player_id: str, access_scope: str, _cache_buster: str = "v2") -> str:
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
    return _get_player_name_cached(str(player_id), _player_cache_scope(), "v2")


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
