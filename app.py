# app.py
# ============================================================
# MVV Dashboard - Main App (Streamlit)
#
# Inhoud
# 1) Config & modes (diag/safe/maintenance)
# 2) Supabase init (cached)
# 3) Cookie & token helpers (30 dagen ingelogd)
# 4) Auth flow (restore -> login -> profile load)
# 5) UI helpers (banner, tiles)
# 6) Home UI (sidebar + tiles)
# ============================================================

import base64
import time
from pathlib import Path

import requests
import streamlit as st
import extra_streamlit_components as stx
from supabase import create_client


# ============================================================
# 1) CONFIG & MODES
# ============================================================

# Cookie lifetimes
ACCESS_COOKIE_SECONDS = 60 * 60           # access token cookie: 1 uur
REFRESH_COOKIE_DAYS = 30                  # refresh token cookie: 30 dagen
REFRESH_COOKIE_SECONDS = 60 * 60 * 24 * REFRESH_COOKIE_DAYS

# Streamlit page config
st.set_page_config(page_title="MVV Dashboard", layout="wide")

# Query param modes (must run BEFORE heavy init)
DIAG_MODE = st.query_params.get("diag") == "1"
SAFE_MODE = st.query_params.get("safe") == "1"

# Maintenance toggle
MAINTENANCE_MODE = False
MAINTENANCE_TITLE = "⚠️ MAINTENANCE"
MAINTENANCE_TEXT = "Er wordt onderhoud uitgevoerd. Je kunt mogelijk (tijdelijk) uitgelogd worden."


# ============================================================
# 2) DIAG / SAFE ROUTES
# ============================================================

if DIAG_MODE:
    st.title("DIAG OK")
    st.write("Als je dit ziet, werkt Streamlit op dit toestel/netwerk.")
    st.write("Test: zet diag uit door ?diag=0 (of verwijder de query param).")
    st.stop()


# ============================================================
# 3) SUPABASE INIT (CACHED)
# ============================================================

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing secrets: SUPABASE_URL / SUPABASE_ANON_KEY")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_sb_client():
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


sb = get_sb_client()


# ============================================================
# 4) SMALL HELPERS
# ============================================================

def normalize_role(v):
    if v is None:
        return "player"
    s = str(v).strip().lower()
    if "." in s:
        s = s.split(".")[-1]
    if "::" in s:
        s = s.split("::")[0]
    return s or "player"


@st.cache_data(show_spinner=False, ttl=3600)
def img_to_b64_safe(path: str) -> str | None:
    """Lees PNG/JPG bytes 1x per uur en cache als base64."""
    p = Path(path)
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode()


# ============================================================
# 5) COOKIE MANAGER & TOKENS
# ============================================================

def cookie_mgr():
    """
    CookieManager singleton:
    - altijd vaste key zodat Streamlit geen duplicate element keys krijgt
    """
    if "_cookie_mgr_instance" not in st.session_state:
        st.session_state["_cookie_mgr_instance"] = stx.CookieManager(key="mvv_cookie_mgr")
    return st.session_state["_cookie_mgr_instance"]


def set_tokens_in_cookie(access_token: str, refresh_token: str, email: str | None = None):
    """
    Access cookie: kort (1 uur)
    Refresh cookie: 30 dagen
    """
    cm = cookie_mgr()
    cm.set("sb_access", str(access_token or ""), max_age=ACCESS_COOKIE_SECONDS, key="set_sb_access")
    cm.set("sb_refresh", str(refresh_token or ""), max_age=REFRESH_COOKIE_SECONDS, key="set_sb_refresh")
    if email:
        cm.set("sb_email", str(email), max_age=REFRESH_COOKIE_SECONDS, key="set_sb_email")


def clear_tokens_in_cookie():
    cm = cookie_mgr()
    cm.set("sb_access", "", max_age=1, key="clear_sb_access")
    cm.set("sb_refresh", "", max_age=1, key="clear_sb_refresh")
    cm.set("sb_email", "", max_age=1, key="clear_sb_email")


def _set_postgrest_auth_safely(token: str | None):
    """Zet auth header voor PostgREST (table queries)."""
    if not token:
        return
    try:
        sb.postgrest.auth(token)
    except Exception:
        pass


# ============================================================
# 6) SESSION RESTORE / REFRESH (30 dagen stay logged in)
# ============================================================

def try_restore_or_refresh_session() -> bool:
    """
    Herstel sessie uit cookies wanneer session_state leeg is geraakt
    (mobiel tab-switch / reconnect / streamlit session reset).
    """
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
            # probeer eerst set_session (kan falen als access verlopen is)
            if access:
                try:
                    sb.auth.set_session(access, refresh)
                except Exception:
                    pass

            # daarna altijd refresh
            refreshed = sb.auth.refresh_session(refresh)
            sess = getattr(refreshed, "session", None)

            if sess and getattr(sess, "access_token", None) and getattr(sess, "refresh_token", None):
                st.session_state["access_token"] = sess.access_token
                st.session_state["sb_session"] = sess

                _set_postgrest_auth_safely(sess.access_token)
                set_tokens_in_cookie(sess.access_token, sess.refresh_token, st.session_state.get("user_email"))

                time.sleep(0.35)
                return True

        except Exception as e:
            last_err = e
            time.sleep(0.35)

    st.session_state["auth_err"] = str(last_err) if last_err else "Unknown auth restore error"
    return False


# ============================================================
# 7) UI: MAINTENANCE BANNER
# ============================================================

def maintenance_banner():
    if not MAINTENANCE_MODE:
        return
    st.markdown(
        f"""
        <div style="
            padding:12px 14px;
            border-radius:12px;
            border:2px solid rgba(255,0,0,.55);
            background:rgba(255,0,0,.12);
            font-weight:800;">
            {MAINTENANCE_TITLE}
            <div style="font-weight:600;opacity:.9;margin-top:6px">{MAINTENANCE_TEXT}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 8) UI: LOGIN / LOGOUT
# ============================================================

def login_ui():
    maintenance_banner()
    st.title("Login")

    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pw")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if not submitted:
        return

    try:
        res = sb.auth.sign_in_with_password({"email": email, "password": password})
        sess = getattr(res, "session", None)
        token = getattr(sess, "access_token", None)
        refresh_token = getattr(sess, "refresh_token", None)

        if not token or not refresh_token:
            st.error("Login mislukt: geen geldige sessie ontvangen.")
            return

        st.session_state["access_token"] = token
        st.session_state["user_email"] = email
        st.session_state["sb_session"] = sess

        set_tokens_in_cookie(token, refresh_token, email)
        time.sleep(0.6)

        _set_postgrest_auth_safely(token)

        # reset role/profile caches (worden opnieuw geladen)
        st.session_state.pop("role", None)
        st.session_state.pop("player_id", None)
        st.session_state.pop("profile_loaded", None)
        st.session_state.pop("_profile_cache", None)

        st.rerun()

    except Exception as e:
        st.error(f"Sign in mislukt: {e}")


def logout_button():
    if st.button("Logout", use_container_width=True, key="btn_logout"):
        try:
            sb.auth.sign_out()
        except Exception:
            pass

        clear_tokens_in_cookie()
        time.sleep(0.35)

        st.session_state.clear()
        st.rerun()


# ============================================================
# 9) PROFILE LOAD (ROLE + PLAYER_ID)
# ============================================================

def load_profile():
    """
    Haal supabase auth user_id op en daarna profiles-rij.
    Wordt 1x per sessie gezet via profile_loaded.
    """
    if st.session_state.get("profile_loaded"):
        return

    token = st.session_state.get("access_token")
    if not token:
        st.error("Niet ingelogd (access_token ontbreekt).")
        st.stop()

    _set_postgrest_auth_safely(token)

    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    r = requests.get(f"{SUPABASE_URL}/auth/v1/user", headers=headers, timeout=30)

    # Token verlopen -> probeer refresh
    if r.status_code in (401, 403):
        st.session_state.pop("access_token", None)
        if try_restore_or_refresh_session():
            st.session_state.pop("profile_loaded", None)
            st.rerun()

        clear_tokens_in_cookie()
        time.sleep(0.35)
        st.session_state.clear()
        st.error("Sessie verlopen. Log opnieuw in.")
        st.stop()

    if not r.ok:
        st.error(f"Kon user niet ophalen: {r.status_code} {r.text}")
        st.stop()

    user_id = r.json().get("id")
    if not user_id:
        clear_tokens_in_cookie()
        time.sleep(0.35)
        st.session_state.clear()
        st.error("Kon user_id niet bepalen. Log opnieuw in.")
        st.stop()

    try:
        prof = (
            sb.table("profiles")
            .select("user_id, role, player_id")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
            .data
        )
    except Exception as e:
        st.error(f"Profiles read fout: {e}")
        st.stop()

    if not prof:
        st.error("Geen profiel gevonden voor dit account.")
        st.stop()

    st.session_state["role"] = normalize_role(prof.get("role"))
    st.session_state["player_id"] = prof.get("player_id")
    st.session_state["profile_loaded"] = True


# ============================================================
# 10) UI: TILE COMPONENT
# ============================================================

def tile(tile_id: str, img_path: str, target_page: str | None, disabled: bool = False):
    """
    Tile met image + knop. In SAFE_MODE geen images (lichter voor mobile/debug).
    """
    if not SAFE_MODE:
        b64 = img_to_b64_safe(img_path)
        if b64:
            st.markdown(
                f"""
                <div class="tile-wrap">
                  <div class="tile-img">
                    <img src="data:image/png;base64,{b64}" />
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="tile-wrap">
                  <div class="tile-img" style="display:flex;align-items:center;justify-content:center;">
                    <div style="opacity:.75;">Missing asset:<br>{img_path}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if disabled:
        st.button("Geen toegang", use_container_width=True, disabled=True, key=f"btn_{tile_id}_noaccess")
        return

    if target_page is None:
        st.button("Coming soon", use_container_width=True, disabled=True, key=f"btn_{tile_id}_soon")
        return

    if st.button("Open", use_container_width=True, key=f"btn_{tile_id}_open"):
        st.switch_page(target_page)


# ============================================================
# 11) CSS (HOME ONLY)
# ============================================================

if not SAFE_MODE:
    st.markdown(
        """
        <style>
          .tile-wrap { width: 100%; }
          .tile-img{
            width: 100%;
            height: 300px;
            border-radius: 22px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,.35);
            border: 1px solid rgba(255,255,255,.10);
          }
          .tile-img img{
            width: 100%;
            height: 100%;
            object-fit: fill;
            object-position: center;
            display: block;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 12) BOOT COOKIE COMPONENT (early)
# ============================================================

try:
    _cm_boot = cookie_mgr()
    _ = _cm_boot.get("sb_refresh")
except Exception:
    pass


# ============================================================
# 13) AUTH GATE (restore -> login)
# ============================================================

if "access_token" not in st.session_state:
    restored = try_restore_or_refresh_session()
    if not restored:
        login_ui()
        st.stop()

_set_postgrest_auth_safely(st.session_state.get("access_token"))
load_profile()


# ============================================================
# 14) SIDEBAR
# ============================================================

st.sidebar.success(f"Ingelogd: {st.session_state.get('user_email','')}")
st.sidebar.info(f"Role: {st.session_state.get('role','')}")

with st.sidebar.expander("Auth debug", expanded=False):
    cm = cookie_mgr()
    st.write("session access:", bool(st.session_state.get("access_token")))
    st.write("cookie access:", bool(cm.get("sb_access")))
    st.write("cookie refresh:", bool(cm.get("sb_refresh")))
    st.write("auth_err:", st.session_state.get("auth_err"))

logout_button()
maintenance_banner()


# ============================================================
# 15) HOME UI (TILES)
# ============================================================

st.title("MVV Dashboard")
if SAFE_MODE:
    st.warning("Safe mode actief (minimale UI). Zet uit door ?safe=0 te gebruiken.")
st.write("Klik op een tegel om een module te openen.")

role = (st.session_state.get("role") or "").lower()
is_player = role == "player"

c1, c2, c3, c4, c5, c6 = st.columns(6, gap="large")

with c1:
    tile("player", "Assets/Afbeeldingen/Script/Player_page.PNG", "pages/01_Player_Page.py")

with c2:
    tile("matchreports", "Assets/Afbeeldingen/Script/Match Report.PNG", "pages/03_Match_Reports.py")

with c3:
    tile("gpsdata", "Assets/Afbeeldingen/Script/GPS_Data.PNG", "pages/02_GPS_Data.py", disabled=is_player)

with c4:
    tile("gpsimport", "Assets/Afbeeldingen/Script/GPS_Import.PNG", "pages/06_GPS_Import.py", disabled=is_player)

with c5:
    tile("medical", "Assets/Afbeeldingen/Script/Medical.PNG", None, disabled=True)

with c6:
    tile("accounts", "Assets/Afbeeldingen/Script/Accounts.PNG", None, disabled=True)
