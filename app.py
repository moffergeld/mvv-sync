# app.py
import base64
from pathlib import Path

import streamlit as st
from supabase import create_client
import extra_streamlit_components as stx  # pip install extra-streamlit-components

# ----------------------------
# Page config (1x!)
# ----------------------------
st.set_page_config(page_title="MVV Dashboard", layout="wide")

# ----------------------------
# Maintenance toggle
# ----------------------------
MAINTENANCE_MODE = True
MAINTENANCE_TITLE = "⚠️ MAINTENANCE"
MAINTENANCE_TEXT = "Er wordt onderhoud uitgevoerd. Je kunt mogelijk (tijdelijk) uitgelogd worden."

# ----------------------------
# Supabase
# ----------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ----------------------------
# Cookie settings
# ----------------------------
COOKIE_ACCESS = "mvv_sb_access"
COOKIE_REFRESH = "mvv_sb_refresh"
COOKIE_EMAIL = "mvv_sb_email"
COOKIE_TTL_DAYS = 14  # pas aan (bijv. 7, 30)

# ----------------------------
# Cookie manager (singleton)
# ----------------------------
def cookie_manager() -> stx.CookieManager:
    if "_cookie_mgr" not in st.session_state:
        st.session_state["_cookie_mgr"] = stx.CookieManager()
    return st.session_state["_cookie_mgr"]

cm = cookie_manager()
_ = cm.get_all()  # initialiseert cookie state

# ----------------------------
# Helpers
# ----------------------------
def img_to_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()

def maintenance_banner():
    if not MAINTENANCE_MODE:
        return
    st.markdown(
        f"""
        <style>
        .maintenance-banner{{
            padding: 16px 18px;
            border-radius: 14px;
en            border: 2px solid rgba(255, 0, 0, 0.55);
            background: rgba(255, 0, 0, 0.12);
            color: #fff;
            font-weight: 800;
            font-size: 20px;
            line-height: 1.25;
            margin: 8px 0 16px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,.25);
        }}
        .maintenance-banner small{{
            display:block;
            font-weight: 600;
            font-size: 14px;
            opacity: .9;
            margin-top: 6px;
        }}
        </style>

        <div class="maintenance-banner">
            {MAINTENANCE_TITLE}
            <small>{MAINTENANCE_TEXT}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

def clear_auth_and_cookies():
    for k in ["access_token", "refresh_token", "user_email", "role", "player_id", "profile_loaded"]:
        st.session_state.pop(k, None)

    cm.delete(COOKIE_ACCESS)
    cm.delete(COOKIE_REFRESH)
    cm.delete(COOKIE_EMAIL)

def persist_auth_to_cookies(access_token: str, refresh_token: str, email: str):
    ttl = COOKIE_TTL_DAYS * 24 * 60 * 60
    cm.set(COOKIE_ACCESS, access_token, expires_at=ttl)
    cm.set(COOKIE_REFRESH, refresh_token, expires_at=ttl)
    cm.set(COOKIE_EMAIL, email, expires_at=ttl)

def restore_auth_from_cookies():
    if "access_token" in st.session_state and "refresh_token" in st.session_state:
        return

    access = cm.get(COOKIE_ACCESS)
    refresh = cm.get(COOKIE_REFRESH)
    email = cm.get(COOKIE_EMAIL)

    if access and refresh:
        st.session_state["access_token"] = access
        st.session_state["refresh_token"] = refresh
        if email:
            st.session_state["user_email"] = email

def refresh_auth_each_run() -> bool:
    rt = st.session_state.get("refresh_token")
    if not rt:
        return False

    try:
        new_sess = sb.auth.refresh_session(rt)
        access_token = new_sess.session.access_token
        refresh_token = new_sess.session.refresh_token

        st.session_state["access_token"] = access_token
        st.session_state["refresh_token"] = refresh_token

        try:
            sb.postgrest.auth(access_token)
        except Exception:
            pass

        persist_auth_to_cookies(
            access_token=access_token,
            refresh_token=refresh_token,
            email=st.session_state.get("user_email", ""),
        )
        return True

    except Exception:
        return False

def login_ui():
    # maintenance ook voor inloggen
    maintenance_banner()

    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Sign in", use_container_width=True, key="btn_signin"):
        try:
            res = sb.auth.sign_in_with_password({"email": email, "password": password})

            access_token = res.session.access_token
            refresh_token = res.session.refresh_token

            st.session_state["access_token"] = access_token
            st.session_state["refresh_token"] = refresh_token
            st.session_state["user_email"] = email

            try:
                sb.postgrest.auth(access_token)
            except Exception:
                pass

            st.session_state.pop("role", None)
            st.session_state.pop("player_id", None)
            st.session_state.pop("profile_loaded", None)

            persist_auth_to_cookies(access_token, refresh_token, email)

            st.rerun()
        except Exception as e:
            st.error(f"Sign in mislukt: {e}")

def logout_button():
    if st.button("Logout", use_container_width=True, key="btn_logout"):
        try:
            sb.auth.sign_out()
        except Exception:
            pass
        clear_auth_and_cookies()
        st.rerun()

def load_profile():
    """
    Verwacht public.profiles:
      - user_id (uuid)
      - role (user_role_v2 / text)
      - player_id (uuid, optional)
    """
    if st.session_state.get("profile_loaded"):
        return

    import requests
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {st.session_state['access_token']}",
        "Content-Type": "application/json",
    }
    r = requests.get(f"{SUPABASE_URL}/auth/v1/user", headers=headers, timeout=30)
    if not r.ok:
        st.error(f"Kon user niet ophalen: {r.status_code} {r.text}")
        st.stop()

    user_id = r.json().get("id")
    if not user_id:
        st.error("Kon user_id niet bepalen.")
        st.stop()

    prof = (
        sb.table("profiles")
        .select("user_id, role, player_id")
        .eq("user_id", user_id)
        .single()
        .execute()
        .data
    )

    role = (prof or {}).get("role")
    player_id = (prof or {}).get("player_id")

    st.session_state["role"] = str(role) if role is not None else None
    st.session_state["player_id"] = player_id
    st.session_state["profile_loaded"] = True

def tile(tile_id: str, img_path: str, target_page: str | None, disabled: bool = False):
    b64 = img_to_b64(img_path)
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

    if disabled:
        st.button("Geen toegang", use_container_width=True, disabled=True, key=f"btn_{tile_id}_noaccess")
        return

    if target_page is None:
        st.button("Coming soon", use_container_width=True, disabled=True, key=f"btn_{tile_id}_soon")
        return

    if st.button("Open", use_container_width=True, key=f"btn_{tile_id}_open"):
        st.switch_page(target_page)

# ----------------------------
# CSS (tiles even breed als Open-knop)
# ----------------------------
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

# ----------------------------
# Auth bootstrap:
# - restore from cookies
# - auto-refresh each run
# - fallback to login if refresh fails
# ----------------------------
restore_auth_from_cookies()

if "refresh_token" in st.session_state:
    ok = refresh_auth_each_run()
    if not ok:
        clear_auth_and_cookies()

if "access_token" not in st.session_state:
    login_ui()
    st.stop()

# ----------------------------
# profiel laden (role + player_id)
# ----------------------------
load_profile()

# ----------------------------
# Sidebar / top
# ----------------------------
st.sidebar.success(f"Ingelogd: {st.session_state.get('user_email','')}")
st.sidebar.info(f"Role: {st.session_state.get('role','')}")
logout_button()

# maintenance ook NA inloggen
maintenance_banner()

st.title("MVV Dashboard")
st.write("Klik op een tegel om een module te openen.")

role = (st.session_state.get("role") or "").lower()
is_player = role == "player"

# ----------------------------
# Tiles (6 naast elkaar)
# ----------------------------
c1, c2, c3, c4, c5, c6 = st.columns(6, gap="large")

with c1:
    tile("player", "Assets/Afbeeldingen/Script/Player_page.PNG", "pages/01_Player_Page.py")

with c2:
    tile("matchreports", "Assets/Afbeeldingen/Script/Match Report.PNG", "pages/03_Match_Reports")

with c3:
    tile("gpsdata", "Assets/Afbeeldingen/Script/GPS_Data.PNG", "pages/02_GPS_Data.py", disabled=is_player)

with c4:
    tile("gpsimport", "Assets/Afbeeldingen/Script/GPS_Import.PNG", "pages/06_GPS_Import.py", disabled=is_player)

with c5:
    tile("medical", "Assets/Afbeeldingen/Script/Medical.PNG", None, disabled=True)

with c6:
    tile("accounts", "Assets/Afbeeldingen/Script/Accounts.PNG", None, disabled=True)
