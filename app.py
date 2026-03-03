# app.py
# ============================================================
# MVV Dashboard - Main App (Streamlit)
# - Player ziet alleen Player + Match Reports tiles
# - Staff ziet alle tiles (in 2 rijen: 3 + 3)
# - Tiles gebruiken st.image() i.p.v. base64 HTML
# - Auth debug expander alleen zichtbaar in DIAG_MODE (?diag=1)
# - Auth/profile centraal via roles.py
# ============================================================

from pathlib import Path

import streamlit as st

from roles import (
    get_sb,
    cookie_mgr,
    set_tokens_in_cookie,
    clear_tokens_in_cookie,
    try_restore_or_refresh_session,
    get_profile,
)

# Optional: alleen voor CookieManager component init (soms nodig)
import extra_streamlit_components as stx  # noqa: F401


# ============================================================
# 1) CONFIG & MODES
# ============================================================

st.set_page_config(page_title="MVV Dashboard", layout="wide")

DIAG_MODE = st.query_params.get("diag") == "1"
SAFE_MODE = st.query_params.get("safe") == "1"

MAINTENANCE_MODE = False
MAINTENANCE_TITLE = "⚠️ MAINTENANCE"
MAINTENANCE_TEXT = "Er wordt onderhoud uitgevoerd. Je kunt mogelijk (tijdelijk) uitgelogd worden."

if DIAG_MODE:
    st.title("DIAG OK")
    st.write("Als je dit ziet, werkt Streamlit op dit toestel/netwerk.")
    st.write("Zet diag uit door ?diag=0 of verwijder de query parameter.")
    st.stop()


# ============================================================
# 2) SUPABASE CLIENT
# ============================================================

sb = get_sb()
if sb is None:
    st.error("Supabase client niet beschikbaar (secrets ontbreken of create_client faalt).")
    st.stop()


# ============================================================
# 3) UI HELPERS
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


def tile(tile_id: str, img_path: str, target_page: str | None, disabled: bool = False):
    """
    Tile met image + knop.
    - In SAFE_MODE geen images (lichter)
    - In normale mode: st.image() i.p.v. base64 HTML (lichter)
    """
    if not SAFE_MODE:
        p = Path(img_path)
        if p.exists():
            st.image(str(p), use_container_width=True)
        else:
            st.warning(f"Missing asset: {img_path}")

    if disabled:
        st.button("Geen toegang", use_container_width=True, disabled=True, key=f"btn_{tile_id}_noaccess")
        return

    if target_page is None:
        st.button("Coming soon", use_container_width=True, disabled=True, key=f"btn_{tile_id}_soon")
        return

    if st.button("Open", use_container_width=True, key=f"btn_{tile_id}_open"):
        st.switch_page(target_page)


# ============================================================
# 4) LOGIN / LOGOUT UI
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

        # minimale settle (optioneel; zet naar 0 als alles stabiel is)
        import time
        time.sleep(0.10)

        # reset profile cache
        st.session_state.pop("_profile_cache", None)
        st.session_state.pop("role", None)
        st.session_state.pop("player_id", None)

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

        import time
        time.sleep(0.10)

        st.session_state.clear()
        st.rerun()


# ============================================================
# 5) CSS (HOME)
# ============================================================

# Styling voor st.image tiles (globaal op deze pagina):
# - ronde hoeken + schaduw
# - voorkomt dat tiles er "vlak" uitzien zonder HTML wrapper
if not SAFE_MODE:
    st.markdown(
        """
        <style>
          /* st.image wrapper */
          [data-testid="stImage"] img{
            border-radius: 22px;
            box-shadow: 0 10px 30px rgba(0,0,0,.35);
            border: 1px solid rgba(255,255,255,.10);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 6) BOOT COOKIE COMPONENT EARLY
# ============================================================

try:
    cm = cookie_mgr()
    _ = cm.get("sb_refresh")
except Exception:
    pass


# ============================================================
# 7) AUTH GATE (restore -> login)
# ============================================================

if "access_token" not in st.session_state:
    restored = try_restore_or_refresh_session(sb)
    if not restored:
        login_ui()
        st.stop()


# ============================================================
# 8) PROFILE LOAD (CENTRAAL IN roles.py)
# ============================================================

profile = get_profile(sb)
if not profile:
    clear_tokens_in_cookie()
    st.session_state.clear()
    st.error("Geen profiel gevonden of geen rechten. Log opnieuw in.")
    st.stop()


# ============================================================
# 9) SIDEBAR
# ============================================================

st.sidebar.success(f"Ingelogd: {st.session_state.get('user_email','')}")
st.sidebar.info(f"Role: {st.session_state.get('role','')}")

# Debug alleen in DIAG_MODE (via ?diag=1)
if DIAG_MODE:
    with st.sidebar.expander("Auth debug", expanded=True):
        cm = cookie_mgr()
        st.write("session access:", bool(st.session_state.get("access_token")))
        st.write("cookie access:", bool(cm.get("sb_access")))
        st.write("cookie refresh:", bool(cm.get("sb_refresh")))
        st.write("auth_err:", st.session_state.get("auth_err"))

logout_button()
maintenance_banner()


# ============================================================
# 10) HOME UI (TILES) - PLAYER vs STAFF
# ============================================================

st.title("MVV Dashboard")
if SAFE_MODE:
    st.warning("Safe mode actief (minimale UI). Zet uit door ?safe=0 te gebruiken.")
st.write("Klik op een tegel om een module te openen.")

role = (st.session_state.get("role") or "").lower()
is_player = role == "player"

if is_player:
    # Alleen 2 tiles renderen
    c1, c2 = st.columns(2, gap="large")
    with c1:
        tile("player", "Assets/Afbeeldingen/Script/Player_page.PNG", "pages/01_Player_Page.py")
    with c2:
        tile("matchreports", "Assets/Afbeeldingen/Script/Match Report.PNG", "pages/03_Match_Reports.py")

else:
    # Staff: 2 rijen (3 + 3) -> lichtere render dan 6 in 1 rij
    r1c1, r1c2, r1c3 = st.columns(3, gap="large")
    with r1c1:
        tile("player", "Assets/Afbeeldingen/Script/Player_page.PNG", "pages/01_Player_Page.py")
    with r1c2:
        tile("matchreports", "Assets/Afbeeldingen/Script/Match Report.PNG", "pages/03_Match_Reports.py")
    with r1c3:
        tile("gpsdata", "Assets/Afbeeldingen/Script/GPS_Data.PNG", "pages/02_GPS_Data.py")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    r2c1, r2c2, r2c3 = st.columns(3, gap="large")
    with r2c1:
        tile("gpsimport", "Assets/Afbeeldingen/Script/GPS_Import.PNG", "pages/06_GPS_Import.py")
    with r2c2:
        tile("medical", "Assets/Afbeeldingen/Script/Medical.PNG", None, disabled=True)
    with r2c3:
        tile("accounts", "Assets/Afbeeldingen/Script/Accounts.PNG", None, disabled=True)
