# app.py
import base64
from pathlib import Path
import time

import requests
import streamlit as st
import extra_streamlit_components as stx
from supabase import create_client

# ----------------------------
# Page config (1x!)
# ----------------------------
st.set_page_config(page_title="MVV Dashboard", layout="wide")

# --------------------------------------------------
# QUERY PARAM MODES (must run BEFORE any heavy init)
# --------------------------------------------------
DIAG_MODE = st.query_params.get("diag") == "1"
SAFE_MODE = st.query_params.get("safe") == "1"

if DIAG_MODE:
    st.title("DIAG OK")
    st.write("Als je dit ziet, werkt Streamlit op dit toestel/netwerk.")
    st.write("Als het nog steeds wit blijft, dan blokkeert het toestel/netwerk Streamlit assets/JS (adblock/VPN/Private DNS/WebView/netwerkfilter).")
    st.write("")
    st.write("Testlinks:")
    st.write("- Normaal: /")
    st.write("- Safe mode: ?safe=1")
    st.stop()

# ----------------------------
# Maintenance toggle
# ----------------------------
MAINTENANCE_MODE = False
MAINTENANCE_TITLE = "⚠️ MAINTENANCE"
MAINTENANCE_TEXT = "Er wordt onderhoud uitgevoerd. Je kunt mogelijk (tijdelijk) uitgelogd worden."

# ----------------------------
# Supabase (fail-safe)
# ----------------------------
try:
    SUPABASE_URL = st.secrets.get("SUPABASE_URL", "").strip()
    SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "").strip()
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.error("Missing secrets: SUPABASE_URL / SUPABASE_ANON_KEY")
        st.stop()
    sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
except Exception as e:
    st.error(f"Boot error (supabase/secrets): {e}")
    st.stop()

# ----------------------------
# Helpers
# ----------------------------
def normalize_role(v):
    if v is None:
        return "player"
    s = str(v).strip().lower()
    if "." in s:
        s = s.split(".")[-1]
    if "::" in s:
        s = s.split("::")[0]
    return s or "player"


def img_to_b64_safe(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode()


def cookie_mgr():
    return stx.CookieManager()


def set_tokens_in_cookie(access_token: str, refresh_token: str, email: str | None = None):
    cm = cookie_mgr()
    # access kort, refresh lang
    cm.set("sb_access", str(access_token or ""), max_age=60 * 60)               # 1 uur
    cm.set("sb_refresh", str(refresh_token or ""), max_age=60 * 60 * 24 * 30)   # 30 dagen
    if email:
        cm.set("sb_email", str(email), max_age=60 * 60 * 24 * 30)


def clear_tokens_in_cookie():
    cm = cookie_mgr()
    # delete bestaat niet in alle versies -> overschrijven + korte expiry
    cm.set("sb_access", "", max_age=1)
    cm.set("sb_refresh", "", max_age=1)
    cm.set("sb_email", "", max_age=1)


def _set_postgrest_auth_safely(token: str | None):
    if not token:
        return
    try:
        sb.postgrest.auth(token)
    except Exception:
        pass


def try_restore_or_refresh_session() -> bool:
    """
    Herstelt sessie vanuit cookies wanneer session_state leeg is geraakt
    (bijv. mobiel tab-switch / reconnect / websocket reset).
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

    # Soft retries bij korte netwerkdip
    for _ in range(2):
        try:
            # Probeer eerst bestaande sessie te zetten
            if access:
                try:
                    sb.auth.set_session(access, refresh)
                except Exception:
                    pass

            # Refresh naar nieuwe tokens
            refreshed = sb.auth.refresh_session(refresh)
            sess = getattr(refreshed, "session", None)

            if sess and getattr(sess, "access_token", None) and getattr(sess, "refresh_token", None):
                st.session_state["access_token"] = sess.access_token
                st.session_state["sb_session"] = sess

                _set_postgrest_auth_safely(sess.access_token)
                set_tokens_in_cookie(
                    sess.access_token,
                    sess.refresh_token,
                    st.session_state.get("user_email"),
                )

                # Geef cookie component tijd om te schrijven
                time.sleep(0.35)
                return True

        except Exception as e:
            last_err = e
            time.sleep(0.35)

    st.session_state["auth_err"] = str(last_err) if last_err else "Unknown auth restore error"
    return False


def maintenance_banner():
    if not MAINTENANCE_MODE:
        return

    st.markdown(
        f"""
        <style>
        .maintenance-banner{{
            padding: 16px 18px;
            border-radius: 14px;
            border: 2px solid rgba(255, 0, 0, 0.55);
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


def login_ui():
    maintenance_banner()
    st.title("Login")

    # form voorkomt reruns tijdens typen (mobiel stabieler)
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
        st.session_state["sb_session"] = sess  # handig voor andere pages

        # Tokens persistent bewaren (belangrijk voor mobiel / tab switch)
        set_tokens_in_cookie(token, refresh_token, email)

        # Geef browser/component tijd om cookies echt te schrijven vóór rerun
        time.sleep(0.6)

        # PostgREST auth direct zetten
        _set_postgrest_auth_safely(token)

        # reset profiel-cache
        st.session_state.pop("role", None)
        st.session_state.pop("player_id", None)
        st.session_state.pop("profile_loaded", None)

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
        # Ook hier kort wachten zodat cookie-clears doorkomen
        time.sleep(0.35)

        st.session_state.clear()
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

    token = st.session_state.get("access_token")
    if not token:
        st.error("Niet ingelogd (access_token ontbreekt).")
        st.stop()

    # Zorg dat sb.table(...) met JWT werkt
    _set_postgrest_auth_safely(token)

    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    r = requests.get(f"{SUPABASE_URL}/auth/v1/user", headers=headers, timeout=30)

    if r.status_code in (401, 403):
        # token mogelijk verlopen -> probeer stil te herstellen via refresh cookie
        st.session_state.pop("access_token", None)
        if try_restore_or_refresh_session():
            st.session_state.pop("profile_loaded", None)
            st.rerun()

        # alleen nu echt logout
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

    # maybe_single i.p.v. single: 0 rows -> None (geen crash)
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
        st.error(
            "Geen profiel gevonden voor dit account (of geen rechten via RLS).\n\n"
            f"user_id = {user_id}\n\n"
            "Controleer in Supabase: Authentication → Users (id) moet exact gelijk zijn aan profiles.user_id "
            "en dat RLS SELECT op profiles voor eigen rij is toegestaan."
        )
        st.stop()

    st.session_state["role"] = normalize_role(prof.get("role"))
    st.session_state["player_id"] = prof.get("player_id")
    st.session_state["profile_loaded"] = True


def tile(tile_id: str, img_path: str, target_page: str | None, disabled: bool = False):
    # In SAFE_MODE geen images/CSS-heavy HTML
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


# ----------------------------
# CSS (alleen als niet safe)
# ----------------------------
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

# ------------------------------------
# Cookie component vroeg initialiseren
# (helpt met betrouwbaarder cookie read/write)
# ------------------------------------
try:
    _cm_boot = cookie_mgr()
    _ = _cm_boot.get("sb_refresh")
except Exception:
    pass

# ----------------------------
# Auth gate
# ----------------------------
if "access_token" not in st.session_state:
    # Probeer sessie te herstellen uit cookies (mobiel/tab-switch proof)
    restored = try_restore_or_refresh_session()
    if not restored:
        login_ui()
        st.stop()

# Zorg opnieuw dat PostgREST auth goed staat (voor het geval rerun)
_set_postgrest_auth_safely(st.session_state.get("access_token"))

# profiel laden (role + player_id)
load_profile()

# ----------------------------
# Sidebar / top
# ----------------------------
st.sidebar.success(f"Ingelogd: {st.session_state.get('user_email','')}")
st.sidebar.info(f"Role: {st.session_state.get('role','')}")

# Tijdelijke debug (kan je later weghalen)
with st.sidebar.expander("Auth debug", expanded=False):
    cm = cookie_mgr()
    st.write("session access:", bool(st.session_state.get("access_token")))
    st.write("cookie access:", bool(cm.get("sb_access")))
    st.write("cookie refresh:", bool(cm.get("sb_refresh")))
    st.write("auth_err:", st.session_state.get("auth_err"))

logout_button()

maintenance_banner()

st.title("MVV Dashboard")
if SAFE_MODE:
    st.warning("Safe mode actief (minimale UI). Zet uit door ?safe=0 te gebruiken.")
st.write("Klik op een tegel om een module te openen.")

role = (st.session_state.get("role") or "").lower()
is_player = role == "player"

# ----------------------------
# Tiles
# ----------------------------
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
