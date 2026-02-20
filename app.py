import base64
from pathlib import Path
import requests
import streamlit as st
from supabase import create_client

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="MVV Dashboard", layout="wide")

SAFE_MODE = st.query_params.get("safe") == "1"

# --------------------------------------------------
# SUPABASE INIT (fail-safe)
# --------------------------------------------------
try:
    SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
    SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "")

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.error("Missing SUPABASE secrets.")
        st.stop()

    sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

except Exception as e:
    st.error(f"Boot error: {e}")
    st.stop()

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def img_to_b64_safe(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode()


def logout():
    try:
        sb.auth.sign_out()
    except Exception:
        pass
    st.session_state.clear()
    st.rerun()


def normalize_role(v):
    if not v:
        return "player"
    s = str(v).strip().lower()
    if "." in s:
        s = s.split(".")[-1]
    if "::" in s:
        s = s.split("::")[0]
    return s


# --------------------------------------------------
# LOGIN (form = mobiel stabiel)
# --------------------------------------------------
def login_ui():
    st.title("Login")

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if not submitted:
        return

    try:
        res = sb.auth.sign_in_with_password({"email": email, "password": password})

        if not res.session or not res.session.access_token:
            st.error("Geen geldige sessie ontvangen.")
            return

        st.session_state["access_token"] = res.session.access_token
        st.session_state["user_email"] = email
        st.session_state["sb_session"] = res.session
        st.rerun()

    except Exception as e:
        st.error(f"Login mislukt: {e}")


# --------------------------------------------------
# PROFILE LOADER (veilig)
# --------------------------------------------------
def load_profile():
    if st.session_state.get("profile_loaded"):
        return

    try:
        headers = {
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {st.session_state['access_token']}",
        }

        r = requests.get(f"{SUPABASE_URL}/auth/v1/user", headers=headers, timeout=20)

        if not r.ok:
            st.session_state.clear()
            st.error("Sessie ongeldig. Log opnieuw in.")
            st.stop()

        user_id = r.json().get("id")

        prof = (
            sb.table("profiles")
            .select("user_id, role, player_id")
            .eq("user_id", user_id)
            .single()
            .execute()
            .data
        )

        st.session_state["role"] = normalize_role((prof or {}).get("role"))
        st.session_state["player_id"] = (prof or {}).get("player_id")
        st.session_state["profile_loaded"] = True

    except Exception as e:
        st.session_state.clear()
        st.error(f"Profiel laden faalde: {e}")
        st.stop()


# --------------------------------------------------
# AUTH GATE
# --------------------------------------------------
if "access_token" not in st.session_state:
    login_ui()
    st.stop()

load_profile()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.success(f"Ingelogd: {st.session_state.get('user_email','')}")
st.sidebar.info(f"Role: {st.session_state.get('role','')}")
if st.sidebar.button("Logout"):
    logout()

st.title("MVV Dashboard")
st.write("Klik op een tegel om een module te openen.")

role = (st.session_state.get("role") or "").lower()
is_player = role == "player"

# --------------------------------------------------
# SAFE MODE (voor probleem-toestellen)
# --------------------------------------------------
if SAFE_MODE:
    st.warning("Safe mode actief (minimale UI). Voeg ?safe=0 toe om uit te zetten.")

# --------------------------------------------------
# TILE RENDER
# --------------------------------------------------
def tile(tile_id, img_path, target_page=None, disabled=False):
    if not SAFE_MODE:
        b64 = img_to_b64_safe(img_path)
        if b64:
            st.markdown(
                f"""
                <div style="border-radius:20px;overflow:hidden;
                            box-shadow:0 8px 25px rgba(0,0,0,.35);
                            margin-bottom:10px;">
                    <img src="data:image/png;base64,{b64}"
                         style="width:100%;height:260px;object-fit:fill;">
                </div>
                """,
                unsafe_allow_html=True,
            )

    if disabled:
        st.button("Geen toegang", disabled=True, use_container_width=True, key=f"{tile_id}_na")
        return

    if target_page is None:
        st.button("Coming soon", disabled=True, use_container_width=True, key=f"{tile_id}_cs")
        return

    if st.button("Open", use_container_width=True, key=f"{tile_id}_open"):
        st.switch_page(target_page)


# --------------------------------------------------
# TILE LAYOUT
# --------------------------------------------------
c1, c2, c3, c4, c5, c6 = st.columns(6)

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
