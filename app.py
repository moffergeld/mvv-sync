# app.py
import base64
from pathlib import Path

import streamlit as st
from supabase import create_client

# ----------------------------
# Page config (1x!)
# ----------------------------
st.set_page_config(page_title="MVV Dashboard", layout="wide")

# ----------------------------
# Supabase
# ----------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ----------------------------
# Helpers
# ----------------------------
def img_to_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()

def login_ui():
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Sign in", use_container_width=True, key="btn_signin"):
        try:
            res = sb.auth.sign_in_with_password({"email": email, "password": password})
            st.session_state["access_token"] = res.session.access_token
            st.session_state["user_email"] = email

            try:
                sb.postgrest.auth(res.session.access_token)
            except Exception:
                pass

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

    # user-id ophalen via auth endpoint (werkt met jouw JWT)
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

    # profile uit profiles
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

    # role kan enum zijn -> cast naar str
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
        height: 350px;
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
# Auth gate
# ----------------------------
if "access_token" not in st.session_state:
    login_ui()
    st.stop()

try:
    sb.postgrest.auth(st.session_state["access_token"])
except Exception:
    pass

# profiel laden (role + player_id)
load_profile()

# ----------------------------
# Sidebar / top
# ----------------------------
st.sidebar.success(f"Ingelogd: {st.session_state.get('user_email','')}")
st.sidebar.info(f"Role: {st.session_state.get('role','')}")
logout_button()

st.title("MVV Dashboard")
st.write("Klik op een tegel om een module te openen.")

role = (st.session_state.get("role") or "").lower()
is_player = role == "player"

# ----------------------------
# Tiles (4 naast elkaar)
# ----------------------------
c1, c2, c3, c4 = st.columns(4, gap="large")

with c1:
    tile("player", "Assets/Afbeeldingen/Script/Player_page.PNG", "pages/01_Player_Page.py")

with c2:
    tile("gpsdata", "Assets/Afbeeldingen/Script/GPS_Data.PNG", "pages/02_GPS_Data.py", disabled=is_player)

with c3:
    tile("gpsimport", "Assets/Afbeeldingen/Script/GPS_Import.PNG", "pages/06_GPS_Import.py", disabled=is_player)

with c4:
    tile("medical", "Assets/Afbeeldingen/Script/Medical.PNG", None, disabled=True)
