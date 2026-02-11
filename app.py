import streamlit as st
from supabase import create_client
from pathlib import Path

# =========================
# Config
# =========================
st.set_page_config(page_title="Inlogpagina", layout="wide")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

BASE_DIR = Path(__file__).resolve().parent

def asset(rel_path: str) -> str:
    return str((BASE_DIR / rel_path).resolve())

# =========================
# Auth UI
# =========================
def login_ui():
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Sign in", use_container_width=True):
        try:
            res = sb.auth.sign_in_with_password({"email": email, "password": password})
            st.session_state["access_token"] = res.session.access_token
            st.session_state["user_email"] = email

            try:
                sb.postgrest.auth(res.session.access_token)
            except Exception:
                pass

            st.rerun()
        except Exception as e:
            st.error(f"Sign in mislukt: {e}")

def logout_button():
    if st.button("Logout", use_container_width=True):
        try:
            sb.auth.sign_out()
        except Exception:
            pass
        st.session_state.clear()
        st.rerun()

# =========================
# Auth gate
# =========================
if "access_token" not in st.session_state:
    st.title("Inlogpagina")
    login_ui()
    st.stop()

try:
    sb.postgrest.auth(st.session_state["access_token"])
except Exception:
    pass

# =========================
# Sidebar
# =========================
st.sidebar.success(f"Ingelogd: {st.session_state.get('user_email','')}")
logout_button()

# =========================
# Tiles config (matcht jouw repo)
# =========================
TILES = [
    {
        "label": "Player Page",
        "img": asset("Assets/Afbeeldingen/Script/Player_page.PNG"),
        "page": "pages/01_Player_Page.py",
        "enabled": True,
    },
    {
        "label": "GPS Data",
        "img": asset("Assets/Afbeeldingen/Script/GPS_Data.PNG"),
        "page": "pages/02_GPS_Data.py",
        "enabled": True,
    },
    {
        "label": "GPS Import",
        "img": asset("Assets/Afbeeldingen/Script/GPS_Import.PNG"),
        "page": "pages/06_GPS_Import.py",
        "enabled": True,
    },
    {
        "label": "Medical",
        "img": asset("Assets/Afbeeldingen/Script/Medical.PNG"),
        "page": None,  # bestaat nog niet
        "enabled": False,
        "badge": "Coming soon",
    },
]

# =========================
# Home (tegels)
# =========================
st.title("MVV Dashboard")
st.write("Klik op een tegel om een module te openen.")

st.markdown(
    """
    <style>
    .tile-wrap {
        display:flex; flex-direction:column; gap:10px;
        padding:14px; border-radius:18px;
        border:1px solid rgba(255,255,255,.08);
        background: rgba(255,255,255,.03);
    }
    .tile-title { font-size:16px; font-weight:700; }
    .tile-badge {
        display:inline-block; padding:4px 10px; border-radius:999px;
        font-size:12px;
        background: rgba(255,0,51,.18);
        border:1px solid rgba(255,0,51,.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

cols = st.columns(3)
for i, t in enumerate(TILES):
    with cols[i % 3]:
        st.image(t["img"], use_container_width=True)

        st.markdown(
            f"<div class='tile-wrap'><div class='tile-title'>{t['label']}</div>",
            unsafe_allow_html=True,
        )

        if t.get("badge"):
            st.markdown(f"<span class='tile-badge'>{t['badge']}</span>", unsafe_allow_html=True)

        if t["enabled"]:
            if st.button("Open", key=f"open_{t['label']}", use_container_width=True):
                st.switch_page(t["page"])
        else:
            st.button("Open", key=f"open_{t['label']}", use_container_width=True, disabled=True)

        st.markdown("</div>", unsafe_allow_html=True)
