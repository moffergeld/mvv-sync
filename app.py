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
# Tiles config
# =========================
TILES = [
    {
        "key": "player",
        "img": asset("Assets/Afbeeldingen/Script/Player_page.PNG"),
        "page": "pages/01_Player_Page.py",
        "enabled": True,
    },
    {
        "key": "gps_data",
        "img": asset("Assets/Afbeeldingen/Script/GPS_Data.PNG"),
        "page": "pages/02_GPS_Data.py",
        "enabled": True,
    },
    {
        "key": "gps_import",
        "img": asset("Assets/Afbeeldingen/Script/GPS_Import.PNG"),
        "page": "pages/06_GPS_Import.py",
        "enabled": True,
    },
    {
        "key": "medical",
        "img": asset("Assets/Afbeeldingen/Script/Medical.PNG"),
        "page": None,  # bestaat nog niet
        "enabled": False,
    },
]

# =========================
# CSS
# - 4 naast elkaar
# - geen lege "boxes" boven de afbeeldingen
# =========================
st.markdown(
    """
    <style>
    /* Verwijder extra witruimte rondom images/containers */
    div[data-testid="stImage"] { margin: 0 !important; padding: 0 !important; }
    div[data-testid="stImage"] img { border-radius: 18px; display:block; }

    /* Compacter layout */
    .tile-card {
        padding: 10px 10px 12px 10px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,.10);
        background: rgba(255,255,255,.03);
    }

    /* Knoppen compacter */
    div.stButton > button {
        padding-top: 0.35rem !important;
        padding-bottom: 0.35rem !important;
        border-radius: 12px !important;
        margin-top: 8px !important;
    }

    /* Belangrijk: zorg dat er geen lege placeholders boven komen door containers */
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stImage"]) {
        margin-top: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Home (tegels)
# =========================
st.title("MVV Dashboard")
st.write("Klik op een tegel om een module te openen.")

# 4 kolommen naast elkaar
cols = st.columns(4, gap="small")

for i, t in enumerate(TILES):
    with cols[i % 4]:
        st.markdown("<div class='tile-card'>", unsafe_allow_html=True)

        # Iets kleiner zodat 4 naast elkaar netjes past
        st.image(t["img"], use_container_width=True)

        if t["enabled"]:
            if st.button("Open", key=f"open_{t['key']}", use_container_width=True):
                st.switch_page(t["page"])
        else:
            st.button("Coming soon", key=f"open_{t['key']}", use_container_width=True, disabled=True)

        st.markdown("</div>", unsafe_allow_html=True)
