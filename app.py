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
        "button": "Open",
    },
    {
        "key": "gps_data",
        "img": asset("Assets/Afbeeldingen/Script/GPS_Data.PNG"),
        "page": "pages/02_GPS_Data.py",
        "enabled": True,
        "button": "Open",
    },
    {
        "key": "gps_import",
        "img": asset("Assets/Afbeeldingen/Script/GPS_Import.PNG"),
        "page": "pages/06_GPS_Import.py",
        "enabled": True,
        "button": "Open",
    },
    {
        "key": "medical",
        "img": asset("Assets/Afbeeldingen/Script/Medical.PNG"),
        "page": None,
        "enabled": False,
        "button": "Coming soon",
    },
]

# =========================
# CSS
# - Image exact even wide as button (same parent width)
# - Remove empty bars above images
# - Force same size (stretch allowed)
# =========================
st.markdown(
    """
    <style>
    :root{
        --tile-img-h: 250px;   /* hoogte van alle tiles */
        --tile-radius: 18px;
    }

    /* container zonder extra borders/ruimte */
    .tile-wrap{
        width:100%;
        margin:0;
        padding:0;
    }

    /* VERWIJDER de “lege boxen” (Streamlit style containers die soms boven komen) */
    div[data-testid="stElementContainer"]:has(> div[data-testid="stImage"]) {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* Zorg dat de image container 100% breedte pakt */
    .tile-wrap div[data-testid="stImage"]{
        width:100% !important;
        margin:0 !important;
        padding:0 !important;
    }

    /* IMG zelf: exact 100% breedte van tile-wrap (dus gelijk aan knopbreedte) */
    .tile-wrap div[data-testid="stImage"] img{
        width:100% !important;
        height: var(--tile-img-h) !important;
        object-fit: fill !important; /* stretchen ok */
        border-radius: var(--tile-radius);
        display:block;
    }

    /* Knoppen compacter */
    div.stButton > button{
        border-radius: 12px !important;
        padding-top: .35rem !important;
        padding-bottom: .35rem !important;
        margin-top: 10px !important;
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

cols = st.columns(4, gap="small")

for i, t in enumerate(TILES):
    with cols[i]:
        st.markdown("<div class='tile-wrap'>", unsafe_allow_html=True)
        st.image(t["img"], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if t["enabled"]:
            if st.button(t["button"], key=f"btn_{t['key']}", use_container_width=True):
                st.switch_page(t["page"])
        else:
            st.button(t["button"], key=f"btn_{t['key']}", use_container_width=True, disabled=True)
