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
    """Robuust pad voor lokaal + Streamlit Cloud."""
    return str((BASE_DIR / rel_path).resolve())


# =========================
# Tiles (afbeeldingen uit GitHub)
# =========================
TILES = [
    {
        "label": "Player Page",
        "img": asset("Assets/Afbeeldingen/Script/Player_page.PNG"),
        "page_key": "Player Page",
        "enabled": True,
    },
    {
        "label": "GPS Data",
        "img": asset("Assets/Afbeeldingen/Script/GPS_Data.PNG"),
        "page_key": "GPS Data",
        "enabled": True,
    },
    {
        "label": "GPS Import",
        "img": asset("Assets/Afbeeldingen/Script/GPS_Import.PNG"),
        "page_key": "GPS Import",
        "enabled": True,
    },
    {
        "label": "Medical",
        "img": asset("Assets/Afbeeldingen/Script/Medical.PNG"),
        "page_key": None,  # nog geen pagina
        "enabled": False,
        "badge": "Coming soon",
    },
]


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

            # Belangrijk voor PostgREST/RLS calls in dezelfde sessie:
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

# Zorg dat de client ook na refresh / rerun geautoriseerd blijft
try:
    sb.postgrest.auth(st.session_state["access_token"])
except Exception:
    pass


# =========================
# Sidebar
# =========================
st.sidebar.success(f"Ingelogd: {st.session_state.get('user_email','')}")
logout_button()

# (optioneel) oude menu blijft bestaan
MENU = ["Home", "Player Page", "GPS Data", "GPS Import"]
current = st.session_state.get("page", "Home")
choice = st.sidebar.radio("Menu", MENU, index=MENU.index(current) if current in MENU else 0)
st.session_state["page"] = choice


# =========================
# Home met tegels
# =========================
def render_home():
    st.title("MVV Dashboard")
    st.write("Gebruik het menu links of klik op een tegel om een module te openen.")

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
                    st.session_state["page"] = t["page_key"]
                    st.rerun()
            else:
                st.button("Open", key=f"open_{t['label']}", use_container_width=True, disabled=True)

            st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Router (koppel later aan jouw echte pagina-modules)
# =========================
page = st.session_state.get("page", "Home")

if page == "Home":
    render_home()

elif page == "Player Page":
    st.title("Player Page")
    st.info("Koppel hier jouw Player Page module/render functie.")

elif page == "GPS Data":
    st.title("GPS Data")
    st.info("Koppel hier jouw GPS Data module/render functie.")

elif page == "GPS Import":
    st.title("GPS Import")
    st.info("Koppel hier jouw GPS Import module/render functie.")

else:
    render_home()
