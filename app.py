import streamlit as st
from supabase import create_client

st.set_page_config(page_title="MVV Dashboard", layout="wide")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


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


# Auth gate
if "access_token" not in st.session_state:
    login_ui()
    st.stop()

# Zorg dat de client ook na refresh/ rerun geautoriseerd blijft
try:
    sb.postgrest.auth(st.session_state["access_token"])
except Exception:
    pass

# Sidebar navigation (zet hier de NAMES zoals jij ze wil)
pg = st.navigation(
    [
        st.Page("pages/01_player_pages.py", title="Player pages", icon="👤"),
        st.Page("pages/02_GPS_Data.py", title="GPS Data", icon="📈"),
        st.Page("pages/06_GPS_Import.py", title="GPS Import", icon="⬆️"),
    ],
    position="sidebar",
)

pg.run()


# Sidebar status + logout
st.sidebar.success(f"Ingelogd: {st.session_state.get('user_email','')}")
logout_button()

# Run de gekozen pagina
pg.run()
