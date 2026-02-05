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
    if st.button("Sign in", width='stretch'):
        res = sb.auth.sign_in_with_password({"email": email, "password": password})
        st.session_state["access_token"] = res.session.access_token
        st.session_state["user_email"] = email
        st.rerun()


def logout_button():
    if st.button("Logout"):
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

# Topbar
st.sidebar.success(f"Ingelogd: {st.session_state.get('user_email','')}")
logout_button()

st.title("MVV Dashboard")
st.write("Gebruik het menu links om een module te openen.")
st.write("• Forms (Wellness/RPE)")
