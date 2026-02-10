import streamlit as st
from supabase import create_client

def get_sb():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, key)

def login_gate(sb):
    """Toont login en stopt de app als er geen sessie is."""
    if "access_token" in st.session_state:
        return

    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Sign in", use_container_width=True):
        try:
            res = sb.auth.sign_in_with_password({"email": email, "password": password})
            st.session_state["access_token"] = res.session.access_token
            st.session_state["refresh_token"] = res.session.refresh_token
            st.session_state["user_email"] = email
            st.rerun()
        except Exception:
            st.error("Inloggen mislukt. Controleer email/wachtwoord.")

    st.stop()

def sidebar_logout(sb):
    st.sidebar.success(f"Ingelogd: {st.session_state.get('user_email','')}")
    if st.sidebar.button("Logout", use_container_width=True):
        try:
            sb.auth.sign_out()
        except Exception:
            pass
        st.session_state.clear()
        st.rerun()
