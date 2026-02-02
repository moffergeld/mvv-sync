import pandas as pd
import requests
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="MVV Dashboard", layout="wide")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def rest_get(table: str, access_token: str, query: str) -> pd.DataFrame:
    url = f"{SUPABASE_URL}/rest/v1/{table}?{query}"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token}",
    }
    r = requests.get(url, headers=headers, timeout=60)

    if not r.ok:
        st.error(f"REST error for table '{table}'")
        st.code(f"URL: {url}")
        st.code(f"Status: {r.status_code}\nBody: {r.text}")
        return pd.DataFrame()

    data = r.json()
    return pd.DataFrame(data)


def get_profile(access_token: str) -> dict:
    # profiles is protected by RLS (self only), so this returns exactly 1 row for the logged-in user
    df = rest_get("profiles", access_token, "select=role,team,player_id")
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def login_ui():
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    cols = st.columns([1, 1, 6])
    with cols[0]:
        if st.button("Sign in", use_container_width=True):
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


if "access_token" not in st.session_state:
    login_ui()
    st.stop()

access_token = st.session_state["access_token"]
profile = get_profile(access_token)

st.title("MVV – Wellness & RPE")
st.caption(f"Ingelogd als: {st.session_state.get('user_email','')}")

c1, c2, c3, c4 = st.columns([1.2, 1.2, 2, 1])
with c1:
    st.metric("Role", profile.get("role", "unknown"))
with c2:
    st.metric("Team", profile.get("team", "") or "-")
with c3:
    st.write("")
with c4:
    logout_button()

tab1, tab2, tab3 = st.tabs(["Wellness", "RPE", "Players"])

with tab1:
    dfw = rest_get("wellness", access_token, "select=*&order=form_date.desc&limit=2000")
    st.subheader("Wellness")
    st.dataframe(dfw, use_container_width=True)

    if not dfw.empty and "form_date" in dfw.columns:
        dfw["form_date"] = pd.to_datetime(dfw["form_date"])
        metrics = [c for c in ["muscle_soreness", "fatigue", "sleep_quality", "stress", "mood"] if c in dfw.columns]
        if metrics:
            daily = dfw.groupby(dfw["form_date"].dt.date)[metrics].mean()
            st.line_chart(daily)

with tab2:
    dfr = rest_get("rpe", access_token, "select=*&order=form_date.desc&limit=2000")
    st.subheader("RPE")
    st.dataframe(dfr, use_container_width=True)

    if not dfr.empty:
        for c in ["ex1_duration_min", "ex1_exertion", "ex2_duration_min", "ex2_exertion"]:
            if c in dfr.columns:
                dfr[c] = pd.to_numeric(dfr[c], errors="coerce")

        dfr["srpe_ex1"] = dfr.get("ex1_duration_min") * dfr.get("ex1_exertion")
        dfr["srpe_ex2"] = dfr.get("ex2_duration_min") * dfr.get("ex2_exertion")
        dfr["srpe_total"] = dfr["srpe_ex1"].fillna(0) + dfr["srpe_ex2"].fillna(0)

        st.subheader("sRPE total (top 50)")
        show_cols = [c for c in ["form_date", "player_id", "srpe_total", "injury", "pain_scale_injury"] if c in dfr.columns]
        st.dataframe(dfr[show_cols].head(50), use_container_width=True)

with tab3:
    dfp = rest_get("players", access_token, "select=*&limit=200")
    st.subheader("Players")
    st.dataframe(dfp, use_container_width=True)
