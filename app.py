import streamlit as st
from auth import get_sb, login_gate, sidebar_logout

st.set_page_config(page_title="MVV Dashboard", layout="wide")

sb = get_sb()

# Auth gate
login_gate(sb)

# Sidebar logout
sidebar_logout(sb)

st.title("MVV Dashboard")
st.write("Gebruik het menu links om een module te openen.")
st.write("• Forms (Wellness/RPE)")
