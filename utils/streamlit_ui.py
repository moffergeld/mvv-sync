from __future__ import annotations

import streamlit as st


def apply_streamlit_chrome() -> None:
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] {
          background: transparent !important;
          border-bottom: none !important;
          box-shadow: none !important;
          backdrop-filter: none !important;
        }

        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"],
        div[data-testid="stHeaderActionElements"] {
          display: none !important;
        }

        #MainMenu,
        footer {
          visibility: hidden !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
