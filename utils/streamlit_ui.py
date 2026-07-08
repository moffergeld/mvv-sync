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
          min-height: 3.2rem !important;
        }

        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"] {
          display: none !important;
        }

        #MainMenu,
        footer {
          visibility: hidden !important;
        }

        [data-testid="collapsedControl"] {
          display: flex !important;
          visibility: visible !important;
          opacity: 1 !important;
          z-index: 1002 !important;
        }

        [data-testid="collapsedControl"] button {
          min-width: 44px !important;
          min-height: 44px !important;
          border-radius: 12px !important;
          background: rgba(11, 16, 32, 0.94) !important;
          border: 1px solid rgba(234, 51, 81, 0.28) !important;
          box-shadow: 0 10px 24px rgba(0, 0, 0, 0.28) !important;
        }

        [data-testid="collapsedControl"] svg {
          fill: #ffffff !important;
        }

        div[data-testid="stHeaderActionElements"] {
          display: flex !important;
          visibility: visible !important;
          opacity: 1 !important;
          z-index: 1001 !important;
        }

        @media (max-width: 1024px) {
          [data-testid="collapsedControl"] {
            position: fixed !important;
            top: 0.7rem !important;
            left: 0.7rem !important;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
