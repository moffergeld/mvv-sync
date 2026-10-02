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

        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"] {
          display: none !important;
        }

        #MainMenu,
        footer {
          visibility: hidden !important;
        }

        div[data-testid="stHeaderActionElements"] {
          display: flex !important;
          visibility: visible !important;
          opacity: 1 !important;
        }

        [data-testid="collapsedControl"] {
          display: flex !important;
          visibility: visible !important;
          opacity: 1 !important;
          z-index: 1002 !important;
          position: fixed !important;
          top: 0.65rem !important;
          left: 0.65rem !important;
        }

        [data-testid="collapsedControl"] button {
          min-width: 42px !important;
          min-height: 42px !important;
          border-radius: 12px !important;
          background: rgba(11, 16, 32, 0.94) !important;
          border: 1px solid rgba(234, 51, 81, 0.24) !important;
          box-shadow: 0 10px 24px rgba(0, 0, 0, 0.28) !important;
        }

        [data-testid="collapsedControl"] svg {
          fill: #ffffff !important;
        }

        div[data-testid="stToolbar"] {
          display: flex !important;
          visibility: visible !important;
          opacity: 1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_dashboard_polish() -> None:
    """Shared visual finish for the staff performance dashboards."""

    st.markdown(
        """
        <style>
        :root {
          --mvv-surface: rgba(14, 21, 36, 0.92);
          --mvv-surface-soft: rgba(18, 27, 46, 0.78);
          --mvv-line: rgba(255,255,255,0.085);
          --mvv-line-strong: rgba(255,255,255,0.14);
          --mvv-accent: #e52b49;
          --mvv-muted: rgba(242,246,252,0.66);
        }

        .block-container {
          max-width: 1320px !important;
          padding-left: 1.6rem !important;
          padding-right: 1.6rem !important;
          padding-bottom: 3rem !important;
        }

        /* Page heroes are compact control bars, not oversized banners. */
        div[data-testid="stVerticalBlock"]:has(.week-report-hero-anchor),
        div[data-testid="stVerticalBlock"]:has(.session-load-beta-hero-anchor),
        div[data-testid="stVerticalBlock"]:has(.mr-page-hero-anchor),
        div[data-testid="stVerticalBlock"]:has(.year-report-hero-anchor),
        div[data-testid="stVerticalBlock"]:has(.month-report-hero-anchor),
        div[data-testid="stVerticalBlock"]:has(.player-report-hero-anchor) {
          min-height: 0 !important;
          padding: .85rem 1rem !important;
          margin-bottom: .8rem !important;
          border-radius: 14px !important;
          border: 1px solid var(--mvv-line) !important;
          background: rgba(13,20,34,.9) !important;
          box-shadow: 0 10px 28px rgba(0,0,0,.14) !important;
        }

        .week-report-head,
        .session-load-beta-head,
        .mr-page-header-row,
        .year-report-head,
        .month-report-head,
        .player-report-head {
          justify-content: flex-start !important;
          gap: .7rem !important;
          margin-bottom: .55rem !important;
        }

        .week-report-logo,
        .session-load-beta-logo,
        .mr-page-logo,
        .year-report-logo,
        .month-report-logo,
        .player-report-logo {
          width: 44px !important;
          height: 44px !important;
        }

        .week-report-title,
        .session-load-beta-title,
        .mr-page-title,
        .year-report-title,
        .month-report-title,
        .player-report-title {
          font-size: 1.65rem !important;
          line-height: 1.05 !important;
          letter-spacing: -.025em;
        }

        .week-report-copy,
        .session-load-beta-copy {
          display: none !important;
        }

        .week-report-filter-note,
        .session-load-beta-filter-note,
        .mr-hero-filter-note,
        .year-report-filter-note,
        .month-report-filter-note,
        .player-report-filter-note {
          margin-top: .75rem !important;
          font-size: .78rem !important;
        }

        .reports-hero,
        .bench-hero,
        .mvv-hero-shell {
          padding: .85rem 1rem !important;
          margin-bottom: .8rem !important;
          border-radius: 14px !important;
          border: 1px solid var(--mvv-line) !important;
          background: rgba(13,20,34,.9) !important;
          box-shadow: 0 10px 28px rgba(0,0,0,.14) !important;
        }

        .mvv-hero-shell .mvv-hero {
          min-height: 0 !important;
          padding: 0 !important;
          border: 0 !important;
          background: transparent !important;
          box-shadow: none !important;
        }

        .reports-head,
        .bench-head,
        .mvv-page-head {
          justify-content: flex-start !important;
          gap: .7rem !important;
          margin-bottom: 0 !important;
        }

        .reports-logo,
        .bench-logo,
        .mvv-page-logo {
          width: 44px !important;
          height: 44px !important;
        }

        .reports-title,
        .bench-title,
        .mvv-page-title {
          font-size: 1.65rem !important;
          line-height: 1.05 !important;
          letter-spacing: -.025em;
        }

        .reports-copy,
        .bench-copy,
        .mvv-page-copy,
        .reports-pill-row,
        .bench-pill-row,
        .mvv-pill-row {
          display: none !important;
        }

        /* One calm surface level: no card-in-card framing around charts. */
        div[data-testid="stVerticalBlock"]:has(.week-report-panel-anchor) {
          padding: 1.15rem 1.1rem 0.55rem !important;
          border-radius: 16px !important;
          border: 1px solid var(--mvv-line) !important;
          background: var(--mvv-surface) !important;
          box-shadow: 0 14px 34px rgba(0,0,0,.18) !important;
        }

        div[data-testid="stPlotlyChart"] {
          border: 0 !important;
          border-radius: 12px !important;
          overflow: hidden;
        }

        .week-report-card-grid,
        .mr-summary-grid {
          gap: .72rem !important;
        }

        .week-report-card,
        .mr-summary-card,
        .mr-kpi-card,
        .mr-panel {
          min-height: 106px !important;
          border-radius: 14px !important;
          border: 1px solid var(--mvv-line) !important;
          border-top-color: rgba(229,43,73,.34) !important;
          background: linear-gradient(145deg, rgba(20,30,50,.94), rgba(12,18,31,.96)) !important;
          box-shadow: 0 10px 28px rgba(0,0,0,.15) !important;
          padding: .9rem 1rem .8rem !important;
        }

        .week-report-card-value,
        .mr-summary-value {
          margin-top: .4rem !important;
          font-size: 1.72rem !important;
          letter-spacing: -.025em;
        }

        .week-report-card-foot,
        .mr-summary-foot {
          margin-top: .48rem !important;
          font-size: .78rem !important;
          line-height: 1.35 !important;
          color: var(--mvv-muted) !important;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
          gap: .4rem !important;
          border-bottom: 1px solid var(--mvv-line) !important;
          padding-bottom: .55rem;
          margin-bottom: .8rem;
        }

        div[data-testid="stTabs"] button {
          min-height: 38px !important;
          padding: .4rem .82rem !important;
          border-radius: 9px !important;
          border: 1px solid transparent !important;
          background: transparent !important;
          color: rgba(255,255,255,.68) !important;
          box-shadow: none !important;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
          color: #fff !important;
          background: rgba(229,43,73,.13) !important;
          border-color: rgba(229,43,73,.3) !important;
        }

        div[data-baseweb="select"] > div,
        [data-testid="stDateInput"] input,
        [data-testid="stTextInput"] input {
          min-height: 44px !important;
          border-radius: 10px !important;
          border-color: var(--mvv-line-strong) !important;
          background: rgba(11,17,30,.88) !important;
          box-shadow: none !important;
        }

        [data-testid="stExpander"] {
          border: 1px solid var(--mvv-line) !important;
          border-radius: 12px !important;
          background: rgba(13,20,34,.72) !important;
        }

        .week-report-table tbody tr:hover td {
          background: rgba(255,255,255,.025);
        }

        @media (max-width: 768px) {
          .block-container {
            padding-left: .8rem !important;
            padding-right: .8rem !important;
          }

          .week-report-head,
          .session-load-beta-head,
          .mr-page-header-row,
          .year-report-head,
          .month-report-head,
          .player-report-head {
            flex-direction: row !important;
            text-align: left !important;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
