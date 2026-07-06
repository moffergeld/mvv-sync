from __future__ import annotations

from pathlib import Path

import streamlit as st

from roles import get_profile, get_sb, is_staff_user, render_sidebar_footer, render_sidebar_navigation, require_auth
from pages.Subscripts.mvv_branding import TEAM_HERO_BG, TEAM_LOGO, build_data_uri
from utils.streamlit_ui import apply_streamlit_chrome


st.set_page_config(page_title="Data Page Beta", layout="wide")
apply_streamlit_chrome()

PAGE_BG_URI = build_data_uri(TEAM_HERO_BG)
TEAM_LOGO_URI = build_data_uri(TEAM_LOGO)


def render_css() -> None:
    background = (
        f"linear-gradient(180deg, rgba(6, 10, 20, 0.82) 0%, rgba(6, 10, 20, 0.80) 100%), "
        f"radial-gradient(circle at top left, rgba(200, 16, 46, 0.16), rgba(200, 16, 46, 0.02) 24%, transparent 46%), "
        f"radial-gradient(circle at top right, rgba(234, 51, 81, 0.10), rgba(234, 51, 81, 0.02) 18%, transparent 42%), "
        f"url('{PAGE_BG_URI}')"
        if PAGE_BG_URI
        else "radial-gradient(circle at top left, rgba(200, 16, 46, 0.28), rgba(200, 16, 46, 0.03) 26%, transparent 48%), radial-gradient(circle at top right, rgba(234, 51, 81, 0.18), rgba(234, 51, 81, 0.03) 18%, transparent 44%), linear-gradient(180deg, #070c18 0%, #0a1020 100%)"
    )
    st.markdown(
        """
        <style>
        .stApp {
          background: __DATA_BETA_BG__;
          background-size: cover;
          background-position: center top;
          background-attachment: fixed;
        }

        .block-container {
          max-width: 1380px;
          padding-top: 1.4rem;
          padding-bottom: 2.4rem;
        }

        .data-beta-hero {
          border-radius: 10px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(135deg, rgba(18, 25, 42, 0.88), rgba(10, 15, 27, 0.84));
          padding: 1.85rem 1.6rem;
          box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22);
          margin-bottom: 1.2rem;
        }

        .data-beta-logo {
          width: 78px;
          height: 78px;
          object-fit: contain;
          margin-bottom: 0;
          flex-shrink: 0;
          filter: drop-shadow(0 8px 22px rgba(0,0,0,0.28));
        }

        .data-beta-head {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 1rem;
          margin-bottom: 1rem;
        }

        .data-beta-copyhead {
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 0.12rem;
          text-align: left;
        }

        .data-beta-kicker {
          color: rgba(255,255,255,0.76);
          font-size: 0.74rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.18em;
          margin-bottom: 0;
        }

        .data-beta-title {
          margin: 0;
          font-size: 2.45rem;
          line-height: 1;
          font-weight: 800;
          color: #ffffff;
        }

        .data-beta-copy {
          margin-top: 0.8rem;
          max-width: 76ch;
          color: rgba(255,255,255,0.84);
          line-height: 1.6;
        }

        .data-beta-pill-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
          margin-top: 1rem;
        }

        .data-beta-pill {
          display: inline-flex;
          align-items: center;
          padding: 0.42rem 0.76rem;
          border-radius: 999px;
          font-size: 0.78rem;
          font-weight: 800;
          border: 1px solid rgba(234, 51, 81, 0.22);
          background: rgba(255,255,255,0.06);
          color: rgba(255,255,255,0.92);
        }

        .data-beta-section-head {
          display: flex;
          justify-content: space-between;
          align-items: flex-end;
          gap: 1rem;
          margin: 1rem 0 0.9rem 0;
        }

        .data-beta-section-kicker {
          color: rgba(255,255,255,0.62);
          font-size: 0.75rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }

        .data-beta-section-title {
          margin-top: 0.25rem;
          color: #ffffff;
          font-size: 1.1rem;
          font-weight: 700;
        }

        .data-beta-section-note {
          color: rgba(255,255,255,0.78);
          font-size: 0.88rem;
          font-weight: 700;
          text-align: right;
        }

        .data-beta-card {
          border-radius: 10px;
          border: 1px solid rgba(234, 51, 81, 0.14);
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
          box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
          padding: 1.15rem 1.05rem 1rem 1.05rem;
          min-height: 250px;
        }

        .data-beta-card-kicker {
          color: rgba(255,255,255,0.62);
          font-size: 0.75rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }

        .data-beta-card-title {
          margin-top: 0.55rem;
          color: #ffffff;
          font-size: 1.55rem;
          line-height: 1.05;
          font-weight: 800;
        }

        .data-beta-card-copy {
          margin-top: 0.7rem;
          color: rgba(255,255,255,0.82);
          line-height: 1.55;
          min-height: 3.2rem;
        }

        .data-beta-card-meta {
          margin-top: 0.9rem;
          color: rgba(255,255,255,0.68);
          font-size: 0.84rem;
          line-height: 1.45;
        }

        .data-beta-button-row {
          margin-top: 0.95rem;
        }

        @media (max-width: 768px) {
          .data-beta-title {
            font-size: 2rem;
          }

          .data-beta-head {
            flex-direction: column;
            gap: 0.8rem;
          }

          .data-beta-copyhead {
            text-align: center;
          }

          .data-beta-section-head {
            flex-direction: column;
            align-items: flex-start;
          }

          .data-beta-section-note {
            text-align: left;
          }
        }
        </style>
        """.replace("__DATA_BETA_BG__", background),
        unsafe_allow_html=True,
    )


def _launch(page_path: str) -> None:
    st.switch_page(page_path)


def render_tile(
    *,
    key: str,
    kicker: str,
    title: str,
    copy: str,
    meta: str,
    button_label: str,
    target_page: str,
) -> None:
    st.markdown(
        f"""
        <div class="data-beta-card">
          <div class="data-beta-card-kicker">{kicker}</div>
          <div class="data-beta-card-title">{title}</div>
          <div class="data-beta-card-copy">{copy}</div>
          <div class="data-beta-card-meta">{meta}</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="data-beta-button-row">', unsafe_allow_html=True)
    if st.button(button_label, use_container_width=True, key=key):
        _launch(target_page)
    st.markdown("</div></div>", unsafe_allow_html=True)


def main() -> None:
    render_css()
    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    profile = get_profile(sb)
    if not is_staff_user(profile):
        st.error("Geen toegang: deze pagina is alleen voor staff.")
        st.stop()

    render_sidebar_navigation(profile)

    logo_markup = (
        f'<img src="{TEAM_LOGO_URI}" alt="MVV Maastricht" class="data-beta-logo" />'
        if TEAM_LOGO_URI
        else ""
    )
    st.markdown(
        f"""
        <div class="data-beta-hero">
          <div class="data-beta-head">
            {logo_markup}
            <div class="data-beta-copyhead">
              <h1 class="data-beta-title">Data</h1>
              <div class="data-beta-kicker">MVV Maastricht | Data Beta | Staff</div>
            </div>
          </div>
          <div class="data-beta-copy">
            Beta-opzet voor een centrale data-ingang. Vanuit deze pagina kun je straks sneller kiezen
            tussen Session Load, ACWR, FFP, Wellness/RPE en Compare zonder dat alles los in de hoofdstructuur hoeft te staan.
          </div>
          <div class="data-beta-pill-row">
            <span class="data-beta-pill">Bestaande pagina's blijven actief</span>
            <span class="data-beta-pill">Nieuwe structuur alleen als beta</span>
            <span class="data-beta-pill">ACWR en FFP openen als eigen beta-routes</span>
          </div>
        </div>
        <div class="data-beta-section-head">
          <div>
            <div class="data-beta-section-kicker">Modules</div>
            <div class="data-beta-section-title">Kies welke data-omgeving je wilt openen</div>
          </div>
          <div class="data-beta-section-note">5 beta-routes, huidige navigatie blijft onaangetast</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    row_one = st.columns(3, gap="large")
    with row_one[0]:
        render_tile(
            key="data_beta_session_load",
            kicker="Load",
            title="Session Load",
            copy="Open een compacte GPS beta-route die alleen de Session Load workflow toont, inclusief dag- en sessiefilter.",
            meta="Nieuwe beta-route voor GPS Session Load",
            button_label="Open Session Load Beta",
            target_page="pages/12_GPS_Session_Load_Beta.py",
        )
    with row_one[1]:
        render_tile(
            key="data_beta_acwr",
            kicker="Ratio",
            title="ACWR",
            copy="Open direct de beta-route voor ACWR met scope-keuze en dezelfde onderliggende berekeningen.",
            meta="Beta-route, bestaande GPS ACWR blijft ook bestaan",
            button_label="Open ACWR Beta",
            target_page="pages/11_ACWR_Page_Beta.py",
        )
    with row_one[2]:
        render_tile(
            key="data_beta_ffp",
            kicker="Model",
            title="Fitness-Fatigue-Performance",
            copy="Open direct de beta-route voor het FFP-model met dezelfde onderliggende Summary-data en modelinstellingen.",
            meta="Losse beta-route voor FFP",
            button_label="Open FFP Beta",
            target_page="pages/13_FFP_Page_Beta.py",
        )

    row_two = st.columns(2, gap="large")
    with row_two[0]:
        render_tile(
            key="data_beta_wr",
            kicker="Monitoring",
            title="Wellness / RPE",
            copy="Ga naar de dagelijkse wellness-, week-, injury- en checklistweergaven voor de selectie.",
            meta="Huidige route: Wellness & RPE Overview",
            button_label="Open Wellness / RPE",
            target_page="pages/04_Wellness_&_RPE_Overview.py",
        )
    with row_two[1]:
        render_tile(
            key="data_beta_compare",
            kicker="Analyse",
            title="Compare",
            copy="Vergelijk GPS, Wellness en RPE over meerdere sessies in een losse analyseflow.",
            meta="Huidige route: Compare",
            button_label="Open Compare",
            target_page="pages/05_Compare.py",
        )

    render_sidebar_footer(profile)


if __name__ == "__main__":
    main()
