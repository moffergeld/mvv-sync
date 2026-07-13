from __future__ import annotations

import streamlit as st

from roles import get_profile, get_sb, is_staff_user, render_sidebar_footer, render_sidebar_navigation, require_auth
from pages.Subscripts.mvv_branding import TEAM_HERO_BG, TEAM_LOGO, build_data_uri
from utils.streamlit_ui import apply_streamlit_chrome


st.set_page_config(page_title="Reports", layout="wide", initial_sidebar_state="expanded")
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
          background: __REPORTS_BG__;
          background-size: cover;
          background-position: center top;
          background-attachment: fixed;
        }

        .block-container {
          max-width: 1380px;
          padding-top: 1.4rem;
          padding-bottom: 2.4rem;
        }

        .reports-hero {
          border-radius: 10px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(135deg, rgba(18, 25, 42, 0.88), rgba(10, 15, 27, 0.84));
          padding: 1.85rem 1.6rem;
          box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22);
          margin-bottom: 1.2rem;
        }

        .reports-logo {
          width: 78px;
          height: 78px;
          object-fit: contain;
          margin-bottom: 0;
          flex-shrink: 0;
          filter: drop-shadow(0 8px 22px rgba(0,0,0,0.28));
        }

        .reports-head {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 1rem;
          margin-bottom: 1rem;
        }

        .reports-copyhead {
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 0.12rem;
          text-align: left;
        }

        .reports-kicker {
          color: rgba(255,255,255,0.76);
          font-size: 0.74rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.18em;
          margin-bottom: 0;
        }

        .reports-title {
          margin: 0;
          font-size: 2.45rem;
          line-height: 1;
          font-weight: 800;
          color: #ffffff;
        }

        .reports-copy {
          margin-top: 0.8rem;
          max-width: 72ch;
          color: rgba(255,255,255,0.84);
          line-height: 1.6;
        }

        .reports-pill-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
          margin-top: 1rem;
        }

        .reports-pill {
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

        .reports-section-head {
          display: flex;
          justify-content: space-between;
          align-items: flex-end;
          gap: 1rem;
          margin: 1rem 0 0.9rem 0;
        }

        .reports-section-kicker {
          color: rgba(255,255,255,0.62);
          font-size: 0.75rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }

        .reports-section-title {
          margin-top: 0.25rem;
          color: #ffffff;
          font-size: 1.1rem;
          font-weight: 700;
        }

        .reports-section-note {
          color: rgba(255,255,255,0.78);
          font-size: 0.88rem;
          font-weight: 700;
          text-align: right;
        }

        .reports-card {
          border-radius: 10px;
          border: 1px solid rgba(234, 51, 81, 0.14);
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
          box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
          padding: 1.15rem 1.05rem 1rem 1.05rem;
          min-height: 250px;
        }

        .reports-card-kicker {
          color: rgba(255,255,255,0.62);
          font-size: 0.75rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }

        .reports-card-title {
          margin-top: 0.55rem;
          color: #ffffff;
          font-size: 1.55rem;
          line-height: 1.05;
          font-weight: 800;
        }

        .reports-card-copy {
          margin-top: 0.7rem;
          color: rgba(255,255,255,0.82);
          line-height: 1.55;
          min-height: 3.2rem;
        }

        .reports-card-meta {
          margin-top: 0.9rem;
          color: rgba(255,255,255,0.68);
          font-size: 0.84rem;
          line-height: 1.45;
        }

        .reports-button-row {
          margin-top: 0.95rem;
        }

        @media (max-width: 768px) {
          .reports-title {
            font-size: 2rem;
          }

          .reports-head {
            flex-direction: column;
            gap: 0.8rem;
          }

          .reports-copyhead {
            text-align: center;
          }

          .reports-section-head {
            flex-direction: column;
            align-items: flex-start;
          }

          .reports-section-note {
            text-align: left;
          }
        }
        </style>
        """.replace("__REPORTS_BG__", background),
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
        <div class="reports-card">
          <div class="reports-card-kicker">{kicker}</div>
          <div class="reports-card-title">{title}</div>
          <div class="reports-card-copy">{copy}</div>
          <div class="reports-card-meta">{meta}</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="reports-button-row">', unsafe_allow_html=True)
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
        f'<img src="{TEAM_LOGO_URI}" alt="MVV Maastricht" class="reports-logo" />'
        if TEAM_LOGO_URI
        else ""
    )
    st.markdown(
        f"""
        <div class="reports-hero">
          <div class="reports-head">
            {logo_markup}
            <div class="reports-copyhead">
              <h1 class="reports-title">Reports</h1>
              <div class="reports-kicker">MVV Maastricht | Reports | Staff</div>
            </div>
          </div>
          <div class="reports-copy">
            Centrale rapportage-ingang voor wedstrijdanalyses, weekoverzichten en staffevaluaties.
          </div>
          <div class="reports-pill-row">
            <span class="reports-pill">Match Reports opent als eigen rapportage-route</span>
            <span class="reports-pill">Week Report volgt de team weekstructuur uit de rapportagemap</span>
          </div>
        </div>
        <div class="reports-section-head">
          <div>
            <div class="reports-section-kicker">Modules</div>
            <div class="reports-section-title">Kies welke rapportage je wilt openen</div>
          </div>
          <div class="reports-section-note">2 modules direct beschikbaar</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    row = st.columns(2, gap="large")
    with row[0]:
        render_tile(
            key="reports_hub_match_reports",
            kicker="Wedstrijd",
            title="Match Reports",
            copy="Open de wedstrijdrapportage met filters, score-overzicht, fasekeuze en per-speler GPS-analyse.",
            meta="Route voor matchselectie, KPI's en teamrapportage",
            button_label="Open Match Reports",
            target_page="pages/02_Match_Reports.py",
        )
    with row[1]:
        render_tile(
            key="reports_hub_week_report",
            kicker="Week",
            title="Week Report",
            copy="Open de webversie van de team weekrapportage met KPI-cards, dagbelasting, spreiding en leaders.",
            meta="Route voor teamweekanalyse, training vs match en weeknotities",
            button_label="Open Week Report",
            target_page="pages/14_Week_Report.py",
        )

    render_sidebar_footer(profile)


if __name__ == "__main__":
    main()
