# pages/04_Wellness_&_RPE_Overview.py
from __future__ import annotations

import streamlit as st

from pages.Subscripts.mvv_branding import TEAM_HERO_BG, TEAM_LOGO, build_data_uri
from roles import require_auth, get_sb, get_profile
from pages.Subscripts.wr_common import fetch_active_players_cached
from pages.Subscripts.wr_tab_day import render_wellness_rpe_tab_day
from pages.Subscripts.wr_tab_week import render_wellness_rpe_tab_week
from pages.Subscripts.wr_tab_checklist import render_wellness_rpe_tab_checklist
from pages.Subscripts.wr_tab_injury import render_wellness_rpe_tab_injury

st.set_page_config(page_title="Wellness & RPE Overview", layout="wide")

PAGE_BG_URI = build_data_uri(TEAM_HERO_BG)
TEAM_LOGO_URI = build_data_uri(TEAM_LOGO)

# CSS styling conform design system
MVV_CSS = """
<style>
:root {
  --mvv-red: #C8102E;
  --mvv-red-light: #E8213F;
  --mvv-red-dark: #8B0A1F;
  --mvv-red-glow: rgba(200,16,46,0.35);
  --mvv-red-subtle: rgba(200,16,46,0.10);
  --glass-border: rgba(255,255,255,0.09);
  --glass-shine: rgba(255,255,255,0.13);
  --metal-grad: linear-gradient(135deg,rgba(255,255,255,0.13) 0%,rgba(255,255,255,0.035) 40%,rgba(255,255,255,0.09) 100%);
  --text-primary: #F0F0F0;
  --text-muted: rgba(240,240,240,0.45);
}

.stApp {
  background:
    radial-gradient(ellipse 80% 60% at 15% -5%, rgba(200,16,46,0.20) 0%, transparent 55%),
    radial-gradient(ellipse 60% 50% at 90% 90%, rgba(200,16,46,0.13) 0%, transparent 50%),
    radial-gradient(ellipse 100% 80% at 50% 50%, #0D0E13 30%, #09090D 100%);
  background-attachment: fixed;
}

.mvv-page-header {
  display: flex;
  align-items: center;
  gap: 18px;
  margin-bottom: 1.8rem;
}

.mvv-page-icon {
  width: 50px;
  height: 50px;
  background: linear-gradient(135deg, var(--mvv-red), var(--mvv-red-dark));
  border-radius: 13px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.8rem;
  box-shadow: 0 4px 18px var(--mvv-red-glow), inset 0 1px 0 var(--glass-shine);
}

.mvv-page-title {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2.4rem;
  letter-spacing: 3px;
  color: var(--text-primary);
  margin: 0;
}

.mvv-page-sub {
  font-family: 'DM Sans', sans-serif;
  font-size: 0.74rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 0.2rem;
}

.mvv-divider {
  height: 1px;
  background: linear-gradient(90deg, var(--mvv-red) 0%, rgba(200,16,46,0.3) 38%, transparent 68%);
  margin: 1.1rem 0 1.8rem;
}

.mvv-section-label {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 0.82rem;
  letter-spacing: 0.22em;
  color: var(--mvv-red-light);
  text-transform: uppercase;
  margin: 1.4rem 0 0.8rem;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.5rem;
  padding: 0.5rem;
  background: rgba(255,255,255,0.03);
  border-radius: 12px;
  border: 1px solid var(--glass-border);
}

.stTabs [data-baseweb="tab"] {
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--glass-border);
  border-radius: 10px;
  color: var(--text-muted);
  padding: 0.5rem 1rem;
  font-family: 'DM Sans', sans-serif;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-size: 0.78rem;
}

.stTabs [data-baseweb="tab"]:hover {
  background: var(--mvv-red-subtle);
  border-color: var(--mvv-red);
  color: white;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, rgba(255,255,255,0.13) 0%, rgba(255,255,255,0.035) 40%, rgba(255,255,255,0.09) 100%);
  border: 1px solid var(--mvv-red);
  color: white;
  box-shadow: 0 0 12px var(--mvv-red-glow), inset 0 1px 0 var(--glass-shine);
}

/* Glass card styling */
.mvv-card {
  background: var(--metal-grad), rgba(255,255,255,0.035);
  border: 1px solid var(--glass-border);
  border-radius: 16px;
  padding: 1.4rem 1.6rem;
  box-shadow: 0 8px 28px rgba(0,0,0,0.38), inset 0 1px 0 var(--glass-shine);
  margin-bottom: 1.5rem;
}
</style>
"""

_WELLNESS_BACKGROUND = (
    f"linear-gradient(180deg, rgba(6, 10, 20, 0.82) 0%, rgba(6, 10, 20, 0.80) 100%), "
    f"radial-gradient(circle at top left, rgba(200, 16, 46, 0.16), rgba(200, 16, 46, 0.02) 24%, transparent 46%), "
    f"radial-gradient(circle at top right, rgba(234, 51, 81, 0.10), rgba(234, 51, 81, 0.02) 18%, transparent 42%), "
    f"url('{PAGE_BG_URI}')"
    if PAGE_BG_URI
    else "radial-gradient(circle at top left, rgba(200, 16, 46, 0.28), rgba(200, 16, 46, 0.03) 26%, transparent 48%), radial-gradient(circle at top right, rgba(234, 51, 81, 0.18), rgba(234, 51, 81, 0.03) 18%, transparent 44%), linear-gradient(180deg, #070c18 0%, #0a1020 100%)"
)

MVV_CSS += """
<style>
.stApp {
  background: __WELLNESS_BACKGROUND__ !important;
  background-size: cover !important;
  background-position: center top !important;
  background-attachment: fixed !important;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(16, 23, 38, 0.98), rgba(9, 13, 23, 0.98)) !important;
  border-right: 1px solid rgba(255,255,255,0.06) !important;
}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
  color: #F8FAFC !important;
}

.block-container {
  padding-top: 1.2rem;
  padding-bottom: 2.2rem;
  max-width: 1380px;
}

.mvv-hero-shell {
  display: flex;
  flex-direction: column;
  gap: 1.1rem;
  margin-bottom: 1.55rem;
}

.mvv-hero {
  min-height: 320px;
  padding: 2rem 1.75rem 1.9rem 1.75rem;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.08);
  background: linear-gradient(135deg, rgba(18, 25, 42, 0.88), rgba(10, 15, 27, 0.84));
  box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22);
}

.mvv-page-logo {
  width: 82px;
  height: 82px;
  object-fit: contain;
  margin-bottom: 0.9rem;
  filter: drop-shadow(0 8px 22px rgba(0,0,0,0.28));
}

.mvv-page-kicker {
  color: rgba(255,255,255,0.76);
  font-size: 0.74rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  margin-bottom: 0.35rem;
}

.mvv-page-title {
  font-family: inherit;
  font-size: 2.55rem;
  line-height: 1;
  letter-spacing: 0;
  color: #FFFFFF;
}

.mvv-page-copy {
  margin-top: 0.8rem;
  max-width: 74ch;
  color: rgba(255,255,255,0.84);
  line-height: 1.62;
}

.mvv-pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
  margin-top: 1rem;
}

.mvv-pill {
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

.mvv-summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1rem;
}

.mvv-summary-card {
  min-height: 122px;
  padding: 1rem 1.05rem 0.95rem 1.05rem;
  border-radius: 8px;
  border: 1px solid rgba(234, 51, 81, 0.14);
  background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
}

.mvv-summary-label {
  color: rgba(255,255,255,0.68);
  font-size: 0.8rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.mvv-summary-value {
  margin-top: 0.55rem;
  font-size: 2rem;
  line-height: 1.05;
  font-weight: 800;
  color: #FFFFFF;
}

.mvv-summary-foot {
  margin-top: 0.65rem;
  color: rgba(255,255,255,0.8);
  font-size: 0.86rem;
  line-height: 1.4;
}

.mvv-section-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 1rem;
  margin: 0.15rem 0 0.95rem 0;
}

.mvv-section-label {
  font-family: inherit;
  color: rgba(255,255,255,0.62);
  font-size: 0.75rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin: 0;
}

.mvv-section-title {
  margin-top: 0.25rem;
  color: #FFFFFF;
  font-size: 1.08rem;
  font-weight: 700;
}

.mvv-section-note {
  color: rgba(255,255,255,0.8);
  font-size: 0.88rem;
  font-weight: 700;
  text-align: right;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0.45rem;
  width: fit-content;
  padding: 0.28rem;
  margin-bottom: 1.2rem;
  border-radius: 999px;
  border: 1px solid rgba(234, 51, 81, 0.18);
  background: rgba(11, 16, 29, 0.86);
  box-shadow: 0 10px 22px rgba(0, 0, 0, 0.16);
}

.stTabs [data-baseweb="tab"] {
  border-radius: 999px;
  padding: 0.55rem 1rem;
  background: transparent;
}

.stTabs [data-baseweb="tab"] p {
  font-size: 0.95rem;
  font-weight: 800;
}

@media (max-width: 1100px) {
  .mvv-summary-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 768px) {
  .mvv-hero {
    min-height: auto;
    padding: 1.55rem 1rem;
  }

  .mvv-page-title {
    font-size: 2rem;
  }

  .mvv-summary-grid {
    grid-template-columns: 1fr;
  }

  .mvv-section-head {
    flex-direction: column;
    align-items: flex-start;
  }

  .mvv-section-note {
    text-align: left;
  }
}
</style>
""".replace("__WELLNESS_BACKGROUND__", _WELLNESS_BACKGROUND)

def render_staff_wellness_rpe_page():
    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    profile = get_profile(sb) or {}
    role = str(profile.get("role") or "").lower()

    # Staff-only
    if role == "player":
        st.error("Geen toegang: deze pagina is alleen voor staff.")
        st.stop()

    # Inject CSS
    st.markdown(MVV_CSS, unsafe_allow_html=True)

    # cache key per project/env
    sb_url_key = str(st.secrets.get("SUPABASE_URL", "sb"))

    players = fetch_active_players_cached(sb_url_key, sb)
    if players.empty:
        st.warning("Geen actieve spelers gevonden.")
        st.stop()

    pid_to_name = dict(zip(players["player_id"], players["full_name"]))

    logo_markup = (
        f'<img src="{TEAM_LOGO_URI}" alt="MVV Maastricht" class="mvv-page-logo" />'
        if TEAM_LOGO_URI
        else ""
    )

    st.markdown(
        f"""
        <div class="mvv-hero-shell">
          <div class="mvv-hero">
            {logo_markup}
            <div class="mvv-page-kicker">MVV Maastricht | Wellness & RPE | Staff</div>
            <div class="mvv-page-title">Wellness & RPE Overview</div>
            <div class="mvv-page-copy">
              Teamoverzicht voor dagelijkse wellness-invoer, RPE-opvolging en medische signalen.
              Gebruik de tabs om snel tussen de dag-, week-, injury- en checklistweergave te schakelen.
            </div>
            <div class="mvv-pill-row">
              <span class="mvv-pill">Dagelijkse wellness en RPE-monitoring</span>
              <span class="mvv-pill">Teamoverzicht voor stafbeslissingen</span>
            </div>
          </div>
        </div>
        <div class="mvv-section-head">
          <div>
            <div class="mvv-section-label">Overzicht</div>
            <div class="mvv-section-title">Kies de juiste laag voor je wellness- en RPE-analyse</div>
          </div>
          <div class="mvv-section-note">{len(players)} spelers actief in deze omgeving</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs with custom styling
    tab_labels = ["Dag", "Week", "Injury", "Checklist"]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_wellness_rpe_tab_day(sb, sb_url_key, pid_to_name)

    with tabs[1]:
        render_wellness_rpe_tab_week(sb, sb_url_key, pid_to_name)

    with tabs[2]:
        render_wellness_rpe_tab_injury(sb, sb_url_key, pid_to_name)

    with tabs[3]:
        render_wellness_rpe_tab_checklist(sb, sb_url_key, pid_to_name)


if __name__ == "__main__":
    render_staff_wellness_rpe_page()
