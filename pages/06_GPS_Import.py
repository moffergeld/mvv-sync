# pages/06_GPS_Import.py
# ============================================================
# MAIN PAGE: GPS Import (router) â€” Redesign MVV
# - Tab: Import GPS -> subtabs: Import (Excel), Manual add, Export
# - Tab: Matches
# NOTE: profiles.team column removed -> code must NOT select it.
# ============================================================
from __future__ import annotations
import streamlit as st
from pages.Subscripts.mvv_branding import TEAM_HERO_BG, TEAM_LOGO, build_data_uri
from pages.Subscripts.gps_import_common import (
    ALLOWED_IMPORT,
    get_access_token,
    get_players_map,
    get_profile_role,
)
from pages.Subscripts.gps_import_tab_excel import tab_import_excel_main
from pages.Subscripts.gps_import_tab_export import tab_export_main
from pages.Subscripts.gps_import_tab_manual import tab_manual_add_main
from pages.Subscripts.gps_import_tab_matches import tab_matches_main

st.set_page_config(page_title="GPS Import | MVV Dashboard", layout="wide")

PAGE_BG_URI = build_data_uri(TEAM_HERO_BG)
TEAM_LOGO_URI = build_data_uri(TEAM_LOGO)


# ============================================================
# GLOBAL CSS â€” MVV brand (zelfde taal als app.py)
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
  --mvv-red:        #C8102E;
  --mvv-red-light:  #E8213F;
  --mvv-red-dark:   #8B0A1F;
  --mvv-red-glow:   rgba(200,16,46,0.35);
  --mvv-red-subtle: rgba(200,16,46,0.10);
  --glass-border:   rgba(255,255,255,0.09);
  --glass-shine:    rgba(255,255,255,0.13);
  --metal-grad:     linear-gradient(135deg,rgba(255,255,255,0.13) 0%,rgba(255,255,255,0.035) 40%,rgba(255,255,255,0.09) 100%);
  --text-primary:   #F0F0F0;
  --text-muted:     rgba(240,240,240,0.45);
}

/* â”€â”€ ACHTERGROND â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.stApp {
  background:
    radial-gradient(ellipse 80% 60% at 15% -5%,  rgba(200,16,46,0.20) 0%, transparent 55%),
    radial-gradient(ellipse 60% 50% at 90% 90%,  rgba(200,16,46,0.13) 0%, transparent 50%),
    radial-gradient(ellipse 100% 80% at 50% 50%, #0D0E13 30%, #09090D 100%);
  background-attachment: fixed;
  font-family: 'DM Sans', sans-serif;
  color: var(--text-primary);
}
.block-container {
  padding: 2rem 2.5rem 3rem !important;
  max-width: 1280px !important;
}

/* â”€â”€ SIDEBAR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,rgba(200,16,46,0.13) 0%,rgba(9,9,13,0.98) 28%) !important;
  border-right: 1px solid var(--glass-border) !important;
}

/* â”€â”€ PAGE HEADER ELEMENTEN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.mvv-page-header {
  display: flex; align-items: center; gap: 18px; margin-bottom: 0;
}
.mvv-page-icon {
  width: 50px; height: 50px; flex-shrink: 0;
  background: linear-gradient(135deg, var(--mvv-red), var(--mvv-red-dark));
  border-radius: 13px;
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
  box-shadow: 0 4px 18px var(--mvv-red-glow), inset 0 1px 0 rgba(255,255,255,0.18);
}
.mvv-page-title {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2.4rem; letter-spacing: 3px;
  color: var(--text-primary); line-height: 1; margin: 0;
}
.mvv-page-sub {
  font-size: 0.74rem; font-weight: 500; color: var(--text-muted);
  letter-spacing: 0.10em; text-transform: uppercase; margin-top: 3px;
}
.mvv-divider {
  height: 1px;
  background: linear-gradient(90deg, var(--mvv-red) 0%, rgba(200,16,46,0.3) 38%, transparent 68%);
  margin: 1.1rem 0 1.6rem;
}

/* â”€â”€ NAV PILLS (st.radio horizontal) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
/* Verberg standaard radio dots en style als pill-tabs */
div[data-testid="stRadio"] > div {
  display: flex !important;
  flex-direction: row !important;
  gap: 8px !important;
  flex-wrap: wrap;
}
div[data-testid="stRadio"] label {
  display: flex !important;
  align-items: center !important;
  padding: 7px 20px !important;
  border-radius: 9px !important;
  border: 1px solid var(--glass-border) !important;
  background: rgba(255,255,255,0.04) !important;
  cursor: pointer !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
  transition: background .2s, border-color .2s, color .2s !important;
}
div[data-testid="stRadio"] label:hover {
  background: rgba(200,16,46,0.10) !important;
  border-color: rgba(200,16,46,0.38) !important;
  color: var(--text-primary) !important;
}
/* Geselecteerde radio pill */
div[data-testid="stRadio"] label:has(input:checked) {
  background: linear-gradient(135deg, rgba(200,16,46,0.28), rgba(139,10,31,0.22)) !important;
  border-color: rgba(200,16,46,0.60) !important;
  color: #F0F0F0 !important;
  box-shadow: 0 0 14px rgba(200,16,46,0.22) !important;
}
/* Verberg de radio dot zelf */
div[data-testid="stRadio"] input[type="radio"] {
  display: none !important;
}

/* â”€â”€ SUBTAB NAV â€” iets kleiner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.subnav-wrap div[data-testid="stRadio"] label {
  padding: 5px 14px !important;
  font-size: 0.76rem !important;
  border-radius: 7px !important;
}

/* â”€â”€ DIVIDER (st.divider) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
hr[data-testid="stDivider"] {
  border: none !important;
  height: 1px !important;
  background: linear-gradient(90deg, rgba(200,16,46,0.4) 0%, rgba(200,16,46,0.15) 40%, transparent 70%) !important;
  margin: 1.2rem 0 1.6rem !important;
}

/* â”€â”€ DEBUG EXPANDER â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
[data-testid="stExpander"] {
  background: rgba(255,255,255,0.025) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 10px !important;
  margin-bottom: 1rem !important;
}
[data-testid="stExpander"] summary {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
}

/* â”€â”€ FORM ELEMENTEN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.stTextInput input,
.stSelectbox [data-baseweb="select"],
.stNumberInput input,
.stDateInput input {
  background: rgba(255,255,255,0.055) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 10px !important;
  color: var(--text-primary) !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus,
.stNumberInput input:focus {
  border-color: rgba(200,16,46,0.55) !important;
  box-shadow: 0 0 0 2px var(--mvv-red-glow) !important;
}

/* â”€â”€ BUTTONS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
div[data-testid="stButton"] > button {
  background: linear-gradient(135deg, #C8102E, #8B0A1F) !important;
  border: none !important;
  border-radius: 10px !important;
  color: white !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.05em !important;
  box-shadow: 0 4px 14px rgba(200,16,46,0.28), inset 0 1px 0 rgba(255,255,255,0.14) !important;
  transition: filter .2s ease, transform .15s ease !important;
}
div[data-testid="stButton"] > button:hover {
  filter: brightness(1.14) !important;
  transform: translateY(-1px) !important;
}
div[data-testid="stButton"] > button:disabled {
  background: rgba(255,255,255,0.05) !important;
  box-shadow: none !important;
  color: rgba(240,240,240,0.28) !important;
  transform: none !important; filter: none !important;
}

/* â”€â”€ FILE UPLOADER â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
[data-testid="stFileUploader"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px dashed rgba(200,16,46,0.35) !important;
  border-radius: 12px !important;
  padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: rgba(200,16,46,0.6) !important;
  background: rgba(200,16,46,0.05) !important;
}

/* â”€â”€ DATAFRAME / TABELLEN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
[data-testid="stDataFrame"] {
  border: 1px solid var(--glass-border) !important;
  border-radius: 12px !important;
  overflow: hidden !important;
}

/* â”€â”€ METRIC CARDS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.035) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 12px !important;
  padding: 1rem 1.2rem !important;
  box-shadow: 0 4px 16px rgba(0,0,0,0.3), inset 0 1px 0 var(--glass-shine) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Bebas Neue', sans-serif !important;
  font-size: 1.9rem !important;
  letter-spacing: 1px !important;
  color: var(--text-primary) !important;
}
[data-testid="stMetricLabel"] {
  font-size: 0.74rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
}
[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="stMetricDelta"] { color: var(--mvv-red-light) !important; font-weight: 600 !important; }

/* â”€â”€ SUCCESS / INFO / WARNING / ERROR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
[data-testid="stAlert"][data-baseweb="notification"] {
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* â”€â”€ LABELS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
label[data-testid="stWidgetLabel"] > div > p {
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important; font-size: 0.78rem !important;
  letter-spacing: 0.06em !important; text-transform: uppercase !important;
  color: var(--text-muted) !important;
}

/* â”€â”€ SECTION LABEL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.mvv-section-label {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 0.82rem; letter-spacing: 0.22em;
  color: var(--mvv-red-light); text-transform: uppercase;
  margin: 1.4rem 0 0.8rem;
}

/* â”€â”€ CONTENT CARD (glassmorphism wrapper) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.mvv-card {
  background: var(--metal-grad), rgba(255,255,255,0.035);
  border: 1px solid var(--glass-border);
  border-radius: 16px;
  padding: 1.4rem 1.6rem;
  box-shadow: 0 8px 28px rgba(0,0,0,0.38), inset 0 1px 0 var(--glass-shine);
  margin-bottom: 1.2rem;
}

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

_GPS_IMPORT_BACKGROUND = (
  f"linear-gradient(180deg, rgba(6, 10, 20, 0.82) 0%, rgba(6, 10, 20, 0.80) 100%), "
  f"radial-gradient(circle at top left, rgba(200, 16, 46, 0.16), rgba(200, 16, 46, 0.02) 24%, transparent 46%), "
  f"radial-gradient(circle at top right, rgba(234, 51, 81, 0.10), rgba(234, 51, 81, 0.02) 18%, transparent 42%), "
  f"url('{PAGE_BG_URI}')"
  if PAGE_BG_URI
  else "radial-gradient(circle at top left, rgba(200, 16, 46, 0.28), rgba(200, 16, 46, 0.03) 26%, transparent 48%), radial-gradient(circle at top right, rgba(234, 51, 81, 0.18), rgba(234, 51, 81, 0.03) 18%, transparent 44%), linear-gradient(180deg, #070c18 0%, #0a1020 100%)"
)

st.markdown(
f"""
<style>
.stApp {{
  background: {_GPS_IMPORT_BACKGROUND} !important;
  background-size: cover !important;
  background-position: center top !important;
  background-attachment: fixed !important;
}}

[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(16, 23, 38, 0.98), rgba(9, 13, 23, 0.98)) !important;
  border-right: 1px solid rgba(255,255,255,0.06) !important;
}}

.block-container {{
  padding-top: 1.2rem !important;
  padding-bottom: 2.4rem !important;
  max-width: 1380px !important;
}}

.mvv-hero-shell {{
  display: flex;
  flex-direction: column;
  gap: 1.1rem;
  margin-bottom: 1.55rem;
}}

.mvv-hero {{
  min-height: 320px;
  padding: 2rem 1.75rem 1.9rem 1.75rem;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.08);
  background: linear-gradient(135deg, rgba(18, 25, 42, 0.88), rgba(10, 15, 27, 0.84));
  box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22);
}}

.mvv-page-logo {{
  width: 82px;
  height: 82px;
  object-fit: contain;
  margin-bottom: 0.9rem;
  filter: drop-shadow(0 8px 22px rgba(0,0,0,0.28));
}}

.mvv-page-kicker {{
  color: rgba(255,255,255,0.76);
  font-size: 0.74rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  margin-bottom: 0.35rem;
}}

.mvv-page-title {{
  font-size: 2.55rem;
  line-height: 1;
  letter-spacing: 0;
  font-family: inherit;
}}

.mvv-page-copy {{
  margin-top: 0.8rem;
  max-width: 74ch;
  color: rgba(255,255,255,0.84);
  line-height: 1.62;
}}

.mvv-pill-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
  margin-top: 1rem;
}}

.mvv-pill {{
  display: inline-flex;
  align-items: center;
  padding: 0.42rem 0.76rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 800;
  border: 1px solid rgba(234, 51, 81, 0.22);
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.92);
}}

.mvv-summary-grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1rem;
}}

.mvv-summary-card,
.mvv-nav-card {{
  border-radius: 8px;
  border: 1px solid rgba(234, 51, 81, 0.14);
  background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
}}

.mvv-summary-card {{
  min-height: 122px;
  padding: 1rem 1.05rem 0.95rem 1.05rem;
}}

.mvv-summary-label {{
  color: rgba(255,255,255,0.68);
  font-size: 0.8rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}}

.mvv-summary-value {{
  margin-top: 0.55rem;
  font-size: 2rem;
  line-height: 1.05;
  font-weight: 800;
  color: #FFFFFF;
}}

.mvv-summary-foot {{
  margin-top: 0.65rem;
  color: rgba(255,255,255,0.8);
  font-size: 0.86rem;
  line-height: 1.4;
}}

.mvv-nav-card {{
  padding: 16px;
  margin-bottom: 18px;
}}

.mvv-nav-head {{
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 1rem;
  margin-bottom: 10px;
}}

.mvv-nav-label {{
  color: rgba(255,255,255,0.62);
  font-size: 0.75rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}}

.mvv-nav-title {{
  margin-top: 0.25rem;
  color: #FFFFFF;
  font-size: 1.05rem;
  font-weight: 700;
}}

.mvv-nav-note {{
  color: rgba(255,255,255,0.8);
  font-size: 0.88rem;
  font-weight: 700;
  text-align: right;
}}

@media (max-width: 1100px) {{
  .mvv-summary-grid {{
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}
}}

@media (max-width: 768px) {{
  .mvv-hero {{
    min-height: auto;
    padding: 1.55rem 1rem;
  }}

  .mvv-page-title {{
    font-size: 2rem;
  }}

  .mvv-summary-grid {{
    grid-template-columns: 1fr;
  }}

  .mvv-nav-head {{
    flex-direction: column;
    align-items: flex-start;
  }}

  .mvv-nav-note {{
    text-align: left;
  }}
}}
</style>
""",
unsafe_allow_html=True,
)


# ============================================================
# AUTH & PROFIEL
# ============================================================

access_token = get_access_token()
if not access_token:
    st.error("Niet ingelogd (access_token ontbreekt).")
    st.stop()

try:
    user_id, email, role, team = get_profile_role(access_token)
except Exception as e:
    st.error(f"Kon profiel/role niet ophalen: {e}")
    st.stop()

role_ui = (role or st.session_state.get("role") or "").strip().lower()
if role_ui == "player":
    st.error("Geen toegang.")
    st.stop()

st.session_state["role"] = role_ui

if role_ui not in ALLOWED_IMPORT:
    st.error("Geen rechten voor GPS import/export.")
    st.stop()

name_to_id, player_options = get_players_map(access_token)


# ============================================================
# PAGE HEADER
# ============================================================

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
    <div class="mvv-page-kicker">MVV Maastricht | GPS Import | Staff</div>
    <div class="mvv-page-title">GPS Import</div>
    <div class="mvv-page-copy">
      Centrale werkruimte voor het importeren, aanvullen, exporteren en koppelen van GPS-data.
      Gebruik de hoofd- en subnavigatie om snel tussen workflowstappen en matchbeheer te schakelen.
    </div>
    <div class="mvv-pill-row">
      <span class="mvv-pill">Import via Excel of handmatige invoer</span>
      <span class="mvv-pill">Matchbeheer en export in dezelfde omgeving</span>
    </div>
  </div>
</div>
""",
unsafe_allow_html=True,
)

# Debug expander - alleen zichtbaar als DIAG_MODE of handmatig
with st.expander("Debug (auth/role)", expanded=False):
    st.write({"email": email, "user_id": user_id, "role": role_ui, "team": team})


# ============================================================
# HOOFD NAVIGATIE
# ============================================================

st.markdown(
    """
    <div class="mvv-nav-card">
      <div class="mvv-nav-head">
        <div>
          <div class="mvv-nav-label">Navigatie</div>
          <div class="mvv-nav-title">Kies de hoofdworkflow voor GPS-beheer</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

main_page = st.radio(
    "Navigatie",
    options=["Import GPS", "Matches"],
    horizontal=True,
    key="nav_main",
    label_visibility="collapsed",
)


# ============================================================
# PAGE: Import GPS
# ============================================================

if main_page == "Import GPS":

    st.markdown(
        """
        <div class="mvv-nav-card">
          <div class="mvv-nav-head">
            <div>
              <div class="mvv-nav-label">Import type</div>
              <div class="mvv-nav-title">Kies hoe je nieuwe GPS-data wilt toevoegen of uitvoeren</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sub-navigatie in eigen wrapper voor kleinere pills
    st.markdown('<div class="subnav-wrap">', unsafe_allow_html=True)
    sub_page = st.radio(
        "Sub",
        options=["Import (Excel)", "Manual add", "Export"],
        horizontal=True,
        key="nav_gps_sub",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    if sub_page == "Import (Excel)":
        tab_import_excel_main(access_token=access_token, name_to_id=name_to_id)

    elif sub_page == "Manual add":
        tab_manual_add_main(access_token=access_token, name_to_id=name_to_id, player_options=player_options)

    elif sub_page == "Export":
        tab_export_main(access_token=access_token, player_options=player_options)


# ============================================================
# PAGE: Matches
# ============================================================

elif main_page == "Matches":
    st.divider()
    tab_matches_main(access_token=access_token)

