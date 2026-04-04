# pages/04_Wellness_&_RPE_Overview.py
from __future__ import annotations

import streamlit as st

from roles import require_auth, get_sb, get_profile
from pages.Subscripts.wr_common import fetch_active_players_cached
from pages.Subscripts.wr_tab_day import render_wellness_rpe_tab_day
from pages.Subscripts.wr_tab_week import render_wellness_rpe_tab_week
from pages.Subscripts.wr_tab_checklist import render_wellness_rpe_tab_checklist
from pages.Subscripts.wr_tab_injury import render_wellness_rpe_tab_injury

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

    # Page header
    st.markdown("""
    <div class="mvv-page-header">
      <div class="mvv-page-icon">📊</div>
      <div>
        <div class="mvv-page-title">WELLNESS & RPE OVERVIEW</div>
        <div class="mvv-page-sub">Team performance metrics and wellness tracking</div>
      </div>
    </div>
    <div class="mvv-divider"></div>
    """, unsafe_allow_html=True)

    # Section label
    st.markdown('<div class="mvv-section-label">Team Overview</div>', unsafe_allow_html=True)

    # cache key per project/env
    sb_url_key = str(st.secrets.get("SUPABASE_URL", "sb"))

    players = fetch_active_players_cached(sb_url_key, sb)
    if players.empty:
        st.warning("Geen actieve spelers gevonden.")
        st.stop()

    pid_to_name = dict(zip(players["player_id"], players["full_name"]))

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
