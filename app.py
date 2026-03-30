# app.py
# ============================================================
# MVV Dashboard - Main App (Streamlit) — Redesign
# ============================================================

from pathlib import Path
import streamlit as st

from roles import (
    get_sb,
    cookie_mgr,
    set_tokens_in_cookie,
    clear_tokens_in_cookie,
    try_restore_or_refresh_session,
    get_profile,
)

import extra_streamlit_components as stx  # noqa: F401


# ============================================================
# 1) CONFIG & MODES
# ============================================================

st.set_page_config(page_title="MVV Dashboard", layout="wide", initial_sidebar_state="expanded")

DIAG_MODE = st.query_params.get("diag") == "1"
SAFE_MODE = st.query_params.get("safe") == "1"

MAINTENANCE_MODE = False
MAINTENANCE_TITLE = "⚠️ ONDERHOUD"
MAINTENANCE_TEXT = "Er wordt onderhoud uitgevoerd. Je kunt mogelijk (tijdelijk) uitgelogd worden."

if DIAG_MODE:
    st.title("DIAG OK")
    st.write("Als je dit ziet, werkt Streamlit op dit toestel/netwerk.")
    st.write("Zet diag uit door ?diag=0 of verwijder de query parameter.")
    st.stop()


# ============================================================
# 2) SUPABASE CLIENT
# ============================================================

sb = get_sb()
if sb is None:
    st.error("Supabase client niet beschikbaar (secrets ontbreken of create_client faalt).")
    st.stop()


# ============================================================
# 3) GLOBAL CSS — MVV BRAND REDESIGN
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

/* ── ROOT TOKENS ─────────────────────────────────────────── */
:root {
  --mvv-red:        #C8102E;
  --mvv-red-light:  #E8213F;
  --mvv-red-dark:   #8B0A1F;
  --mvv-red-glow:   rgba(200, 16, 46, 0.35);
  --mvv-red-subtle: rgba(200, 16, 46, 0.10);
  --bg-deep:        #0B0C10;
  --bg-card:        rgba(255,255,255,0.04);
  --bg-card-hover:  rgba(255,255,255,0.075);
  --glass-border:   rgba(255,255,255,0.09);
  --glass-shine:    rgba(255,255,255,0.13);
  --text-primary:   #F0F0F0;
  --text-muted:     rgba(240,240,240,0.48);
  --metal-grad: linear-gradient(
      135deg,
      rgba(255,255,255,0.14) 0%,
      rgba(255,255,255,0.04) 40%,
      rgba(255,255,255,0.10) 100%
  );
}

/* ── FULL PAGE BACKGROUND ────────────────────────────────── */
.stApp {
  background:
    radial-gradient(ellipse 80% 60% at 15% -5%,  rgba(200,16,46,0.18) 0%, transparent 55%),
    radial-gradient(ellipse 60% 50% at 90% 90%,  rgba(200,16,46,0.12) 0%, transparent 50%),
    radial-gradient(ellipse 100% 80% at 50% 50%, #0D0E13 30%, #09090D 100%);
  background-attachment: fixed;
  font-family: 'DM Sans', sans-serif;
  color: var(--text-primary);
}

/* ── MAIN BLOCK CONTAINER ────────────────────────────────── */
.block-container {
  padding: 2rem 2.5rem 3rem !important;
  max-width: 1280px !important;
}

/* ── SIDEBAR ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg, rgba(200,16,46,0.12) 0%, rgba(11,12,16,0.98) 30%) !important;
  border-right: 1px solid var(--glass-border) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stAlert {
  font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] .stAlert {
  background: var(--mvv-red-subtle) !important;
  border: 1px solid rgba(200,16,46,0.35) !important;
  border-radius: 10px !important;
  color: var(--text-primary) !important;
}

/* ── PAGE HEADER ─────────────────────────────────────────── */
.mvv-header {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 0.4rem;
}
.mvv-header-logo {
  width: 52px;
  height: 52px;
  background: linear-gradient(135deg, var(--mvv-red), var(--mvv-red-dark));
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: 'Bebas Neue', sans-serif;
  font-size: 22px;
  color: white;
  letter-spacing: 1px;
  box-shadow: 0 4px 18px var(--mvv-red-glow), inset 0 1px 0 rgba(255,255,255,0.18);
  flex-shrink: 0;
}
.mvv-header-title {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2.8rem;
  letter-spacing: 3px;
  color: var(--text-primary);
  line-height: 1;
  margin: 0;
}
.mvv-header-sub {
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--text-muted);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-top: 4px;
}
.mvv-divider {
  height: 1px;
  background: linear-gradient(90deg, var(--mvv-red) 0%, rgba(200,16,46,0.3) 40%, transparent 70%);
  margin: 1.2rem 0 2rem 0;
}
.mvv-section-label {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 0.9rem;
  letter-spacing: 0.22em;
  color: var(--mvv-red-light);
  text-transform: uppercase;
  margin-bottom: 1.1rem;
}

/* ── TILE CARDS ──────────────────────────────────────────── */
.tile-card {
  position: relative;
  background: var(--metal-grad), var(--bg-card);
  border: 1px solid var(--glass-border);
  border-radius: 18px;
  overflow: hidden;
  transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
  cursor: pointer;
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  box-shadow:
    0 8px 32px rgba(0,0,0,0.45),
    inset 0 1px 0 var(--glass-shine);
}
.tile-card:hover {
  transform: translateY(-5px) scale(1.012);
  border-color: rgba(200,16,46,0.5);
  box-shadow:
    0 16px 48px rgba(0,0,0,0.55),
    0 0 28px var(--mvv-red-glow),
    inset 0 1px 0 var(--glass-shine);
}
.tile-card.disabled {
  opacity: 0.42;
  cursor: not-allowed;
  filter: grayscale(0.35);
}
.tile-card.disabled:hover {
  transform: none;
  box-shadow: 0 8px 32px rgba(0,0,0,0.45), inset 0 1px 0 var(--glass-shine);
  border-color: var(--glass-border);
}
.tile-img-wrap {
  width: 100%;
  aspect-ratio: 16/9;
  overflow: hidden;
  position: relative;
}
.tile-img-wrap img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
/* metallic top sheen */
.tile-img-wrap::after {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 40%;
  background: linear-gradient(180deg, rgba(255,255,255,0.07) 0%, transparent 100%);
  pointer-events: none;
}
.tile-footer {
  padding: 12px 14px 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-top: 1px solid var(--glass-border);
  background: rgba(0,0,0,0.18);
}
.tile-name {
  font-family: 'DM Sans', sans-serif;
  font-weight: 600;
  font-size: 0.82rem;
  letter-spacing: 0.05em;
  color: var(--text-primary);
  text-transform: uppercase;
}
.tile-badge {
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  padding: 3px 9px;
  border-radius: 20px;
  text-transform: uppercase;
}
.tile-badge.open {
  background: rgba(200,16,46,0.18);
  color: var(--mvv-red-light);
  border: 1px solid rgba(200,16,46,0.4);
}
.tile-badge.soon {
  background: rgba(255,255,255,0.06);
  color: var(--text-muted);
  border: 1px solid rgba(255,255,255,0.1);
}
.tile-badge.locked {
  background: rgba(255,255,255,0.04);
  color: rgba(240,240,240,0.28);
  border: 1px solid rgba(255,255,255,0.07);
}

/* ── MAINTENANCE BANNER ──────────────────────────────────── */
.maintenance-wrap {
  padding: 13px 18px;
  border-radius: 12px;
  border: 1px solid rgba(255,80,80,0.4);
  background: rgba(200,16,46,0.1);
  margin-bottom: 1.2rem;
  font-weight: 700;
  font-size: 0.9rem;
}
.maintenance-wrap .sub {
  font-weight: 400;
  opacity: 0.8;
  margin-top: 4px;
  font-size: 0.85rem;
}

/* ── LOGIN FORM ──────────────────────────────────────────── */
.login-wrap {
  max-width: 420px;
  margin: 8vh auto 0;
  background: var(--metal-grad), rgba(255,255,255,0.035);
  border: 1px solid var(--glass-border);
  border-radius: 22px;
  padding: 2.4rem 2.2rem;
  box-shadow:
    0 24px 64px rgba(0,0,0,0.6),
    0 0 40px var(--mvv-red-glow),
    inset 0 1px 0 var(--glass-shine);
  backdrop-filter: blur(12px);
}
.login-brand {
  text-align: center;
  margin-bottom: 1.8rem;
}
.login-brand-mark {
  width: 64px; height: 64px;
  background: linear-gradient(135deg, var(--mvv-red), var(--mvv-red-dark));
  border-radius: 18px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-family: 'Bebas Neue', sans-serif;
  font-size: 28px;
  color: white;
  letter-spacing: 1px;
  box-shadow: 0 6px 24px var(--mvv-red-glow), inset 0 1px 0 rgba(255,255,255,0.2);
  margin-bottom: 14px;
}
.login-brand h2 {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2rem;
  letter-spacing: 4px;
  margin: 0;
  color: var(--text-primary);
}
.login-brand p {
  font-size: 0.78rem;
  color: var(--text-muted);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin: 4px 0 0;
}

/* ── STREAMLIT OVERRIDES ─────────────────────────────────── */
/* text inputs */
.stTextInput input {
  background: rgba(255,255,255,0.055) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 10px !important;
  color: var(--text-primary) !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus {
  border-color: rgba(200,16,46,0.6) !important;
  box-shadow: 0 0 0 2px var(--mvv-red-glow) !important;
}
/* primary buttons */
.stButton > button[kind="primary"],
.stFormSubmitButton > button {
  background: linear-gradient(135deg, var(--mvv-red), var(--mvv-red-dark)) !important;
  border: none !important;
  border-radius: 10px !important;
  color: white !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: 0.05em !important;
  box-shadow: 0 4px 18px var(--mvv-red-glow), inset 0 1px 0 rgba(255,255,255,0.18) !important;
  transition: filter 0.2s ease, transform 0.15s ease !important;
}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button:hover {
  filter: brightness(1.12) !important;
  transform: translateY(-1px) !important;
}
/* secondary/other buttons */
.stButton > button:not([kind="primary"]) {
  background: rgba(255,255,255,0.055) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 10px !important;
  color: var(--text-primary) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  transition: background 0.2s ease, border-color 0.2s ease !important;
}
.stButton > button:not([kind="primary"]):hover {
  background: rgba(200,16,46,0.12) !important;
  border-color: rgba(200,16,46,0.4) !important;
}
/* hide default streamlit image styling — we handle it ourselves */
[data-testid="stImage"] img {
  border-radius: 0 !important;
  box-shadow: none !important;
  border: none !important;
}
/* labels */
label[data-testid="stWidgetLabel"] > div > p {
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
}
/* title override */
h1[data-testid="stHeading"] {
  font-family: 'Bebas Neue', sans-serif !important;
  letter-spacing: 3px !important;
  font-size: 2.4rem !important;
}
/* hide streamlit default menu for cleaner look */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 4) UI HELPERS
# ============================================================

def maintenance_banner():
    if not MAINTENANCE_MODE:
        return
    st.markdown(
        f"""
        <div class="maintenance-wrap">
            {MAINTENANCE_TITLE}
            <div class="sub">{MAINTENANCE_TEXT}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Map tile_id -> display name
TILE_LABELS = {
    "player":       "Player Page",
    "matchreports": "Match Reports",
    "gpsdata":      "GPS Data",
    "gpsimport":    "GPS Import",
    "medical":      "Medical",
    "accounts":     "Accounts",
}


def tile(tile_id: str, img_path: str, target_page: str | None, disabled: bool = False):
    """
    Premium glassmorphism tile card.
    - Image rendered via st.image (no base64 overhead)
    - HTML wrapper for card chrome/styling
    - MVV brand accents + metallic sheen
    """
    label = TILE_LABELS.get(tile_id, tile_id.replace("_", " ").title())

    if disabled:
        badge_html = '<span class="tile-badge locked">Geen toegang</span>'
        card_class = "tile-card disabled"
    elif target_page is None:
        badge_html = '<span class="tile-badge soon">Coming Soon</span>'
        card_class = "tile-card"
    else:
        badge_html = '<span class="tile-badge open">Open</span>'
        card_class = "tile-card"

    # Card top chrome (opens the div)
    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)

    # Image area
    if not SAFE_MODE:
        p = Path(img_path)
        if p.exists():
            st.markdown('<div class="tile-img-wrap">', unsafe_allow_html=True)
            st.image(str(p), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Placeholder gradient if image missing
            st.markdown(
                f"""
                <div class="tile-img-wrap" style="
                    background: linear-gradient(135deg, rgba(200,16,46,0.18) 0%, rgba(11,12,16,1) 100%);
                    display:flex; align-items:center; justify-content:center;
                    color:rgba(240,240,240,0.25); font-size:0.75rem; letter-spacing:0.1em;
                    text-transform:uppercase; font-family:'DM Sans',sans-serif;
                ">
                    {label}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Footer with name + badge
    st.markdown(
        f"""
        <div class="tile-footer">
            <span class="tile-name">{label}</span>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Spacer + functional Streamlit button (invisible trigger)
    if not disabled and target_page is not None:
        if st.button("Open", use_container_width=True, key=f"btn_{tile_id}_open"):
            st.switch_page(target_page)
    elif disabled:
        st.button("Geen toegang", use_container_width=True, disabled=True, key=f"btn_{tile_id}_noaccess")
    else:
        st.button("Coming soon", use_container_width=True, disabled=True, key=f"btn_{tile_id}_soon")


# ============================================================
# 5) LOGIN UI
# ============================================================

def login_ui():
    maintenance_banner()

    st.markdown("""
    <div class="login-wrap">
        <div class="login-brand">
            <div class="login-brand-mark">MVV</div>
            <h2>MVV DASHBOARD</h2>
            <p>Performance Analytics Platform</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Streamlit form (rendered below the card visually, inside the page center)
    with st.form("login_form", clear_on_submit=False):
        col_l, col_m, col_r = st.columns([1, 2, 1])
        with col_m:
            email    = st.text_input("Email", key="login_email", placeholder="jouw@email.nl")
            password = st.text_input("Wachtwoord", type="password", key="login_pw", placeholder="••••••••")
            submitted = st.form_submit_button("Inloggen", use_container_width=True)

    if not submitted:
        return

    try:
        res = sb.auth.sign_in_with_password({"email": email, "password": password})
        sess = getattr(res, "session", None)
        token = getattr(sess, "access_token", None)
        refresh_token = getattr(sess, "refresh_token", None)

        if not token or not refresh_token:
            st.error("Login mislukt: geen geldige sessie ontvangen.")
            return

        st.session_state["access_token"] = token
        st.session_state["user_email"] = email
        st.session_state["sb_session"] = sess

        set_tokens_in_cookie(token, refresh_token, email)

        import time
        time.sleep(0.10)

        st.session_state.pop("_profile_cache", None)
        st.session_state.pop("role", None)
        st.session_state.pop("player_id", None)

        st.rerun()

    except Exception as e:
        st.error(f"Sign in mislukt: {e}")


# ============================================================
# 6) LOGOUT BUTTON
# ============================================================

def logout_button():
    if st.button("Uitloggen", use_container_width=True, key="btn_logout"):
        try:
            sb.auth.sign_out()
        except Exception:
            pass
        clear_tokens_in_cookie()
        import time
        time.sleep(0.10)
        st.session_state.clear()
        st.rerun()


# ============================================================
# 7) BOOT COOKIE COMPONENT EARLY
# ============================================================

try:
    cm = cookie_mgr()
    _ = cm.get("sb_refresh")
except Exception:
    pass


# ============================================================
# 8) AUTH GATE
# ============================================================

if "access_token" not in st.session_state:
    restored = try_restore_or_refresh_session(sb)
    if not restored:
        login_ui()
        st.stop()


# ============================================================
# 9) PROFILE LOAD
# ============================================================

profile = get_profile(sb)
if not profile:
    clear_tokens_in_cookie()
    st.session_state.clear()
    st.error("Geen profiel gevonden of geen rechten. Log opnieuw in.")
    st.stop()


# ============================================================
# 10) SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
        <div style="
            font-family:'Bebas Neue',sans-serif;
            font-size:1.5rem;
            letter-spacing:3px;
            color:#F0F0F0;
            margin-bottom:1.2rem;
            padding-bottom:10px;
            border-bottom:1px solid rgba(200,16,46,0.35);
        ">MVV DASHBOARD</div>
    """, unsafe_allow_html=True)

    st.success(f"✓  {st.session_state.get('user_email','')}")
    st.info(f"Rol: **{st.session_state.get('role','').upper()}**")

    if DIAG_MODE:
        with st.expander("Auth debug", expanded=True):
            cm = cookie_mgr()
            st.write("session access:", bool(st.session_state.get("access_token")))
            st.write("cookie access:",  bool(cm.get("sb_access")))
            st.write("cookie refresh:", bool(cm.get("sb_refresh")))
            st.write("auth_err:", st.session_state.get("auth_err"))

    st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)
    logout_button()

maintenance_banner()


# ============================================================
# 11) PAGE HEADER
# ============================================================

st.markdown("""
<div class="mvv-header">
    <div class="mvv-header-logo">MVV</div>
    <div>
        <div class="mvv-header-title">Dashboard</div>
        <div class="mvv-header-sub">Performance Analytics Platform</div>
    </div>
</div>
<div class="mvv-divider"></div>
""", unsafe_allow_html=True)

if SAFE_MODE:
    st.warning("Safe mode actief (minimale UI). Zet uit via ?safe=0.")


# ============================================================
# 12) TILES — PLAYER vs STAFF
# ============================================================

role = (st.session_state.get("role") or "").lower()
is_player = role == "player"

if is_player:
    st.markdown('<div class="mvv-section-label">Jouw modules</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        tile("player",       "Assets/Afbeeldingen/Script/Player_page.PNG",  "pages/01_Player_Page.py")
    with c2:
        tile("matchreports", "Assets/Afbeeldingen/Script/Match Report.PNG",  "pages/02_Match_Reports.py")

else:
    st.markdown('<div class="mvv-section-label">Analyse & Rapportage</div>', unsafe_allow_html=True)
    r1c1, r1c2, r1c3 = st.columns(3, gap="large")
    with r1c1:
        tile("player",       "Assets/Afbeeldingen/Script/Player_page.PNG",   "pages/01_Player_Page.py")
    with r1c2:
        tile("matchreports", "Assets/Afbeeldingen/Script/Match Report.PNG",  "pages/02_Match_Reports.py")
    with r1c3:
        tile("gpsdata",      "Assets/Afbeeldingen/Script/GPS_Data.PNG",      "pages/03_GPS_Data.py")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="mvv-section-label">Data & Beheer</div>', unsafe_allow_html=True)

    r2c1, r2c2, r2c3 = st.columns(3, gap="large")
    with r2c1:
        tile("gpsimport", "Assets/Afbeeldingen/Script/GPS_Import.PNG",   "pages/06_GPS_Import.py")
    with r2c2:
        tile("medical",   "Assets/Afbeeldingen/Script/Medical.PNG",      None, disabled=True)
    with r2c3:
        tile("accounts",  "Assets/Afbeeldingen/Script/Accounts.PNG",     None, disabled=True)
