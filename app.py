# app.py
# ============================================================
# MVV Dashboard - Main App (Streamlit) — Redesign v2
# Tiles gebruiken base64-embedded images in pure HTML zodat
# de kaart-structuur intact blijft (st.image() werkt NIET
# genest in een HTML div in Streamlit).
# ============================================================

from pathlib import Path
import base64
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
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
  --mvv-red:        #C8102E;
  --mvv-red-light:  #E8213F;
  --mvv-red-dark:   #8B0A1F;
  --mvv-red-glow:   rgba(200,16,46,0.35);
  --mvv-red-subtle: rgba(200,16,46,0.10);
  --glass-border:   rgba(255,255,255,0.09);
  --glass-shine:    rgba(255,255,255,0.13);
  --metal-grad: linear-gradient(135deg,rgba(255,255,255,0.13) 0%,rgba(255,255,255,0.035) 40%,rgba(255,255,255,0.09) 100%);
  --text-primary: #F0F0F0;
  --text-muted:   rgba(240,240,240,0.45);
}

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

[data-testid="stSidebar"] {
  background: linear-gradient(180deg,rgba(200,16,46,0.13) 0%,rgba(9,9,13,0.98) 28%) !important;
  border-right: 1px solid var(--glass-border) !important;
}
[data-testid="stSidebar"] .stAlert {
  background: var(--mvv-red-subtle) !important;
  border: 1px solid rgba(200,16,46,0.35) !important;
  border-radius: 10px !important;
  color: var(--text-primary) !important;
}

.mvv-header { display:flex; align-items:center; gap:18px; margin-bottom:0; }
.mvv-header-logo {
  width:50px; height:50px; flex-shrink:0;
  background: linear-gradient(135deg,var(--mvv-red),var(--mvv-red-dark));
  border-radius:13px;
  display:flex; align-items:center; justify-content:center;
  font-family:'Bebas Neue',sans-serif; font-size:20px; color:white; letter-spacing:1px;
  box-shadow:0 4px 18px var(--mvv-red-glow),inset 0 1px 0 rgba(255,255,255,0.18);
}
.mvv-header-title {
  font-family:'Bebas Neue',sans-serif; font-size:2.6rem; letter-spacing:3px;
  color:var(--text-primary); line-height:1; margin:0;
}
.mvv-header-sub {
  font-size:0.75rem; font-weight:500; color:var(--text-muted);
  letter-spacing:0.1em; text-transform:uppercase; margin-top:3px;
}
.mvv-divider {
  height:1px;
  background:linear-gradient(90deg,var(--mvv-red) 0%,rgba(200,16,46,0.3) 38%,transparent 68%);
  margin:1.1rem 0 1.8rem;
}
.mvv-section-label {
  font-family:'Bebas Neue',sans-serif; font-size:0.82rem; letter-spacing:0.22em;
  color:var(--mvv-red-light); text-transform:uppercase; margin-bottom:1rem;
}

/* ── TILE CARD ──────────────────────────────────────────── */
.mvv-tile {
  position:relative;
  background: var(--metal-grad), rgba(255,255,255,0.04);
  border:1px solid var(--glass-border);
  border-radius:16px 16px 0 0;
  overflow:hidden;
  box-shadow:0 8px 28px rgba(0,0,0,0.48),inset 0 1px 0 var(--glass-shine);
  transition:transform .25s ease,box-shadow .25s ease,border-color .25s ease;
  cursor:pointer;
}
.mvv-tile:hover {
  transform:translateY(-4px) scale(1.012);
  border-color:rgba(200,16,46,0.52);
  box-shadow:0 18px 48px rgba(0,0,0,0.56),0 0 28px var(--mvv-red-glow),inset 0 1px 0 var(--glass-shine);
}
.mvv-tile.tile-disabled {
  opacity:0.4; cursor:not-allowed; filter:grayscale(0.4); pointer-events:none;
}
.mvv-tile-img { width:100%; aspect-ratio:16/9; overflow:hidden; position:relative; display:block; }
.mvv-tile-img img { width:100%; height:100%; object-fit:cover; display:block; }
.mvv-tile-img::after {
  content:''; position:absolute; top:0; left:0; right:0; height:42%;
  background:linear-gradient(180deg,rgba(255,255,255,0.07) 0%,transparent 100%);
  pointer-events:none;
}
.mvv-tile-placeholder {
  width:100%; aspect-ratio:16/9;
  background:linear-gradient(135deg,rgba(200,16,46,0.15) 0%,#0D0E13 100%);
  display:flex; align-items:center; justify-content:center;
  color:rgba(240,240,240,0.18); font-size:0.72rem; letter-spacing:0.12em; text-transform:uppercase;
}
.mvv-tile-footer {
  padding:11px 14px 13px;
  display:flex; align-items:center; justify-content:space-between;
  border-top:1px solid rgba(255,255,255,0.07);
  background:rgba(0,0,0,0.22);
}
.mvv-tile-name {
  font-family:'DM Sans',sans-serif; font-weight:600; font-size:0.76rem;
  letter-spacing:0.07em; text-transform:uppercase; color:var(--text-primary);
}
.mvv-badge { font-size:0.64rem; font-weight:600; letter-spacing:0.08em; padding:3px 9px; border-radius:20px; text-transform:uppercase; }
.mvv-badge-open   { background:rgba(200,16,46,0.18); color:#E8213F; border:1px solid rgba(200,16,46,0.38); }
.mvv-badge-soon   { background:rgba(255,255,255,0.06); color:rgba(240,240,240,0.35); border:1px solid rgba(255,255,255,0.09); }
.mvv-badge-locked { background:rgba(255,255,255,0.04); color:rgba(240,240,240,0.22); border:1px solid rgba(255,255,255,0.07); }

/* ── BUTTONS ─────────────────────────────────────────────── */
div[data-testid="stButton"] > button {
  background: linear-gradient(135deg, #C8102E, #8B0A1F) !important;
  border: none !important;
  border-radius: 0 0 14px 14px !important;
  color: white !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.05em !important;
  margin-top: -4px !important;
  box-shadow: 0 4px 14px rgba(200,16,46,0.28), inset 0 1px 0 rgba(255,255,255,0.14) !important;
  transition: filter .2s ease !important;
}
div[data-testid="stButton"] > button:hover {
  filter: brightness(1.14) !important;
}
div[data-testid="stButton"] > button:disabled {
  background: rgba(255,255,255,0.05) !important;
  box-shadow: none !important;
  color: rgba(240,240,240,0.28) !important;
  filter: none !important;
}

/* Logout — ghost */
[data-testid="stSidebar"] div[data-testid="stButton"] > button {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 10px !important;
  box-shadow: none !important;
  color: rgba(240,240,240,0.6) !important;
  margin-top: 0 !important;
}
[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
  background: rgba(200,16,46,0.12) !important;
  border-color: rgba(200,16,46,0.38) !important;
  color: #F0F0F0 !important;
  filter: none !important;
}

/* ── LOGIN FORM ──────────────────────────────────────────── */
.stTextInput input {
  background: rgba(255,255,255,0.055) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 10px !important;
  color: var(--text-primary) !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus {
  border-color: rgba(200,16,46,0.55) !important;
  box-shadow: 0 0 0 2px var(--mvv-red-glow) !important;
}
.stFormSubmitButton > button {
  background: linear-gradient(135deg,#C8102E,#8B0A1F) !important;
  border: none !important; border-radius: 10px !important;
  color: white !important; font-family:'DM Sans',sans-serif !important; font-weight:600 !important;
  box-shadow: 0 4px 18px var(--mvv-red-glow),inset 0 1px 0 rgba(255,255,255,0.18) !important;
}

.sb-brand-label {
  font-family:'Bebas Neue',sans-serif; font-size:1.4rem; letter-spacing:3px; color:#F0F0F0;
  padding-bottom:10px; border-bottom:1px solid rgba(200,16,46,0.3); margin-bottom:4px;
}
.mvv-maintenance {
  padding:12px 16px; border-radius:12px;
  border:1px solid rgba(255,80,80,0.4); background:rgba(200,16,46,0.10);
  margin-bottom:1rem; font-weight:700; font-size:0.88rem;
}
.mvv-maintenance .sub { font-weight:400; opacity:.8; margin-top:4px; font-size:.83rem; }

#MainMenu, footer { visibility: hidden; }
label[data-testid="stWidgetLabel"] > div > p {
  font-family:'DM Sans',sans-serif !important; font-weight:500 !important;
  font-size:0.8rem !important; letter-spacing:0.06em !important;
  text-transform:uppercase !important; color:var(--text-muted) !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 4) IMAGE HELPER — base64 embed
# ============================================================

def _img_b64(path: str) -> str | None:
    """Return a base64 data-URI for an image file, or None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    ext  = p.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext, "png")
    data = base64.b64encode(p.read_bytes()).decode()
    return f"data:image/{mime};base64,{data}"


# ============================================================
# 5) UI HELPERS
# ============================================================

def maintenance_banner():
    if not MAINTENANCE_MODE:
        return
    st.markdown(
        f'<div class="mvv-maintenance">{MAINTENANCE_TITLE}'
        f'<div class="sub">{MAINTENANCE_TEXT}</div></div>',
        unsafe_allow_html=True,
    )


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
    Glassmorphism tile card met base64-embedded image in één HTML blok.
    De Streamlit button rendert direct erna (margin-top:-4px sluit het aan).
    """
    label = TILE_LABELS.get(tile_id, tile_id.replace("_", " ").title())

    if disabled:
        badge = '<span class="mvv-badge mvv-badge-locked">Geen toegang</span>'
    elif target_page is None:
        badge = '<span class="mvv-badge mvv-badge-soon">Coming Soon</span>'
    else:
        badge = '<span class="mvv-badge mvv-badge-open">Open</span>'

    if SAFE_MODE:
        img_block = ""
    else:
        b64 = _img_b64(img_path)
        if b64:
            img_block = (
                f'<div class="mvv-tile-img">'
                f'<img src="{b64}" alt="{label}" />'
                f'</div>'
            )
        else:
            img_block = f'<div class="mvv-tile-placeholder">{label}</div>'

    card_class = "mvv-tile tile-disabled" if disabled else "mvv-tile"

    # One HTML block = card chrome + image + footer all together
    st.markdown(
        f"""
        <div class="{card_class}">
          {img_block}
          <div class="mvv-tile-footer">
            <span class="mvv-tile-name">{label}</span>
            {badge}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Streamlit button connects visually via border-radius:0 0 14px 14px + margin-top:-4px
    if disabled:
        st.button("Geen toegang", use_container_width=True, disabled=True, key=f"btn_{tile_id}_noaccess")
    elif target_page is None:
        st.button("Coming soon", use_container_width=True, disabled=True, key=f"btn_{tile_id}_soon")
    else:
        if st.button("Open", use_container_width=True, key=f"btn_{tile_id}_open"):
            st.switch_page(target_page)


# ============================================================
# 6) LOGIN UI
# ============================================================

def login_ui():
    maintenance_banner()

    st.markdown("""
    <div style="text-align:center;margin:8vh auto 1.4rem;">
      <div style="
        width:62px;height:62px;
        background:linear-gradient(135deg,#C8102E,#8B0A1F);
        border-radius:17px;
        display:inline-flex;align-items:center;justify-content:center;
        font-family:'Bebas Neue',sans-serif;font-size:26px;color:white;letter-spacing:1px;
        box-shadow:0 6px 24px rgba(200,16,46,0.45),inset 0 1px 0 rgba(255,255,255,0.2);
        margin-bottom:12px;">MVV</div>
      <div style="font-family:'Bebas Neue',sans-serif;font-size:2rem;letter-spacing:4px;color:#F0F0F0;">
        MVV DASHBOARD</div>
      <div style="font-size:0.76rem;color:rgba(240,240,240,0.42);letter-spacing:0.1em;text-transform:uppercase;margin-top:4px;">
        Performance Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        with st.form("login_form", clear_on_submit=False):
            email    = st.text_input("Email",      key="login_email", placeholder="jouw@email.nl")
            password = st.text_input("Wachtwoord", type="password", key="login_pw", placeholder="••••••••")
            submitted = st.form_submit_button("Inloggen", use_container_width=True)

        # Inject autocomplete attributes AFTER the form renders.
        # Streamlit does not set autocomplete/name on inputs, which breaks
        # mobile autofill and password managers. This JS waits for the DOM
        # and patches the attributes on the two login fields.
        st.markdown("""
        <script>
        (function patchLoginAutofill() {
            function patch() {
                // Find the Streamlit form element
                var form = document.querySelector('form[data-testid="stForm"]');
                if (!form) return false;

                var inputs = form.querySelectorAll('input');
                if (inputs.length < 2) return false;

                // First visible non-password input = email
                var emailInput = null;
                var pwInput    = null;
                inputs.forEach(function(inp) {
                    if (inp.type === 'password') {
                        pwInput = inp;
                    } else if (!emailInput && inp.type !== 'hidden') {
                        emailInput = inp;
                    }
                });

                if (emailInput) {
                    emailInput.setAttribute('autocomplete', 'email');
                    emailInput.setAttribute('name',         'email');
                    emailInput.setAttribute('type',         'email');
                    emailInput.setAttribute('inputmode',    'email');
                }
                if (pwInput) {
                    pwInput.setAttribute('autocomplete', 'current-password');
                    pwInput.setAttribute('name',         'password');
                }
                // Mark the form itself
                if (form) {
                    form.setAttribute('autocomplete', 'on');
                }
                return true;
            }

            // Retry until Streamlit has rendered the inputs
            var attempts = 0;
            var timer = setInterval(function() {
                attempts++;
                if (patch() || attempts > 40) clearInterval(timer);
            }, 150);
        })();
        </script>
        """, unsafe_allow_html=True)

    if not submitted:
        return

    try:
        res = sb.auth.sign_in_with_password({"email": email, "password": password})
        sess          = getattr(res, "session", None)
        token         = getattr(sess, "access_token",  None)
        refresh_token = getattr(sess, "refresh_token", None)

        if not token or not refresh_token:
            st.error("Login mislukt: geen geldige sessie ontvangen.")
            return

        st.session_state["access_token"] = token
        st.session_state["user_email"]   = email
        st.session_state["sb_session"]   = sess

        set_tokens_in_cookie(token, refresh_token, email)

        import time; time.sleep(0.10)

        st.session_state.pop("_profile_cache", None)
        st.session_state.pop("role",      None)
        st.session_state.pop("player_id", None)

        st.rerun()

    except Exception as e:
        st.error(f"Sign in mislukt: {e}")


# ============================================================
# 7) LOGOUT BUTTON
# ============================================================

def logout_button():
    if st.button("Uitloggen", use_container_width=True, key="btn_logout"):
        try:
            sb.auth.sign_out()
        except Exception:
            pass
        clear_tokens_in_cookie()
        import time; time.sleep(0.10)
        st.session_state.clear()
        st.rerun()


# ============================================================
# 8) BOOT COOKIE COMPONENT EARLY
# ============================================================

try:
    cm = cookie_mgr()
    _ = cm.get("sb_refresh")
except Exception:
    pass


# ============================================================
# 9) AUTH GATE
# ============================================================

if "access_token" not in st.session_state:
    restored = try_restore_or_refresh_session(sb)
    if not restored:
        login_ui()
        st.stop()


# ============================================================
# 10) PROFILE LOAD
# ============================================================

profile = get_profile(sb)
if not profile:
    clear_tokens_in_cookie()
    st.session_state.clear()
    st.error("Geen profiel gevonden of geen rechten. Log opnieuw in.")
    st.stop()


# ============================================================
# 11) SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="sb-brand-label">MVV DASHBOARD</div>', unsafe_allow_html=True)
    st.success(f"✓  {st.session_state.get('user_email', '')}")
    st.info(f"Rol: **{st.session_state.get('role', '').upper()}**")

    if DIAG_MODE:
        with st.expander("Auth debug", expanded=True):
            cm = cookie_mgr()
            st.write("session access:", bool(st.session_state.get("access_token")))
            st.write("cookie access:",  bool(cm.get("sb_access")))
            st.write("cookie refresh:", bool(cm.get("sb_refresh")))
            st.write("auth_err:", st.session_state.get("auth_err"))

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    logout_button()

maintenance_banner()


# ============================================================
# 12) PAGE HEADER
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
# 13) TILES — PLAYER vs STAFF
# ============================================================

role      = (st.session_state.get("role") or "").lower()
is_player = role == "player"

if is_player:
    st.markdown('<div class="mvv-section-label">Jouw modules</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        tile("player",       "Assets/Afbeeldingen/Script/Player_page.PNG",  "pages/01_Player_Page.py")
    with c2:
        tile("matchreports", "Assets/Afbeeldingen/Script/Match Report.PNG", "pages/02_Match_Reports.py")

else:
    st.markdown('<div class="mvv-section-label">Analyse &amp; Rapportage</div>', unsafe_allow_html=True)
    r1c1, r1c2, r1c3 = st.columns(3, gap="large")
    with r1c1:
        tile("player",       "Assets/Afbeeldingen/Script/Player_page.PNG",   "pages/01_Player_Page.py")
    with r1c2:
        tile("matchreports", "Assets/Afbeeldingen/Script/Match Report.PNG",  "pages/02_Match_Reports.py")
    with r1c3:
        tile("gpsdata",      "Assets/Afbeeldingen/Script/GPS_Data.PNG",      "pages/03_GPS_Data.py")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="mvv-section-label">Data &amp; Beheer</div>', unsafe_allow_html=True)

    r2c1, r2c2, r2c3 = st.columns(3, gap="large")
    with r2c1:
        tile("gpsimport", "Assets/Afbeeldingen/Script/GPS_Import.PNG",  "pages/06_GPS_Import.py")
    with r2c2:
        tile("medical",   "Assets/Afbeeldingen/Script/Medical.PNG",     None, disabled=True)
    with r2c3:
        tile("accounts",  "Assets/Afbeeldingen/Script/Accounts.PNG",    None, disabled=True)
