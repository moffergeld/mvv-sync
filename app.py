import base64
from pathlib import Path

import extra_streamlit_components as stx  # noqa: F401
import streamlit as st

from roles import (
    clear_tokens_in_cookie,
    cookie_mgr,
    get_profile,
    get_sb,
    set_tokens_in_cookie,
    try_restore_or_refresh_session,
)


st.set_page_config(page_title="MVV Dashboard", layout="wide")

DIAG_MODE = st.query_params.get("diag") == "1"
SAFE_MODE = st.query_params.get("safe") == "1"

MAINTENANCE_MODE = False
MAINTENANCE_TITLE = "Onderhoud"
MAINTENANCE_TEXT = "Er wordt onderhoud uitgevoerd. Je kunt mogelijk tijdelijk uitgelogd worden."

ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "Assets" / "Afbeeldingen"
TEAM_LOGO = ASSETS_DIR / "Team_Logos" / "MVV Maastricht.png"
HOME_BG = ASSETS_DIR / "Backgrounds" / "team_page_hero.png"

MODULES = [
    {
        "id": "player",
        "title": "Player Page",
        "description": "Individuele spelerstatus, readiness en dagelijkse check-in op een plek.",
        "badge": "Core",
        "img_path": "Assets/Afbeeldingen/Script/Player_page.PNG",
        "target_page": "pages/01_Player_Page.py",
        "roles": {"player", "staff"},
    },
    {
        "id": "matchreports",
        "title": "Match Reports",
        "description": "Rapportage, wedstrijdnotities en context voor staf en evaluatie.",
        "badge": "Analyse",
        "img_path": "Assets/Afbeeldingen/Script/Match Report.PNG",
        "target_page": "pages/02_Match_Reports.py",
        "roles": {"player", "staff"},
    },
    {
        "id": "gpsdata",
        "title": "GPS Data",
        "description": "Belastingsdata, trainingsoutput en trends per speler of sessie.",
        "badge": "Performance",
        "img_path": "Assets/Afbeeldingen/Script/GPS_Data.PNG",
        "target_page": "pages/03_GPS_Data.py",
        "roles": {"staff"},
    },
    {
        "id": "gpsimport",
        "title": "GPS Import",
        "description": "Nieuwe GPS-bestanden inladen en klaarzetten voor verwerking.",
        "badge": "Workflow",
        "img_path": "Assets/Afbeeldingen/Script/GPS_Import.PNG",
        "target_page": "pages/06_GPS_Import.py",
        "roles": {"staff"},
    },
    {
        "id": "medical",
        "title": "Medical",
        "description": "Medische opvolging en beschikbaarheid, straks in dezelfde stijl.",
        "badge": "Binnenkort",
        "img_path": "Assets/Afbeeldingen/Script/Medical.PNG",
        "target_page": None,
        "disabled": True,
        "roles": {"staff"},
    },
    {
        "id": "accounts",
        "title": "Accounts",
        "description": "Gebruikers, rechten en teamtoegang vanuit een centraal paneel.",
        "badge": "Binnenkort",
        "img_path": "Assets/Afbeeldingen/Script/Accounts.PNG",
        "target_page": None,
        "disabled": True,
        "roles": {"staff"},
    },
    {
        "id": "compare",
        "title": "Analytics",
        "description": "Vergelijk spelers en sessies om sneller patronen te spotten.",
        "badge": "Insights",
        "img_path": "Assets/Afbeeldingen/Script/Analytics.PNG",
        "target_page": "pages/05_Compare.py",
        "roles": {"staff"},
    },
]


def build_image_data_uri(path_like: str | Path) -> str:
    path = Path(path_like)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        return ""

    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def maintenance_banner() -> None:
    if not MAINTENANCE_MODE:
        return
    st.markdown(
        f"""
        <div style="
            padding:12px 14px;
            border-radius:8px;
            border:1px solid rgba(234,51,81,.42);
            background:rgba(200,16,46,.16);
            font-weight:800;
            color:#ffffff;">
            {MAINTENANCE_TITLE}
            <div style="font-weight:600;opacity:.9;margin-top:6px">{MAINTENANCE_TEXT}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home_css() -> None:
    page_bg_uri = "" if SAFE_MODE else build_image_data_uri(HOME_BG)
    app_background = (
        f"linear-gradient(180deg, rgba(6, 10, 20, 0.82) 0%, rgba(6, 10, 20, 0.80) 100%), "
        f"radial-gradient(circle at top left, rgba(200, 16, 46, 0.16), rgba(200, 16, 46, 0.02) 24%, transparent 46%), "
        f"radial-gradient(circle at top right, rgba(234, 51, 81, 0.10), rgba(234, 51, 81, 0.02) 18%, transparent 42%), "
        f"url('{page_bg_uri}')"
        if page_bg_uri
        else "radial-gradient(circle at top left, rgba(200, 16, 46, 0.28), rgba(200, 16, 46, 0.03) 26%, transparent 48%), radial-gradient(circle at top right, rgba(234, 51, 81, 0.18), rgba(234, 51, 81, 0.03) 18%, transparent 44%), linear-gradient(180deg, #070c18 0%, #0a1020 100%)"
    )
    st.markdown(
        """
        <style>
          :root {
            --mvv-red: #c8102e;
            --mvv-red-bright: #ea3351;
            --mvv-navy: #0b1020;
            --mvv-panel: #12192a;
            --mvv-text: #f8fafc;
            --mvv-muted: rgba(226, 232, 240, 0.74);
          }

          .stApp {
            background: __APP_BACKGROUND__;
            background-size: cover;
            background-position: center top;
            background-attachment: fixed;
            color: var(--mvv-text);
          }

          [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(16, 23, 38, 0.98), rgba(9, 13, 23, 0.98));
            border-right: 1px solid rgba(255,255,255,0.06);
          }

          [data-testid="stSidebar"] .stMarkdown,
          [data-testid="stSidebar"] label,
          [data-testid="stSidebar"] div {
            color: var(--mvv-text);
          }

          .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1380px;
          }

          .home-hero-shell {
            display: flex;
            flex-direction: column;
            gap: 1.1rem;
            margin-bottom: 1.55rem;
          }

          .home-hero {
            min-height: 320px;
            padding: 2rem 1.75rem 1.9rem 1.75rem;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.08);
            background: linear-gradient(135deg, rgba(18, 25, 42, 0.88), rgba(10, 15, 27, 0.84));
            box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22);
          }

          .home-hero-logo {
            width: 82px;
            height: 82px;
            object-fit: contain;
            margin-bottom: 0.9rem;
            filter: drop-shadow(0 8px 22px rgba(0,0,0,0.28));
          }

          .home-kicker {
            color: rgba(255,255,255,0.76);
            font-size: 0.74rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            margin-bottom: 0.35rem;
          }

          .home-title {
            margin: 0;
            font-size: 2.55rem;
            line-height: 1;
            font-weight: 800;
          }

          .home-copy {
            margin-top: 0.8rem;
            max-width: 74ch;
            color: rgba(255,255,255,0.84);
            line-height: 1.62;
          }

          .home-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
          }

          .home-pill {
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

          .home-summary-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
          }

          .home-summary-card {
            min-height: 120px;
            padding: 1rem 1.05rem 0.95rem 1.05rem;
            border-radius: 8px;
            border: 1px solid rgba(234, 51, 81, 0.14);
            background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
          }

          .home-summary-label {
            color: rgba(255,255,255,0.68);
            font-size: 0.8rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
          }

          .home-summary-value {
            margin-top: 0.55rem;
            font-size: 1.95rem;
            line-height: 1.1;
            font-weight: 800;
            color: #ffffff;
            word-break: break-word;
          }

          .home-summary-foot {
            margin-top: 0.65rem;
            color: rgba(255,255,255,0.8);
            font-size: 0.86rem;
            line-height: 1.4;
          }

          .home-section-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 1rem;
            margin-bottom: 0.95rem;
          }

          .home-section-kicker {
            color: rgba(255,255,255,0.62);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
          }

          .home-section-title {
            margin-top: 0.25rem;
            color: #ffffff;
            font-size: 1.1rem;
            font-weight: 700;
          }

          .home-section-note {
            color: rgba(255,255,255,0.8);
            font-size: 0.88rem;
            font-weight: 700;
            text-align: right;
          }

          .home-module-thumb {
            position: relative;
            aspect-ratio: 16 / 10;
            display: flex;
            align-items: flex-end;
            padding: 1.05rem;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 14px 26px rgba(0,0,0,0.2);
            background-size: cover;
            background-position: center;
          }

          .home-module-thumb::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, rgba(7,12,24,0.02), rgba(7,12,24,0.9));
          }

          .home-module-badge {
            position: absolute;
            top: 0.85rem;
            left: 0.85rem;
            z-index: 1;
            padding: 0.38rem 0.65rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 800;
            color: #ffffff;
            border: 1px solid rgba(255,255,255,0.1);
            background: rgba(11,16,29,0.74);
            backdrop-filter: blur(10px);
          }

          .home-module-overlay {
            position: relative;
            z-index: 1;
          }

          .home-module-name {
            font-size: 1.55rem;
            font-weight: 800;
            color: #ffffff;
            line-height: 1.06;
          }

          .home-module-copy {
            margin-top: 0.38rem;
            max-width: 28ch;
            color: rgba(255,255,255,0.88);
            line-height: 1.48;
            font-size: 0.92rem;
          }

          .home-module-gap {
            height: 0.7rem;
          }

          .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(234, 51, 81, 0.24);
            background: linear-gradient(135deg, rgba(234, 51, 81, 0.18), rgba(200, 16, 46, 0.28));
            color: #ffffff;
            font-weight: 800;
            min-height: 2.8rem;
            box-shadow: 0 10px 22px rgba(0,0,0,0.14);
          }

          .stButton > button:hover {
            border-color: rgba(234, 51, 81, 0.38);
            background: linear-gradient(135deg, rgba(234, 51, 81, 0.24), rgba(200, 16, 46, 0.36));
          }

          .stButton > button:disabled {
            background: rgba(255,255,255,0.04);
            border-color: rgba(255,255,255,0.08);
            color: rgba(255,255,255,0.48);
          }

          [data-testid="stTextInputRootElement"] {
            background: rgba(11, 16, 29, 0.88);
            border-radius: 8px;
          }

          @media (max-width: 1100px) {
            .home-summary-grid {
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }
          }

          @media (max-width: 768px) {
            .home-hero {
              min-height: auto;
              padding: 1.55rem 1rem;
            }

            .home-title {
              font-size: 2rem;
            }

            .home-summary-grid {
              grid-template-columns: 1fr;
            }

            .home-section-head {
              flex-direction: column;
              align-items: flex-start;
            }

            .home-section-note {
              text-align: left;
            }

            .home-module-thumb {
              aspect-ratio: 16 / 11;
            }
          }
        </style>
        """.replace("__APP_BACKGROUND__", app_background),
        unsafe_allow_html=True,
    )


def login_ui() -> None:
    maintenance_banner()
    st.markdown("## Inloggen")

    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pw")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if not submitted:
        return

    try:
        res = get_sb().auth.sign_in_with_password({"email": email, "password": password})
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
        st.session_state.pop("_profile_cache", None)
        st.session_state.pop("role", None)
        st.session_state.pop("player_id", None)
        st.rerun()

    except Exception as exc:
        st.error(f"Sign in mislukt: {exc}")


def logout_button() -> None:
    if st.button("Logout", use_container_width=True, key="btn_logout"):
        try:
            get_sb().auth.sign_out()
        except Exception:
            pass
        clear_tokens_in_cookie()
        st.session_state.clear()
        st.rerun()


def modules_for_role(role: str) -> list[dict]:
    role = (role or "").lower()
    return [module for module in MODULES if role in module["roles"]]


def render_home_hero(role: str, email: str, modules: list[dict]) -> None:
    logo_uri = build_image_data_uri(TEAM_LOGO) if TEAM_LOGO.exists() else ""
    logo_markup = f'<img src="{logo_uri}" alt="MVV Maastricht" class="home-hero-logo" />' if logo_uri else ""
    available_count = sum(1 for module in modules if module.get("target_page") and not module.get("disabled"))
    disabled_count = len(modules) - available_count
    role_label = "Staff" if role == "staff" else "Speler"
    summary_cards = [
        ("Rol", role_label, "Actieve toegangslaag voor deze sessie"),
        ("Modules", str(len(modules)), "Onderdelen zichtbaar op deze startpagina"),
        ("Beschikbaar", str(available_count), "Direct te openen modules voor vandaag"),
        ("Account", email or "--", f"{disabled_count} modules staan nog in opbouw"),
    ]
    summary_markup = "".join(
        f"""
        <div class="home-summary-card">
          <div class="home-summary-label">{label}</div>
          <div class="home-summary-value">{value}</div>
          <div class="home-summary-foot">{foot}</div>
        </div>
        """
        for label, value, foot in summary_cards
    )

    st.markdown(
        f"""
        <div class="home-hero-shell">
          <div class="home-hero">
            {logo_markup}
            <div class="home-kicker">MVV Maastricht | Dashboard | Seizoensoverzicht</div>
            <h1 class="home-title">MVV Dashboard</h1>
            <div class="home-copy">
              Centrale ingang voor de performance-omgeving van MVV Maastricht. Open snel de juiste module voor
              player monitoring, GPS, rapportage en dagelijkse stafworkflow.
            </div>
            <div class="home-pill-row">
              <span class="home-pill">Zelfde MVV-stijl als de beta-pagina's</span>
              <span class="home-pill">Directe toegang per rol en workflow</span>
            </div>
          </div>
          <div class="home-summary-grid">
            {summary_markup}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_module_tile(module: dict) -> None:
    image_uri = "" if SAFE_MODE else build_image_data_uri(module["img_path"])
    thumb_style = (
        f"background-image: linear-gradient(135deg, rgba(10, 15, 27, 0.08), rgba(10, 15, 27, 0.04)), url('{image_uri}');"
        if image_uri
        else "background: linear-gradient(135deg, rgba(234, 51, 81, 0.16), rgba(10, 15, 27, 0.92));"
    )
    badge = module.get("badge") or "Module"
    st.markdown(
        f"""
        <div class="home-module-thumb" style="{thumb_style}">
          <div class="home-module-badge">{badge}</div>
          <div class="home-module-overlay">
            <div class="home-module-name">{module['title']}</div>
            <div class="home-module-copy">{module['description']}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="home-module-gap"></div>', unsafe_allow_html=True)

    if module.get("disabled"):
        st.button("Geen toegang", use_container_width=True, disabled=True, key=f"btn_{module['id']}_disabled")
        return

    if module.get("target_page") is None:
        st.button("Binnenkort", use_container_width=True, disabled=True, key=f"btn_{module['id']}_soon")
        return

    if st.button("Open module", use_container_width=True, key=f"btn_{module['id']}_open"):
        st.switch_page(module["target_page"])


def render_module_grid(modules: list[dict], cols_per_row: int) -> None:
    rows = [modules[i:i + cols_per_row] for i in range(0, len(modules), cols_per_row)]
    for row_index, row_modules in enumerate(rows):
        cols = st.columns(cols_per_row, gap="large")
        for col, module in zip(cols, row_modules):
            with col:
                render_module_tile(module)
        if row_index < len(rows) - 1:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


if DIAG_MODE:
    st.title("DIAG OK")
    st.write("Als je dit ziet, werkt Streamlit op dit toestel/netwerk.")
    st.write("Zet diag uit door ?diag=0 of verwijder de query parameter.")
    st.stop()


render_home_css()

sb = get_sb()
if sb is None:
    st.error("Supabase client niet beschikbaar (secrets ontbreken of create_client faalt).")
    st.stop()


try:
    cm = cookie_mgr()
    _ = cm.get("sb_refresh")
except Exception:
    pass


if "access_token" not in st.session_state:
    restored = try_restore_or_refresh_session(sb)
    if not restored:
        login_ui()
        st.stop()


profile = get_profile(sb)
if not profile:
    clear_tokens_in_cookie()
    st.session_state.clear()
    st.error("Geen profiel gevonden of geen rechten. Log opnieuw in.")
    st.stop()


role = str(st.session_state.get("role") or profile.get("role") or "").lower()
st.sidebar.success(f"Ingelogd: {st.session_state.get('user_email', '')}")
st.sidebar.info(f"Role: {role}")
with st.sidebar:
    logout_button()

if DIAG_MODE:
    with st.sidebar.expander("Auth debug", expanded=True):
        cm = cookie_mgr()
        st.write("session access:", bool(st.session_state.get("access_token")))
        st.write("cookie access:", bool(cm.get("sb_access")))
        st.write("cookie refresh:", bool(cm.get("sb_refresh")))
        st.write("auth_err:", st.session_state.get("auth_err"))

maintenance_banner()

if SAFE_MODE:
    st.warning("Safe mode actief (minimale UI). Zet uit door ?safe=0 te gebruiken.")

home_modules = modules_for_role(role)
render_home_hero(role, st.session_state.get("user_email", ""), home_modules)

available_count = sum(1 for module in home_modules if module.get("target_page") and not module.get("disabled"))
st.markdown(
    f"""
    <div class="home-section-head">
      <div>
        <div class="home-section-kicker">Modules</div>
        <div class="home-section-title">Kies de juiste werkruimte voor je volgende actie</div>
      </div>
      <div class="home-section-note">{available_count} direct beschikbaar</div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_module_grid(home_modules, 2 if role == "player" else 3)
