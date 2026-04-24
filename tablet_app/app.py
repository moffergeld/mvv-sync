from __future__ import annotations

import base64
import html
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import extra_streamlit_components as stx
import streamlit as st
from supabase import create_client


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
PAGES_DIR = ROOT_DIR / "pages"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PAGES_DIR))

from Subscripts.player_tab_forms import (  # noqa: E402
    _legend_asrm,
    _legend_rpe,
    load_asrm,
    load_rpe,
    save_asrm as original_save_asrm,
    save_rpe as original_save_rpe,
)


APP_TITLE = "MVV Tablet Check-in"
CLUB_NAME = "MVV Maastricht"
ACCESS_COOKIE_NAME = "mvv_tablet_access"
ACCESS_COOKIE_SECONDS = 60 * 60 * 24 * 30


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;700;800;900&display=swap');

      :root {
        --mvv-red: #c8102e;
        --mvv-dark-red: #8f0b20;
        --mvv-deep: #14070a;
        --mvv-cream: #fff7ef;
        --mvv-soft: #f7e9e7;
        --mvv-gold: #d6a94a;
        --mvv-border: rgba(200, 16, 46, 0.20);
      }

      #MainMenu,
      header,
      footer,
      [data-testid="stSidebar"],
      [data-testid="collapsedControl"] {
        display: none !important;
      }

      .stApp {
        background:
          radial-gradient(circle at top left, rgba(200, 16, 46, 0.22), transparent 34rem),
          linear-gradient(135deg, #fff9f5 0%, #f7eeee 48%, #ffffff 100%);
      }

      .block-container {
        max-width: 1240px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        font-family: 'Inter', sans-serif;
      }

      h1, h2, h3 {
        letter-spacing: 0.01em;
      }

      div.stButton > button {
        min-height: 5.25rem;
        border-radius: 18px;
        border: 1px solid var(--mvv-border);
        background: linear-gradient(145deg, #ffffff 0%, #fff4f1 100%);
        box-shadow: 0 12px 24px rgba(78, 8, 18, 0.10);
        color: var(--mvv-deep);
        font-size: 1rem;
        font-weight: 900;
        line-height: 1.35;
        white-space: pre-line;
        transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease;
      }

      div.stButton > button:hover {
        transform: translateY(-2px);
        border-color: rgba(200, 16, 46, 0.55);
        box-shadow: 0 16px 32px rgba(78, 8, 18, 0.16);
      }

      div.stButton > button:active {
        transform: translateY(0px);
      }

      div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(200, 16, 46, 0.14);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 26px rgba(78, 8, 18, 0.08);
      }

      div[data-testid="stMetricValue"],
      div[data-testid="stMetricValue"] > div,
      div[data-testid="stMetricValue"] p,
      div[data-testid="stMetricValue"] span,
      [data-testid="stMetric"] [data-testid="stMetricValue"],
      [data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: var(--mvv-red) !important;
        font-size: 2.15rem;
        font-weight: 900 !important;
      }

      div[data-testid="stMetricLabel"] p {
        font-weight: 800;
        color: rgba(20, 7, 10, 0.72);
      }

      .tablet-hero {
        position: relative;
        overflow: hidden;
        display: flex;
        gap: 1rem;
        align-items: center;
        padding: 1.35rem 1.45rem;
        border: 1px solid rgba(255, 255, 255, 0.42);
        border-radius: 24px;
        background:
          linear-gradient(135deg, rgba(200, 16, 46, 0.96) 0%, rgba(143, 11, 32, 0.96) 54%, rgba(20, 7, 10, 0.96) 100%);
        color: white;
        box-shadow: 0 18px 46px rgba(78, 8, 18, 0.24);
        margin-bottom: 1.1rem;
      }

      .tablet-hero::after {
        content: '';
        position: absolute;
        right: -5rem;
        top: -6rem;
        width: 18rem;
        height: 18rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.10);
      }

      .mvv-logo-wrap {
        z-index: 1;
        width: 72px;
        height: 72px;
        min-width: 72px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.92);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.4), 0 12px 24px rgba(0,0,0,0.16);
      }

      .mvv-logo-wrap img {
        max-width: 58px;
        max-height: 58px;
        object-fit: contain;
      }

      .mvv-logo-fallback {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.65rem;
        color: var(--mvv-red);
        letter-spacing: 0.02em;
      }

      .tablet-hero-content {
        z-index: 1;
      }

      .tablet-hero-kicker {
        margin: 0 0 0.2rem 0;
        text-transform: uppercase;
        font-size: 0.78rem;
        font-weight: 900;
        letter-spacing: 0.16em;
        color: rgba(255, 255, 255, 0.78) !important;
      }

      .tablet-hero-title {
        margin: 0;
        font-family: 'Bebas Neue', sans-serif;
        font-size: clamp(2.4rem, 5vw, 4.2rem);
        line-height: 0.95;
        letter-spacing: 0.02em;
        color: #ffffff !important;
      }

      .tablet-hero-subtitle {
        margin: 0.45rem 0 0 0;
        font-size: 1.02rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.88) !important;
      }

      .mvv-section-card {
        padding: 1rem 1.1rem;
        border-radius: 20px;
        border: 1px solid rgba(200, 16, 46, 0.16);
        background: rgba(255, 255, 255, 0.84);
        box-shadow: 0 12px 30px rgba(78, 8, 18, 0.08);
        margin: 0.6rem 0 1rem 0;
      }


      .mvv-kpi-card {
        padding: 1rem 1.1rem;
        border-radius: 20px;
        border: 1px solid rgba(200, 16, 46, 0.12);
        background: rgba(255, 255, 255, 0.88);
        box-shadow: 0 12px 30px rgba(78, 8, 18, 0.08);
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin-bottom: 0.6rem;
      }

      .mvv-kpi-label {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--mvv-deep) !important;
        margin-bottom: 0.25rem;
      }

      .mvv-kpi-value {
        font-size: 2.25rem;
        line-height: 1;
        font-weight: 900;
        color: var(--mvv-red) !important;
      }

      .tablet-hero a,
      .tablet-hero a:visited,
      .tablet-hero a:hover,
      .tablet-hero a:active {
        color: #ffffff !important;
        text-decoration: none !important;
      }

      .tablet-hero svg,
      .tablet-hero [data-testid="stHeaderActionElements"] {
        display: none !important;
      }

      .mvv-note {
        padding: 0.9rem 1rem;
        border-radius: 16px;
        border-left: 6px solid var(--mvv-red);
        background: rgba(255, 255, 255, 0.80);
        font-weight: 800;
      }

      [data-testid="stTextInput"] input,
      [data-testid="stNumberInput"] input,
      textarea {
        border-radius: 14px !important;
      }

      .stApp [data-testid="stTextInput"] input,
      .stApp [data-testid="stNumberInput"] input,
      .stApp [data-baseweb="select"] > div,
      .stApp textarea {
        background: #ffffff !important;
        color: var(--mvv-deep) !important;
      }

      .stApp [data-baseweb="select"] * {
        color: var(--mvv-deep) !important;
      }

      .stApp [data-testid="stTextInput"] input,
      .stApp [data-testid="stNumberInput"] input,
      .stApp [data-baseweb="select"] > div,
      .stApp textarea {
        border: 1px solid rgba(200, 16, 46, 0.18) !important;
        box-shadow: none !important;
      }

      div[role="radiogroup"] label {
        background: rgba(255,255,255,0.80);
        border: 1px solid rgba(200,16,46,0.16);
        border-radius: 999px;
        padding: 0.45rem 0.8rem;
        margin-right: 0.45rem;
        font-weight: 900;
      }


      /* Force readable text even when Streamlit/account dark mode is enabled */
      .stApp,
      .stApp label,
      .stApp .stMarkdown,
      .stApp .stMarkdown p,
      .stApp .stMarkdown li,
      .stApp .stMarkdown span,
      .stApp .stMarkdown strong,
      .stApp [data-testid="stMarkdownContainer"] p,
      .stApp [data-testid="stMarkdownContainer"] li,
      .stApp [data-testid="stMetricLabel"] p,
      .stApp [data-testid="stTextInputRootElement"] label,
      .stApp [data-testid="stNumberInput"] label,
      .stApp [data-testid="stSelectbox"] label,
      .stApp [data-testid="stTextArea"] label,
      .stApp [data-testid="stSlider"] label,
      .stApp [data-testid="stRadio"] label,
      .stApp [data-testid="stToggle"] label,
      .stApp [data-testid="stWidgetLabel"],
      .stApp [data-testid="stCaptionContainer"] {
        color: var(--mvv-deep) !important;
      }

      .stApp [data-testid="stMetricLabel"] {
        opacity: 1 !important;
      }

      .stApp input::placeholder,
      .stApp textarea::placeholder {
        color: rgba(20, 7, 10, 0.62) !important;
      }

      .mvv-section-card,
      .mvv-note {
        color: var(--mvv-deep) !important;
      }

      .tablet-hero,
      .tablet-hero *,
      .mvv-logo-wrap,
      .mvv-logo-wrap * {
        color: white !important;
      }

      .tablet-hero-content,
      .tablet-hero-content *,
      .tablet-hero-title,
      .tablet-hero-subtitle,
      .tablet-hero-kicker {
        color: #ffffff !important;
      }

      .mvv-logo-fallback {
        color: var(--mvv-red) !important;
      }

      div.stFormSubmitButton > button {
        background: linear-gradient(135deg, #182033 0%, #0f172a 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(15, 23, 42, 0.45) !important;
      }

      div.stFormSubmitButton > button * {
        color: #ffffff !important;
      }

      @media (max-width: 768px) {
        .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
        .tablet-hero { border-radius: 20px; padding: 1rem; }
        .mvv-logo-wrap { width: 58px; height: 58px; min-width: 58px; border-radius: 16px; }
        div.stButton > button { min-height: 4.8rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def amsterdam_today():
    return datetime.now(ZoneInfo("Europe/Amsterdam")).date()


def cookie_mgr():
    if "_tablet_cookie_mgr" not in st.session_state:
        st.session_state["_tablet_cookie_mgr"] = stx.CookieManager(key="tablet_cookie_mgr")
    return st.session_state["_tablet_cookie_mgr"]


def logo_html() -> str:
    """Use an MVV logo file when present; otherwise show a clean MVV fallback mark."""
    candidates = [
        ROOT_DIR / "assets" / "mvv-logo.png",
        ROOT_DIR / "assets" / "mvv_logo.png",
        ROOT_DIR / "assets" / "mvv.png",
        THIS_DIR / "assets" / "mvv-logo.png",
        THIS_DIR / "assets" / "mvv_logo.png",
        THIS_DIR / "assets" / "mvv.png",
    ]
    for path in candidates:
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            return f'<img src="data:image/png;base64,{encoded}" alt="MVV Maastricht logo" />'
    return '<div class="mvv-logo-fallback">MVV</div>'


def render_hero(title: str, subtitle: str, kicker: str = CLUB_NAME) -> None:
    st.markdown(
        f"""
        <div class="tablet-hero">
          <div class="mvv-logo-wrap">{logo_html()}</div>
          <div class="tablet-hero-content">
            <div class="tablet-hero-kicker">{html.escape(kicker)}</div>
            <div class="tablet-hero-title">{html.escape(title)}</div>
            <div class="tablet-hero-subtitle">{html.escape(subtitle)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="mvv-kpi-card"><div class="mvv-kpi-label">{html.escape(str(label))}</div><div class="mvv-kpi-value">{html.escape(str(value))}</div></div>',
        unsafe_allow_html=True,
    )


def get_tablet_code() -> str:
    for key in ("TABLET_SHARED_CODE", "TABLET_CODE", "KIOSK_CODE"):
        value = str(st.secrets.get(key, "") or "").strip()
        if value:
            return value
    return ""


def get_service_key() -> str:
    for key in ("SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_SERVICE_KEY", "SUPABASE_SERVICE_ROLE"):
        value = str(st.secrets.get(key, "") or "").strip()
        if value:
            return value
    return ""


def get_tablet_sb():
    url = str(st.secrets.get("SUPABASE_URL", "") or "").strip()
    service_key = get_service_key()
    if not url or not service_key:
        return None

    if "_tablet_sb_client" not in st.session_state or st.session_state.get("_tablet_sb_client") is None:
        st.session_state["_tablet_sb_client"] = create_client(url, service_key)

    return st.session_state["_tablet_sb_client"]




def get_tablet_created_by(player_id: str = "") -> str:
    """
    Tablet/kiosk mode heeft geen ingelogde Supabase-user, terwijl de database
    created_by verplicht maakt. Zet bij voorkeur TABLET_CREATED_BY_USER_ID in
    Streamlit secrets met de UUID van een admin/system-user. Zonder die secret
    gebruikt de app player_id als UUID-fallback.
    """
    for key in (
        "TABLET_CREATED_BY_USER_ID",
        "TABLET_CREATED_BY",
        "CHECKIN_CREATED_BY_USER_ID",
        "DEFAULT_CREATED_BY_USER_ID",
    ):
        value = str(st.secrets.get(key, "") or "").strip()
        if value:
            return value
    return str(player_id or "").strip()


def _execute_upsert_with_fallback(sb, table_name: str, payload: Dict[str, Any], keys: Dict[str, Any]):
    try:
        return sb.table(table_name).upsert(payload, on_conflict=",".join(keys.keys())).execute()
    except Exception:
        existing = sb.table(table_name).select("id")
        for key, value in keys.items():
            existing = existing.eq(key, value)
        rows = existing.limit(1).execute().data or []
        if rows:
            row_id = rows[0].get("id")
            update_payload = dict(payload)
            update_payload.pop("id", None)
            update_payload.pop("created_by", None)
            return sb.table(table_name).update(update_payload).eq("id", row_id).execute()
        return sb.table(table_name).insert(payload).execute()


def save_asrm_tablet(sb, player_id, entry_date, muscle_soreness, fatigue, sleep_quality, stress, mood) -> None:
    entry_date_iso = entry_date.isoformat() if hasattr(entry_date, "isoformat") else str(entry_date)
    now_iso = datetime.now(ZoneInfo("UTC")).isoformat()
    payload = {
        "player_id": str(player_id),
        "entry_date": entry_date_iso,
        "muscle_soreness": int(muscle_soreness),
        "fatigue": int(fatigue),
        "sleep_quality": int(sleep_quality),
        "stress": int(stress),
        "mood": int(mood),
        "created_by": get_tablet_created_by(str(player_id)),
        "updated_at": now_iso,
    }
    _execute_upsert_with_fallback(
        sb,
        "asrm_entries",
        payload,
        {"player_id": str(player_id), "entry_date": entry_date_iso},
    )


def save_rpe_tablet(sb, player_id: str, entry_date, injury: bool, injury_type: str | None, injury_pain: int | None, notes: str, sessions: List[Dict[str, int]]) -> None:
    try:
        original_save_rpe(
            sb,
            player_id=player_id,
            entry_date=entry_date,
            injury=injury,
            injury_type=injury_type,
            injury_pain=injury_pain,
            notes=notes,
            sessions=sessions,
        )
        return
    except Exception as exc:
        # De originele save_rpe kan falen in tablet-mode door created_by/auth.uid(),
        # of doordat oudere code een kolom gebruikt die niet in jouw rpe_sessions schema staat.
        # In die gevallen slaan we hieronder handmatig op volgens de actuele database-tabellen.
        if "created_by" not in str(exc) and "training_load" not in str(exc) and "schema cache" not in str(exc):
            raise

    entry_date_iso = entry_date.isoformat() if hasattr(entry_date, "isoformat") else str(entry_date)
    now_iso = datetime.now(ZoneInfo("UTC")).isoformat()
    header_payload = {
        "player_id": str(player_id),
        "entry_date": entry_date_iso,
        "injury": bool(injury),
        "injury_type": injury_type,
        "injury_pain": injury_pain,
        "notes": str(notes or ""),
        "created_by": get_tablet_created_by(str(player_id)),
        "updated_at": now_iso,
    }

    existing = (
        sb.table("rpe_entries")
        .select("id")
        .eq("player_id", str(player_id))
        .eq("entry_date", entry_date_iso)
        .limit(1)
        .execute()
        .data
        or []
    )

    if existing:
        rpe_entry_id = existing[0].get("id")
        update_payload = dict(header_payload)
        update_payload.pop("created_by", None)
        sb.table("rpe_entries").update(update_payload).eq("id", rpe_entry_id).execute()
    else:
        inserted = sb.table("rpe_entries").insert(header_payload).execute().data or []
        rpe_entry_id = inserted[0].get("id") if inserted else None

    if not rpe_entry_id:
        raise RuntimeError("RPE-header kon niet worden opgeslagen.")

    try:
        sb.table("rpe_sessions").delete().eq("rpe_entry_id", rpe_entry_id).execute()
        session_fk = "rpe_entry_id"
    except Exception:
        sb.table("rpe_sessions").delete().eq("rpe_id", rpe_entry_id).execute()
        session_fk = "rpe_id"

    for session in sessions:
        duration_min = int(session.get("duration_min", 0) or 0)
        rpe_value = int(session.get("rpe", 0) or 0)
        session_payload = {
            session_fk: rpe_entry_id,
            "session_index": int(session.get("session_index", 1) or 1),
            "duration_min": duration_min,
            "rpe": rpe_value,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        try:
            sb.table("rpe_sessions").insert(session_payload).execute()
        except Exception as exc:
            # Sommige rpe_sessions-tabellen hebben geen created_at/updated_at.
            # Dan opnieuw proberen met alleen de noodzakelijke kolommen.
            if "schema cache" not in str(exc) and "Could not find" not in str(exc):
                raise
            minimal_payload = {
                session_fk: rpe_entry_id,
                "session_index": int(session.get("session_index", 1) or 1),
                "duration_min": duration_min,
                "rpe": rpe_value,
            }
            sb.table("rpe_sessions").insert(minimal_payload).execute()

def grant_tablet_access() -> None:
    cm = cookie_mgr()
    cm.set(ACCESS_COOKIE_NAME, "1", max_age=ACCESS_COOKIE_SECONDS, key="tablet_access_set")
    st.session_state["tablet_unlocked"] = True
    time.sleep(0.10)
    st.rerun()


def lock_tablet() -> None:
    cm = cookie_mgr()
    cm.set(ACCESS_COOKIE_NAME, "", max_age=1, key="tablet_access_clear")
    for key in (
        "tablet_unlocked",
        "tablet_player_id",
        "tablet_player_name",
        "tablet_active_form",
        "tablet_flash",
    ):
        st.session_state.pop(key, None)
    time.sleep(0.10)
    st.rerun()


def ensure_tablet_access() -> None:
    shared_code = get_tablet_code()
    if not shared_code:
        st.error("Tabletcode ontbreekt in Streamlit secrets. Voeg TABLET_SHARED_CODE toe.")
        st.stop()

    cm = cookie_mgr()
    _ = cm.get(ACCESS_COOKIE_NAME)

    if st.session_state.get("tablet_unlocked"):
        return

    query_code = str(st.query_params.get("code", "") or "").strip()
    if query_code and query_code == shared_code:
        grant_tablet_access()

    cookie_value = str(cm.get(ACCESS_COOKIE_NAME) or "").strip()
    if cookie_value == "1":
        st.session_state["tablet_unlocked"] = True
        return

    render_hero("Tablet toegang", "Voer de teamcode in om de check-in pagina te openen.")

    with st.form("tablet_unlock_form", clear_on_submit=False):
        code_value = st.text_input("Tabletcode", type="password")
        submitted = st.form_submit_button("Open tablet", use_container_width=True)

    if submitted:
        if code_value.strip() == shared_code:
            grant_tablet_access()
        else:
            st.error("Code klopt niet.")

    st.stop()


@st.cache_data(show_spinner=False, ttl=120)
def fetch_active_players(_sb) -> List[Dict[str, str]]:
    try:
        rows = (
            _sb.table("players")
            .select("player_id,full_name")
            .eq("is_active", True)
            .order("full_name")
            .execute()
            .data
            or []
        )
    except Exception:
        rows = []

    out: List[Dict[str, str]] = []
    for row in rows:
        player_id = row.get("player_id")
        full_name = str(row.get("full_name") or "").strip()
        if player_id and full_name:
            out.append({"player_id": str(player_id), "full_name": full_name})
    return out


@st.cache_data(show_spinner=False, ttl=30)
def fetch_daily_completion(_sb, entry_date_iso: str) -> Dict[str, List[str]]:
    try:
        asrm_rows = (
            _sb.table("asrm_entries")
            .select("player_id")
            .eq("entry_date", entry_date_iso)
            .execute()
            .data
            or []
        )
    except Exception:
        asrm_rows = []

    try:
        rpe_rows = (
            _sb.table("rpe_entries")
            .select("player_id")
            .eq("entry_date", entry_date_iso)
            .execute()
            .data
            or []
        )
    except Exception:
        rpe_rows = []

    asrm_ids = sorted({str(row.get("player_id")) for row in asrm_rows if row.get("player_id")})
    rpe_ids = sorted({str(row.get("player_id")) for row in rpe_rows if row.get("player_id")})
    return {"asrm_ids": asrm_ids, "rpe_ids": rpe_ids}


def clear_daily_cache() -> None:
    fetch_daily_completion.clear()


def show_flash() -> None:
    flash = st.session_state.pop("tablet_flash", None)
    if flash:
        st.success(str(flash))


def render_top_actions(show_back: bool = False) -> None:
    # Geen navigatieknoppen bovenaan de tabletpagina's.
    return

def render_player_picker(sb) -> None:
    entry_date = amsterdam_today()
    players = fetch_active_players(sb)
    if not players:
        st.warning("Geen actieve spelers gevonden.")
        return

    completion = fetch_daily_completion(sb, entry_date.isoformat())
    asrm_ids = set(completion.get("asrm_ids", []))
    rpe_ids = set(completion.get("rpe_ids", []))

    render_top_actions(show_back=False)
    show_flash()

    render_hero(
        APP_TITLE,
        f"Selecteer een speler voor de invoer van vandaag ({entry_date.strftime('%d-%m-%Y')}).",
    )

    stat_1, stat_2, stat_3 = st.columns(3)
    with stat_1:
        render_kpi_card("Actieve spelers", len(players))
    with stat_2:
        render_kpi_card("Wellness ingevuld", sum(1 for player in players if player["player_id"] in asrm_ids))
    with stat_3:
        render_kpi_card("RPE ingevuld", sum(1 for player in players if player["player_id"] in rpe_ids))

    search = st.text_input("Zoek speler", value="", placeholder="Typ een naam")
    search_value = search.strip().lower()
    filtered_players = [
        player for player in players
        if not search_value or search_value in player["full_name"].lower()
    ]

    if not filtered_players:
        st.info("Geen speler gevonden.")
        return

    st.markdown('<div class="mvv-section-card"><b>Spelerslijst</b><br>Klik op een speler. Als wellness al is ingevuld, opent RPE automatisch.</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for idx, player in enumerate(filtered_players):
        player_id = player["player_id"]
        player_name = player["full_name"]
        wellness_done = player_id in asrm_ids
        rpe_done = player_id in rpe_ids
        wellness_state = "OK" if wellness_done else "OPEN"
        rpe_state = "OK" if rpe_done else "OPEN"
        next_step = "Open RPE" if wellness_done and not rpe_done else "Open wellness"
        if wellness_done and rpe_done:
            next_step = "Controleer invoer"
        label = f"{player_name}\nWellness: {wellness_state} | RPE: {rpe_state}\n{next_step}"

        with cols[idx % 3]:
            if st.button(label, use_container_width=True, key=f"tablet_pick_{player_id}"):
                st.session_state["tablet_player_id"] = player_id
                st.session_state["tablet_player_name"] = player_name
                # Belangrijk: als wellness vandaag bestaat, start direct op RPE.
                st.session_state["tablet_active_form"] = "RPE" if wellness_done else "Wellness"
                st.rerun()


def _session_value(rpe_sessions: List[Dict[str, Any]], idx: int, key: str, default: int) -> int:
    hit = next((row for row in rpe_sessions if int(row.get("session_index", 0) or 0) == idx), None)
    if not hit:
        return default
    value = hit.get(key)
    return int(value) if value is not None else default


def render_player_forms(sb, player_id: str, player_name: str) -> None:
    entry_date = amsterdam_today()
    existing_asrm = load_asrm(sb, player_id, entry_date) or {}
    rpe_header, rpe_sessions = load_rpe(sb, player_id, entry_date)
    rpe_header = rpe_header or {}
    rpe_sessions = rpe_sessions or []

    has_wellness = bool(existing_asrm)
    has_rpe = bool(rpe_header)
    has_s2 = any(int(row.get("session_index", 0) or 0) == 2 for row in rpe_sessions)
    injury_default = bool(rpe_header.get("injury", False))

    if "tablet_active_form" not in st.session_state:
        st.session_state["tablet_active_form"] = "RPE" if has_wellness else "Wellness"

    render_top_actions(show_back=True)
    show_flash()

    render_hero(
        player_name,
        f"Invoer voor vandaag ({entry_date.strftime('%d-%m-%Y')}).",
        kicker=f"{CLUB_NAME} · speler check-in",
    )

    status_1, status_2 = st.columns(2)
    with status_1:
        render_kpi_card("Wellness", "OK" if has_wellness else "Open")
    with status_2:
        render_kpi_card("RPE", "OK" if has_rpe else "Open")

    form_options = ["Wellness", "RPE"]
    default_form = st.session_state.get("tablet_active_form", "RPE" if has_wellness else "Wellness")
    if default_form not in form_options:
        default_form = "Wellness"

    active_form = st.radio(
        "Onderdeel",
        options=form_options,
        index=form_options.index(default_form),
        horizontal=True,
        key=f"tablet_form_selector_{player_id}",
    )
    st.session_state["tablet_active_form"] = active_form

    if active_form == "Wellness":
        st.markdown('<div class="mvv-section-card">', unsafe_allow_html=True)
        if has_wellness:
            st.success("Wellness staat al ingevuld voor vandaag.")
        else:
            st.info("Wellness staat nog open voor vandaag.")

        with st.form(f"tablet_asrm_form_{player_id}", clear_on_submit=False):
            _legend_asrm()

            ms = st.slider(
                "Muscle soreness (1-10)",
                1,
                10,
                value=int(existing_asrm.get("muscle_soreness", 5)),
                key=f"tablet_asrm_ms_{player_id}",
            )
            fat = st.slider(
                "Fatigue (1-10)",
                1,
                10,
                value=int(existing_asrm.get("fatigue", 5)),
                key=f"tablet_asrm_fat_{player_id}",
            )
            sleep = st.slider(
                "Sleep quality (1-10)",
                1,
                10,
                value=int(existing_asrm.get("sleep_quality", 5)),
                key=f"tablet_asrm_sleep_{player_id}",
            )
            stress = st.slider(
                "Stress (1-10)",
                1,
                10,
                value=int(existing_asrm.get("stress", 5)),
                key=f"tablet_asrm_stress_{player_id}",
            )
            mood = st.slider(
                "Mood (1-10)",
                1,
                10,
                value=int(existing_asrm.get("mood", 5)),
                key=f"tablet_asrm_mood_{player_id}",
            )

            asrm_submit = st.form_submit_button("Wellness opslaan", use_container_width=True)

        if asrm_submit:
            try:
                save_asrm_tablet(sb, player_id, entry_date, ms, fat, sleep, stress, mood)
                clear_daily_cache()
                st.session_state["tablet_active_form"] = "RPE"
                st.session_state["tablet_flash"] = f"Wellness opgeslagen voor {player_name}. RPE is nu geopend."
                st.rerun()
            except Exception as exc:
                st.error(f"Opslaan faalde: {exc}")
        st.markdown('</div>', unsafe_allow_html=True)

    if active_form == "RPE":
        st.markdown('<div class="mvv-section-card">', unsafe_allow_html=True)
        if has_rpe:
            st.success("RPE staat al ingevuld voor vandaag.")
        else:
            st.info("RPE staat nog open voor vandaag.")

        injury_locations = [
            "None",
            "Foot",
            "Ankle",
            "Lower leg",
            "Knee",
            "Upper leg",
            "Hip",
            "Groin",
            "Glute",
            "Lower back",
            "Abdomen",
            "Chest",
            "Shoulder",
            "Upper arm",
            "Elbow",
            "Forearm",
            "Wrist",
            "Hand",
            "Neck",
            "Head",
            "Other",
        ]

        existing_loc = str(rpe_header.get("injury_type") or "None").strip() or "None"
        if existing_loc not in injury_locations:
            existing_loc = "Other"

        with st.form(f"tablet_rpe_form_{player_id}", clear_on_submit=False):
            _legend_rpe()

            st.markdown("### Session 1")
            s1_dur = st.number_input(
                "[1] Duration (min)",
                min_value=0,
                max_value=600,
                value=_session_value(rpe_sessions, 1, "duration_min", 0),
                key=f"tablet_rpe_s1_dur_{player_id}",
            )
            s1_rpe = st.slider(
                "[1] RPE (1-10)",
                1,
                10,
                value=_session_value(rpe_sessions, 1, "rpe", 5),
                key=f"tablet_rpe_s1_rpe_{player_id}",
            )

            st.divider()

            enable_s2 = st.toggle(
                "Add 2nd session?",
                value=has_s2,
                key=f"tablet_rpe_enable_s2_{player_id}",
            )

            st.markdown("### Session 2")
            s2_dur = st.number_input(
                "[2] Duration (min)",
                min_value=0,
                max_value=600,
                value=_session_value(rpe_sessions, 2, "duration_min", 0),
                key=f"tablet_rpe_s2_dur_{player_id}",
            )
            s2_rpe = st.slider(
                "[2] RPE (1-10)",
                1,
                10,
                value=_session_value(rpe_sessions, 2, "rpe", 5),
                key=f"tablet_rpe_s2_rpe_{player_id}",
            )

            st.divider()
            st.markdown("### Injury")

            injury = st.toggle("Injury?", value=injury_default, key=f"tablet_rpe_injury_{player_id}")

            loc_col, pain_col = st.columns([1.2, 2.0])
            with loc_col:
                injury_loc = st.selectbox(
                    "Location",
                    options=injury_locations,
                    index=injury_locations.index(existing_loc),
                    key=f"tablet_rpe_loc_{player_id}",
                )
            with pain_col:
                injury_pain = st.slider(
                    "Pain (0-10)",
                    0,
                    10,
                    value=int(rpe_header.get("injury_pain", 0) or 0),
                    key=f"tablet_rpe_pain_{player_id}",
                )

            notes = st.text_area(
                "Notes (optional)",
                value=str(rpe_header.get("notes") or ""),
                key=f"tablet_rpe_notes_{player_id}",
            )
            rpe_submit = st.form_submit_button("RPE opslaan", use_container_width=True)

        if rpe_submit:
            try:
                sessions_payload: List[Dict[str, int]] = []

                if int(s1_dur) > 0:
                    sessions_payload.append(
                        {"session_index": 1, "duration_min": int(s1_dur), "rpe": int(s1_rpe)}
                    )

                if bool(enable_s2) and int(s2_dur) > 0:
                    sessions_payload.append(
                        {"session_index": 2, "duration_min": int(s2_dur), "rpe": int(s2_rpe)}
                    )

                injury_type_to_save = None
                injury_pain_to_save = None
                if bool(injury):
                    injury_type_to_save = None if injury_loc == "None" else injury_loc
                    injury_pain_to_save = int(injury_pain)

                save_rpe_tablet(
                    sb,
                    player_id=player_id,
                    entry_date=entry_date,
                    injury=bool(injury),
                    injury_type=injury_type_to_save,
                    injury_pain=injury_pain_to_save,
                    notes=notes,
                    sessions=sessions_payload,
                )
                clear_daily_cache()
                st.session_state["tablet_flash"] = f"RPE opgeslagen voor {player_name}."
                st.rerun()
            except Exception as exc:
                st.error(f"Opslaan faalde: {exc}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    if has_wellness and has_rpe:
        st.success("Alles staat erin voor vandaag.")
    else:
        st.info("Sla beide onderdelen op en ga daarna terug naar de spelerslijst.")

    if st.button("Klaar / volgende speler", use_container_width=True, key=f"tablet_done_{player_id}"):
        st.session_state.pop("tablet_player_id", None)
        st.session_state.pop("tablet_player_name", None)
        st.session_state.pop("tablet_active_form", None)
        st.rerun()


def main() -> None:
    ensure_tablet_access()

    sb = get_tablet_sb()
    if sb is None:
        st.error(
            "Supabase service key ontbreekt. Voeg SUPABASE_SERVICE_ROLE_KEY toe in Streamlit secrets "
            "voor deze losse tablet-app."
        )
        st.stop()

    selected_player_id = str(st.session_state.get("tablet_player_id") or "").strip()
    selected_player_name = str(st.session_state.get("tablet_player_name") or "").strip()

    if selected_player_id and selected_player_name:
        render_player_forms(sb, selected_player_id, selected_player_name)
    else:
        render_player_picker(sb)


if __name__ == "__main__":
    main()
