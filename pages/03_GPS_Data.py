# pages/03_GPS_Data.py
# ==========================================================
# GPS Data (Main) - STYLED + SAME DATAFLOW
#
# Afgesproken:
# - Summary-only data voor analyses
# - Data scope in sidebar
# - Tabs op pagina: Session Load / ACWR / FFP / Benchmarks
# - Session Load + ACWR gebruiken scope
# - FFP laadt altijd ALL Summary-data
# - Benchmarks via v_gps_match_events
# - Alleen design / page shell aangepast, dataflow gelijk
# ==========================================================

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

import pages.Subscripts.gps_data_session_load_pages as session_load_pages
import pages.Subscripts.gps_data_acwr_pages as acwr_pages
import pages.Subscripts.gps_data_ffp_pages as ffp_pages
import pages.Subscripts.gps_data_benchmarks_pages as benchmarks_pages
from pages.Subscripts.mvv_branding import TEAM_HERO_BG, TEAM_LOGO, build_data_uri

from auth_session import ensure_auth_restored, get_sb_client
from roles import get_profile, is_staff_user

st.set_page_config(page_title="GPS Data", layout="wide")

PAGE_BG_URI = build_data_uri(TEAM_HERO_BG)
TEAM_LOGO_URI = build_data_uri(TEAM_LOGO)


# ==========================================================
# PAGE STYLING
# ==========================================================
st.markdown(
    """
    <style>
    :root {
        --mvv-red: #C8102E;
        --mvv-red-light: #E8213F;
        --mvv-red-dark: #8B0A1F;
        --bg-main: #060B16;
        --bg-card: rgba(255,255,255,0.035);
        --bg-card-2: rgba(255,255,255,0.02);
        --line-soft: rgba(255,255,255,0.10);
        --text-main: #F5F7FB;
        --text-soft: rgba(245,247,251,0.72);
        --text-dim: rgba(245,247,251,0.52);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(200,16,46,0.18) 0%, rgba(200,16,46,0.04) 22%, rgba(0,0,0,0) 45%),
            radial-gradient(circle at bottom right, rgba(200,16,46,0.16) 0%, rgba(200,16,46,0.04) 18%, rgba(0,0,0,0) 40%),
            linear-gradient(180deg, #040915 0%, #050A16 100%);
        color: var(--text-main);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.025) 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: var(--text-main);
    }

    .gps-hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 26px;
        padding: 20px 22px 16px 22px;
        margin-bottom: 18px;
        background:
            radial-gradient(circle at top left, rgba(200,16,46,0.20) 0%, rgba(200,16,46,0.07) 25%, rgba(0,0,0,0) 60%),
            linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 40%, rgba(255,255,255,0.01) 100%);
        box-shadow:
            0 16px 40px rgba(0,0,0,0.22),
            inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .gps-kicker {
        font-size: 11px;
        letter-spacing: 0.24em;
        font-weight: 800;
        text-transform: uppercase;
        color: rgba(255,255,255,0.72);
        margin-bottom: 8px;
    }

    .gps-title {
        font-size: 28px;
        line-height: 1.05;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0 0 8px 0;
    }

    .gps-subtitle {
        max-width: 1100px;
        font-size: 14px;
        line-height: 1.6;
        color: rgba(255,255,255,0.82);
        margin-bottom: 14px;
    }

    .gps-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }

    .gps-pill {
        border-radius: 999px;
        padding: 8px 12px;
        font-size: 12px;
        font-weight: 700;
        color: #FFFFFF;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.02) 100%);
    }

    .gps-section-card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 14px 16px;
        margin: 10px 0 18px 0;
        background:
            linear-gradient(180deg, rgba(255,255,255,0.028) 0%, rgba(255,255,255,0.018) 100%);
        box-shadow: 0 10px 24px rgba(0,0,0,0.14);
    }

    .gps-section-label {
        font-size: 11px;
        letter-spacing: 0.22em;
        font-weight: 800;
        text-transform: uppercase;
        color: rgba(255,255,255,0.72);
        margin-bottom: 8px;
    }

    .gps-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }

    .gps-badge {
        border-radius: 999px;
        padding: 8px 12px;
        font-size: 12px;
        font-weight: 700;
        color: white;
        background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .gps-divider {
        height: 1px;
        background: linear-gradient(90deg, rgba(255,255,255,0.0) 0%, rgba(255,255,255,0.18) 15%, rgba(255,255,255,0.18) 85%, rgba(255,255,255,0.0) 100%);
        margin: 18px 0 14px 0;
    }

    div[data-testid="stTabs"] button {
        border-radius: 999px !important;
        padding: 10px 16px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        background: rgba(255,255,255,0.02) !important;
        color: rgba(255,255,255,0.78) !important;
        font-weight: 700 !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: linear-gradient(180deg, rgba(200,16,46,0.30) 0%, rgba(200,16,46,0.16) 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(232,33,63,0.40) !important;
        box-shadow: 0 8px 18px rgba(200,16,46,0.18);
    }

    div[data-testid="stTabs"] {
        margin-top: 6px;
    }

    div[data-baseweb="select"] > div,
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 14px !important;
    }

    .st-emotion-cache-16txtl3, .st-emotion-cache-1kyxreq {
        color: var(--text-main);
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_GPS_PAGE_BACKGROUND = (
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
        background: __GPS_PAGE_BACKGROUND__ !important;
        background-size: cover !important;
        background-position: center top !important;
        background-attachment: fixed !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(16, 23, 38, 0.98), rgba(9, 13, 23, 0.98)) !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }

    .gps-hero-shell {
        display: flex;
        flex-direction: column;
        gap: 1.1rem;
        margin-bottom: 1.55rem;
    }

    .gps-hero {
        min-height: 320px;
        border-radius: 8px;
        padding: 2rem 1.75rem 1.9rem 1.75rem;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(135deg, rgba(18, 25, 42, 0.88), rgba(10, 15, 27, 0.84));
        box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22);
    }

    .gps-hero-logo {
        width: 82px;
        height: 82px;
        object-fit: contain;
        margin-bottom: 0.9rem;
        filter: drop-shadow(0 8px 22px rgba(0,0,0,0.28));
    }

    .gps-kicker {
        color: rgba(255,255,255,0.76);
        font-size: 0.74rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        margin-bottom: 0.35rem;
    }

    .gps-title {
        font-size: 2.55rem;
        line-height: 1;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
    }

    .gps-subtitle {
        margin-top: 0.8rem;
        max-width: 74ch;
        color: rgba(255,255,255,0.84);
        line-height: 1.62;
        font-size: 0.96rem;
        margin-bottom: 0;
    }

    .gps-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1rem;
    }

    .gps-pill {
        border-radius: 999px;
        padding: 0.42rem 0.76rem;
        font-size: 0.78rem;
        font-weight: 800;
        border: 1px solid rgba(234, 51, 81, 0.22);
        background: rgba(255,255,255,0.06);
        color: #FFFFFF;
    }

    .gps-summary-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
    }

    .gps-summary-card,
    .gps-section-card {
        border-radius: 8px;
        border: 1px solid rgba(234, 51, 81, 0.14);
        background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
    }

    .gps-summary-card {
        min-height: 122px;
        padding: 1rem 1.05rem 0.95rem 1.05rem;
    }

    .gps-summary-label {
        color: rgba(255,255,255,0.68);
        font-size: 0.8rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .gps-summary-value {
        margin-top: 0.55rem;
        font-size: 2rem;
        line-height: 1.05;
        font-weight: 800;
        color: #FFFFFF;
    }

    .gps-summary-foot {
        margin-top: 0.65rem;
        color: rgba(255,255,255,0.8);
        font-size: 0.86rem;
        line-height: 1.4;
    }

    .gps-filter-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 1rem;
        margin-bottom: 12px;
    }

    .gps-filter-title {
        color: #FFFFFF;
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    .gps-filter-note {
        color: rgba(255,255,255,0.8);
        font-size: 0.88rem;
        font-weight: 700;
        text-align: right;
    }

    .gps-section-card {
        padding: 16px;
        margin: 0 0 18px 0;
    }

    .gps-section-label {
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        font-weight: 800;
        text-transform: uppercase;
        color: rgba(255,255,255,0.62);
        margin-bottom: 0.25rem;
    }

    .gps-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 0.5rem;
    }

    .gps-badge {
        border-radius: 999px;
        padding: 8px 12px;
        font-size: 12px;
        font-weight: 700;
        color: white;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
    }

    @media (max-width: 1100px) {
        .gps-summary-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 768px) {
        .gps-hero {
            min-height: auto;
            padding: 1.55rem 1rem;
        }

        .gps-title {
            font-size: 2rem;
        }

        .gps-summary-grid {
            grid-template-columns: 1fr;
        }

        .gps-filter-head {
            flex-direction: column;
            align-items: flex-start;
        }

        .gps-filter-note {
            text-align: left;
        }
    }
    </style>
    """.replace("__GPS_PAGE_BACKGROUND__", _GPS_PAGE_BACKGROUND),
    unsafe_allow_html=True,
)


# ==========================================================
# AUTH / ACCESS
# ==========================================================
sb = get_sb_client()
ok, token = ensure_auth_restored(sb)

if not ok or not token:
    st.error("Sessie verlopen. Log opnieuw in.")
    try:
        st.switch_page("app.py")
    except Exception:
        pass
    st.stop()

profile = get_profile(sb)
if not is_staff_user(profile):
    st.error("Geen toegang.")
    st.stop()

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "").strip()
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing secrets: SUPABASE_URL / SUPABASE_ANON_KEY")
    st.stop()


# ==========================================================
# TOKEN HELPERS
# ==========================================================
def get_access_token() -> str | None:
    tok = st.session_state.get("access_token")
    if tok:
        return str(tok)

    sess = st.session_state.get("sb_session")
    if sess is not None:
        t2 = getattr(sess, "access_token", None)
        if t2:
            return str(t2)

    return str(token) if token else None


def rest_headers(access_token: str) -> dict:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Prefer": "count=exact",
    }


def auth_get_user(access_token: str) -> dict:
    url = f"{SUPABASE_URL}/auth/v1/user"
    r = requests.get(url, headers=rest_headers(access_token), timeout=30)
    if not r.ok:
        raise RuntimeError(f"AUTH user fetch failed ({r.status_code}): {r.text}")
    return r.json()


# ==========================================================
# REST PAGING
# ==========================================================
def rest_get_paged(
    access_token: str,
    table: str,
    base_query: str,
    page_size: int = 5000,
    timeout: int = 120,
) -> pd.DataFrame:
    url = f"{SUPABASE_URL}/rest/v1/{table}?{base_query}"
    headers = rest_headers(access_token) | {"Range-Unit": "items"}

    all_rows: list[dict] = []
    start = 0

    while True:
        end = start + page_size - 1
        h = headers | {"Range": f"{start}-{end}"}

        r = requests.get(url, headers=h, timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"GET {table} failed ({r.status_code}): {r.text}")

        batch = r.json()
        if not batch:
            break

        all_rows.extend(batch)

        if len(batch) < page_size:
            break

        start += page_size

    return pd.DataFrame(all_rows)


# ==========================================================
# SELECT COLUMNS
# ==========================================================
GPS_SELECT_COLS = [
    "gps_id",
    "datum",
    "week",
    "year",
    "player_name",
    "type",
    "event",
    "duration",
    "total_distance",
    "running",
    "sprint",
    "high_sprint",
    "max_speed",
    "playerload2d",
    "total_accelerations",
    "high_accelerations",
    "total_decelerations",
    "high_decelerations",
    "hrzone1",
    "hrzone2",
    "hrzone3",
    "hrzone4",
    "hrzone5",
    "hrtrimp",
]


# ==========================================================
# TRANSFORM
# ==========================================================
_DB_TO_DASH = {
    "datum": "Datum",
    "player_name": "Speler",
    "type": "Type",
    "event": "Event",
    "duration": "Duration",
    "total_distance": "Total Distance",
    "running": "Running",
    "sprint": "Sprint",
    "high_sprint": "High Sprint",
    "max_speed": "Max Speed",
    "playerload2d": "playerload2D",
    "total_accelerations": "Total Accelerations",
    "high_accelerations": "High Accelerations",
    "total_decelerations": "Total Decelerations",
    "high_decelerations": "High Decelerations",
    "hrzone1": "HRzone1",
    "hrzone2": "HRzone2",
    "hrzone3": "HRzone3",
    "hrzone4": "HRzone4",
    "hrzone5": "HRzone5",
    "hrtrimp": "HRtrimp",
}


def _supabase_to_dashboard_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    rename_map = {k: v for k, v in _DB_TO_DASH.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    if "Datum" in df.columns:
        df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce").dt.date

    for c in ["Speler", "Type", "Event"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    non_num = {"Datum", "Speler", "Type", "Event", "gps_id", "week", "year"}
    for c in df.columns:
        if c not in non_num:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _supabase_to_calendar_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["Datum", "Type", "Event"])

    df = df_raw.copy()
    df = df.rename(columns={"datum": "Datum", "type": "Type", "event": "Event"})
    df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce").dt.date
    df["Type"] = df["Type"].astype(str).str.strip()
    df["Event"] = df["Event"].astype(str).str.strip()
    return df


# ==========================================================
# SCOPE HELPERS
# ==========================================================
def _season_start(today: date) -> date:
    return date(today.year, 7, 1) if today.month >= 7 else date(today.year - 1, 7, 1)


def _scope_to_dates(scope_key: str) -> tuple[date | None, date | None]:
    today = date.today()
    if scope_key == "Laatste 8 weken":
        return today - timedelta(days=56 - 1), today
    if scope_key == "Laatste 12 weken":
        return today - timedelta(days=84 - 1), today
    if scope_key == "Seizoen":
        return _season_start(today), today
    if scope_key == "Alles":
        return None, None
    return today - timedelta(days=56 - 1), today


# ==========================================================
# CACHED FETCHERS
# ==========================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_summary_scope_cached(access_token: str, scope_key: str) -> pd.DataFrame:
    d0, d1 = _scope_to_dates(scope_key)

    select = ",".join(GPS_SELECT_COLS)
    q = f"select={select}&event=eq.Summary&order=datum.asc,gps_id.asc"
    if d0 and d1:
        q += f"&datum=gte.{d0.isoformat()}&datum=lte.{d1.isoformat()}"

    raw = rest_get_paged(access_token, "gps_records", q, page_size=5000, timeout=120)
    return _supabase_to_dashboard_df(raw)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_summary_all_cached(access_token: str) -> pd.DataFrame:
    select = ",".join(GPS_SELECT_COLS)
    q = f"select={select}&event=eq.Summary&order=datum.asc,gps_id.asc"
    raw = rest_get_paged(access_token, "gps_records", q, page_size=5000, timeout=180)
    return _supabase_to_dashboard_df(raw)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_calendar_dates_all_cached(access_token: str) -> pd.DataFrame:
    q = "select=datum,type,event&event=eq.Summary&order=datum.asc"
    raw = rest_get_paged(access_token, "gps_records", q, page_size=5000, timeout=120)
    return _supabase_to_calendar_df(raw)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_summary_day_cached(access_token: str, day_iso: str) -> pd.DataFrame:
    select = ",".join(GPS_SELECT_COLS)
    q = f"select={select}&event=eq.Summary&datum=eq.{day_iso}&order=gps_id.asc"
    raw = rest_get_paged(access_token, "gps_records", q, page_size=5000, timeout=120)
    return _supabase_to_dashboard_df(raw)


# ==========================================================
# UI
# ==========================================================
access_token = get_access_token()
if not access_token:
    st.error("Niet ingelogd (access_token ontbreekt).")
    st.stop()

u = auth_get_user(access_token)
st.session_state["user_id"] = u.get("id") or ""

calendar_df_all = fetch_calendar_dates_all_cached(access_token)
session_days = (
    int(calendar_df_all["datum"].dropna().nunique())
    if not calendar_df_all.empty and "datum" in calendar_df_all.columns
    else 0
)
session_types = (
    int(calendar_df_all["type"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().nunique())
    if not calendar_df_all.empty and "type" in calendar_df_all.columns
    else 0
)
summary_cards = [
    ("Sessiedagen", str(session_days), "Unieke dagen met Summary-data"),
    ("Kalenderitems", str(len(calendar_df_all)), "Records beschikbaar in de GPS-kalender"),
    ("Datatypen", str(session_types), "Trainings- en wedstrijdbuckets in de dataset"),
    ("Modules", "4", "Session Load, ACWR, FFP en Benchmarks"),
]
summary_markup = "".join(
    f"""<div class="gps-summary-card">
<div class="gps-summary-label">{label}</div>
<div class="gps-summary-value">{value}</div>
<div class="gps-summary-foot">{foot}</div>
</div>"""
    for label, value, foot in summary_cards
)
hero_logo_markup = (
    f'<img src="{TEAM_LOGO_URI}" alt="MVV Maastricht" class="gps-hero-logo" />'
    if TEAM_LOGO_URI
    else ""
)

st.markdown(
    f"""
    <div class="gps-hero-shell">
        <div class="gps-hero">
            {hero_logo_markup}
            <div class="gps-kicker">MVV Performance Dashboard | GPS Data</div>
            <div class="gps-title">GPS Data Overview</div>
            <div class="gps-subtitle">
                Summary-only analyses voor Session Load, ACWR, FFP en benchmarks.
                De dataflow blijft gelijk, maar de pagina leest nu visueel mee met de rest van de MVV-omgeving.
            </div>
            <div class="gps-pill-row">
                <div class="gps-pill">Summary-only analyses</div>
                <div class="gps-pill">ACWR thresholds per week</div>
                <div class="gps-pill">FFP laadt altijd alle Summary-data</div>
                <div class="gps-pill">Benchmarks uit match events</div>
            </div>
        </div>
        <div class="gps-summary-grid">
            {summary_markup}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="gps-section-card">
        <div class="gps-filter-head">
            <div>
                <div class="gps-section-label">Data scope</div>
                <div class="gps-filter-title">Kies welke periode je wilt gebruiken voor de Summary-analyses</div>
            </div>
            <div class="gps-filter-note">FFP blijft altijd alle Summary-data laden</div>
        </div>
        <div class="gps-badge-row">
            <div class="gps-badge">Modules: Session Load, ACWR, FFP, Benchmarks</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

scope_col, note_col = st.columns([1.15, 1.85], gap="large")
with scope_col:
    scope_key = st.selectbox(
        "Data scope (Summary-only)",
        options=["Laatste 8 weken", "Laatste 12 weken", "Seizoen", "Alles"],
        index=0,
        key="gps_scope",
    )
with note_col:
    st.markdown(
        f"""
        <div class="gps-badge-row" style="justify-content:flex-end; margin-top: 1.95rem;">
            <div class="gps-badge">Actieve scope: {scope_key}</div>
            <div class="gps-badge">{session_days} sessiedagen in de kalender</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab_session, tab_acwr, tab_ffp, tab_bench = st.tabs(
    ["Session Load", "ACWR", "FFP", "Benchmarks"]
)

# --------------------------------------------------
# Session Load
# --------------------------------------------------
with tab_session:
    with st.spinner(f"Summary data laden ({scope_key})..."):
        df_scope = fetch_summary_scope_cached(access_token, scope_key)

    if df_scope.empty:
        st.info("Geen Summary GPS data gevonden in deze scope.")
    else:
        def _fetch_day(day_iso: str) -> pd.DataFrame:
            return fetch_summary_day_cached(access_token, day_iso)

        session_load_pages.session_load_pages_main(
            df_gps_scope=df_scope,
            calendar_df_all=calendar_df_all,
            fetch_day_fn=_fetch_day,
        )

# --------------------------------------------------
# ACWR
# --------------------------------------------------
with tab_acwr:
    with st.spinner(f"Summary data laden ({scope_key})..."):
        df_scope = fetch_summary_scope_cached(access_token, scope_key)

    if df_scope.empty:
        st.info("Geen Summary GPS data gevonden in deze scope.")
    else:
        acwr_pages.acwr_pages_main(df_scope)

# --------------------------------------------------
# FFP
# --------------------------------------------------
with tab_ffp:
    with st.spinner("FFP: Summary data laden (ALLES)..."):
        df_ffp_all = fetch_summary_all_cached(access_token)

    if df_ffp_all.empty:
        st.info("Geen Summary GPS data gevonden.")
    else:
        ffp_pages.ffp_pages_main(df_ffp_all)

# --------------------------------------------------
# Benchmarks
# --------------------------------------------------
with tab_bench:
    benchmarks_pages.benchmarks_pages_main(
        supabase_url=SUPABASE_URL,
        supabase_anon_key=SUPABASE_ANON_KEY,
        access_token=access_token,
        user_id=st.session_state.get("user_id", ""),
    )
