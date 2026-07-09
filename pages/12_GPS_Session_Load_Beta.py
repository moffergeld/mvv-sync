from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

import pages.Subscripts.gps_data_session_load_pages as session_load_pages
from auth_session import ensure_auth_restored, get_sb_client
from pages.Subscripts.mvv_branding import TEAM_HERO_BG, TEAM_LOGO, build_data_uri
from roles import get_profile, is_staff_user, render_sidebar_footer, render_sidebar_navigation
from utils.streamlit_ui import apply_streamlit_chrome


st.set_page_config(page_title="Session Load Beta", layout="wide", initial_sidebar_state="expanded")
apply_streamlit_chrome()

PAGE_BG_URI = build_data_uri(TEAM_HERO_BG)
TEAM_LOGO_URI = build_data_uri(TEAM_LOGO)

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "").strip()


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
          background: __SESSION_LOAD_BETA_BG__;
          background-size: cover;
          background-position: center top;
          background-attachment: fixed;
        }

        .block-container {
          max-width: 1380px;
          padding-top: 1.4rem;
          padding-bottom: 2rem;
        }

        div[data-testid="stVerticalBlock"]:has(.session-load-beta-hero-anchor),
        .session-load-beta-panel {
          border-radius: 10px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(135deg, rgba(18, 25, 42, 0.88), rgba(10, 15, 27, 0.84));
          box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22);
        }

        div[data-testid="stVerticalBlock"]:has(.session-load-beta-hero-anchor) {
          padding: 1.75rem 1.55rem;
          margin-bottom: 1rem;
        }

        .session-load-beta-hero {
          margin-bottom: 1rem;
        }

        .session-load-beta-panel {
          padding: 1rem 1.05rem 0.95rem 1.05rem;
          margin-bottom: 1rem;
        }

        .session-load-beta-filter-label {
          color: rgba(255,255,255,0.92);
          font-size: 0.92rem;
          font-weight: 700;
          margin-bottom: 0.35rem;
        }

        .session-load-beta-logo {
          width: 78px;
          height: 78px;
          object-fit: contain;
          margin-bottom: 0;
          flex-shrink: 0;
          filter: drop-shadow(0 8px 22px rgba(0,0,0,0.28));
        }

        .session-load-beta-head {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 1rem;
        }

        .session-load-beta-kicker {
          color: rgba(255,255,255,0.76);
          font-size: 0.74rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.18em;
          margin-bottom: 0.35rem;
        }

        .session-load-beta-title {
          margin: 0;
          font-size: 2.3rem;
          line-height: 1;
          font-weight: 800;
          color: #ffffff;
        }

        .session-load-beta-copy {
          margin-top: 0.8rem;
          color: rgba(255,255,255,0.84);
          line-height: 1.6;
          max-width: 76ch;
        }

        .session-load-beta-pill-row,
        .session-load-beta-badge-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
        }

        .session-load-beta-pill,
        .session-load-beta-badge {
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

        .session-load-beta-pill-row {
          margin-top: 1rem;
        }

        @media (max-width: 768px) {
          .session-load-beta-head {
            flex-direction: column;
            gap: 0.8rem;
          }

          .session-load-beta-title {
            font-size: 2rem;
            text-align: center;
          }

          div[data-testid="stVerticalBlock"]:has(.session-load-beta-hero-anchor) {
            padding: 1.35rem 1rem;
          }
        }
        </style>
        """.replace("__SESSION_LOAD_BETA_BG__", background),
        unsafe_allow_html=True,
    )


def rest_headers(access_token: str) -> dict:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Prefer": "count=exact",
    }


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
        batch_headers = headers | {"Range": f"{start}-{end}"}
        response = requests.get(url, headers=batch_headers, timeout=timeout)
        if not response.ok:
            raise RuntimeError(f"GET {table} failed ({response.status_code}): {response.text}")

        batch = response.json()
        if not batch:
            break

        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size

    return pd.DataFrame(all_rows)


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

DB_TO_DASH = {
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
    "week": "Week",
    "year": "Year",
}


def to_dashboard_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=list(DB_TO_DASH.values()))

    df = raw_df.copy()
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    for numeric_col in [col for col in GPS_SELECT_COLS if col not in {"datum", "player_name", "type", "event"}]:
        if numeric_col in df.columns:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

    dashboard_df = df.rename(columns=DB_TO_DASH)
    if "Datum" in dashboard_df.columns:
        dashboard_df["Datum"] = dashboard_df["Datum"].dt.date
    return dashboard_df


def to_calendar_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=["Datum", "Type", "Event"])

    df = raw_df.copy().rename(columns={"datum": "Datum", "type": "Type", "event": "Event"})
    df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce").dt.date
    df["Type"] = df["Type"].astype(str).str.strip()
    df["Event"] = df["Event"].astype(str).str.strip()
    return df


def _scope_to_dates(scope_key: str) -> tuple[date | None, date | None]:
    today_value = date.today()
    if scope_key == "Laatste 8 weken":
        return today_value - timedelta(days=56 - 1), today_value
    if scope_key == "Laatste 12 weken":
        return today_value - timedelta(days=84 - 1), today_value
    if scope_key == "Seizoen":
        season_year = today_value.year if today_value.month >= 7 else today_value.year - 1
        return date(season_year, 7, 1), today_value
    return None, None


@st.cache_data(show_spinner=False, ttl=120)
def fetch_summary_scope_cached(access_token: str, scope_key: str) -> pd.DataFrame:
    d0, d1 = _scope_to_dates(scope_key)
    date_clause = ""
    if d0 and d1:
        date_clause = f"&datum=gte.{d0.isoformat()}&datum=lte.{d1.isoformat()}"

    raw = rest_get_paged(
        access_token,
        "gps_records",
        f"select={','.join(GPS_SELECT_COLS)}&event=eq.Summary{date_clause}&order=datum.asc,gps_id.asc",
    )
    return to_dashboard_df(raw)


@st.cache_data(show_spinner=False, ttl=120)
def fetch_calendar_scope_cached(access_token: str, scope_key: str) -> pd.DataFrame:
    d0, d1 = _scope_to_dates(scope_key)
    date_clause = ""
    if d0 and d1:
        date_clause = f"&datum=gte.{d0.isoformat()}&datum=lte.{d1.isoformat()}"

    raw = rest_get_paged(
        access_token,
        "gps_records",
        f"select=datum,type,event&event=eq.Summary{date_clause}&order=datum.asc,gps_id.asc",
    )
    return to_calendar_df(raw)


@st.cache_data(show_spinner=False, ttl=120)
def fetch_summary_day_cached(access_token: str, day_iso: str) -> pd.DataFrame:
    raw = rest_get_paged(
        access_token,
        "gps_records",
        f"select={','.join(GPS_SELECT_COLS)}&event=eq.Summary&datum=eq.{day_iso}&order=gps_id.asc",
    )
    return to_dashboard_df(raw)


def main() -> None:
    render_css()
    sb = get_sb_client()
    ok, token = ensure_auth_restored(sb)
    if not ok or not token:
        st.error("Sessie verlopen. Log opnieuw in.")
        st.switch_page("app.py")
        st.stop()

    profile = get_profile(sb)
    if not is_staff_user(profile):
        st.error("Geen toegang: deze pagina is alleen voor staff.")
        st.stop()
    render_sidebar_navigation(profile)

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.error("Missing secrets: SUPABASE_URL / SUPABASE_ANON_KEY")
        st.stop()

    logo_markup = (
        f'<img src="{TEAM_LOGO_URI}" alt="MVV Maastricht" class="session-load-beta-logo" />'
        if TEAM_LOGO_URI
        else ""
    )
    hero_shell = st.container()
    with hero_shell:
        st.markdown('<div class="session-load-beta-hero-anchor"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="session-load-beta-hero">
              <div class="session-load-beta-head">
                {logo_markup}
                <h1 class="session-load-beta-title">Session Load</h1>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        scope_col, calendar_col = st.columns(2, gap="large")
        with scope_col:
            st.markdown(
                '<div class="session-load-beta-filter-label">Data scope (Summary-only)</div>',
                unsafe_allow_html=True,
            )
            scope_key = st.selectbox(
                "Data scope (Summary-only)",
                options=["Laatste 8 weken", "Laatste 12 weken", "Seizoen", "Alles"],
                index=0,
                key="session_load_beta_scope",
                label_visibility="collapsed",
            )

    try:
        with st.spinner(f"Summary data laden ({scope_key})..."):
            df_scope = fetch_summary_scope_cached(str(token), scope_key)
            calendar_df_scope = fetch_calendar_scope_cached(str(token), scope_key)
    except Exception as exc:
        st.error(f"Kon Session Load data niet laden: {exc}")
        df_scope = pd.DataFrame()
        calendar_df_scope = pd.DataFrame()

    with calendar_col:
        st.markdown(
            '<div class="session-load-beta-filter-label">Kalender</div>',
            unsafe_allow_html=True,
        )
        selected_day = session_load_pages.pick_day_from_calendar(calendar_df_scope, key_prefix="sl_beta_hero")

    if df_scope.empty:
        st.info("Geen Summary GPS data gevonden in deze scope.")
    else:
        session_load_pages.session_load_pages_main(
            df_gps_scope=df_scope,
            calendar_df_all=calendar_df_scope,
            fetch_day_fn=lambda day_iso: fetch_summary_day_cached(str(token), day_iso),
            selected_day=selected_day,
        )

    render_sidebar_footer(profile)


if __name__ == "__main__":
    main()
