from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

import pages.Subscripts.gps_data_ffp_pages as ffp_pages
from auth_session import ensure_auth_restored, get_sb_client
from pages.Subscripts.mvv_branding import TEAM_HERO_BG, TEAM_LOGO, build_data_uri
from roles import get_profile, is_staff_user, render_sidebar_footer, render_sidebar_navigation
from utils.streamlit_ui import apply_streamlit_chrome


st.set_page_config(page_title="FFP Beta", layout="wide")
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
          background: __FFP_BETA_BG__;
          background-size: cover;
          background-position: center top;
          background-attachment: fixed;
        }

        .block-container {
          max-width: 1380px;
          padding-top: 1.4rem;
          padding-bottom: 2rem;
        }

        .ffp-beta-hero,
        .ffp-beta-panel {
          border-radius: 10px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(135deg, rgba(18, 25, 42, 0.88), rgba(10, 15, 27, 0.84));
          box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22);
        }

        .ffp-beta-hero {
          padding: 1.75rem 1.55rem;
          margin-bottom: 1rem;
        }

        .ffp-beta-panel {
          padding: 1rem 1.05rem 0.95rem 1.05rem;
          margin-bottom: 1rem;
        }

        .ffp-beta-logo {
          width: 78px;
          height: 78px;
          object-fit: contain;
          margin-bottom: 0;
          flex-shrink: 0;
          filter: drop-shadow(0 8px 22px rgba(0,0,0,0.28));
        }

        .ffp-beta-head {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 1rem;
        }

        .ffp-beta-kicker {
          color: rgba(255,255,255,0.76);
          font-size: 0.74rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.18em;
          margin-bottom: 0.35rem;
        }

        .ffp-beta-title {
          margin: 0;
          font-size: 2.3rem;
          line-height: 1;
          font-weight: 800;
          color: #ffffff;
        }

        .ffp-beta-copy {
          margin-top: 0.8rem;
          color: rgba(255,255,255,0.84);
          line-height: 1.6;
          max-width: 76ch;
        }

        .ffp-beta-pill-row,
        .ffp-beta-badge-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
        }

        .ffp-beta-pill,
        .ffp-beta-badge {
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

        .ffp-beta-pill-row {
          margin-top: 1rem;
        }

        .ffp-beta-panel-head {
          display: flex;
          justify-content: space-between;
          align-items: flex-end;
          gap: 1rem;
          margin-bottom: 0.75rem;
        }

        .ffp-beta-panel-kicker {
          color: rgba(255,255,255,0.62);
          font-size: 0.75rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }

        .ffp-beta-panel-title {
          margin-top: 0.25rem;
          color: #ffffff;
          font-size: 1.05rem;
          font-weight: 700;
        }

        .ffp-beta-panel-note {
          color: rgba(255,255,255,0.78);
          font-size: 0.88rem;
          font-weight: 700;
          text-align: right;
        }

        @media (max-width: 768px) {
          .ffp-beta-head {
            flex-direction: column;
            gap: 0.8rem;
          }

          .ffp-beta-title {
            font-size: 2rem;
            text-align: center;
          }

          .ffp-beta-panel-head {
            flex-direction: column;
            align-items: flex-start;
          }

          .ffp-beta-panel-note {
            text-align: left;
          }
        }
        </style>
        """.replace("__FFP_BETA_BG__", background),
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
def fetch_summary_all_cached(access_token: str) -> pd.DataFrame:
    raw = rest_get_paged(
        access_token,
        "gps_records",
        f"select={','.join(GPS_SELECT_COLS)}&event=eq.Summary&order=datum.asc,gps_id.asc",
        timeout=180,
    )
    return to_dashboard_df(raw)


@st.cache_data(show_spinner=False, ttl=120)
def fetch_summary_scope_day_count_cached(access_token: str, scope_key: str) -> int:
    d0, d1 = _scope_to_dates(scope_key)
    date_clause = ""
    if d0 and d1:
        date_clause = f"&datum=gte.{d0.isoformat()}&datum=lte.{d1.isoformat()}"

    raw = rest_get_paged(
        access_token,
        "gps_records",
        f"select=datum&event=eq.Summary{date_clause}&order=datum.asc",
    )
    if raw.empty or "datum" not in raw.columns:
        return 0
    dates = pd.to_datetime(raw["datum"], errors="coerce").dt.date.dropna()
    return int(dates.nunique())


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
        f'<img src="{TEAM_LOGO_URI}" alt="MVV Maastricht" class="ffp-beta-logo" />'
        if TEAM_LOGO_URI
        else ""
    )
    st.markdown(
        f"""
        <div class="ffp-beta-hero">
          <div class="ffp-beta-head">
            {logo_markup}
            <h1 class="ffp-beta-title">Fitness-Fatigue-Performance</h1>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scope_key = st.selectbox(
        "Contextscope voor indicatie",
        options=["Laatste 8 weken", "Laatste 12 weken", "Seizoen", "Alles"],
        index=0,
        key="ffp_beta_scope_indicator",
        help="FFP zelf blijft alle Summary-data gebruiken. Deze scope is alleen een contextbadge op de beta-pagina.",
    )

    try:
        with st.spinner("FFP: Summary data laden (ALLES)..."):
            df_ffp_all = fetch_summary_all_cached(str(token))
        scope_day_count = fetch_summary_scope_day_count_cached(str(token), scope_key)
    except Exception as exc:
        st.error(f"Kon FFP data niet laden: {exc}")
        df_ffp_all = pd.DataFrame()
        scope_day_count = 0

    st.markdown(
        f"""
        <div class="ffp-beta-panel">
          <div class="ffp-beta-panel-head">
            <div>
              <div class="ffp-beta-panel-kicker">Model</div>
              <div class="ffp-beta-panel-title">FFP gebruikt altijd alle Summary-data</div>
            </div>
            <div class="ffp-beta-panel-note">{scope_day_count} sessiedagen in de gekozen contextscope</div>
          </div>
          <div class="ffp-beta-badge-row">
            <span class="ffp-beta-badge">Contextscope: {scope_key}</span>
            <span class="ffp-beta-badge">Rekenbasis: alle Summary-data</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_ffp_all.empty:
        st.info("Geen Summary GPS data gevonden.")
    else:
        ffp_pages.ffp_pages_main(df_ffp_all)

    render_sidebar_footer(profile)


if __name__ == "__main__":
    main()
