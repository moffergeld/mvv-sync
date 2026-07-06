from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

import pages.Subscripts.gps_data_acwr_pages as acwr_pages
from auth_session import ensure_auth_restored, get_sb_client
from pages.Subscripts.mvv_branding import TEAM_HERO_BG, TEAM_LOGO, build_data_uri
from roles import get_profile, is_staff_user, render_sidebar_footer, render_sidebar_navigation
from utils.streamlit_ui import apply_streamlit_chrome


st.set_page_config(page_title="ACWR Beta", layout="wide")
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
          background: __ACWR_BETA_BG__;
          background-size: cover;
          background-position: center top;
          background-attachment: fixed;
        }

        .block-container {
          max-width: 1380px;
          padding-top: 1.4rem;
          padding-bottom: 2rem;
        }

        .acwr-beta-hero {
          border-radius: 10px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(135deg, rgba(18, 25, 42, 0.88), rgba(10, 15, 27, 0.84));
          padding: 1.75rem 1.55rem;
          box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22);
          margin-bottom: 1rem;
        }

        .acwr-beta-logo {
          width: 78px;
          height: 78px;
          object-fit: contain;
          margin-bottom: 0.8rem;
          filter: drop-shadow(0 8px 22px rgba(0,0,0,0.28));
        }

        .acwr-beta-kicker {
          color: rgba(255,255,255,0.76);
          font-size: 0.74rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.18em;
          margin-bottom: 0.35rem;
        }

        .acwr-beta-title {
          margin: 0;
          font-size: 2.3rem;
          line-height: 1;
          font-weight: 800;
          color: #ffffff;
        }

        .acwr-beta-copy {
          margin-top: 0.8rem;
          color: rgba(255,255,255,0.84);
          line-height: 1.6;
          max-width: 76ch;
        }

        .acwr-beta-pill-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
          margin-top: 1rem;
        }

        .acwr-beta-pill {
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
        </style>
        """.replace("__ACWR_BETA_BG__", background),
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
        return today_value - timedelta(days=56), today_value
    if scope_key == "Laatste 12 weken":
        return today_value - timedelta(days=84), today_value
    if scope_key == "Seizoen":
        return date(today_value.year if today_value.month >= 7 else today_value.year - 1, 7, 1), today_value
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
        f"select={','.join(GPS_SELECT_COLS)}&event=eq.Summary{date_clause}&order=datum.desc",
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
        f'<img src="{TEAM_LOGO_URI}" alt="MVV Maastricht" class="acwr-beta-logo" />'
        if TEAM_LOGO_URI
        else ""
    )
    st.markdown(
        f"""
        <div class="acwr-beta-hero">
          {logo_markup}
          <h1 class="acwr-beta-title">ACWR</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scope_key = st.selectbox(
        "Data scope (Summary-only)",
        options=["Laatste 8 weken", "Laatste 12 weken", "Seizoen", "Alles"],
        index=0,
        key="acwr_beta_scope",
    )

    try:
        with st.spinner(f"Summary data laden ({scope_key})..."):
            df_scope = fetch_summary_scope_cached(str(token), scope_key)
    except Exception as exc:
        st.error(f"Kon ACWR data niet laden: {exc}")
        df_scope = pd.DataFrame()

    if df_scope.empty:
        st.info("Geen Summary GPS data gevonden in deze scope.")
    else:
        acwr_pages.acwr_pages_main(df_scope)

    render_sidebar_footer(profile)


if __name__ == "__main__":
    main()
