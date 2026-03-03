# pages/03_GPS_Data.py
# ==========================================================
# GPS Data (Main) - OPTIMALIZED + MINIMAL METRICS
#
# Afgesproken:
# - Summary-only data voor analyses
# - Default scope: laatste 8 weken (keuze: 8w/12w/Seizoen/Alles)
# - df_all (scope) = Summary-only + behoud Type (Practice/Match splits)
# - FFP: altijd ALL, maar alleen laden wanneer FFP wordt geopend
# - Benchmarks: beschikbaar
# - Data preview: verwijderd
# - Kalender (Session Load): all-time tonen via aparte lichte calendar df
#
# Database -> Dashboard kolomnamen (belangrijk voor bestaande subscripts):
# - duration            -> Duration
# - total_distance      -> Total Distance
# - running             -> Running
# - sprint              -> Sprint
# - high_sprint         -> High Sprint
# - max_speed           -> Max Speed
# - playerload2d        -> PlayerLoad2D
# - total_accelerations -> Total Accelerations
# - high_accelerations  -> High Accelerations
# - total_decelerations -> Total Decelerations
# - high_decelerations  -> High Decelerations
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

from auth_session import ensure_auth_restored, get_sb_client
from roles import get_profile, is_staff_user

st.set_page_config(page_title="GPS Data", layout="wide")

# -------------------------
# Auth restore
# -------------------------
sb = get_sb_client()
ok, token = ensure_auth_restored(sb)

if not ok or not token:
    st.error("Sessie verlopen. Log opnieuw in.")
    try:
        st.switch_page("app.py")
    except Exception:
        pass
    st.stop()

# -------------------------
# Staff-only gate
# -------------------------
profile = get_profile(sb)
if not is_staff_user(profile):
    st.error("Geen toegang.")
    st.stop()

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "").strip()
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing secrets: SUPABASE_URL / SUPABASE_ANON_KEY")
    st.stop()


# -------------------------
# Token helpers
# -------------------------
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


# -------------------------
# REST paging
# -------------------------
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


# -------------------------
# Minimal select columns (db names)
# -------------------------
GPS_SELECT_COLS = [
    "gps_id",
    "datum",
    "week",
    "year",
    "player_name",
    "type",
    "event",
    # metrics
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
]


# -------------------------
# Transform: Supabase -> Dashboard df
# -------------------------
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
    "playerload2d": "PlayerLoad2D",
    "total_accelerations": "Total Accelerations",
    "high_accelerations": "High Accelerations",
    "total_decelerations": "Total Decelerations",
    "high_decelerations": "High Decelerations",
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


# -------------------------
# Scope helpers
# -------------------------
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


# -------------------------
# Cached fetchers
# -------------------------
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


# -------------------------
# UI
# -------------------------
st.title("GPS Data")

access_token = get_access_token()
if not access_token:
    st.error("Niet ingelogd (access_token ontbreekt).")
    st.stop()

u = auth_get_user(access_token)
st.session_state["user_id"] = u.get("id") or ""

scope_key = st.selectbox(
    "Data scope (Summary-only)",
    options=["Laatste 8 weken", "Laatste 12 weken", "Seizoen", "Alles"],
    index=0,
    key="gps_scope",
)

sub_page = st.radio(
    "Subpagina",
    options=["Session Load", "ACWR", "FFP"],
    horizontal=True,
    key="gpsdata_subpage",
)

st.divider()

calendar_df_all = fetch_calendar_dates_all_cached(access_token)

if sub_page == "Session Load":
    tab_dash, tab_bench = st.tabs(["Dashboard", "Benchmarks"])

    with tab_dash:
        with st.spinner(f"Summary data laden ({scope_key})..."):
            df_scope = fetch_summary_scope_cached(access_token, scope_key)

        if df_scope.empty:
            st.info("Geen Summary GPS data gevonden in deze scope.")
            st.stop()

        def _fetch_day(day_iso: str) -> pd.DataFrame:
            return fetch_summary_day_cached(access_token, day_iso)

        session_load_pages.session_load_pages_main(
            df_gps_scope=df_scope,
            calendar_df_all=calendar_df_all,
            fetch_day_fn=_fetch_day,
        )

    with tab_bench:
        benchmarks_pages.benchmarks_pages_main(
            supabase_url=SUPABASE_URL,
            supabase_anon_key=SUPABASE_ANON_KEY,
            access_token=access_token,
            user_id=st.session_state.get("user_id", ""),
        )

elif sub_page == "ACWR":
    tab_acwr, tab_bench = st.tabs(["ACWR module", "Benchmarks"])

    with tab_acwr:
        with st.spinner(f"Summary data laden ({scope_key})..."):
            df_scope = fetch_summary_scope_cached(access_token, scope_key)

        if df_scope.empty:
            st.info("Geen Summary GPS data gevonden in deze scope.")
            st.stop()

        acwr_pages.acwr_pages_main(df_scope)

    with tab_bench:
        benchmarks_pages.benchmarks_pages_main(
            supabase_url=SUPABASE_URL,
            supabase_anon_key=SUPABASE_ANON_KEY,
            access_token=access_token,
            user_id=st.session_state.get("user_id", ""),
        )

elif sub_page == "FFP":
    tab_ffp, tab_bench = st.tabs(["FFP module", "Benchmarks"])

    with tab_ffp:
        with st.spinner("FFP: Summary data laden (ALLES)..."):
            df_ffp_all = fetch_summary_all_cached(access_token)

        if df_ffp_all.empty:
            st.info("Geen Summary GPS data gevonden.")
            st.stop()

        ffp_pages.ffp_pages_main(df_ffp_all)

    with tab_bench:
        benchmarks_pages.benchmarks_pages_main(
            supabase_url=SUPABASE_URL,
            supabase_anon_key=SUPABASE_ANON_KEY,
            access_token=access_token,
            user_id=st.session_state.get("user_id", ""),
        )
