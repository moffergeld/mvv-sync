# pages/03_GPS_Data.py
# ==========================================================
# GPS Data pagina (OPTIMALIZED)
#
# Afgesproken gedrag:
# - Default: laatste 8 weken (Summary-only) -> snelle load
# - Scope selector: Laatste 8 weken / Laatste 12 weken / Seizoen / Alles
# - df_all voor Session Load + ACWR = Summary-only binnen scope
# - Practice/Match splits blijven werken via kolom 'Type'
# - FFP: laadt altijd ALLES (Summary-only) maar alleen als FFP geopend wordt
# - Benchmarks tab blijft "alles" (eigen fetch in benchmarks subscript)
# - Kalender (Session Load) moet all-time blijven:
#   -> lichte calendar_df_all (Datum/Type/Event) over ALL summary records
#   -> als gebruiker een oude datum klikt buiten scope: on-demand 1-dag fetch
#
# Tech:
# - REST paging blijft, maar nu server-side gefilterd op event=Summary + datum-range
# - Caching per scope en per dag
# - use_container_width vervangen door width="stretch"
# ==========================================================

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

import pages.Subscripts.gps_data_session_load_pages as session_load_pages
import pages.Subscripts.gps_data_acwr_pages as acwr_pages
import pages.Subscripts.gps_data_ffp_pages as ffp_pages
import pages.Subscripts.gps_data_benchmarks_pages as benchmarks_pages  # Benchmarks tab (Gref)

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
# Auth / REST helpers
# -------------------------
def get_access_token() -> str | None:
    tok = st.session_state.get("access_token")
    if tok:
        return str(tok)

    sess = st.session_state.get("sb_session")
    if sess is not None:
        token2 = getattr(sess, "access_token", None)
        if token2:
            return str(token2)

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
# Supabase fetch -> DataFrame voor dashboards
# (zelfde mapping als je oorspronkelijke script)
# -------------------------
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
    "walking",
    "jogging",
    "running",
    "sprint",
    "high_sprint",
    "number_of_sprints",
    "number_of_high_sprints",
    "number_of_repeated_sprints",
    "max_speed",
    "avg_speed",
    "playerload3d",
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
    "hrzoneanaerobic",
    "avg_hr",
    "max_hr",
]


def _supabase_to_dashboard_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")

    rename_map = {
        "gps_id": "gps_id",
        "datum": "Datum",
        "player_name": "Speler",
        "type": "Type",
        "event": "Event",
        "week": "Week",
        "year": "Year",
        "duration": "Duration",
        "total_distance": "Total Distance",
        "walking": "Walk Distance",
        "jogging": "Jog Distance",
        "running": "Run Distance",
        "sprint": "Sprint",
        "high_sprint": "High Sprint",
        "number_of_sprints": "Number of Sprints",
        "number_of_high_sprints": "Number of High Sprints",
        "number_of_repeated_sprints": "Number of Repeated Sprints",
        "max_speed": "Max Speed",
        "avg_speed": "Avg Speed",
        "playerload3d": "Playerload3D",
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
        "hrtrimp": "HRTrimp",
        "hrzoneanaerobic": "HRzoneAnaerobic",
        "avg_hr": "Avg HR",
        "max_hr": "Max HR",
    }
    df = df.rename(columns=rename_map)

    non_num = {"Datum", "Speler", "Type", "Event"}
    for c in df.columns:
        if c not in non_num and c != "gps_id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Event" in df.columns:
        df["Event"] = df["Event"].astype(str).str.strip()
    if "Type" in df.columns:
        df["Type"] = df["Type"].astype(str).str.strip()

    return df


def _supabase_to_calendar_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Datum", "Type", "Event"])
    out = df.copy()
    out["datum"] = pd.to_datetime(out["datum"], errors="coerce")
    out = out.rename(columns={"datum": "Datum", "type": "Type", "event": "Event"})
    out = out.dropna(subset=["Datum"]).copy()
    out["Datum"] = out["Datum"].dt.date
    out["Type"] = out["Type"].astype(str).str.strip()
    out["Event"] = out["Event"].astype(str).str.strip()
    return out[["Datum", "Type", "Event"]]


# -------------------------
# Scope helpers
# -------------------------
def _season_start(today: date) -> date:
    if today.month >= 7:
        return date(today.year, 7, 1)
    return date(today.year - 1, 7, 1)


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
def fetch_gps_summary_scope_cached(access_token: str, user_id: str, scope_key: str) -> pd.DataFrame:
    d0, d1 = _scope_to_dates(scope_key)
    select = ",".join(GPS_SELECT_COLS)
    base_query = f"select={select}&event=eq.Summary&order=datum.asc,gps_id.asc"
    if d0 and d1:
        base_query += f"&datum=gte.{d0.isoformat()}&datum=lte.{d1.isoformat()}"

    raw = rest_get_paged(
        access_token=access_token,
        table="gps_records",
        base_query=base_query,
        page_size=5000,
        timeout=120,
    )
    return _supabase_to_dashboard_df(raw)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_gps_summary_all_cached(access_token: str, user_id: str) -> pd.DataFrame:
    select = ",".join(GPS_SELECT_COLS)
    base_query = f"select={select}&event=eq.Summary&order=datum.asc,gps_id.asc"
    raw = rest_get_paged(
        access_token=access_token,
        table="gps_records",
        base_query=base_query,
        page_size=5000,
        timeout=180,
    )
    return _supabase_to_dashboard_df(raw)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_calendar_all_cached(access_token: str, user_id: str) -> pd.DataFrame:
    # lichte fetch: alleen datum/type/event, alleen Summary
    base_query = "select=datum,type,event&event=eq.Summary&order=datum.asc"
    raw = rest_get_paged(
        access_token=access_token,
        table="gps_records",
        base_query=base_query,
        page_size=5000,
        timeout=120,
    )
    return _supabase_to_calendar_df(raw)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_gps_summary_day_cached(access_token: str, user_id: str, day_iso: str) -> pd.DataFrame:
    select = ",".join(GPS_SELECT_COLS)
    base_query = f"select={select}&event=eq.Summary&datum=eq.{day_iso}&order=gps_id.asc"
    raw = rest_get_paged(
        access_token=access_token,
        table="gps_records",
        base_query=base_query,
        page_size=5000,
        timeout=120,
    )
    return _supabase_to_dashboard_df(raw)


# -------------------------
# UI
# -------------------------
st.title("GPS Data")

access_token = get_access_token()
if not access_token:
    st.error("Niet ingelogd (access_token ontbreekt).")
    st.stop()

# user_id alleen voor cache-key stabiliteit
try:
    u = auth_get_user(access_token)
    user_id = u.get("id") or "unknown"
except Exception as e:
    sb2 = get_sb_client()
    ok2, token2 = ensure_auth_restored(sb2)
    if ok2 and token2:
        access_token = token2
        try:
            u = auth_get_user(access_token)
            user_id = u.get("id") or "unknown"
        except Exception as e2:
            st.error(f"Kon user niet ophalen: {e2}")
            st.stop()
    else:
        st.error(f"Kon user niet ophalen: {e}")
        st.stop()

# scope selector
scope_key = st.selectbox(
    "Data scope (Summary-only)",
    options=["Laatste 8 weken", "Laatste 12 weken", "Seizoen", "Alles"],
    index=0,
    key="gps_scope",
    help="Default is snel (8 weken). 'Alles' kan zwaar zijn. FFP laadt altijd 'Alles' maar pas als je FFP opent.",
)

sub_page = st.radio(
    "Subpagina",
    options=["Session Load", "ACWR", "FFP"],
    horizontal=True,
    key="gpsdata_subpage",
    label_visibility="collapsed",
)

st.divider()

# kalender data (all-time, light)
with st.spinner("Kalender laden (all-time)..."):
    calendar_df_all = fetch_calendar_all_cached(access_token, user_id)

# =========================
# SESSION LOAD
# =========================
if sub_page == "Session Load":
    tabs = st.tabs(["Dashboard", "Benchmarks", "Data preview"])

    with tabs[0]:
        with st.spinner(f"Summary data laden ({scope_key})..."):
            df_scope = fetch_gps_summary_scope_cached(access_token, user_id, scope_key)

        if df_scope.empty:
            st.info("Geen Summary GPS data gevonden in deze scope.")
            st.stop()

        def _fetch_day(day_iso: str) -> pd.DataFrame:
            return fetch_gps_summary_day_cached(access_token, user_id, day_iso)

        session_load_pages.session_load_pages_main(
            df_scope,
            calendar_df_all=calendar_df_all,
            fetch_day_fn=_fetch_day,
        )

    with tabs[1]:
        benchmarks_pages.benchmarks_pages_main(
            supabase_url=SUPABASE_URL,
            supabase_anon_key=SUPABASE_ANON_KEY,
            access_token=access_token,
            user_id=user_id,
        )

    with tabs[2]:
        with st.spinner(f"Summary data laden ({scope_key})..."):
            df_scope = fetch_gps_summary_scope_cached(access_token, user_id, scope_key)
        st.dataframe(df_scope, width="stretch", height=520)

# =========================
# ACWR (Summary-only; uses scope df)
# =========================
elif sub_page == "ACWR":
    tabs = st.tabs(["ACWR module", "Benchmarks", "Data preview"])

    with tabs[0]:
        with st.spinner(f"Summary data laden ({scope_key})..."):
            df_scope = fetch_gps_summary_scope_cached(access_token, user_id, scope_key)

        if df_scope.empty:
            st.info("Geen Summary GPS data gevonden in deze scope.")
            st.stop()

        acwr_pages.acwr_pages_main(df_scope)

    with tabs[1]:
        benchmarks_pages.benchmarks_pages_main(
            supabase_url=SUPABASE_URL,
            supabase_anon_key=SUPABASE_ANON_KEY,
            access_token=access_token,
            user_id=user_id,
        )

    with tabs[2]:
        with st.spinner(f"Summary data laden ({scope_key})..."):
            df_scope = fetch_gps_summary_scope_cached(access_token, user_id, scope_key)
        st.dataframe(df_scope, width="stretch", height=520)

# =========================
# FFP (ALWAYS ALL, lazy-load)
# =========================
elif sub_page == "FFP":
    tabs = st.tabs(["Dashboard", "Benchmarks", "Data preview (ALL)"])

    with tabs[0]:
        with st.spinner("FFP: Summary data laden (ALLES)..."):
            df_all = fetch_gps_summary_all_cached(access_token, user_id)

        if df_all.empty:
            st.info("Geen Summary GPS data gevonden.")
            st.stop()

        ffp_pages.ffp_pages_main(df_all)

    with tabs[1]:
        benchmarks_pages.benchmarks_pages_main(
            supabase_url=SUPABASE_URL,
            supabase_anon_key=SUPABASE_ANON_KEY,
            access_token=access_token,
            user_id=user_id,
        )

    with tabs[2]:
        with st.spinner("FFP: Summary data laden (ALLES)..."):
            df_all = fetch_gps_summary_all_cached(access_token, user_id)
        st.dataframe(df_all, width="stretch", height=520)
