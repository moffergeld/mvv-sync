# pages/07_GPS_Data.py
# ==========================================================
# GPS Data pagina (AUTO-LOAD + per-subpagina)
# - Subpagina's: Session Load, ACWR, FFP
# - Auto-load + caching
# - Laadt ALLE gps_records via limit+offset pagination (robust)
# - Session Load: kalender in session_load_pages.py is de enige dag-filter
# - ACWR: forced Summary-only
# ==========================================================

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

import session_load_pages
import acwr_pages
import ffp_pages

st.set_page_config(page_title="GPS Data", layout="wide")

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
        return tok
    sess = st.session_state.get("sb_session")
    if sess is not None:
        token = getattr(sess, "access_token", None)
        if token:
            return token
    return None


def rest_headers(access_token: str) -> dict:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def auth_get_user(access_token: str) -> dict:
    url = f"{SUPABASE_URL}/auth/v1/user"
    r = requests.get(url, headers=rest_headers(access_token), timeout=30)
    if not r.ok:
        raise RuntimeError(f"AUTH user fetch failed ({r.status_code}): {r.text}")
    return r.json()


def rest_get_limit_offset(
    access_token: str,
    table: str,
    base_query: str,
    limit: int,
    offset: int,
    timeout: int = 180,
) -> list[dict]:
    """
    Supabase PostgREST pagination via limit+offset (robust).
    base_query: query WITHOUT limit/offset.
    """
    url = f"{SUPABASE_URL}/rest/v1/{table}?{base_query}&limit={int(limit)}&offset={int(offset)}"
    r = requests.get(url, headers=rest_headers(access_token), timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"GET {table} failed ({r.status_code}): {r.text}")
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError("Unexpected response (expected list).")
    return data


# -------------------------
# Supabase fetch -> DataFrame voor dashboards
# -------------------------
GPS_SELECT_COLS = [
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
    if df.empty:
        return df

    df = df.copy()
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")

    rename_map = {
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

    num_cols = [c for c in df.columns if c not in ["Datum", "Speler", "Type", "Event"]]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Event" in df.columns:
        df["Event"] = df["Event"].astype(str).str.strip()
    if "Type" in df.columns:
        df["Type"] = df["Type"].astype(str).str.strip()

    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_gps_df_all_cached(access_token: str, user_id: str) -> pd.DataFrame:
    """
    Laadt ALLE gps_records via limit+offset tot batch leeg is.
    """
    page_size = 10000
    max_rows = 1000000  # safety
    base_query = f"select={','.join(GPS_SELECT_COLS)}&order=datum.asc"

    all_rows: list[dict] = []
    offset = 0

    while True:
        batch = rest_get_limit_offset(
            access_token=access_token,
            table="gps_records",
            base_query=base_query,
            limit=page_size,
            offset=offset,
            timeout=180,
        )
        if not batch:
            break

        all_rows.extend(batch)
        offset += page_size

        if len(all_rows) >= max_rows:
            break

    return _supabase_to_dashboard_df(pd.DataFrame(all_rows))


# -------------------------
# UI
# -------------------------
st.title("GPS Data")

access_token = get_access_token()
if not access_token:
    st.error("Niet ingelogd (access_token ontbreekt).")
    st.stop()

try:
    u = auth_get_user(access_token)
    user_id = u.get("id") or "unknown"
except Exception as e:
    st.error(f"Kon user niet ophalen: {e}")
    st.stop()

with st.spinner("GPS data laden..."):
    df_all = fetch_gps_df_all_cached(access_token=access_token, user_id=user_id)

with st.expander("Debug (data)", expanded=False):
    st.write("Rows:", len(df_all))
    if "Datum" in df_all.columns and not df_all.empty:
        st.write("Min/Max Datum:", df_all["Datum"].min(), df_all["Datum"].max())
    if "Event" in df_all.columns and not df_all.empty:
        st.write("Event counts:", df_all["Event"].astype(str).value_counts().head(10))

if df_all.empty:
    st.info("Geen GPS data gevonden in Supabase.")
    st.stop()

sub_page = st.radio(
    "Subpagina",
    options=["Session Load", "ACWR", "FFP"],
    horizontal=True,
    key="gpsdata_subpage",
    label_visibility="collapsed",
)

st.divider()

if sub_page == "Session Load":
    tabs = st.tabs(["Dashboard", "Data preview"])
    with tabs[0]:
        session_load_pages.session_load_pages_main(df_all)
    with tabs[1]:
        st.dataframe(df_all, use_container_width=True, height=520)

elif sub_page == "ACWR":
    if "Event" not in df_all.columns:
        st.info("Geen Event-kolom beschikbaar; ACWR vereist Summary.")
        st.stop()

    df_acwr = df_all[df_all["Event"].astype(str).str.strip().str.lower() == "summary"].copy()
    if df_acwr.empty:
        st.info("Geen Summary GPS data gevonden.")
        st.stop()

    tabs = st.tabs(["ACWR module", "Data preview"])
    with tabs[0]:
        acwr_pages.acwr_pages_main(df_acwr)
    with tabs[1]:
        st.dataframe(df_acwr, use_container_width=True, height=520)

elif sub_page == "FFP":
    tabs = st.tabs(["Dashboard", "Data preview"])
    with tabs[0]:
        ffp_pages.ffp_pages_main(df_all)
    with tabs[1]:
        st.dataframe(df_all, use_container_width=True, height=520)
