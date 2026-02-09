# pages/07_GPS_Data.py
# ==========================================================
# GPS Data pagina (AUTO-LOAD + per-subpagina)
# - Subpagina's: Session Load, ACWR, FFP
# - Auto-load + caching (st.cache_data)
# - GEEN date range / limit UI meer (laadt alles via pagination)
# - Session Load: kalender in session_load_pages.py is de enige dag-filter
# - ACWR: forced Summary-only (via v_gps_summary)
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


def rest_get_range(access_token: str, table: str, query: str, start: int, end: int) -> tuple[list[dict], str | None]:
    """
    Paginated fetch using Range header.
    Returns: (json_rows, content_range_header)
    """
    url = f"{SUPABASE_URL}/rest/v1/{table}?{query}"
    headers = rest_headers(access_token)
    headers["Range"] = f"{start}-{end}"
    # ask PostgREST to include Content-Range (optional but helpful)
    headers["Prefer"] = "count=exact"
    r = requests.get(url, headers=headers, timeout=180)
    if not r.ok:
        raise RuntimeError(f"GET {table} failed ({r.status_code}): {r.text}")
    return r.json(), r.headers.get("content-range")


def auth_get_user(access_token: str) -> dict:
    url = f"{SUPABASE_URL}/auth/v1/user"
    r = requests.get(url, headers=rest_headers(access_token), timeout=30)
    if not r.ok:
        raise RuntimeError(f"AUTH user fetch failed ({r.status_code}): {r.text}")
    return r.json()


# -------------------------
# Supabase -> Dashboard DF mapping
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

    # numeric
    num_cols = [c for c in df.columns if c not in ["Datum", "Speler", "Type", "Event"]]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Event" in df.columns:
        df["Event"] = df["Event"].astype(str).str.strip()

    if "Type" in df.columns:
        df["Type"] = df["Type"].astype(str).str.strip()

    return df


def fetch_all_rows_paginated(
    access_token: str,
    table: str,
    select_cols: list[str],
    order_col: str = "datum",
    batch_size: int = 5000,
) -> pd.DataFrame:
    """
    Fetch ALL rows from a table/view via pagination (Range headers),
    bypassing Supabase default max-rows (1000).
    """
    q = f"select={','.join(select_cols)}&order={order_col}.asc"

    all_rows: list[dict] = []
    start = 0

    while True:
        end = start + batch_size - 1
        chunk, _cr = rest_get_range(access_token, table, q, start=start, end=end)
        if not chunk:
            break
        all_rows.extend(chunk)
        if len(chunk) < batch_size:
            break
        start += batch_size

    return pd.DataFrame(all_rows)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_view_summary_cached(access_token: str, user_id: str) -> pd.DataFrame:
    # Session Load + ACWR -> v_gps_summary
    raw = fetch_all_rows_paginated(
        access_token=access_token,
        table="v_gps_summary",
        select_cols=GPS_SELECT_COLS,   # view should expose same names; if not, adjust
        order_col="datum",
        batch_size=5000,
    )
    return _supabase_to_dashboard_df(raw)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_gps_records_cached(access_token: str, user_id: str) -> pd.DataFrame:
    # For FFP (or if you want everything incl events)
    raw = fetch_all_rows_paginated(
        access_token=access_token,
        table="gps_records",
        select_cols=GPS_SELECT_COLS,
        order_col="datum",
        batch_size=5000,
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

try:
    u = auth_get_user(access_token)
    user_id = u.get("id") or "unknown"
except Exception as e:
    st.error(f"Kon user niet ophalen: {e}")
    st.stop()

sub_page = st.radio(
    "Subpagina",
    options=["Session Load", "ACWR", "FFP"],
    horizontal=True,
    key="gpsdata_subpage",
    label_visibility="collapsed",
)

st.divider()

# Load data (cached)
if sub_page in ("Session Load", "ACWR"):
    with st.spinner("GPS Summary laden..."):
        df_summary = fetch_view_summary_cached(access_token=access_token, user_id=user_id)

    if df_summary.empty:
        st.info("Geen data gevonden in v_gps_summary.")
        st.stop()

    tabs = st.tabs(["Dashboard", "Data preview"])
    with tabs[0]:
        if sub_page == "Session Load":
            session_load_pages.session_load_pages_main(df_summary)
        else:
            # ACWR expects Summary only -> already summary view
            acwr_pages.acwr_pages_main(df_summary)

    with tabs[1]:
        st.dataframe(df_summary, use_container_width=True, height=520)

else:
    # FFP
    with st.spinner("GPS data laden..."):
        df_all = fetch_gps_records_cached(access_token=access_token, user_id=user_id)

    if df_all.empty:
        st.info("Geen GPS data gevonden in gps_records.")
        st.stop()

    tabs = st.tabs(["Dashboard", "Data preview"])
    with tabs[0]:
        ffp_pages.ffp_pages_main(df_all)
    with tabs[1]:
        st.dataframe(df_all, use_container_width=True, height=520)
