# pages/02_GPS_Data.py
# ==========================================================
# GPS Data pagina (AUTO-LOAD + per-subpagina)
# - Subpagina's: Session Load, ACWR, FFP
# - Auto-load (geen knop nodig) + caching (st.cache_data)
# - Laadt ALLE data via betrouwbare Range-pagination (PostgREST)
# - Session Load: kalender in session_load_pages.py is de enige dag-filter
# - ACWR: forced Summary-only
# - FFP: alles (pas aan indien nodig)
#
# UI beveiliging:
# - role=player heeft GEEN toegang tot deze pagina (staff-only)
#
# Vereist:
#   st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"]
#   st.session_state["access_token"] (JWT)
#   st.session_state["role"] (geladen in app.py na login)
#
# Gebruikt modules:
#   session_load_pages.py  -> session_load_pages_main(df)
#   acwr_pages.py          -> acwr_pages_main(df)
#   ffp_pages.py           -> ffp_pages_main(df)
# ==========================================================

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

import pages.Subscripts.session_load_pages
import pages.Subscripts.acwr_pages
import pages.Subscripts.ffp_pages

st.set_page_config(page_title="GPS Data", layout="wide")

# -------------------------
# UI access gate (staff-only)
# -------------------------
role = (st.session_state.get("role") or "").lower()
if role == "player":
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
        # zorgt vaker voor Content-Range header
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
    """
    Betrouwbare pagination via Range headers (PostgREST/Supabase).
    Vereist stabiele order in base_query (bv. &order=datum.asc,gps_id.asc).
    """
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
    if df.empty:
        return df

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

    # numeriek
    non_num = {"Datum", "Speler", "Type", "Event"}
    for c in df.columns:
        if c not in non_num and c != "gps_id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Event" in df.columns:
        df["Event"] = df["Event"].astype(str).str.strip()

    if "Type" in df.columns:
        df["Type"] = df["Type"].astype(str).str.strip()

    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_gps_df_all_cached(access_token: str, user_id: str) -> pd.DataFrame:
    """
    Laadt ALLE gps_records via Range-pagination.
    user_id zit in signature zodat cache per user apart is.
    """
    select = ",".join(GPS_SELECT_COLS)

    # ✅ Belangrijk: stabiele order (datum + unieke id)
    base_query = f"select={select}&order=datum.asc,gps_id.asc"

    raw = rest_get_paged(
        access_token=access_token,
        table="gps_records",
        base_query=base_query,
        page_size=5000,   # pas aan indien nodig
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

try:
    u = auth_get_user(access_token)
    user_id = u.get("id") or "unknown"
except Exception as e:
    st.error(f"Kon user niet ophalen: {e}")
    st.stop()

# Auto-load alles
with st.spinner("GPS data laden..."):
    df_all = fetch_gps_df_all_cached(access_token=access_token, user_id=user_id)

if df_all.empty:
    st.info("Geen GPS data gevonden in Supabase.")
    st.stop()

# Subpagina navigatie
sub_page = st.radio(
    "Subpagina",
    options=["Session Load", "ACWR", "FFP"],
    horizontal=True,
    key="gpsdata_subpage",
    label_visibility="collapsed",
)

st.divider()

# =========================
# SESSION LOAD
# =========================
if sub_page == "Session Load":
    tabs = st.tabs(["Dashboard", "Data preview"])
    with tabs[0]:
        session_load_pages.session_load_pages_main(df_all)
    with tabs[1]:
        st.dataframe(df_all, use_container_width=True, height=520)

# =========================
# ACWR (forced Summary-only)
# =========================
elif sub_page == "ACWR":
    if "Event" in df_all.columns:
        df_acwr = df_all[df_all["Event"].astype(str).str.strip().str.lower() == "summary"].copy()
    else:
        df_acwr = df_all.iloc[0:0].copy()

    if df_acwr.empty:
        st.info("Geen Summary GPS data gevonden.")
        st.stop()

    tabs = st.tabs(["ACWR module", "Data preview"])
    with tabs[0]:
        acwr_pages.acwr_pages_main(df_acwr)
    with tabs[1]:
        st.dataframe(df_acwr, use_container_width=True, height=520)

# =========================
# FFP
# =========================
elif sub_page == "FFP":
    tabs = st.tabs(["Dashboard", "Data preview"])
    with tabs[0]:
        ffp_pages.ffp_pages_main(df_all)
    with tabs[1]:
        st.dataframe(df_all, use_container_width=True, height=520)
