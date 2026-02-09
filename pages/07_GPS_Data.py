# pages/07_GPS_Data.py
# ==========================================================
# GPS Data pagina (AUTO-LOAD + per-subpagina)
# - Subpagina's: Session Load, ACWR, FFP
# - Auto-load (geen knop nodig) + caching (st.cache_data)
# - GEEN date range / limit UI meer (laadt alles)
# - Session Load: kalender in session_load_pages.py is de enige dag-filter
# - ACWR: forced Summary-only (robust: ook "Summary (2)" etc.)
# - FFP: laadt alles (pas aan indien je Summary-only wil)
#
# Vereist:
#   st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"]
#   st.session_state["access_token"] (JWT) óf st.session_state["sb_session"].access_token
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


def rest_get(access_token: str, table: str, query: str) -> pd.DataFrame:
    url = f"{SUPABASE_URL}/rest/v1/{table}?{query}"
    r = requests.get(url, headers=rest_headers(access_token), timeout=120)
    if not r.ok:
        raise RuntimeError(f"GET {table} failed ({r.status_code}): {r.text}")
    return pd.DataFrame(r.json())


def auth_get_user(access_token: str) -> dict:
    url = f"{SUPABASE_URL}/auth/v1/user"
    r = requests.get(url, headers=rest_headers(access_token), timeout=30)
    if not r.ok:
        raise RuntimeError(f"AUTH user fetch failed ({r.status_code}): {r.text}")
    return r.json()


def _safe_q(v: str) -> str:
    return requests.utils.quote(str(v), safe="")


def _event_is_summary(series: pd.Series) -> pd.Series:
    """
    Robust Summary-detectie:
    - accepteert 'Summary'
    - accepteert 'Summary (2)', 'Summary (3)' etc.
    """
    s = series.astype(str).str.strip().str.lower()
    return s.str.startswith("summary")


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
    """
    Mapt Supabase gps_records kolommen naar de kolomnamen die jouw dashboards verwachten.
    """
    if df.empty:
        return df

    df = df.copy()

    if "datum" in df.columns:
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

    # Numeriek coercen
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
    Laadt ALLE gps_records (zonder datumfilter UI).
    Let op:
      - PostgREST heeft server-side limieten/performance afhankelijk van je dataset.
      - limit verhoog je hier indien nodig.
      - user_id zit in args zodat cache-key per gebruiker uniek kan zijn.
    """
    limit = 500000  # pas aan als nodig

    q = (
        f"select={','.join(GPS_SELECT_COLS)}"
        f"&order=datum.asc"
        f"&limit={int(limit)}"
    )
    raw = rest_get(access_token, "gps_records", q)
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

# Basic sanity checks
required_cols = ["Datum", "Speler", "Type", "Event"]
missing = [c for c in required_cols if c not in df_all.columns]
if missing:
    st.error(f"GPS data mist kolommen na mapping: {missing}")
    st.stop()

# Drop rows zonder datum of speler om downstream issues te voorkomen
df_all = df_all.dropna(subset=["Datum", "Speler"]).copy()
if df_all.empty:
    st.info("Geen bruikbare GPS data gevonden (na opschonen).")
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
        # Session Load kalender gebruikt df_all (ALLE data) voor dagkleuren
        # en haalt zelf Summary data eruit voor de grafieken.
        session_load_pages.session_load_pages_main(df_all)
    with tabs[1]:
        st.dataframe(df_all, use_container_width=True, height=520)

# =========================
# ACWR (forced Summary-only)
# =========================
elif sub_page == "ACWR":
    df_acwr = df_all[_event_is_summary(df_all["Event"])].copy()

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
    # Alles (geen filters). Als je FFP ook Summary-only wil:
    # df_ffp = df_all[_event_is_summary(df_all["Event"])].copy()
    df_ffp = df_all

    if df_ffp.empty:
        st.info("Geen GPS data gevonden.")
        st.stop()

    tabs = st.tabs(["Dashboard", "Data preview"])
    with tabs[0]:
        ffp_pages.ffp_pages_main(df_ffp)
    with tabs[1]:
        st.dataframe(df_ffp, use_container_width=True, height=520)
