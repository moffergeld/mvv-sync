# pages/07_GPS_Data.py
# ==========================================================
# GPS Data pagina (AUTO-LOAD + per-subpagina filters)
# - Subpagina's: Session Load, ACWR, FFP
# - Sub-subpagina's (tabs) per subpagina
# - Auto-load (geen knop nodig) + caching (st.cache_data)
# - Per subpagina eigen filters:
#     * Session Load: eigen date range + (optioneel) Summary-only toggle
#     * ACWR: ALWAYS Summary-only (forced)
#     * FFP: eigen date range + (optioneel) Summary-only toggle
#
# Vereist:
#   st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"]
#   st.session_state["access_token"] (JWT)
#
# Gebruikt modules:
#   session_load_pages.py  -> session_load_pages_main(df)
#   acwr_pages.py          -> acwr_pages_main(df)
#   ffp_pages.py           -> ffp_pages_main(df)
# ==========================================================

import re
from datetime import date

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
    r = requests.get(url, headers=rest_headers(access_token), timeout=60)
    if not r.ok:
        raise RuntimeError(f"GET {table} failed ({r.status_code}): {r.text}")
    return pd.DataFrame(r.json())


def auth_get_user(access_token: str) -> dict:
    url = f"{SUPABASE_URL}/auth/v1/user"
    r = requests.get(url, headers=rest_headers(access_token), timeout=30)
    if not r.ok:
        raise RuntimeError(f"AUTH user fetch failed ({r.status_code}): {r.text}")
    return r.json()


def toast_ok(msg: str) -> None:
    try:
        st.toast(msg, icon="✅")
    except Exception:
        st.success(msg)


def toast_err(msg: str) -> None:
    try:
        st.toast(msg, icon="❌")
    except Exception:
        st.error(msg)


def _safe_q(v: str) -> str:
    return requests.utils.quote(str(v), safe="")


def _norm_event(v) -> str:
    return str(v).strip().lower()


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

    # Numeriek
    skip = {"Datum", "Speler", "Type", "Event"}
    for c in [c for c in df.columns if c not in skip]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Event" in df.columns:
        df["Event"] = df["Event"].astype(str).str.strip()

    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_gps_df_cached(
    access_token: str,
    user_id: str,
    d_from_iso: str,
    d_to_iso: str,
    only_summary: bool,
    limit: int,
) -> pd.DataFrame:
    """
    Cache key bevat user_id + params.
    access_token zit in args omdat we hem nodig hebben voor headers; ttl maakt dit praktisch.
    """
    d_from = date.fromisoformat(d_from_iso)
    d_to = date.fromisoformat(d_to_iso)

    q = (
        f"select={','.join(GPS_SELECT_COLS)}"
        f"&datum=gte.{d_from.isoformat()}"
        f"&datum=lte.{d_to.isoformat()}"
        f"&order=datum.asc"
        f"&limit={int(limit)}"
    )

    # ✅ Summary filter: tolerant voor hoofdletters (we filteren server-side op "Summary" voor performance,
    # en client-side nogmaals tolerant voor oude varianten)
    if only_summary:
        q += f"&event=eq.{_safe_q('Summary')}"

    raw = rest_get(access_token, "gps_records", q)
    df = _supabase_to_dashboard_df(raw)

    # extra tolerant client-side
    if only_summary and not df.empty and "Event" in df.columns:
        df = df[df["Event"].map(_norm_event) == "summary"].copy()

    return df


def load_df_for(
    access_token: str,
    user_id: str,
    d_from: date,
    d_to: date,
    only_summary: bool,
    limit: int,
    cache_key: str,
) -> pd.DataFrame:
    """
    Houdt ook per-subpagina een session_state kopie bij zodat navigeren instant voelt.
    """
    params = {
        "from": d_from.isoformat(),
        "to": d_to.isoformat(),
        "only_summary": bool(only_summary),
        "limit": int(limit),
    }

    last = st.session_state.get(f"{cache_key}_params")
    need_reload = last != params or f"{cache_key}_df" not in st.session_state

    if need_reload:
        with st.spinner("GPS data laden..."):
            df = fetch_gps_df_cached(
                access_token=access_token,
                user_id=user_id,
                d_from_iso=params["from"],
                d_to_iso=params["to"],
                only_summary=params["only_summary"],
                limit=params["limit"],
            )
        st.session_state[f"{cache_key}_df"] = df
        st.session_state[f"{cache_key}_params"] = params
    else:
        df = st.session_state.get(f"{cache_key}_df", pd.DataFrame())

    return df


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

# Subpagina navigatie (blijft staan bij reruns)
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
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.2])
        with c1:
            sl_from = st.date_input("Van", value=date.today().replace(day=1), key="sl_from")
        with c2:
            sl_to = st.date_input("Tot", value=date.today(), key="sl_to")
        with c3:
            sl_limit = st.number_input("Max rijen", 1000, 500000, 200000, 10000, key="sl_limit")
        with c4:
            sl_only_summary = st.toggle("Alleen Summary", value=False, key="sl_only_summary")

        if sl_from > sl_to:
            st.error("Van-datum kan niet na Tot-datum liggen.")
            st.stop()

    df_sl = load_df_for(
        access_token=access_token,
        user_id=user_id,
        d_from=sl_from,
        d_to=sl_to,
        only_summary=sl_only_summary,
        limit=int(sl_limit),
        cache_key="sl",
    )

    if df_sl.empty:
        st.info("Geen GPS data gevonden voor deze filters.")
        st.stop()

    tabs = st.tabs(["Dashboard", "Data preview"])
    with tabs[0]:
        session_load_pages.session_load_pages_main(df_sl)
    with tabs[1]:
        st.dataframe(df_sl, use_container_width=True, height=520)

# =========================
# ACWR (forced Summary-only)
# =========================
elif sub_page == "ACWR":
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 1.4])
        with c1:
            acwr_from = st.date_input("Van", value=date.today().replace(day=1), key="acwr_from")
        with c2:
            acwr_to = st.date_input("Tot", value=date.today(), key="acwr_to")
        with c3:
            acwr_limit = st.number_input("Max rijen", 1000, 500000, 200000, 10000, key="acwr_limit")

        st.caption("ACWR gebruikt altijd Event='Summary' (automatisch geforceerd).")

        if acwr_from > acwr_to:
            st.error("Van-datum kan niet na Tot-datum liggen.")
            st.stop()

    df_acwr = load_df_for(
        access_token=access_token,
        user_id=user_id,
        d_from=acwr_from,
        d_to=acwr_to,
        only_summary=True,  # ✅ forced
        limit=int(acwr_limit),
        cache_key="acwr",
    )

    if df_acwr.empty:
        st.info("Geen Summary GPS data gevonden voor deze periode.")
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
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.2])
        with c1:
            ffp_from = st.date_input("Van", value=date.today().replace(day=1), key="ffp_from")
        with c2:
            ffp_to = st.date_input("Tot", value=date.today(), key="ffp_to")
        with c3:
            ffp_limit = st.number_input("Max rijen", 1000, 500000, 200000, 10000, key="ffp_limit")
        with c4:
            ffp_only_summary = st.toggle("Alleen Summary", value=False, key="ffp_only_summary")

        if ffp_from > ffp_to:
            st.error("Van-datum kan niet na Tot-datum liggen.")
            st.stop()

    df_ffp = load_df_for(
        access_token=access_token,
        user_id=user_id,
        d_from=ffp_from,
        d_to=ffp_to,
        only_summary=ffp_only_summary,
        limit=int(ffp_limit),
        cache_key="ffp",
    )

    if df_ffp.empty:
        st.info("Geen GPS data gevonden voor deze filters.")
        st.stop()

    tabs = st.tabs(["Dashboard", "Data preview"])
    with tabs[0]:
        ffp_pages.ffp_pages_main(df_ffp)
    with tabs[1]:
        st.dataframe(df_ffp, use_container_width=True, height=520)
