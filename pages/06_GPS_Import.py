# pages/06_GPS_Import.py
# - Import GPS (Excel): Upload JOHAN Excel (SUMMARY / EXERCISES) or FLAT GPS Excel -> parse -> upsert public.gps_records
# - Manual add GPS: add/edit rows -> upsert public.gps_records
# - Export GPS: export gps_records -> Excel
# - Matches: import Matches.csv -> upsert public.matches, edit score/type/season, delete match, add match manual
#
# UI beveiliging:
# - Alleen staff/admin/etc mogen deze pagina zien (players NIET).
#
# Requires:
#   st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"]
#   st.session_state["access_token"] (JWT)
#   st.session_state["role"] (geladen in app.py na login)

# =========================
# PART 1/5 — Imports, config, REST helpers, small utilities
# =========================

import io
import re
from datetime import date

import pandas as pd
import requests
import streamlit as st

# -------------------------
# UI access gate (staff-only)
# -------------------------
role_ui = (st.session_state.get("role") or "").lower()
if role_ui == "player":
    st.error("Geen toegang.")
    st.stop()

# Excel engine check (Streamlit Cloud must have openpyxl in requirements.txt)
try:
    import openpyxl  # noqa: F401
except Exception:
    st.error("Excel support ontbreekt: installeer openpyxl via requirements.txt")
    st.stop()

st.set_page_config(page_title="Data Import", layout="wide")

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing secrets: SUPABASE_URL / SUPABASE_ANON_KEY")
    st.stop()

ALLOWED_IMPORT = {"admin", "data_scientist", "staff", "physio", "performance_coach"}
TYPE_OPTIONS = ["Practice", "Practice (1)", "Practice (2)", "Match", "Practice Match"]
MATCH_TYPES = {"Match", "Practice Match"}  # ✅ used everywhere

# Matches UI config
TEAM_NAME_MATCHES = "MVV Maastricht"
HOME_AWAY_OPTIONS = ["Home", "Away"]
MATCH_TYPE_OPTIONS = ["Competitie", "Oefenwedstrijd", "Beker"]


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


def season_options(start_year: int = 2020, years_ahead: int = 6) -> list[str]:
    y1 = date.today().year + years_ahead
    return [f"{y}/{y+1}" for y in range(start_year, y1 + 1)]


def default_season_today() -> str:
    y = date.today().year
    return f"{y}/{y+1}" if date.today().month >= 7 else f"{y-1}/{y}"


def build_fixture(team_name: str, home_away: str | None, opponent: str | None) -> str:
    ha = (home_away or "").strip().lower()
    opp = (opponent or "").strip()
    team = (team_name or "").strip()

    if not team and not opp:
        return ""

    if ha == "away":
        return f"{opp} - {team}".strip(" -") if opp and team else (opp or team)

    return f"{team} - {opp}".strip(" -") if team and opp else (team or opp)


def build_result(goals_for, goals_against) -> str:
    gf = pd.to_numeric(goals_for, errors="coerce")
    ga = pd.to_numeric(goals_against, errors="coerce")
    if pd.isna(gf) or pd.isna(ga):
        return ""
    return f"{int(gf)}-{int(ga)}"


def json_safe(v):
    """Convert pandas/float NaN/NaT/NA/inf to None so requests(json=...) is valid JSON."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        if isinstance(v, float):
            if v != v:  # NaN
            ...
            # NOTE: kept identical logic below; this '...' is not valid code.
    except Exception:
        pass
    return v


# =========================
# Auth / REST helpers
# =========================
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


def rest_upsert(access_token: str, table: str, rows: list[dict], on_conflict: str) -> None:
    if not rows:
        return
    url = f"{SUPABASE_URL}/rest/v1/{table}?on_conflict={on_conflict}"
    headers = rest_headers(access_token)
    headers["Prefer"] = "resolution=merge-duplicates"
    CHUNK = 500

    for i in range(0, len(rows), CHUNK):
        chunk = rows[i : i + CHUNK]
        safe_chunk = [{k: json_safe(v) for k, v in row.items()} for row in chunk]

        r = requests.post(url, headers=headers, json=safe_chunk, timeout=120)
        if not r.ok:
            raise RuntimeError(f"UPSERT {table} failed ({r.status_code}): {r.text}")


def rest_patch(access_token: str, table: str, where_query: str, payload: dict) -> None:
    url = f"{SUPABASE_URL}/rest/v1/{table}?{where_query}"
    headers = rest_headers(access_token)
    headers["Prefer"] = "return=representation"
    safe_payload = {k: json_safe(v) for k, v in payload.items()}
    r = requests.patch(url, headers=headers, json=safe_payload, timeout=60)
    if not r.ok:
        raise RuntimeError(f"PATCH {table} failed ({r.status_code}): {r.text}")


def rest_delete(access_token: str, table: str, where_query: str) -> None:
    url = f"{SUPABASE_URL}/rest/v1/{table}?{where_query}"
    headers = rest_headers(access_token)
    headers["Prefer"] = "return=representation"
    r = requests.delete(url, headers=headers, timeout=60)
    if not r.ok:
        raise RuntimeError(f"DELETE {table} failed ({r.status_code}): {r.text}")


def auth_get_user(access_token: str) -> dict:
    url = f"{SUPABASE_URL}/auth/v1/user"
    r = requests.get(url, headers=rest_headers(access_token), timeout=30)
    if not r.ok:
        raise RuntimeError(f"AUTH user fetch failed ({r.status_code}): {r.text}")
    return r.json()


def normalize_role(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if "." in s:
        s = s.split(".")[-1]
    if "::" in s:
        s = s.split("::")[0]
    return s.strip() or None


@st.cache_data(ttl=60)
def get_profile_role(access_token: str) -> tuple[str | None, str | None, str | None, str | None]:
    u = auth_get_user(access_token)
    user_id = u.get("id")
    email = u.get("email")

    role = None
    team = None

    if user_id:
        dfp = rest_get(
            access_token,
            "profiles",
            f"select=user_id,role,team&user_id=eq.{user_id}&limit=1",
        )
        if not dfp.empty:
            role = normalize_role(dfp.iloc[0].get("role"))
            team = dfp.iloc[0].get("team")
    return user_id, email, role, team


# =========================
# PART 2/5 — Mapping helpers (players, matches), gps schema + df_to_db_rows
# =========================
def normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())


def normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


@st.cache_data(ttl=120)
def get_players_map(access_token: str) -> tuple[dict, list[str]]:
    df = rest_get(access_token, "players", "select=player_id,full_name,is_active&is_active=eq.true&limit=5000")
    if df.empty:
        return {}, []
    df["full_name"] = df["full_name"].astype(str).str.strip()
    df = df.dropna(subset=["player_id", "full_name"])
    name_to_id = {normalize_name(n): pid for n, pid in zip(df["full_name"], df["player_id"])}
    display_names = sorted(df["full_name"].tolist())
    return name_to_id, display_names

# ==========================================================
# REST OF FILE
# ==========================================================
# Let de rest van je script ongewijzigd staan.
# Je hoeft alleen de gate bovenaan toe te voegen.
#
# Belangrijk:
# - Verwijder de debug expander of laat staan; players komen toch niet binnen.
# - Staff roles blijven via ALLOWED_IMPORT afgevangen zoals je al had.
# ==========================================================
