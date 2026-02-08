# pages/06_GPS_Import.py
# - Import GPS (Excel): Upload JOHAN Excel (SUMMARY / EXERCISES) -> parse -> upsert public.gps_records
# - Manual add GPS: add/edit rows -> upsert public.gps_records
# - Export GPS: export gps_records -> Excel
# - Matches: import Matches.csv -> upsert public.matches, edit score/type/season, delete match, add match manual
#
# Requires:
#   st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"]
#   st.session_state["access_token"] (JWT)

# =========================
# PART 1/5 — Imports, config, REST helpers, small utilities
# =========================

import io
import re
from datetime import date

import pandas as pd
import requests
import streamlit as st

# Excel engine check (Streamlit Cloud must have openpyxl in requirements.txt)
try:
    import openpyxl  # noqa: F401
except Exception:
    st.error("Excel support ontbreekt: installeer openpyxl via requirements.txt")
    st.stop()

st.set_page_config(page_title="GPS Import", layout="wide")

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing secrets: SUPABASE_URL / SUPABASE_ANON_KEY")
    st.stop()

ALLOWED_IMPORT = {"admin", "data_scientist", "staff", "physio", "performance_coach"}
TYPE_OPTIONS = ["Practice", "Practice (1)", "Practice (2)", "Match", "Practice Match"]

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
    # in NL start seizoen typisch juli/aug -> vanaf juli nemen y/y+1, anders y-1/y
    return f"{y}/{y+1}" if date.today().month >= 7 else f"{y-1}/{y}"


def build_fixture(team_name: str, home_away: str | None, opponent: str | None) -> str:
    ha = (home_away or "").strip().lower()
    opp = (opponent or "").strip()
    team = (team_name or "").strip()

    if not team and not opp:
        return ""

    if ha == "away":
        return f"{opp} - {team}".strip(" -") if opp and team else (opp or team)

    # default home
    return f"{team} - {opp}".strip(" -") if team and opp else (team or opp)


def build_result(goals_for, goals_against) -> str:
    gf = pd.to_numeric(goals_for, errors="coerce")
    ga = pd.to_numeric(goals_against, errors="coerce")
    if pd.isna(gf) or pd.isna(ga):
        return ""
    return f"{int(gf)}-{int(ga)}"


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
        r = requests.post(url, headers=headers, json=chunk, timeout=120)
        if not r.ok:
            raise RuntimeError(f"UPSERT {table} failed ({r.status_code}): {r.text}")


def rest_patch(access_token: str, table: str, where_query: str, payload: dict) -> None:
    """
    where_query example: "match_id=eq.27"
    """
    url = f"{SUPABASE_URL}/rest/v1/{table}?{where_query}"
    headers = rest_headers(access_token)
    headers["Prefer"] = "return=representation"
    r = requests.patch(url, headers=headers, json=payload, timeout=60)
    if not r.ok:
        raise RuntimeError(f"PATCH {table} failed ({r.status_code}): {r.text}")


def rest_delete(access_token: str, table: str, where_query: str) -> None:
    """
    where_query example: "match_id=eq.27"
    """
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


def parse_matches_csv(file_bytes: bytes) -> pd.DataFrame:
    # Matches.csv is ; separated
    df = pd.read_csv(io.BytesIO(file_bytes), sep=";")

    df = df.rename(
        columns={
            "Datum": "match_date",
            "Wedstrijd": "fixture",
            "Home/Away": "home_away",
            "Tegenstander": "opponent",
            "Type": "match_type",
            "Seizoen": "season",
            "Result": "result",
            "Goals for": "goals_for",
            "Goals against": "goals_against",
        }
    )

    df["match_date"] = pd.to_datetime(df["match_date"], dayfirst=True, errors="coerce").dt.date
    if df["match_date"].isna().any():
        bad = df.loc[df["match_date"].isna()].head(5)
        raise ValueError(f"Kon Datum niet parsen in Matches.csv. Voorbeelden:\n{bad}")

    for c in ["goals_for", "goals_against"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    for c in ["fixture", "home_away", "opponent", "match_type", "season", "result"]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace("nan", "").str.strip()

    # force fixture/result based on home_away+opponent and goals
    df["fixture"] = df.apply(lambda r: build_fixture(TEAM_NAME_MATCHES, r.get("home_away"), r.get("opponent")), axis=1)
    df["result"] = df.apply(lambda r: build_result(r.get("goals_for"), r.get("goals_against")), axis=1)

    keep = ["match_date", "fixture", "home_away", "opponent", "match_type", "season", "result", "goals_for", "goals_against"]
    return df[keep]


def matches_df_to_rows(df: pd.DataFrame, source_file: str) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "match_date": str(r["match_date"]),
                "fixture": r.get("fixture") or None,
                "home_away": r.get("home_away") or None,
                "opponent": r.get("opponent") or None,
                "match_type": r.get("match_type") or None,
                "season": r.get("season") or None,
                "result": r.get("result") or None,
                "goals_for": int(r["goals_for"]) if pd.notna(r.get("goals_for")) else None,
                "goals_against": int(r["goals_against"]) if pd.notna(r.get("goals_against")) else None,
                "source_file": source_file,
            }
        )
    return rows


def fetch_matches_on_date(access_token: str, d: date) -> pd.DataFrame:
    q = (
        "select=match_id,match_date,fixture,opponent,home_away,match_type,season,result,goals_for,goals_against"
        f"&match_date=eq.{d.isoformat()}"
        "&order=match_id.desc&limit=200"
    )
    return rest_get(access_token, "matches", q)


def fetch_matches_range(access_token: str, d_from: date, d_to: date, season_filter: str = "") -> pd.DataFrame:
    q = (
        "select=match_id,match_date,fixture,opponent,home_away,match_type,season,result,goals_for,goals_against"
        f"&match_date=gte.{d_from.isoformat()}"
        f"&match_date=lte.{d_to.isoformat()}"
        "&order=match_date.desc&limit=2000"
    )
    if season_filter.strip():
        q += f"&season=eq.{requests.utils.quote(season_filter.strip(), safe='')}"
    return rest_get(access_token, "matches", q)


@st.cache_data(ttl=30)
def fetch_gps_match_ids_on_date(access_token: str, d: date, match_type: str) -> pd.Series:
    """
    Gets match_id(s) present in gps_records for date+type (non-null).
    """
    t = requests.utils.quote(str(match_type), safe="")
    q = (
        "select=match_id"
        f"&datum=eq.{d.isoformat()}"
        f"&type=eq.{t}"
        "&match_id=is.not.null"
        "&limit=20000"
    )
    df = rest_get(access_token, "gps_records", q)
    if df.empty or "match_id" not in df.columns:
        return pd.Series(dtype="Int64")
    return pd.to_numeric(df["match_id"], errors="coerce").dropna().astype(int)


def resolve_match_id_for_date(access_token: str, d: date, match_type: str) -> tuple[int | None, pd.DataFrame]:
    """
    Resolving order:
      1) existing gps_records match_id mode for that date+type
      2) matches table on date:
          - exactly 1 match => auto
          - multiple => return None + df (caller can show dropdown)
    """
    s = fetch_gps_match_ids_on_date(access_token, d, match_type)
    if not s.empty:
        return int(s.value_counts().idxmax()), pd.DataFrame()

    dfm = fetch_matches_on_date(access_token, d)
    if dfm.empty:
        return None, dfm

    if dfm["match_id"].nunique() == 1:
        return int(pd.to_numeric(dfm["match_id"], errors="coerce").dropna().iloc[0]), dfm

    return None, dfm


# =========================
# gps_records columns
# =========================
GPS_COLS = [
    "player_id",
    "player_name",
    "datum",
    "week",
    "year",
    "type",
    "event",
    "match_id",
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
    "source_file",
]

METRIC_MAP = {
    "duration": "duration",
    "totaldistance": "total_distance",
    "walkdistance": "walking",
    "jogdistance": "jogging",
    "rundistance": "running",
    "sprintdistance": "sprint",
    "hisprintdistance": "high_sprint",
    "highsprintdistance": "high_sprint",
    "numberofsprints": "number_of_sprints",
    "numberofhisprints": "number_of_high_sprints",
    "numberofhighsprints": "number_of_high_sprints",
    "numberofrepeatedsprints": "number_of_repeated_sprints",
    "maxspeed": "max_speed",
    "avgspeed": "avg_speed",
    "playerload3d": "playerload3d",
    "playerload2d": "playerload2d",
    "totalaccelerations": "total_accelerations",
    "highaccelerations": "high_accelerations",
    "totaldecelerations": "total_decelerations",
    "highdecelerations": "high_decelerations",
    "hrzone1": "hrzone1",
    "hrzone2": "hrzone2",
    "hrzone3": "hrzone3",
    "hrzone4": "hrzone4",
    "hrzone5": "hrzone5",
    "hrtrimp": "hrtrimp",
    "hrzoneanaerobic": "hrzoneanaerobic",
    "avghr": "avg_hr",
    "maxhr": "max_hr",
}

ID_COLS_IN_PARSER = ["Speler", "Datum", "Week", "Year", "Type", "Event"]

INT_DB_COLS = {
    "number_of_sprints",
    "number_of_high_sprints",
    "number_of_repeated_sprints",
    "total_accelerations",
    "high_accelerations",
    "total_decelerations",
    "high_decelerations",
}


def drop_min_columns(df: pd.DataFrame) -> pd.DataFrame:
    min_cols = [c for c in df.columns if str(c).strip().endswith("/min")]
    return df.drop(columns=min_cols) if min_cols else df


def coerce_num(v):
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return None
    if pd.isna(v):
        return None
    if isinstance(v, str):
        v = v.replace(",", ".")
    num = pd.to_numeric(v, errors="coerce")
    return float(num) if pd.notna(num) else None


def df_to_db_rows(
    df: pd.DataFrame,
    source_file: str,
    name_to_id: dict,
    match_id: int | None = None,
) -> tuple[list[dict], list[str]]:
    rows = []
    unmapped = set()

    parsed_dates = pd.to_datetime(df["Datum"], dayfirst=True, errors="coerce")
    if parsed_dates.isna().any():
        bad = df.loc[parsed_dates.isna(), "Datum"].head(5).tolist()
        raise ValueError(f"Kon sommige Datum waarden niet parsen: {bad}")

    dates_iso = parsed_dates.dt.date.astype(str)

    for idx, r in df.iterrows():
        speler = str(r.get("Speler", "")).strip()
        if not speler:
            continue

        pid = name_to_id.get(normalize_name(speler))
        if not pid:
            unmapped.add(speler)

        dt = parsed_dates.iloc[idx].date()

        base = {
            "player_id": pid,
            "player_name": speler,
            "datum": dates_iso.iloc[idx],
            "week": int(dt.isocalendar().week),
            "year": int(dt.year),
            "type": str(r.get("Type", "")).strip(),
            "event": str(r.get("Event", "")).strip(),
            "match_id": match_id,
            "source_file": source_file,
        }

        for c in df.columns:
            if c in ID_COLS_IN_PARSER or str(c).strip().endswith("/min"):
                continue

            key = normalize_key(c)
            if key not in METRIC_MAP:
                continue

            db_col = METRIC_MAP[key]
            val = r[c]

            if db_col in INT_DB_COLS:
                v = pd.to_numeric(val, errors="coerce")
                base[db_col] = int(v) if pd.notna(v) else None
            else:
                base[db_col] = coerce_num(val)

        rows.append(base)

    return rows, sorted(unmapped)


# =========================
# PART 3/5 — JOHAN parsers (SUMMARY / EXERCISES) + unique Event
# =========================
def parse_summary_excel(file_bytes: bytes, selected_date: date, selected_type: str) -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(file_bytes), header=None)

    total_work_start = raw[raw[0] == "Total Work"].index[0]
    intensity_start = raw[raw[0] == "Intensity"].index[0]

    total_work_df = raw.iloc[total_work_start + 1 : intensity_start].dropna(how="all")
    total_work_df.columns = ["Variabele", "Eenheid"] + raw.iloc[0, 2:].tolist()
    total_work_df.set_index("Variabele", inplace=True)
    total_work_df = total_work_df.drop(columns=["Eenheid"])

    intensity_df = raw.iloc[intensity_start + 1 :].dropna(how="all")
    intensity_df.columns = ["Variabele", "Eenheid"] + raw.iloc[0, 2:].tolist()
    intensity_df.set_index("Variabele", inplace=True)
    intensity_df = intensity_df.drop(columns=["Eenheid"])

    intensity_df_renamed = intensity_df.copy()
    intensity_df_renamed.index = intensity_df_renamed.index + "/min"

    combined_df = pd.concat([total_work_df, intensity_df_renamed])
    combined_df.columns.name = None

    result_df = combined_df.transpose().reset_index().rename(columns={"index": "Speler"})

    result_df["Datum"] = pd.to_datetime(selected_date).strftime("%d-%m-%Y")
    result_df["Type"] = selected_type
    result_df["Event"] = "Summary"

    result_df = drop_min_columns(result_df)

    metric_cols = [c for c in result_df.columns if c not in ["Speler", "Datum", "Type", "Event"]]
    result_df[metric_cols] = result_df[metric_cols].fillna(0)

    dt = pd.to_datetime(selected_date)
    result_df["Week"] = int(dt.isocalendar().week)
    result_df["Year"] = int(dt.year)

    fixed = ["Speler", "Datum", "Week", "Year", "Type", "Event"]
    rest = [c for c in result_df.columns if c not in fixed]
    return result_df[fixed + rest]


def maak_lijst_uniek(lijst):
    seen = {}
    out = []
    for item in lijst:
        if item in seen:
            seen[item] += 1
            out.append(f"{item}_{seen[item]}")
        else:
            seen[item] = 1
            out.append(item)
    return out


def ensure_unique_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    keys = ["Speler", "Datum", "Type", "Event"]
    df["Event"] = df["Event"].astype(str).str.strip()

    idx = df.groupby(keys).cumcount()
    grp_size = df.groupby(keys)["Event"].transform("size")

    mask = grp_size > 1
    df.loc[mask, "Event"] = df.loc[mask, "Event"] + " (" + (idx[mask] + 1).astype(str) + ")"
    return df


def parse_exercises_excel(file_bytes: bytes, selected_date: date, selected_type: str) -> pd.DataFrame:
    xlsx = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = [s for s in xlsx.sheet_names if s.lower() != "spelerlijst"]

    alle = []
    for sheet in sheets:
        df = pd.read_excel(xlsx, sheet_name=sheet, header=None)

        speler = df.iloc[1, 0]
        oefenvormen = df.iloc[0, 2:].dropna().tolist()

        total_work_start = df[df[0] == "Total Work"].index[0]
        intensity_start = df[df[0] == "Intensity"].index[0]

        total_work_df = df.iloc[total_work_start + 1 : intensity_start].dropna(how="all")
        huidige_oefenvormen = oefenvormen[: total_work_df.shape[1] - 2]
        total_work_df.columns = maak_lijst_uniek(["Variabele", "Eenheid"] + huidige_oefenvormen)
        total_work_df.set_index("Variabele", inplace=True)

        intensity_df = df.iloc[intensity_start + 1 :].dropna(how="all")
        huidige_oefenvormen_i = oefenvormen[: intensity_df.shape[1] - 2]
        intensity_df.columns = maak_lijst_uniek(["Variabele", "Eenheid"] + huidige_oefenvormen_i)
        intensity_df.set_index("Variabele", inplace=True)

        for oef in [c for c in total_work_df.columns if c != "Eenheid"]:
            rec = {
                "Speler": speler,
                "Datum": pd.to_datetime(selected_date).strftime("%d-%m-%Y"),
                "Type": selected_type,
                "Event": str(oef).split("_")[0],
            }

            for var in total_work_df.index:
                rec[var] = total_work_df.at[var, oef]

            for var in intensity_df.index:
                if oef in intensity_df.columns:
                    rec[f"{var}/min"] = intensity_df.at[var, oef]

            alle.append(rec)

    out = pd.DataFrame(alle)
    out = drop_min_columns(out)

    metric_cols = [c for c in out.columns if c not in ["Speler", "Datum", "Type", "Event"]]
    out[metric_cols] = out[metric_cols].fillna(0)

    dt = pd.to_datetime(selected_date)
    out["Week"] = int(dt.isocalendar().week)
    out["Year"] = int(dt.year)

    fixed = ["Speler", "Datum", "Week", "Year", "Type", "Event"]
    rest = [c for c in out.columns if c not in fixed]

    out = ensure_unique_events(out)
    return out[fixed + rest]


# =========================
# PART 4/5 — Export helpers
# =========================
def fetch_all_gps_records(access_token: str, limit: int = 200000) -> pd.DataFrame:
    query = f"select={','.join(GPS_COLS)}&order=datum.desc&limit={limit}"
    return rest_get(access_token, "gps_records", query)


def df_to_excel_bytes_single(df: pd.DataFrame, sheet_name: str = "gps_records") -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return bio.getvalue()


def safe_sheet_name(name: str, used: set[str]) -> str:
    s = str(name).strip()
    s = re.sub(r"[:\\/?*\[\]]", "_", s)
    s = s[:31] if len(s) > 31 else s
    if not s:
        s = "Sheet"
    base = s
    i = 1
    while s in used:
        suffix = f"_{i}"
        s = (base[: 31 - len(suffix)] + suffix) if len(base) + len(suffix) > 31 else (base + suffix)
        i += 1
    used.add(s)
    return s


# =========================
# PART 5/5 — UI
# ✅ Subtabs onder Import GPS (radio)
# ✅ Confirm messages (toast) bij import/export/save/delete
# ✅ Geen “terugspringen” naar Import GPS
# ✅ Matches: select/edit/delete + confirm; import CSV; add manual
# ✅ Manual add: match_id volledig auto (op datum/type + existing gps_records)
# =========================

st.title("GPS Import")

access_token = get_access_token()
if not access_token:
    st.error("Niet ingelogd (access_token ontbreekt).")
    st.stop()

try:
    user_id, email, role, team = get_profile_role(access_token)
except Exception as e:
    st.error(f"Kon profiel/role niet ophalen: {e}")
    st.stop()

with st.expander("Debug (auth/role)", expanded=False):
    st.write({"email": email, "user_id": user_id, "role": role, "team": team})
    st.write({"session_role_raw": st.session_state.get("role")})

if role not in ALLOWED_IMPORT:
    st.error("Geen rechten voor GPS import/export.")
    st.stop()

name_to_id, player_options = get_players_map(access_token)

# Navigation (persists across reruns)
main_page = st.radio(
    "Navigatie",
    options=["Import GPS", "Matches"],
    horizontal=True,
    key="nav_main",
    label_visibility="collapsed",
)

# -------------------------
# PAGE: Import GPS
# -------------------------
if main_page == "Import GPS":
    sub_page = st.radio(
        "Sub",
        options=["Import (Excel)", "Manual add", "Export"],
        horizontal=True,
        key="nav_gps_sub",
        label_visibility="collapsed",
    )

    # -------------------------
    # SUB: Import (Excel)
    # -------------------------
    if sub_page == "Import (Excel)":
        st.subheader("Import JOHAN Excel")

        col1, col2, col3 = st.columns([1.2, 1.2, 2.6])
        with col1:
            selected_date = st.date_input("Datum", value=date.today(), key="gps_imp_date")
        with col2:
            selected_type = st.selectbox("Type", TYPE_OPTIONS, key="gps_imp_type")
        with col3:
            uploaded_files = st.file_uploader(
                "Upload Excel (je mag SUMMARY én EXERCISES tegelijk selecteren)",
                type=["xlsx", "xls"],
                accept_multiple_files=True,
                key="gps_imp_files",
            )

        selected_match_id = None
        if selected_type in ["Match", "Practice Match"]:
            try:
                df_matches = fetch_matches_on_date(access_token, selected_date)
            except Exception as e:
                toast_err(f"Kon matches niet ophalen: {e}")
                df_matches = pd.DataFrame()

            if df_matches.empty:
                st.warning("Geen match gevonden op deze datum in tabel matches.")
            else:
                df_matches = df_matches.copy()
                df_matches["label"] = df_matches.apply(
                    lambda r: f"#{int(r['match_id'])} | {(r.get('fixture') or '').strip()} | {build_result(r.get('goals_for'), r.get('goals_against'))}",
                    axis=1,
                )
                pick = st.selectbox("Koppel aan match", options=df_matches["label"].tolist(), key="gps_imp_match_pick")
                selected_match_id = int(df_matches.loc[df_matches["label"] == pick, "match_id"].iloc[0])

        if uploaded_files:
            st.divider()
            st.subheader("Preview / Parse")

            if st.button("Preview (parse)", type="secondary", key="gps_preview_btn"):
                all_parsed = []
                errors = []

                for up in uploaded_files:
                    filename = up.name
                    file_bytes = up.getvalue()

                    is_summary = "summary" in filename.lower()
                    is_exercises = "exercises" in filename.lower()

                    try:
                        if is_summary:
                            df_parsed = parse_summary_excel(file_bytes, selected_date, selected_type)
                            kind = "SUMMARY"
                        elif is_exercises:
                            df_parsed = parse_exercises_excel(file_bytes, selected_date, selected_type)
                            kind = "EXERCISES"
                        else:
                            try:
                                df_parsed = parse_summary_excel(file_bytes, selected_date, selected_type)
                                kind = "SUMMARY (auto)"
                            except Exception:
                                df_parsed = parse_exercises_excel(file_bytes, selected_date, selected_type)
                                kind = "EXERCISES (auto)"

                        all_parsed.append((filename, kind, df_parsed))
                    except Exception as e:
                        errors.append((filename, str(e)))

                st.session_state["gps_parsed_multi"] = all_parsed

                if errors:
                    st.error("Sommige bestanden konden niet geparsed worden:")
                    for fn, msg in errors:
                        st.write(f"- {fn}: {msg}")

                if all_parsed:
                    toast_ok(f"Preview bevestigd: parsed bestanden = {len(all_parsed)}")
                    for fn, kind, dfp in all_parsed:
                        st.markdown(f"**{fn}** — {kind} — rijen: {len(dfp)}")
                        st.dataframe(dfp.head(120), width="stretch")

            all_parsed = st.session_state.get("gps_parsed_multi", [])
            if all_parsed:
                st.divider()
                st.subheader("Import → Supabase (upsert)")

                if st.button("Import (upsert naar gps_records)", type="primary", key="gps_upsert_btn"):
                    try:
                        all_rows: list[dict] = []
                        all_unmapped: set[str] = set()

                        for (filename, kind, dfp) in all_parsed:
                            rows, unmapped = df_to_db_rows(
                                dfp,
                                source_file=filename,
                                name_to_id=name_to_id,
                                match_id=selected_match_id,
                            )
                            all_rows.extend(rows)
                            all_unmapped.update(unmapped)

                        if all_unmapped:
                            st.warning(
                                "Niet gematchte namen (player_id blijft NULL, import gaat wel door):\n- "
                                + "\n- ".join(sorted(list(all_unmapped))[:30])
                                + (f"\n... (+{len(all_unmapped)-30})" if len(all_unmapped) > 30 else "")
                            )

                        rest_upsert(
                            access_token,
                            "gps_records",
                            all_rows,
                            on_conflict="player_name,datum,type,event",
                        )

                        st.session_state["gps_parsed_multi"] = []
                        toast_ok(f"Import bevestigd: upserted rows = {len(all_rows)}")
                    except Exception as e:
                        toast_err(f"Import fout: {e}")

    # -------------------------
    # SUB: Manual add
    # -------------------------
    if sub_page == "Manual add":
        st.subheader("Manual add (tabel)")
        st.caption("Voeg rijen toe met het + icoon. match_id wordt automatisch gezet bij Match / Practice Match.")

        template_cols = [
            "player_name", "datum", "type", "event", "match_id",
            "duration", "total_distance", "walking", "jogging", "running", "sprint", "high_sprint",
            "number_of_sprints", "number_of_high_sprints", "number_of_repeated_sprints",
            "max_speed", "avg_speed", "playerload3d", "playerload2d",
            "total_accelerations", "high_accelerations", "total_decelerations", "high_decelerations",
            "hrzone1", "hrzone2", "hrzone3", "hrzone4", "hrzone5", "hrtrimp", "hrzoneanaerobic", "avg_hr", "max_hr",
            "source_file",
        ]

        if "manual_df" not in st.session_state:
            st.session_state["manual_df"] = pd.DataFrame(
                [{
                    "player_name": player_options[0] if player_options else "",
                    "datum": date.today(),
                    "type": "Practice",
                    "event": "Summary",
                    "match_id": None,
                    "source_file": "manual",
                }],
                columns=template_cols,
            )

        # Match pick (only if needed) is shown AFTER editor; we keep column disabled to avoid manual typing.
        colcfg = {
            "player_name": st.column_config.SelectboxColumn("player_name", options=player_options, required=True),
            "datum": st.column_config.DateColumn("datum", required=True),
            "type": st.column_config.SelectboxColumn("type", options=TYPE_OPTIONS, required=True),
            "event": st.column_config.TextColumn("event", required=True),
            "match_id": st.column_config.NumberColumn(
                "match_id",
                help="Wordt automatisch gevuld bij Match/Practice Match (op basis van datum).",
                step=1,
                disabled=True,
            ),
            "source_file": st.column_config.TextColumn("source_file"),
        }

        edited = st.data_editor(
            st.session_state["manual_df"],
            width="stretch",
            num_rows="dynamic",
            column_config=colcfg,
            key="manual_editor",
        )

        # -------- AUTO match_id (based on first row date+type + existing gps_records)
        df_preview = edited.copy()
        try:
            first_date = pd.to_datetime(df_preview.iloc[0].get("datum")).date()
        except Exception:
            first_date = date.today()

        first_type = str(df_preview.iloc[0].get("type") or "").strip()

        auto_match_id = None
        matches_on_date = pd.DataFrame()

        if first_type in ["Match", "Practice Match"]:
            auto_match_id, matches_on_date = resolve_match_id_for_date(access_token, first_date, first_type)

            # If multiple matches exist and no GPS match_id yet -> allow 1 dropdown choice
            if auto_match_id is None and matches_on_date is not None and not matches_on_date.empty:
                matches_on_date = matches_on_date.copy()
                matches_on_date["label"] = matches_on_date.apply(
                    lambda r: f"#{int(r['match_id'])} | {(r.get('fixture') or '').strip()} | {build_result(r.get('goals_for'), r.get('goals_against'))}",
                    axis=1,
                )
                pick_key = f"manual_match_pick_{first_date.isoformat()}_{first_type}"
                pick = st.selectbox(
                    "Kies match voor deze datum (wordt toegepast op alle rijen zonder match_id)",
                    options=matches_on_date["label"].tolist(),
                    key=pick_key,
                )
                auto_match_id = int(matches_on_date.loc[matches_on_date["label"] == pick, "match_id"].iloc[0])

            # fill missing match_id for all rows
            if auto_match_id is not None:
                cur = pd.to_numeric(df_preview.get("match_id"), errors="coerce")
                df_preview.loc[cur.isna(), "match_id"] = int(auto_match_id)

                # keep session consistent for next rerun
                st.session_state["manual_df"] = df_preview

        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("Reset table", key="manual_reset_btn"):
                st.session_state["manual_df"] = pd.DataFrame(
                    [{
                        "player_name": player_options[0] if player_options else "",
                        "datum": date.today(),
                        "type": "Practice",
                        "event": "Summary",
                        "match_id": None,
                        "source_file": "manual",
                    }],
                    columns=template_cols,
                )
                toast_ok("Reset bevestigd.")

        with colB:
            if st.button("Save rows (upsert)", type="primary", key="manual_save_btn"):
                try:
                    dfm = df_preview.copy()

                    # basic cleanup
                    dfm["player_name"] = dfm["player_name"].astype(str).str.strip()
                    dfm["event"] = dfm["event"].astype(str).str.strip()
                    dfm["type"] = dfm["type"].astype(str).str.strip()

                    dfm = dfm.dropna(subset=["player_name", "datum", "type", "event"])
                    dfm = dfm[(dfm["player_name"] != "") & (dfm["event"] != "")]
                    if dfm.empty:
                        toast_err("Geen geldige rijen om op te slaan.")
                        st.stop()

                    # dates
                    dt_series = pd.to_datetime(dfm["datum"], errors="coerce")
                    if dt_series.isna().any():
                        toast_err("Ongeldige datum in één of meer rijen.")
                        st.stop()

                    dfm["week"] = dt_series.dt.isocalendar().week.astype(int)
                    dfm["year"] = dt_series.dt.year.astype(int)
                    dfm["datum"] = dt_series.dt.date.astype(str)

                    # player_id mapping
                    dfm["player_id"] = dfm["player_name"].map(lambda x: name_to_id.get(normalize_name(x)))

                    # ensure match_id int/None
                    dfm["match_id"] = pd.to_numeric(dfm.get("match_id"), errors="coerce")
                    dfm["match_id"] = dfm["match_id"].map(lambda v: int(v) if pd.notna(v) else None)

                    # enforce match_id consistency per (datum,type) for match types
                    MATCH_TYPES = {"Match", "Practice Match"}
                    for (d_iso, t), g in dfm.groupby(["datum", "type"], dropna=False):
                        if t not in MATCH_TYPES:
                            continue
                        d_obj = pd.to_datetime(d_iso).date()

                        existing_ids = fetch_gps_match_ids_on_date(access_token, d_obj, t)
                        if not existing_ids.empty:
                            forced_id = int(existing_ids.value_counts().idxmax())
                        else:
                            forced_id, dfm_on_date = resolve_match_id_for_date(access_token, d_obj, t)
                            if forced_id is None:
                                toast_err(
                                    f"Geen unieke match gevonden op {d_iso} voor type '{t}'. "
                                    "Ga naar Matches en voeg/selecteer de match (of kies hier de match dropdown)."
                                )
                                st.stop()

                        dfm.loc[g.index, "match_id"] = forced_id

                    metric_keys = [
                        "duration", "total_distance", "walking", "jogging", "running", "sprint", "high_sprint",
                        "number_of_sprints", "number_of_high_sprints", "number_of_repeated_sprints",
                        "max_speed", "avg_speed", "playerload3d", "playerload2d",
                        "total_accelerations", "high_accelerations", "total_decelerations", "high_decelerations",
                        "hrzone1", "hrzone2", "hrzone3", "hrzone4", "hrzone5", "hrtrimp", "hrzoneanaerobic", "avg_hr", "max_hr"
                    ]

                    for c in metric_keys:
                        if c in dfm.columns:
                            dfm[c] = pd.to_numeric(dfm[c], errors="coerce")

                    rows = []
                    for _, r in dfm.iterrows():
                        row = {
                            "player_id": r.get("player_id"),
                            "player_name": r.get("player_name"),
                            "datum": r.get("datum"),
                            "week": int(r.get("week")) if pd.notna(r.get("week")) else None,
                            "year": int(r.get("year")) if pd.notna(r.get("year")) else None,
                            "type": r.get("type"),
                            "event": r.get("event"),
                            "match_id": r.get("match_id"),
                            "source_file": (r.get("source_file") or "manual"),
                        }

                        for k in metric_keys:
                            v = r.get(k)
                            if k in INT_DB_COLS:
                                vv = pd.to_numeric(v, errors="coerce")
                                row[k] = int(vv) if pd.notna(vv) else None
                            else:
                                row[k] = float(v) if pd.notna(v) else None

                        rows.append(row)

                    rest_upsert(
                        access_token,
                        "gps_records",
                        rows,
                        on_conflict="player_name,datum,type,event",
                    )

                    st.session_state["manual_df"] = dfm[template_cols].copy()
                    toast_ok(f"Save bevestigd: rows = {len(rows)}")
                except Exception as e:
                    toast_err(f"Save fout: {e}")

    # -------------------------
    # SUB: Export
    # -------------------------
    if sub_page == "Export":
        st.subheader("Export gps_records → Excel")

        st.markdown("### 1) Export alles")
        c1, c2 = st.columns([1.4, 2.6])
        with c1:
            limit = st.number_input(
                "Max rows (veiligheidslimiet)",
                min_value=1,
                max_value=500000,
                value=200000,
                step=10000,
                key="exp_all_limit",
            )
        with c2:
            st.caption("Klik op generate, daarna verschijnt de downloadknop.")

        if st.button("Generate export (ALL)", key="exp_all_btn"):
            try:
                df = fetch_all_gps_records(access_token, limit=int(limit))
                if df.empty:
                    st.warning("Geen data gevonden.")
                else:
                    ordered = [c for c in GPS_COLS if c in df.columns]
                    df = df[ordered]
                    xbytes = df_to_excel_bytes_single(df, sheet_name="gps_records")
                    st.session_state["export_all_bytes"] = xbytes
                    toast_ok(f"Export bevestigd: {len(df)} rijen klaar voor download.")
            except Exception as e:
                toast_err(str(e))

        if st.session_state.get("export_all_bytes"):
            st.download_button(
                "Download gps_records_ALL.xlsx",
                data=st.session_state["export_all_bytes"],
                file_name="gps_records_ALL.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="exp_all_dl",
            )

        st.divider()
        st.markdown("### 2) Export geselecteerde speler(s) (elk tabblad = speler)")

        export_players = st.multiselect(
            "Selecteer speler(s)",
            options=player_options if player_options else [],
            key="exp_sel_players",
        )

        date_col1, date_col2 = st.columns([1, 1])
        with date_col1:
            exp_from = st.date_input("Van datum", value=date.today().replace(day=1), key="exp_sel_from")
        with date_col2:
            exp_to = st.date_input("Tot datum", value=date.today(), key="exp_sel_to")

        if st.button("Generate export (selected players)", key="exp_sel_btn"):
            try:
                if not export_players:
                    toast_err("Selecteer minimaal 1 speler.")
                    st.stop()

                bio = io.BytesIO()
                used = set()
                with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                    for p in export_players:
                        pname = requests.utils.quote(str(p), safe="")
                        q = (
                            f"select={','.join(GPS_COLS)}"
                            f"&player_name=eq.{pname}"
                            f"&datum=gte.{exp_from.isoformat()}"
                            f"&datum=lte.{exp_to.isoformat()}"
                            f"&order=datum.asc"
                            f"&limit=200000"
                        )
                        dfp = rest_get(access_token, "gps_records", q)
                        ordered = [c for c in GPS_COLS if c in dfp.columns]
                        dfp = dfp[ordered] if not dfp.empty else pd.DataFrame(columns=GPS_COLS)
                        sheet = safe_sheet_name(p, used)
                        dfp.to_excel(writer, index=False, sheet_name=sheet)

                st.session_state["export_sel_bytes"] = bio.getvalue()
                toast_ok("Export bevestigd: bestand klaar voor download.")
            except Exception as e:
                toast_err(str(e))

        if st.session_state.get("export_sel_bytes"):
            st.download_button(
                "Download gps_records_SELECTED_PLAYERS.xlsx",
                data=st.session_state["export_sel_bytes"],
                file_name="gps_records_SELECTED_PLAYERS.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="exp_sel_dl",
            )

# -------------------------
# PAGE: Matches
# -------------------------
if main_page == "Matches":
    st.subheader("Matches")

    # 1) Import CSV
    st.markdown("### 1) Import Matches.csv → Supabase")

    matches_file = st.file_uploader("Upload Matches.csv", type=["csv"], key="matches_csv")
    if matches_file:
        b = matches_file.getvalue()
        fname = matches_file.name

        if st.button("Preview matches", key="m_prev_btn"):
            try:
                dfm = parse_matches_csv(b)
                st.session_state["matches_preview"] = dfm
                toast_ok(f"Preview bevestigd: {len(dfm)} rijen.")
                st.dataframe(dfm, width="stretch")
            except Exception as e:
                toast_err(str(e))

        dfm = st.session_state.get("matches_preview")
        if dfm is not None and not dfm.empty:
            if st.button("Import matches (upsert)", type="primary", key="m_import_btn"):
                try:
                    rows = matches_df_to_rows(dfm, source_file=fname)

                    # Requires UNIQUE constraint in Supabase:
                    # alter table public.matches add constraint matches_unique_key unique (match_date, fixture, season);
                    rest_upsert(access_token, "matches", rows, on_conflict="match_date,fixture,season")

                    st.session_state["matches_preview"] = None
                    toast_ok(f"Import bevestigd: matches = {len(rows)}")
                except Exception as e:
                    toast_err(f"Import fout: {e}")

    st.divider()

    # 2) Select/edit/delete
    st.markdown("### 2) Selecteer & pas score aan")

    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        d_from = st.date_input("Van", value=date.today().replace(month=1, day=1), key="m_edit_from")
    with c2:
        d_to = st.date_input("Tot", value=date.today(), key="m_edit_to")
    with c3:
        season_filter = st.selectbox(
            "Seizoen filter (optioneel)",
            options=["(alles)"] + season_options(start_year=2020, years_ahead=6),
            index=0,
            key="m_edit_season",
        )

    try:
        df_list = fetch_matches_range(access_token, d_from, d_to, "" if season_filter == "(alles)" else season_filter)
    except Exception as e:
        toast_err(f"Kon matches niet ophalen: {e}")
        df_list = pd.DataFrame()

    if df_list.empty:
        st.info("Geen matches gevonden in deze periode.")
    else:
        df_list = df_list.copy()
        df_list["label"] = df_list.apply(
            lambda r: f"{r.get('match_date')} | {(r.get('fixture') or '').strip()} | {build_result(r.get('goals_for'), r.get('goals_against'))}",
            axis=1,
        )

        pick = st.selectbox("Kies wedstrijd", options=df_list["label"].tolist(), key="m_pick_match")
        match_id = int(df_list.loc[df_list["label"] == pick, "match_id"].iloc[0])
        row = df_list[df_list["match_id"] == match_id].iloc[0].to_dict()

        st.markdown("**Aanpassen:** (alleen score + match type + season)")
        e1, e2, e3 = st.columns([1, 1, 1.2])
        with e1:
            goals_for = st.number_input("Goals for", min_value=0, step=1, value=int(row.get("goals_for") or 0), key="m_gf")
        with e2:
            goals_against = st.number_input("Goals against", min_value=0, step=1, value=int(row.get("goals_against") or 0), key="m_ga")
        with e3:
            match_type = st.selectbox(
                "Match type",
                options=MATCH_TYPE_OPTIONS,
                index=(MATCH_TYPE_OPTIONS.index(row.get("match_type")) if row.get("match_type") in MATCH_TYPE_OPTIONS else 0),
                key="m_mt",
            )

        seasons = season_options(start_year=2020, years_ahead=6)
        current_season = str(row.get("season") or "").strip()
        if current_season and current_season not in seasons:
            seasons = [current_season] + seasons

        season = st.selectbox(
            "Season",
            options=seasons,
            index=(seasons.index(current_season) if current_season in seasons else 0),
            key="m_season",
        )

        csave, cdel = st.columns([1, 1])
        with csave:
            if st.button("Save changes", type="primary", key="m_save_btn"):
                try:
                    payload = {
                        "goals_for": int(goals_for),
                        "goals_against": int(goals_against),
                        "result": build_result(goals_for, goals_against),
                        "match_type": match_type,
                        "season": season if season else None,
                    }
                    rest_patch(access_token, "matches", f"match_id=eq.{match_id}", payload)
                    toast_ok("Save bevestigd.")
                except Exception as e:
                    toast_err(f"Opslaan mislukt: {e}")

        with cdel:
            if st.button("Delete match", type="secondary", key="m_del_btn"):
                st.session_state["confirm_delete_match_id"] = match_id

        if st.session_state.get("confirm_delete_match_id") == match_id:
            st.warning("Bevestig verwijderen van deze wedstrijd. Dit kan niet ongedaan gemaakt worden.")
            y, n = st.columns([1, 1])
            with y:
                if st.button("Ja, verwijderen", type="primary", key="m_del_yes"):
                    try:
                        rest_delete(access_token, "matches", f"match_id=eq.{match_id}")
                        st.session_state["confirm_delete_match_id"] = None
                        toast_ok("Verwijderen bevestigd.")
                    except Exception as e:
                        toast_err(f"Verwijderen mislukt: {e}")
            with n:
                if st.button("Nee, annuleren", key="m_del_no"):
                    st.session_state["confirm_delete_match_id"] = None
                    toast_ok("Verwijderen geannuleerd.")

    st.divider()

    # 3) Add match manual
    st.markdown("### 3) Nieuwe wedstrijd handmatig toevoegen")

    a1, a2, a3 = st.columns([1, 1.4, 1])
    with a1:
        new_date = st.date_input("Match date", value=date.today(), key="m_new_date")
    with a2:
        new_opponent = st.text_input("Opponent", value="", key="m_new_opp")
    with a3:
        new_home_away = st.selectbox("Home/Away", options=HOME_AWAY_OPTIONS, key="m_new_ha")

    b1, b2 = st.columns([1, 1])
    with b1:
        new_match_type = st.selectbox("Match type", options=MATCH_TYPE_OPTIONS, key="m_new_type")
    with b2:
        seasons_new = season_options(start_year=2020, years_ahead=6)
        ds = default_season_today()
        idx = seasons_new.index(ds) if ds in seasons_new else 0
        new_season = st.selectbox("Season", options=seasons_new, index=idx, key="m_new_season")

    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        new_gf = st.number_input("Goals for", min_value=0, step=1, value=0, key="m_new_gf")
    with c2:
        new_ga = st.number_input("Goals against", min_value=0, step=1, value=0, key="m_new_ga")
    with c3:
        new_result = st.text_input("Result (optioneel override)", value="", key="m_new_result")

    auto_fixture = build_fixture(TEAM_NAME_MATCHES, new_home_away, new_opponent)
    st.text_input("Fixture (auto)", value=auto_fixture, disabled=True, key="m_new_fix")

    if st.button("Add match", type="primary", key="m_add_btn"):
        try:
            if not new_opponent.strip():
                toast_err("Opponent is verplicht.")
                st.stop()

            payload = [{
                "match_date": new_date.isoformat(),
                "home_away": new_home_away,
                "opponent": new_opponent.strip(),
                "fixture": auto_fixture,
                "match_type": new_match_type,
                "season": new_season.strip(),
                "goals_for": int(new_gf),
                "goals_against": int(new_ga),
                "result": (new_result.strip() if new_result.strip() else build_result(new_gf, new_ga)),
                "source_file": "manual",
            }]

            # Requires UNIQUE constraint:
            # alter table public.matches add constraint matches_unique_key unique (match_date, fixture, season);
            rest_upsert(access_token, "matches", payload, on_conflict="match_date,fixture,season")
            toast_ok("Toevoegen bevestigd.")
        except Exception as e:
            toast_err(f"Toevoegen mislukt: {e}")
