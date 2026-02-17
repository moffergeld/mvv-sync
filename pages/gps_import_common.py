# gps_import_common.py
# ============================================================
# Shared helpers for GPS Import suite
# NOTE: profiles.team column removed -> do NOT select it anywhere.
# ============================================================

from __future__ import annotations

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

# -------------------------
# Config / secrets
# -------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing secrets: SUPABASE_URL / SUPABASE_ANON_KEY")
    st.stop()

ALLOWED_IMPORT = {"admin", "data_scientist", "staff", "physio", "performance_coach"}
TYPE_OPTIONS = ["Practice", "Practice (1)", "Practice (2)", "Match", "Practice Match"]
MATCH_TYPES = {"Match", "Practice Match"}  # used everywhere

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
            if v != v:
                return None
            if v in (float("inf"), float("-inf")):
                return None
    except Exception:
        pass
    return v


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
def get_profile_role(access_token: str) -> tuple[str | None, str | None, str | None, None]:
    """
    profiles.team verwijderd -> return (user_id, email, role, team=None)
    """
    u = auth_get_user(access_token)
    user_id = u.get("id")
    email = u.get("email")

    role = None
    team = None

    if user_id:
        dfp = rest_get(
            access_token,
            "profiles",
            f"select=user_id,role&user_id=eq.{user_id}&limit=1",
        )
        if not dfp.empty:
            role = normalize_role(dfp.iloc[0].get("role"))
            team = None

    return user_id, email, role, team


# -------------------------
# Players mapping
# -------------------------
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


# -------------------------
# Matches helpers (used by GPS + Matches page)
# -------------------------
def parse_matches_csv(file_bytes: bytes) -> pd.DataFrame:
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
    t = requests.utils.quote(str(match_type), safe="")
    q = (
        "select=match_id"
        f"&datum=eq.{d.isoformat()}"
        f"&type=eq.{t}"
        "&match_id=is.not_null"
        "&limit=20000"
    )
    df = rest_get(access_token, "gps_records", q)
    if df.empty or "match_id" not in df.columns:
        return pd.Series(dtype="Int64")
    return pd.to_numeric(df["match_id"], errors="coerce").dropna().astype(int)


def resolve_match_id_for_date(access_token: str, d: date, match_type: str) -> tuple[int | None, pd.DataFrame]:
    if match_type not in MATCH_TYPES:
        return None, pd.DataFrame()

    s = fetch_gps_match_ids_on_date(access_token, d, match_type)
    if not s.empty:
        return int(s.value_counts().idxmax()), pd.DataFrame()

    dfm = fetch_matches_on_date(access_token, d)
    if dfm.empty:
        return None, dfm

    if dfm["match_id"].nunique() == 1:
        return int(pd.to_numeric(dfm["match_id"], errors="coerce").dropna().iloc[0]), dfm

    return None, dfm


def ui_pick_match_if_needed(access_token: str, d: date, match_type: str, key_prefix: str) -> int | None:
    if match_type not in MATCH_TYPES:
        return None

    auto_id, dfm = resolve_match_id_for_date(access_token, d, match_type)
    if auto_id is not None:
        return int(auto_id)

    if dfm is None or dfm.empty:
        st.warning(f"Geen match gevonden op {d.isoformat()} in tabel matches (match_id blijft leeg).")
        return None

    dfm = dfm.copy()
    dfm["label"] = dfm.apply(
        lambda r: f"#{int(r['match_id'])} | {(r.get('fixture') or '').strip()} | {build_result(r.get('goals_for'), r.get('goals_against'))}",
        axis=1,
    )
    pick_key = f"{key_prefix}_{d.isoformat()}_{match_type}"
    pick = st.selectbox(
        f"Kies match voor {d.isoformat()} ({match_type})",
        options=dfm["label"].tolist(),
        key=pick_key,
    )
    return int(dfm.loc[dfm["label"] == pick, "match_id"].iloc[0])


def apply_auto_match_ids_to_rows(access_token: str, rows: list[dict], ui_key_prefix: str) -> list[dict]:
    if not rows:
        return rows

    keys = sorted({(r.get("datum"), r.get("type")) for r in rows if r.get("type") in MATCH_TYPES and r.get("datum")})
    chosen: dict[tuple[str, str], int | None] = {}

    for d_iso, t in keys:
        try:
            d_obj = pd.to_datetime(d_iso).date()
        except Exception:
            continue
        mid = ui_pick_match_if_needed(access_token, d_obj, t, key_prefix=ui_key_prefix)
        chosen[(d_iso, t)] = mid

    for r in rows:
        k = (r.get("datum"), r.get("type"))
        if r.get("type") in MATCH_TYPES and k in chosen:
            r["match_id"] = chosen[k]
        else:
            r["match_id"] = None
    return rows


# -------------------------
# GPS schema + parsers
# -------------------------
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
    # extra aliases (zonder "distance")
    "walking": "walking",
    "jogging": "jogging",
    "running": "running",
    "sprint": "sprint",
    "highsprint": "high_sprint",
    "hisprint": "high_sprint",
    "walk": "walking",
    "jog": "jogging",
    "run": "running",
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


def normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())


def drop_min_columns(df: pd.DataFrame) -> pd.DataFrame:
    min_cols = [c for c in df.columns if str(c).strip().endswith("/min")]
    return df.drop(columns=min_cols) if min_cols else df


def coerce_num(v):
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, str):
        v = v.replace(",", ".")
    num = pd.to_numeric(v, errors="coerce")
    return float(num) if pd.notna(num) else None


def df_to_db_rows(df: pd.DataFrame, source_file: str, name_to_id: dict) -> tuple[list[dict], list[str]]:
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
        t = str(r.get("Type", "")).strip()
        ev = str(r.get("Event", "")).strip()

        base = {
            "player_id": pid,
            "player_name": speler,
            "datum": dates_iso.iloc[idx],
            "week": int(dt.isocalendar().week),
            "year": int(dt.year),
            "type": t,
            "event": ev,
            "match_id": None,
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


def is_flat_gps_excel(file_bytes: bytes) -> bool:
    try:
        df0 = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0, nrows=3)
        cols = {str(c).strip().lower() for c in df0.columns}
        return {"speler", "datum", "type", "event"}.issubset(cols)
    except Exception:
        return False


def parse_flat_gps_excel(file_bytes: bytes) -> pd.DataFrame:
    xlsx = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet = "GPS" if "GPS" in xlsx.sheet_names else xlsx.sheet_names[0]
    df = pd.read_excel(xlsx, sheet_name=sheet)

    need = ["Speler", "Datum", "Type", "Event"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Flat GPS Excel mist kolommen: {missing}")

    dt = pd.to_datetime(df["Datum"], dayfirst=True, errors="coerce")
    if dt.isna().any():
        bad = df.loc[dt.isna(), "Datum"].head(5).tolist()
        raise ValueError(f"Kon sommige Datum waarden niet parsen: {bad}")

    df["Datum"] = dt.dt.strftime("%d-%m-%Y")
    df["Week"] = dt.dt.isocalendar().week.astype(int)
    df["Year"] = dt.dt.year.astype(int)

    df = drop_min_columns(df)

    fixed = ["Speler", "Datum", "Week", "Year", "Type", "Event"]
    rest = [c for c in df.columns if c not in fixed]
    df = df[fixed + rest]

    df = ensure_unique_events(df)
    return df


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


# -------------------------
# Export helpers
# -------------------------
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
