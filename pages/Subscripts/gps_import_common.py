# gps_import_common.py
# ============================================================
# Shared helpers for GPS Import suite
# NOTE: profiles.team column removed -> do NOT select it anywhere.
# Auth fix:
# - Restores session from cookies via auth_session.py when session_state resets
#   (mobile/tab switch/reconnect)
# ============================================================

from __future__ import annotations

import csv
import io
import math
import numbers
import re
import unicodedata
from datetime import date

import pandas as pd
import requests
import streamlit as st
import roles as roles_mod

# ✅ auth restore helpers (jij hebt auth_session.py al toegevoegd)
from auth_session import ensure_auth_restored, get_sb_client


def _fallback_redirect_to_login(message: str = "Sessie verlopen. Log opnieuw in.", clear_cookies: bool = False) -> None:
    if message:
        st.error(message)
    try:
        st.switch_page("app.py")
    except Exception:
        pass
    st.stop()


redirect_to_login = getattr(roles_mod, "redirect_to_login", _fallback_redirect_to_login)

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
    if isinstance(v, dict):
        return {str(k): json_safe(value) for k, value in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(value) for value in v]
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, (date,)):
        return v.isoformat()
    if isinstance(v, numbers.Integral) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, numbers.Real) and not isinstance(v, bool):
        value = float(v)
        return None if not math.isfinite(value) else value
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
    if isinstance(v, str):
        value = v.strip()
        if not value:
            return None
        # CSV readers keep broad vendor exports as strings. Convert plain
        # numeric cells so extra_metrics remains useful for later analysis.
        if re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?", value):
            numeric = float(value.replace(",", "."))
            return int(numeric) if numeric.is_integer() else numeric
        return value
    return v


# -------------------------
# Auth / REST helpers
# -------------------------
def get_access_token() -> str | None:
    """
    Haalt access token op uit session_state.
    Als session_state weg is (mobiel/tab-switch), probeert cookie-restore via auth_session.py.
    """
    tok = st.session_state.get("access_token")
    if tok:
        return str(tok)

    sess = st.session_state.get("sb_session")
    if sess is not None:
        token = getattr(sess, "access_token", None)
        if token:
            return str(token)

    # ✅ fallback: restore from cookie
    try:
        sb = get_sb_client()
        ok, tok2 = ensure_auth_restored(sb)
        if ok and tok2:
            return str(tok2)
    except Exception:
        pass

    return None


def require_access_token() -> str:
    tok = get_access_token()
    if not tok:
        st.error("Sessie verlopen. Log opnieuw in.")
        try:
            st.switch_page("app.py")
        except Exception:
            pass
        st.stop()
    return tok


def get_access_token() -> str | None:
    """
    Gevalideerde token-opvraag voor import/management flows.
    Controleert eerst of de sessie nog geldig is en herstelt anders via cookies.
    """
    try:
        sb = get_sb_client()
        ok, tok = ensure_auth_restored(sb)
        if ok and tok:
            return str(tok)
    except Exception:
        pass
    return None


def require_access_token() -> str:
    tok = get_access_token()
    if not tok:
        redirect_to_login("Sessie verlopen. Log opnieuw in.", clear_cookies=True)
    return tok


def rest_headers(access_token: str) -> dict:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def _retry_with_refreshed_token(func, *args, **kwargs):
    """
    Voert een REST/auth call uit; bij auth-fout 401/403 probeert 1x token refresh via auth_session.
    Verwacht dat func RuntimeError kan gooien met statuscode in tekst.
    """
    try:
        return func(*args, **kwargs)
    except RuntimeError as e:
        msg = str(e)
        if "(401)" in msg or "(403)" in msg:
            sb = get_sb_client()
            ok, tok = ensure_auth_restored(sb)
            if ok and tok:
                # vervang eerste arg als dat access_token is
                new_args = list(args)
                if new_args:
                    new_args[0] = tok
                return func(*new_args, **kwargs)
        raise


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
            # 1x retry bij auth expiry
            if r.status_code in (401, 403):
                sb = get_sb_client()
                ok, tok = ensure_auth_restored(sb)
                if ok and tok:
                    headers = rest_headers(tok)
                    headers["Prefer"] = "resolution=merge-duplicates"
                    r = requests.post(url, headers=headers, json=safe_chunk, timeout=120)

            if not r.ok:
                raise RuntimeError(f"UPSERT {table} failed ({r.status_code}): {r.text}")


def rest_patch(access_token: str, table: str, where_query: str, payload: dict) -> None:
    url = f"{SUPABASE_URL}/rest/v1/{table}?{where_query}"
    headers = rest_headers(access_token)
    headers["Prefer"] = "return=representation"
    safe_payload = {k: json_safe(v) for k, v in payload.items()}
    r = requests.patch(url, headers=headers, json=safe_payload, timeout=60)

    if not r.ok and r.status_code in (401, 403):
        sb = get_sb_client()
        ok, tok = ensure_auth_restored(sb)
        if ok and tok:
            headers = rest_headers(tok)
            headers["Prefer"] = "return=representation"
            r = requests.patch(url, headers=headers, json=safe_payload, timeout=60)

    if not r.ok:
        raise RuntimeError(f"PATCH {table} failed ({r.status_code}): {r.text}")


def rest_delete(access_token: str, table: str, where_query: str) -> None:
    url = f"{SUPABASE_URL}/rest/v1/{table}?{where_query}"
    headers = rest_headers(access_token)
    headers["Prefer"] = "return=representation"
    r = requests.delete(url, headers=headers, timeout=60)

    if not r.ok and r.status_code in (401, 403):
        sb = get_sb_client()
        ok, tok = ensure_auth_restored(sb)
        if ok and tok:
            headers = rest_headers(tok)
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
    try:
        u = auth_get_user(access_token)
    except RuntimeError as e:
        # 1x retry via cookie restore
        if "(401)" in str(e) or "(403)" in str(e):
            sb = get_sb_client()
            ok, tok = ensure_auth_restored(sb)
            if ok and tok:
                access_token = tok
                u = auth_get_user(access_token)
            else:
                raise
        else:
            raise

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
    # retry wrapper voor auth-expiry
    df = _retry_with_refreshed_token(
        rest_get,
        access_token,
        "players",
        "select=player_id,full_name,is_active&is_active=eq.true&limit=5000",
    )
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
MATCH_IMPORT_COLUMNS = [
    "match_date",
    "fixture",
    "home_away",
    "opponent",
    "match_type",
    "season",
    "result",
    "goals_for",
    "goals_against",
]

MATCH_IMPORT_COLUMN_ALIASES = {
    "datum": "match_date",
    "date": "match_date",
    "matchdate": "match_date",
    "matchdatum": "match_date",
    "wedstrijd": "fixture",
    "fixture": "fixture",
    "match": "fixture",
    "wedstrijdnaam": "fixture",
    "homeaway": "home_away",
    "thuisuit": "home_away",
    "locatie": "home_away",
    "tegenstander": "opponent",
    "opponent": "opponent",
    "opponentteam": "opponent",
    "type": "match_type",
    "matchtype": "match_type",
    "wedstrijdtype": "match_type",
    "seizoen": "season",
    "season": "season",
    "seasonname": "season",
    "result": "result",
    "uitslag": "result",
    "score": "result",
    "goalsfor": "goals_for",
    "goalsvoor": "goals_for",
    "doelpuntenvoor": "goals_for",
    "goalsagainst": "goals_against",
    "goalstegen": "goals_against",
    "doelpuntentegen": "goals_against",
    "doelpuntenegen": "goals_against",
    "doelpuntenant": "goals_against",
    "matchid": "match_id",
    "wedstrijdid": "match_id",
}


def _match_header_key(value: object) -> str:
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = value.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _clean_match_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _read_matches_csv(file_bytes: bytes) -> pd.DataFrame:
    if not file_bytes:
        raise ValueError("Het CSV-bestand is leeg.")

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = file_bytes.decode(encoding)
            sample = text[:8192]
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
            except csv.Error:
                first_line = sample.splitlines()[0] if sample.splitlines() else ""
                delimiter = ";" if first_line.count(";") >= first_line.count(",") else ","

            df = pd.read_csv(
                io.StringIO(text),
                sep=delimiter,
                dtype=object,
                keep_default_na=False,
            )
            if len(df.columns) > 1 or delimiter == ",":
                return df
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Kon het Matches.csv-bestand niet lezen: {last_error}")


def _coalesce_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mapt Nederlandse/Engelse headers zonder dubbele doelkolommen te maken."""
    output = pd.DataFrame(index=df.index)
    sources: dict[str, list[object]] = {}
    for column in df.columns:
        target = MATCH_IMPORT_COLUMN_ALIASES.get(_match_header_key(column))
        if target:
            sources.setdefault(target, []).append(column)

    for target, columns in sources.items():
        values = pd.Series([""] * len(df), index=df.index, dtype=object)
        for column in columns:
            candidate = df[column].map(_clean_match_text)
            values = values.mask(values.eq(""), candidate)
        output[target] = values
    return output


def _parse_match_date(value: object) -> date | None:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    text = _clean_match_text(value)
    if not text:
        return None

    # ISO dates are unambiguous; the remaining common exports are day-first.
    if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text):
        parsed = pd.to_datetime(text, errors="coerce")
    else:
        parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    return parsed.date() if pd.notna(parsed) else None


def _normalise_home_away(value: object) -> str | None:
    key = _match_header_key(value)
    if key in {"home", "thuis", "h", "1"}:
        return "Home"
    if key in {"away", "uit", "a", "2"}:
        return "Away"
    return None


def _normalise_match_type(value: object) -> tuple[str, bool]:
    raw = _clean_match_text(value)
    key = _match_header_key(raw)
    if not raw:
        return "Competitie", True
    if key in {"oefen", "oefenwedstrijd", "friendly", "vriendelijk", "vriendschappelijk", "test", "practicematch"}:
        return "Oefenwedstrijd", False
    if key in {"competitie", "competition", "league", "official", "match", "normaal"}:
        return "Competitie", False
    if key in {"beker", "cup"}:
        return "Beker", False
    return raw, True


def _season_for_match_date(value: date | None) -> str:
    if value is None:
        return ""
    start_year = value.year if value.month >= 7 else value.year - 1
    return f"{start_year}/{start_year + 1}"


def _parse_result(value: object) -> str:
    raw = _clean_match_text(value)
    if not raw:
        return ""
    match = re.fullmatch(r"(\d+)\s*[-:\u2013]\s*(\d+)", raw)
    return f"{int(match.group(1))}-{int(match.group(2))}" if match else ""


def _parse_goal(value: object) -> tuple[int | None, bool]:
    raw = _clean_match_text(value)
    if not raw:
        return None, False
    number = pd.to_numeric(raw.replace(",", "."), errors="coerce")
    if pd.isna(number) or float(number) < 0 or float(number) != int(float(number)):
        return None, True
    return int(number), False


def _infer_match_context(fixture: str, home_away: str, opponent: str) -> tuple[str, str, str]:
    """Vult opponent/home-away aan als alleen een volledige fixture is aangeleverd."""
    if opponent or not fixture:
        return fixture, home_away, opponent
    parts = re.split(r"\s+-\s+", fixture, maxsplit=1)
    if len(parts) != 2:
        return fixture, home_away, opponent
    left, right = parts[0].strip(), parts[1].strip()
    team_key = _match_header_key(TEAM_NAME_MATCHES)
    if _match_header_key(left) == team_key:
        return fixture, "Home", right
    if _match_header_key(right) == team_key:
        return fixture, "Away", left
    return fixture, home_away, opponent


def match_identity_key(row: dict | pd.Series) -> tuple[str, str, str]:
    """Dezelfde sleutel als de matches_unique_key constraint in Supabase."""
    match_date = _clean_match_text(row.get("match_date"))
    fixture = re.sub(r"\s+", " ", _clean_match_text(row.get("fixture"))).casefold()
    season = re.sub(r"\s+", " ", _clean_match_text(row.get("season"))).casefold()
    return match_date, fixture, season


def normalize_matches_dataframe(raw_df: pd.DataFrame, source_file: str = "") -> pd.DataFrame:
    """Normaliseert een matchbestand en voegt per rij import_status/import_message toe."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=MATCH_IMPORT_COLUMNS + ["_import_status", "_import_message"])

    df = _coalesce_match_columns(raw_df)
    if "match_date" not in df.columns:
        raise ValueError("Verplichte kolom ontbreekt: Datum/Date.")
    if "opponent" not in df.columns and "fixture" not in df.columns:
        raise ValueError("Verplichte kolom ontbreekt: Tegenstander/Opponent of Wedstrijd/Fixture.")

    rows: list[dict] = []
    for index, source in df.iterrows():
        errors: list[str] = []
        warnings: list[str] = []
        match_date = _parse_match_date(source.get("match_date"))
        if match_date is None:
            errors.append("Datum ontbreekt of is ongeldig")

        fixture = _clean_match_text(source.get("fixture"))
        home_away_raw = _clean_match_text(source.get("home_away"))
        home_away = _normalise_home_away(home_away_raw) if home_away_raw else "Home"
        opponent = _clean_match_text(source.get("opponent"))
        fixture, home_away, opponent = _infer_match_context(fixture, home_away, opponent)

        if home_away_raw and home_away not in HOME_AWAY_OPTIONS:
            errors.append(f"Home/Away onbekend: {home_away_raw}")
        elif not home_away_raw:
            warnings.append("Home/Away ontbrak; Home gebruikt")
            home_away = "Home"
        if not opponent:
            errors.append("Tegenstander ontbreekt")

        match_type, type_warning = _normalise_match_type(source.get("match_type"))
        if type_warning:
            warnings.append("Match type gecontroleerd of standaard Competitie gebruikt")

        season = _clean_match_text(source.get("season"))
        if not season:
            season = _season_for_match_date(match_date)
            if season:
                warnings.append(f"Seizoen automatisch bepaald: {season}")

        goals_for, gf_invalid = _parse_goal(source.get("goals_for"))
        goals_against, ga_invalid = _parse_goal(source.get("goals_against"))
        if gf_invalid:
            errors.append("Goals for moet een positief geheel getal zijn")
        if ga_invalid:
            errors.append("Goals against moet een positief geheel getal zijn")

        calculated_result = build_result(goals_for, goals_against)
        supplied_result = _parse_result(source.get("result"))
        raw_result = _clean_match_text(source.get("result"))
        if raw_result and not supplied_result:
            warnings.append("Resultaat kon niet worden gelezen; score gebruikt")
        result = calculated_result or supplied_result
        if calculated_result and supplied_result and calculated_result != supplied_result:
            warnings.append("Resultaat overschreven door Goals for/against")
        if not result:
            result = None

        canonical_fixture = build_fixture(TEAM_NAME_MATCHES, home_away, opponent) or fixture
        if not canonical_fixture:
            errors.append("Wedstrijd/Fixture ontbreekt")

        row = {
            "match_date": match_date,
            "fixture": canonical_fixture or None,
            "home_away": home_away or None,
            "opponent": opponent or None,
            "match_type": match_type or None,
            "season": season or None,
            "result": result,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "_source_row": int(index) + 2,
            "_import_status": "FOUT" if errors else ("WAARSCHUWING" if warnings else "OK"),
            "_import_message": "; ".join(errors + warnings),
        }
        source_match_id = _clean_match_text(source.get("match_id"))
        if source_match_id:
            parsed_match_id = pd.to_numeric(source_match_id, errors="coerce")
            if pd.notna(parsed_match_id) and float(parsed_match_id).is_integer():
                row["match_id"] = int(parsed_match_id)
            else:
                errors.append("Match ID moet een geheel getal zijn")
                row["_import_status"] = "FOUT"
                row["_import_message"] = "; ".join(errors + warnings)
        rows.append(row)

    result_df = pd.DataFrame(rows)
    seen: set[tuple[str, str, str]] = set()
    for idx, row in result_df.iterrows():
        if row["_import_status"] == "FOUT":
            continue
        key = match_identity_key(row)
        if key in seen:
            result_df.at[idx, "_import_status"] = "DUPLICAAT"
            previous = _clean_match_text(result_df.at[idx, "_import_message"])
            result_df.at[idx, "_import_message"] = "; ".join(filter(None, [previous, "Dubbele wedstrijd in dit bestand"]))
        else:
            seen.add(key)
    return result_df


def parse_matches_csv(file_bytes: bytes) -> pd.DataFrame:
    return normalize_matches_dataframe(_read_matches_csv(file_bytes))


def matches_df_to_rows(df: pd.DataFrame, source_file: str) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        if str(r.get("_import_status", "OK")) in {"FOUT", "DUPLICAAT"}:
            continue
        match_id = r.get("match_id")
        payload = {
            "match_date": str(r["match_date"]) if pd.notna(r.get("match_date")) else None,
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
        if pd.notna(match_id) and _clean_match_text(match_id):
            payload["match_id"] = int(match_id)
        rows.append(
            payload
        )
    return rows


def sync_matches(access_token: str, df: pd.DataFrame, source_file: str) -> dict[str, int]:
    """Synchroniseert geldige wedstrijden zonder bestaande records te dupliceren."""
    rows = matches_df_to_rows(df, source_file=source_file)
    if not rows:
        return {"total": 0, "inserted": 0, "updated": 0}

    dates = [pd.to_datetime(row["match_date"]).date() for row in rows if row.get("match_date")]
    existing = fetch_matches_range(access_token, min(dates), max(dates)) if dates else pd.DataFrame()
    existing_keys = {match_identity_key(row) for _, row in existing.iterrows()} if not existing.empty else set()
    input_ids = sorted(
        {
            int(row["match_id"])
            for row in rows
            if row.get("match_id") is not None and _clean_match_text(row.get("match_id"))
        }
    )
    existing_ids: set[int] = set()
    if input_ids:
        id_query = ",".join(str(match_id) for match_id in input_ids)
        existing_by_id = _retry_with_refreshed_token(
            rest_get,
            access_token,
            "matches",
            f"select=match_id&match_id=in.({id_query})&limit=2000",
        )
        if not existing_by_id.empty and "match_id" in existing_by_id.columns:
            existing_ids = set(
                pd.to_numeric(existing_by_id["match_id"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )

    unique_upsert_rows: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()
    inserted = 0
    updated = 0
    total = 0
    for row in rows:
        key = match_identity_key(row)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        row_match_id = int(row["match_id"]) if row.get("match_id") is not None else None
        if row_match_id is not None and row_match_id in existing_ids:
            patch_payload = {key: value for key, value in row.items() if key != "match_id"}
            rest_patch(access_token, "matches", f"match_id=eq.{row_match_id}", patch_payload)
            updated += 1
            total += 1
            continue

        # Never trust an unknown imported match_id: let the unique key decide.
        safe_row = dict(row)
        safe_row.pop("match_id", None)
        if key in existing_keys:
            updated += 1
        else:
            inserted += 1
        unique_upsert_rows.append(safe_row)
        total += 1

    rest_upsert(access_token, "matches", unique_upsert_rows, on_conflict="match_date,fixture,season")
    return {"total": total, "inserted": inserted, "updated": updated}


def fetch_matches_on_date(access_token: str, d: date) -> pd.DataFrame:
    q = (
        "select=match_id,match_date,fixture,opponent,home_away,match_type,season,result,goals_for,goals_against"
        f"&match_date=eq.{d.isoformat()}"
        "&order=match_id.desc&limit=200"
    )
    return _retry_with_refreshed_token(rest_get, access_token, "matches", q)


def fetch_matches_range(access_token: str, d_from: date, d_to: date, season_filter: str = "") -> pd.DataFrame:
    q = (
        "select=match_id,match_date,fixture,opponent,home_away,match_type,season,result,goals_for,goals_against"
        f"&match_date=gte.{d_from.isoformat()}"
        f"&match_date=lte.{d_to.isoformat()}"
        "&order=match_date.desc&limit=2000"
    )
    if season_filter.strip():
        q += f"&season=eq.{requests.utils.quote(season_filter.strip(), safe='')}"
    return _retry_with_refreshed_token(rest_get, access_token, "matches", q)


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
    df = _retry_with_refreshed_token(rest_get, access_token, "gps_records", q)
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
    "total_distance_td",
    "td_zone_1",
    "td_zone_2",
    "td_zone_1_2",
    "td_zone_3",
    "td_zone_4",
    "td_zone_5",
    "td_zone_6",
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
    # Frequently used fields from broad player-metrics exports. Less stable
    # vendor-specific fields continue to be preserved in extra_metrics.
    "distance_per_min",
    "high_sprint_relative",
    "sprint_relative",
    "accelerations_relative",
    "decelerations_relative",
    "acceleration_impulse",
    "total_acceleration_loading",
    "total_deceleration_loading",
    "max_acceleration",
    "max_deceleration",
    "explosive_distance",
    "metabolic_distance_relative",
    "hml_distance",
    "hml_efforts_maximum_speed",
    "lower_speed_loading",
    "total_loading",
    "player_max_speed",
    "heart_rate_load",
    "heart_rate_exertion",
    "heart_rate_recovery_pct",
    "heart_rate_recovery_beats",
    "heart_rate_variability",
    "min_hr",
    "acute_load",
    "chronic_load",
    "acwr",
    "steps",
    "source_file",
    "extra_metrics",
]

METRIC_MAP = {
    "duration": "duration",
    "totaldistance": "total_distance_td",
    "walkdistance": "td_zone_1_2",
    "jogdistance": "td_zone_3",
    "rundistance": "td_zone_4",
    "sprintdistance": "td_zone_5",
    "hisprintdistance": "td_zone_6",
    "highsprintdistance": "td_zone_6",
    "highspeedrunningabsolute": "td_zone_6",
    "numberofsprints": "number_of_sprints",
    "sprints": "number_of_sprints",
    "numberofhisprints": "number_of_high_sprints",
    "numberofhighsprints": "number_of_high_sprints",
    "numberofrepeatedsprints": "number_of_repeated_sprints",
    "maxspeed": "max_speed",
    "avgspeed": "avg_speed",
    "averagespeed": "avg_speed",
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
    "averageheartrate": "avg_hr",
    "maxhr": "max_hr",
    "maximumheartrate": "max_hr",
    "distancepermin": "distance_per_min",
    "highspeedrunningrelative": "high_sprint_relative",
    "sprintdistancerelative": "sprint_relative",
    "accelerationsrelative": "accelerations_relative",
    "decelerationsrelative": "decelerations_relative",
    "accelerationimpulse": "acceleration_impulse",
    "totalaccelerationloading": "total_acceleration_loading",
    "totaldecelerationloading": "total_deceleration_loading",
    "maxacceleration": "max_acceleration",
    "maxdeceleration": "max_deceleration",
    "explosivedistanceabsolute": "explosive_distance",
    "metabolicdistancerelative": "metabolic_distance_relative",
    "hmldistance": "hml_distance",
    "hmleffortsmaximumspeed": "hml_efforts_maximum_speed",
    "lowerspeedloading": "lower_speed_loading",
    "totalloading": "total_loading",
    "playermaxspeed": "player_max_speed",
    "heartrateload": "heart_rate_load",
    "heartrateexertion": "heart_rate_exertion",
    "heartraterecovery": "heart_rate_recovery_pct",
    "heartraterecoverynumberofbeats": "heart_rate_recovery_beats",
    "heartratevariability": "heart_rate_variability",
    "minimumheartrate": "min_hr",
    "acute": "acute_load",
    "chronic": "chronic_load",
    "acutechronicratio": "acwr",
    "steps": "steps",
    # extra aliases (zonder "distance")
    "td_zone_1_2": "td_zone_1_2",
    "td_zone_3": "td_zone_3",
    "td_zone_4": "td_zone_4",
    "td_zone_5": "td_zone_5",
    "highsprint": "td_zone_6",
    "hisprint": "td_zone_6",
    "walk": "td_zone_1_2",
    "jog": "td_zone_3",
    "run": "td_zone_4",
}

# Source-complete fields from the STATSports/Johan Sports player-metrics export.
# Exact headers are kept so similarly normalized fields (for example time and time %) do not collide.
CSV_SOURCE_COLUMN_MAP = {'timeinredzonerelative': 'csv_time_in_red_zone_relative', 'accelerationstotaldistancezone1relative': 'csv_accelerations_total_distance_zone_1_relative', 'accelerationstotaldistancezone2relative': 'csv_accelerations_total_distance_zone_2_relative', 'accelerationstotaldistancezone3relative': 'csv_accelerations_total_distance_zone_3_relative', 'accelerationstotaldistancezone4absolute': 'csv_accelerations_total_distance_zone_4_absolute', 'accelerationstotaldistancezone4relative': 'csv_accelerations_total_distance_zone_4_relative', 'accelerationstotaldistancezone5relative': 'csv_accelerations_total_distance_zone_5_relative', 'accelerationstotaldistancezone6relative': 'csv_accelerations_total_distance_zone_6_relative', 'accelerationstotaltimezone1relative': 'csv_accelerations_total_time_zone_1_relative', 'accelerationstotaltimezone2relative': 'csv_accelerations_total_time_zone_2_relative', 'accelerationstotaltimezone3relative': 'csv_accelerations_total_time_zone_3_relative', 'accelerationstotaltimezone4relative': 'csv_accelerations_total_time_zone_4_relative', 'accelerationstotaltimezone5relative': 'csv_accelerations_total_time_zone_5_relative', 'accelerationstotaltimezone6relative': 'csv_accelerations_total_time_zone_6_relative', 'accelerationszone1relative': 'csv_accelerations_zone_1_relative', 'accelerationszone2relative': 'csv_accelerations_zone_2_relative', 'accelerationszone3relative': 'csv_accelerations_zone_3_relative', 'accelerationszone3zone6relative': 'csv_accelerations_zone_3_zone_6_relative', 'accelerationszone4relative': 'csv_accelerations_zone_4_relative', 'accelerationszone4zone6relative': 'csv_accelerations_zone_4_zone_6_relative', 'accelerationszone5relative': 'csv_accelerations_zone_5_relative', 'accelerationszone5zone6relative': 'csv_accelerations_zone_5_zone_6_relative', 'accelerationszone6relative': 'csv_accelerations_zone_6_relative', 'accelerationsperminrelative': 'csv_accelerations_per_min_relative', 'averagediveimpact': 'csv_average_dive_impact', 'averagegkpower': 'csv_average_gk_power', 'averagemetabolicpower': 'csv_average_metabolic_power', 'averagetimesincelastaccel': 'csv_average_time_since_last_accel', 'averagetimesincelastdecel': 'csv_average_time_since_last_decel', 'averagetimesincelastdive': 'csv_average_time_since_last_dive', 'averagetimesincelasthib': 'csv_average_time_since_last_hib', 'averagetimesincelasthmleffort': 'csv_average_time_since_last_hml_effort', 'averagetimesincelastsprint': 'csv_average_time_since_last_sprint', 'ballinplaytime': 'csv_ball_in_play_time_pct', 'playercustomid': 'csv_player_custom_id', 'decelerationstotaldistancezone1relative': 'csv_decelerations_total_distance_zone_1_relative', 'decelerationstotaldistancezone2relative': 'csv_decelerations_total_distance_zone_2_relative', 'decelerationstotaldistancezone3relative': 'csv_decelerations_total_distance_zone_3_relative', 'decelerationstotaldistancezone4relative': 'csv_decelerations_total_distance_zone_4_relative', 'decelerationstotaldistancezone5relative': 'csv_decelerations_total_distance_zone_5_relative', 'decelerationstotaldistancezone6relative': 'csv_decelerations_total_distance_zone_6_relative', 'decelerationstotaltimezone1relative': 'csv_decelerations_total_time_zone_1_relative', 'decelerationstotaltimezone2relative': 'csv_decelerations_total_time_zone_2_relative', 'decelerationstotaltimezone3relative': 'csv_decelerations_total_time_zone_3_relative', 'decelerationstotaltimezone4relative': 'csv_decelerations_total_time_zone_4_relative', 'decelerationstotaltimezone5relative': 'csv_decelerations_total_time_zone_5_relative', 'decelerationstotaltimezone6relative': 'csv_decelerations_total_time_zone_6_relative', 'decelerationszone1relative': 'csv_decelerations_zone_1_relative', 'decelerationszone2relative': 'csv_decelerations_zone_2_relative', 'decelerationszone3relative': 'csv_decelerations_zone_3_relative', 'decelerationszone3zone6relative': 'csv_decelerations_zone_3_zone_6_relative', 'decelerationszone4relative': 'csv_decelerations_zone_4_relative', 'decelerationszone4zone6relative': 'csv_decelerations_zone_4_zone_6_relative', 'decelerationszone5relative': 'csv_decelerations_zone_5_relative', 'decelerationszone5zone6relative': 'csv_decelerations_zone_5_zone_6_relative', 'decelerationszone6relative': 'csv_decelerations_zone_6_relative', 'decelerationsperminrelative': 'csv_decelerations_per_min_relative', 'distancezone1absolute': 'csv_distance_zone_1_absolute', 'distancezone1relative': 'csv_distance_zone_1_relative', 'distancezone2absolute': 'csv_distance_zone_2_absolute', 'distancezone2relative': 'csv_distance_zone_2_relative', 'distancezone2zone6absolute': 'csv_distance_zone_2_zone_6_absolute', 'distancezone2zone6relative': 'csv_distance_zone_2_zone_6_relative', 'distancezone3absolute': 'csv_distance_zone_3_absolute', 'distancezone3relative': 'csv_distance_zone_3_relative', 'distancezone3zone6absolute': 'csv_distance_zone_3_zone_6_absolute', 'distancezone3zone6relative': 'csv_distance_zone_3_zone_6_relative', 'distancezone4absolute': 'csv_distance_zone_4_absolute', 'distancezone4relative': 'csv_distance_zone_4_relative', 'distancezone4zone6absolute': 'csv_distance_zone_4_zone_6_absolute', 'distancezone4zone6relative': 'csv_distance_zone_4_zone_6_relative', 'distancezone5absolute': 'csv_distance_zone_5_absolute', 'distancezone5relative': 'csv_distance_zone_5_relative', 'distancezone6absolute': 'csv_distance_zone_6_absolute', 'distancezone6relative': 'csv_distance_zone_6_relative', 'dives': 'csv_dives', 'divesleft': 'csv_dives_left', 'divesright': 'csv_dives_right', 'drilldate': 'csv_drill_date', 'drillendtime': 'csv_drill_end_time', 'drillstarttime': 'csv_drill_start_time', 'drilltitle': 'csv_drill_title', 'durationofhighintensitybursts': 'csv_duration_of_high_intensity_bursts', 'dynamicloadanterior': 'csv_dynamic_load_anterior', 'dynamicloadlateral': 'csv_dynamic_load_lateral', 'dynamicloadvertical': 'csv_dynamic_load_vertical', 'dynamicstressload': 'csv_dynamic_stress_load', 'dynamicstressloadtimezone1': 'csv_dynamic_stress_load_time_zone_1', 'dynamicstressloadtimezone2': 'csv_dynamic_stress_load_time_zone_2', 'dynamicstressloadtimezone3': 'csv_dynamic_stress_load_time_zone_3', 'dynamicstressloadtimezone4': 'csv_dynamic_stress_load_time_zone_4', 'dynamicstressloadtimezone5': 'csv_dynamic_stress_load_time_zone_5', 'dynamicstressloadtimezone6': 'csv_dynamic_stress_load_time_zone_6', 'dynamicstressloadzone1': 'csv_dynamic_stress_load_zone_1', 'dynamicstressloadzone2': 'csv_dynamic_stress_load_zone_2', 'dynamicstressloadzone3': 'csv_dynamic_stress_load_zone_3', 'dynamicstressloadzone3zone6': 'csv_dynamic_stress_load_zone_3_zone_6', 'dynamicstressloadzone4': 'csv_dynamic_stress_load_zone_4', 'dynamicstressloadzone4zone6': 'csv_dynamic_stress_load_zone_4_zone_6', 'dynamicstressloadzone5': 'csv_dynamic_stress_load_zone_5', 'dynamicstressloadzone5zone6': 'csv_dynamic_stress_load_zone_5_zone_6', 'dynamicstressloadzone6': 'csv_dynamic_stress_load_zone_6', 'edi': 'csv_edi_pct', 'energyexpenditurekcal': 'csv_energy_expenditure_kcal', 'entrieszone3absolute': 'csv_entries_zone_3_absolute', 'entrieszone3relative': 'csv_entries_zone_3_relative', 'entrieszone4absolute': 'csv_entries_zone_4_absolute', 'entrieszone4relative': 'csv_entries_zone_4_relative', 'entrieszone5absolute': 'csv_entries_zone_5_absolute', 'entrieszone5relative': 'csv_entries_zone_5_relative', 'entrieszone6absolute': 'csv_entries_zone_6_absolute', 'entrieszone6relative': 'csv_entries_zone_6_relative', 'equivalentmetabolicdistance': 'csv_equivalent_metabolic_distance', 'explosivedistancerelative': 'csv_explosive_distance_relative', 'externalwork': 'csv_external_work', 'fatigueindex': 'csv_fatigue_index', 'gkload': 'csv_gk_load', 'highintensityburstsmaximumspeed': 'csv_high_intensity_bursts_maximum_speed', 'highintensityburststotaldistance': 'csv_high_intensity_bursts_total_distance', 'hmlefforts': 'csv_hml_efforts', 'hmleffortstotaldistance': 'csv_hml_efforts_total_distance', 'hmltime': 'csv_hml_time', 'hmldperminute': 'csv_hmld_per_minute', 'hsrperminuteabsolute': 'csv_hsr_per_minute_absolute', 'hsrperminuterelative': 'csv_hsr_per_minute_relative', 'impactsrelative': 'csv_impacts_relative', 'impactszone1relative': 'csv_impacts_zone_1_relative', 'impactszone2relative': 'csv_impacts_zone_2_relative', 'impactszone3relative': 'csv_impacts_zone_3_relative', 'impactszone3zone6relative': 'csv_impacts_zone_3_zone_6_relative', 'impactszone4relative': 'csv_impacts_zone_4_relative', 'impactszone4zone6relative': 'csv_impacts_zone_4_zone_6_relative', 'impactszone5relative': 'csv_impacts_zone_5_relative', 'impactszone5zone6relative': 'csv_impacts_zone_5_zone_6_relative', 'impactszone6relative': 'csv_impacts_zone_6_relative', 'leftanteriorpostimpact': 'csv_left_anterior_post_impact', 'leftaverageverticalimpact': 'csv_left_average_vertical_impact', 'leftlateralimpact': 'csv_left_lateral_impact', 'leftmagimpact': 'csv_left_mag_impact', 'leftverticalimpact': 'csv_left_vertical_impact', 'maxheartrate': 'csv_max_heart_rate', 'metabolicdistancezone1relative': 'csv_metabolic_distance_zone_1_relative', 'metabolicdistancezone2relative': 'csv_metabolic_distance_zone_2_relative', 'metabolicdistancezone3relative': 'csv_metabolic_distance_zone_3_relative', 'metabolicdistancezone4relative': 'csv_metabolic_distance_zone_4_relative', 'metabolicdistancezone5relative': 'csv_metabolic_distance_zone_5_relative', 'metabolicdistancezone6relative': 'csv_metabolic_distance_zone_6_relative', 'metabolictimerelative': 'csv_metabolic_time_relative', 'metabolictimezone1absolute': 'csv_metabolic_time_zone_1_absolute', 'metabolictimezone1relative': 'csv_metabolic_time_zone_1_relative', 'metabolictimezone2absolute': 'csv_metabolic_time_zone_2_absolute', 'metabolictimezone2relative': 'csv_metabolic_time_zone_2_relative', 'metabolictimezone3relative': 'csv_metabolic_time_zone_3_relative', 'metabolictimezone4relative': 'csv_metabolic_time_zone_4_relative', 'metabolictimezone5relative': 'csv_metabolic_time_zone_5_relative', 'metabolictimezone6relative': 'csv_metabolic_time_zone_6_relative', 'noofsatellites': 'csv_no_of_satellites', 'numberofhighintensitybursts': 'csv_number_of_high_intensity_bursts', 'playerdateofbirth': 'csv_player_date_of_birth', 'playerdisplayname': 'csv_player_display_name', 'playerfirstname': 'csv_player_first_name', 'playerheight': 'csv_player_height', 'playerlastname': 'csv_player_last_name', 'playermaxaccel': 'csv_player_max_accel', 'playermaxdecel': 'csv_player_max_decel', 'playermaxheartrate': 'csv_player_max_heart_rate', 'playername': 'csv_player_name', 'playerprimaryposition': 'csv_player_primary_position', 'playerrestingheartrate': 'csv_player_resting_heart_rate', 'playersecondaryposition': 'csv_player_secondary_position', 'playersprintthreshold': 'csv_player_sprint_threshold', 'playerweight': 'csv_player_weight', 'qualityofsignal': 'csv_quality_of_signal', 'rightanteriorpostimpact': 'csv_right_anterior_post_impact', 'rightaverageverticalimpact': 'csv_right_average_vertical_impact', 'rightlateralimpact': 'csv_right_lateral_impact', 'rightmagimpact': 'csv_right_mag_impact', 'rightverticalimpact': 'csv_right_vertical_impact', 'sessiondate': 'csv_session_date', 'sessiondayofweek': 'csv_session_day_of_week', 'sessionendtime': 'csv_session_end_time', 'sessionstarttime': 'csv_session_start_time', 'sessiontitle': 'csv_session_title', 'sessiontype': 'csv_session_type', 'sessionweeknumber': 'csv_session_week_number', 'speedintensity': 'csv_speed_intensity', 'speedintensityzone1absolute': 'csv_speed_intensity_zone_1_absolute', 'speedintensityzone1relative': 'csv_speed_intensity_zone_1_relative', 'speedintensityzone2absolute': 'csv_speed_intensity_zone_2_absolute', 'speedintensityzone2relative': 'csv_speed_intensity_zone_2_relative', 'speedintensityzone3absolute': 'csv_speed_intensity_zone_3_absolute', 'speedintensityzone3relative': 'csv_speed_intensity_zone_3_relative', 'speedintensityzone3zone6absolute': 'csv_speed_intensity_zone_3_zone_6_absolute', 'speedintensityzone3zone6relative': 'csv_speed_intensity_zone_3_zone_6_relative', 'speedintensityzone4absolute': 'csv_speed_intensity_zone_4_absolute', 'speedintensityzone4relative': 'csv_speed_intensity_zone_4_relative', 'speedintensityzone4zone6absolute': 'csv_speed_intensity_zone_4_zone_6_absolute', 'speedintensityzone4zone6relative': 'csv_speed_intensity_zone_4_zone_6_relative', 'speedintensityzone5absolute': 'csv_speed_intensity_zone_5_absolute', 'speedintensityzone5relative': 'csv_speed_intensity_zone_5_relative', 'speedintensityzone5zone6absolute': 'csv_speed_intensity_zone_5_zone_6_absolute', 'speedintensityzone5zone6relative': 'csv_speed_intensity_zone_5_zone_6_relative', 'speedintensityzone6absolute': 'csv_speed_intensity_zone_6_absolute', 'speedintensityzone6relative': 'csv_speed_intensity_zone_6_relative', 'stepbalance': 'csv_step_balance', 'timeinheartratezone1relative': 'csv_time_in_heart_rate_zone_1_relative', 'timeinheartratezone2relative': 'csv_time_in_heart_rate_zone_2_relative', 'timeinheartratezone2zone6relative': 'csv_time_in_heart_rate_zone_2_zone_6_relative', 'timeinheartratezone3relative': 'csv_time_in_heart_rate_zone_3_relative', 'timeinheartratezone3zone6relative': 'csv_time_in_heart_rate_zone_3_zone_6_relative', 'timeinheartratezone4relative': 'csv_time_in_heart_rate_zone_4_relative', 'timeinheartratezone4zone6relative': 'csv_time_in_heart_rate_zone_4_zone_6_relative', 'timeinheartratezone5relative': 'csv_time_in_heart_rate_zone_5_relative', 'timeinheartratezone6relative': 'csv_time_in_heart_rate_zone_6_relative', 'timezone1absolute': 'csv_time_zone_1_absolute', 'timezone1relative': 'csv_time_zone_1_relative', 'timezone2absolute': 'csv_time_zone_2_absolute', 'timezone2relative': 'csv_time_zone_2_relative', 'timezone3absolute': 'csv_time_zone_3_absolute', 'timezone3relative': 'csv_time_zone_3_relative', 'timezone4absolute': 'csv_time_zone_4_absolute', 'timezone4relative': 'csv_time_zone_4_relative', 'timezone5absolute': 'csv_time_zone_5_absolute', 'timezone5relative': 'csv_time_zone_5_relative', 'timezone6absolute': 'csv_time_zone_6_absolute', 'timezone6relative': 'csv_time_zone_6_relative', 'totalleftsteps': 'csv_total_left_steps', 'totalmetabolicpower': 'csv_total_metabolic_power', 'totalrightsteps': 'csv_total_right_steps', 'totaltime': 'csv_total_time'}
CSV_SOURCE_HEADER_MAP = {'% Time In Red Zone (Relative)': 'csv_pct_time_in_red_zone_relative', 'Accelerations Total Distance Zone 1 (Relative)': 'csv_accelerations_total_distance_zone_1_relative', 'Accelerations Total Distance Zone 2 (Relative)': 'csv_accelerations_total_distance_zone_2_relative', 'Accelerations Total Distance Zone 3 (Relative)': 'csv_accelerations_total_distance_zone_3_relative', 'Accelerations Total Distance Zone 4 (Absolute)': 'csv_accelerations_total_distance_zone_4_absolute', 'Accelerations Total Distance Zone 4 (Relative)': 'csv_accelerations_total_distance_zone_4_relative', 'Accelerations Total Distance Zone 5 (Relative)': 'csv_accelerations_total_distance_zone_5_relative', 'Accelerations Total Distance Zone 6 (Relative)': 'csv_accelerations_total_distance_zone_6_relative', 'Accelerations Total Time Zone 1 (Relative)': 'csv_accelerations_total_time_zone_1_relative', 'Accelerations Total Time Zone 2 (Relative)': 'csv_accelerations_total_time_zone_2_relative', 'Accelerations Total Time Zone 3 (Relative)': 'csv_accelerations_total_time_zone_3_relative', 'Accelerations Total Time Zone 4 (Relative)': 'csv_accelerations_total_time_zone_4_relative', 'Accelerations Total Time Zone 5 (Relative)': 'csv_accelerations_total_time_zone_5_relative', 'Accelerations Total Time Zone 6 (Relative)': 'csv_accelerations_total_time_zone_6_relative', 'Accelerations Zone 1 (Relative)': 'csv_accelerations_zone_1_relative', 'Accelerations Zone 2 (Relative)': 'csv_accelerations_zone_2_relative', 'Accelerations Zone 3 (Relative)': 'csv_accelerations_zone_3_relative', 'Accelerations Zone 3 - Zone 6 (Relative)': 'csv_accelerations_zone_3_zone_6_relative', 'Accelerations Zone 4 (Relative)': 'csv_accelerations_zone_4_relative', 'Accelerations Zone 4 - Zone 6 (Relative)': 'csv_accelerations_zone_4_zone_6_relative', 'Accelerations Zone 5 (Relative)': 'csv_accelerations_zone_5_relative', 'Accelerations Zone 5 - Zone 6 (Relative)': 'csv_accelerations_zone_5_zone_6_relative', 'Accelerations Zone 6 (Relative)': 'csv_accelerations_zone_6_relative', 'Accelerations Per Min (Relative)': 'csv_accelerations_per_min_relative', 'Average Dive Impact': 'csv_average_dive_impact', 'Average GK Power': 'csv_average_gk_power', 'Average Metabolic Power': 'csv_average_metabolic_power', 'Average Time Since Last Accel': 'csv_average_time_since_last_accel', 'Average Time Since Last Decel': 'csv_average_time_since_last_decel', 'Average Time Since Last Dive': 'csv_average_time_since_last_dive', 'Average Time Since Last HIB': 'csv_average_time_since_last_hib', 'Average Time Since Last HML Effort': 'csv_average_time_since_last_hml_effort', 'Average Time Since Last Sprint': 'csv_average_time_since_last_sprint', 'Ball In Play Time': 'csv_ball_in_play_time', 'Ball In Play Time %': 'csv_ball_in_play_time_pct', 'Player Custom ID': 'csv_player_custom_id', 'Decelerations Total Distance Zone 1 (Relative)': 'csv_decelerations_total_distance_zone_1_relative', 'Decelerations Total Distance Zone 2 (Relative)': 'csv_decelerations_total_distance_zone_2_relative', 'Decelerations Total Distance Zone 3 (Relative)': 'csv_decelerations_total_distance_zone_3_relative', 'Decelerations Total Distance Zone 4 (Relative)': 'csv_decelerations_total_distance_zone_4_relative', 'Decelerations Total Distance Zone 5 (Relative)': 'csv_decelerations_total_distance_zone_5_relative', 'Decelerations Total Distance Zone 6 (Relative)': 'csv_decelerations_total_distance_zone_6_relative', 'Decelerations Total Time Zone 1 (Relative)': 'csv_decelerations_total_time_zone_1_relative', 'Decelerations Total Time Zone 2 (Relative)': 'csv_decelerations_total_time_zone_2_relative', 'Decelerations Total Time Zone 3 (Relative)': 'csv_decelerations_total_time_zone_3_relative', 'Decelerations Total Time Zone 4 (Relative)': 'csv_decelerations_total_time_zone_4_relative', 'Decelerations Total Time Zone 5 (Relative)': 'csv_decelerations_total_time_zone_5_relative', 'Decelerations Total Time Zone 6 (Relative)': 'csv_decelerations_total_time_zone_6_relative', 'Decelerations Zone 1 (Relative)': 'csv_decelerations_zone_1_relative', 'Decelerations Zone 2 (Relative)': 'csv_decelerations_zone_2_relative', 'Decelerations Zone 3 (Relative)': 'csv_decelerations_zone_3_relative', 'Decelerations Zone 3 - Zone 6 (Relative)': 'csv_decelerations_zone_3_zone_6_relative', 'Decelerations Zone 4 (Relative)': 'csv_decelerations_zone_4_relative', 'Decelerations Zone 4 - Zone 6 (Relative)': 'csv_decelerations_zone_4_zone_6_relative', 'Decelerations Zone 5 (Relative)': 'csv_decelerations_zone_5_relative', 'Decelerations Zone 5 - Zone 6 (Relative)': 'csv_decelerations_zone_5_zone_6_relative', 'Decelerations Zone 6 (Relative)': 'csv_decelerations_zone_6_relative', 'Decelerations Per Min (Relative)': 'csv_decelerations_per_min_relative', 'Distance Zone 1 (Absolute)': 'csv_distance_zone_1_absolute', 'Distance Zone 1 (Relative)': 'csv_distance_zone_1_relative', 'Distance Zone 2 (Absolute)': 'csv_distance_zone_2_absolute', 'Distance Zone 2 (Relative)': 'csv_distance_zone_2_relative', 'Distance Zone 2 - Zone 6 (Absolute)': 'csv_distance_zone_2_zone_6_absolute', 'Distance Zone 2 - Zone 6 (Relative)': 'csv_distance_zone_2_zone_6_relative', 'Distance Zone 3 (Absolute)': 'csv_distance_zone_3_absolute', 'Distance Zone 3 (Relative)': 'csv_distance_zone_3_relative', 'Distance Zone 3 - Zone 6 (Absolute)': 'csv_distance_zone_3_zone_6_absolute', 'Distance Zone 3 - Zone 6 (Relative)': 'csv_distance_zone_3_zone_6_relative', 'Distance Zone 4 (Absolute)': 'csv_distance_zone_4_absolute', 'Distance Zone 4 (Relative)': 'csv_distance_zone_4_relative', 'Distance Zone 4 - Zone 6 (Absolute)': 'csv_distance_zone_4_zone_6_absolute', 'Distance Zone 4 - Zone 6 (Relative)': 'csv_distance_zone_4_zone_6_relative', 'Distance Zone 5 (Absolute)': 'csv_distance_zone_5_absolute', 'Distance Zone 5 (Relative)': 'csv_distance_zone_5_relative', 'Distance Zone 6 (Absolute)': 'csv_distance_zone_6_absolute', 'Distance Zone 6 (Relative)': 'csv_distance_zone_6_relative', 'Dives': 'csv_dives', 'Dives Left': 'csv_dives_left', 'Dives Right': 'csv_dives_right', 'Drill Date': 'csv_drill_date', 'Drill End Time': 'csv_drill_end_time', 'Drill Start Time': 'csv_drill_start_time', 'Drill Title': 'csv_drill_title', 'Duration Of High Intensity Bursts': 'csv_duration_of_high_intensity_bursts', 'Dynamic Load Anterior': 'csv_dynamic_load_anterior', 'Dynamic Load Lateral': 'csv_dynamic_load_lateral', 'Dynamic Load Vertical': 'csv_dynamic_load_vertical', 'Dynamic Stress Load': 'csv_dynamic_stress_load', 'Dynamic Stress Load Time Zone 1': 'csv_dynamic_stress_load_time_zone_1', 'Dynamic Stress Load Time Zone 2': 'csv_dynamic_stress_load_time_zone_2', 'Dynamic Stress Load Time Zone 3': 'csv_dynamic_stress_load_time_zone_3', 'Dynamic Stress Load Time Zone 4': 'csv_dynamic_stress_load_time_zone_4', 'Dynamic Stress Load Time Zone 5': 'csv_dynamic_stress_load_time_zone_5', 'Dynamic Stress Load Time Zone 6': 'csv_dynamic_stress_load_time_zone_6', 'Dynamic Stress Load Zone 1': 'csv_dynamic_stress_load_zone_1', 'Dynamic Stress Load Zone 2': 'csv_dynamic_stress_load_zone_2', 'Dynamic Stress Load Zone 3': 'csv_dynamic_stress_load_zone_3', 'Dynamic Stress Load Zone 3 - Zone 6': 'csv_dynamic_stress_load_zone_3_zone_6', 'Dynamic Stress Load Zone 4': 'csv_dynamic_stress_load_zone_4', 'Dynamic Stress Load Zone 4 - Zone 6': 'csv_dynamic_stress_load_zone_4_zone_6', 'Dynamic Stress Load Zone 5': 'csv_dynamic_stress_load_zone_5', 'Dynamic Stress Load Zone 5 - Zone 6': 'csv_dynamic_stress_load_zone_5_zone_6', 'Dynamic Stress Load Zone 6': 'csv_dynamic_stress_load_zone_6', 'EDI %': 'csv_edi_pct', 'Energy Expenditure (Kcal)': 'csv_energy_expenditure_kcal', 'Entries Zone 3 (Absolute)': 'csv_entries_zone_3_absolute', 'Entries Zone 3 (Relative)': 'csv_entries_zone_3_relative', 'Entries Zone 4 (Absolute)': 'csv_entries_zone_4_absolute', 'Entries Zone 4 (Relative)': 'csv_entries_zone_4_relative', 'Entries Zone 5 (Absolute)': 'csv_entries_zone_5_absolute', 'Entries Zone 5 (Relative)': 'csv_entries_zone_5_relative', 'Entries Zone 6 (Absolute)': 'csv_entries_zone_6_absolute', 'Entries Zone 6 (Relative)': 'csv_entries_zone_6_relative', 'Equivalent Metabolic Distance': 'csv_equivalent_metabolic_distance', 'Explosive Distance (Relative)': 'csv_explosive_distance_relative', 'External Work': 'csv_external_work', 'Fatigue Index': 'csv_fatigue_index', 'GK Load': 'csv_gk_load', 'High Intensity Bursts Maximum Speed': 'csv_high_intensity_bursts_maximum_speed', 'High Intensity Bursts Total Distance': 'csv_high_intensity_bursts_total_distance', 'HML Efforts': 'csv_hml_efforts', 'HML Efforts Total Distance': 'csv_hml_efforts_total_distance', 'HML Time': 'csv_hml_time', 'HMLD Per Minute': 'csv_hmld_per_minute', 'HSR Per Minute (Absolute)': 'csv_hsr_per_minute_absolute', 'HSR Per Minute (Relative)': 'csv_hsr_per_minute_relative', 'Impacts (Relative)': 'csv_impacts_relative', 'Impacts Zone 1 (Relative)': 'csv_impacts_zone_1_relative', 'Impacts Zone 2 (Relative)': 'csv_impacts_zone_2_relative', 'Impacts Zone 3 (Relative)': 'csv_impacts_zone_3_relative', 'Impacts Zone 3 - Zone 6 (Relative)': 'csv_impacts_zone_3_zone_6_relative', 'Impacts Zone 4 (Relative)': 'csv_impacts_zone_4_relative', 'Impacts Zone 4 - Zone 6 (Relative)': 'csv_impacts_zone_4_zone_6_relative', 'Impacts Zone 5 (Relative)': 'csv_impacts_zone_5_relative', 'Impacts Zone 5 - Zone 6 (Relative)': 'csv_impacts_zone_5_zone_6_relative', 'Impacts Zone 6 (Relative)': 'csv_impacts_zone_6_relative', 'Left Anterior Post Impact': 'csv_left_anterior_post_impact', 'Left Average Vertical Impact': 'csv_left_average_vertical_impact', 'Left Lateral Impact': 'csv_left_lateral_impact', 'Left Mag Impact': 'csv_left_mag_impact', 'Left Vertical Impact': 'csv_left_vertical_impact', 'Max Heart Rate': 'csv_max_heart_rate', 'Metabolic Distance Zone 1 (Relative)': 'csv_metabolic_distance_zone_1_relative', 'Metabolic Distance Zone 2 (Relative)': 'csv_metabolic_distance_zone_2_relative', 'Metabolic Distance Zone 3 (Relative)': 'csv_metabolic_distance_zone_3_relative', 'Metabolic Distance Zone 4 (Relative)': 'csv_metabolic_distance_zone_4_relative', 'Metabolic Distance Zone 5 (Relative)': 'csv_metabolic_distance_zone_5_relative', 'Metabolic Distance Zone 6 (Relative)': 'csv_metabolic_distance_zone_6_relative', 'Metabolic Time (Relative)': 'csv_metabolic_time_relative', 'Metabolic Time Zone 1 (Absolute)': 'csv_metabolic_time_zone_1_absolute', 'Metabolic Time Zone 1 (Relative)': 'csv_metabolic_time_zone_1_relative', 'Metabolic Time Zone 2 (Absolute)': 'csv_metabolic_time_zone_2_absolute', 'Metabolic Time Zone 2 (Relative)': 'csv_metabolic_time_zone_2_relative', 'Metabolic Time Zone 3 (Relative)': 'csv_metabolic_time_zone_3_relative', 'Metabolic Time Zone 4 (Relative)': 'csv_metabolic_time_zone_4_relative', 'Metabolic Time Zone 5 (Relative)': 'csv_metabolic_time_zone_5_relative', 'Metabolic Time Zone 6 (Relative)': 'csv_metabolic_time_zone_6_relative', 'No of Satellites': 'csv_no_of_satellites', 'Number Of High Intensity Bursts': 'csv_number_of_high_intensity_bursts', 'Player Date of Birth': 'csv_player_date_of_birth', 'Player Display Name': 'csv_player_display_name', 'Player First Name': 'csv_player_first_name', 'Player Height': 'csv_player_height', 'Player Last Name': 'csv_player_last_name', 'Player Max Accel': 'csv_player_max_accel', 'Player Max Decel': 'csv_player_max_decel', 'Player Max Heart Rate': 'csv_player_max_heart_rate', 'Player Name': 'csv_player_name', 'Player Primary Position': 'csv_player_primary_position', 'Player Resting Heart Rate': 'csv_player_resting_heart_rate', 'Player Secondary Position': 'csv_player_secondary_position', 'Player Sprint Threshold': 'csv_player_sprint_threshold', 'Player Weight': 'csv_player_weight', 'Quality of Signal': 'csv_quality_of_signal', 'Right Anterior Post Impact': 'csv_right_anterior_post_impact', 'Right Average Vertical Impact': 'csv_right_average_vertical_impact', 'Right Lateral Impact': 'csv_right_lateral_impact', 'Right Mag Impact': 'csv_right_mag_impact', 'Right Vertical Impact': 'csv_right_vertical_impact', 'Session Date': 'csv_session_date', 'Session Day of Week': 'csv_session_day_of_week', 'Session End Time': 'csv_session_end_time', 'Session Start Time': 'csv_session_start_time', 'Session Title': 'csv_session_title', 'Session Type': 'csv_session_type', 'Session Week Number': 'csv_session_week_number', 'Speed Intensity': 'csv_speed_intensity', 'Speed Intensity Zone 1 (Absolute)': 'csv_speed_intensity_zone_1_absolute', 'Speed Intensity Zone 1(Relative)': 'csv_speed_intensity_zone_1_relative', 'Speed Intensity Zone 2 (Absolute)': 'csv_speed_intensity_zone_2_absolute', 'Speed Intensity Zone 2 (Relative)': 'csv_speed_intensity_zone_2_relative', 'Speed Intensity Zone 3 (Absolute)': 'csv_speed_intensity_zone_3_absolute', 'Speed Intensity Zone 3 (Relative)': 'csv_speed_intensity_zone_3_relative', 'Speed Intensity Zone 3 - Zone 6 (Absolute)': 'csv_speed_intensity_zone_3_zone_6_absolute', 'Speed Intensity Zone 3 - Zone 6 (Relative)': 'csv_speed_intensity_zone_3_zone_6_relative', 'Speed Intensity Zone 4 (Absolute)': 'csv_speed_intensity_zone_4_absolute', 'Speed Intensity Zone 4 (Relative)': 'csv_speed_intensity_zone_4_relative', 'Speed Intensity Zone 4 - Zone 6 (Absolute)': 'csv_speed_intensity_zone_4_zone_6_absolute', 'Speed Intensity Zone 4 - Zone 6 (Relative)': 'csv_speed_intensity_zone_4_zone_6_relative', 'Speed Intensity Zone 5 (Absolute)': 'csv_speed_intensity_zone_5_absolute', 'Speed Intensity Zone 5 (Relative)': 'csv_speed_intensity_zone_5_relative', 'Speed Intensity Zone 5 - Zone 6 (Absolute)': 'csv_speed_intensity_zone_5_zone_6_absolute', 'Speed Intensity Zone 5 - Zone 6 (Relative)': 'csv_speed_intensity_zone_5_zone_6_relative', 'Speed Intensity Zone 6 (Absolute)': 'csv_speed_intensity_zone_6_absolute', 'Speed Intensity Zone 6 (Relative)': 'csv_speed_intensity_zone_6_relative', 'Step Balance': 'csv_step_balance', 'Time In Heart Rate Zone 1 (Relative)': 'csv_time_in_heart_rate_zone_1_relative', 'Time In Heart Rate Zone 2 (Relative)': 'csv_time_in_heart_rate_zone_2_relative', 'Time In Heart Rate Zone 2 - Zone 6 (Relative)': 'csv_time_in_heart_rate_zone_2_zone_6_relative', 'Time In Heart Rate Zone 3 (Relative)': 'csv_time_in_heart_rate_zone_3_relative', 'Time In Heart Rate Zone 3 - Zone 6 (Relative)': 'csv_time_in_heart_rate_zone_3_zone_6_relative', 'Time In Heart Rate Zone 4 (Relative)': 'csv_time_in_heart_rate_zone_4_relative', 'Time In Heart Rate Zone 4 - Zone 6 (Relative)': 'csv_time_in_heart_rate_zone_4_zone_6_relative', 'Time In Heart Rate Zone 5 (Relative)': 'csv_time_in_heart_rate_zone_5_relative', 'Time In Heart Rate Zone 6 (Relative)': 'csv_time_in_heart_rate_zone_6_relative', 'Time In Red Zone (Relative)': 'csv_time_in_red_zone_relative', 'Time Zone 1 (Absolute)': 'csv_time_zone_1_absolute', 'Time Zone 1 (Relative)': 'csv_time_zone_1_relative', 'Time Zone 2 (Absolute)': 'csv_time_zone_2_absolute', 'Time Zone 2 (Relative)': 'csv_time_zone_2_relative', 'Time Zone 3 (Absolute)': 'csv_time_zone_3_absolute', 'Time Zone 3 (Relative)': 'csv_time_zone_3_relative', 'Time Zone 4 (Absolute)': 'csv_time_zone_4_absolute', 'Time Zone 4 (Relative)': 'csv_time_zone_4_relative', 'Time Zone 5 (Absolute)': 'csv_time_zone_5_absolute', 'Time Zone 5 (Relative)': 'csv_time_zone_5_relative', 'Time Zone 6 (Absolute)': 'csv_time_zone_6_absolute', 'Time Zone 6 (Relative)': 'csv_time_zone_6_relative', 'Total Left Steps': 'csv_total_left_steps', 'Total Metabolic Power': 'csv_total_metabolic_power', 'Total Right Steps': 'csv_total_right_steps', 'Total Time': 'csv_total_time'}
CSV_SOURCE_TEXT_COLUMNS = frozenset(['csv_accelerations_total_time_zone_1_relative', 'csv_accelerations_total_time_zone_2_relative', 'csv_accelerations_total_time_zone_3_relative', 'csv_accelerations_total_time_zone_4_relative', 'csv_accelerations_total_time_zone_5_relative', 'csv_accelerations_total_time_zone_6_relative', 'csv_average_time_since_last_accel', 'csv_average_time_since_last_decel', 'csv_average_time_since_last_dive', 'csv_average_time_since_last_hib', 'csv_average_time_since_last_hml_effort', 'csv_average_time_since_last_sprint', 'csv_ball_in_play_time', 'csv_player_custom_id', 'csv_decelerations_total_time_zone_1_relative', 'csv_decelerations_total_time_zone_2_relative', 'csv_decelerations_total_time_zone_3_relative', 'csv_decelerations_total_time_zone_4_relative', 'csv_decelerations_total_time_zone_5_relative', 'csv_decelerations_total_time_zone_6_relative', 'csv_drill_date', 'csv_drill_end_time', 'csv_drill_start_time', 'csv_drill_title', 'csv_duration_of_high_intensity_bursts', 'csv_dynamic_stress_load_time_zone_1', 'csv_dynamic_stress_load_time_zone_2', 'csv_dynamic_stress_load_time_zone_3', 'csv_dynamic_stress_load_time_zone_4', 'csv_dynamic_stress_load_time_zone_5', 'csv_dynamic_stress_load_time_zone_6', 'csv_hml_time', 'csv_metabolic_time_relative', 'csv_metabolic_time_zone_1_absolute', 'csv_metabolic_time_zone_1_relative', 'csv_metabolic_time_zone_2_absolute', 'csv_metabolic_time_zone_2_relative', 'csv_metabolic_time_zone_3_relative', 'csv_metabolic_time_zone_4_relative', 'csv_metabolic_time_zone_5_relative', 'csv_metabolic_time_zone_6_relative', 'csv_player_date_of_birth', 'csv_player_display_name', 'csv_player_first_name', 'csv_player_last_name', 'csv_player_name', 'csv_player_primary_position', 'csv_player_secondary_position', 'csv_session_date', 'csv_session_day_of_week', 'csv_session_end_time', 'csv_session_start_time', 'csv_session_title', 'csv_session_type', 'csv_time_in_heart_rate_zone_1_relative', 'csv_time_in_heart_rate_zone_2_relative', 'csv_time_in_heart_rate_zone_2_zone_6_relative', 'csv_time_in_heart_rate_zone_3_relative', 'csv_time_in_heart_rate_zone_3_zone_6_relative', 'csv_time_in_heart_rate_zone_4_relative', 'csv_time_in_heart_rate_zone_4_zone_6_relative', 'csv_time_in_heart_rate_zone_5_relative', 'csv_time_in_heart_rate_zone_6_relative', 'csv_time_in_red_zone_relative', 'csv_time_zone_1_absolute', 'csv_time_zone_1_relative', 'csv_time_zone_2_absolute', 'csv_time_zone_2_relative', 'csv_time_zone_3_absolute', 'csv_time_zone_3_relative', 'csv_time_zone_4_absolute', 'csv_time_zone_4_relative', 'csv_time_zone_5_absolute', 'csv_time_zone_5_relative', 'csv_time_zone_6_absolute', 'csv_time_zone_6_relative', 'csv_total_time'])
CSV_DIRECT_COLS = ('csv_pct_time_in_red_zone_relative', 'csv_accelerations_total_distance_zone_1_relative', 'csv_accelerations_total_distance_zone_2_relative', 'csv_accelerations_total_distance_zone_3_relative', 'csv_accelerations_total_distance_zone_4_absolute', 'csv_accelerations_total_distance_zone_4_relative', 'csv_accelerations_total_distance_zone_5_relative', 'csv_accelerations_total_distance_zone_6_relative', 'csv_accelerations_total_time_zone_1_relative', 'csv_accelerations_total_time_zone_2_relative', 'csv_accelerations_total_time_zone_3_relative', 'csv_accelerations_total_time_zone_4_relative', 'csv_accelerations_total_time_zone_5_relative', 'csv_accelerations_total_time_zone_6_relative', 'csv_accelerations_zone_1_relative', 'csv_accelerations_zone_2_relative', 'csv_accelerations_zone_3_relative', 'csv_accelerations_zone_3_zone_6_relative', 'csv_accelerations_zone_4_relative', 'csv_accelerations_zone_4_zone_6_relative', 'csv_accelerations_zone_5_relative', 'csv_accelerations_zone_5_zone_6_relative', 'csv_accelerations_zone_6_relative', 'csv_accelerations_per_min_relative', 'csv_average_dive_impact', 'csv_average_gk_power', 'csv_average_metabolic_power', 'csv_average_time_since_last_accel', 'csv_average_time_since_last_decel', 'csv_average_time_since_last_dive', 'csv_average_time_since_last_hib', 'csv_average_time_since_last_hml_effort', 'csv_average_time_since_last_sprint', 'csv_ball_in_play_time', 'csv_ball_in_play_time_pct', 'csv_player_custom_id', 'csv_decelerations_total_distance_zone_1_relative', 'csv_decelerations_total_distance_zone_2_relative', 'csv_decelerations_total_distance_zone_3_relative', 'csv_decelerations_total_distance_zone_4_relative', 'csv_decelerations_total_distance_zone_5_relative', 'csv_decelerations_total_distance_zone_6_relative', 'csv_decelerations_total_time_zone_1_relative', 'csv_decelerations_total_time_zone_2_relative', 'csv_decelerations_total_time_zone_3_relative', 'csv_decelerations_total_time_zone_4_relative', 'csv_decelerations_total_time_zone_5_relative', 'csv_decelerations_total_time_zone_6_relative', 'csv_decelerations_zone_1_relative', 'csv_decelerations_zone_2_relative', 'csv_decelerations_zone_3_relative', 'csv_decelerations_zone_3_zone_6_relative', 'csv_decelerations_zone_4_relative', 'csv_decelerations_zone_4_zone_6_relative', 'csv_decelerations_zone_5_relative', 'csv_decelerations_zone_5_zone_6_relative', 'csv_decelerations_zone_6_relative', 'csv_decelerations_per_min_relative', 'csv_distance_zone_1_absolute', 'csv_distance_zone_1_relative', 'csv_distance_zone_2_absolute', 'csv_distance_zone_2_relative', 'csv_distance_zone_2_zone_6_absolute', 'csv_distance_zone_2_zone_6_relative', 'csv_distance_zone_3_absolute', 'csv_distance_zone_3_relative', 'csv_distance_zone_3_zone_6_absolute', 'csv_distance_zone_3_zone_6_relative', 'csv_distance_zone_4_absolute', 'csv_distance_zone_4_relative', 'csv_distance_zone_4_zone_6_absolute', 'csv_distance_zone_4_zone_6_relative', 'csv_distance_zone_5_absolute', 'csv_distance_zone_5_relative', 'csv_distance_zone_6_absolute', 'csv_distance_zone_6_relative', 'csv_dives', 'csv_dives_left', 'csv_dives_right', 'csv_drill_date', 'csv_drill_end_time', 'csv_drill_start_time', 'csv_drill_title', 'csv_duration_of_high_intensity_bursts', 'csv_dynamic_load_anterior', 'csv_dynamic_load_lateral', 'csv_dynamic_load_vertical', 'csv_dynamic_stress_load', 'csv_dynamic_stress_load_time_zone_1', 'csv_dynamic_stress_load_time_zone_2', 'csv_dynamic_stress_load_time_zone_3', 'csv_dynamic_stress_load_time_zone_4', 'csv_dynamic_stress_load_time_zone_5', 'csv_dynamic_stress_load_time_zone_6', 'csv_dynamic_stress_load_zone_1', 'csv_dynamic_stress_load_zone_2', 'csv_dynamic_stress_load_zone_3', 'csv_dynamic_stress_load_zone_3_zone_6', 'csv_dynamic_stress_load_zone_4', 'csv_dynamic_stress_load_zone_4_zone_6', 'csv_dynamic_stress_load_zone_5', 'csv_dynamic_stress_load_zone_5_zone_6', 'csv_dynamic_stress_load_zone_6', 'csv_edi_pct', 'csv_energy_expenditure_kcal', 'csv_entries_zone_3_absolute', 'csv_entries_zone_3_relative', 'csv_entries_zone_4_absolute', 'csv_entries_zone_4_relative', 'csv_entries_zone_5_absolute', 'csv_entries_zone_5_relative', 'csv_entries_zone_6_absolute', 'csv_entries_zone_6_relative', 'csv_equivalent_metabolic_distance', 'csv_explosive_distance_relative', 'csv_external_work', 'csv_fatigue_index', 'csv_gk_load', 'csv_high_intensity_bursts_maximum_speed', 'csv_high_intensity_bursts_total_distance', 'csv_hml_efforts', 'csv_hml_efforts_total_distance', 'csv_hml_time', 'csv_hmld_per_minute', 'csv_hsr_per_minute_absolute', 'csv_hsr_per_minute_relative', 'csv_impacts_relative', 'csv_impacts_zone_1_relative', 'csv_impacts_zone_2_relative', 'csv_impacts_zone_3_relative', 'csv_impacts_zone_3_zone_6_relative', 'csv_impacts_zone_4_relative', 'csv_impacts_zone_4_zone_6_relative', 'csv_impacts_zone_5_relative', 'csv_impacts_zone_5_zone_6_relative', 'csv_impacts_zone_6_relative', 'csv_left_anterior_post_impact', 'csv_left_average_vertical_impact', 'csv_left_lateral_impact', 'csv_left_mag_impact', 'csv_left_vertical_impact', 'csv_max_heart_rate', 'csv_metabolic_distance_zone_1_relative', 'csv_metabolic_distance_zone_2_relative', 'csv_metabolic_distance_zone_3_relative', 'csv_metabolic_distance_zone_4_relative', 'csv_metabolic_distance_zone_5_relative', 'csv_metabolic_distance_zone_6_relative', 'csv_metabolic_time_relative', 'csv_metabolic_time_zone_1_absolute', 'csv_metabolic_time_zone_1_relative', 'csv_metabolic_time_zone_2_absolute', 'csv_metabolic_time_zone_2_relative', 'csv_metabolic_time_zone_3_relative', 'csv_metabolic_time_zone_4_relative', 'csv_metabolic_time_zone_5_relative', 'csv_metabolic_time_zone_6_relative', 'csv_no_of_satellites', 'csv_number_of_high_intensity_bursts', 'csv_player_date_of_birth', 'csv_player_display_name', 'csv_player_first_name', 'csv_player_height', 'csv_player_last_name', 'csv_player_max_accel', 'csv_player_max_decel', 'csv_player_max_heart_rate', 'csv_player_name', 'csv_player_primary_position', 'csv_player_resting_heart_rate', 'csv_player_secondary_position', 'csv_player_sprint_threshold', 'csv_player_weight', 'csv_quality_of_signal', 'csv_right_anterior_post_impact', 'csv_right_average_vertical_impact', 'csv_right_lateral_impact', 'csv_right_mag_impact', 'csv_right_vertical_impact', 'csv_session_date', 'csv_session_day_of_week', 'csv_session_end_time', 'csv_session_start_time', 'csv_session_title', 'csv_session_type', 'csv_session_week_number', 'csv_speed_intensity', 'csv_speed_intensity_zone_1_absolute', 'csv_speed_intensity_zone_1_relative', 'csv_speed_intensity_zone_2_absolute', 'csv_speed_intensity_zone_2_relative', 'csv_speed_intensity_zone_3_absolute', 'csv_speed_intensity_zone_3_relative', 'csv_speed_intensity_zone_3_zone_6_absolute', 'csv_speed_intensity_zone_3_zone_6_relative', 'csv_speed_intensity_zone_4_absolute', 'csv_speed_intensity_zone_4_relative', 'csv_speed_intensity_zone_4_zone_6_absolute', 'csv_speed_intensity_zone_4_zone_6_relative', 'csv_speed_intensity_zone_5_absolute', 'csv_speed_intensity_zone_5_relative', 'csv_speed_intensity_zone_5_zone_6_absolute', 'csv_speed_intensity_zone_5_zone_6_relative', 'csv_speed_intensity_zone_6_absolute', 'csv_speed_intensity_zone_6_relative', 'csv_step_balance', 'csv_time_in_heart_rate_zone_1_relative', 'csv_time_in_heart_rate_zone_2_relative', 'csv_time_in_heart_rate_zone_2_zone_6_relative', 'csv_time_in_heart_rate_zone_3_relative', 'csv_time_in_heart_rate_zone_3_zone_6_relative', 'csv_time_in_heart_rate_zone_4_relative', 'csv_time_in_heart_rate_zone_4_zone_6_relative', 'csv_time_in_heart_rate_zone_5_relative', 'csv_time_in_heart_rate_zone_6_relative', 'csv_time_in_red_zone_relative', 'csv_time_zone_1_absolute', 'csv_time_zone_1_relative', 'csv_time_zone_2_absolute', 'csv_time_zone_2_relative', 'csv_time_zone_3_absolute', 'csv_time_zone_3_relative', 'csv_time_zone_4_absolute', 'csv_time_zone_4_relative', 'csv_time_zone_5_absolute', 'csv_time_zone_5_relative', 'csv_time_zone_6_absolute', 'csv_time_zone_6_relative', 'csv_total_left_steps', 'csv_total_metabolic_power', 'csv_total_right_steps', 'csv_total_time')
ID_COLS_IN_PARSER = ["Speler", "Datum", "Week", "Year", "Type", "Event"]

# Session/player metadata is useful for parsing and auditability, but is not a
# performance metric. It stays out of extra_metrics to avoid duplicating the
# canonical gps_records identity fields.
CSV_METADATA_KEYS = {
    "playerdisplayname",
    "playername",
    "playercustomid",
    "playerfirstname",
    "playerlastname",
    "playerprimaryposition",
    "playersecondaryposition",
    "sessiondate",
    "sessionstarttime",
    "sessionendtime",
    "sessiontype",
    "sessiontitle",
    "drilldate",
    "drillstarttime",
    "drillendtime",
    "drilltitle",
    "teamname",
}

INT_DB_COLS = {
    "number_of_sprints",
    "number_of_high_sprints",
    "number_of_repeated_sprints",
    "total_accelerations",
    "high_accelerations",
    "total_decelerations",
    "high_decelerations",
    "heart_rate_recovery_beats",
    "steps",
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

def _source_text_value(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    value = str(v).strip()
    return value or None


def _speed_zone_source_columns(columns: list[str]) -> dict[int, str]:
    by_key = {}
    for column in columns:
        by_key.setdefault(normalize_key(column), column)

    result = {}
    for zone in range(1, 7):
        absolute = by_key.get(f"distancezone{zone}absolute")
        relative = by_key.get(f"distancezone{zone}relative")
        if absolute is not None:
            result[zone] = absolute
        elif relative is not None:
            result[zone] = relative
    return result


def _apply_speed_zone_mapping(base: dict, source_row, source_columns: dict[int, str]) -> None:
    if not all(zone in source_columns for zone in range(1, 6)):
        return

    if 6 in source_columns:
        # STATSports exposes all six speed zones. Keep both low-speed zones and
        # their combined total so they can be compared with Johan's zone_1_2 field.
        groups = {
            "td_zone_1": (1,),
            "td_zone_2": (2,),
            "td_zone_1_2": (1, 2),
            "td_zone_3": (3,),
            "td_zone_4": (4,),
            "td_zone_5": (5,),
            "td_zone_6": (6,),
        }
    else:
        # Johan Sports has five zones; its first zone remains the combined
        # Zone 1-2 comparison value, followed by Zones 3 through 6.
        groups = {
            "td_zone_1_2": (1,),
            "td_zone_3": (2,),
            "td_zone_4": (3,),
            "td_zone_5": (4,),
            "td_zone_6": (5,),
        }

    for target, zones in groups.items():
        values = [coerce_num(source_row[source_columns[zone]]) for zone in zones]
        values = [value for value in values if value is not None]
        if values:
            base[target] = sum(values)


def df_to_db_rows(df: pd.DataFrame, source_file: str, name_to_id: dict) -> tuple[list[dict], list[str]]:
    rows = []
    unmapped = set()

    parsed_dates = pd.to_datetime(df["Datum"], dayfirst=True, errors="coerce")
    if parsed_dates.isna().any():
        bad = df.loc[parsed_dates.isna(), "Datum"].head(5).tolist()
        raise ValueError(f"Kon sommige Datum waarden niet parsen: {bad}")

    dates_iso = parsed_dates.dt.date.astype(str)
    speed_zone_columns = _speed_zone_source_columns(list(df.columns))

    for idx, r in df.iterrows():
        speler = str(r.get("Speler", "")).strip()
        if not speler:
            continue

        pid = name_to_id.get(normalize_name(speler))
        if not pid:
            unmapped.add(speler)

        # Keep the original dataframe index; blank-player rows may have been
        # filtered before this function is called.
        dt = parsed_dates.loc[idx].date()
        t = str(r.get("Type", "")).strip()
        ev = str(r.get("Event", "")).strip()

        base = {
            "player_id": pid,
            "player_name": speler,
            "datum": dates_iso.loc[idx],
            "week": int(dt.isocalendar().week),
            "year": int(dt.year),
            "type": t,
            "event": ev,
            "match_id": None,
            "source_file": source_file,
            "extra_metrics": {},
        }

        for c in df.columns:
            if c in ID_COLS_IN_PARSER:
                continue

            key = normalize_key(c)
            val = r[c]
            direct_col = CSV_SOURCE_HEADER_MAP.get(str(c).strip()) or CSV_SOURCE_COLUMN_MAP.get(key)
            if direct_col is not None:
                if direct_col in CSV_SOURCE_TEXT_COLUMNS:
                    base[direct_col] = _source_text_value(val)
                else:
                    base[direct_col] = coerce_num(val)
                continue

            if key not in METRIC_MAP:
                extra_value = json_safe(val)
                if extra_value is not None and not (isinstance(extra_value, str) and not extra_value.strip()):
                    base["extra_metrics"][str(c).strip()] = extra_value
                continue

            db_col = METRIC_MAP[key]

            if db_col in INT_DB_COLS:
                v = pd.to_numeric(val, errors="coerce")
                base[db_col] = int(v) if pd.notna(v) else None
            else:
                base[db_col] = coerce_num(val)

        _apply_speed_zone_mapping(base, r, speed_zone_columns)
        rows.append(base)

    return rows, sorted(unmapped)


def _csv_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    by_key = {normalize_key(c): c for c in df.columns}
    for alias in aliases:
        found = by_key.get(normalize_key(alias))
        if found is not None:
            return found
    return None


def _read_csv_dataframe(file_bytes: bytes) -> pd.DataFrame:
    """Read vendor CSVs with a small amount of delimiter/encoding tolerance."""
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            text = file_bytes.decode(encoding)
            sample = text[:8192]
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                delimiter = ","
            df = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=object, keep_default_na=False)
            df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
            blank_columns = [c for c in df.columns if not c or c.lower().startswith("unnamed:")]
            if blank_columns:
                df = df.drop(columns=blank_columns)
            df.columns = maak_lijst_uniek(list(df.columns))
            if len(df.columns) < 2:
                raise ValueError("CSV bevat minder dan twee kolommen")
            return df
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("CSV kon niet worden gelezen. " + " | ".join(errors))


def parse_player_metrics_csv(file_bytes: bytes, selected_type: str) -> pd.DataFrame:
    """Parse a broad player/session export into the canonical GPS import shape."""
    df = _read_csv_dataframe(file_bytes)

    player_col = _csv_column(df, ["Player Display Name", "Player Name", "Speler", "Player"])
    date_col = _csv_column(df, ["Session Date", "Drill Date", "Datum", "Date"])
    # Session Title identifies the exported session; Drill Title is often only "Entire Session".
    event_col = _csv_column(df, ["Session Title", "Drill Title", "Event"])

    missing = []
    if player_col is None:
        missing.append("Player Display Name/Player Name")
    if date_col is None:
        missing.append("Session Date/Drill Date")
    if missing:
        raise ValueError("CSV mist verplichte kolommen: " + ", ".join(missing))

    parsed_dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    if parsed_dates.isna().any():
        bad = df.loc[parsed_dates.isna(), date_col].head(5).tolist()
        raise ValueError(f"Kon sommige sessiedatums niet parsen: {bad}")

    result = df.copy()
    player_values = result[player_col].astype(str).str.strip()
    if event_col is not None:
        event_values = result[event_col].astype(str).str.strip()
        fallback_event_col = _csv_column(df, ["Drill Title", "Event"])
        if fallback_event_col is not None and fallback_event_col != event_col:
            fallback_values = result[fallback_event_col].astype(str).str.strip()
            event_values = event_values.where(event_values.ne(""), fallback_values)
    else:
        event_values = pd.Series(["CSV import"] * len(result), index=result.index)

    # Some vendors already use one of the dashboard's canonical column names.
    # Remove those source columns before inserting the normalized values to
    # prevent duplicate labels and ambiguous upsert payloads.
    canonical_columns = ["Speler", "Datum", "Week", "Year", "Type", "Event"]
    result = result.drop(columns=[c for c in canonical_columns if c in result.columns])
    result.insert(0, "Speler", player_values)
    result.insert(1, "Datum", parsed_dates.dt.strftime("%d-%m-%Y"))
    result.insert(2, "Week", parsed_dates.dt.isocalendar().week.astype(int))
    result.insert(3, "Year", parsed_dates.dt.year.astype(int))
    result.insert(4, "Type", str(selected_type).strip())
    result.insert(5, "Event", event_values.where(event_values.ne(""), "CSV import"))

    result = result[result["Speler"].ne("")].copy()
    return ensure_unique_events(result)


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
    return _retry_with_refreshed_token(rest_get, access_token, "gps_records", query)


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
