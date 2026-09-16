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
GPS_COLS = ['player_id',
 'player_name',
 'datum',
 'week',
 'year',
 'type',
 'event',
 'match_id',
 'duration',
 'total_distance',
 'total_distance_zone_1',
 'total_distance_zone_2',
 'total_distance_zone_1_and_2',
 'total_distance_zone_3',
 'total_distance_zone_4',
 'total_distance_zone_5',
 'total_distance_zone_6',
 'number_of_sprints',
 'number_of_high_sprints',
 'number_of_repeated_sprints',
 'maximum_speed',
 'average_speed',
 'player_load_three_dimensional',
 'player_load_two_dimensional',
 'total_accelerations',
 'high_accelerations',
 'total_decelerations',
 'high_decelerations',
 'heart_rate_zone_1',
 'heart_rate_zone_2',
 'heart_rate_zone_3',
 'heart_rate_zone_4',
 'heart_rate_zone_5',
 'heart_rate_training_impulse',
 'heart_rate_anaerobic_zone',
 'average_heart_rate',
 'maximum_heart_rate',
 'acceleration_impulse',
 'total_acceleration_loading',
 'total_deceleration_loading',
 'maximum_acceleration',
 'maximum_deceleration',
 'explosive_distance',
 'high_metabolic_load_distance',
 'high_metabolic_load_efforts_maximum_speed',
 'lower_speed_loading',
 'heart_rate_load',
 'heart_rate_exertion',
 'heart_rate_recovery_percentage',
 'heart_rate_recovery_beats',
 'heart_rate_variability',
 'minimum_heart_rate',
 'extra_metrics']

METRIC_MAP = {'duration': 'duration',
 'totaldistance': 'total_distance',
 'distancetotal': 'total_distance',
 'walkdistance': 'total_distance_zone_1_and_2',
 'jogdistance': 'total_distance_zone_3',
 'rundistance': 'total_distance_zone_4',
 'sprintdistance': 'total_distance_zone_5',
 'hisprintdistance': 'total_distance_zone_6',
 'highsprintdistance': 'total_distance_zone_6',
 'highspeedrunningabsolute': 'total_distance_zone_6',
 'numberofsprints': 'number_of_sprints',
 'sprints': 'number_of_sprints',
 'numberofhisprints': 'number_of_high_sprints',
 'numberofhighsprints': 'number_of_high_sprints',
 'numberofrepeatedsprints': 'number_of_repeated_sprints',
 'maxspeed': 'maximum_speed',
 'avgspeed': 'average_speed',
 'averagespeed': 'average_speed',
 'playerload3d': 'player_load_three_dimensional',
 'playerload2d': 'player_load_two_dimensional',
 'totalaccelerations': 'total_accelerations',
 'highaccelerations': 'high_accelerations',
 'totaldecelerations': 'total_decelerations',
 'highdecelerations': 'high_decelerations',
 'hrzone1': 'heart_rate_zone_1',
 'hrzone2': 'heart_rate_zone_2',
 'hrzone3': 'heart_rate_zone_3',
 'hrzone4': 'heart_rate_zone_4',
 'hrzone5': 'heart_rate_zone_5',
 'hrtrimp': 'heart_rate_training_impulse',
 'hrzoneanaerobic': 'heart_rate_anaerobic_zone',
 'avghr': 'average_heart_rate',
 'averageheartrate': 'average_heart_rate',
 'avgheartrate': 'average_heart_rate',
 'maxhr': 'maximum_heart_rate',
 'maximumheartrate': 'maximum_heart_rate',
 'maxheartrate': 'maximum_heart_rate',
 'accelerationimpulse': 'acceleration_impulse',
 'totalaccelerationloading': 'total_acceleration_loading',
 'totaldecelerationloading': 'total_deceleration_loading',
 'maxacceleration': 'maximum_acceleration',
 'maxdeceleration': 'maximum_deceleration',
 'explosivedistanceabsolute': 'explosive_distance',
 'hmldistance': 'high_metabolic_load_distance',
 'hmleffortsmaximumspeed': 'high_metabolic_load_efforts_maximum_speed',
 'lowerspeedloading': 'lower_speed_loading',
 'heartrateload': 'heart_rate_load',
 'heartrateexertion': 'heart_rate_exertion',
 'heartraterecovery': 'heart_rate_recovery_percentage',
 'heartraterecoverynumberofbeats': 'heart_rate_recovery_beats',
 'heartratevariability': 'heart_rate_variability',
 'minimumheartrate': 'minimum_heart_rate',
 'walking': 'total_distance_zone_1_and_2',
 'jogging': 'total_distance_zone_3',
 'running': 'total_distance_zone_4',
 'sprint': 'total_distance_zone_5',
 'highsprint': 'total_distance_zone_6',
 'hisprint': 'total_distance_zone_6',
 'walk': 'total_distance_zone_1_and_2',
 'jog': 'total_distance_zone_3',
 'run': 'total_distance_zone_4'}

# Source-complete fields from the STATSports/Johan Sports player-metrics export.
# Exact headers are kept so similarly normalized fields (for example time and time %) do not collide.
CSV_SOURCE_COLUMN_MAP = {'accelerationstotaldistancezone4absolute': 'accelerations_total_distance_zone_4_absolute',
 'averagediveimpact': 'average_dive_impact',
 'averagemetabolicpower': 'average_metabolic_power',
 'averagetimesincelastaccel': 'average_time_since_last_acceleration',
 'averagetimesincelastdecel': 'average_time_since_last_deceleration',
 'averagetimesincelastdive': 'average_time_since_last_dive',
 'averagetimesincelasthib': 'average_time_since_last_high_intensity_burst',
 'averagetimesincelasthmleffort': 'average_time_since_last_high_metabolic_load_effort',
 'averagetimesincelastsprint': 'average_time_since_last_sprint',
 'distancezone1absolute': 'distance_zone_1_absolute',
 'distancezone2absolute': 'distance_zone_2_absolute',
 'distancezone3absolute': 'distance_zone_3_absolute',
 'distancezone4absolute': 'distance_zone_4_absolute',
 'distancezone5absolute': 'distance_zone_5_absolute',
 'distancezone6absolute': 'distance_zone_6_absolute',
 'dives': 'dives',
 'divesleft': 'dives_left',
 'divesright': 'dives_right',
 'drilldate': 'drill_date',
 'drillendtime': 'drill_end_time',
 'drillstarttime': 'drill_start_time',
 'drilltitle': 'drill_title',
 'durationofhighintensitybursts': 'duration_of_high_intensity_bursts',
 'dynamicloadanterior': 'dynamic_load_anterior',
 'dynamicloadlateral': 'dynamic_load_lateral',
 'dynamicloadvertical': 'dynamic_load_vertical',
 'dynamicstressload': 'dynamic_stress_load',
 'dynamicstressloadtimezone1': 'dynamic_stress_load_time_zone_1',
 'dynamicstressloadtimezone2': 'dynamic_stress_load_time_zone_2',
 'dynamicstressloadtimezone3': 'dynamic_stress_load_time_zone_3',
 'dynamicstressloadtimezone4': 'dynamic_stress_load_time_zone_4',
 'dynamicstressloadtimezone5': 'dynamic_stress_load_time_zone_5',
 'dynamicstressloadtimezone6': 'dynamic_stress_load_time_zone_6',
 'dynamicstressloadzone1': 'dynamic_stress_load_zone_1',
 'dynamicstressloadzone2': 'dynamic_stress_load_zone_2',
 'dynamicstressloadzone3': 'dynamic_stress_load_zone_3',
 'dynamicstressloadzone4': 'dynamic_stress_load_zone_4',
 'dynamicstressloadzone5': 'dynamic_stress_load_zone_5',
 'dynamicstressloadzone6': 'dynamic_stress_load_zone_6',
 'edi': 'equivalent_distance_index_percentage',
 'energyexpenditurekcal': 'energy_expenditure_kilocalories',
 'entrieszone3absolute': 'entries_zone_3_absolute',
 'entrieszone4absolute': 'entries_zone_4_absolute',
 'entrieszone5absolute': 'entries_zone_5_absolute',
 'entrieszone6absolute': 'entries_zone_6_absolute',
 'equivalentmetabolicdistance': 'equivalent_metabolic_distance',
 'externalwork': 'external_work',
 'fatigueindex': 'fatigue_index',
 'highintensityburstsmaximumspeed': 'high_intensity_bursts_maximum_speed',
 'highintensityburststotaldistance': 'high_intensity_bursts_total_distance',
 'hmlefforts': 'high_metabolic_load_efforts',
 'hmleffortstotaldistance': 'high_metabolic_load_efforts_total_distance',
 'hmltime': 'high_metabolic_load_time',
 'leftanteriorpostimpact': 'left_anterior_posterior_impact',
 'leftaverageverticalimpact': 'left_average_vertical_impact',
 'leftlateralimpact': 'left_lateral_impact',
 'leftmagimpact': 'left_magnitude_impact',
 'leftverticalimpact': 'left_vertical_impact',
 'maxheartrate': 'maximum_heart_rate',
 'metabolictimezone1absolute': 'metabolic_time_zone_1_absolute',
 'metabolictimezone2absolute': 'metabolic_time_zone_2_absolute',
 'playername': 'source_player_name',
 'rightanteriorpostimpact': 'right_anterior_posterior_impact',
 'rightaverageverticalimpact': 'right_average_vertical_impact',
 'rightlateralimpact': 'right_lateral_impact',
 'rightmagimpact': 'right_magnitude_impact',
 'rightverticalimpact': 'right_vertical_impact',
 'sessiondate': 'session_date',
 'sessiondayofweek': 'session_day_of_week',
 'sessionendtime': 'session_end_time',
 'sessionstarttime': 'session_start_time',
 'sessiontitle': 'session_title',
 'sessiontype': 'session_type',
 'sessionweeknumber': 'session_week_number',
 'speedintensity': 'speed_intensity',
 'speedintensityzone1absolute': 'speed_intensity_zone_1_absolute',
 'speedintensityzone2absolute': 'speed_intensity_zone_2_absolute',
 'speedintensityzone3absolute': 'speed_intensity_zone_3_absolute',
 'speedintensityzone4absolute': 'speed_intensity_zone_4_absolute',
 'speedintensityzone5absolute': 'speed_intensity_zone_5_absolute',
 'speedintensityzone6absolute': 'speed_intensity_zone_6_absolute',
 'stepbalance': 'step_balance',
 'totalleftsteps': 'total_left_steps',
 'totalmetabolicpower': 'total_metabolic_power',
 'totalrightsteps': 'total_right_steps',
 'sessionweekno': 'session_week_number',
 'distancetotal': 'total_distance',
 'distancez1abs': 'distance_zone_1_absolute',
 'distancez2abs': 'distance_zone_2_absolute',
 'distancez3abs': 'distance_zone_3_absolute',
 'distancez4abs': 'distance_zone_4_absolute',
 'distancez5abs': 'distance_zone_5_absolute',
 'distancez6abs': 'distance_zone_6_absolute',
 'highspeedrunningabs': 'high_speed_running_distance_absolute',
 'averagespeed': 'average_speed',
 'maxspeed': 'maximum_speed',
 'totalaccelloading': 'total_acceleration_loading',
 'accelerationdistancez4abs': 'accelerations_total_distance_zone_4_absolute',
 'accelimpulsetotal': 'acceleration_impulse',
 'maxacceleration': 'maximum_acceleration',
 'timesincelastaccel': 'average_time_since_last_acceleration',
 'totaldecelloading': 'total_deceleration_loading',
 'maxdeceleration': 'maximum_deceleration',
 'timesincelastdecel': 'average_time_since_last_deceleration',
 'dsl': 'dynamic_stress_load',
 'dslz1time': 'dynamic_stress_load_time_zone_1',
 'dslz2time': 'dynamic_stress_load_time_zone_2',
 'dslz3time': 'dynamic_stress_load_time_zone_3',
 'dslz4time': 'dynamic_stress_load_time_zone_4',
 'dslz5time': 'dynamic_stress_load_time_zone_5',
 'dslz6time': 'dynamic_stress_load_time_zone_6',
 'dslz1': 'dynamic_stress_load_zone_1',
 'dslz2': 'dynamic_stress_load_zone_2',
 'dslz3': 'dynamic_stress_load_zone_3',
 'dslz4': 'dynamic_stress_load_zone_4',
 'dslz5': 'dynamic_stress_load_zone_5',
 'dslz6': 'dynamic_stress_load_zone_6',
 'dynamicloadant': 'dynamic_load_anterior',
 'dynamicloadlat': 'dynamic_load_lateral',
 'dynamicloadvert': 'dynamic_load_vertical',
 'entriesz3abs': 'entries_zone_3_absolute',
 'entriesz4abs': 'entries_zone_4_absolute',
 'entriesz5abs': 'entries_zone_5_absolute',
 'entriesz6abs': 'entries_zone_6_absolute',
 'hmld': 'high_metabolic_load_distance',
 'effortsnumber': 'high_metabolic_load_efforts',
 'effortsmaxspeed': 'high_metabolic_load_efforts_maximum_speed',
 'effortsdistance': 'high_metabolic_load_efforts_total_distance',
 'leftantpostimpact': 'left_anterior_posterior_impact',
 'leftaveragevertimpact': 'left_average_vertical_impact',
 'rightantpostimpact': 'right_anterior_posterior_impact',
 'rightaveragevertimpact': 'right_average_vertical_impact',
 'metabolictimez1abs': 'metabolic_time_zone_1_absolute',
 'metabolictimez2abs': 'metabolic_time_zone_2_absolute',
 'speedintensityz1abs': 'speed_intensity_zone_1_absolute',
 'speedintensityz2abs': 'speed_intensity_zone_2_absolute',
 'speedintensityz3abs': 'speed_intensity_zone_3_absolute',
 'speedintensityz4abs': 'speed_intensity_zone_4_absolute',
 'speedintensityz5abs': 'speed_intensity_zone_5_absolute',
 'speedintensityz6abs': 'speed_intensity_zone_6_absolute',
 'sprints': 'number_of_sprints',
 'hibsduration': 'duration_of_high_intensity_bursts',
 'hibsmaxspeed': 'high_intensity_bursts_maximum_speed',
 'timesincelastsprint': 'average_time_since_last_sprint',
 'dynamicloadmag': 'dynamic_load_magnitude',
 'energyexpenditure': 'energy_expenditure_kilocalories',
 'emd': 'equivalent_metabolic_distance',
 'explosivedistanceabs': 'explosive_distance',
 'hibsdistance': 'high_intensity_bursts_total_distance',
 'dynamicloadslow': 'lower_speed_loading',
 'avgheartrate': 'average_heart_rate',
 'minimumheartrate': 'minimum_heart_rate',
 'hrexertion': 'heart_rate_exertion',
 'hrload': 'heart_rate_load',
 'hrrpercent': 'heart_rate_recovery_percentage',
 'hrrbeats': 'heart_rate_recovery_beats',
 'hrv': 'heart_rate_variability',
 'percenttimeinredzonerel': 'percentage_time_in_red_zone_relative',
 'divepower': 'dive_power',
 'timesincelastdive': 'average_time_since_last_dive',
 'timesincelasthib': 'average_time_since_last_high_intensity_burst',
 'timesincelasthmleffort': 'average_time_since_last_high_metabolic_load_effort',
 'divesl': 'dives_left',
 'divesr': 'dives_right',
 'diveload': 'dive_load',
 'primarylabel': 'primary_label',
 'secondarylabel': 'secondary_label',
 'tertiarylabel': 'tertiary_label',
 'freetext': 'free_text',
 'accelerationsabs': 'accelerations_absolute',
 'accelerationdistancez3abs': 'accelerations_total_distance_zone_3_absolute',
 'accelerationdistancez5abs': 'accelerations_total_distance_zone_5_absolute',
 'accelerationdistancez6abs': 'accelerations_total_distance_zone_6_absolute',
 'accelerationsz3abs': 'accelerations_zone_3_absolute',
 'accelerationsz4abs': 'accelerations_zone_4_absolute',
 'accelerationsz5abs': 'accelerations_zone_5_absolute',
 'accelerationsz6abs': 'accelerations_zone_6_absolute',
 'decelerationsabs': 'decelerations_absolute',
 'decelerationdistancez3abs': 'decelerations_total_distance_zone_3_absolute',
 'decelerationdistancez4abs': 'decelerations_total_distance_zone_4_absolute',
 'decelerationdistancez5abs': 'decelerations_total_distance_zone_5_absolute',
 'decelerationdistancez6abs': 'decelerations_total_distance_zone_6_absolute',
 'decelerationsz3abs': 'decelerations_zone_3_absolute',
 'decelerationsz4abs': 'decelerations_zone_4_absolute',
 'decelerationsz5abs': 'decelerations_zone_5_absolute',
 'decelerationsz6abs': 'decelerations_zone_6_absolute',
 'impactsabs': 'impacts_absolute',
 'impactsz1abs': 'impacts_zone_1_absolute',
 'impactsz2abs': 'impacts_zone_2_absolute',
 'impactsz3abs': 'impacts_zone_3_absolute',
 'impactsz4abs': 'impacts_zone_4_absolute',
 'impactsz5abs': 'impacts_zone_5_absolute',
 'impactsz6abs': 'impacts_zone_6_absolute',
 'metabolicdistanceabs': 'metabolic_distance_absolute',
 'metabolicdistancez1abs': 'metabolic_distance_zone_1_absolute',
 'metabolicdistancez2abs': 'metabolic_distance_zone_2_absolute',
 'metabolicdistancez3abs': 'metabolic_distance_zone_3_absolute',
 'metabolicdistancez4abs': 'metabolic_distance_zone_4_absolute',
 'metabolicdistancez5abs': 'metabolic_distance_zone_5_absolute',
 'metabolicdistancez6abs': 'metabolic_distance_zone_6_absolute',
 'metabolictimeabs': 'metabolic_time_absolute',
 'metabolictimez3abs': 'metabolic_time_zone_3_absolute',
 'metabolictimez4abs': 'metabolic_time_zone_4_absolute',
 'metabolictimez5abs': 'metabolic_time_zone_5_absolute',
 'metabolictimez6abs': 'metabolic_time_zone_6_absolute',
 'sprintdistance': 'sprint_distance',
 'mechanicalload': 'mechanical_load',
 'percenttimeinredzoneabs': 'percentage_time_in_red_zone_absolute',
 'timeheartratez1abs': 'time_in_heart_rate_zone_1_absolute',
 'timeheartratez2abs': 'time_in_heart_rate_zone_2_absolute',
 'timeheartratez3abs': 'time_in_heart_rate_zone_3_absolute',
 'timeheartratez4abs': 'time_in_heart_rate_zone_4_absolute',
 'timeheartratez5abs': 'time_in_heart_rate_zone_5_absolute',
 'timeheartratez6abs': 'time_in_heart_rate_zone_6_absolute',
 'timeinredzoneabs': 'time_in_red_zone_absolute',
 'totaldistancezone5': 'total_distance_zone_5',
 'highspeedrunningdistanceabsolute': 'high_speed_running_distance_absolute',
 'percentagetimeinredzoneabsolute': 'percentage_time_in_red_zone_absolute',
 'heartrateload': 'heart_rate_load',
 'accelerationszone4absolute': 'accelerations_zone_4_absolute',
 'heartratezone5': 'heart_rate_zone_5',
 'playerloadtwodimensional': 'player_load_two_dimensional',
 'equivalentdistanceindexpercentage': 'equivalent_distance_index_percentage',
 'highdecelerations': 'high_decelerations',
 'decelerationstotaldistancezone6absolute': 'decelerations_total_distance_zone_6_absolute',
 'metabolicdistancezone1absolute': 'metabolic_distance_zone_1_absolute',
 'extrametrics': 'extra_metrics',
 'totaldistancezone2': 'total_distance_zone_2',
 'metabolicdistanceabsolute': 'metabolic_distance_absolute',
 'highmetabolicloadeffortsmaximumspeed': 'high_metabolic_load_efforts_maximum_speed',
 'totaldistancezone3': 'total_distance_zone_3',
 'accelerationstotaldistancezone3absolute': 'accelerations_total_distance_zone_3_absolute',
 'totaldecelerations': 'total_decelerations',
 'rightanteriorposteriorimpact': 'right_anterior_posterior_impact',
 'timeinheartratezone6absolute': 'time_in_heart_rate_zone_6_absolute',
 'maximumheartrate': 'maximum_heart_rate',
 'metabolicdistancezone6absolute': 'metabolic_distance_zone_6_absolute',
 'duration': 'duration',
 'accelerationstotaldistancezone5absolute': 'accelerations_total_distance_zone_5_absolute',
 'impactszone6absolute': 'impacts_zone_6_absolute',
 'accelerationimpulse': 'acceleration_impulse',
 'averagetimesincelastacceleration': 'average_time_since_last_acceleration',
 'playerid': 'player_id',
 'numberofsprints': 'number_of_sprints',
 'metabolictimeabsolute': 'metabolic_time_absolute',
 'highmetabolicloadefforts': 'high_metabolic_load_efforts',
 'metabolictimezone3absolute': 'metabolic_time_zone_3_absolute',
 'heartratezone1': 'heart_rate_zone_1',
 'timeinheartratezone1absolute': 'time_in_heart_rate_zone_1_absolute',
 'event': 'event',
 'highaccelerations': 'high_accelerations',
 'datum': 'datum',
 'impactsabsolute': 'impacts_absolute',
 'heartratezone3': 'heart_rate_zone_3',
 'impactszone3absolute': 'impacts_zone_3_absolute',
 'highmetabolicloadeffortstotaldistance': 'high_metabolic_load_efforts_total_distance',
 'numberofhighsprints': 'number_of_high_sprints',
 'impactszone5absolute': 'impacts_zone_5_absolute',
 'accelerationszone6absolute': 'accelerations_zone_6_absolute',
 'impactszone2absolute': 'impacts_zone_2_absolute',
 'percentagetimeinredzonerelative': 'percentage_time_in_red_zone_relative',
 'decelerationstotaldistancezone4absolute': 'decelerations_total_distance_zone_4_absolute',
 'sourceplayername': 'source_player_name',
 'heartratetrainingimpulse': 'heart_rate_training_impulse',
 'gpsid': 'gps_id',
 'playerloadthreedimensional': 'player_load_three_dimensional',
 'decelerationstotaldistancezone3absolute': 'decelerations_total_distance_zone_3_absolute',
 'decelerationszone3absolute': 'decelerations_zone_3_absolute',
 'accelerationstotaldistancezone6absolute': 'accelerations_total_distance_zone_6_absolute',
 'decelerationsabsolute': 'decelerations_absolute',
 'impactszone4absolute': 'impacts_zone_4_absolute',
 'numberofrepeatedsprints': 'number_of_repeated_sprints',
 'metabolictimezone5absolute': 'metabolic_time_zone_5_absolute',
 'leftanteriorposteriorimpact': 'left_anterior_posterior_impact',
 'highmetabolicloaddistance': 'high_metabolic_load_distance',
 'decelerationszone6absolute': 'decelerations_zone_6_absolute',
 'decelerationszone5absolute': 'decelerations_zone_5_absolute',
 'metabolicdistancezone2absolute': 'metabolic_distance_zone_2_absolute',
 'averagetimesincelasthighintensityburst': 'average_time_since_last_high_intensity_burst',
 'totaldistancezone4': 'total_distance_zone_4',
 'metabolicdistancezone5absolute': 'metabolic_distance_zone_5_absolute',
 'metabolictimezone4absolute': 'metabolic_time_zone_4_absolute',
 'highmetabolicloadtime': 'high_metabolic_load_time',
 'dynamicloadmagnitude': 'dynamic_load_magnitude',
 'totaldecelerationloading': 'total_deceleration_loading',
 'metabolicdistancezone4absolute': 'metabolic_distance_zone_4_absolute',
 'totaldistancezone1': 'total_distance_zone_1',
 'averageheartrate': 'average_heart_rate',
 'totalaccelerations': 'total_accelerations',
 'rightmagnitudeimpact': 'right_magnitude_impact',
 'impactszone1absolute': 'impacts_zone_1_absolute',
 'matchid': 'match_id',
 'maximumspeed': 'maximum_speed',
 'heartratezone4': 'heart_rate_zone_4',
 'explosivedistance': 'explosive_distance',
 'heartraterecoverybeats': 'heart_rate_recovery_beats',
 'year': 'year',
 'totaldistance': 'total_distance',
 'lowerspeedloading': 'lower_speed_loading',
 'averagetimesincelastdeceleration': 'average_time_since_last_deceleration',
 'timeinheartratezone4absolute': 'time_in_heart_rate_zone_4_absolute',
 'metabolicdistancezone3absolute': 'metabolic_distance_zone_3_absolute',
 'leftmagnitudeimpact': 'left_magnitude_impact',
 'heartratezone2': 'heart_rate_zone_2',
 'accelerationsabsolute': 'accelerations_absolute',
 'energyexpenditurekilocalories': 'energy_expenditure_kilocalories',
 'timeinheartratezone2absolute': 'time_in_heart_rate_zone_2_absolute',
 'timeinredzoneabsolute': 'time_in_red_zone_absolute',
 'heartrateanaerobiczone': 'heart_rate_anaerobic_zone',
 'totaldistancezone1and2': 'total_distance_zone_1_and_2',
 'totaldistancezone6': 'total_distance_zone_6',
 'decelerationszone4absolute': 'decelerations_zone_4_absolute',
 'timeinheartratezone5absolute': 'time_in_heart_rate_zone_5_absolute',
 'week': 'week',
 'decelerationstotaldistancezone5absolute': 'decelerations_total_distance_zone_5_absolute',
 'accelerationszone5absolute': 'accelerations_zone_5_absolute',
 'accelerationszone3absolute': 'accelerations_zone_3_absolute',
 'totalaccelerationloading': 'total_acceleration_loading',
 'heartrateexertion': 'heart_rate_exertion',
 'type': 'type',
 'heartratevariability': 'heart_rate_variability',
 'maximumacceleration': 'maximum_acceleration',
 'maximumdeceleration': 'maximum_deceleration',
 'heartraterecoverypercentage': 'heart_rate_recovery_percentage',
 'averagetimesincelasthighmetabolicloadeffort': 'average_time_since_last_high_metabolic_load_effort',
 'timeinheartratezone3absolute': 'time_in_heart_rate_zone_3_absolute',
 'metabolictimezone6absolute': 'metabolic_time_zone_6_absolute',
 'totaltime': 'duration'}
CSV_SOURCE_HEADER_MAP = {'Accelerations Total Distance Zone 4 (Absolute)': 'accelerations_total_distance_zone_4_absolute',
 'Average Dive Impact': 'average_dive_impact',
 'Average Metabolic Power': 'average_metabolic_power',
 'Average Time Since Last Accel': 'average_time_since_last_acceleration',
 'Average Time Since Last Decel': 'average_time_since_last_deceleration',
 'Average Time Since Last Dive': 'average_time_since_last_dive',
 'Average Time Since Last HIB': 'average_time_since_last_high_intensity_burst',
 'Average Time Since Last HML Effort': 'average_time_since_last_high_metabolic_load_effort',
 'Average Time Since Last Sprint': 'average_time_since_last_sprint',
 'Distance Zone 1 (Absolute)': 'distance_zone_1_absolute',
 'Distance Zone 2 (Absolute)': 'distance_zone_2_absolute',
 'Distance Zone 3 (Absolute)': 'distance_zone_3_absolute',
 'Distance Zone 4 (Absolute)': 'distance_zone_4_absolute',
 'Distance Zone 5 (Absolute)': 'distance_zone_5_absolute',
 'Distance Zone 6 (Absolute)': 'distance_zone_6_absolute',
 'Dives': 'dives',
 'Dives Left': 'dives_left',
 'Dives Right': 'dives_right',
 'Drill Date': 'drill_date',
 'Drill End Time': 'drill_end_time',
 'Drill Start Time': 'drill_start_time',
 'Drill Title': 'drill_title',
 'Duration Of High Intensity Bursts': 'duration_of_high_intensity_bursts',
 'Dynamic Load Anterior': 'dynamic_load_anterior',
 'Dynamic Load Lateral': 'dynamic_load_lateral',
 'Dynamic Load Vertical': 'dynamic_load_vertical',
 'Dynamic Stress Load': 'dynamic_stress_load',
 'Dynamic Stress Load Time Zone 1': 'dynamic_stress_load_time_zone_1',
 'Dynamic Stress Load Time Zone 2': 'dynamic_stress_load_time_zone_2',
 'Dynamic Stress Load Time Zone 3': 'dynamic_stress_load_time_zone_3',
 'Dynamic Stress Load Time Zone 4': 'dynamic_stress_load_time_zone_4',
 'Dynamic Stress Load Time Zone 5': 'dynamic_stress_load_time_zone_5',
 'Dynamic Stress Load Time Zone 6': 'dynamic_stress_load_time_zone_6',
 'Dynamic Stress Load Zone 1': 'dynamic_stress_load_zone_1',
 'Dynamic Stress Load Zone 2': 'dynamic_stress_load_zone_2',
 'Dynamic Stress Load Zone 3': 'dynamic_stress_load_zone_3',
 'Dynamic Stress Load Zone 4': 'dynamic_stress_load_zone_4',
 'Dynamic Stress Load Zone 5': 'dynamic_stress_load_zone_5',
 'Dynamic Stress Load Zone 6': 'dynamic_stress_load_zone_6',
 'EDI %': 'equivalent_distance_index_percentage',
 'Energy Expenditure (Kcal)': 'energy_expenditure_kilocalories',
 'Entries Zone 3 (Absolute)': 'entries_zone_3_absolute',
 'Entries Zone 4 (Absolute)': 'entries_zone_4_absolute',
 'Entries Zone 5 (Absolute)': 'entries_zone_5_absolute',
 'Entries Zone 6 (Absolute)': 'entries_zone_6_absolute',
 'Equivalent Metabolic Distance': 'equivalent_metabolic_distance',
 'External Work': 'external_work',
 'Fatigue Index': 'fatigue_index',
 'High Intensity Bursts Maximum Speed': 'high_intensity_bursts_maximum_speed',
 'High Intensity Bursts Total Distance': 'high_intensity_bursts_total_distance',
 'HML Efforts': 'high_metabolic_load_efforts',
 'HML Efforts Total Distance': 'high_metabolic_load_efforts_total_distance',
 'HML Time': 'high_metabolic_load_time',
 'Left Anterior Post Impact': 'left_anterior_posterior_impact',
 'Left Average Vertical Impact': 'left_average_vertical_impact',
 'Left Lateral Impact': 'left_lateral_impact',
 'Left Mag Impact': 'left_magnitude_impact',
 'Left Vertical Impact': 'left_vertical_impact',
 'Max Heart Rate': 'maximum_heart_rate',
 'Metabolic Time Zone 1 (Absolute)': 'metabolic_time_zone_1_absolute',
 'Metabolic Time Zone 2 (Absolute)': 'metabolic_time_zone_2_absolute',
 'Player Name': 'source_player_name',
 'Right Anterior Post Impact': 'right_anterior_posterior_impact',
 'Right Average Vertical Impact': 'right_average_vertical_impact',
 'Right Lateral Impact': 'right_lateral_impact',
 'Right Mag Impact': 'right_magnitude_impact',
 'Right Vertical Impact': 'right_vertical_impact',
 'Session Date': 'session_date',
 'Session Day of Week': 'session_day_of_week',
 'Session End Time': 'session_end_time',
 'Session Start Time': 'session_start_time',
 'Session Title': 'session_title',
 'Session Type': 'session_type',
 'Session Week Number': 'session_week_number',
 'Speed Intensity': 'speed_intensity',
 'Speed Intensity Zone 1 (Absolute)': 'speed_intensity_zone_1_absolute',
 'Speed Intensity Zone 2 (Absolute)': 'speed_intensity_zone_2_absolute',
 'Speed Intensity Zone 3 (Absolute)': 'speed_intensity_zone_3_absolute',
 'Speed Intensity Zone 4 (Absolute)': 'speed_intensity_zone_4_absolute',
 'Speed Intensity Zone 5 (Absolute)': 'speed_intensity_zone_5_absolute',
 'Speed Intensity Zone 6 (Absolute)': 'speed_intensity_zone_6_absolute',
 'Step Balance': 'step_balance',
 'Total Left Steps': 'total_left_steps',
 'Total Metabolic Power': 'total_metabolic_power',
 'Total Right Steps': 'total_right_steps',
 'sessionDate': 'session_date',
 'sessionWeekNo': 'session_week_number',
 'sessionDayOfWeek': 'session_day_of_week',
 'sessionTitle': 'session_title',
 'sessionType': 'session_type',
 'sessionStartTime': 'session_start_time',
 'sessionEndTime': 'session_end_time',
 'drillDate': 'drill_date',
 'playerName': 'source_player_name',
 'drillStartTime': 'drill_start_time',
 'drillEndTime': 'drill_end_time',
 'drillTitle': 'drill_title',
 'totalTime': 'duration',
 'distanceTotal': 'total_distance',
 'distanceZ1Abs': 'distance_zone_1_absolute',
 'distanceZ2Abs': 'distance_zone_2_absolute',
 'distanceZ3Abs': 'distance_zone_3_absolute',
 'distanceZ4Abs': 'distance_zone_4_absolute',
 'distanceZ5Abs': 'distance_zone_5_absolute',
 'distanceZ6Abs': 'distance_zone_6_absolute',
 'highSpeedRunningAbs': 'high_speed_running_distance_absolute',
 'averageSpeed': 'average_speed',
 'maxSpeed': 'maximum_speed',
 'totalAccelLoading': 'total_acceleration_loading',
 'accelerationDistanceZ4Abs': 'accelerations_total_distance_zone_4_absolute',
 'accelImpulseTotal': 'acceleration_impulse',
 'maxAcceleration': 'maximum_acceleration',
 'timeSinceLastAccel': 'average_time_since_last_acceleration',
 'totalDecelLoading': 'total_deceleration_loading',
 'maxDeceleration': 'maximum_deceleration',
 'timeSinceLastDecel': 'average_time_since_last_deceleration',
 'dsl': 'dynamic_stress_load',
 'dslZ1Time': 'dynamic_stress_load_time_zone_1',
 'dslZ2Time': 'dynamic_stress_load_time_zone_2',
 'dslZ3Time': 'dynamic_stress_load_time_zone_3',
 'dslZ4Time': 'dynamic_stress_load_time_zone_4',
 'dslZ5Time': 'dynamic_stress_load_time_zone_5',
 'dslZ6Time': 'dynamic_stress_load_time_zone_6',
 'dslZ1': 'dynamic_stress_load_zone_1',
 'dslZ2': 'dynamic_stress_load_zone_2',
 'dslZ3': 'dynamic_stress_load_zone_3',
 'dslZ4': 'dynamic_stress_load_zone_4',
 'dslZ5': 'dynamic_stress_load_zone_5',
 'dslZ6': 'dynamic_stress_load_zone_6',
 'dynamicLoadAnt': 'dynamic_load_anterior',
 'dynamicLoadLat': 'dynamic_load_lateral',
 'dynamicLoadVert': 'dynamic_load_vertical',
 'entriesZ3Abs': 'entries_zone_3_absolute',
 'entriesZ4Abs': 'entries_zone_4_absolute',
 'entriesZ5Abs': 'entries_zone_5_absolute',
 'entriesZ6Abs': 'entries_zone_6_absolute',
 'hmld': 'high_metabolic_load_distance',
 'effortsNumber': 'high_metabolic_load_efforts',
 'effortsMaxSpeed': 'high_metabolic_load_efforts_maximum_speed',
 'effortsDistance': 'high_metabolic_load_efforts_total_distance',
 'hmltime': 'high_metabolic_load_time',
 'leftAntPostImpact': 'left_anterior_posterior_impact',
 'leftAverageVertImpact': 'left_average_vertical_impact',
 'leftLateralImpact': 'left_lateral_impact',
 'leftMagImpact': 'left_magnitude_impact',
 'leftVerticalImpact': 'left_vertical_impact',
 'rightAntPostImpact': 'right_anterior_posterior_impact',
 'rightAverageVertImpact': 'right_average_vertical_impact',
 'rightLateralImpact': 'right_lateral_impact',
 'rightVerticalImpact': 'right_vertical_impact',
 'totalMetabolicPower': 'total_metabolic_power',
 'metabolicTimeZ1Abs': 'metabolic_time_zone_1_absolute',
 'metabolicTimeZ2Abs': 'metabolic_time_zone_2_absolute',
 'rightMagImpact': 'right_magnitude_impact',
 'speedIntensity': 'speed_intensity',
 'speedIntensityZ1Abs': 'speed_intensity_zone_1_absolute',
 'speedIntensityZ2Abs': 'speed_intensity_zone_2_absolute',
 'speedIntensityZ3Abs': 'speed_intensity_zone_3_absolute',
 'speedIntensityZ4Abs': 'speed_intensity_zone_4_absolute',
 'speedIntensityZ5Abs': 'speed_intensity_zone_5_absolute',
 'speedIntensityZ6Abs': 'speed_intensity_zone_6_absolute',
 'sprints': 'number_of_sprints',
 'hibsDuration': 'duration_of_high_intensity_bursts',
 'hibsMaxSpeed': 'high_intensity_bursts_maximum_speed',
 'timeSinceLastSprint': 'average_time_since_last_sprint',
 'stepBalance': 'step_balance',
 'totalLeftSteps': 'total_left_steps',
 'totalRightSteps': 'total_right_steps',
 'dynamicLoadMag': 'dynamic_load_magnitude',
 'edi': 'equivalent_distance_index_percentage',
 'energyExpenditure': 'energy_expenditure_kilocalories',
 'emd': 'equivalent_metabolic_distance',
 'explosiveDistanceAbs': 'explosive_distance',
 'hibsDistance': 'high_intensity_bursts_total_distance',
 'externalWork': 'external_work',
 'dynamicLoadSlow': 'lower_speed_loading',
 'fatigueIndex': 'fatigue_index',
 'avgHeartrate': 'average_heart_rate',
 'minimumHeartrate': 'minimum_heart_rate',
 'maxHeartrate': 'maximum_heart_rate',
 'hrexertion': 'heart_rate_exertion',
 'hrLoad': 'heart_rate_load',
 'hrrPercent': 'heart_rate_recovery_percentage',
 'hrrBeats': 'heart_rate_recovery_beats',
 'hrv': 'heart_rate_variability',
 'percentTimeInRedZoneRel': 'percentage_time_in_red_zone_relative',
 'averageDiveImpact': 'average_dive_impact',
 'divePower': 'dive_power',
 'averageMetabolicPower': 'average_metabolic_power',
 'timeSinceLastDive': 'average_time_since_last_dive',
 'timeSinceLastHib': 'average_time_since_last_high_intensity_burst',
 'timeSinceLastHmlEffort': 'average_time_since_last_high_metabolic_load_effort',
 'dives': 'dives',
 'divesL': 'dives_left',
 'divesR': 'dives_right',
 'diveLoad': 'dive_load',
 'Primary Label': 'primary_label',
 'Secondary Label': 'secondary_label',
 'Tertiary Label': 'tertiary_label',
 'Free Text': 'free_text',
 'accelerationsAbs': 'accelerations_absolute',
 'accelerationDistanceZ3Abs': 'accelerations_total_distance_zone_3_absolute',
 'accelerationDistanceZ5Abs': 'accelerations_total_distance_zone_5_absolute',
 'accelerationDistanceZ6Abs': 'accelerations_total_distance_zone_6_absolute',
 'accelerationsZ3Abs': 'accelerations_zone_3_absolute',
 'accelerationsZ4Abs': 'accelerations_zone_4_absolute',
 'accelerationsZ5Abs': 'accelerations_zone_5_absolute',
 'accelerationsZ6Abs': 'accelerations_zone_6_absolute',
 'decelerationsAbs': 'decelerations_absolute',
 'decelerationDistanceZ3Abs': 'decelerations_total_distance_zone_3_absolute',
 'decelerationDistanceZ4Abs': 'decelerations_total_distance_zone_4_absolute',
 'decelerationDistanceZ5Abs': 'decelerations_total_distance_zone_5_absolute',
 'decelerationDistanceZ6Abs': 'decelerations_total_distance_zone_6_absolute',
 'decelerationsZ3Abs': 'decelerations_zone_3_absolute',
 'decelerationsZ4Abs': 'decelerations_zone_4_absolute',
 'decelerationsZ5Abs': 'decelerations_zone_5_absolute',
 'decelerationsZ6Abs': 'decelerations_zone_6_absolute',
 'impactsAbs': 'impacts_absolute',
 'impactsZ1Abs': 'impacts_zone_1_absolute',
 'impactsZ2Abs': 'impacts_zone_2_absolute',
 'impactsZ3Abs': 'impacts_zone_3_absolute',
 'impactsZ4Abs': 'impacts_zone_4_absolute',
 'impactsZ5Abs': 'impacts_zone_5_absolute',
 'impactsZ6Abs': 'impacts_zone_6_absolute',
 'metabolicDistanceAbs': 'metabolic_distance_absolute',
 'metabolicDistanceZ1Abs': 'metabolic_distance_zone_1_absolute',
 'metabolicDistanceZ2Abs': 'metabolic_distance_zone_2_absolute',
 'metabolicDistanceZ3Abs': 'metabolic_distance_zone_3_absolute',
 'metabolicDistanceZ4Abs': 'metabolic_distance_zone_4_absolute',
 'metabolicDistanceZ5Abs': 'metabolic_distance_zone_5_absolute',
 'metabolicDistanceZ6Abs': 'metabolic_distance_zone_6_absolute',
 'metabolicTimeAbs': 'metabolic_time_absolute',
 'metabolicTimeZ3Abs': 'metabolic_time_zone_3_absolute',
 'metabolicTimeZ4Abs': 'metabolic_time_zone_4_absolute',
 'metabolicTimeZ5Abs': 'metabolic_time_zone_5_absolute',
 'metabolicTimeZ6Abs': 'metabolic_time_zone_6_absolute',
 'sprintDistance': 'sprint_distance',
 'mechanicalLoad': 'mechanical_load',
 'percentTimeInRedZoneAbs': 'percentage_time_in_red_zone_absolute',
 'timeHeartRateZ1Abs': 'time_in_heart_rate_zone_1_absolute',
 'timeHeartRateZ2Abs': 'time_in_heart_rate_zone_2_absolute',
 'timeHeartRateZ3Abs': 'time_in_heart_rate_zone_3_absolute',
 'timeHeartRateZ4Abs': 'time_in_heart_rate_zone_4_absolute',
 'timeHeartRateZ5Abs': 'time_in_heart_rate_zone_5_absolute',
 'timeHeartRateZ6Abs': 'time_in_heart_rate_zone_6_absolute',
 'timeInRedZoneAbs': 'time_in_red_zone_absolute'}
CSV_SOURCE_TEXT_COLUMNS = ['average_time_since_last_acceleration',
 'average_time_since_last_deceleration',
 'average_time_since_last_dive',
 'average_time_since_last_high_intensity_burst',
 'average_time_since_last_high_metabolic_load_effort',
 'average_time_since_last_sprint',
 'drill_date',
 'drill_end_time',
 'drill_start_time',
 'drill_title',
 'duration_of_high_intensity_bursts',
 'dynamic_stress_load_time_zone_1',
 'dynamic_stress_load_time_zone_2',
 'dynamic_stress_load_time_zone_3',
 'dynamic_stress_load_time_zone_4',
 'dynamic_stress_load_time_zone_5',
 'dynamic_stress_load_time_zone_6',
 'event',
 'free_text',
 'high_metabolic_load_time',
 'metabolic_time_absolute',
 'metabolic_time_zone_1_absolute',
 'metabolic_time_zone_2_absolute',
 'metabolic_time_zone_3_absolute',
 'metabolic_time_zone_4_absolute',
 'metabolic_time_zone_5_absolute',
 'metabolic_time_zone_6_absolute',
 'player_name',
 'primary_label',
 'secondary_label',
 'session_date',
 'session_day_of_week',
 'session_end_time',
 'session_start_time',
 'session_title',
 'session_type',
 'source_player_name',
 'tertiary_label',
 'time_in_heart_rate_zone_1_absolute',
 'time_in_heart_rate_zone_2_absolute',
 'time_in_heart_rate_zone_3_absolute',
 'time_in_heart_rate_zone_4_absolute',
 'time_in_heart_rate_zone_5_absolute',
 'time_in_heart_rate_zone_6_absolute',
 'time_in_red_zone_absolute',
 'type']
CSV_DIRECT_COLS = ['acceleration_impulse',
 'accelerations_absolute',
 'accelerations_total_distance_zone_3_absolute',
 'accelerations_total_distance_zone_4_absolute',
 'accelerations_total_distance_zone_5_absolute',
 'accelerations_total_distance_zone_6_absolute',
 'accelerations_zone_3_absolute',
 'accelerations_zone_4_absolute',
 'accelerations_zone_5_absolute',
 'accelerations_zone_6_absolute',
 'average_dive_impact',
 'average_heart_rate',
 'average_metabolic_power',
 'average_speed',
 'average_time_since_last_acceleration',
 'average_time_since_last_deceleration',
 'average_time_since_last_dive',
 'average_time_since_last_high_intensity_burst',
 'average_time_since_last_high_metabolic_load_effort',
 'average_time_since_last_sprint',
 'datum',
 'decelerations_absolute',
 'decelerations_total_distance_zone_3_absolute',
 'decelerations_total_distance_zone_4_absolute',
 'decelerations_total_distance_zone_5_absolute',
 'decelerations_total_distance_zone_6_absolute',
 'decelerations_zone_3_absolute',
 'decelerations_zone_4_absolute',
 'decelerations_zone_5_absolute',
 'decelerations_zone_6_absolute',
 'distance_zone_1_absolute',
 'distance_zone_2_absolute',
 'distance_zone_3_absolute',
 'distance_zone_4_absolute',
 'distance_zone_5_absolute',
 'distance_zone_6_absolute',
 'dive_load',
 'dive_power',
 'dives',
 'dives_left',
 'dives_right',
 'drill_date',
 'drill_end_time',
 'drill_start_time',
 'drill_title',
 'duration',
 'duration_of_high_intensity_bursts',
 'dynamic_load_anterior',
 'dynamic_load_lateral',
 'dynamic_load_magnitude',
 'dynamic_load_vertical',
 'dynamic_stress_load',
 'dynamic_stress_load_time_zone_1',
 'dynamic_stress_load_time_zone_2',
 'dynamic_stress_load_time_zone_3',
 'dynamic_stress_load_time_zone_4',
 'dynamic_stress_load_time_zone_5',
 'dynamic_stress_load_time_zone_6',
 'dynamic_stress_load_zone_1',
 'dynamic_stress_load_zone_2',
 'dynamic_stress_load_zone_3',
 'dynamic_stress_load_zone_4',
 'dynamic_stress_load_zone_5',
 'dynamic_stress_load_zone_6',
 'energy_expenditure_kilocalories',
 'entries_zone_3_absolute',
 'entries_zone_4_absolute',
 'entries_zone_5_absolute',
 'entries_zone_6_absolute',
 'equivalent_distance_index_percentage',
 'equivalent_metabolic_distance',
 'event',
 'explosive_distance',
 'external_work',
 'extra_metrics',
 'fatigue_index',
 'free_text',
 'gps_id',
 'heart_rate_anaerobic_zone',
 'heart_rate_exertion',
 'heart_rate_load',
 'heart_rate_recovery_beats',
 'heart_rate_recovery_percentage',
 'heart_rate_training_impulse',
 'heart_rate_variability',
 'heart_rate_zone_1',
 'heart_rate_zone_2',
 'heart_rate_zone_3',
 'heart_rate_zone_4',
 'heart_rate_zone_5',
 'high_accelerations',
 'high_decelerations',
 'high_intensity_bursts_maximum_speed',
 'high_intensity_bursts_total_distance',
 'high_metabolic_load_distance',
 'high_metabolic_load_efforts',
 'high_metabolic_load_efforts_maximum_speed',
 'high_metabolic_load_efforts_total_distance',
 'high_metabolic_load_time',
 'high_speed_running_distance_absolute',
 'impacts_absolute',
 'impacts_zone_1_absolute',
 'impacts_zone_2_absolute',
 'impacts_zone_3_absolute',
 'impacts_zone_4_absolute',
 'impacts_zone_5_absolute',
 'impacts_zone_6_absolute',
 'left_anterior_posterior_impact',
 'left_average_vertical_impact',
 'left_lateral_impact',
 'left_magnitude_impact',
 'left_vertical_impact',
 'lower_speed_loading',
 'match_id',
 'maximum_acceleration',
 'maximum_deceleration',
 'maximum_heart_rate',
 'maximum_speed',
 'mechanical_load',
 'metabolic_distance_absolute',
 'metabolic_distance_zone_1_absolute',
 'metabolic_distance_zone_2_absolute',
 'metabolic_distance_zone_3_absolute',
 'metabolic_distance_zone_4_absolute',
 'metabolic_distance_zone_5_absolute',
 'metabolic_distance_zone_6_absolute',
 'metabolic_time_absolute',
 'metabolic_time_zone_1_absolute',
 'metabolic_time_zone_2_absolute',
 'metabolic_time_zone_3_absolute',
 'metabolic_time_zone_4_absolute',
 'metabolic_time_zone_5_absolute',
 'metabolic_time_zone_6_absolute',
 'minimum_heart_rate',
 'number_of_high_sprints',
 'number_of_repeated_sprints',
 'number_of_sprints',
 'percentage_time_in_red_zone_absolute',
 'percentage_time_in_red_zone_relative',
 'player_id',
 'player_load_three_dimensional',
 'player_load_two_dimensional',
 'player_name',
 'primary_label',
 'right_anterior_posterior_impact',
 'right_average_vertical_impact',
 'right_lateral_impact',
 'right_magnitude_impact',
 'right_vertical_impact',
 'secondary_label',
 'session_date',
 'session_day_of_week',
 'session_end_time',
 'session_start_time',
 'session_title',
 'session_type',
 'session_week_number',
 'source_player_name',
 'speed_intensity',
 'speed_intensity_zone_1_absolute',
 'speed_intensity_zone_2_absolute',
 'speed_intensity_zone_3_absolute',
 'speed_intensity_zone_4_absolute',
 'speed_intensity_zone_5_absolute',
 'speed_intensity_zone_6_absolute',
 'sprint_distance',
 'step_balance',
 'tertiary_label',
 'time_in_heart_rate_zone_1_absolute',
 'time_in_heart_rate_zone_2_absolute',
 'time_in_heart_rate_zone_3_absolute',
 'time_in_heart_rate_zone_4_absolute',
 'time_in_heart_rate_zone_5_absolute',
 'time_in_heart_rate_zone_6_absolute',
 'time_in_red_zone_absolute',
 'total_acceleration_loading',
 'total_accelerations',
 'total_deceleration_loading',
 'total_decelerations',
 'total_distance',
 'total_distance_zone_1',
 'total_distance_zone_1_and_2',
 'total_distance_zone_2',
 'total_distance_zone_3',
 'total_distance_zone_4',
 'total_distance_zone_5',
 'total_distance_zone_6',
 'total_left_steps',
 'total_metabolic_power',
 'total_right_steps',
 'type',
 'week',
 'year']

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

INT_DB_COLS = ['number_of_repeated_sprints', 'high_accelerations', 'heart_rate_recovery_beats', 'number_of_sprints', 'total_decelerations', 'total_accelerations', 'steps', 'number_of_high_sprints', 'high_decelerations']


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
        # The broad exports use full labels, while the compact Statsports
        # export uses distanceZ1Abs/distanceZ1Rel (and so on).
        absolute = by_key.get(f"distancezone{zone}absolute") or by_key.get(f"distancez{zone}abs")
        relative = by_key.get(f"distancezone{zone}relative") or by_key.get(f"distancez{zone}rel")
        if absolute is not None:
            result[zone] = absolute
    return result


def _apply_speed_zone_mapping(base: dict, source_row, source_columns: dict[int, str]) -> None:
    if not all(zone in source_columns for zone in range(1, 6)):
        return

    if 6 in source_columns:
        # STATSports exposes all six speed zones. Keep both low-speed zones and
        # their combined total so they can be compared with Johan's total_distance_zone_1_and_2 field.
        groups = {
            "total_distance_zone_1": (1,),
            "total_distance_zone_2": (2,),
            "total_distance_zone_1_and_2": (1, 2),
            "total_distance_zone_3": (3,),
            "total_distance_zone_4": (4,),
            "total_distance_zone_5": (5,),
            "total_distance_zone_6": (6,),
        }
    else:
        # Johan Sports has five zones; its first zone remains the combined
        # Zone 1-2 comparison value, followed by Zones 3 through 6.
        groups = {
            "total_distance_zone_1_and_2": (1,),
            "total_distance_zone_3": (2,),
            "total_distance_zone_4": (3,),
            "total_distance_zone_5": (4,),
            "total_distance_zone_6": (5,),
        }

    for target, zones in groups.items():
        values = [coerce_num(source_row[source_columns[zone]]) for zone in zones]
        values = [value for value in values if value is not None]
        if values:
            base[target] = sum(values)


def _duration_minutes(value) -> float | None:
    """Convert vendor hh:mm:ss durations to the dashboard's minute unit."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    match = re.fullmatch(r"(\d+):(\d{1,2}):(\d{1,2})", text)
    if not match:
        return coerce_num(value)
    hours, minutes, seconds = (int(part) for part in match.groups())
    return hours * 60 + minutes + seconds / 60


CSV_BLOCKED_KEYS = ['accelerationdistancez1rel', 'accelerationdistancez2rel', 'accelerationdistancez3rel', 'accelerationdistancez4rel', 'accelerationdistancez5rel', 'accelerationdistancez6rel', 'accelerationsperminrelative', 'accelerationsperminuterelative', 'accelerationsrel', 'accelerationsrelative', 'accelerationstotaldistancezone1relative', 'accelerationstotaldistancezone2relative', 'accelerationstotaldistancezone3relative', 'accelerationstotaldistancezone4relative', 'accelerationstotaldistancezone5relative', 'accelerationstotaldistancezone6relative', 'accelerationstotaltimezone1relative', 'accelerationstotaltimezone2relative', 'accelerationstotaltimezone3relative', 'accelerationstotaltimezone4relative', 'accelerationstotaltimezone5relative', 'accelerationstotaltimezone6relative', 'accelerationsz1rel', 'accelerationsz2rel', 'accelerationsz3rel', 'accelerationsz3z6rel', 'accelerationsz4rel', 'accelerationsz4z6rel', 'accelerationsz5rel', 'accelerationsz5z6rel', 'accelerationsz6rel', 'accelerationszone1relative', 'accelerationszone2relative', 'accelerationszone3relative', 'accelerationszone3zone6relative', 'accelerationszone4relative', 'accelerationszone4zone6relative', 'accelerationszone5relative', 'accelerationszone5zone6relative', 'accelerationszone6relative', 'accelerationtimez1rel', 'accelerationtimez2rel', 'accelerationtimez3rel', 'accelerationtimez4rel', 'accelerationtimez5rel', 'accelerationtimez6rel', 'accelsperminrel', 'acute', 'acutechronicratio', 'acutechronicworkloadratio', 'acuteload', 'averagegkpower', 'averagegoalkeeperpower', 'ballinplaypercentage', 'ballinplaytime', 'ballinplaytimepercentage', 'chronic', 'chronicload', 'decelerationdistancez1rel', 'decelerationdistancez2rel', 'decelerationdistancez3rel', 'decelerationdistancez4rel', 'decelerationdistancez5rel', 'decelerationdistancez6rel', 'decelerationsperminrelative', 'decelerationsperminuterelative', 'decelerationsrel', 'decelerationsrelative', 'decelerationstotaldistancezone1relative', 'decelerationstotaldistancezone2relative', 'decelerationstotaldistancezone3relative', 'decelerationstotaldistancezone4relative', 'decelerationstotaldistancezone5relative', 'decelerationstotaldistancezone6relative', 'decelerationstotaltimezone1relative', 'decelerationstotaltimezone2relative', 'decelerationstotaltimezone3relative', 'decelerationstotaltimezone4relative', 'decelerationstotaltimezone5relative', 'decelerationstotaltimezone6relative', 'decelerationsz1rel', 'decelerationsz2rel', 'decelerationsz3rel', 'decelerationsz3z6rel', 'decelerationsz4rel', 'decelerationsz4z6rel', 'decelerationsz5rel', 'decelerationsz5z6rel', 'decelerationsz6rel', 'decelerationszone1relative', 'decelerationszone2relative', 'decelerationszone3relative', 'decelerationszone3zone6relative', 'decelerationszone4relative', 'decelerationszone4zone6relative', 'decelerationszone5relative', 'decelerationszone5zone6relative', 'decelerationszone6relative', 'decelerationtimez1rel', 'decelerationtimez2rel', 'decelerationtimez3rel', 'decelerationtimez4rel', 'decelerationtimez5rel', 'decelerationtimez6rel', 'decelsperminrel', 'distancepermin', 'distanceperminute', 'distancez1rel', 'distancez2rel', 'distancez2z6abs', 'distancez2z6rel', 'distancez3rel', 'distancez3z6abs', 'distancez3z6rel', 'distancez4rel', 'distancez4z6abs', 'distancez4z6rel', 'distancez5rel', 'distancez6rel', 'distancezone1relative', 'distancezone2relative', 'distancezone2zone6absolute', 'distancezone2zone6relative', 'distancezone3relative', 'distancezone3zone6absolute', 'distancezone3zone6relative', 'distancezone4relative', 'distancezone4zone6absolute', 'distancezone4zone6relative', 'distancezone5relative', 'distancezone6relative', 'dslz3z6', 'dslz4z6', 'dslz5z6', 'dynamicstressloadzone3zone6', 'dynamicstressloadzone4zone6', 'dynamicstressloadzone5zone6', 'entriesz3rel', 'entriesz4rel', 'entriesz5rel', 'entriesz6rel', 'entrieszone3relative', 'entrieszone4relative', 'entrieszone5relative', 'entrieszone6relative', 'explosivedistancerel', 'explosivedistancerelative', 'gkload', 'goalkeeperload', 'hibsnumber', 'highmetabolicloaddistanceperminute', 'highspeedrunningdistancerelative', 'highspeedrunningperminuteabsolute', 'highspeedrunningperminuterelative', 'highspeedrunningrel', 'hmldpermin', 'hmldperminute', 'hsrabspermin', 'hsrperminuteabsolute', 'hsrperminuterelative', 'hsrrelpermin', 'impactsrel', 'impactsrelative', 'impactsz1rel', 'impactsz2rel', 'impactsz3rel', 'impactsz4rel', 'impactsz5rel', 'impactsz6rel', 'impactszone1relative', 'impactszone2relative', 'impactszone3relative', 'impactszone3zone6relative', 'impactszone4relative', 'impactszone4zone6relative', 'impactszone5relative', 'impactszone5zone6relative', 'impactszone6relative', 'impactz3z6rel', 'impactz4z6rel', 'impactz5z6rel', 'insertedat', 'metabolicdistancerel', 'metabolicdistancerelative', 'metabolicdistancez1rel', 'metabolicdistancez2rel', 'metabolicdistancez3rel', 'metabolicdistancez4rel', 'metabolicdistancez5rel', 'metabolicdistancez6rel', 'metabolicdistancezone1relative', 'metabolicdistancezone2relative', 'metabolicdistancezone3relative', 'metabolicdistancezone4relative', 'metabolicdistancezone5relative', 'metabolicdistancezone6relative', 'metabolictimerel', 'metabolictimerelative', 'metabolictimez1rel', 'metabolictimez2rel', 'metabolictimez3rel', 'metabolictimez4rel', 'metabolictimez5rel', 'metabolictimez6rel', 'metabolictimezone1relative', 'metabolictimezone2relative', 'metabolictimezone3relative', 'metabolictimezone4relative', 'metabolictimezone5relative', 'metabolictimezone6relative', 'noofsatellites', 'numberofhighintensitybursts', 'numberofsatellites', 'playercustomid', 'playerdateofbirth', 'playerdisplayname', 'playerfirstname', 'playerheight', 'playerlastname', 'playermaxaccel', 'playermaxdecel', 'playermaxheartrate', 'playermaximumacceleration', 'playermaximumdeceleration', 'playermaximumheartrate', 'playermaximumspeed', 'playermaxspeed', 'playerprimaryposition', 'playerrestingheartrate', 'playersecondaryposition', 'playersprintthreshold', 'playerweight', 'qualityofsignal', 'sourcefile', 'speedintensityz1rel', 'speedintensityz2rel', 'speedintensityz3rel', 'speedintensityz3z6abs', 'speedintensityz3z6rel', 'speedintensityz4rel', 'speedintensityz4z6abs', 'speedintensityz4z6rel', 'speedintensityz5rel', 'speedintensityz5z6abs', 'speedintensityz5z6rel', 'speedintensityz6rel', 'speedintensityzone1relative', 'speedintensityzone2relative', 'speedintensityzone3relative', 'speedintensityzone3zone6absolute', 'speedintensityzone3zone6relative', 'speedintensityzone4relative', 'speedintensityzone4zone6absolute', 'speedintensityzone4zone6relative', 'speedintensityzone5relative', 'speedintensityzone5zone6absolute', 'speedintensityzone5zone6relative', 'speedintensityzone6relative', 'sprintdistancerelative', 'steps', 'timeheartratez1rel', 'timeheartratez2rel', 'timeheartratez2z6rel', 'timeheartratez3rel', 'timeheartratez3z6rel', 'timeheartratez4rel', 'timeheartratez4z6rel', 'timeheartratez5rel', 'timeheartratez6rel', 'timeinheartratezone1relative', 'timeinheartratezone2relative', 'timeinheartratezone2zone6relative', 'timeinheartratezone3relative', 'timeinheartratezone3zone6relative', 'timeinheartratezone4relative', 'timeinheartratezone4zone6relative', 'timeinheartratezone5relative', 'timeinheartratezone6relative', 'timeinredzonerel', 'timeinredzonerelative', 'timez1abs', 'timez1rel', 'timez2abs', 'timez2rel', 'timez3abs', 'timez3rel', 'timez4abs', 'timez4rel', 'timez5abs', 'timez5rel', 'timez6abs', 'timez6rel', 'timezone1absolute', 'timezone1relative', 'timezone2absolute', 'timezone2relative', 'timezone3absolute', 'timezone3relative', 'timezone4absolute', 'timezone4relative', 'timezone5absolute', 'timezone5relative', 'timezone6absolute', 'timezone6relative', 'totalloading']


def df_to_db_rows(df: pd.DataFrame, source_file: str, name_to_id: dict) -> tuple[list[dict], list[str]]:
    rows = []
    unmapped = set()

    parsed_dates = pd.to_datetime(df["Datum"], dayfirst=True, errors="coerce")
    if parsed_dates.isna().any():
        bad = df.loc[parsed_dates.isna(), "Datum"].head(5).tolist()
        raise ValueError(f"Kon sommige Datum waarden niet parsen: {bad}")

    dates_iso = parsed_dates.dt.date.astype(str)
    is_statsports = bool(df.attrs.get("statsports")) or any(normalize_key(c) in {"sessiondate", "drilldate", "sessiontitle", "playername", "playerdisplayname"} for c in df.columns)
    speed_zone_columns = _speed_zone_source_columns(list(df.columns))
    total_time_column = _csv_column(df, ["Total Time", "totalTime"])

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

            "extra_metrics": {},
        }

        for c in df.columns:
            if c in ID_COLS_IN_PARSER or str(c).strip() in {"source_file", "inserted_at"}:
                continue

            key = normalize_key(c)
            if key in CSV_BLOCKED_KEYS:
                continue
            val = r[c]
            direct_col = CSV_SOURCE_HEADER_MAP.get(str(c).strip()) or CSV_SOURCE_COLUMN_MAP.get(key)
            if direct_col is None and str(c).strip() in CSV_DIRECT_COLS:
                direct_col = str(c).strip()
            if direct_col is not None:
                if direct_col == "duration":
                    base[direct_col] = _duration_minutes(val)
                elif direct_col in CSV_SOURCE_TEXT_COLUMNS:
                    base[direct_col] = _source_text_value(val)
                else:
                    base[direct_col] = coerce_num(val)
                continue

            if key not in METRIC_MAP:
                if is_statsports:
                    continue
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
        if base.get("duration") is None and total_time_column is not None:
            base["duration"] = _duration_minutes(r[total_time_column])
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
    # Each drill is its own event; Session Title is only a fallback when no
    # drill title is included in a vendor export.
    event_col = _csv_column(df, ["Drill Title", "Event", "Session Title"])
    session_col = _csv_column(df, ["Session Title"])
    session_type_col = _csv_column(df, ["Session Type"])
    drill_start_col = _csv_column(df, ["Drill Start Time", "Drill Start", "Start Time"])
    primary_label_col = _csv_column(df, ["Primary Label"])

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
        fallback_event_col = _csv_column(df, ["Session Title", "Event"])
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
    event_values = event_values.where(event_values.ne(""), "CSV import")
    # Statsport uses MD-1/MD-2/etc. for training days and MD (usually with
    # the opponent in the title) for the actual match. Match Day -N is also a
    # training context. The session title/type therefore takes precedence over
    # incomplete Primary Label values such as Match-Topups on MD-5.
    session_values = result[session_col].astype(str).str.strip() if session_col else pd.Series("", index=result.index)
    session_type_values = result[session_type_col].astype(str).str.strip() if session_type_col else pd.Series("", index=result.index)
    primary_values = result[primary_label_col].astype(str).str.strip().str.lower() if primary_label_col else pd.Series("", index=result.index)
    session_norm = session_values.str.lower()
    session_type_norm = session_type_values.str.lower()
    is_relative_training = session_norm.str.startswith("md-") | session_norm.str.startswith("md+") | session_type_norm.str.startswith("match day -")
    is_md_match = ((session_norm == "md") | (session_norm.str.startswith("md "))) & ~is_relative_training
    is_named_match = session_norm.str.contains(r"\bmatch\b", case=False, na=False) & ~is_relative_training
    is_explicit_match_session = is_md_match | is_named_match | (session_type_norm == "match day")
    is_match = is_explicit_match_session | ((primary_values == "match") & is_md_match)
    type_values = pd.Series("Practice", index=result.index).mask(is_match, "Match")

    drill_norm = event_values.map(normalize_key)
    start_values = result[drill_start_col].astype(str).str.strip() if drill_start_col else pd.Series("", index=result.index)
    session_keys = list(zip(player_values, parsed_dates.dt.strftime("%Y-%m-%d"), session_values, start_values))
    live_keys = {key for key, drill in zip(session_keys, drill_norm) if drill == "entiresessionlive"}

    keep_mask = pd.Series(True, index=result.index)
    for idx, (key, drill) in enumerate(zip(session_keys, drill_norm)):
        if drill == "entiresession" and key in live_keys:
            keep_mask.iloc[idx] = False
        elif drill in {"match", "matchentirematch"} and key in live_keys:
            keep_mask.iloc[idx] = False

    event_values = event_values.mask(drill_norm == "entiresessionlive", "Summary")
    event_values = event_values.mask(drill_norm == "entiresession", "Summary")
    event_values = event_values.mask(drill_norm == "matchentirematch", "Summary")
    event_values = event_values.mask(drill_norm == "match", "Summary")

    result.insert(4, "Type", type_values)
    result.insert(5, "Event", event_values)

    result = result[result["Speler"].ne("")].copy()
    result = result.loc[keep_mask.reindex(result.index, fill_value=True)].copy()
    result.attrs["statsports"] = True
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

    start_col = _csv_column(df, ["Drill Start Time", "Drill Start", "Start Time"])
    if start_col:
        df["_event_order"] = pd.to_datetime(df[start_col], dayfirst=True, errors="coerce")
    else:
        df["_event_order"] = pd.NaT
    df["_event_original_order"] = range(len(df))
    ordered = df.sort_values(keys + ["_event_order", "_event_original_order"], na_position="last")
    idx = ordered.groupby(keys).cumcount()
    grp_size = ordered.groupby(keys)["Event"].transform("size")
    mask = grp_size > 1
    ordered.loc[mask, "Event"] = ordered.loc[mask, "Event"] + " (" + (idx[mask] + 1).astype(str) + ")"
    df = ordered.sort_values("_event_original_order").drop(columns=["_event_order", "_event_original_order"])
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
