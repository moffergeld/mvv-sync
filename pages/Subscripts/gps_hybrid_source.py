"""Canonical GPS reader for the complete Streamlit dashboard.

Rows before 24 August 2026 come from Supabase. Rows on and after that date
come live from the STATSports API. Keeping this rule in one module prevents
individual dashboard pages from drifting back to a different data source.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date, timedelta
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st

from pages.Subscripts.statsports_api import (
    STATSPORTS_FIRST_DATE,
    fetch_statsports_range,
    sessions_to_dataframe,
    validate_api_key,
)


GPS_API_CUTOVER_DATE = STATSPORTS_FIRST_DATE
GPS_SUPABASE_LAST_DATE = GPS_API_CUTOVER_DATE - timedelta(days=1)


class HybridGpsDataError(RuntimeError):
    """Safe error shown when one of the mandatory GPS sources is unavailable."""


def get_statsports_api_key() -> str:
    """Read the server-side API ID without ever exposing it in the UI."""

    value = (
        st.secrets.get("STATSPORTS_API_KEY", "")
        or st.secrets.get("STATSPORTS_API_ID", "")
        or st.secrets.get("STATSport", "")
    )
    try:
        return validate_api_key(str(value or ""))
    except ValueError as exc:
        raise HybridGpsDataError(
            "STATSPORTS_API_KEY ontbreekt of is ongeldig; recente GPS-data kan niet worden geladen."
        ) from exc


def _as_date(value: date | str | pd.Timestamp | None) -> date | None:
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Ongeldige GPS-datum: {value}")
    return parsed.date()


def _supabase_config() -> tuple[str, str]:
    url = str(st.secrets.get("SUPABASE_URL", "") or "").strip().rstrip("/")
    anon_key = str(st.secrets.get("SUPABASE_ANON_KEY", "") or "").strip()
    if not url or not anon_key:
        raise HybridGpsDataError("Supabase-configuratie ontbreekt; historische GPS-data kan niet worden geladen.")
    return url, anon_key


def _rest_get_paged(access_token: str, table: str, query: str, page_size: int = 5000) -> pd.DataFrame:
    url, anon_key = _supabase_config()
    headers = {
        "apikey": anon_key,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Range-Unit": "items",
    }
    rows: list[dict] = []
    start = 0
    while True:
        response = requests.get(
            f"{url}/rest/v1/{table}?{query}",
            headers={**headers, "Range": f"{start}-{start + page_size - 1}"},
            timeout=120,
        )
        if not response.ok:
            raise HybridGpsDataError(
                f"Historische GPS-data kon niet uit Supabase worden geladen (HTTP {response.status_code})."
            )
        batch = response.json()
        if not isinstance(batch, list):
            raise HybridGpsDataError("Supabase gaf een ongeldig antwoord voor de historische GPS-data.")
        rows.extend(item for item in batch if isinstance(item, dict))
        if len(batch) < page_size:
            break
        start += page_size
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=300)
def _statsports_frame_cached(api_key: str, end_iso: str) -> pd.DataFrame:
    end_date = _as_date(end_iso)
    if end_date is None or end_date < GPS_API_CUTOVER_DATE:
        return pd.DataFrame()
    sessions = fetch_statsports_range(api_key, GPS_API_CUTOVER_DATE, end_date)
    return sessions_to_dataframe(sessions)


@st.cache_data(show_spinner=False, ttl=300)
def _player_name_map_cached(access_token: str) -> dict[str, str]:
    players = _rest_get_paged(
        access_token,
        "players",
        "select=player_id,full_name&order=full_name.asc",
    )
    if players.empty:
        return {}
    from pages.Subscripts.gps_import_common import normalize_name

    result: dict[str, str] = {}
    for _, row in players.iterrows():
        player_id = str(row.get("player_id") or "").strip()
        full_name = str(row.get("full_name") or "").strip()
        if player_id and full_name:
            result[normalize_name(full_name)] = player_id
    return result


@st.cache_data(show_spinner=False, ttl=300)
def _match_ids_by_date_cached(access_token: str) -> dict[str, int]:
    matches = _rest_get_paged(
        access_token,
        "matches",
        (
            "select=match_id,match_date"
            f"&match_date=gte.{GPS_API_CUTOVER_DATE.isoformat()}"
            "&order=match_date.asc,match_id.asc"
        ),
    )
    if matches.empty:
        return {}
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    matches["match_id"] = pd.to_numeric(matches["match_id"], errors="coerce")
    matches = matches.dropna(subset=["match_date", "match_id"])
    counts = matches.groupby("match_date")["match_id"].nunique()
    unique_dates = set(counts[counts == 1].index)
    return {
        str(row["match_date"]): int(row["match_id"])
        for _, row in matches.iterrows()
        if row["match_date"] in unique_dates
    }


def _synthetic_gps_id(row: pd.Series) -> int:
    def clean(value: object) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value)

    identity = "|".join(
        clean(row.get(key))
        for key in ("player_id", "player_name", "datum", "type", "event", "session_id", "drill_id")
    )
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:15]
    return -int(digest, 16)


def _loose_name_parts(value: str) -> list[str]:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.findall(r"[a-z0-9]+", text.lower())


def _extend_player_map_for_api_names(name_to_id: dict[str, str], api_names: pd.Series) -> dict[str, str]:
    """Resolve harmless encoding differences without guessing between players."""

    result = dict(name_to_id)
    candidate_parts = {key: _loose_name_parts(key) for key in result}
    for api_name in api_names.dropna().astype(str).unique():
        api_key = re.sub(r"\s+", " ", api_name.strip().lower())
        if api_key in result:
            continue
        parts = _loose_name_parts(api_name)
        if len(parts) < 2:
            continue
        matches = [
            key
            for key, stored_parts in candidate_parts.items()
            if len(stored_parts) >= 2
            and stored_parts[-1] == parts[-1]
            and stored_parts[0][:1] == parts[0][:1]
        ]
        if len(matches) == 1:
            result[api_key] = result[matches[0]]
    return result


def _statsports_db_frame(access_token: str, api_end: date) -> pd.DataFrame:
    api_frame = _statsports_frame_cached(get_statsports_api_key(), api_end.isoformat())
    if api_frame.empty:
        return pd.DataFrame()

    from pages.Subscripts.gps_import_common import df_to_db_rows

    name_to_id = _extend_player_map_for_api_names(
        _player_name_map_cached(access_token),
        api_frame["Speler"],
    )
    rows, _unmapped = df_to_db_rows(
        api_frame,
        source_file="STATSports API (live dashboard)",
        name_to_id=name_to_id,
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    match_ids = _match_ids_by_date_cached(access_token)
    match_mask = frame["type"].isin({"Match", "Practice Match"})
    frame.loc[match_mask, "match_id"] = frame.loc[match_mask, "datum"].map(match_ids)
    frame["gps_id"] = frame.apply(_synthetic_gps_id, axis=1)
    frame["data_source"] = "STATSports API"
    return frame


def _supabase_query(
    columns: list[str],
    start_date: date | None,
    end_date: date | None,
    *,
    event: str | None,
    session_type: str | None,
    player_id: str | None,
    player_name: str | None,
    match_id: int | str | None,
) -> str:
    parts = [f"select={','.join(dict.fromkeys(columns))}", f"datum=lt.{GPS_API_CUTOVER_DATE.isoformat()}"]
    if start_date is not None:
        parts.append(f"datum=gte.{start_date.isoformat()}")
    if end_date is not None:
        parts.append(f"datum=lte.{min(end_date, GPS_SUPABASE_LAST_DATE).isoformat()}")
    if event is not None:
        parts.append(f"event=eq.{quote(str(event), safe='')}")
    if session_type is not None:
        parts.append(f"type=eq.{quote(str(session_type), safe='')}")
    if player_id is not None:
        parts.append(f"player_id=eq.{quote(str(player_id), safe='')}")
    if player_name is not None:
        parts.append(f"player_name=eq.{quote(str(player_name), safe='')}")
    if match_id is not None:
        parts.append(f"match_id=eq.{quote(str(match_id), safe='')}")
    parts.append("order=datum.asc,gps_id.asc")
    return "&".join(parts)


def _filter_api_rows(
    frame: pd.DataFrame,
    start_date: date | None,
    end_date: date | None,
    *,
    event: str | None,
    session_type: str | None,
    player_id: str | None,
    player_name: str | None,
    match_id: int | str | None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    dates = pd.to_datetime(result["datum"], errors="coerce").dt.date
    if start_date is not None:
        result = result.loc[dates >= start_date].copy()
        dates = pd.to_datetime(result["datum"], errors="coerce").dt.date
    if end_date is not None:
        result = result.loc[dates <= end_date].copy()
    if event is not None:
        result = result.loc[result["event"].astype(str) == str(event)].copy()
    if session_type is not None:
        result = result.loc[result["type"].astype(str) == str(session_type)].copy()
    if player_id is not None:
        result = result.loc[result["player_id"].fillna("").astype(str) == str(player_id)].copy()
    if player_name is not None:
        result = result.loc[result["player_name"].fillna("").astype(str) == str(player_name)].copy()
    if match_id is not None:
        result = result.loc[pd.to_numeric(result["match_id"], errors="coerce") == int(match_id)].copy()
    return result


def load_hybrid_gps(
    access_token: str,
    columns: list[str] | tuple[str, ...],
    *,
    start: date | str | pd.Timestamp | None = None,
    end: date | str | pd.Timestamp | None = None,
    event: str | None = None,
    session_type: str | None = None,
    player_id: str | None = None,
    player_name: str | None = None,
    match_id: int | str | None = None,
    descending: bool = False,
) -> pd.DataFrame:
    """Return one canonical frame while enforcing the source boundary."""

    if not str(access_token or "").strip():
        raise HybridGpsDataError("Je sessie is verlopen; GPS-data kan niet worden geladen.")

    requested_columns = list(dict.fromkeys(str(column) for column in columns))
    required_columns = list(
        dict.fromkeys(requested_columns + ["gps_id", "player_id", "player_name", "datum", "type", "event", "match_id"])
    )
    start_date = _as_date(start)
    end_date = _as_date(end)
    if start_date and end_date and end_date < start_date:
        return pd.DataFrame(columns=requested_columns)

    old_frame = pd.DataFrame()
    if start_date is None or start_date < GPS_API_CUTOVER_DATE:
        old_frame = _rest_get_paged(
            access_token,
            "gps_records",
            _supabase_query(
                required_columns,
                start_date,
                end_date,
                event=event,
                session_type=session_type,
                player_id=player_id,
                player_name=player_name,
                match_id=match_id,
            ),
        )
        if not old_frame.empty:
            old_frame["data_source"] = "Supabase"

    api_frame = pd.DataFrame()
    effective_api_end = min(end_date or date.today(), date.today())
    if effective_api_end >= GPS_API_CUTOVER_DATE and (start_date is None or end_date is None or end_date >= GPS_API_CUTOVER_DATE):
        api_frame = _filter_api_rows(
            _statsports_db_frame(access_token, effective_api_end),
            max(start_date, GPS_API_CUTOVER_DATE) if start_date else GPS_API_CUTOVER_DATE,
            end_date,
            event=event,
            session_type=session_type,
            player_id=player_id,
            player_name=player_name,
            match_id=match_id,
        )

    combined = pd.concat([old_frame, api_frame], ignore_index=True, sort=False)
    if combined.empty:
        return pd.DataFrame(columns=requested_columns)

    combined["datum"] = pd.to_datetime(combined["datum"], errors="coerce")
    combined = combined.dropna(subset=["datum"])
    combined = combined.sort_values(["datum", "gps_id"], ascending=[not descending, not descending])
    for column in requested_columns:
        if column not in combined.columns:
            combined[column] = pd.NA
    result = combined[requested_columns].reset_index(drop=True)
    result.attrs["gps_source_boundary"] = GPS_API_CUTOVER_DATE.isoformat()
    result.attrs["supabase_rows"] = int(len(old_frame))
    result.attrs["statsports_rows"] = int(len(api_frame))
    return result
