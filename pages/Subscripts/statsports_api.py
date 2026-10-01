"""STATSports Pro Series API client and payload normalisation.

This module intentionally has no Streamlit or Supabase dependency. The UI owns
authentication and writes the returned rows directly to Supabase.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests


STATSPORTS_API_BASE = "https://statsportsproseries.com/thirdpartyapi/api/ThirdPartyData/"
STATSPORTS_API_VERSION = "7"
STATSPORTS_FIRST_DATE = date(2026, 8, 24)
_API_KEY_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


class StatsportsApiError(RuntimeError):
    """A safe, user-facing STATSports API error."""


def validate_api_key(api_key: str) -> str:
    value = str(api_key or "").strip()
    if not _API_KEY_RE.fullmatch(value):
        raise ValueError("Vul een geldige STATSports API-ID in.")
    return value


def _date_chunks(start: date, end: date, chunk_days: int = 7) -> list[tuple[date, date]]:
    if end < start:
        raise ValueError("De einddatum mag niet voor de begindatum liggen.")
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def fetch_statsports_range(
    api_key: str,
    start: date,
    end: date,
    *,
    on_chunk: Callable[[int, int, date, date, list[dict]], None] | None = None,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch full sessions in API-friendly blocks of seven days."""

    key = validate_api_key(api_key)
    chunks = _date_chunks(start, end)
    http = session or requests.Session()
    result: list[dict] = []

    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        try:
            response = http.post(
                f"{STATSPORTS_API_BASE}getFullSessionsByDateRange",
                headers={"Content-Type": "application/json", "api-version": STATSPORTS_API_VERSION},
                json={
                    "thirdPartyApiId": key,
                    "sessionStartDate": f"{chunk_start.isoformat()}T00:00:00Z",
                    "sessionEndDate": f"{chunk_end.isoformat()}T23:59:59Z",
                },
                timeout=120,
            )
        except requests.Timeout as exc:
            raise StatsportsApiError(
                f"STATSports reageerde niet op tijd voor {chunk_start:%d-%m-%Y} t/m {chunk_end:%d-%m-%Y}."
            ) from exc
        except requests.RequestException as exc:
            raise StatsportsApiError("STATSports kon niet worden bereikt.") from exc

        if response.status_code == 204 or not response.text.strip():
            sessions: list[dict] = []
        elif not response.ok:
            raise StatsportsApiError(
                f"STATSports weigerde {chunk_start:%d-%m-%Y} t/m {chunk_end:%d-%m-%Y} "
                f"(HTTP {response.status_code})."
            )
        else:
            try:
                payload = response.json()
            except ValueError as exc:
                raise StatsportsApiError("STATSports gaf een ongeldig antwoord terug.") from exc
            sessions = payload if isinstance(payload, list) else [payload]
            sessions = [item for item in sessions if isinstance(item, dict)]

        result.extend(sessions)
        if on_chunk:
            on_chunk(index, len(chunks), chunk_start, chunk_end, sessions)

    return result


def _details(session: dict) -> dict:
    value = session.get("sessionDetails") or session.get("session") or {}
    return value if isinstance(value, dict) else {}


def _players(session: dict) -> list[dict]:
    value = session.get("sessionPlayers") or session.get("players") or []
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _player_details(item: dict) -> dict:
    value = item.get("playerDetails") or item.get("player") or {}
    return value if isinstance(value, dict) else {}


def _player_name(item: dict) -> str:
    player = _player_details(item)
    display_name = str(player.get("displayName") or "").strip()
    if display_name:
        return display_name
    return " ".join(str(player.get(key) or "").strip() for key in ("firstName", "lastName")).strip()


def _session_date(session: dict) -> str:
    details = _details(session)
    return str(details.get("sessionDate") or session.get("shareDate") or "")[:10]


def _session_type(session: dict) -> str:
    return str(_details(session).get("sessionType") or "").strip()


def _session_title(session: dict) -> str:
    return str(session.get("sessionName") or session.get("name") or _session_type(session)).strip()


def _has_match_part(item: dict) -> bool:
    drills = item.get("drills") or []
    return any(
        re.fullmatch(r"(?:first|second)\s*half\s*", str(drill.get("drillName") or drill.get("name") or ""), re.I)
        for drill in drills
        if isinstance(drill, dict)
    )


def dashboard_sessions(sessions: Iterable[dict]) -> list[dict]:
    """Match the desktop app's filtering and match-supplement behaviour."""

    source = [session for session in sessions if isinstance(session, dict)]
    labeled = [session for session in source if _session_type(session) and _session_type(session).lower() != "general"]
    supplements: list[dict] = []

    for session in source:
        players = _players(session)
        if _session_type(session).lower() != "general" or len(players) != 1 or not _has_match_part(players[0]):
            continue
        player_name = _player_name(players[0])
        match = next(
            (
                candidate
                for candidate in labeled
                if _session_date(candidate) == _session_date(session)
                and "match day" in _session_type(candidate).lower()
                and any(_player_name(item) == player_name for item in _players(candidate))
            ),
            None,
        )
        if not match:
            continue
        linked = dict(session)
        linked["sessionName"] = _session_title(match)
        linked["sessionDetails"] = {**_details(session), **{
            key: _details(match).get(key)
            for key in ("sessionDate", "startTime", "endTime", "sessionType")
            if _details(match).get(key) is not None
        }}
        supplements.append(linked)

    return sorted(labeled + supplements, key=lambda item: str(_details(item).get("startTime") or ""))


def _canonical_type(session_title: str, session_type: str) -> str:
    title = session_title.strip().lower()
    kind = session_type.strip().lower()
    relative_training = title.startswith(("md-", "md+")) or kind.startswith("match day -")
    match = (
        not relative_training
        and (
            title == "md"
            or title.startswith("md ")
            or bool(re.search(r"\bmatch\b", title))
            or kind == "match day"
        )
    )
    return "Match" if match else "Practice"


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _raw_context(session: dict, item: dict, drill: dict) -> dict:
    """Keep the complete API context without duplicating every player in a session."""

    session_data = {key: value for key, value in session.items() if key not in {"sessionPlayers", "players"}}
    player_data = {key: value for key, value in item.items() if key != "drills"}
    return {"session": session_data, "player": player_data, "drill": drill}


def _summary_priority(event_key: str) -> int | None:
    """Only the live whole-session drill is valid for dashboard load totals."""

    if event_key == "entiresessionlive":
        return 0
    return None


def sessions_to_dataframe(sessions: Iterable[dict]) -> pd.DataFrame:
    """Flatten STATSports sessions to the existing GPS import contract."""

    rows: list[dict] = []
    for session in dashboard_sessions(sessions):
        details = _details(session)
        session_title = _session_title(session)
        session_type = _session_type(session)
        canonical_type = _canonical_type(session_title, session_type)
        session_date = _session_date(session)

        for item in _players(session):
            player = _player_details(item)
            player_name = _player_name(item)
            for drill in item.get("drills") or []:
                if not isinstance(drill, dict):
                    continue
                kpi = drill.get("drillKpi") or {}
                metrics = dict(kpi) if isinstance(kpi, dict) else {}
                if _number(metrics.get("totalTime")):
                    metrics["totalTime"] = metrics["totalTime"] / 60
                if _number(metrics.get("maxSpeed")):
                    metrics["maxSpeed"] = metrics["maxSpeed"] * 3.6

                event = str(drill.get("drillName") or drill.get("name") or session_title or "STATSports API").strip()
                event_key = re.sub(r"[^a-z0-9]", "", event.lower())
                summary_priority = _summary_priority(event_key)
                if summary_priority is not None:
                    event = "Summary"

                rows.append({
                    "Speler": player_name,
                    "Datum": session_date,
                    "Type": canonical_type,
                    "Event": event,
                    "Session ID": session.get("id"),
                    "Session Title": session_title,
                    "Session Type": session_type,
                    "Session Start Time": details.get("startTime"),
                    "Session End Time": details.get("endTime"),
                    "Player ID (STATSports)": player.get("id") or player.get("playerId"),
                    "Player Position": player.get("primaryPosition"),
                    "Drill ID": drill.get("id") or drill.get("drillId"),
                    "Drill Start Time": drill.get("startTime"),
                    "Drill End Time": drill.get("endTime"),
                    "Primary Label": drill.get("primaryLabel"),
                    "Secondary Label": drill.get("secondaryLabel"),
                    "Tertiary Label": drill.get("tertiaryLabel"),
                    "STATSports Raw": _raw_context(session, item, drill),
                    "_summary_priority": summary_priority,
                    **metrics,
                })

    if not rows:
        return pd.DataFrame(columns=["Speler", "Datum", "Type", "Event"])

    frame = pd.DataFrame(rows)
    frame["Datum"] = pd.to_datetime(frame["Datum"], errors="coerce").dt.strftime("%d-%m-%Y")
    frame = frame[frame["Speler"].astype(str).str.strip().ne("") & frame["Datum"].notna()].copy()

    summary_mask = frame["Event"].eq("Summary")
    if summary_mask.any():
        summary_rows = frame.loc[summary_mask].sort_values(
            ["Speler", "Datum", "Type", "_summary_priority", "Drill Start Time"],
            na_position="last",
        )
        keep_summary_indices = summary_rows.drop_duplicates(
            subset=["Speler", "Datum", "Type", "Session ID", "Session Start Time", "Session Title"],
            keep="first",
        ).index
        frame = frame.loc[~summary_mask | frame.index.isin(keep_summary_indices)].copy()

    keys = ["Speler", "Datum", "Type", "Event"]
    order = pd.to_datetime(frame["Drill Start Time"], errors="coerce", utc=True)
    frame["_event_order"] = order
    frame["_original_order"] = range(len(frame))
    ordered = frame.sort_values(keys + ["_event_order", "_original_order"], na_position="last")
    duplicate_index = ordered.groupby(keys).cumcount()
    duplicate_count = ordered.groupby(keys)["Event"].transform("size")
    duplicate_mask = (duplicate_count > 1) & ordered["Event"].ne("Summary")
    ordered.loc[duplicate_mask, "Event"] = (
        ordered.loc[duplicate_mask, "Event"].astype(str)
        + " ("
        + (duplicate_index[duplicate_mask] + 1).astype(str)
        + ")"
    )
    frame = ordered.sort_values("_original_order").drop(
        columns=["_event_order", "_original_order", "_summary_priority"]
    )
    frame.attrs["statsports"] = True
    frame.attrs["preserve_extra_metrics"] = True
    frame.attrs["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return frame
