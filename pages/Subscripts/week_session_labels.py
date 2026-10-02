from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence


_MD_PATTERN = re.compile(r"\b(?:MD|MATCH\s*DAY)\s*([+-]\s*\d+)?\b", re.IGNORECASE)


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _mapping_layers(value: object) -> list[Mapping]:
    if not isinstance(value, Mapping):
        return []
    layers: list[Mapping] = [value]
    for key in ("source_columns", "sessionDetails", "session", "STATSports Raw"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            layers.append(nested)
            details = nested.get("sessionDetails")
            if isinstance(details, Mapping):
                layers.append(details)
    return layers


def source_value(value: object, *keys: str) -> str:
    wanted = {_normalized_key(key) for key in keys}
    for layer in _mapping_layers(value):
        for key, item in layer.items():
            if _normalized_key(key) in wanted and item is not None and str(item).strip():
                return str(item).strip()
    return ""


def md_label(value: object, canonical_type: object = "", session_type: object = "") -> str:
    """Return the desktop-style MD notation for one GPS session."""

    candidates = [
        str(session_type or "").strip(),
        source_value(value, "Session Type", "sessionType"),
        source_value(value, "Session Title", "sessionTitle", "sessionName"),
    ]
    for candidate in candidates:
        match = _MD_PATTERN.search(candidate)
        if not match:
            continue
        offset = re.sub(r"\s+", "", match.group(1) or "")
        return f"MD{offset}"

    type_text = str(canonical_type or "").strip().lower()
    if "match" in type_text or "wedstrijd" in type_text:
        return "MD"
    return "MD onbekend"


def session_time(value: object, direct_value: object = "") -> str:
    raw = str(direct_value or "").strip() or source_value(
        value,
        "Session Start Time",
        "sessionStartTime",
        "Drill Start Time",
        "drillStartTime",
    )
    match = re.search(r"(?:^|[T ])(\d{1,2}):(\d{2})", raw)
    if not match:
        return ""
    return f"{int(match.group(1)):02d}:{match.group(2)}"


def is_goalkeeper_position(value: object) -> bool:
    """Recognise the keeper labels used by MVV and STATSports."""

    text = str(value or "").strip().lower()
    normalized = re.sub(r"[^a-z]", "", text)
    return bool(
        re.search(r"(?:^|[^a-z])gk(?:$|[^a-z])", text)
        or any(label in normalized for label in ("goalkeeper", "keeper", "doelman", "goalie"))
    )


def moment_axis(records: Sequence[Mapping]) -> list[dict[str, object]]:
    """Create one outer MD/date label with separate inner session moments."""

    keys = [(record.get("datum"), str(record.get("md_label") or "MD onbekend")) for record in records]
    counts = Counter(keys)
    seen: Counter = Counter()
    result: list[dict[str, object]] = []
    for record, key in zip(records, keys):
        seen[key] += 1
        date_value, md_value = key
        date_label = date_value.strftime("%d/%m") if hasattr(date_value, "strftime") else str(date_value)
        time_value = str(record.get("session_time") or "")
        event_group = str(record.get("event_group") or "Training")
        if counts[key] > 1:
            moment = f"Moment {seen[key]}"
            if time_value:
                moment += f" · {time_value}"
        else:
            moment = time_value or ("Wedstrijd" if event_group == "Match" else "Training")
        result.append(
            {
                "events_in_day": counts[key],
                "moment_index": seen[key],
                "group_label": f"{md_value} · {date_label}",
                "moment_label": moment,
            }
        )
    return result
