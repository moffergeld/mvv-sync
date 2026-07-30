from __future__ import annotations

from datetime import date, timedelta
import re
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from pages.Subscripts.mvv_branding import TEAM_HERO_BG, TEAM_LOGO, build_data_uri
from roles import (
    get_access_token,
    get_profile,
    get_sb,
    is_staff_user,
    render_sidebar_footer,
    render_sidebar_navigation,
    require_auth,
)
from utils.streamlit_ui import apply_streamlit_chrome


st.set_page_config(page_title="Benchmarks", layout="wide", initial_sidebar_state="expanded")
apply_streamlit_chrome()

PAGE_BG_URI = build_data_uri(TEAM_HERO_BG)
TEAM_LOGO_URI = build_data_uri(TEAM_LOGO)

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "").strip()
BENCH_TEMP_SUBPOSITION_KEY = "bench_temp_sub_positions"

MVV_RED = "#EA3351"
MVV_RED_DEEP = "#8F1230"
MVV_RED_SOFT = "#F3A0AF"
MVV_GOLD = "#E8B24D"
MVV_GREEN = "#2FB67A"
MVV_AMBER = "#F5A524"
MVV_TEXT = "#F8FAFC"
MVV_TEXT_SOFT = "rgba(248,250,252,0.72)"
MVV_GRID = "rgba(255,255,255,0.10)"

TABLE_COLUMNS = [
    "Positie",
    "Totale afstand (m)",
    "HI afstand >20.0 (m)",
    "Sprintafstand >25.0 (m)",
    "Runs count >15.0 (#)",
    "Sprint count >25.0 (#)",
    "Tot. dist. (m/min)",
    "Intensiteit (%)",
]

KKD_BENCHMARKS = pd.DataFrame(
    [
        ["AM", "12.561", "1.058", "218", "46,4", "10,6", "132", "8,4%"],
        ["CB", "10.913", "651", "149", "29,6", "7,3", "115", "6,0%"],
        ["CF", "11.720", "1.087", "278", "44,6", "13,2", "123", "9,3%"],
        ["CM", "12.478", "948", "179", "43,2", "9,0", "131", "7,6%"],
        ["DM", "12.282", "845", "158", "39,1", "7,9", "129", "6,9%"],
        ["GK", "5.973", "39", "6", "2,9", "0,3", "63", "0,7%"],
        ["LB", "11.309", "973", "273", "37,7", "12,2", "119", "8,6%"],
        ["LW", "11.602", "1.126", "318", "43,1", "14,1", "122", "9,7%"],
        ["RB", "11.488", "1.000", "283", "38,6", "12,4", "121", "8,7%"],
        ["RW", "11.659", "1.171", "345", "44,3", "15,1", "123", "10,0%"],
    ],
    columns=TABLE_COLUMNS,
)

EREDIVISIE_BENCHMARKS = pd.DataFrame(
    [
        ["AM", "12.139", "1.019", "202", "38,1", "9,0", "128", "8,4%"],
        ["CB", "10.579", "615", "138", "22,6", "5,9", "111", "5,8%"],
        ["CF", "11.193", "992", "235", "35,1", "10,0", "118", "8,9%"],
        ["CM", "11.700", "970", "215", "34,2", "9,1", "123", "8,3%"],
        ["DEF", "9.931", "635", "183", "19,3", "6,3", "105", "6,4%"],
        ["DM", "11.912", "882", "171", "33,7", "7,6", "125", "7,4%"],
        ["FOR", "10.931", "1.225", "352", "37,0", "14,8", "115", "11,2%"],
        ["GK", "5.436", "25", "3", "1,2", "0,1", "57", "0,5%"],
        ["LB", "11.029", "990", "277", "30,9", "10,9", "116", "9,0%"],
        ["LW", "11.415", "1.120", "308", "36,0", "11,9", "120", "9,8%"],
        ["MID", "11.201", "1.032", "263", "33,6", "9,2", "118", "9,2%"],
        ["RB", "11.203", "981", "274", "30,5", "10,7", "118", "8,8%"],
        ["RW", "11.463", "1.143", "309", "36,7", "12,2", "121", "10,0%"],
    ],
    columns=TABLE_COLUMNS,
)

COMPARISON_BENCHMARKS = pd.DataFrame(
    [
        ["AM", "-422", "-39", "-16", "-8,3", "-1,6", "-4", "0,0%"],
        ["CB", "-334", "-36", "-11", "-7,0", "-1,4", "-4", "-0,2%"],
        ["CF", "-527", "-95", "-43", "-9,5", "-3,2", "-5", "-0,4%"],
        ["CM", "-778", "+22", "+36", "-9,0", "+0,1", "-8", "+0,7%"],
        ["DM", "-370", "+37", "+13", "-5,4", "-0,3", "-4", "+0,5%"],
        ["GK", "-537", "-14", "-3", "-1,7", "-0,2", "-6", "-0,2%"],
        ["LB", "-280", "+17", "+4", "-6,8", "-1,3", "-3", "+0,4%"],
        ["LW", "-187", "-6", "-10", "-7,1", "-2,2", "-2", "+0,1%"],
        ["RB", "-285", "-19", "-9", "-8,1", "-1,7", "-3", "+0,1%"],
        ["RW", "-196", "-28", "-36", "-7,6", "-2,9", "-2", "0,0%"],
    ],
    columns=TABLE_COLUMNS,
)

MARKS_TABLES = [
    ("KKD", "Keuken Kampioen Divisie 2024/2025", "10 posities, per 90 minuten", KKD_BENCHMARKS),
    ("Eredivisie", "Dutch Eredivisie 2025/2026", "13 posities, per 90 minuten", EREDIVISIE_BENCHMARKS),
    ("Vergelijking", "Eredivisie minus KKD", "Alleen overlappende posities", COMPARISON_BENCHMARKS),
]

BENCHMARK_SOURCE_TABLES = {
    "KKD": KKD_BENCHMARKS,
    "Eredivisie": EREDIVISIE_BENCHMARKS,
}

COMPARE_POSITION_CODES = set(KKD_BENCHMARKS["Positie"].astype(str)).union(EREDIVISIE_BENCHMARKS["Positie"].astype(str))

PERIOD_OPTIONS = {
    "Laatste 4 weken": 4,
    "Laatste 6 weken": 6,
    "Laatste 8 weken": 8,
    "Laatste 12 weken": 12,
}

COMPARE_MATCH_SCOPE_OPTIONS = {
    "Laatste 3 wedstrijden": 3,
    "Laatste 5 wedstrijden": 5,
    "Laatste 8 wedstrijden": 8,
    "Alle wedstrijden": None,
}

MATCH_EVENT_SELECT_VARIANTS = [
    "gps_id,match_id,datum,player_id,player_name,type,event,duration,total_distance,sprint,high_sprint,number_of_sprints,playerload2d,total_accelerations,high_accelerations,total_decelerations,high_decelerations",
    "gps_id,match_id,datum,player_id,player_name,type,event,duration,total_distance,sprint,high_sprint,playerload2d,total_accelerations,high_accelerations,total_decelerations,high_decelerations",
    "gps_id,datum,player_id,player_name,type,event,duration,total_distance,sprint,high_sprint,number_of_sprints,playerload2d,total_accelerations,total_decelerations",
    "gps_id,datum,player_id,player_name,type,event,duration,total_distance,sprint,high_sprint,playerload2d,total_accelerations,total_decelerations",
]

MATCH_NUMERIC_COLS = [
    "duration",
    "total_distance",
    "sprint",
    "high_sprint",
    "number_of_sprints",
    "playerload2d",
    "total_accelerations",
    "high_accelerations",
    "total_decelerations",
    "high_decelerations",
]

METRIC_SPECS = {
    "total_distance_90": {
        "label": "Totale afstand /90",
        "benchmark_col": "Totale afstand (m)",
        "kind": "distance",
        "color": MVV_RED,
        "tolerance": 200.0,
    },
    "hsr_hsd_90": {
        "label": "HSR/HSD /90",
        "benchmark_col": "HI afstand >20.0 (m)",
        "kind": "distance",
        "color": "#FF6B7F",
        "tolerance": 40.0,
    },
    "sprint_distance_90": {
        "label": "Sprintafstand /90",
        "benchmark_col": "Sprintafstand >25.0 (m)",
        "kind": "distance",
        "color": MVV_GOLD,
        "tolerance": 20.0,
    },
    "sprint_count_90": {
        "label": "Sprint count /90",
        "benchmark_col": "Sprint count >25.0 (#)",
        "kind": "count",
        "color": "#F97316",
        "tolerance": 1.0,
    },
    "total_distance_per_min": {
        "label": "Totale afstand /min",
        "benchmark_col": "Tot. dist. (m/min)",
        "kind": "rate",
        "color": "#38BDF8",
        "tolerance": 2.0,
    },
    "intensity_pct": {
        "label": "Intensiteit",
        "benchmark_col": "Intensiteit (%)",
        "kind": "percent",
        "color": "#34D399",
        "tolerance": 0.4,
    },
}

PLAYER_DETAIL_METRIC_ORDER = [
    "total_distance_90",
    "hsr_hsd_90",
    "sprint_distance_90",
    "sprint_count_90",
    "total_distance_per_min",
    "intensity_pct",
]

GPS_SELECT_COLS = [
    "gps_id",
    "datum",
    "player_id",
    "player_name",
    "type",
    "event",
    "duration",
    "total_distance",
    "sprint",
    "high_sprint",
    "number_of_sprints",
]

NUMERIC_GPS_COLS = [
    "duration",
    "total_distance",
    "sprint",
    "high_sprint",
    "number_of_sprints",
]

POSITION_EXACT_CODES = {"AM", "CB", "CF", "CM", "DM", "GK", "LB", "LW", "RB", "RW"}
SUBPOSITION_OPTIONS = [""] + sorted(POSITION_EXACT_CODES)
POSITION_DISPLAY_ORDER = ["GK", "CB", "LB", "RB", "DM", "CM", "AM", "LW", "RW", "CF"]


def parse_metric_value(value: object) -> float:
    text = str(value).strip().replace("%", "")
    if not text:
        return 0.0
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    else:
        text = text.replace(".", "")
    try:
        return float(text)
    except ValueError:
        return 0.0


def _safe_divide_series(numerator: pd.Series, denominator: pd.Series, multiplier: float = 1.0) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce").replace(0, pd.NA)
    result = pd.to_numeric(numerator, errors="coerce") / denom
    return result.astype(float) * multiplier


def _format_int(value: object) -> str:
    if pd.isna(value):
        return "--"
    return f"{int(round(float(value))):,}".replace(",", ".")


def _format_decimal(value: object, decimals: int = 1) -> str:
    if pd.isna(value):
        return "--"
    formatted = f"{float(value):,.{decimals}f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def _sort_compare_positions(values: list[str]) -> list[str]:
    order_map = {code: index for index, code in enumerate(POSITION_DISPLAY_ORDER)}
    cleaned_values = [str(value).strip().upper() for value in values if str(value).strip()]
    return sorted(cleaned_values, key=lambda value: (order_map.get(value, 999), value))


def _count_exact_subpositions(values: pd.Series) -> int:
    cleaned = values.fillna("").astype(str).str.strip().str.upper()
    return int(cleaned.isin(POSITION_EXACT_CODES).sum())


def _format_distance(value: object) -> str:
    base = _format_int(value)
    return "--" if base == "--" else f"{base} m"


def _format_rate(value: object) -> str:
    base = _format_decimal(value, 1)
    return "--" if base == "--" else f"{base} m/min"


def _format_percent(value: object) -> str:
    base = _format_decimal(value, 1)
    return "--" if base == "--" else f"{base}%"


def _format_count(value: object) -> str:
    return _format_decimal(value, 1)


def _format_metric(metric_key: str, value: object) -> str:
    kind = METRIC_SPECS[metric_key]["kind"]
    if kind == "distance":
        return _format_distance(value)
    if kind == "rate":
        return _format_rate(value)
    if kind == "percent":
        return _format_percent(value)
    return _format_count(value)


def _format_gap(metric_key: str, value: object) -> str:
    if pd.isna(value):
        return "--"
    prefix = "+" if float(value) >= 0 else ""
    kind = METRIC_SPECS[metric_key]["kind"]
    if kind == "distance":
        return f"{prefix}{_format_distance(value)}"
    if kind == "rate":
        return f"{prefix}{_format_rate(value)}"
    if kind == "percent":
        return f"{prefix}{_format_percent(value)}"
    return f"{prefix}{_format_count(value)}"


def _format_score(value: object) -> str:
    if pd.isna(value):
        return "--"
    return f"{_format_decimal(value, 0)}%"


def _status_html(status: str) -> str:
    tone = {
        "Vooruit": "is-up",
        "Achteruit": "is-down",
        "Stabiel": "is-flat",
        "Nieuw": "is-new",
    }.get(str(status), "is-flat")
    return f'<span class="bench-status-pill {tone}">{status}</span>'


def _status_class(status: str) -> str:
    return {
        "Vooruit": "is-up",
        "Achteruit": "is-down",
        "Stabiel": "is-flat",
        "Nieuw": "is-new",
    }.get(str(status), "is-flat")


def build_stat_card(label: str, value: str, note: str) -> str:
    return f"""
    <div class="bench-stat-card">
      <div class="bench-stat-label">{label}</div>
      <div class="bench-stat-value">{value}</div>
      <div class="bench-stat-note">{note}</div>
    </div>
    """


def render_stat_cards(cards: list[tuple[str, str, str]], columns_per_row: int) -> None:
    for start in range(0, len(cards), columns_per_row):
        row_cards = cards[start : start + columns_per_row]
        columns = st.columns(columns_per_row, gap="small")
        for column, (label, value, note) in zip(columns, row_cards):
            with column:
                st.markdown(build_stat_card(label, value, note), unsafe_allow_html=True)


def build_signal_card(label: str, title: str, note: str, status: str = "Stabiel") -> str:
    tone = _status_class(status)
    return f"""
    <div class="bench-signal-card {tone}">
      <div class="bench-signal-label">{label}</div>
      <div class="bench-signal-title">{title}</div>
      <div class="bench-signal-note">{note}</div>
    </div>
    """


def build_position_card(position: str, row: pd.Series, metric_key: str) -> str:
    status = str(row.get("overall_status", "--"))
    tone = _status_class(status)
    current_value = _format_metric(metric_key, row.get(f"{metric_key}_current"))
    benchmark_value = _format_metric(metric_key, row.get(f"{metric_key}_benchmark"))
    gap_value = _format_gap(metric_key, row.get(f"{metric_key}_gap_current"))
    score_value = _format_score(row.get("current_score"))
    delta_value = row.get("score_change")
    score_delta = "--" if pd.isna(delta_value) else f"{'+' if float(delta_value) >= 0 else ''}{_format_decimal(delta_value, 1)}"
    return f"""
    <div class="bench-position-card">
      <div class="bench-position-head">
        <div class="bench-position-code">{position}</div>
        <span class="bench-status-pill {tone}">{status}</span>
      </div>
      <div class="bench-position-metric">{METRIC_SPECS[metric_key]['label']}</div>
      <div class="bench-position-value">{current_value}</div>
      <div class="bench-position-grid">
        <div><span>Benchmark</span><strong>{benchmark_value}</strong></div>
        <div><span>Gap</span><strong>{gap_value}</strong></div>
        <div><span>Bench-score</span><strong>{score_value}</strong></div>
        <div><span>Delta</span><strong>{score_delta}</strong></div>
      </div>
    </div>
    """

def rest_headers(access_token: str) -> dict[str, str]:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Prefer": "count=exact",
    }


def rest_get_paged(
    access_token: str,
    table: str,
    base_query: str,
    page_size: int = 5000,
    timeout: int = 120,
) -> pd.DataFrame:
    url = f"{SUPABASE_URL}/rest/v1/{table}?{base_query}"
    headers = rest_headers(access_token) | {"Range-Unit": "items"}
    all_rows: list[dict[str, Any]] = []
    start = 0

    while True:
        end = start + page_size - 1
        batch_headers = headers | {"Range": f"{start}-{end}"}
        response = requests.get(url, headers=batch_headers, timeout=timeout)
        if not response.ok:
            raise RuntimeError(f"GET {table} failed ({response.status_code}): {response.text}")

        batch = response.json()
        if not batch:
            break

        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size

    return pd.DataFrame(all_rows)


@st.cache_data(show_spinner=False, ttl=300)
def fetch_active_players_cached(_sb, cache_scope: str = "benchmarks") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    select_variants = [
        {"select": "player_id,full_name,is_active,position,sub_position", "filter_active": True},
        {"select": 'player_id,full_name,is_active,"Position",sub_position', "filter_active": True},
        {"select": 'player_id,full_name,is_active,"Position"', "filter_active": True},
        {"select": "player_id,full_name,is_active,position", "filter_active": True},
        {"select": "player_id,full_name,is_active", "filter_active": True},
        {"select": "player_id,display_name,position,sub_position", "filter_active": False},
        {"select": 'player_id,display_name,"Position",sub_position', "filter_active": False},
        {"select": "player_id,display_name,position", "filter_active": False},
        {"select": "player_id,display_name", "filter_active": False},
    ]

    for variant in select_variants:
        try:
            query = _sb.table("players").select(variant["select"])
            if variant["filter_active"]:
                query = query.eq("is_active", True)
            order_column = "full_name" if "full_name" in variant["select"] else "display_name"
            rows = query.order(order_column).execute().data or []
            break
        except Exception:
            rows = []

    if not rows:
        return pd.DataFrame(columns=["player_id", "full_name", "position", "sub_position"])

    records = []
    for row in rows:
        player_id = row.get("player_id")
        full_name = str(row.get("full_name") or row.get("display_name") or "").strip()
        position_value = row.get("Position")
        if position_value is None:
            position_value = row.get("position")
        sub_position_value = row.get("sub_position")
        if player_id and full_name:
            records.append(
                {
                    "player_id": str(player_id),
                    "full_name": full_name,
                    "position": str(position_value or "").strip(),
                    "sub_position": str(sub_position_value or "").strip().upper(),
                }
            )

    return pd.DataFrame(records)


def get_temp_subposition_overrides() -> dict[str, str]:
    raw = st.session_state.get(BENCH_TEMP_SUBPOSITION_KEY)
    if not isinstance(raw, dict):
        raw = {}
        st.session_state[BENCH_TEMP_SUBPOSITION_KEY] = raw
    return {str(key): str(value or "").strip().upper() for key, value in raw.items() if str(key).strip()}


def set_temp_subposition_override(player_id: str, sub_position: str) -> None:
    overrides = dict(get_temp_subposition_overrides())
    normalized_player_id = str(player_id or "").strip()
    normalized_value = str(sub_position or "").strip().upper()
    if not normalized_player_id:
        return
    if normalized_value:
        overrides[normalized_player_id] = normalized_value
    else:
        overrides.pop(normalized_player_id, None)
    st.session_state[BENCH_TEMP_SUBPOSITION_KEY] = overrides


def resolve_benchmark_source_position(position_value: object, sub_position_value: object, temp_override: object) -> str:
    for candidate in (temp_override, sub_position_value, position_value):
        normalized = str(candidate or "").strip().upper()
        if normalized:
            return normalized
    return ""


def apply_benchmark_position_overrides(players_df: pd.DataFrame) -> pd.DataFrame:
    if players_df.empty:
        return players_df

    working_df = players_df.copy()
    if "sub_position" not in working_df.columns:
        working_df["sub_position"] = ""

    temp_overrides = get_temp_subposition_overrides()
    working_df["temp_sub_position"] = working_df["player_id"].astype(str).map(temp_overrides).fillna("")
    working_df["benchmark_position_source"] = working_df.apply(
        lambda row: resolve_benchmark_source_position(
            row.get("position"),
            row.get("sub_position"),
            row.get("temp_sub_position"),
        ),
        axis=1,
    )
    return working_df


def save_player_sub_position(sb, player_id: str, sub_position: str) -> tuple[bool, str]:
    normalized_player_id = str(player_id or "").strip()
    normalized_value = str(sub_position or "").strip().upper()
    if not normalized_player_id:
        return False, "Geen speler geselecteerd."

    payload = {"sub_position": normalized_value or None}
    try:
        (
            sb.table("players")
            .update(payload)
            .eq("player_id", normalized_player_id)
            .execute()
        )
        fetch_active_players_cached.clear()
        return True, "Subpositie permanent opgeslagen."
    except Exception as exc:
        message = str(exc)
        if "sub_position" in message.lower():
            return False, "Permanent opslaan lukt nog niet omdat de kolom `sub_position` nog niet in Supabase staat. Voer eerst de migratie uit."
        return False, f"Permanent opslaan faalde: {exc}"


@st.cache_data(show_spinner=False, ttl=300)
def fetch_summary_history_cached(access_token: str, start_iso: str) -> pd.DataFrame:
    raw = rest_get_paged(
        access_token,
        "gps_records",
        f"select={','.join(GPS_SELECT_COLS)}&event=eq.Summary&datum=gte.{start_iso}&order=datum.asc,gps_id.asc",
    )
    if raw.empty:
        return raw

    df = raw.copy()
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce").dt.normalize()
    df["player_id"] = df["player_id"].fillna("").astype(str) if "player_id" in df.columns else ""
    df["player_name"] = (
        df["player_name"].fillna("Onbekend").astype(str).str.strip() if "player_name" in df.columns else "Onbekend"
    )
    df["type"] = df["type"].fillna("").astype(str).str.strip() if "type" in df.columns else ""
    for column in NUMERIC_GPS_COLS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["datum"]).copy()
    return df


def build_benchmark_numeric_table(source_key: str) -> pd.DataFrame:
    source_df = BENCHMARK_SOURCE_TABLES[source_key]
    numeric = pd.DataFrame({"Positie": source_df["Positie"].astype(str)})
    for metric_key, spec in METRIC_SPECS.items():
        numeric[f"{metric_key}_benchmark"] = source_df[spec["benchmark_col"]].apply(parse_metric_value)
    return numeric


def _canonical_player_name(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def map_position(raw_position: object, source_key: str) -> str | None:
    value = str(raw_position or "").strip().upper()
    if not value:
        return None

    valid_codes = set(BENCHMARK_SOURCE_TABLES[source_key]["Positie"].astype(str))

    candidates = [segment.strip() for segment in re.split(r"[/,|;]+", value) if segment.strip()]
    if not candidates:
        candidates = [value]

    for candidate in candidates:
        cleaned = re.sub(r"[^A-Z ]", " ", candidate)
        cleaned = " ".join(cleaned.split())
        if cleaned in valid_codes:
            return cleaned
        if cleaned in POSITION_EXACT_CODES and cleaned in valid_codes:
            return cleaned

        if any(token in cleaned for token in ("KEEPER", "GOAL", "DOEL", "GK")):
            return "GK" if "GK" in valid_codes else None
        if "LEFT" in cleaned and "BACK" in cleaned:
            return "LB" if "LB" in valid_codes else ("DEF" if "DEF" in valid_codes else "CB")
        if "RIGHT" in cleaned and "BACK" in cleaned:
            return "RB" if "RB" in valid_codes else ("DEF" if "DEF" in valid_codes else "CB")
        if "WING" in cleaned and "LEFT" in cleaned:
            return "LW" if "LW" in valid_codes else ("FOR" if "FOR" in valid_codes else "CF")
        if "WING" in cleaned and "RIGHT" in cleaned:
            return "RW" if "RW" in valid_codes else ("FOR" if "FOR" in valid_codes else "CF")
        if "ATTACK" in cleaned and "MID" in cleaned:
            return "AM" if "AM" in valid_codes else ("MID" if "MID" in valid_codes else "CM")
        if "DEFENS" in cleaned and "MID" in cleaned:
            return "DM" if "DM" in valid_codes else ("MID" if "MID" in valid_codes else "CM")
        if "CENTRAL" in cleaned and "MID" in cleaned:
            return "CM" if "CM" in valid_codes else "MID"
        if "CENTER" in cleaned and "BACK" in cleaned:
            return "CB" if "CB" in valid_codes else "DEF"
        if "CENTRE" in cleaned and "BACK" in cleaned:
            return "CB" if "CB" in valid_codes else "DEF"
        if any(token in cleaned for token in ("VERDEDIG", "DEFENDER", "DEFENCE", "DEFENSE")):
            return "DEF" if "DEF" in valid_codes else "CB"
        if any(token in cleaned for token in ("MIDDENVELD", "MIDFIELD", "MIDFIELDER")):
            return "MID" if "MID" in valid_codes else "CM"
        if any(token in cleaned for token in ("AANVALL", "ATTACKER", "FORWARD", "STRIKER", "SPITS")):
            return "FOR" if "FOR" in valid_codes else "CF"

        fallback_map = {
            "MID": "MID" if "MID" in valid_codes else "CM",
            "FOR": "FOR" if "FOR" in valid_codes else "CF",
            "DEF": "DEF" if "DEF" in valid_codes else "CB",
        }
        if cleaned in fallback_map:
            return fallback_map[cleaned]

    return None


def build_position_snapshot(
    summary_df: pd.DataFrame,
    players_df: pd.DataFrame,
    source_key: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if summary_df.empty or players_df.empty:
        return pd.DataFrame()

    window_df = summary_df.loc[
        summary_df["datum"].between(pd.Timestamp(start_date), pd.Timestamp(end_date), inclusive="both")
    ].copy()
    if window_df.empty:
        return pd.DataFrame()

    players_df = apply_benchmark_position_overrides(players_df)
    player_map = players_df[["player_id", "full_name", "position", "benchmark_position_source"]].copy()
    player_map["player_id"] = player_map["player_id"].astype(str)
    player_map["full_name_key"] = player_map["full_name"].map(_canonical_player_name)

    name_position_map = (
        player_map.sort_values("full_name")
        .drop_duplicates("full_name_key")
        .set_index("full_name_key")["benchmark_position_source"]
        .to_dict()
    )

    window_df["player_id"] = window_df["player_id"].astype(str)
    window_df["player_name_key"] = window_df["player_name"].map(_canonical_player_name)
    window_df = window_df.merge(player_map[["player_id", "benchmark_position_source"]], on="player_id", how="left")
    window_df["benchmark_position_source"] = window_df["benchmark_position_source"].fillna(window_df["player_name_key"].map(name_position_map))
    window_df["Positie"] = window_df["benchmark_position_source"].apply(lambda value: map_position(value, source_key))
    window_df = window_df.loc[window_df["Positie"].notna()].copy()
    if window_df.empty:
        return pd.DataFrame()

    player_totals = (
        window_df.groupby(["Positie", "player_id", "player_name"], as_index=False)
        .agg(
            session_count=("gps_id", "size"),
            duration=("duration", "sum"),
            total_distance=("total_distance", "sum"),
            sprint=("sprint", "sum"),
            high_sprint=("high_sprint", "sum"),
            number_of_sprints=("number_of_sprints", "sum"),
        )
    )
    player_totals = player_totals.loc[player_totals["duration"] > 0].copy()
    if player_totals.empty:
        return pd.DataFrame()

    player_totals["hsr_hsd"] = player_totals["sprint"] + player_totals["high_sprint"]
    player_totals["total_distance_90"] = _safe_divide_series(player_totals["total_distance"], player_totals["duration"], 90.0)
    player_totals["hsr_hsd_90"] = _safe_divide_series(player_totals["hsr_hsd"], player_totals["duration"], 90.0)
    player_totals["sprint_distance_90"] = _safe_divide_series(player_totals["high_sprint"], player_totals["duration"], 90.0)
    player_totals["sprint_count_90"] = _safe_divide_series(player_totals["number_of_sprints"], player_totals["duration"], 90.0)
    player_totals["total_distance_per_min"] = _safe_divide_series(player_totals["total_distance"], player_totals["duration"])
    player_totals["intensity_pct"] = _safe_divide_series(player_totals["hsr_hsd"], player_totals["total_distance"], 100.0)

    position_snapshot = (
        player_totals.groupby("Positie", as_index=False)
        .agg(
            active_players=("player_name", "nunique"),
            total_sessions=("session_count", "sum"),
            total_distance_90=("total_distance_90", "mean"),
            hsr_hsd_90=("hsr_hsd_90", "mean"),
            sprint_distance_90=("sprint_distance_90", "mean"),
            sprint_count_90=("sprint_count_90", "mean"),
            total_distance_per_min=("total_distance_per_min", "mean"),
            intensity_pct=("intensity_pct", "mean"),
        )
    )
    return position_snapshot


def build_player_snapshot(
    summary_df: pd.DataFrame,
    players_df: pd.DataFrame,
    source_key: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if summary_df.empty or players_df.empty:
        return pd.DataFrame()

    window_df = summary_df.loc[
        summary_df["datum"].between(pd.Timestamp(start_date), pd.Timestamp(end_date), inclusive="both")
    ].copy()
    if window_df.empty:
        return pd.DataFrame()

    players_df = apply_benchmark_position_overrides(players_df)
    player_map = players_df[["player_id", "full_name", "position", "benchmark_position_source"]].copy()
    player_map["player_id"] = player_map["player_id"].astype(str)
    player_map["full_name_key"] = player_map["full_name"].map(_canonical_player_name)

    name_position_map = (
        player_map.sort_values("full_name")
        .drop_duplicates("full_name_key")
        .set_index("full_name_key")["benchmark_position_source"]
        .to_dict()
    )

    window_df["player_id"] = window_df["player_id"].astype(str)
    window_df["player_name_key"] = window_df["player_name"].map(_canonical_player_name)
    window_df = window_df.merge(player_map[["player_id", "benchmark_position_source"]], on="player_id", how="left")
    window_df["benchmark_position_source"] = window_df["benchmark_position_source"].fillna(window_df["player_name_key"].map(name_position_map))
    window_df["Positie"] = window_df["benchmark_position_source"].apply(lambda value: map_position(value, source_key))
    window_df = window_df.loc[window_df["Positie"].notna()].copy()
    if window_df.empty:
        return pd.DataFrame()

    player_totals = (
        window_df.groupby(["Positie", "player_id", "player_name"], as_index=False)
        .agg(
            session_count=("gps_id", "size"),
            duration=("duration", "sum"),
            total_distance=("total_distance", "sum"),
            sprint=("sprint", "sum"),
            high_sprint=("high_sprint", "sum"),
            number_of_sprints=("number_of_sprints", "sum"),
        )
    )
    player_totals = player_totals.loc[player_totals["duration"] > 0].copy()
    if player_totals.empty:
        return pd.DataFrame()

    player_totals["hsr_hsd"] = player_totals["sprint"] + player_totals["high_sprint"]
    player_totals["total_distance_90"] = _safe_divide_series(player_totals["total_distance"], player_totals["duration"], 90.0)
    player_totals["hsr_hsd_90"] = _safe_divide_series(player_totals["hsr_hsd"], player_totals["duration"], 90.0)
    player_totals["sprint_distance_90"] = _safe_divide_series(player_totals["high_sprint"], player_totals["duration"], 90.0)
    player_totals["sprint_count_90"] = _safe_divide_series(player_totals["number_of_sprints"], player_totals["duration"], 90.0)
    player_totals["total_distance_per_min"] = _safe_divide_series(player_totals["total_distance"], player_totals["duration"])
    player_totals["intensity_pct"] = _safe_divide_series(player_totals["hsr_hsd"], player_totals["total_distance"], 100.0)
    return player_totals


def lookup_runs_benchmark_value(position: str) -> str:
    for source_df in BENCHMARK_SOURCE_TABLES.values():
        match = source_df.loc[source_df["Positie"].astype(str) == str(position), "Runs count >15.0 (#)"]
        if not match.empty:
            return str(match.iloc[0])
    return "--"


def _metric_status(current: object, previous: object, benchmark: object, tolerance: float) -> str:
    if pd.isna(current) or pd.isna(benchmark):
        return "--"
    if pd.isna(previous):
        return "Nieuw"
    current_gap = abs(float(current) - float(benchmark))
    previous_gap = abs(float(previous) - float(benchmark))
    delta = previous_gap - current_gap
    if delta > tolerance:
        return "Vooruit"
    if delta < -tolerance:
        return "Achteruit"
    return "Stabiel"


def _score_against_benchmark(row: pd.Series, prefix: str) -> float:
    penalties: list[float] = []
    for metric_key in METRIC_SPECS:
        value = row.get(f"{metric_key}_{prefix}")
        benchmark = row.get(f"{metric_key}_benchmark")
        if pd.isna(value) or pd.isna(benchmark) or float(benchmark) == 0:
            continue
        penalties.append(abs(float(value) - float(benchmark)) / abs(float(benchmark)))
    if not penalties:
        return float("nan")
    return max(0.0, 100.0 - (sum(penalties) / len(penalties)) * 100.0)


def build_compare_report(
    summary_df: pd.DataFrame,
    players_df: pd.DataFrame,
    source_key: str,
    weeks: int,
) -> dict[str, Any]:
    if summary_df.empty:
        return {"report_df": pd.DataFrame(), "note": "Geen Summary-data beschikbaar."}

    latest_timestamp = pd.to_datetime(summary_df["datum"], errors="coerce").max()
    if pd.isna(latest_timestamp):
        return {"report_df": pd.DataFrame(), "note": "Geen geldige datums in de Summary-data."}

    current_end = latest_timestamp.date()
    current_start = current_end - timedelta(days=(weeks * 7) - 1)
    previous_end = current_start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=(weeks * 7) - 1)

    benchmark_df = build_benchmark_numeric_table(source_key)
    current_df = build_position_snapshot(summary_df, players_df, source_key, current_start, current_end)
    previous_df = build_position_snapshot(summary_df, players_df, source_key, previous_start, previous_end)
    player_current_df = build_player_snapshot(summary_df, players_df, source_key, current_start, current_end)
    player_previous_df = build_player_snapshot(summary_df, players_df, source_key, previous_start, previous_end)

    if current_df.empty:
        return {
            "report_df": pd.DataFrame(),
            "player_report_df": pd.DataFrame(),
            "note": "Geen posities met bruikbare actuele Summary-data gevonden voor deze periode.",
            "current_start": current_start,
            "current_end": current_end,
            "previous_start": previous_start,
            "previous_end": previous_end,
        }

    current_df = current_df.rename(
        columns={column: f"{column}_current" for column in current_df.columns if column != "Positie"}
    )
    previous_df = previous_df.rename(
        columns={column: f"{column}_previous" for column in previous_df.columns if column != "Positie"}
    )

    report_df = benchmark_df.merge(current_df, on="Positie", how="left").merge(previous_df, on="Positie", how="left")

    for metric_key, spec in METRIC_SPECS.items():
        benchmark_col = f"{metric_key}_benchmark"
        current_col = f"{metric_key}_current"
        previous_col = f"{metric_key}_previous"
        report_df[f"{metric_key}_gap_current"] = report_df[current_col] - report_df[benchmark_col]
        report_df[f"{metric_key}_gap_previous"] = report_df[previous_col] - report_df[benchmark_col]
        report_df[f"{metric_key}_status"] = report_df.apply(
            lambda row: _metric_status(
                row.get(current_col),
                row.get(previous_col),
                row.get(benchmark_col),
                float(spec["tolerance"]),
            ),
            axis=1,
        )

    status_columns = [f"{metric_key}_status" for metric_key in METRIC_SPECS]
    report_df["tracked_metrics"] = report_df[status_columns].apply(
        lambda row: int(pd.Series(row).isin(["Vooruit", "Achteruit", "Stabiel"]).sum()),
        axis=1,
    )
    report_df["improved_metrics"] = report_df[status_columns].apply(
        lambda row: int((pd.Series(row) == "Vooruit").sum()),
        axis=1,
    )
    report_df["worsened_metrics"] = report_df[status_columns].apply(
        lambda row: int((pd.Series(row) == "Achteruit").sum()),
        axis=1,
    )
    report_df["current_score"] = report_df.apply(lambda row: _score_against_benchmark(row, "current"), axis=1)
    report_df["previous_score"] = report_df.apply(lambda row: _score_against_benchmark(row, "previous"), axis=1)
    report_df["score_change"] = report_df["current_score"] - report_df["previous_score"]
    report_df["progress_pct"] = _safe_divide_series(
        report_df["improved_metrics"],
        report_df["tracked_metrics"],
        100.0,
    )

    def _overall_status(row: pd.Series) -> str:
        if row.get("tracked_metrics", 0) <= 0 or pd.isna(row.get("score_change")):
            return "--"
        if float(row["score_change"]) > 2.0 or int(row["improved_metrics"]) > int(row["worsened_metrics"]):
            return "Vooruit"
        if float(row["score_change"]) < -2.0 or int(row["worsened_metrics"]) > int(row["improved_metrics"]):
            return "Achteruit"
        return "Stabiel"

    report_df["overall_status"] = report_df.apply(_overall_status, axis=1)
    report_df["active_players_current"] = pd.to_numeric(report_df["active_players_current"], errors="coerce").fillna(0).astype(int)
    report_df["active_players_previous"] = pd.to_numeric(report_df["active_players_previous"], errors="coerce").fillna(0).astype(int)

    player_report_df = pd.DataFrame()
    if not player_current_df.empty:
        player_current_df = player_current_df.rename(
            columns={column: f"{column}_current" for column in player_current_df.columns if column not in {"Positie", "player_id", "player_name"}}
        )
        if not player_previous_df.empty:
            player_previous_df = player_previous_df.rename(
                columns={column: f"{column}_previous" for column in player_previous_df.columns if column not in {"Positie", "player_id", "player_name"}}
            )
        player_report_df = (
            player_current_df.merge(benchmark_df, on="Positie", how="left")
            .merge(player_previous_df, on=["Positie", "player_id", "player_name"], how="left")
        )
        for metric_key, spec in METRIC_SPECS.items():
            benchmark_col = f"{metric_key}_benchmark"
            current_col = f"{metric_key}_current"
            previous_col = f"{metric_key}_previous"
            player_report_df[f"{metric_key}_gap_current"] = player_report_df[current_col] - player_report_df[benchmark_col]
            player_report_df[f"{metric_key}_gap_previous"] = player_report_df[previous_col] - player_report_df[benchmark_col]
            player_report_df[f"{metric_key}_status"] = player_report_df.apply(
                lambda row: _metric_status(
                    row.get(current_col),
                    row.get(previous_col),
                    row.get(benchmark_col),
                    float(spec["tolerance"]),
                ),
                axis=1,
            )
        status_columns = [f"{metric_key}_status" for metric_key in METRIC_SPECS]
        player_report_df["tracked_metrics"] = player_report_df[status_columns].apply(
            lambda row: int(pd.Series(row).isin(["Vooruit", "Achteruit", "Stabiel"]).sum()),
            axis=1,
        )
        player_report_df["improved_metrics"] = player_report_df[status_columns].apply(
            lambda row: int((pd.Series(row) == "Vooruit").sum()),
            axis=1,
        )
        player_report_df["worsened_metrics"] = player_report_df[status_columns].apply(
            lambda row: int((pd.Series(row) == "Achteruit").sum()),
            axis=1,
        )
        player_report_df["current_score"] = player_report_df.apply(lambda row: _score_against_benchmark(row, "current"), axis=1)
        player_report_df["previous_score"] = player_report_df.apply(lambda row: _score_against_benchmark(row, "previous"), axis=1)
        player_report_df["score_change"] = player_report_df["current_score"] - player_report_df["previous_score"]
        player_report_df["progress_pct"] = _safe_divide_series(
            player_report_df["improved_metrics"],
            player_report_df["tracked_metrics"],
            100.0,
        )
        player_report_df["overall_status"] = player_report_df.apply(_overall_status, axis=1)

    return {
        "report_df": report_df,
        "player_report_df": player_report_df,
        "note": None,
        "current_start": current_start,
        "current_end": current_end,
        "previous_start": previous_start,
        "previous_end": previous_end,
    }


def build_focus_table(report_df: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    current_col = f"{metric_key}_current"
    previous_col = f"{metric_key}_previous"
    benchmark_col = f"{metric_key}_benchmark"
    gap_col = f"{metric_key}_gap_current"
    status_col = f"{metric_key}_status"

    focus_df = report_df.loc[report_df[current_col].notna(), [
        "Positie",
        "active_players_current",
        current_col,
        previous_col,
        benchmark_col,
        gap_col,
        status_col,
        "current_score",
        "score_change",
    ]].copy()
    focus_df = focus_df.rename(columns={"active_players_current": "Spelers"})
    focus_df["Huidig"] = focus_df[current_col].apply(lambda value: _format_metric(metric_key, value))
    focus_df["Vorig"] = focus_df[previous_col].apply(lambda value: _format_metric(metric_key, value))
    focus_df["Benchmark"] = focus_df[benchmark_col].apply(lambda value: _format_metric(metric_key, value))
    focus_df["Gap"] = focus_df[gap_col].apply(lambda value: _format_gap(metric_key, value))
    focus_df["Trend"] = focus_df[status_col].astype(str)
    focus_df["Bench-score"] = focus_df["current_score"].apply(_format_score)
    focus_df["Score delta"] = focus_df["score_change"].apply(
        lambda value: "--" if pd.isna(value) else f"{'+' if float(value) >= 0 else ''}{_format_decimal(value, 1)}"
    )
    return focus_df[["Positie", "Spelers", "Huidig", "Vorig", "Benchmark", "Gap", "Trend", "Bench-score", "Score delta"]]


def build_player_focus_table(player_report_df: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    current_col = f"{metric_key}_current"
    previous_col = f"{metric_key}_previous"
    benchmark_col = f"{metric_key}_benchmark"
    gap_col = f"{metric_key}_gap_current"
    status_col = f"{metric_key}_status"
    focus_df = player_report_df.loc[player_report_df[current_col].notna(), [
        "player_name",
        "Positie",
        "session_count_current",
        current_col,
        previous_col,
        benchmark_col,
        gap_col,
        status_col,
        "current_score",
        "score_change",
    ]].copy()
    focus_df = focus_df.rename(columns={"player_name": "Speler", "session_count_current": "Sessies"})
    focus_df["Huidig"] = focus_df[current_col].apply(lambda value: _format_metric(metric_key, value))
    focus_df["Vorig"] = focus_df[previous_col].apply(lambda value: _format_metric(metric_key, value))
    focus_df["Benchmark"] = focus_df[benchmark_col].apply(lambda value: _format_metric(metric_key, value))
    focus_df["Gap"] = focus_df[gap_col].apply(lambda value: _format_gap(metric_key, value))
    focus_df["Trend"] = focus_df[status_col].astype(str)
    focus_df["Bench-score"] = focus_df["current_score"].apply(_format_score)
    focus_df["Score delta"] = focus_df["score_change"].apply(
        lambda value: "--" if pd.isna(value) else f"{'+' if float(value) >= 0 else ''}{_format_decimal(value, 1)}"
    )
    return focus_df[["Speler", "Positie", "Sessies", "Huidig", "Vorig", "Benchmark", "Gap", "Trend", "Bench-score", "Score delta"]]


def build_position_detail_table(report_df: pd.DataFrame, position: str) -> pd.DataFrame:
    row = report_df.loc[report_df["Positie"] == position]
    if row.empty:
        return pd.DataFrame()

    record = row.iloc[0]
    detail_rows = []
    for metric_key, spec in METRIC_SPECS.items():
        detail_rows.append(
            {
                "Metric": spec["label"],
                "Benchmark": _format_metric(metric_key, record.get(f"{metric_key}_benchmark")),
                "Huidig": _format_metric(metric_key, record.get(f"{metric_key}_current")),
                "Vorig": _format_metric(metric_key, record.get(f"{metric_key}_previous")),
                "Gap": _format_gap(metric_key, record.get(f"{metric_key}_gap_current")),
                "Trend": record.get(f"{metric_key}_status", "--"),
            }
        )
    detail_rows.append(
        {
            "Metric": "Runs count >15.0 (#)",
            "Benchmark": lookup_runs_benchmark_value(position),
            "Huidig": "Niet beschikbaar",
            "Vorig": "Niet beschikbaar",
            "Gap": "--",
            "Trend": "Niet in Summary-bron",
        }
    )
    return pd.DataFrame(detail_rows)


def _chart_layout(title: str, height: int = 370) -> dict[str, Any]:
    return dict(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=20, color=MVV_TEXT)),
        height=height,
        margin=dict(l=18, r=18, t=58, b=34),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color=MVV_TEXT, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        barmode="group",
    )


def build_metric_compare_chart(
    report_df: pd.DataFrame,
    metric_key: str,
    source_key: str,
    period_label: str,
) -> go.Figure:
    current_col = f"{metric_key}_current"
    previous_col = f"{metric_key}_previous"
    benchmark_col = f"{metric_key}_benchmark"
    chart_df = report_df.loc[
        report_df[current_col].notna(),
        ["Positie", current_col, previous_col, benchmark_col, "overall_status"],
    ].copy()
    if not chart_df.empty:
        chart_df["gap_abs"] = (chart_df[current_col] - chart_df[benchmark_col]).abs()
        chart_df = chart_df.sort_values("gap_abs", ascending=False)

    fig = go.Figure()
    fig.update_layout(**_chart_layout(f"{METRIC_SPECS[metric_key]['label']} vs benchmark", height=430))
    fig.update_xaxes(gridcolor=MVV_GRID, zeroline=False, tickfont=dict(color=MVV_TEXT_SOFT))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=MVV_TEXT_SOFT))

    if chart_df.empty:
        return fig

    status_color_map = {
        "Vooruit": MVV_GREEN,
        "Achteruit": MVV_RED,
        "Stabiel": MVV_AMBER,
        "Nieuw": "#38BDF8",
    }

    for _, row in chart_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row[benchmark_col], row[current_col]],
                y=[row["Positie"], row["Positie"]],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.22)", width=7),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            name=f"Benchmark {source_key}",
            x=chart_df[benchmark_col],
            y=chart_df["Positie"],
            mode="markers",
            marker=dict(size=12, color=MVV_GOLD, symbol="diamond", line=dict(color="#fff4cc", width=1)),
            customdata=chart_df[benchmark_col],
            hovertemplate="<b>%{y}</b><br>Benchmark: %{customdata:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            name=f"Huidig ({period_label})",
            x=chart_df[current_col],
            y=chart_df["Positie"],
            mode="markers+text",
            marker=dict(
                size=15,
                color=[status_color_map.get(status, METRIC_SPECS[metric_key]["color"]) for status in chart_df["overall_status"]],
                line=dict(color="#ffffff", width=1.5),
            ),
            text=[_format_metric(metric_key, value) for value in chart_df[current_col]],
            textposition="middle right",
            textfont=dict(color=MVV_TEXT, size=11),
            customdata=chart_df[current_col],
            hovertemplate="<b>%{y}</b><br>Huidig: %{customdata:.1f}<extra></extra>",
        )
    )

    if chart_df[previous_col].notna().any():
        fig.add_trace(
            go.Scatter(
                name="Vorige periode",
                x=chart_df[previous_col],
                y=chart_df["Positie"],
                mode="markers",
                marker=dict(size=11, color="#0f172a", line=dict(color="#93C5FD", width=2)),
                customdata=chart_df[previous_col],
                hovertemplate="<b>%{y}</b><br>Vorig: %{customdata:.1f}<extra></extra>",
            )
        )

    return fig


def build_progress_chart(report_df: pd.DataFrame) -> go.Figure:
    chart_df = report_df.loc[report_df["tracked_metrics"] > 0, [
        "Positie",
        "progress_pct",
        "improved_metrics",
        "tracked_metrics",
        "overall_status",
    ]].copy()

    fig = go.Figure()
    fig.update_layout(**_chart_layout("Bench-score per positie", height=430))
    fig.update_yaxes(automargin=True, gridcolor=MVV_GRID, tickfont=dict(color=MVV_TEXT_SOFT))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", showgrid=True, gridcolor=MVV_GRID, tickfont=dict(color=MVV_TEXT_SOFT))

    if chart_df.empty:
        return fig

    score_df = report_df.loc[report_df["current_score"].notna(), [
        "Positie",
        "current_score",
        "previous_score",
        "score_change",
        "overall_status",
    ]].copy()
    score_df = score_df.sort_values("current_score", ascending=True)

    color_map = {
        "Vooruit": MVV_GREEN,
        "Achteruit": MVV_RED,
        "Stabiel": MVV_AMBER,
    }
    fig.add_trace(
        go.Bar(
            x=score_df["current_score"].fillna(0),
            y=score_df["Positie"],
            orientation="h",
            marker_color=[color_map.get(status, "#64748B") for status in score_df["overall_status"]],
            text=[
                "--" if pd.isna(value) else f"{_format_decimal(value, 0)}%"
                for value in score_df["current_score"]
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Bench-score: %{x:.0f}%<extra></extra>",
        )
    )
    if score_df["previous_score"].notna().any():
        fig.add_trace(
            go.Scatter(
                name="Vorige score",
                x=score_df["previous_score"],
                y=score_df["Positie"],
                mode="markers",
                marker=dict(size=10, color="#0f172a", line=dict(color="#93C5FD", width=2)),
                hovertemplate="%{y}<br>Vorige score: %{x:.0f}%<extra></extra>",
            )
        )
    fig.add_vline(x=75, line_dash="dot", line_color="rgba(255,255,255,0.18)")
    return fig


def build_status_heatmap(report_df: pd.DataFrame) -> go.Figure:
    status_columns = [f"{metric_key}_status" for metric_key in METRIC_SPECS]
    labels = [str(METRIC_SPECS[metric_key]["label"]) for metric_key in METRIC_SPECS]
    chart_df = report_df[["Positie"] + status_columns].copy()
    chart_df = chart_df.sort_values("current_score", ascending=False, na_position="last")

    mapping = {"Achteruit": -1, "Stabiel": 0, "Vooruit": 1, "Nieuw": 2, "--": None}
    z_values: list[list[float | None]] = []
    text_values: list[list[str]] = []
    for _, row in chart_df.iterrows():
        z_row: list[float | None] = []
        text_row: list[str] = []
        for column in status_columns:
            status = str(row.get(column, "--"))
            z_row.append(mapping.get(status))
            text_row.append(status)
        z_values.append(z_row)
        text_values.append(text_row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=labels,
            y=chart_df["Positie"],
            text=text_values,
            texttemplate="%{text}",
            colorscale=[
                [0.0, "rgba(234,51,81,0.85)"],
                [0.25, "rgba(234,51,81,0.85)"],
                [0.5, "rgba(245,165,36,0.85)"],
                [0.75, "rgba(47,182,122,0.85)"],
                [1.0, "rgba(56,189,248,0.85)"],
            ],
            zmin=-1,
            zmax=2,
            hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
            showscale=False,
            xgap=6,
            ygap=6,
        )
    )
    fig.update_layout(**_chart_layout("Voortgangsmatrix per metric", height=400))
    fig.update_xaxes(side="top", tickfont=dict(color=MVV_TEXT_SOFT, size=11))
    fig.update_yaxes(tickfont=dict(color=MVV_TEXT_SOFT))
    return fig


def build_player_compare_chart(player_report_df: pd.DataFrame, metric_key: str) -> go.Figure:
    current_col = f"{metric_key}_current"
    previous_col = f"{metric_key}_previous"
    benchmark_col = f"{metric_key}_benchmark"
    chart_df = player_report_df.loc[
        player_report_df[current_col].notna(),
        ["player_name", "Positie", current_col, previous_col, benchmark_col, "overall_status"],
    ].copy()
    if not chart_df.empty:
        chart_df["label"] = chart_df["player_name"].astype(str) + " (" + chart_df["Positie"].astype(str) + ")"
        chart_df["gap_abs"] = (chart_df[current_col] - chart_df[benchmark_col]).abs()
        chart_df = chart_df.sort_values("gap_abs", ascending=False).head(14)

    fig = go.Figure()
    fig.update_layout(**_chart_layout("Spelers vs benchmark", height=470))
    fig.update_xaxes(gridcolor=MVV_GRID, zeroline=False, tickfont=dict(color=MVV_TEXT_SOFT))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=MVV_TEXT_SOFT, size=11))

    if chart_df.empty:
        return fig

    status_color_map = {
        "Vooruit": MVV_GREEN,
        "Achteruit": MVV_RED,
        "Stabiel": MVV_AMBER,
        "Nieuw": "#38BDF8",
    }

    for _, row in chart_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row[benchmark_col], row[current_col]],
                y=[row["label"], row["label"]],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.18)", width=6),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            name="Benchmark",
            x=chart_df[benchmark_col],
            y=chart_df["label"],
            mode="markers",
            marker=dict(size=11, color=MVV_GOLD, symbol="diamond", line=dict(color="#fff4cc", width=1)),
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Huidig",
            x=chart_df[current_col],
            y=chart_df["label"],
            mode="markers+text",
            marker=dict(
                size=14,
                color=[status_color_map.get(status, MVV_RED_SOFT) for status in chart_df["overall_status"]],
                line=dict(color="#ffffff", width=1.4),
            ),
            text=[_format_metric(metric_key, value) for value in chart_df[current_col]],
            textposition="middle right",
            textfont=dict(color=MVV_TEXT, size=11),
        )
    )
    if chart_df[previous_col].notna().any():
        fig.add_trace(
            go.Scatter(
                name="Vorig",
                x=chart_df[previous_col],
                y=chart_df["label"],
                mode="markers",
                marker=dict(size=10, color="#0f172a", line=dict(color="#93C5FD", width=2)),
            )
        )
    return fig


def build_player_score_chart(player_report_df: pd.DataFrame) -> go.Figure:
    chart_df = player_report_df.loc[player_report_df["current_score"].notna(), [
        "player_name",
        "Positie",
        "current_score",
        "previous_score",
        "overall_status",
    ]].copy()
    if not chart_df.empty:
        chart_df["label"] = chart_df["player_name"].astype(str) + " (" + chart_df["Positie"].astype(str) + ")"
        chart_df = chart_df.sort_values("current_score", ascending=True).tail(14)

    fig = go.Figure()
    fig.update_layout(**_chart_layout("Bench-score spelers", height=470))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", gridcolor=MVV_GRID, tickfont=dict(color=MVV_TEXT_SOFT))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=MVV_TEXT_SOFT, size=11))

    if chart_df.empty:
        return fig

    color_map = {
        "Vooruit": MVV_GREEN,
        "Achteruit": MVV_RED,
        "Stabiel": MVV_AMBER,
    }
    fig.add_trace(
        go.Bar(
            x=chart_df["current_score"],
            y=chart_df["label"],
            orientation="h",
            marker_color=[color_map.get(status, "#64748B") for status in chart_df["overall_status"]],
            text=[_format_score(value) for value in chart_df["current_score"]],
            textposition="outside",
            cliponaxis=False,
            name="Huidig",
        )
    )
    if chart_df["previous_score"].notna().any():
        fig.add_trace(
            go.Scatter(
                name="Vorig",
                x=chart_df["previous_score"],
                y=chart_df["label"],
                mode="markers",
                marker=dict(size=9, color="#0f172a", line=dict(color="#93C5FD", width=2)),
            )
        )
    return fig


def map_compare_position(raw_position: object) -> str | None:
    value = str(raw_position or "").strip().upper()
    if not value:
        return None

    candidates = [segment.strip() for segment in re.split(r"[/,|;]+", value) if segment.strip()]
    if not candidates:
        candidates = [value]

    for candidate in candidates:
        cleaned = re.sub(r"[^A-Z ]", " ", candidate)
        cleaned = " ".join(cleaned.split())
        if cleaned in POSITION_EXACT_CODES:
            return cleaned

        if any(token in cleaned for token in ("KEEPER", "GOAL", "DOEL", "GK")):
            return "GK"
        if "LEFT" in cleaned and "BACK" in cleaned:
            return "LB"
        if "RIGHT" in cleaned and "BACK" in cleaned:
            return "RB"
        if "WING" in cleaned and "LEFT" in cleaned:
            return "LW"
        if "WING" in cleaned and "RIGHT" in cleaned:
            return "RW"
        if "ATTACK" in cleaned and "MID" in cleaned:
            return "AM"
        if "DEFENS" in cleaned and "MID" in cleaned:
            return "DM"
        if "CENTRAL" in cleaned and "MID" in cleaned:
            return "CM"
        if "CENTER" in cleaned and "BACK" in cleaned:
            return "CB"
        if "CENTRE" in cleaned and "BACK" in cleaned:
            return "CB"
        if "ATTACK" in cleaned and "FORWARD" in cleaned:
            return "CF"
        if "CENTRE" in cleaned and "FORWARD" in cleaned:
            return "CF"
        if "CENTER" in cleaned and "FORWARD" in cleaned:
            return "CF"
        if "STRIKER" in cleaned or "SPITS" in cleaned:
            return "CF"
        if "LEFT" in cleaned and "WING" in cleaned:
            return "LW"
        if "RIGHT" in cleaned and "WING" in cleaned:
            return "RW"

    return None


def resolve_benchmark_position(position: object, source_key: str) -> str | None:
    value = str(position or "").strip().upper()
    if not value:
        return None

    valid_codes = set(BENCHMARK_SOURCE_TABLES[source_key]["Positie"].astype(str))
    if value in valid_codes:
        return value

    fallback_map = {
        "KKD": {
            "DEF": "CB",
            "MID": "CM",
            "FOR": "CF",
        },
        "Eredivisie": {
            "CM": "MID",
        },
    }
    fallback = fallback_map.get(source_key, {}).get(value)
    if fallback and fallback in valid_codes:
        return fallback
    return None


@st.cache_data(show_spinner=False, ttl=300)
def fetch_match_events_history_cached(access_token: str, start_iso: str) -> pd.DataFrame:
    last_error: Exception | None = None

    for select_clause in MATCH_EVENT_SELECT_VARIANTS:
        try:
            raw = rest_get_paged(
                access_token,
                "v_gps_match_events",
                f"select={select_clause}&datum=gte.{start_iso}&order=datum.asc,gps_id.asc",
            )
            df = raw.copy()
            if df.empty:
                return df

            df["datum"] = pd.to_datetime(df["datum"], errors="coerce").dt.normalize()
            if "player_id" in df.columns:
                df["player_id"] = df["player_id"].fillna("").astype(str)
            if "player_name" in df.columns:
                df["player_name"] = df["player_name"].fillna("Onbekend").astype(str).str.strip()
            if "type" in df.columns:
                df["type"] = df["type"].fillna("").astype(str).str.strip().str.lower()
            if "event" in df.columns:
                df["event"] = df["event"].fillna("").astype(str).str.strip()
            if "match_id" in df.columns:
                df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").astype("Int64")
            for column in MATCH_NUMERIC_COLS:
                if column in df.columns:
                    df[column] = pd.to_numeric(df[column], errors="coerce")
            df = df.dropna(subset=["datum"]).copy()
            return df
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Kon v_gps_match_events niet laden: {last_error}")


def _select_match_event_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    matches_df = df.loc[df["type"].eq("match")].copy()
    if matches_df.empty:
        return matches_df

    event_group_cols = ["player_id", "player_name"]
    if "match_id" in matches_df.columns and matches_df["match_id"].notna().any():
        event_group_cols.append("match_id")
    else:
        event_group_cols.append("datum")

    has_summary = matches_df.groupby(event_group_cols)["event"].transform(lambda values: values.astype(str).eq("Summary").any())
    use_summary = has_summary & matches_df["event"].eq("Summary")
    use_halves = (~has_summary) & matches_df["event"].isin(["First Half", "Second Half"])
    return matches_df.loc[use_summary | use_halves].copy()


def prepare_match_totals_for_compare(
    match_df: pd.DataFrame,
    players_df: pd.DataFrame,
    min_minutes: float,
) -> pd.DataFrame:
    if match_df.empty or players_df.empty:
        return pd.DataFrame()

    working_df = _select_match_event_rows(match_df)
    if working_df.empty:
        return pd.DataFrame()

    players_df = apply_benchmark_position_overrides(players_df)
    player_map = players_df[["player_id", "full_name", "position", "benchmark_position_source", "sub_position", "temp_sub_position"]].copy()
    player_map["player_id"] = player_map["player_id"].astype(str)
    player_map["full_name_key"] = player_map["full_name"].map(_canonical_player_name)

    name_position_map = (
        player_map.sort_values("full_name")
        .drop_duplicates("full_name_key")
        .set_index("full_name_key")["benchmark_position_source"]
        .to_dict()
    )

    working_df["player_id"] = working_df["player_id"].astype(str)
    working_df["player_name_key"] = working_df["player_name"].map(_canonical_player_name)
    working_df = working_df.merge(
        player_map[["player_id", "position", "sub_position", "temp_sub_position", "benchmark_position_source"]],
        on="player_id",
        how="left",
    )
    working_df["benchmark_position_source"] = working_df["benchmark_position_source"].fillna(
        working_df["player_name_key"].map(name_position_map)
    )
    working_df["Positie"] = working_df["benchmark_position_source"].apply(map_compare_position)
    working_df = working_df.loc[working_df["Positie"].notna()].copy()
    if working_df.empty:
        return pd.DataFrame()

    group_cols = ["player_id", "player_name", "datum", "Positie"]
    if "match_id" in working_df.columns:
        group_cols.append("match_id")

    agg_columns = [column for column in MATCH_NUMERIC_COLS if column in working_df.columns]
    if not agg_columns:
        return pd.DataFrame()

    match_totals = working_df.groupby(group_cols, as_index=False)[agg_columns].sum(min_count=1)
    match_totals = match_totals.loc[pd.to_numeric(match_totals["duration"], errors="coerce").fillna(0) >= float(min_minutes)].copy()
    if match_totals.empty:
        return pd.DataFrame()

    match_totals["hsr_hsd"] = (
        pd.to_numeric(match_totals.get("sprint"), errors="coerce").fillna(0.0)
        + pd.to_numeric(match_totals.get("high_sprint"), errors="coerce").fillna(0.0)
    )
    match_totals["total_distance_90"] = _safe_divide_series(match_totals["total_distance"], match_totals["duration"], 90.0)
    match_totals["hsr_hsd_90"] = _safe_divide_series(match_totals["hsr_hsd"], match_totals["duration"], 90.0)
    match_totals["sprint_distance_90"] = _safe_divide_series(match_totals["high_sprint"], match_totals["duration"], 90.0)
    if "number_of_sprints" in match_totals.columns:
        match_totals["sprint_count_90"] = _safe_divide_series(match_totals["number_of_sprints"], match_totals["duration"], 90.0)
    else:
        match_totals["sprint_count_90"] = pd.NA
    match_totals["total_distance_per_min"] = _safe_divide_series(match_totals["total_distance"], match_totals["duration"])
    match_totals["intensity_pct"] = _safe_divide_series(match_totals["hsr_hsd"], match_totals["total_distance"], 100.0)
    match_totals = match_totals.sort_values(["player_name", "datum"], ascending=[True, False]).reset_index(drop=True)
    return match_totals


def build_player_match_compare_bundle(
    match_df: pd.DataFrame,
    players_df: pd.DataFrame,
    match_limit: int | None,
    min_minutes: float,
) -> dict[str, Any]:
    match_totals = prepare_match_totals_for_compare(match_df, players_df, min_minutes)
    if match_totals.empty:
        exact_subposition_count = _count_exact_subpositions(players_df.get("benchmark_position_source", pd.Series(dtype=str)))
        if exact_subposition_count == 0:
            note = "Geen vergelijking beschikbaar: er zijn nog geen exacte subposities gekoppeld. Stel per speler eerst bijvoorbeeld CB, CM, AM, LW of CF in."
        else:
            note = "Geen bruikbare matchdata gevonden voor deze spelers en minutenfilter."
        return {
            "player_compare_df": pd.DataFrame(),
            "match_totals_df": pd.DataFrame(),
            "note": note,
        }

    player_rows: list[dict[str, Any]] = []
    metric_keys = list(METRIC_SPECS.keys())
    group_columns = ["player_id", "player_name", "Positie"]
    for (player_id, player_name, position), player_matches in match_totals.groupby(group_columns, dropna=False):
        sorted_matches = player_matches.sort_values("datum", ascending=False).copy()
        scoped_matches = sorted_matches if match_limit is None else sorted_matches.head(int(match_limit)).copy()
        if scoped_matches.empty:
            continue

        row: dict[str, Any] = {
            "player_id": player_id,
            "player_name": player_name,
            "Positie": position,
            "match_count": int(len(scoped_matches)),
            "available_matches": int(len(sorted_matches)),
            "sample_start": scoped_matches["datum"].min(),
            "sample_end": scoped_matches["datum"].max(),
            "last_match": sorted_matches["datum"].max(),
        }
        for metric_key in metric_keys:
            row[f"{metric_key}_current"] = pd.to_numeric(scoped_matches.get(metric_key), errors="coerce").mean()
        player_rows.append(row)

    player_compare_df = pd.DataFrame(player_rows)
    if player_compare_df.empty:
        return {
            "player_compare_df": pd.DataFrame(),
            "match_totals_df": match_totals,
            "note": "Geen spelervergelijkingen beschikbaar na het aggregeren van de matchdata.",
        }

    kkd_bench_df = build_benchmark_numeric_table("KKD").rename(
        columns={
            "Positie": "kkd_position",
            **{f"{metric_key}_benchmark": f"{metric_key}_kkd_benchmark" for metric_key in metric_keys},
        }
    )
    eredivisie_bench_df = build_benchmark_numeric_table("Eredivisie").rename(
        columns={
            "Positie": "eredivisie_position",
            **{f"{metric_key}_benchmark": f"{metric_key}_eredivisie_benchmark" for metric_key in metric_keys},
        }
    )

    player_compare_df["kkd_position"] = player_compare_df["Positie"].apply(lambda value: resolve_benchmark_position(value, "KKD"))
    player_compare_df["eredivisie_position"] = player_compare_df["Positie"].apply(lambda value: resolve_benchmark_position(value, "Eredivisie"))
    player_compare_df = player_compare_df.merge(kkd_bench_df, on="kkd_position", how="left")
    player_compare_df = player_compare_df.merge(eredivisie_bench_df, on="eredivisie_position", how="left")

    for metric_key in metric_keys:
        player_compare_df[f"{metric_key}_gap_kkd"] = player_compare_df[f"{metric_key}_current"] - player_compare_df[f"{metric_key}_kkd_benchmark"]
        player_compare_df[f"{metric_key}_gap_eredivisie"] = player_compare_df[f"{metric_key}_current"] - player_compare_df[f"{metric_key}_eredivisie_benchmark"]

    return {
        "player_compare_df": player_compare_df,
        "match_totals_df": match_totals,
        "note": None,
    }


def classify_focus_level(row: pd.Series, metric_key: str) -> str:
    current_value = row.get(f"{metric_key}_current")
    kkd_value = row.get(f"{metric_key}_kkd_benchmark")
    eredivisie_value = row.get(f"{metric_key}_eredivisie_benchmark")
    tolerance = float(METRIC_SPECS[metric_key]["tolerance"])

    if not pd.isna(eredivisie_value) and not pd.isna(current_value) and float(current_value) >= float(eredivisie_value) - tolerance:
        return "Eredivisie-niveau"
    if not pd.isna(kkd_value) and not pd.isna(current_value) and float(current_value) >= float(kkd_value) - tolerance:
        return "KKD-niveau"
    if pd.isna(kkd_value) and pd.isna(eredivisie_value):
        return "Geen koppeling"
    return "Onder KKD"


def build_dual_benchmark_chart(
    player_compare_df: pd.DataFrame,
    metric_key: str,
    match_totals_df: pd.DataFrame | None = None,
) -> go.Figure:
    current_col = f"{metric_key}_current"
    kkd_col = f"{metric_key}_kkd_benchmark"
    eredivisie_col = f"{metric_key}_eredivisie_benchmark"
    chart_df = player_compare_df.loc[player_compare_df[current_col].notna()].copy()
    fig = go.Figure()
    fig.update_layout(**_chart_layout(f"Spelerload vs benchmarks | {METRIC_SPECS[metric_key]['label']}", height=520))
    fig.update_layout(
        margin=dict(l=18, r=18, t=108, b=58),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="left",
            x=0,
            bgcolor="rgba(10,16,28,0.82)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(color=MVV_TEXT, size=11),
        ),
    )
    fig.update_xaxes(gridcolor=MVV_GRID, zeroline=False, tickfont=dict(color=MVV_TEXT_SOFT))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=MVV_TEXT_SOFT, size=11))
    if chart_df.empty:
        return fig

    chart_df["label"] = chart_df["player_name"].astype(str) + " | " + chart_df["Positie"].astype(str)
    chart_df["focus_level"] = chart_df.apply(lambda row: classify_focus_level(row, metric_key), axis=1)
    chart_df["sort_gap"] = (
        chart_df[f"{metric_key}_gap_eredivisie"].abs().where(chart_df[eredivisie_col].notna(), chart_df[f"{metric_key}_gap_kkd"].abs())
    )
    chart_df["position_order"] = chart_df["Positie"].map({code: idx for idx, code in enumerate(POSITION_DISPLAY_ORDER)}).fillna(999)
    chart_df = chart_df.sort_values(["position_order", "sort_gap", current_col], ascending=[True, False, False]).head(16)

    level_colors = {
        "Eredivisie-niveau": MVV_GREEN,
        "KKD-niveau": MVV_GOLD,
        "Onder KKD": MVV_RED,
        "Geen koppeling": "#64748B",
    }

    for _, row in chart_df.iterrows():
        bench_values = [value for value in [row.get(kkd_col), row.get(eredivisie_col)] if not pd.isna(value)]
        if bench_values:
            fig.add_trace(
                go.Scatter(
                    x=[min(bench_values), max(bench_values)],
                    y=[row["label"], row["label"]],
                    mode="lines",
                    line=dict(color="rgba(255,255,255,0.16)", width=8),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    if match_totals_df is not None and not match_totals_df.empty and metric_key in match_totals_df.columns:
        recent_matches = (
            match_totals_df.loc[
                match_totals_df["player_id"].astype(str).isin(chart_df["player_id"].astype(str)),
                ["player_id", "datum", metric_key, "duration"],
            ]
            .copy()
        )
        if not recent_matches.empty:
            recent_matches["player_id"] = recent_matches["player_id"].astype(str)
            recent_matches["datum"] = pd.to_datetime(recent_matches["datum"], errors="coerce")
            recent_matches = recent_matches.dropna(subset=["datum"])
            recent_matches = recent_matches.sort_values(["player_id", "datum"], ascending=[True, False])
            recent_matches["match_rank"] = recent_matches.groupby("player_id").cumcount() + 1
            recent_matches = recent_matches.loc[recent_matches["match_rank"] <= 5].copy()
            if not recent_matches.empty:
                label_lookup = chart_df.set_index(chart_df["player_id"].astype(str))["label"].to_dict()
                recent_matches["label"] = recent_matches["player_id"].map(label_lookup)
                recent_matches = recent_matches.dropna(subset=["label", metric_key])
                if not recent_matches.empty:
                    recent_matches["match_label"] = recent_matches["datum"].dt.strftime("%d/%m/%Y")
                    recent_matches["metric_label"] = recent_matches[metric_key].apply(lambda value: _format_metric(metric_key, value))
                    fig.add_trace(
                        go.Scatter(
                            name="Laatste 5 matches",
                            x=recent_matches[metric_key],
                            y=recent_matches["label"],
                            mode="markers",
                            marker=dict(
                                size=8,
                                color="rgba(255,255,255,0.28)",
                                line=dict(color="rgba(248,250,252,0.85)", width=1.1),
                            ),
                            customdata=recent_matches[["match_rank", "match_label", "metric_label", "duration"]],
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "Wedstrijd %{customdata[0]}: %{customdata[1]}<br>"
                                "Waarde: %{customdata[2]}<br>"
                                "Minuten: %{customdata[3]:.0f}<extra></extra>"
                            ),
                        )
                    )

    if chart_df[kkd_col].notna().any():
        fig.add_trace(
            go.Scatter(
                name="KKD",
                x=chart_df[kkd_col],
                y=chart_df["label"],
                mode="markers",
                marker=dict(size=11, color=MVV_GOLD, symbol="diamond", line=dict(color="#fff4cc", width=1)),
                hovertemplate="<b>%{y}</b><br>KKD: %{x:.1f}<extra></extra>",
            )
        )
    if chart_df[eredivisie_col].notna().any():
        fig.add_trace(
            go.Scatter(
                name="Eredivisie",
                x=chart_df[eredivisie_col],
                y=chart_df["label"],
                mode="markers",
                marker=dict(size=11, color="#38BDF8", symbol="circle", line=dict(color="#DBF5FF", width=1)),
                hovertemplate="<b>%{y}</b><br>Eredivisie: %{x:.1f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            name="MVV",
            x=chart_df[current_col],
            y=chart_df["label"],
            mode="markers+text",
            marker=dict(
                size=15,
                color=[level_colors.get(level, MVV_RED_SOFT) for level in chart_df["focus_level"]],
                line=dict(color="#ffffff", width=1.5),
            ),
            text=[_format_metric(metric_key, value) for value in chart_df[current_col]],
            textposition="middle right",
            textfont=dict(color=MVV_TEXT, size=11),
            hovertemplate="<b>%{y}</b><br>MVV: %{x:.1f}<extra></extra>",
        )
    )
    return fig


def build_position_benchmark_chart(player_compare_df: pd.DataFrame, metric_key: str) -> go.Figure:
    current_col = f"{metric_key}_current"
    kkd_col = f"{metric_key}_kkd_benchmark"
    eredivisie_col = f"{metric_key}_eredivisie_benchmark"
    chart_df = (
        player_compare_df.loc[player_compare_df[current_col].notna()]
        .groupby("Positie", as_index=False)
        .agg(
            mvv_current=(current_col, "mean"),
            kkd_benchmark=(kkd_col, "mean"),
            eredivisie_benchmark=(eredivisie_col, "mean"),
            players=("player_name", "nunique"),
        )
    )

    if not chart_df.empty:
        chart_df["Positie"] = pd.Categorical(
            chart_df["Positie"],
            categories=POSITION_DISPLAY_ORDER,
            ordered=True,
        )
        chart_df = chart_df.sort_values("Positie").copy()
        chart_df["Positie"] = chart_df["Positie"].astype(str)

    fig = go.Figure()
    fig.update_layout(**_chart_layout(f"Subpositieprofiel | {METRIC_SPECS[metric_key]['label']}", height=460))
    fig.update_layout(
        margin=dict(l=18, r=18, t=108, b=52),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="left",
            x=0,
            bgcolor="rgba(10,16,28,0.82)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(color=MVV_TEXT, size=11),
        ),
    )
    fig.update_xaxes(gridcolor=MVV_GRID, tickfont=dict(color=MVV_TEXT_SOFT))
    fig.update_yaxes(gridcolor=MVV_GRID, tickfont=dict(color=MVV_TEXT_SOFT))
    if chart_df.empty:
        return fig

    if chart_df["kkd_benchmark"].notna().any():
        fig.add_trace(
            go.Scatter(
                name="KKD",
                x=chart_df["Positie"],
                y=chart_df["kkd_benchmark"],
                mode="lines+markers",
                line=dict(color=MVV_GOLD, width=2.5, dash="dot"),
                marker=dict(size=9, color=MVV_GOLD, line=dict(color="#fff4cc", width=1)),
                hovertemplate="<b>%{x}</b><br>KKD: %{y:.1f}<extra></extra>",
            )
        )
    if chart_df["eredivisie_benchmark"].notna().any():
        fig.add_trace(
            go.Scatter(
                name="Eredivisie",
                x=chart_df["Positie"],
                y=chart_df["eredivisie_benchmark"],
                mode="lines+markers",
                line=dict(color="#38BDF8", width=2.5, dash="dash"),
                marker=dict(size=9, color="#38BDF8", line=dict(color="#DBF5FF", width=1)),
                hovertemplate="<b>%{x}</b><br>Eredivisie: %{y:.1f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            name="MVV",
            x=chart_df["Positie"],
            y=chart_df["mvv_current"],
            mode="lines+markers+text",
            line=dict(color=MVV_RED, width=4),
            marker=dict(size=12, color=MVV_RED, line=dict(color="#ffffff", width=1.2)),
            text=[_format_metric(metric_key, value) for value in chart_df["mvv_current"]],
            textposition="top center",
            textfont=dict(color=MVV_TEXT, size=10),
            customdata=chart_df["players"],
            hovertemplate="<b>%{x}</b><br>MVV: %{y:.1f}<br>Spelers: %{customdata}<extra></extra>",
        )
    )
    return fig


def build_player_match_timeline(
    player_matches_df: pd.DataFrame,
    metric_key: str,
    kkd_benchmark: float | None,
    eredivisie_benchmark: float | None,
    player_name: str,
    *,
    title_override: str | None = None,
    height: int = 440,
) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**_chart_layout(title_override or f"{player_name} | matchtrend", height=height))
    fig.update_xaxes(gridcolor=MVV_GRID, tickfont=dict(color=MVV_TEXT_SOFT))
    fig.update_yaxes(gridcolor=MVV_GRID, tickfont=dict(color=MVV_TEXT_SOFT))
    if player_matches_df.empty:
        return fig

    chart_df = player_matches_df.sort_values("datum", ascending=True).copy()
    chart_df["day_rank"] = chart_df.groupby("datum").cumcount() + 1
    chart_df["day_total"] = chart_df.groupby("datum")["datum"].transform("size")
    chart_df["x_label"] = chart_df.apply(
        lambda row: (
            f"{pd.Timestamp(row['datum']).strftime('%d/%m')} ({int(row['day_rank'])})"
            if int(row["day_total"]) > 1
            else pd.Timestamp(row["datum"]).strftime("%d/%m")
        ),
        axis=1,
    )
    current_col = metric_key

    fig.add_trace(
        go.Scatter(
            x=chart_df["x_label"],
            y=chart_df[current_col],
            mode="lines+markers+text",
            name="Matchload",
            line=dict(color=MVV_RED_DEEP, width=3),
            marker=dict(size=10, color=MVV_RED, line=dict(color="#ffffff", width=1.6)),
            text=[_format_metric(metric_key, value) for value in chart_df[current_col]],
            textposition="top center",
            hovertemplate="<b>%{x}</b><br>Waarde: %{y:.1f}<br>Minuten: %{customdata:.0f}<extra></extra>",
            customdata=chart_df["duration"],
        )
    )

    if kkd_benchmark is not None and not pd.isna(kkd_benchmark):
        fig.add_hline(
            y=float(kkd_benchmark),
            line_dash="dot",
            line_color=MVV_GOLD,
            annotation_text=f"KKD {_format_metric(metric_key, kkd_benchmark)}",
            annotation_position="top left",
        )
    if eredivisie_benchmark is not None and not pd.isna(eredivisie_benchmark):
        fig.add_hline(
            y=float(eredivisie_benchmark),
            line_dash="dash",
            line_color="#38BDF8",
            annotation_text=f"Eredivisie {_format_metric(metric_key, eredivisie_benchmark)}",
            annotation_position="bottom left",
        )
    return fig


def build_compare_overview_table(player_compare_df: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    current_col = f"{metric_key}_current"
    kkd_col = f"{metric_key}_kkd_benchmark"
    eredivisie_col = f"{metric_key}_eredivisie_benchmark"
    overview_df = player_compare_df.loc[player_compare_df[current_col].notna(), [
        "player_name",
        "Positie",
        "match_count",
        "available_matches",
        "last_match",
        current_col,
        kkd_col,
        f"{metric_key}_gap_kkd",
        eredivisie_col,
        f"{metric_key}_gap_eredivisie",
    ]].copy()
    overview_df["Niveau"] = overview_df.apply(lambda row: classify_focus_level(row, metric_key), axis=1)
    overview_df["Laatste match"] = pd.to_datetime(overview_df["last_match"], errors="coerce").dt.strftime("%d/%m/%Y")
    overview_df["Huidig"] = overview_df[current_col].apply(lambda value: _format_metric(metric_key, value))
    overview_df["KKD"] = overview_df[kkd_col].apply(lambda value: _format_metric(metric_key, value))
    overview_df["Gap KKD"] = overview_df[f"{metric_key}_gap_kkd"].apply(lambda value: _format_gap(metric_key, value))
    overview_df["Eredivisie"] = overview_df[eredivisie_col].apply(lambda value: _format_metric(metric_key, value))
    overview_df["Gap Eredivisie"] = overview_df[f"{metric_key}_gap_eredivisie"].apply(lambda value: _format_gap(metric_key, value))
    overview_df = overview_df.rename(
        columns={
            "player_name": "Speler",
            "match_count": "Matches in sample",
            "available_matches": "Totaal matches",
        }
    )
    return overview_df[
        ["Speler", "Positie", "Matches in sample", "Totaal matches", "Laatste match", "Huidig", "KKD", "Gap KKD", "Eredivisie", "Gap Eredivisie", "Niveau"]
    ]


def build_player_metric_detail_table(player_row: pd.Series) -> pd.DataFrame:
    detail_rows = []
    for metric_key, spec in METRIC_SPECS.items():
        detail_rows.append(
            {
                "Metric": spec["label"],
                "MVV speler": _format_metric(metric_key, player_row.get(f"{metric_key}_current")),
                "KKD": _format_metric(metric_key, player_row.get(f"{metric_key}_kkd_benchmark")),
                "Gap KKD": _format_gap(metric_key, player_row.get(f"{metric_key}_gap_kkd")),
                "Eredivisie": _format_metric(metric_key, player_row.get(f"{metric_key}_eredivisie_benchmark")),
                "Gap Eredivisie": _format_gap(metric_key, player_row.get(f"{metric_key}_gap_eredivisie")),
            }
        )
    return pd.DataFrame(detail_rows)


def build_player_matches_table(
    player_matches_df: pd.DataFrame,
    metric_key: str,
    kkd_benchmark: float | None,
    eredivisie_benchmark: float | None,
) -> pd.DataFrame:
    table_df = player_matches_df.sort_values("datum", ascending=False).copy()
    if table_df.empty:
        return pd.DataFrame()

    table_df["Datum"] = pd.to_datetime(table_df["datum"], errors="coerce").dt.strftime("%d/%m/%Y")
    table_df["Min"] = table_df["duration"].apply(_format_decimal)
    table_df["Totale afstand /90"] = table_df["total_distance_90"].apply(lambda value: _format_metric("total_distance_90", value))
    table_df["HSR/HSD /90"] = table_df["hsr_hsd_90"].apply(lambda value: _format_metric("hsr_hsd_90", value))
    table_df["Sprintafstand /90"] = table_df["sprint_distance_90"].apply(lambda value: _format_metric("sprint_distance_90", value))
    table_df["Totale afstand /min"] = table_df["total_distance_per_min"].apply(lambda value: _format_metric("total_distance_per_min", value))
    table_df["Intensiteit"] = table_df["intensity_pct"].apply(lambda value: _format_metric("intensity_pct", value))
    table_df["Focus"] = table_df[metric_key].apply(lambda value: _format_metric(metric_key, value))
    table_df["Gap KKD"] = table_df[metric_key].sub(kkd_benchmark).apply(lambda value: _format_gap(metric_key, value) if kkd_benchmark is not None and not pd.isna(kkd_benchmark) else "--")
    table_df["Gap Eredivisie"] = table_df[metric_key].sub(eredivisie_benchmark).apply(lambda value: _format_gap(metric_key, value) if eredivisie_benchmark is not None and not pd.isna(eredivisie_benchmark) else "--")
    return table_df[
        ["Datum", "Min", "Focus", "Gap KKD", "Gap Eredivisie", "Totale afstand /90", "HSR/HSD /90", "Sprintafstand /90", "Totale afstand /min", "Intensiteit"]
    ]


def render_css() -> None:
    background = (
        f"linear-gradient(180deg, rgba(6, 10, 20, 0.82) 0%, rgba(6, 10, 20, 0.80) 100%), "
        f"radial-gradient(circle at top left, rgba(200, 16, 46, 0.16), rgba(200, 16, 46, 0.02) 24%, transparent 46%), "
        f"radial-gradient(circle at top right, rgba(234, 51, 81, 0.10), rgba(234, 51, 81, 0.02) 18%, transparent 42%), "
        f"url('{PAGE_BG_URI}')"
        if PAGE_BG_URI
        else "radial-gradient(circle at top left, rgba(200, 16, 46, 0.28), rgba(200, 16, 46, 0.03) 26%, transparent 48%), radial-gradient(circle at top right, rgba(234, 51, 81, 0.18), rgba(234, 51, 81, 0.03) 18%, transparent 44%), linear-gradient(180deg, #070c18 0%, #0a1020 100%)"
    )
    st.markdown(
        """
        <style>
        .stApp {
          background: __BENCH_BG__;
          background-size: cover;
          background-position: center top;
          background-attachment: fixed;
        }

        .block-container {
          max-width: 1380px;
          padding-top: 1.4rem;
          padding-bottom: 2.4rem;
        }

        .bench-hero {
          border-radius: 10px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(135deg, rgba(18, 25, 42, 0.88), rgba(10, 15, 27, 0.84));
          padding: 1.85rem 1.6rem;
          box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22);
          margin-bottom: 1.2rem;
        }

        .bench-head {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 1rem;
          margin-bottom: 0.9rem;
        }

        .bench-logo {
          width: 78px;
          height: 78px;
          object-fit: contain;
          flex-shrink: 0;
          filter: drop-shadow(0 8px 22px rgba(0,0,0,0.28));
        }

        .bench-copyhead {
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 0.12rem;
          text-align: left;
        }

        .bench-title {
          margin: 0;
          font-size: 2.45rem;
          line-height: 1;
          font-weight: 800;
          color: #ffffff;
        }

        .bench-kicker {
          color: rgba(255,255,255,0.76);
          font-size: 0.74rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.18em;
        }

        .bench-copy {
          max-width: 72ch;
          color: rgba(255,255,255,0.84);
          line-height: 1.55;
        }

        .bench-pill-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
          margin-top: 0.95rem;
        }

        .bench-pill {
          display: inline-flex;
          align-items: center;
          padding: 0.42rem 0.76rem;
          border-radius: 999px;
          font-size: 0.78rem;
          font-weight: 800;
          border: 1px solid rgba(234, 51, 81, 0.22);
          background: rgba(255,255,255,0.06);
          color: rgba(255,255,255,0.92);
        }

        .bench-section-copy {
          color: rgba(255,255,255,0.74);
          line-height: 1.55;
          margin: 0.25rem 0 0.85rem 0;
        }

        .bench-sheet-card {
          border-radius: 12px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(180deg, rgba(17, 23, 38, 0.96), rgba(11, 16, 29, 0.96));
          padding: 0.95rem;
          box-shadow: 0 14px 28px rgba(0, 0, 0, 0.18);
        }

        .bench-sheet-kicker {
          color: rgba(255,255,255,0.62);
          font-size: 0.74rem;
          font-weight: 800;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          margin-bottom: 0.35rem;
        }

        .bench-sheet-title {
          color: #ffffff;
          font-size: 1.05rem;
          font-weight: 800;
          margin-bottom: 0.18rem;
        }

        .bench-sheet-note {
          color: rgba(255,255,255,0.72);
          font-size: 0.88rem;
          margin-bottom: 0.8rem;
        }

        .bench-table-note {
          color: rgba(255,255,255,0.66);
          font-size: 0.8rem;
          margin: 0.75rem 0 0 0;
        }

        .bench-download-card {
          border-radius: 12px;
          border: 1px solid rgba(234, 51, 81, 0.16);
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
          padding: 1rem 1rem 0.9rem 1rem;
          box-shadow: 0 14px 28px rgba(0, 0, 0, 0.18);
          margin-bottom: 1rem;
        }

        .bench-download-label {
          color: rgba(255,255,255,0.62);
          font-size: 0.74rem;
          font-weight: 800;
          letter-spacing: 0.14em;
          text-transform: uppercase;
        }

        .bench-download-title {
          color: #ffffff;
          font-size: 1.15rem;
          font-weight: 800;
          margin-top: 0.35rem;
        }

        .bench-download-copy {
          color: rgba(255,255,255,0.76);
          font-size: 0.88rem;
          line-height: 1.5;
          margin-top: 0.35rem;
        }

        .bench-stat-card {
          border-radius: 12px;
          border: 1px solid rgba(234, 51, 81, 0.18);
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.94), rgba(11, 16, 29, 0.96));
          padding: 0.9rem 1rem;
          box-shadow: 0 12px 24px rgba(0, 0, 0, 0.16);
          min-height: 126px;
        }

        .bench-stat-label {
          color: rgba(255,255,255,0.62);
          font-size: 0.72rem;
          font-weight: 800;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          margin-bottom: 0.4rem;
        }

        .bench-stat-value {
          color: #ffffff;
          font-size: 1.45rem;
          font-weight: 800;
          line-height: 1.05;
          margin-bottom: 0.28rem;
        }

        .bench-stat-note {
          color: rgba(255,255,255,0.74);
          font-size: 0.84rem;
          line-height: 1.4;
        }

        .bench-compare-strip {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 0.75rem;
          margin: 0.95rem 0 1rem 0;
        }

        .bench-compare-cell {
          border-radius: 12px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.94), rgba(11, 16, 29, 0.96));
          padding: 0.9rem 1rem;
        }

        .bench-compare-key {
          color: rgba(255,255,255,0.6);
          font-size: 0.72rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          margin-bottom: 0.35rem;
        }

        .bench-compare-main {
          color: #ffffff;
          font-size: 1.06rem;
          font-weight: 800;
          line-height: 1.2;
        }

        .bench-compare-sub {
          color: rgba(255,255,255,0.72);
          font-size: 0.82rem;
          line-height: 1.4;
          margin-top: 0.25rem;
        }

        .bench-signal-card {
          border-radius: 12px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.94), rgba(11, 16, 29, 0.96));
          padding: 0.95rem 1rem;
          min-height: 132px;
        }

        .bench-signal-card.is-up {
          border-color: rgba(47,182,122,0.38);
          box-shadow: inset 0 0 0 1px rgba(47,182,122,0.12);
        }

        .bench-signal-card.is-down {
          border-color: rgba(234,51,81,0.34);
          box-shadow: inset 0 0 0 1px rgba(234,51,81,0.10);
        }

        .bench-signal-card.is-flat {
          border-color: rgba(245,165,36,0.28);
        }

        .bench-signal-card.is-new {
          border-color: rgba(56,189,248,0.28);
        }

        .bench-signal-label {
          color: rgba(255,255,255,0.6);
          font-size: 0.72rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          margin-bottom: 0.4rem;
        }

        .bench-signal-title {
          color: #ffffff;
          font-size: 1.2rem;
          font-weight: 800;
          line-height: 1.15;
          margin-bottom: 0.35rem;
        }

        .bench-signal-note {
          color: rgba(255,255,255,0.74);
          font-size: 0.84rem;
          line-height: 1.45;
        }

        .bench-subsection {
          color: rgba(255,255,255,0.62);
          font-size: 0.72rem;
          font-weight: 800;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          margin: 1rem 0 0.4rem 0;
        }

        .bench-position-card {
          border-radius: 12px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.94), rgba(11, 16, 29, 0.96));
          padding: 0.95rem 1rem;
          box-shadow: 0 12px 24px rgba(0, 0, 0, 0.16);
          min-height: 186px;
        }

        .bench-position-head {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.6rem;
          margin-bottom: 0.7rem;
        }

        .bench-position-code {
          color: #ffffff;
          font-size: 1.18rem;
          font-weight: 900;
          letter-spacing: 0.04em;
        }

        .bench-position-metric {
          color: rgba(255,255,255,0.64);
          font-size: 0.76rem;
          font-weight: 800;
          letter-spacing: 0.1em;
          text-transform: uppercase;
        }

        .bench-position-value {
          color: #ffffff;
          font-size: 1.45rem;
          font-weight: 800;
          line-height: 1.05;
          margin: 0.35rem 0 0.7rem 0;
        }

        .bench-position-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0.55rem 0.75rem;
        }

        .bench-position-grid span {
          display: block;
          color: rgba(255,255,255,0.58);
          font-size: 0.7rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          margin-bottom: 0.18rem;
        }

        .bench-position-grid strong {
          color: #ffffff;
          font-size: 0.93rem;
          font-weight: 800;
        }

        .bench-status-pill {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          min-width: 84px;
          padding: 0.24rem 0.62rem;
          border-radius: 999px;
          font-size: 0.76rem;
          font-weight: 800;
          border: 1px solid transparent;
        }

        .bench-status-pill.is-up {
          color: #062714;
          background: rgba(47, 182, 122, 0.92);
        }

        .bench-status-pill.is-down {
          color: #ffffff;
          background: rgba(234, 51, 81, 0.92);
        }

        .bench-status-pill.is-flat {
          color: #1f2937;
          background: rgba(245, 165, 36, 0.92);
        }

        .bench-status-pill.is-new {
          color: #082f49;
          background: rgba(56, 189, 248, 0.92);
        }

        .bench-empty {
          border-radius: 12px;
          border: 1px dashed rgba(255,255,255,0.18);
          padding: 1rem 1.05rem;
          color: rgba(255,255,255,0.74);
          background: rgba(255,255,255,0.03);
        }

        .stTabs [data-baseweb="tab-list"] {
          gap: 0.55rem;
          margin-bottom: 0.85rem;
        }

        .stTabs [data-baseweb="tab"] {
          border-radius: 999px;
          background: rgba(12, 18, 31, 0.82);
          border: 1px solid rgba(255,255,255,0.08);
          color: rgba(255,255,255,0.82);
          font-weight: 800;
          padding: 0.5rem 0.95rem;
        }

        .stTabs [aria-selected="true"] {
          border-color: rgba(234, 51, 81, 0.28);
          color: #ffffff;
        }

        @media (max-width: 768px) {
          .bench-head {
            flex-direction: column;
            gap: 0.8rem;
          }

          .bench-copyhead {
            text-align: center;
          }

          .bench-title {
            font-size: 2rem;
          }

          .bench-compare-strip {
            grid-template-columns: 1fr;
          }

          .bench-position-grid {
            grid-template-columns: 1fr;
          }
        }
        </style>
        """.replace("__BENCH_BG__", background),
        unsafe_allow_html=True,
    )


def render_marks_tab() -> None:
    st.markdown(
        '<div class="bench-section-copy">Hier staan de vaste benchmarktabellen precies zoals bronreferentie: KKD, Eredivisie en het directe competitieverschil.</div>',
        unsafe_allow_html=True,
    )
    table_tabs = st.tabs([label for label, _, _, _ in MARKS_TABLES])
    for tab, (_, title, note, table_df) in zip(table_tabs, MARKS_TABLES):
        with tab:
            st.markdown(
                f"""
                <div class="bench-sheet-card">
                  <div class="bench-sheet-kicker">Benchmarkblad</div>
                  <div class="bench-sheet-title">{title}</div>
                  <div class="bench-sheet-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(table_df, width="stretch", hide_index=True)
            st.markdown(
                '<div class="bench-table-note">Waardes per 90 minuten. Afstanden in meters, totale afstand in m/min en intensiteit in %.</div>',
                unsafe_allow_html=True,
            )


def render_compare_tab(sb) -> None:
    scope_labels = list(COMPARE_MATCH_SCOPE_OPTIONS.keys())
    default_scope_index = scope_labels.index("Laatste 5 wedstrijden")

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.markdown('<div class="bench-empty">Supabase-config ontbreekt, daardoor is de compare-rapportage nu niet beschikbaar.</div>', unsafe_allow_html=True)
        return

    try:
        access_token = get_access_token()
        players_df = fetch_active_players_cached(sb)
        fetch_start = (date.today() - timedelta(days=540)).isoformat()
        match_df = fetch_match_events_history_cached(access_token, fetch_start)
    except Exception as exc:
        st.markdown(
            f'<div class="bench-empty">Kon benchmarkdata niet laden: {exc}</div>',
            unsafe_allow_html=True,
        )
        return

    players_df = apply_benchmark_position_overrides(players_df)
    exact_subposition_count = _count_exact_subpositions(players_df.get("benchmark_position_source", pd.Series(dtype=str)))
    total_player_count = int(len(players_df))
    if total_player_count:
        st.caption(
            f"Compare gebruikt nu alleen exacte subposities. "
            f"{exact_subposition_count}/{total_player_count} spelers hebben momenteel een geldige subpositie-instelling."
        )
    player_manager_df = players_df.sort_values("full_name").copy()
    if not player_manager_df.empty:
        with st.expander("Subpositie beheer", expanded=False):
            manager_cols = st.columns([1.2, 0.7, 0.9, 0.9], gap="small")
            player_lookup = {
                str(row["player_id"]): str(row["full_name"])
                for _, row in player_manager_df.iterrows()
                if str(row.get("player_id") or "").strip() and str(row.get("full_name") or "").strip()
            }
            player_ids = list(player_lookup.keys())
            selected_player_id = manager_cols[0].selectbox(
                "Speler",
                options=player_ids,
                format_func=lambda player_id: player_lookup.get(player_id, player_id),
                key="bench_compare_player_selector",
            )
            selected_player_row = player_manager_df.loc[
                player_manager_df["player_id"].astype(str) == str(selected_player_id)
            ].iloc[0]
            base_position = str(selected_player_row.get("position") or "--").strip() or "--"
            permanent_sub_position = str(selected_player_row.get("sub_position") or "").strip().upper()
            temp_sub_position = str(selected_player_row.get("temp_sub_position") or "").strip().upper()
            active_benchmark_position = (
                str(selected_player_row.get("benchmark_position_source") or "").strip()
                or permanent_sub_position
                or base_position
            )
            active_source_label = "Tijdelijk" if temp_sub_position else ("Permanent" if permanent_sub_position else "Hoofdpositie")

            manager_cols[1].text_input("Hoofdpositie", value=base_position, disabled=True)
            permanent_choice = manager_cols[2].selectbox(
                "Permanente subpositie",
                options=SUBPOSITION_OPTIONS,
                index=SUBPOSITION_OPTIONS.index(permanent_sub_position) if permanent_sub_position in SUBPOSITION_OPTIONS else 0,
                key=f"bench_perm_sub_position_{selected_player_id}",
            )
            temp_choice = manager_cols[3].selectbox(
                "Tijdelijke subpositie",
                options=SUBPOSITION_OPTIONS,
                index=SUBPOSITION_OPTIONS.index(temp_sub_position) if temp_sub_position in SUBPOSITION_OPTIONS else 0,
                key=f"bench_temp_sub_position_{selected_player_id}",
            )
            st.caption(f"Actief in compare: {active_benchmark_position} via {active_source_label.lower()} instelling.")

            action_cols = st.columns(3, gap="small")
            if action_cols[0].button("Tijdelijk toepassen", width="stretch", key=f"bench_apply_temp_{selected_player_id}"):
                set_temp_subposition_override(selected_player_id, temp_choice)
                st.rerun()
            if action_cols[1].button("Tijdelijke override wissen", width="stretch", key=f"bench_clear_temp_{selected_player_id}"):
                set_temp_subposition_override(selected_player_id, "")
                st.rerun()
            if action_cols[2].button("Permanent opslaan", width="stretch", key=f"bench_save_perm_{selected_player_id}"):
                ok, message = save_player_sub_position(sb, selected_player_id, permanent_choice)
                if ok:
                    set_temp_subposition_override(selected_player_id, "")
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

    filter_cols = st.columns(5, gap="small")
    scope_label = filter_cols[0].selectbox("Wedstrijdscope", options=scope_labels, index=default_scope_index)
    min_minutes = float(filter_cols[1].slider("Minimum minuten per match", min_value=45, max_value=90, value=60, step=5))

    match_limit = COMPARE_MATCH_SCOPE_OPTIONS[scope_label]
    compare_bundle = build_player_match_compare_bundle(match_df, players_df, match_limit, min_minutes)
    player_compare_df = compare_bundle.get("player_compare_df", pd.DataFrame())
    match_totals_df = compare_bundle.get("match_totals_df", pd.DataFrame())
    if player_compare_df.empty:
        st.markdown(
            f'<div class="bench-empty">{compare_bundle.get("note") or "Geen vergelijkingsdata beschikbaar."}</div>',
            unsafe_allow_html=True,
        )
        return

    available_metric_options = [
        metric_key for metric_key in METRIC_SPECS if player_compare_df.get(f"{metric_key}_current", pd.Series(dtype=float)).notna().any()
    ]
    if not available_metric_options:
        st.markdown(
            '<div class="bench-empty">Geen benchmarkmetrics beschikbaar op basis van de huidige matchdata.</div>',
            unsafe_allow_html=True,
        )
        return

    focus_metric = filter_cols[2].selectbox(
        "Focus metric",
        options=available_metric_options,
        format_func=lambda key: str(METRIC_SPECS[key]["label"]),
    )
    position_options = ["Alle posities"] + _sort_compare_positions(player_compare_df["Positie"].dropna().astype(str).unique().tolist())
    selected_position_filter = filter_cols[3].selectbox("Subpositiefilter", options=position_options)
    player_sort_mode = filter_cols[4].selectbox(
        "Sorteer spelers op",
        options=["Grootste Eredivisie-gap", "Grootste KKD-gap", "Hoogste load", "Naam"],
    )

    player_compare_df = player_compare_df.copy()
    player_compare_df["focus_level"] = player_compare_df.apply(lambda row: classify_focus_level(row, focus_metric), axis=1)

    filtered_player_df = player_compare_df.copy()
    if selected_position_filter != "Alle posities":
        filtered_player_df = filtered_player_df.loc[filtered_player_df["Positie"] == selected_position_filter].copy()

    if player_sort_mode == "Naam":
        filtered_player_df = filtered_player_df.sort_values("player_name", ascending=True)
    elif player_sort_mode == "Hoogste load":
        filtered_player_df = filtered_player_df.sort_values(f"{focus_metric}_current", ascending=False, na_position="last")
    elif player_sort_mode == "Grootste KKD-gap":
        filtered_player_df["focus_sort_gap"] = filtered_player_df[f"{focus_metric}_gap_kkd"].abs()
        filtered_player_df = filtered_player_df.sort_values("focus_sort_gap", ascending=False, na_position="last")
    else:
        filtered_player_df["focus_sort_gap"] = filtered_player_df[f"{focus_metric}_gap_eredivisie"].abs().where(
            filtered_player_df[f"{focus_metric}_eredivisie_benchmark"].notna(),
            filtered_player_df[f"{focus_metric}_gap_kkd"].abs(),
        )
        filtered_player_df = filtered_player_df.sort_values("focus_sort_gap", ascending=False, na_position="last")

    if filtered_player_df.empty:
        st.markdown(
            '<div class="bench-empty">Geen spelers over na dit positiefilter.</div>',
            unsafe_allow_html=True,
        )
        return

    chart_cols = st.columns([0.58, 0.42], gap="large")
    with chart_cols[0]:
        st.plotly_chart(
            build_dual_benchmark_chart(filtered_player_df, focus_metric, match_totals_df),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
    with chart_cols[1]:
        st.plotly_chart(
            build_position_benchmark_chart(filtered_player_df, focus_metric),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )

    player_options = filtered_player_df["player_name"].dropna().astype(str).tolist()
    selected_player = st.selectbox("Speler detailgrafieken", options=player_options)
    selected_player_row = filtered_player_df.loc[filtered_player_df["player_name"] == selected_player].iloc[0]
    selected_player_matches = match_totals_df.loc[match_totals_df["player_name"] == selected_player].copy()
    detail_metric_options = [
        metric_key
        for metric_key in PLAYER_DETAIL_METRIC_ORDER
        if metric_key in METRIC_SPECS
        and metric_key in selected_player_matches.columns
        and selected_player_matches[metric_key].notna().any()
    ]
    if detail_metric_options:
        detail_cols = st.columns(2, gap="large")
        for index, metric_key in enumerate(detail_metric_options):
            with detail_cols[index % 2]:
                st.plotly_chart(
                    build_player_match_timeline(
                        selected_player_matches,
                        metric_key,
                        selected_player_row.get(f"{metric_key}_kkd_benchmark"),
                        selected_player_row.get(f"{metric_key}_eredivisie_benchmark"),
                        selected_player,
                        title_override=f"{selected_player} | {METRIC_SPECS[metric_key]['label']}",
                        height=340,
                    ),
                    width="stretch",
                    config={"displayModeBar": False, "responsive": True},
                )


def main() -> None:
    render_css()
    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    profile = get_profile(sb)
    if not is_staff_user(profile):
        st.error("Geen toegang: deze pagina is alleen voor staff.")
        st.stop()

    render_sidebar_navigation(profile)

    logo_markup = f'<img src="{TEAM_LOGO_URI}" alt="MVV Maastricht" class="bench-logo" />' if TEAM_LOGO_URI else ""
    st.markdown(
        f"""
        <div class="bench-hero">
          <div class="bench-head">
            {logo_markup}
            <div class="bench-copyhead">
              <h1 class="bench-title">Benchmarks</h1>
              <div class="bench-kicker">MVV Maastricht | Data | Benchmarks</div>
            </div>
          </div>
          <div class="bench-copy">
            Gebruik <strong>Marks</strong> voor de vaste referentietabellen en <strong>Compare</strong> om de matchload van jouw spelers direct naast KKD- en Eredivisiebenchmarks te leggen.
          </div>
          <div class="bench-pill-row">
            <span class="bench-pill">KKD 2024/2025</span>
            <span class="bench-pill">Eredivisie 2025/2026</span>
            <span class="bench-pill">Speler vs benchmark</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    marks_tab, compare_tab = st.tabs(["Marks", "Compare"])
    with marks_tab:
        render_marks_tab()
    with compare_tab:
        render_compare_tab(sb)

    render_sidebar_footer(profile)


if __name__ == "__main__":
    main()
