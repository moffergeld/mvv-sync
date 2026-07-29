from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
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

ROOT_DIR = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT_DIR / "Assets" / "Benchmarks"
BENCHMARKS_PDF = BENCHMARKS_DIR / "Positional_Benchmarks_MVV.pdf"

PAGE_BG_URI = build_data_uri(TEAM_HERO_BG)
TEAM_LOGO_URI = build_data_uri(TEAM_LOGO)

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "").strip()

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

PERIOD_OPTIONS = {
    "Laatste 4 weken": 4,
    "Laatste 6 weken": 6,
    "Laatste 8 weken": 8,
    "Laatste 12 weken": 12,
}

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

POSITION_EXACT_CODES = {"AM", "CB", "CF", "CM", "DEF", "DM", "FOR", "GK", "LB", "LW", "MID", "RB", "RW"}


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


@st.cache_data(show_spinner=False)
def load_pdf_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


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
        "player_id,full_name,is_active,position",
        'player_id,full_name,is_active,"Position"',
        "player_id,full_name,is_active",
    ]

    for select_clause in select_variants:
        try:
            rows = (
                _sb.table("players")
                .select(select_clause)
                .eq("is_active", True)
                .order("full_name")
                .execute()
                .data
                or []
            )
            break
        except Exception:
            rows = []

    if not rows:
        return pd.DataFrame(columns=["player_id", "full_name", "position"])

    records = []
    for row in rows:
        player_id = row.get("player_id")
        full_name = str(row.get("full_name") or "").strip()
        position_value = row.get("Position")
        if position_value is None:
            position_value = row.get("position")
        if player_id and full_name:
            records.append(
                {
                    "player_id": str(player_id),
                    "full_name": full_name,
                    "position": str(position_value or "").strip(),
                }
            )

    return pd.DataFrame(records)


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

    player_map = players_df[["player_id", "full_name", "position"]].copy()
    player_map["player_id"] = player_map["player_id"].astype(str)
    player_map["full_name_key"] = player_map["full_name"].map(_canonical_player_name)

    name_position_map = (
        player_map.sort_values("full_name")
        .drop_duplicates("full_name_key")
        .set_index("full_name_key")["position"]
        .to_dict()
    )

    window_df["player_id"] = window_df["player_id"].astype(str)
    window_df["player_name_key"] = window_df["player_name"].map(_canonical_player_name)
    window_df = window_df.merge(player_map[["player_id", "position"]], on="player_id", how="left")
    window_df["position"] = window_df["position"].fillna(window_df["player_name_key"].map(name_position_map))
    window_df["Positie"] = window_df["position"].apply(lambda value: map_position(value, source_key))
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

    player_map = players_df[["player_id", "full_name", "position"]].copy()
    player_map["player_id"] = player_map["player_id"].astype(str)
    player_map["full_name_key"] = player_map["full_name"].map(_canonical_player_name)

    name_position_map = (
        player_map.sort_values("full_name")
        .drop_duplicates("full_name_key")
        .set_index("full_name_key")["position"]
        .to_dict()
    )

    window_df["player_id"] = window_df["player_id"].astype(str)
    window_df["player_name_key"] = window_df["player_name"].map(_canonical_player_name)
    window_df = window_df.merge(player_map[["player_id", "position"]], on="player_id", how="left")
    window_df["position"] = window_df["position"].fillna(window_df["player_name_key"].map(name_position_map))
    window_df["Positie"] = window_df["position"].apply(lambda value: map_position(value, source_key))
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
    st.markdown(
        """
        <div class="bench-sheet-card">
          <div class="bench-sheet-kicker">Benchmark Report</div>
          <div class="bench-sheet-title">Vergelijking met progressie</div>
          <div class="bench-sheet-note">Bekijk per positie of MVV in de gekozen periode dichter bij of verder van de benchmark komt dan in de vorige referentieperiode.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    filter_cols = st.columns(3, gap="small")
    source_key = filter_cols[0].selectbox("Benchmarkbron", options=list(BENCHMARK_SOURCE_TABLES.keys()), index=1)
    period_labels = list(PERIOD_OPTIONS.keys())
    default_period_index = period_labels.index("Laatste 8 weken")
    period_label = filter_cols[1].selectbox("Vergelijkingsperiode", options=period_labels, index=default_period_index)
    metric_options = list(METRIC_SPECS.keys())
    focus_metric = filter_cols[2].selectbox(
        "Focus metric",
        options=metric_options,
        format_func=lambda key: str(METRIC_SPECS[key]["label"]),
    )

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.markdown('<div class="bench-empty">Supabase-config ontbreekt, daardoor is de compare-rapportage nu niet beschikbaar.</div>', unsafe_allow_html=True)
        return

    try:
        access_token = get_access_token()
        max_weeks = max(PERIOD_OPTIONS.values())
        fetch_start = (date.today() - timedelta(days=(max_weeks * 14) + 21)).isoformat()
        players_df = fetch_active_players_cached(sb)
        summary_df = fetch_summary_history_cached(access_token, fetch_start)
    except Exception as exc:
        st.markdown(
            f'<div class="bench-empty">Kon benchmarkdata niet laden: {exc}</div>',
            unsafe_allow_html=True,
        )
        return

    compare_bundle = build_compare_report(summary_df, players_df, source_key, PERIOD_OPTIONS[period_label])
    report_df = compare_bundle.get("report_df", pd.DataFrame())
    player_report_df = compare_bundle.get("player_report_df", pd.DataFrame())
    if report_df.empty:
        st.markdown(
            f'<div class="bench-empty">{compare_bundle.get("note") or "Geen vergelijkingsdata beschikbaar."}</div>',
            unsafe_allow_html=True,
        )
        return

    current_start = compare_bundle["current_start"]
    current_end = compare_bundle["current_end"]
    previous_start = compare_bundle["previous_start"]
    previous_end = compare_bundle["previous_end"]

    tracked_total = int(report_df["tracked_metrics"].sum())
    improved_total = int(report_df["improved_metrics"].sum())
    progressing_positions = int((report_df["overall_status"] == "Vooruit").sum())
    matched_positions = int(report_df[f"{focus_metric}_current"].notna().sum())

    biggest_gap_position = "--"
    biggest_gap_note = "Alle posities liggen dicht bij de benchmark."
    gap_candidates: list[tuple[float, str, str]] = []
    for _, row in report_df.iterrows():
        for metric_key, spec in METRIC_SPECS.items():
            benchmark = row.get(f"{metric_key}_benchmark")
            current = row.get(f"{metric_key}_current")
            if pd.isna(benchmark) or pd.isna(current) or float(benchmark) == 0:
                continue
            relative_gap = abs(float(current) - float(benchmark)) / abs(float(benchmark))
            gap_candidates.append((relative_gap, str(row["Positie"]), str(spec["label"])))
    if gap_candidates:
        relative_gap, biggest_gap_position, biggest_gap_metric = max(gap_candidates, key=lambda item: item[0])
        biggest_gap_note = f"{biggest_gap_metric} | {_format_decimal(relative_gap * 100, 1)}% afwijking"

    player_up_count = int((player_report_df["overall_status"] == "Vooruit").sum()) if not player_report_df.empty else 0
    total_players = int(player_report_df["player_name"].nunique()) if not player_report_df.empty else 0
    best_player_name = "--"
    best_player_note = "Geen individuele benchmarkscore beschikbaar."
    if not player_report_df.empty and player_report_df["current_score"].notna().any():
        best_player_row = player_report_df.sort_values("current_score", ascending=False).iloc[0]
        best_player_name = str(best_player_row["player_name"])
        best_player_note = f"{best_player_row['Positie']} | Bench-score {_format_score(best_player_row['current_score'])}"

    watch_player_name = "--"
    watch_player_note = "Geen individuele gap gevonden."
    if not player_report_df.empty:
        player_gap_df = player_report_df.loc[player_report_df[f"{focus_metric}_current"].notna()].copy()
        if not player_gap_df.empty:
            player_gap_df["focus_gap_abs"] = (player_gap_df[f"{focus_metric}_gap_current"]).abs()
            watch_row = player_gap_df.sort_values("focus_gap_abs", ascending=False).iloc[0]
            watch_player_name = str(watch_row["player_name"])
            watch_player_note = f"{watch_row['Positie']} | Gap { _format_gap(focus_metric, watch_row[f'{focus_metric}_gap_current']) }"

    st.markdown(
        f"""
        <div class="bench-compare-strip">
          <div class="bench-compare-cell">
            <div class="bench-compare-key">Benchmarkbron</div>
            <div class="bench-compare-main">{source_key}</div>
            <div class="bench-compare-sub">Referentietabel voor alle vergelijkingen in dit rapport.</div>
          </div>
          <div class="bench-compare-cell">
            <div class="bench-compare-key">Periode</div>
            <div class="bench-compare-main">{period_label}</div>
            <div class="bench-compare-sub">{current_start.strftime("%d/%m/%Y")} t/m {current_end.strftime("%d/%m/%Y")}</div>
          </div>
          <div class="bench-compare-cell">
            <div class="bench-compare-key">Vorige referentie</div>
            <div class="bench-compare-main">{previous_start.strftime("%d/%m/%Y")} t/m {previous_end.strftime("%d/%m/%Y")}</div>
            <div class="bench-compare-sub">Wordt gebruikt om vooruit of achteruit te bepalen.</div>
          </div>
          <div class="bench-compare-cell">
            <div class="bench-compare-key">Focus metric</div>
            <div class="bench-compare-main">{METRIC_SPECS[focus_metric]['label']}</div>
            <div class="bench-compare-sub">Hoofdvergelijking in de charts en detailtabellen hieronder.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="bench-table-note">
          Huidige periode: {current_start.strftime("%d/%m/%Y")} t/m {current_end.strftime("%d/%m/%Y")} |
          Vorige periode: {previous_start.strftime("%d/%m/%Y")} t/m {previous_end.strftime("%d/%m/%Y")} |
          Runs count &gt;15.0 (#) ontbreekt in de huidige Summary-bron en telt daarom niet mee in de voortgangsscore.
        </div>
        """,
        unsafe_allow_html=True,
    )

    compare_cards = [
        ("Spelers in scope", str(total_players), f"{total_players} MVV-spelers met benchmark-koppeling in de gekozen periode"),
        ("Gekoppelde posities", str(matched_positions), f"{matched_positions} van {len(report_df)} benchmarkposities hebben actuele data"),
        ("Metrics vooruit", f"{improved_total}/{tracked_total or 0}", "Aantal vergelijkbare metrics die dichter bij de benchmark kwamen"),
        ("Posities vooruit", str(progressing_positions), "Posities met een hogere benchmarks score dan in de vorige periode"),
    ]
    render_stat_cards(compare_cards, columns_per_row=4)

    signal_cols = st.columns(3, gap="small")
    with signal_cols[0]:
        st.markdown(
            build_signal_card(
                "Beste match",
                best_player_name,
                best_player_note,
                "Vooruit",
            ),
            unsafe_allow_html=True,
        )
    with signal_cols[1]:
        st.markdown(
            build_signal_card(
                "Grootste watch",
                watch_player_name,
                watch_player_note,
                "Achteruit",
            ),
            unsafe_allow_html=True,
        )
    with signal_cols[2]:
        st.markdown(
            build_signal_card(
                "Spelers vooruit",
                f"{player_up_count}/{total_players or 0}",
                "Aantal individuele spelers dat in deze periode dichter bij de benchmark komt.",
                "Stabiel" if total_players == 0 else ("Vooruit" if player_up_count >= max(1, total_players / 2) else "Achteruit"),
            ),
            unsafe_allow_html=True,
        )

    st.markdown('<div class="bench-subsection">Positie-overzicht</div>', unsafe_allow_html=True)
    position_cards = []
    for _, row in report_df.sort_values("current_score", ascending=False, na_position="last").iterrows():
        if pd.isna(row.get(f"{focus_metric}_current")):
            continue
        position_cards.append(build_position_card(str(row["Positie"]), row, focus_metric))
    if position_cards:
        for start in range(0, len(position_cards), 4):
            row_cards = position_cards[start : start + 4]
            cols = st.columns(4, gap="small")
            for col, card_html in zip(cols, row_cards):
                with col:
                    st.markdown(card_html, unsafe_allow_html=True)

    chart_col, progress_col = st.columns([0.56, 0.44], gap="large")
    with chart_col:
        fig = build_metric_compare_chart(report_df, focus_metric, source_key, period_label)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})
    with progress_col:
        fig = build_progress_chart(report_df)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})

    matrix_col, focus_table_col = st.columns([0.48, 0.52], gap="large")
    with matrix_col:
        fig = build_status_heatmap(report_df)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})
    with focus_table_col:
        focus_table = build_focus_table(report_df, focus_metric)
        st.markdown(
            f"""
            <div class="bench-sheet-card">
              <div class="bench-sheet-kicker">Focus metric</div>
              <div class="bench-sheet-title">{METRIC_SPECS[focus_metric]['label']} per positie</div>
              <div class="bench-sheet-note">Huidige periode, vorige periode, benchmark en netto gap in dezelfde tabel.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(focus_table, width="stretch", hide_index=True)

    if not player_report_df.empty:
        st.markdown('<div class="bench-subsection">Spelers vs benchmark</div>', unsafe_allow_html=True)
        player_filter_cols = st.columns([0.44, 0.56], gap="small")
        position_options = ["Alle posities"] + sorted(player_report_df["Positie"].dropna().astype(str).unique().tolist())
        selected_position_filter = player_filter_cols[0].selectbox("Positiefilter spelers", options=position_options)
        player_sort_mode = player_filter_cols[1].selectbox(
            "Spelers sorteren op",
            options=["Grootste gap", "Beste bench-score", "Naam"],
        )
        filtered_player_df = player_report_df.copy()
        if selected_position_filter != "Alle posities":
            filtered_player_df = filtered_player_df.loc[filtered_player_df["Positie"] == selected_position_filter].copy()
        if player_sort_mode == "Beste bench-score":
            filtered_player_df = filtered_player_df.sort_values("current_score", ascending=False, na_position="last")
        elif player_sort_mode == "Naam":
            filtered_player_df = filtered_player_df.sort_values("player_name", ascending=True)
        else:
            filtered_player_df["focus_gap_abs"] = filtered_player_df[f"{focus_metric}_gap_current"].abs()
            filtered_player_df = filtered_player_df.sort_values("focus_gap_abs", ascending=False, na_position="last")

        player_chart_cols = st.columns([0.58, 0.42], gap="large")
        with player_chart_cols[0]:
            fig = build_player_compare_chart(filtered_player_df, focus_metric)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})
        with player_chart_cols[1]:
            fig = build_player_score_chart(filtered_player_df)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})

        st.markdown(
            """
            <div class="bench-sheet-card">
              <div class="bench-sheet-kicker">Speler detail</div>
              <div class="bench-sheet-title">Individuele benchmarkvergelijking</div>
              <div class="bench-sheet-note">Hier vergelijk je jouw spelers direct met hun positiebenchmark, niet alleen het teamgemiddelde per positie.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(build_player_focus_table(filtered_player_df, focus_metric), width="stretch", hide_index=True)

    detail_positions = report_df.loc[report_df[f"{focus_metric}_current"].notna(), "Positie"].astype(str).tolist()
    detail_cols = st.columns([0.42, 0.58], gap="large")
    with detail_cols[0]:
        selected_position = st.selectbox("Positie detail", options=detail_positions)
        position_row = report_df.loc[report_df["Positie"] == selected_position].iloc[0]
        st.markdown(
            build_stat_card(
                "Bench-score nu",
                _format_score(position_row.get("current_score")),
                f"Vorige periode: {_format_score(position_row.get('previous_score'))} | Trend: {position_row.get('overall_status', '--')}",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            build_stat_card(
                "Spelers in scope",
                str(int(position_row.get("active_players_current", 0))),
                f"Vorige periode: {int(position_row.get('active_players_previous', 0))} | Vergelijkbare metrics: {int(position_row.get('tracked_metrics', 0))}",
            ),
            unsafe_allow_html=True,
        )
    with detail_cols[1]:
        st.markdown(
            """
            <div class="bench-sheet-card">
              <div class="bench-sheet-kicker">Positie detail</div>
              <div class="bench-sheet-title">Alle metrics voor de geselecteerde positie</div>
              <div class="bench-sheet-note">Per metric zie je benchmark, actuele output, vorige referentie en of MVV daar dichter bij komt.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        detail_df = build_position_detail_table(report_df, selected_position)
        st.dataframe(detail_df, width="stretch", hide_index=True)


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

    if not BENCHMARKS_PDF.exists():
        st.error("De benchmark-PDF is niet gevonden in de assets.")
        render_sidebar_footer(profile)
        st.stop()

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
            Gebruik <strong>Marks</strong> voor de vaste referentietabellen en <strong>Compare</strong> voor een echte benchmarkrapportage op basis van de actuele Summary-data van MVV.
          </div>
          <div class="bench-pill-row">
            <span class="bench-pill">KKD 2024/2025</span>
            <span class="bench-pill">Eredivisie 2025/2026</span>
            <span class="bench-pill">Progressie: vooruit of achteruit</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info_col, download_col = st.columns([0.72, 0.28], gap="large")
    with info_col:
        marks_tab, compare_tab = st.tabs(["Marks", "Compare"])
        with marks_tab:
            render_marks_tab()
        with compare_tab:
            render_compare_tab(sb)

    with download_col:
        st.markdown(
            """
            <div class="bench-download-card">
              <div class="bench-download-label">Bronbestand</div>
              <div class="bench-download-title">Positional Benchmarks [MVV]</div>
              <div class="bench-download-copy">
                Download hier de originele PDF zoals die in de Benchmarks-pagina is opgenomen.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download PDF",
            data=load_pdf_bytes(str(BENCHMARKS_PDF)),
            file_name="Positional_Benchmarks_MVV.pdf",
            mime="application/pdf",
            width="stretch",
            key="benchmarks_download_pdf",
        )
        st.page_link("pages/10_Data_Page_Beta.py", label="Terug naar Data")

    render_sidebar_footer(profile)


if __name__ == "__main__":
    main()
