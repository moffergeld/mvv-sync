from __future__ import annotations

from html import escape
import re
from typing import Callable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

from auth_session import ensure_auth_restored, get_sb_client
from pages.Subscripts.gps_hybrid_source import load_hybrid_gps
from pages.Subscripts.mvv_branding import TEAM_HERO_BG, TEAM_LOGO, build_data_uri
from pages.Subscripts.week_session_labels import md_label as desktop_md_label
from pages.Subscripts.week_session_labels import session_time as desktop_session_time
from pages.Subscripts.week_session_labels import source_value as session_source_value
from roles import get_profile, is_staff_user, render_sidebar_footer, render_sidebar_navigation, require_auth
from speed_outlier_utils import sanitize_progressive_max_speed
from utils.streamlit_ui import apply_dashboard_polish, apply_streamlit_chrome


st.set_page_config(page_title="Weekoverzicht", layout="wide", initial_sidebar_state="expanded")
apply_streamlit_chrome()

PAGE_BG_URI = build_data_uri(TEAM_HERO_BG)
TEAM_LOGO_URI = build_data_uri(TEAM_LOGO)

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "").strip()

MVV_RED = "#C8102E"
MVV_RED_BRIGHT = "#EA3351"
MVV_RED_DEEP = "#6E1222"
MVV_TEXT = "#F8FAFC"
MVV_TEXT_SOFT = "rgba(248,250,252,0.76)"
MVV_TEXT_MUTED = "rgba(248,250,252,0.62)"
MVV_GRID = "rgba(255,255,255,0.10)"
MVV_PANEL_BG = "rgba(18, 25, 42, 0.92)"

GPS_SELECT_COLS = [
    "gps_id",
    "player_id",
    "player_name",
    "datum",
    "week",
    "year",
    "type",
    "event",
    "extra_metrics",
    "duration",
    "total_distance",
    "total_distance_zone_1_and_2",
    "total_distance_zone_3",
    "total_distance_zone_4",
    "total_distance_zone_5",
    "total_distance_zone_6",
    "number_of_sprints",
    "player_load_two_dimensional",
    "player_load_three_dimensional",
    "total_accelerations",
    "total_decelerations",
    "heart_rate_training_impulse",
    "high_metabolic_load_distance",
    "maximum_speed",
]

GPS_INDEX_SELECT_COLS = [
    "gps_id",
    "player_id",
    "player_name",
    "datum",
    "type",
    "total_distance",
    "total_distance_zone_5",
    "total_distance_zone_6",
    "number_of_sprints",
    "maximum_speed",
]

SUM_COLUMNS = [
    "duration",
    "total_distance",
    "total_distance_zone_1_and_2",
    "total_distance_zone_3",
    "total_distance_zone_4",
    "total_distance_zone_5",
    "total_distance_zone_6",
    "number_of_sprints",
    "player_load_two_dimensional",
    "player_load_three_dimensional",
    "total_accelerations",
    "total_decelerations",
    "heart_rate_training_impulse",
    "high_metabolic_load_distance",
]

INDEX_SUM_COLUMNS = [
    "total_distance",
    "total_distance_zone_5",
    "total_distance_zone_6",
    "number_of_sprints",
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
          background: __WEEK_REPORT_BG__;
          background-size: cover;
          background-position: center top;
          background-attachment: fixed;
        }

        .block-container {
          max-width: 1380px;
          padding-top: 1.25rem;
          padding-bottom: 2.4rem;
        }

        div[data-testid="stVerticalBlock"]:has(.week-report-hero-anchor) {
          padding: 1.75rem 1.6rem 1.3rem 1.6rem;
          border-radius: 10px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(135deg, rgba(18, 25, 42, 0.88), rgba(10, 15, 27, 0.84));
          box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22);
          margin-bottom: 1.1rem;
        }

        div[data-testid="stVerticalBlock"]:has(.week-report-panel-anchor) {
          padding: 1rem 1rem 0.8rem 1rem;
          border-radius: 10px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
          box-shadow: 0 14px 26px rgba(0, 0, 0, 0.18);
          margin-bottom: 1rem;
        }

        .week-report-hero-anchor,
        .week-report-panel-anchor {
          height: 0;
        }

        .week-report-head {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 1rem;
          margin-bottom: 1rem;
        }

        .week-report-logo {
          width: 78px;
          height: 78px;
          object-fit: contain;
          flex-shrink: 0;
          filter: drop-shadow(0 8px 22px rgba(0,0,0,0.28));
        }

        .week-report-copyhead {
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 0.12rem;
          text-align: left;
        }

        .week-report-kicker {
          color: rgba(255,255,255,0.76);
          font-size: 0.74rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.18em;
          margin-bottom: 0;
        }

        .week-report-title {
          margin: 0;
          font-size: 2.45rem;
          line-height: 1;
          font-weight: 800;
          color: #ffffff;
        }

        .week-report-copy {
          margin-top: 0.85rem;
          max-width: 78ch;
          color: rgba(255,255,255,0.84);
          line-height: 1.6;
        }

        .week-report-filter-label {
          color: rgba(255,255,255,0.92);
          font-size: 0.92rem;
          font-weight: 700;
          margin-bottom: 0.35rem;
        }

        .week-report-filter-note {
          color: rgba(255,255,255,0.80);
          font-size: 0.88rem;
          font-weight: 700;
          text-align: right;
          margin-top: 2rem;
        }

        .week-report-badge-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
          margin-top: 1rem;
        }

        .week-report-badge {
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

        [class*="st-key-week_report_back"] button,
        [class*="st-key-week_report_pdf_download"] button {
          min-height: 2.65rem !important;
          border-radius: 10px !important;
          border: 1px solid rgba(234, 51, 81, 0.22) !important;
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96)) !important;
          color: #ffffff !important;
          font-weight: 800 !important;
          box-shadow: 0 10px 22px rgba(0, 0, 0, 0.18) !important;
        }

        [class*="st-key-week_report_back"] button:hover,
        [class*="st-key-week_report_pdf_download"] button:hover {
          border-color: rgba(234, 51, 81, 0.36) !important;
          color: #ffffff !important;
        }

        .week-report-card-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 1rem;
          margin: 0.2rem 0 1.15rem 0;
        }

        .week-report-card {
          border-radius: 10px;
          border: 1px solid rgba(234, 51, 81, 0.14);
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
          box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
          padding: 1rem 1rem 0.9rem 1rem;
          min-height: 132px;
        }

        .week-report-card-label {
          color: rgba(255,255,255,0.62);
          font-size: 0.72rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }

        .week-report-card-value {
          margin-top: 0.55rem;
          color: #ffffff;
          font-size: 2rem;
          line-height: 1;
          font-weight: 800;
        }

        .week-report-card-foot {
          margin-top: 0.72rem;
          color: rgba(255,255,255,0.76);
          line-height: 1.45;
          font-size: 0.84rem;
        }

        .week-report-panel-title {
          color: #ffffff;
          font-size: 1.08rem;
          line-height: 1.2;
          font-weight: 800;
          margin-bottom: 0.2rem;
        }

        .week-report-panel-subtitle {
          color: rgba(255,255,255,0.70);
          font-size: 0.84rem;
          margin-bottom: 0.85rem;
        }

        .week-report-table-wrap {
          overflow-x: auto;
        }

        .week-report-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.88rem;
        }

        .week-report-table thead th {
          text-align: left;
          padding: 0.8rem 0.8rem;
          color: rgba(255,255,255,0.68);
          font-size: 0.73rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          border-bottom: 1px solid rgba(255,255,255,0.10);
        }

        .week-report-table tbody td {
          padding: 0.76rem 0.8rem;
          color: rgba(255,255,255,0.90);
          border-bottom: 1px solid rgba(255,255,255,0.06);
          white-space: nowrap;
        }

        .week-report-table tbody tr:last-child td {
          border-bottom: none;
        }

        .week-report-note-list {
          margin: 0.25rem 0 0 0;
          padding-left: 1.1rem;
          color: rgba(255,255,255,0.90);
          line-height: 1.65;
        }

        .week-report-note-foot {
          margin-top: 0.9rem;
          color: rgba(255,255,255,0.58);
          font-size: 0.82rem;
        }

        div[data-testid="stTabs"] button {
          background: rgba(255,255,255,0.03);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 999px;
          color: rgba(255,255,255,0.74);
          font-weight: 800;
          padding: 0.38rem 0.95rem;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
          color: #ffffff;
          border-color: rgba(234, 51, 81, 0.55);
          background: rgba(200, 16, 46, 0.22);
        }

        @media (max-width: 1100px) {
          .week-report-card-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
        }

        @media (max-width: 768px) {
          div[data-testid="stVerticalBlock"]:has(.week-report-hero-anchor) {
            padding: 1.35rem 1rem 1rem 1rem;
          }

          .week-report-head {
            flex-direction: column;
            gap: 0.8rem;
          }

          .week-report-copyhead {
            text-align: center;
          }

          .week-report-title {
            font-size: 2rem;
          }

          .week-report-card-grid {
            grid-template-columns: repeat(1, minmax(0, 1fr));
          }

          .week-report-filter-note {
            text-align: left;
            margin-top: 0.25rem;
          }
        }
        </style>
        """.replace("__WEEK_REPORT_BG__", background),
        unsafe_allow_html=True,
    )


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
    all_rows: list[dict] = []
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


def _safe_divide(numerator: pd.Series, denominator: pd.Series, multiplier: float = 1.0) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").astype(float)
    den = pd.to_numeric(denominator, errors="coerce").astype(float)
    den = den.where(den.ne(0), float("nan"))
    return num.div(den).mul(multiplier)


def _session_category(type_value: object) -> str:
    value = str(type_value or "").strip().lower()
    if "match" in value or "wedstrijd" in value:
        return "Match"
    return "Training"


SESSION_INDEX_PATTERN = re.compile(r"\((\d+)\)")


def _session_display(type_value: object) -> str:
    value = str(type_value or "").strip()
    return value or "Onbekende sessie"


def _session_index(type_value: object) -> int:
    value = _session_display(type_value)
    match = SESSION_INDEX_PATTERN.search(value)
    if match:
        return int(match.group(1))
    return 1


def _session_sort_value(type_value: object) -> tuple[int, int, str]:
    label = _session_display(type_value)
    category_rank = 1 if _session_category(label) == "Match" else 0
    return (category_rank, _session_index(label), label.lower())


def _session_short_code(type_value: object) -> str:
    label = _session_display(type_value)
    category = _session_category(label)
    index = _session_index(label)
    if category == "Match":
        return f"M{index}"
    return f"T{index}"


def _extra_metric_value(value: object, *keys: str) -> str:
    return session_source_value(value, *keys)


def _session_context(row: pd.Series) -> dict:
    extra = row.get("extra_metrics")
    context = dict(extra) if isinstance(extra, dict) else {}
    direct_fields = {
        "Session ID": row.get("session_id"),
        "Session Title": row.get("session_title"),
        "Session Type": row.get("session_type"),
        "Session Start Time": row.get("session_start_time"),
    }
    for key, value in direct_fields.items():
        if value is not None and not pd.isna(value) and str(value).strip():
            context[key] = value

    raw = context.get("STATSports Raw")
    if isinstance(raw, dict):
        session = raw.get("session") if isinstance(raw.get("session"), dict) else {}
        details = session.get("sessionDetails") if isinstance(session.get("sessionDetails"), dict) else {}
        player = raw.get("player") if isinstance(raw.get("player"), dict) else {}
        player_details = player.get("playerDetails") if isinstance(player.get("playerDetails"), dict) else {}
        raw_fields = {
            "Session ID": session.get("id"),
            "Session Title": session.get("sessionName") or session.get("name"),
            "Session Type": details.get("sessionType"),
            "Session Start Time": details.get("startTime"),
            "Player Position": player_details.get("primaryPosition"),
        }
        for key, value in raw_fields.items():
            if key not in context and value is not None and str(value).strip():
                context[key] = value
    return context


def _row_source_value(row: pd.Series, direct_column: str, *keys: str) -> str:
    direct_value = row.get(direct_column)
    if direct_value is not None and not pd.isna(direct_value) and str(direct_value).strip():
        return str(direct_value).strip()
    return _extra_metric_value(_session_context(row), *keys)


def _session_identity(row: pd.Series) -> str:
    session_id = _row_source_value(row, "session_id", "Session ID", "sessionId")
    start = _row_source_value(row, "session_start_time", "Session Start Time", "sessionStartTime")
    title = _row_source_value(row, "session_title", "Session Title", "sessionTitle")
    session_type = str(row.get("type") or "Sessie").strip()
    return session_id or "|".join((session_type, start, title))


def _session_label(row: pd.Series) -> str:
    start = _row_source_value(row, "session_start_time", "Session Start Time", "sessionStartTime")
    title = _row_source_value(row, "session_title", "Session Title", "sessionTitle")
    session_type = _session_display(row.get("type"))
    time_label = ""
    if "T" in start:
        time_label = start.split("T", 1)[1][:5]
    elif len(start) >= 5:
        time_label = start[:5]
    pieces = [session_type]
    if time_label:
        pieces.append(time_label)
    if title and title.lower() not in session_type.lower():
        pieces.append(title)
    return " · ".join(pieces)


def _weekday_label(value: object) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "--"
    weekdays = ["Ma", "Di", "Wo", "Do", "Vr", "Za", "Zo"]
    return f"{weekdays[int(ts.weekday())]} {ts:%d/%m}"


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


def _format_speed(value: object) -> str:
    base = _format_decimal(value, 1)
    return "--" if base == "--" else f"{base} km/h"


def _format_signed_pct(value: object) -> str:
    if pd.isna(value):
        return "--"
    prefix = "+" if float(value) >= 0 else ""
    return f"{prefix}{_format_decimal(value, 1)}%"


def _week_pdf_filename(week_start: pd.Timestamp) -> str:
    iso = week_start.isocalendar()
    return f"week_report_{iso.year}_W{int(iso.week):02d}.pdf"


def _report_file_name(base_name: str, report_style: str, report_revision: str | None = None) -> str:
    stem = base_name[:-4] if base_name.lower().endswith(".pdf") else base_name
    normalized_style = str(report_style or "").strip().lower()
    style_slug = "html" if normalized_style == "html" or "nieuw" in normalized_style else "report"
    revision_slug = (
        str(report_revision or "")
        .strip()
        .lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("_", "-")
    )
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    if revision_slug:
        return f"{stem}_{style_slug}_{revision_slug}_{stamp}.pdf"
    return f"{stem}_{style_slug}_{stamp}.pdf"


def _prepare_summary_index_df(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    df = raw.copy()
    df["player_id"] = df["player_id"].fillna("").astype(str)
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce").dt.normalize()
    df["player_name"] = df["player_name"].fillna("Onbekend").astype(str).str.strip()
    df["type"] = df["type"].fillna("").astype(str).str.strip()

    for column in INDEX_SUM_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df["maximum_speed"] = pd.to_numeric(df["maximum_speed"], errors="coerce")
    df = df.dropna(subset=["datum"]).copy()
    df["maximum_speed"] = sanitize_progressive_max_speed(df, group_cols=["player_id", "player_name"], order_cols=["gps_id"])
    df["hsr_hsd"] = df["total_distance_zone_5"].fillna(0.0) + df["total_distance_zone_6"].fillna(0.0)
    df["session_category"] = df["type"].apply(_session_category)
    df["week_start"] = (df["datum"] - pd.to_timedelta(df["datum"].dt.weekday, unit="D")).dt.normalize()
    season_max_speed = df.groupby("player_id")["maximum_speed"].transform("max")
    df["speed_exposure_flag"] = season_max_speed.gt(0) & df["maximum_speed"].ge(season_max_speed * 0.9)
    return df


def _prepare_summary_period_df(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    df = raw.copy()
    df["player_id"] = df["player_id"].fillna("").astype(str)
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce").dt.normalize()
    df["player_name"] = df["player_name"].fillna("Onbekend").astype(str).str.strip()
    df["type"] = df["type"].fillna("").astype(str).str.strip()
    df["event"] = df["event"].fillna("").astype(str).str.strip()
    if "extra_metrics" not in df.columns:
        df["extra_metrics"] = None

    for column in SUM_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df["maximum_speed"] = pd.to_numeric(df["maximum_speed"], errors="coerce")
    df = df.dropna(subset=["datum"]).copy()
    df["hsr_hsd"] = df["total_distance_zone_5"].fillna(0.0) + df["total_distance_zone_6"].fillna(0.0)
    df["session_category"] = df["type"].apply(_session_category)
    df["session_key"] = df.apply(_session_identity, axis=1)
    df["session_label"] = df.apply(_session_label, axis=1)
    df["md_label"] = df.apply(lambda row: desktop_md_label(_session_context(row), row.get("type")), axis=1)
    df["session_time"] = df.apply(lambda row: desktop_session_time(_session_context(row)), axis=1)
    df["week_start"] = (df["datum"] - pd.to_timedelta(df["datum"].dt.weekday, unit="D")).dt.normalize()
    return df


def _merge_summary_context(scope_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    if scope_df.empty or history_df.empty:
        return scope_df

    context_df = (
        history_df[["gps_id", "maximum_speed", "speed_exposure_flag"]]
        .drop_duplicates(subset=["gps_id"])
        .rename(columns={"maximum_speed": "history_max_speed"})
    )
    merged = scope_df.merge(context_df, on="gps_id", how="left")
    merged["maximum_speed"] = merged["history_max_speed"].combine_first(merged["maximum_speed"])
    merged["speed_exposure_flag"] = merged["speed_exposure_flag"].fillna(False).astype(bool)
    return merged.drop(columns=["history_max_speed"])


@st.cache_data(show_spinner=False, ttl=180)
def fetch_summary_index_cached(access_token: str) -> pd.DataFrame:
    raw = load_hybrid_gps(
        access_token,
        GPS_INDEX_SELECT_COLS,
        event="Summary",
    )
    return _prepare_summary_index_df(raw)


@st.cache_data(show_spinner=False, ttl=180)
def fetch_summary_period_cached(access_token: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    raw = load_hybrid_gps(
        access_token,
        GPS_SELECT_COLS,
        start=start_iso,
        end=end_iso,
        event="Summary",
    )
    return _prepare_summary_period_df(raw)


@st.cache_data(show_spinner=False, ttl=300)
def fetch_player_positions_cached(_sb, access_scope: str) -> dict[str, str]:
    rows: list[dict] = []
    for select_clause in ("player_id,position", 'player_id,"Position"', "player_id"):
        try:
            rows = _sb.table("players").select(select_clause).execute().data or []
            break
        except Exception:
            rows = []

    positions: dict[str, str] = {}
    for row in rows:
        player_id = str(row.get("player_id") or "").strip()
        position = row.get("Position") if row.get("Position") is not None else row.get("position")
        if player_id:
            positions[player_id] = str(position or "").strip()
    return positions


def exclude_goalkeepers(df: pd.DataFrame, positions: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return df
    player_positions = df["player_id"].fillna("").astype(str).map(positions).fillna("")
    if "extra_metrics" in df.columns:
        source_positions = df.apply(
            lambda row: session_source_value(
                _session_context(row),
                "Player Position",
                "playerPrimaryPosition",
                "primaryPosition",
            ),
            axis=1,
        )
        player_positions = player_positions.where(player_positions.astype(str).str.strip().ne(""), source_positions)
    def is_goalkeeper(value: object) -> bool:
        text = str(value or "").strip().lower()
        normalized = re.sub(r"[^a-z]", "", text)
        return bool(
            re.search(r"(?:^|[^a-z])gk(?:$|[^a-z])", text)
            or any(label in normalized for label in ("goalkeeper", "keeper", "doelman", "goalie"))
        )

    return df.loc[~player_positions.map(is_goalkeeper)].copy()


def _week_moment_axis(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Group all sessions from one date under one shared day/date axis label."""

    weekdays = ("Ma", "Di", "Wo", "Do", "Vr", "Za", "Zo")
    day_counts: dict[object, int] = {}
    day_seen: dict[object, int] = {}
    for record in records:
        day = record.get("datum")
        day_counts[day] = day_counts.get(day, 0) + 1

    labels: list[dict[str, object]] = []
    for record in records:
        day = record.get("datum")
        day_seen[day] = day_seen.get(day, 0) + 1
        date_label = day.strftime("%d/%m") if hasattr(day, "strftime") else str(day)
        weekday = weekdays[day.weekday()] if hasattr(day, "weekday") else "Dag"
        md_value = str(record.get("md_label") or "MD onbekend")
        time_value = str(record.get("session_time") or "")
        labels.append(
            {
                "events_in_day": day_counts[day],
                "moment_index": day_seen[day],
                "group_label": f"{weekday} · {date_label}",
                "moment_label": f"{md_value} · {time_value}" if time_value else md_value,
            }
        )
    return labels


def _week_label(week_start: pd.Timestamp) -> str:
    iso = week_start.isocalendar()
    week_end = week_start + pd.Timedelta(days=6)
    return f"{iso.year}-W{int(iso.week):02d} | {week_start:%d/%m/%Y} - {week_end:%d/%m/%Y}"


def build_week_history(all_df: pd.DataFrame) -> pd.DataFrame:
    if all_df.empty:
        return pd.DataFrame()
    history = (
        all_df.groupby("week_start", dropna=False)
        .agg(
            active_players=("player_name", "nunique"),
            player_sessions=("datum", "size"),
            total_distance=("total_distance", "sum"),
            hsr_hsd=("hsr_hsd", "sum"),
            sprints=("number_of_sprints", "sum"),
        )
        .reset_index()
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    history["week_label"] = history["week_start"].apply(_week_label)
    history["td_rolling4_prev"] = history["total_distance"].shift(1).rolling(4, min_periods=1).mean()
    history["hsr_rolling4_prev"] = history["hsr_hsd"].shift(1).rolling(4, min_periods=1).mean()
    return history


def build_week_player_table(week_df: pd.DataFrame) -> pd.DataFrame:
    if week_df.empty:
        return pd.DataFrame()
    grouped = (
        week_df.groupby("player_name", dropna=False)
        .agg(
            sessions=("datum", "size"),
            total_distance=("total_distance", "sum"),
            high_metabolic_load_distance=("high_metabolic_load_distance", "sum"),
            total_distance_zone_5=("total_distance_zone_5", "sum"),
            total_distance_zone_6=("total_distance_zone_6", "sum"),
            hsr_hsd=("hsr_hsd", "sum"),
            sprints=("number_of_sprints", "sum"),
            total_accelerations=("total_accelerations", "sum"),
            total_decelerations=("total_decelerations", "sum"),
            player_load_two_dimensional=("player_load_two_dimensional", "sum"),
            maximum_speed=("maximum_speed", "max"),
            duration=("duration", "sum"),
        )
        .reset_index()
        .sort_values("total_distance", ascending=False)
        .reset_index(drop=True)
    )
    grouped["distance_per_minute"] = _safe_divide(grouped["total_distance"], grouped["duration"])
    return grouped


def build_week_day_table(week_df: pd.DataFrame) -> pd.DataFrame:
    if week_df.empty:
        return pd.DataFrame()
    grouped = (
        week_df.groupby("datum", dropna=False)
        .agg(
            active_players=("player_name", "nunique"),
            player_sessions=("datum", "size"),
            total_distance=("total_distance", "sum"),
            hsr_hsd=("hsr_hsd", "sum"),
            sprints=("number_of_sprints", "sum"),
            speed_exposures=("speed_exposure_flag", "sum"),
            total_accelerations=("total_accelerations", "sum"),
            total_decelerations=("total_decelerations", "sum"),
            maximum_speed=("maximum_speed", "max"),
            duration=("duration", "sum"),
        )
        .reset_index()
        .sort_values("datum")
        .reset_index(drop=True)
    )
    grouped["label"] = grouped["datum"].dt.strftime("%d/%m")
    grouped["distance_per_player"] = _safe_divide(grouped["total_distance"], grouped["active_players"])
    return grouped


def build_week_session_table(week_df: pd.DataFrame) -> pd.DataFrame:
    if week_df.empty:
        return pd.DataFrame()
    grouped = (
        week_df.groupby(["datum", "type", "session_category", "session_key", "session_label"], dropna=False)
        .agg(
            active_players=("player_name", "nunique"),
            player_sessions=("datum", "size"),
            total_distance=("total_distance", "sum"),
            hsr_hsd=("hsr_hsd", "sum"),
            sprints=("number_of_sprints", "sum"),
            speed_exposures=("speed_exposure_flag", "sum"),
            total_accelerations=("total_accelerations", "sum"),
            total_decelerations=("total_decelerations", "sum"),
            maximum_speed=("maximum_speed", "max"),
            duration=("duration", "sum"),
        )
        .reset_index()
    )
    grouped["session_display"] = grouped["type"].apply(_session_display)
    grouped["session_sort"] = grouped["type"].apply(_session_sort_value)
    grouped = grouped.sort_values(["datum", "session_sort", "session_display"]).reset_index(drop=True)
    grouped["event_index"] = grouped.groupby("datum").cumcount() + 1
    grouped["label"] = grouped["datum"].dt.strftime("%d/%m")
    grouped["day_label"] = grouped["datum"].apply(_weekday_label)
    grouped["session_code"] = grouped["type"].apply(_session_short_code)
    grouped["event_group"] = grouped["session_category"].fillna("Training").astype(str)
    grouped["events_in_day"] = grouped.groupby("datum")["datum"].transform("size")
    grouped["session_code_display"] = grouped.apply(
        lambda row: row["session_code"] if int(row.get("events_in_day", 1) or 1) > 1 else ("M" if row["event_group"] == "Match" else "T"),
        axis=1,
    )
    grouped["distance_per_player"] = _safe_divide(grouped["total_distance"], grouped["active_players"])
    return grouped


def build_week_zone_day_table(week_df: pd.DataFrame) -> pd.DataFrame:
    if week_df.empty:
        return pd.DataFrame()
    weekdays = ["Ma", "Di", "Wo", "Do", "Vr", "Za", "Zo"]
    grouped = (
        week_df.groupby("datum", dropna=False)
        .agg(
            total_distance_zone_1_and_2=("total_distance_zone_1_and_2", "sum"),
            total_distance_zone_3=("total_distance_zone_3", "sum"),
            total_distance_zone_4=("total_distance_zone_4", "sum"),
            total_distance_zone_5=("total_distance_zone_5", "sum"),
            total_distance_zone_6=("total_distance_zone_6", "sum"),
        )
        .reset_index()
        .sort_values("datum")
        .reset_index(drop=True)
    )
    grouped["label"] = grouped["datum"].apply(
        lambda value: f"{weekdays[int(value.weekday())]} {value:%d/%m}" if pd.notna(value) else "--"
    )
    return grouped


def build_week_zone_session_table(week_df: pd.DataFrame) -> pd.DataFrame:
    if week_df.empty:
        return pd.DataFrame()
    grouped = (
        week_df.groupby(["datum", "type", "session_category", "session_key", "session_label"], dropna=False)
        .agg(
            total_distance_zone_1_and_2=("total_distance_zone_1_and_2", "sum"),
            total_distance_zone_3=("total_distance_zone_3", "sum"),
            total_distance_zone_4=("total_distance_zone_4", "sum"),
            total_distance_zone_5=("total_distance_zone_5", "sum"),
            total_distance_zone_6=("total_distance_zone_6", "sum"),
        )
        .reset_index()
    )
    grouped["session_display"] = grouped["type"].apply(_session_display)
    grouped["session_sort"] = grouped["type"].apply(_session_sort_value)
    grouped = grouped.sort_values(["datum", "session_sort", "session_display"]).reset_index(drop=True)
    grouped["event_index"] = grouped.groupby("datum").cumcount() + 1
    grouped["label"] = grouped["datum"].apply(_weekday_label)
    grouped["session_code"] = grouped["type"].apply(_session_short_code)
    grouped["event_group"] = grouped["session_category"].fillna("Training").astype(str)
    grouped["events_in_day"] = grouped.groupby("datum")["datum"].transform("size")
    grouped["session_code_display"] = grouped.apply(
        lambda row: row["session_code"] if int(row.get("events_in_day", 1) or 1) > 1 else ("M" if row["event_group"] == "Match" else "T"),
        axis=1,
    )
    return grouped


def build_week_day_stats(week_df: pd.DataFrame) -> pd.DataFrame:
    if week_df.empty:
        return pd.DataFrame()
    player_day = (
        week_df.groupby(["datum", "player_name"], dropna=False)
        .agg(
            total_distance=("total_distance", "sum"),
            hsr_hsd=("hsr_hsd", "sum"),
            sprints=("number_of_sprints", "sum"),
            total_accelerations=("total_accelerations", "sum"),
            total_decelerations=("total_decelerations", "sum"),
            duration=("duration", "sum"),
        )
        .reset_index()
    )
    player_day["distance_per_minute"] = _safe_divide(player_day["total_distance"], player_day["duration"])
    for metric in ["total_distance", "hsr_hsd", "sprints", "total_accelerations", "total_decelerations", "distance_per_minute"]:
        player_day[metric] = pd.to_numeric(player_day[metric], errors="coerce").astype(float)

    grouped = player_day.groupby("datum", dropna=False).agg(player_count=("player_name", "nunique")).reset_index()
    for metric in ["total_distance", "hsr_hsd", "sprints", "total_accelerations", "total_decelerations", "distance_per_minute"]:
        stats = (
            player_day.groupby("datum", dropna=False)[metric]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
        )
        grouped = grouped.merge(stats, on="datum", how="left")

    grouped["label"] = grouped["datum"].dt.strftime("%d/%m")
    return grouped.sort_values("datum").reset_index(drop=True)


def build_week_session_stats(week_df: pd.DataFrame) -> pd.DataFrame:
    if week_df.empty:
        return pd.DataFrame()
    player_session = (
        week_df.groupby(
            ["datum", "type", "session_category", "session_key", "session_label", "md_label", "session_time", "player_name"],
            dropna=False,
        )
        .agg(
            total_distance=("total_distance", "sum"),
            total_distance_zone_5=("total_distance_zone_5", "sum"),
            total_distance_zone_6=("total_distance_zone_6", "sum"),
            high_metabolic_load_distance=("high_metabolic_load_distance", "sum"),
            hsr_hsd=("hsr_hsd", "sum"),
            sprints=("number_of_sprints", "sum"),
            total_accelerations=("total_accelerations", "sum"),
            total_decelerations=("total_decelerations", "sum"),
            duration=("duration", "sum"),
        )
        .reset_index()
    )
    player_session["distance_per_minute"] = _safe_divide(player_session["total_distance"], player_session["duration"])
    metric_columns = [
        "total_distance",
        "total_distance_zone_5",
        "total_distance_zone_6",
        "high_metabolic_load_distance",
        "hsr_hsd",
        "sprints",
        "total_accelerations",
        "total_decelerations",
        "distance_per_minute",
    ]
    for metric in metric_columns:
        player_session[metric] = pd.to_numeric(player_session[metric], errors="coerce").astype(float)

    grouped = (
        player_session.groupby(
            ["datum", "type", "session_category", "session_key", "session_label", "md_label", "session_time"],
            dropna=False,
        )
        .agg(player_count=("player_name", "nunique"))
        .reset_index()
    )
    for metric in metric_columns:
        stats = (
            player_session.groupby(
                ["datum", "type", "session_category", "session_key", "session_label", "md_label", "session_time"],
                dropna=False,
            )[metric]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
        )
        grouped = grouped.merge(
            stats,
            on=["datum", "type", "session_category", "session_key", "session_label", "md_label", "session_time"],
            how="left",
        )

    grouped["session_display"] = grouped["type"].apply(_session_display)
    grouped["session_sort"] = grouped["type"].apply(_session_sort_value)
    grouped["session_time_sort"] = grouped["session_time"].replace("", "99:99")
    grouped = grouped.sort_values(["datum", "session_time_sort", "session_sort", "session_display"]).reset_index(drop=True)
    grouped["event_index"] = grouped.groupby("datum").cumcount() + 1
    grouped["label"] = grouped["datum"].dt.strftime("%d/%m")
    grouped["day_label"] = grouped["datum"].apply(_weekday_label)
    grouped["session_code"] = grouped["type"].apply(_session_short_code)
    grouped["event_group"] = grouped["session_category"].fillna("Training").astype(str)
    axis_rows = _week_moment_axis(grouped.to_dict("records"))
    for column in ("events_in_day", "moment_index", "group_label", "moment_label"):
        grouped[column] = [row[column] for row in axis_rows]
    grouped["session_code_display"] = grouped.apply(
        lambda row: row["session_code"] if int(row.get("events_in_day", 1) or 1) > 1 else ("M" if row["event_group"] == "Match" else "T"),
        axis=1,
    )
    grouped["label"] = grouped["session_label"]
    return grouped.drop(columns=["session_time_sort"])


def build_week_type_table(week_df: pd.DataFrame) -> pd.DataFrame:
    if week_df.empty:
        return pd.DataFrame()
    grouped = (
        week_df.groupby("session_category", dropna=False)
        .agg(
            player_sessions=("datum", "size"),
            active_players=("player_name", "nunique"),
            total_distance=("total_distance", "sum"),
            hsr_hsd=("hsr_hsd", "sum"),
            sprints=("number_of_sprints", "sum"),
            maximum_speed=("maximum_speed", "max"),
        )
        .reset_index()
    )
    order = pd.Categorical(grouped["session_category"], categories=["Training", "Match"], ordered=True)
    grouped["session_category"] = order
    grouped = grouped.sort_values("session_category").reset_index(drop=True)
    grouped["session_category"] = grouped["session_category"].astype(str)
    return grouped


def build_week_notes(summary: dict[str, object], day_table: pd.DataFrame, player_table: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    week_start = summary["week_start"]
    notes.append(
        f"Week {week_start:%d/%m/%Y}: {_format_int(summary['active_players'])} actieve GPS-spelers en {_format_int(summary['player_sessions'])} player-sessies."
    )
    notes.append(
        f"Teamload: {_format_int(summary['total_distance'])} m total distance, {_format_int(summary['hsr_hsd'])} m HSR en {_format_int(summary['sprints'])} sprints."
    )
    if not day_table.empty:
        peak_day = day_table.sort_values("total_distance", ascending=False).iloc[0]
        notes.append(
            f"Hoogste dag binnen deze week: {peak_day['datum']:%d/%m/%Y} met {_format_int(peak_day['total_distance'])} m."
        )
    if not player_table.empty:
        top_player = player_table.sort_values("total_distance", ascending=False).iloc[0]
        notes.append(
            f"Hoogste individuele volume: {top_player['player_name']} met {_format_int(top_player['total_distance'])} m."
        )
    if not pd.isna(summary["speed_exposures"]):
        notes.append(
            f"Speed exposure: {_format_int(summary['speed_exposures'])} spelersessies bereikten >=90% van de individuele seizoensmax."
        )
    return notes


def build_week_summary(week_df: pd.DataFrame, history_row: pd.Series | None) -> dict[str, object]:
    active_players = week_df["player_name"].nunique() if not week_df.empty else 0
    player_sessions = len(week_df.index)
    total_distance = float(week_df["total_distance"].sum()) if not week_df.empty else 0.0
    hsr_hsd = float(week_df["hsr_hsd"].sum()) if not week_df.empty else 0.0
    sprints = float(week_df["number_of_sprints"].sum()) if not week_df.empty else 0.0
    top_speed = float(week_df["maximum_speed"].max()) if not week_df["maximum_speed"].dropna().empty else float("nan")
    speed_exposures = float(week_df["speed_exposure_flag"].sum()) if not week_df.empty else 0.0
    match_sessions = int((week_df["session_category"] == "Match").sum()) if not week_df.empty else 0
    training_sessions = int((week_df["session_category"] == "Training").sum()) if not week_df.empty else 0
    active_days = int(week_df["datum"].nunique()) if not week_df.empty else 0
    dist_per_player = total_distance / active_players if active_players else float("nan")

    td_vs_prev = float("nan")
    hsr_vs_prev = float("nan")
    if history_row is not None:
        td_base = history_row.get("td_rolling4_prev")
        hsr_base = history_row.get("hsr_rolling4_prev")
        if pd.notna(td_base) and float(td_base) != 0:
            td_vs_prev = ((total_distance - float(td_base)) / float(td_base)) * 100
        if pd.notna(hsr_base) and float(hsr_base) != 0:
            hsr_vs_prev = ((hsr_hsd - float(hsr_base)) / float(hsr_base)) * 100

    week_start = week_df["week_start"].iloc[0] if not week_df.empty else pd.Timestamp.today().normalize()
    return {
        "week_start": week_start,
        "week_end": week_start + pd.Timedelta(days=6),
        "active_players": active_players,
        "player_sessions": player_sessions,
        "total_distance": total_distance,
        "hsr_hsd": hsr_hsd,
        "sprints": sprints,
        "top_speed": top_speed,
        "speed_exposures": speed_exposures,
        "dist_per_player": dist_per_player,
        "match_sessions": match_sessions,
        "training_sessions": training_sessions,
        "active_days": active_days,
        "td_vs_prev": td_vs_prev,
        "hsr_vs_prev": hsr_vs_prev,
    }


def base_figure(title: str, height: int = 330) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=20, color=MVV_TEXT)),
        height=height,
        margin=dict(l=18, r=18, t=56, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color=MVV_TEXT, size=12),
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(color=MVV_TEXT_SOFT))
    fig.update_yaxes(gridcolor=MVV_GRID, zeroline=False, tickfont=dict(color=MVV_TEXT_SOFT))
    return fig


def build_daily_bar_chart(
    day_table: pd.DataFrame,
    column: str,
    title: str,
    color: str,
    value_formatter: Callable[[object], str],
    hover_format: str = ":,.0f",
    y_range: tuple[float, float] | None = None,
    error_column: str | None = None,
) -> go.Figure:
    fig = base_figure(title, height=350)
    if day_table.empty or column not in day_table.columns:
        return fig
    error_values = None
    if error_column and error_column in day_table.columns:
        error_values = day_table[error_column].fillna(0)
    fig.add_trace(
        go.Bar(
            x=day_table["label"],
            y=day_table[column],
            marker_color=color,
            error_y=dict(type="data", array=error_values, color=MVV_RED_BRIGHT, thickness=1.4) if error_values is not None else None,
            text=[value_formatter(value) for value in day_table[column]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"%{{x}}<br>%{{y{hover_format}}}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    return fig


def build_weekly_player_load_chart(player_table: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.13,
        row_heights=[0.58, 0.42],
        subplot_titles=("Totale afstand", "Intensieve afstand"),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )
    required = {
        "player_name",
        "total_distance",
        "high_metabolic_load_distance",
        "total_distance_zone_5",
        "total_distance_zone_6",
    }
    if player_table.empty or not required.issubset(player_table.columns):
        fig.update_layout(height=540, paper_bgcolor="rgba(0,0,0,0)")
        return fig

    data = player_table.sort_values("total_distance", ascending=False).reset_index(drop=True)
    players = data["player_name"].fillna("Onbekend").astype(str)
    total_distance = pd.to_numeric(data["total_distance"], errors="coerce").fillna(0)
    hmld = pd.to_numeric(data["high_metabolic_load_distance"], errors="coerce").fillna(0)
    fig.add_trace(
        go.Bar(
            name="Total Distance",
            x=players,
            y=total_distance,
            marker_color="#526986",
            text=[_format_distance(value) for value in total_distance],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>Total Distance: %{y:,.0f} m<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name="HMLD",
            x=players,
            y=hmld,
            mode="lines+markers",
            line=dict(color="#69D5CB", width=3),
            marker=dict(size=7, symbol="circle", line=dict(color="#101827", width=1.5)),
            hovertemplate="<b>%{x}</b><br>HMLD: %{y:,.0f} m<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    for column, name, color in (
        ("total_distance_zone_5", "Zone 5", "#EDB45B"),
        ("total_distance_zone_6", "Zone 6", "#E84664"),
    ):
        values = pd.to_numeric(data[column], errors="coerce").fillna(0)
        fig.add_trace(
            go.Bar(
                name=name,
                x=players,
                y=values,
                marker=dict(color=color, line=dict(color="#101827", width=1)),
                offsetgroup=name,
                hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:,.0f}} m<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=56, b=70),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.012)",
        font=dict(color=MVV_TEXT, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
        barmode="group",
        bargap=0.34,
        bargroupgap=0.1,
    )
    fig.update_annotations(font=dict(color=MVV_TEXT_SOFT, size=12), xanchor="left", x=0)
    fig.update_xaxes(showgrid=False, tickfont=dict(color=MVV_TEXT_SOFT), tickangle=-35, automargin=True)
    fig.update_yaxes(gridcolor=MVV_GRID, zeroline=False, tickfont=dict(color=MVV_TEXT_SOFT))
    fig.update_yaxes(title_text="Meters", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="HMLD (m)", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="Meters", row=2, col=1)
    return fig


def build_session_load_chart(session_stats: pd.DataFrame) -> go.Figure:
    """Team average per session with volume and intensity on honest separate scales."""

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.13,
        row_heights=[0.56, 0.44],
        subplot_titles=("Totale afstand", "Intensieve afstand"),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )
    required = {
        "label",
        "total_distance_mean",
        "total_distance_zone_5_mean",
        "total_distance_zone_6_mean",
        "high_metabolic_load_distance_mean",
    }
    if session_stats.empty or not required.issubset(session_stats.columns):
        fig.update_layout(height=460, paper_bgcolor="rgba(0,0,0,0)")
        return fig

    labels = [session_stats["group_label"].tolist(), session_stats["moment_label"].tolist()]
    fig.add_trace(
        go.Bar(
            name="Total Distance",
            x=labels,
            y=session_stats["total_distance_mean"],
            marker_color="#526986",
            text=[_format_distance(value) for value in session_stats["total_distance_mean"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>TD %{y:,.0f} m<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name="HMLD",
            x=labels,
            y=session_stats["high_metabolic_load_distance_mean"],
            mode="lines+markers",
            line=dict(color="#69D5CB", width=3),
            marker=dict(size=7, symbol="circle", line=dict(color="#101827", width=1.5)),
            hovertemplate="%{x}<br>HMLD %{y:,.0f} m<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    for column, name, color in (
        ("total_distance_zone_5_mean", "Zone 5", "#EDB45B"),
        ("total_distance_zone_6_mean", "Zone 6", "#E84664"),
    ):
        fig.add_trace(
            go.Bar(
                name=name,
                x=labels,
                y=session_stats[column],
                marker=dict(color=color, line=dict(color="#101827", width=1)),
                offsetgroup=name,
                hovertemplate=f"%{{x}}<br>{name} %{{y:,.0f}} m<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        height=510,
        margin=dict(l=20, r=20, t=56, b=42),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.012)",
        font=dict(color=MVV_TEXT, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
        barmode="group",
        bargap=0.42,
        bargroupgap=0.12,
    )
    fig.update_annotations(font=dict(color=MVV_TEXT_SOFT, size=12), xanchor="left", x=0)
    fig.update_xaxes(showgrid=False, tickfont=dict(color=MVV_TEXT_SOFT), tickangle=0, automargin=True)
    fig.update_yaxes(gridcolor=MVV_GRID, zeroline=False, tickfont=dict(color=MVV_TEXT_SOFT))
    fig.update_yaxes(title_text="Meters", row=1, col=1)
    fig.update_yaxes(title_text="HMLD (m)", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="Meters", row=2, col=1)
    return fig


def build_error_bar_chart(day_stats: pd.DataFrame, mean_column: str, std_column: str, title: str, color: str, value_formatter: Callable[[object], str]) -> go.Figure:
    fig = base_figure(title, height=350)
    if day_stats.empty or mean_column not in day_stats.columns:
        return fig
    means = day_stats[mean_column].fillna(0)
    stds = day_stats[std_column].fillna(0)
    fig.add_trace(
        go.Bar(
            x=day_stats["label"],
            y=means,
            marker_color=color,
            error_y=dict(type="data", array=stds, color=MVV_RED_BRIGHT, thickness=1.4),
            text=[value_formatter(value) for value in means],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>%{y:,.1f}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    return fig


def build_grouped_error_chart(day_stats: pd.DataFrame) -> go.Figure:
    fig = base_figure("Daily Player Average Accelerations / Decelerations +/- SD", height=350)
    if day_stats.empty:
        return fig
    accelerations = day_stats["total_accelerations_mean"].fillna(0)
    decelerations = day_stats["total_decelerations_mean"].fillna(0)
    accel_std = day_stats["total_accelerations_std"].fillna(0)
    decel_std = day_stats["total_decelerations_std"].fillna(0)
    fig.add_trace(
        go.Bar(
            name="Accelerations",
            x=day_stats["label"],
            y=accelerations,
            marker_color=MVV_RED_BRIGHT,
            error_y=dict(type="data", array=accel_std, color=MVV_TEXT_SOFT, thickness=1.3),
            text=[_format_int(value) for value in accelerations],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Decelerations",
            x=day_stats["label"],
            y=decelerations,
            marker_color=MVV_RED_DEEP,
            error_y=dict(type="data", array=decel_std, color=MVV_TEXT_SOFT, thickness=1.3),
            text=[_format_int(value) for value in decelerations],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(barmode="group")
    return fig


def build_leaderboard_chart(player_table: pd.DataFrame, column: str, title: str, value_formatter: Callable[[object], str]) -> go.Figure:
    fig = base_figure(title, height=360)
    if player_table.empty or column not in player_table.columns:
        return fig
    top_df = player_table.nlargest(10, column).sort_values(column, ascending=True)
    fig.add_trace(
        go.Bar(
            x=top_df[column],
            y=top_df["player_name"],
            orientation="h",
            marker_color=MVV_RED_DEEP,
            text=[value_formatter(value) for value in top_df[column]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False, margin=dict(l=18, r=28, t=56, b=24))
    fig.update_yaxes(automargin=True)
    return fig


def build_cards_html(summary: dict[str, object], monitoring_summary: dict[str, object]) -> str:
    cards = [
        ("Active Players", _format_int(summary["active_players"]), "Unieke GPS-spelers in deze week"),
        ("Player Sessions", _format_int(summary["player_sessions"]), "Totaal aantal Entire Session - Live-sessies"),
        ("Total Distance", _format_distance(summary["total_distance"]), "Opgetelde teamload binnen de week"),
        ("HSR", _format_distance(summary["hsr_hsd"]), "Sprint + high total_distance_zone_5 distance"),
        ("Sprints", _format_int(summary["sprints"]), "Totale sprintacties in deze week"),
        ("Speed Exposures", _format_int(summary["speed_exposures"]), "Sessies >= 90% van individuele seizoensmax"),
        ("Dist / Player", _format_distance(summary["dist_per_player"]), "Team totaal gedeeld door actieve spelers"),
        ("Top Speed", _format_speed(summary["top_speed"]), "Hoogste topsnelheid in de gekozen week"),
    ]
    return _render_card_grid(cards)


def _render_card_grid(cards: list[tuple[str, str, str]]) -> str:
    html_blocks = []
    for label, value, foot in cards:
        html_blocks.append(
            '<div class="week-report-card">'
            f'<div class="week-report-card-label">{escape(label)}</div>'
            f'<div class="week-report-card-value">{escape(value)}</div>'
            f'<div class="week-report-card-foot">{escape(foot)}</div>'
            "</div>"
        )
    return f'<div class="week-report-card-grid">{"".join(html_blocks)}</div>'


def build_table_html(df: pd.DataFrame, columns: list[tuple[str, str, Callable[[object], str] | None]]) -> str:
    if df.empty:
        return '<div class="week-report-panel-subtitle">Geen data beschikbaar voor deze selectie.</div>'

    header_html = "".join(f"<th>{escape(label)}</th>" for _, label, _ in columns)
    row_html: list[str] = []
    for _, row in df.iterrows():
        cells = []
        for column, _, formatter in columns:
            value = row.get(column)
            display = formatter(value) if formatter else ("--" if pd.isna(value) else str(value))
            cells.append(f"<td>{escape(display)}</td>")
        row_html.append(f"<tr>{''.join(cells)}</tr>")
    return f"""
    <div class="week-report-table-wrap">
      <table class="week-report-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{''.join(row_html)}</tbody>
      </table>
    </div>
    """


def render_panel_header(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f'<div class="week-report-panel-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="week-report-panel-anchor"></div>
        <div class="week-report-panel-title">{escape(title)}</div>
        {subtitle_html}
        """,
        unsafe_allow_html=True,
    )


def render_plot_panel(title: str, fig: go.Figure, subtitle: str | None = None) -> None:
    with st.container():
        render_panel_header(title, subtitle)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_html_panel(title: str, html_content: str, subtitle: str | None = None) -> None:
    with st.container():
        render_panel_header(title, subtitle)
        st.markdown(html_content, unsafe_allow_html=True)


def main() -> None:
    render_css()
    apply_dashboard_polish()
    require_auth()

    sb = get_sb_client()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    ok, access_token = ensure_auth_restored(sb)
    if not ok or not access_token:
        st.error("Kon geen geldige sessie herstellen.")
        st.stop()

    profile = get_profile(sb)
    if not is_staff_user(profile):
        st.error("Geen toegang: deze pagina is alleen voor staff.")
        st.stop()

    render_sidebar_navigation(profile)

    with st.spinner("Weekoverzicht laden..."):
        history_source_df = fetch_summary_index_cached(access_token)
        player_positions = fetch_player_positions_cached(sb, access_token)
        history_source_df = exclude_goalkeepers(history_source_df, player_positions)

    if history_source_df.empty:
        st.info("Geen Entire Session - Live-data gevonden voor het weekoverzicht.")
        st.stop()

    week_options = sorted(
        (pd.Timestamp(value).normalize() for value in history_source_df["week_start"].dropna().unique()),
        reverse=True,
    )
    if not week_options:
        st.info("Geen weken beschikbaar in de Entire Session - Live-data.")
        st.stop()

    default_week = week_options[0]
    if (
        "week_report_selected_week" not in st.session_state
        or st.session_state["week_report_selected_week"] not in week_options
    ):
        st.session_state["week_report_selected_week"] = default_week

    logo_markup = (
        f'<img src="{TEAM_LOGO_URI}" alt="MVV Maastricht" class="week-report-logo" />'
        if TEAM_LOGO_URI
        else ""
    )

    hero_container = st.container()
    with hero_container:
        st.markdown(
            f"""
            <div class="week-report-hero-anchor"></div>
            <div class="week-report-head">
              {logo_markup}
              <div class="week-report-copyhead">
                <h1 class="week-report-title">Weekoverzicht</h1>
                <div class="week-report-kicker">MVV Maastricht | GPS, Wellness &amp; RPE</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        back_col, meta_col = st.columns([0.34, 1.66], gap="large")
        with back_col:
            if st.button("Open dashboard", key="week_overview_back", width="stretch"):
                st.switch_page("app.py")
        with meta_col:
            st.markdown(
                f'<div class="week-report-filter-note">{len(week_options)} weken beschikbaar in totaal</div>',
                unsafe_allow_html=True,
            )

        filter_col, detail_col = st.columns([1.2, 0.8], gap="large")
        with filter_col:
            st.markdown('<div class="week-report-filter-label">Week</div>', unsafe_allow_html=True)
            selected_week = st.selectbox(
                "Week",
                options=week_options,
                format_func=_week_label,
                label_visibility="collapsed",
                key="week_report_selected_week",
            )
        with detail_col:
            selected_iso = selected_week.isocalendar()
            st.markdown(
                f'<div class="week-report-filter-note">ISO week {selected_iso.year}-W{int(selected_iso.week):02d}</div>',
                unsafe_allow_html=True,
            )

    selected_week = pd.Timestamp(selected_week).normalize()
    week_end = selected_week + pd.Timedelta(days=6)
    week_df = fetch_summary_period_cached(access_token, selected_week.date().isoformat(), week_end.date().isoformat())
    week_df = exclude_goalkeepers(week_df, player_positions)
    if week_df.empty:
        st.info("Geen data gevonden voor deze week.")
        st.stop()

    session_stats = build_week_session_stats(week_df)
    player_table = build_week_player_table(week_df)
    render_plot_panel(
        "GPS & belasting per sessie",
        build_session_load_chart(session_stats),
        "Teamgemiddelde van veldspelers uit uitsluitend Entire Session - Live; dubbele sessies blijven apart.",
    )

    render_plot_panel(
        "Totale weekbelasting per speler",
        build_weekly_player_load_chart(player_table),
        "Total Distance, HMLD en gegroepeerde Zone 5/6-belasting van alle Entire Session - Live-sessies.",
    )

    render_sidebar_footer(profile)


if __name__ == "__main__":
    main()
