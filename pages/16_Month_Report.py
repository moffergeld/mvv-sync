from __future__ import annotations

from html import escape
from typing import Callable

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from auth_session import ensure_auth_restored, get_sb_client
from pages.Subscripts.mvv_branding import TEAM_HERO_BG, TEAM_LOGO, build_data_uri
from report_monitoring import (
    WELLNESS_PARAMETER_SPECS,
    build_monitoring_dataset,
    build_monitoring_grouped_summary,
    build_monitoring_player_summary,
    summarize_monitoring_dataset,
)
from roles import get_profile, is_staff_user, render_sidebar_footer, render_sidebar_navigation, require_auth
from utils.streamlit_ui import apply_streamlit_chrome


st.set_page_config(page_title="Month Report", layout="wide", initial_sidebar_state="expanded")
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
MVV_GRID = "rgba(255,255,255,0.10)"

GPS_SELECT_COLS = [
    "gps_id",
    "player_id",
    "player_name",
    "datum",
    "week",
    "year",
    "type",
    "event",
    "duration",
    "total_distance",
    "walking",
    "jogging",
    "running",
    "sprint",
    "high_sprint",
    "number_of_sprints",
    "playerload2d",
    "playerload3d",
    "total_accelerations",
    "total_decelerations",
    "hrtrimp",
    "max_speed",
]

SUM_COLUMNS = [
    "duration",
    "total_distance",
    "walking",
    "jogging",
    "running",
    "sprint",
    "high_sprint",
    "number_of_sprints",
    "playerload2d",
    "playerload3d",
    "total_accelerations",
    "total_decelerations",
    "hrtrimp",
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
          background: __MONTH_REPORT_BG__;
          background-size: cover;
          background-position: center top;
          background-attachment: fixed;
        }

        .block-container {
          max-width: 1380px;
          padding-top: 1.25rem;
          padding-bottom: 2.4rem;
        }

        div[data-testid="stVerticalBlock"]:has(.month-report-hero-anchor) {
          padding: 1.75rem 1.6rem 1.3rem 1.6rem;
          border-radius: 10px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(135deg, rgba(18, 25, 42, 0.88), rgba(10, 15, 27, 0.84));
          box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22);
          margin-bottom: 1.1rem;
        }

        div[data-testid="stVerticalBlock"]:has(.month-report-panel-anchor) {
          padding: 1rem 1rem 0.8rem 1rem;
          border-radius: 10px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
          box-shadow: 0 14px 26px rgba(0, 0, 0, 0.18);
          margin-bottom: 1rem;
        }

        .month-report-hero-anchor,
        .month-report-panel-anchor {
          height: 0;
        }

        .month-report-head {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 1rem;
          margin-bottom: 1rem;
        }

        .month-report-logo {
          width: 78px;
          height: 78px;
          object-fit: contain;
          flex-shrink: 0;
          filter: drop-shadow(0 8px 22px rgba(0,0,0,0.28));
        }

        .month-report-copyhead {
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 0.12rem;
          text-align: left;
        }

        .month-report-kicker {
          color: rgba(255,255,255,0.76);
          font-size: 0.74rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.18em;
          margin-bottom: 0;
        }

        .month-report-title {
          margin: 0;
          font-size: 2.45rem;
          line-height: 1;
          font-weight: 800;
          color: #ffffff;
        }

        .month-report-copy {
          margin-top: 0.85rem;
          max-width: 78ch;
          color: rgba(255,255,255,0.84);
          line-height: 1.6;
        }

        .month-report-filter-label {
          color: rgba(255,255,255,0.92);
          font-size: 0.92rem;
          font-weight: 700;
          margin-bottom: 0.35rem;
        }

        .month-report-filter-note {
          color: rgba(255,255,255,0.80);
          font-size: 0.88rem;
          font-weight: 700;
          text-align: right;
          margin-top: 2rem;
        }

        .month-report-badge-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
          margin-top: 1rem;
        }

        .month-report-badge {
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

        [class*="st-key-month_report_back"] button {
          min-height: 2.65rem !important;
          border-radius: 10px !important;
          border: 1px solid rgba(234, 51, 81, 0.22) !important;
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96)) !important;
          color: #ffffff !important;
          font-weight: 800 !important;
          box-shadow: 0 10px 22px rgba(0, 0, 0, 0.18) !important;
        }

        [class*="st-key-month_report_back"] button:hover {
          border-color: rgba(234, 51, 81, 0.36) !important;
          color: #ffffff !important;
        }

        .month-report-card-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 1rem;
          margin: 0.2rem 0 1.15rem 0;
        }

        .month-report-card {
          border-radius: 10px;
          border: 1px solid rgba(234, 51, 81, 0.14);
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
          box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
          padding: 1rem 1rem 0.9rem 1rem;
          min-height: 132px;
        }

        .month-report-card-label {
          color: rgba(255,255,255,0.62);
          font-size: 0.72rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }

        .month-report-card-value {
          margin-top: 0.55rem;
          color: #ffffff;
          font-size: 2rem;
          line-height: 1;
          font-weight: 800;
        }

        .month-report-card-foot {
          margin-top: 0.72rem;
          color: rgba(255,255,255,0.76);
          line-height: 1.45;
          font-size: 0.84rem;
        }

        .month-report-panel-title {
          color: #ffffff;
          font-size: 1.08rem;
          line-height: 1.2;
          font-weight: 800;
          margin-bottom: 0.2rem;
        }

        .month-report-panel-subtitle {
          color: rgba(255,255,255,0.70);
          font-size: 0.84rem;
          margin-bottom: 0.85rem;
        }

        .month-report-table-wrap {
          overflow-x: auto;
        }

        .month-report-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.88rem;
        }

        .month-report-table thead th {
          text-align: left;
          padding: 0.8rem 0.8rem;
          color: rgba(255,255,255,0.68);
          font-size: 0.73rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          border-bottom: 1px solid rgba(255,255,255,0.10);
        }

        .month-report-table tbody td {
          padding: 0.76rem 0.8rem;
          color: rgba(255,255,255,0.90);
          border-bottom: 1px solid rgba(255,255,255,0.06);
          white-space: nowrap;
        }

        .month-report-table tbody tr:last-child td {
          border-bottom: none;
        }

        .month-report-note-list {
          margin: 0.25rem 0 0 0;
          padding-left: 1.1rem;
          color: rgba(255,255,255,0.90);
          line-height: 1.65;
        }

        .month-report-note-foot {
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
          .month-report-card-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
        }

        @media (max-width: 768px) {
          div[data-testid="stVerticalBlock"]:has(.month-report-hero-anchor) {
            padding: 1.35rem 1rem 1rem 1rem;
          }

          .month-report-head {
            flex-direction: column;
            gap: 0.8rem;
          }

          .month-report-copyhead {
            text-align: center;
          }

          .month-report-title {
            font-size: 2rem;
          }

          .month-report-card-grid {
            grid-template-columns: repeat(1, minmax(0, 1fr));
          }

          .month-report-filter-note {
            text-align: left;
            margin-top: 0.25rem;
          }
        }
        </style>
        """.replace("__MONTH_REPORT_BG__", background),
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


@st.cache_data(show_spinner=False, ttl=180)
def fetch_summary_history_cached(access_token: str) -> pd.DataFrame:
    raw = rest_get_paged(
        access_token,
        "gps_records",
        f"select={','.join(GPS_SELECT_COLS)}&event=eq.Summary&order=datum.asc,gps_id.asc",
    )
    if raw.empty:
        return raw

    df = raw.copy()
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce").dt.normalize()
    df["player_name"] = df["player_name"].fillna("Onbekend").astype(str).str.strip()
    df["type"] = df["type"].fillna("").astype(str).str.strip()

    for column in SUM_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df["max_speed"] = pd.to_numeric(df.get("max_speed"), errors="coerce")
    df = df.dropna(subset=["datum"]).copy()

    df["hsr_hsd"] = df["sprint"].fillna(0.0) + df["high_sprint"].fillna(0.0)
    df["session_category"] = df["type"].apply(_session_category)
    df["week_start"] = (df["datum"] - pd.to_timedelta(df["datum"].dt.weekday, unit="D")).dt.normalize()
    df["month_start"] = df["datum"].dt.to_period("M").dt.to_timestamp()
    season_max_speed = df.groupby("player_name")["max_speed"].transform("max")
    df["speed_exposure_flag"] = season_max_speed.gt(0) & df["max_speed"].ge(season_max_speed * 0.9)
    return df


def _month_end(month_start: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(month_start) + pd.offsets.MonthEnd(0)


def _month_label(month_start: pd.Timestamp) -> str:
    month_start = pd.Timestamp(month_start).normalize()
    month_end = _month_end(month_start)
    return f"{month_start:%Y-%m} | {month_start:%d/%m/%Y} - {month_end:%d/%m/%Y}"


def _week_short_label(week_start: pd.Timestamp) -> str:
    iso = pd.Timestamp(week_start).isocalendar()
    return f"W{int(iso.week):02d} | {pd.Timestamp(week_start):%d/%m}"


def build_month_history(all_df: pd.DataFrame) -> pd.DataFrame:
    if all_df.empty:
        return pd.DataFrame()
    history = (
        all_df.groupby("month_start", dropna=False)
        .agg(
            active_players=("player_name", "nunique"),
            player_sessions=("datum", "size"),
            total_distance=("total_distance", "sum"),
            hsr_hsd=("hsr_hsd", "sum"),
            sprints=("number_of_sprints", "sum"),
        )
        .reset_index()
        .sort_values("month_start")
        .reset_index(drop=True)
    )
    history["month_label"] = history["month_start"].apply(_month_label)
    history["td_rolling3_prev"] = history["total_distance"].shift(1).rolling(3, min_periods=1).mean()
    history["hsr_rolling3_prev"] = history["hsr_hsd"].shift(1).rolling(3, min_periods=1).mean()
    return history


def build_month_player_table(month_df: pd.DataFrame) -> pd.DataFrame:
    if month_df.empty:
        return pd.DataFrame()
    grouped = (
        month_df.groupby("player_name", dropna=False)
        .agg(
            sessions=("datum", "size"),
            total_distance=("total_distance", "sum"),
            hsr_hsd=("hsr_hsd", "sum"),
            sprints=("number_of_sprints", "sum"),
            total_accelerations=("total_accelerations", "sum"),
            total_decelerations=("total_decelerations", "sum"),
            playerload2d=("playerload2d", "sum"),
            max_speed=("max_speed", "max"),
            duration=("duration", "sum"),
        )
        .reset_index()
        .sort_values("total_distance", ascending=False)
        .reset_index(drop=True)
    )
    grouped["distance_per_min"] = _safe_divide(grouped["total_distance"], grouped["duration"])
    return grouped


def build_month_day_table(month_df: pd.DataFrame) -> pd.DataFrame:
    if month_df.empty:
        return pd.DataFrame()
    grouped = (
        month_df.groupby("datum", dropna=False)
        .agg(
            active_players=("player_name", "nunique"),
            player_sessions=("datum", "size"),
            total_distance=("total_distance", "sum"),
            hsr_hsd=("hsr_hsd", "sum"),
            sprints=("number_of_sprints", "sum"),
            total_accelerations=("total_accelerations", "sum"),
            total_decelerations=("total_decelerations", "sum"),
            max_speed=("max_speed", "max"),
            duration=("duration", "sum"),
        )
        .reset_index()
        .sort_values("datum")
        .reset_index(drop=True)
    )
    grouped["label"] = grouped["datum"].dt.strftime("%d/%m")
    grouped["distance_per_player"] = _safe_divide(grouped["total_distance"], grouped["active_players"])
    return grouped


def build_month_day_stats(month_df: pd.DataFrame) -> pd.DataFrame:
    if month_df.empty:
        return pd.DataFrame()
    player_day = (
        month_df.groupby(["datum", "player_name"], dropna=False)
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
    player_day["distance_per_min"] = _safe_divide(player_day["total_distance"], player_day["duration"])
    for metric in ["total_distance", "hsr_hsd", "sprints", "total_accelerations", "total_decelerations", "distance_per_min"]:
        player_day[metric] = pd.to_numeric(player_day[metric], errors="coerce").astype(float)

    grouped = player_day.groupby("datum", dropna=False).agg(player_count=("player_name", "nunique")).reset_index()
    for metric in ["total_distance", "hsr_hsd", "sprints", "total_accelerations", "total_decelerations", "distance_per_min"]:
        stats = (
            player_day.groupby("datum", dropna=False)[metric]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
        )
        grouped = grouped.merge(stats, on="datum", how="left")

    grouped["label"] = grouped["datum"].dt.strftime("%d/%m")
    return grouped.sort_values("datum").reset_index(drop=True)


def build_month_week_table(month_df: pd.DataFrame) -> pd.DataFrame:
    if month_df.empty:
        return pd.DataFrame()
    grouped = (
        month_df.groupby("week_start", dropna=False)
        .agg(
            active_players=("player_name", "nunique"),
            player_sessions=("datum", "size"),
            total_distance=("total_distance", "sum"),
            hsr_hsd=("hsr_hsd", "sum"),
            sprints=("number_of_sprints", "sum"),
            max_speed=("max_speed", "max"),
        )
        .reset_index()
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    grouped["label"] = grouped["week_start"].apply(_week_short_label)
    grouped["distance_per_player"] = _safe_divide(grouped["total_distance"], grouped["active_players"])
    return grouped


def build_month_type_table(month_df: pd.DataFrame) -> pd.DataFrame:
    if month_df.empty:
        return pd.DataFrame()
    grouped = (
        month_df.groupby("session_category", dropna=False)
        .agg(
            player_sessions=("datum", "size"),
            active_players=("player_name", "nunique"),
            total_distance=("total_distance", "sum"),
            hsr_hsd=("hsr_hsd", "sum"),
            sprints=("number_of_sprints", "sum"),
            max_speed=("max_speed", "max"),
        )
        .reset_index()
    )
    order = pd.Categorical(grouped["session_category"], categories=["Training", "Match"], ordered=True)
    grouped["session_category"] = order
    grouped = grouped.sort_values("session_category").reset_index(drop=True)
    grouped["session_category"] = grouped["session_category"].astype(str)
    return grouped


def build_zone_totals(month_df: pd.DataFrame) -> pd.DataFrame:
    zone_rows = [
        ("Walking", float(month_df["walking"].sum())),
        ("Jogging", float(month_df["jogging"].sum())),
        ("Running", float(month_df["running"].sum())),
        ("Sprint", float(month_df["sprint"].sum())),
        ("High Sprint", float(month_df["high_sprint"].sum())),
    ]
    zone_df = pd.DataFrame(zone_rows, columns=["zone", "value"])
    return zone_df[zone_df["value"] > 0].reset_index(drop=True)


def build_month_summary(month_df: pd.DataFrame, history_row: pd.Series | None) -> dict[str, object]:
    active_players = month_df["player_name"].nunique() if not month_df.empty else 0
    player_sessions = len(month_df.index)
    total_distance = float(month_df["total_distance"].sum()) if not month_df.empty else 0.0
    hsr_hsd = float(month_df["hsr_hsd"].sum()) if not month_df.empty else 0.0
    sprints = float(month_df["number_of_sprints"].sum()) if not month_df.empty else 0.0
    top_speed = float(month_df["max_speed"].max()) if not month_df["max_speed"].dropna().empty else float("nan")
    speed_exposures = float(month_df["speed_exposure_flag"].sum()) if not month_df.empty else 0.0
    match_sessions = int((month_df["session_category"] == "Match").sum()) if not month_df.empty else 0
    training_sessions = int((month_df["session_category"] == "Training").sum()) if not month_df.empty else 0
    active_days = int(month_df["datum"].nunique()) if not month_df.empty else 0
    weeks_in_month = int(month_df["week_start"].nunique()) if not month_df.empty else 0
    dist_per_player = total_distance / active_players if active_players else float("nan")

    td_vs_prev = float("nan")
    hsr_vs_prev = float("nan")
    if history_row is not None:
        td_base = history_row.get("td_rolling3_prev")
        hsr_base = history_row.get("hsr_rolling3_prev")
        if pd.notna(td_base) and float(td_base) != 0:
            td_vs_prev = ((total_distance - float(td_base)) / float(td_base)) * 100
        if pd.notna(hsr_base) and float(hsr_base) != 0:
            hsr_vs_prev = ((hsr_hsd - float(hsr_base)) / float(hsr_base)) * 100

    month_start = month_df["month_start"].iloc[0] if not month_df.empty else pd.Timestamp.today().normalize().replace(day=1)
    return {
        "month_start": month_start,
        "month_end": _month_end(month_start),
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
        "weeks_in_month": weeks_in_month,
        "td_vs_prev": td_vs_prev,
        "hsr_vs_prev": hsr_vs_prev,
    }


def build_month_notes(
    summary: dict[str, object],
    day_table: pd.DataFrame,
    week_table: pd.DataFrame,
    player_table: pd.DataFrame,
) -> list[str]:
    notes: list[str] = []
    notes.append(
        f"Maand {_month_label(summary['month_start'])}: {_format_int(summary['active_players'])} actieve GPS-spelers en {_format_int(summary['player_sessions'])} player-sessies."
    )
    notes.append(
        f"Teamload: {_format_int(summary['total_distance'])} m total distance, {_format_int(summary['hsr_hsd'])} m HSR/HSD en {_format_int(summary['sprints'])} sprints."
    )
    if not day_table.empty:
        peak_day = day_table.sort_values("total_distance", ascending=False).iloc[0]
        notes.append(
            f"Hoogste dag binnen deze maand: {peak_day['datum']:%d/%m/%Y} met {_format_int(peak_day['total_distance'])} m."
        )
    if not week_table.empty:
        peak_week = week_table.sort_values("total_distance", ascending=False).iloc[0]
        notes.append(
            f"Zwaarste weekblok: {peak_week['label']} met {_format_int(peak_week['total_distance'])} m."
        )
    if not player_table.empty:
        top_player = player_table.sort_values("total_distance", ascending=False).iloc[0]
        notes.append(
            f"Hoogste individuele maandload: {top_player['player_name']} met {_format_int(top_player['total_distance'])} m."
        )
    if not pd.isna(summary["speed_exposures"]):
        notes.append(
            f"Speed exposure: {_format_int(summary['speed_exposures'])} sessies bereikten >=90% van de individuele seizoensmax."
        )
    return notes[:6]


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


def build_bar_chart(
    df: pd.DataFrame,
    label_column: str,
    value_column: str,
    title: str,
    color: str,
    value_formatter: Callable[[object], str],
    height: int = 350,
    hover_format: str = ":,.0f",
    y_range: tuple[float, float] | None = None,
    error_column: str | None = None,
) -> go.Figure:
    fig = base_figure(title, height=height)
    if df.empty or value_column not in df.columns or label_column not in df.columns:
        return fig
    error_values = None
    if error_column and error_column in df.columns:
        error_values = df[error_column].fillna(0)
    fig.add_trace(
        go.Bar(
            x=df[label_column],
            y=df[value_column],
            marker_color=color,
            error_y=dict(type="data", array=error_values, color=MVV_RED_BRIGHT, thickness=1.4) if error_values is not None else None,
            text=[value_formatter(value) for value in df[value_column]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"%{{x}}<br>%{{y{hover_format}}}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    return fig


def build_error_bar_chart(
    day_stats: pd.DataFrame,
    mean_column: str,
    std_column: str,
    title: str,
    color: str,
    value_formatter: Callable[[object], str],
) -> go.Figure:
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


def build_zone_share_chart(zone_df: pd.DataFrame) -> go.Figure:
    fig = base_figure("Distance Zone Share", height=340)
    if zone_df.empty:
        return fig
    fig = go.Figure(
        data=[
            go.Pie(
                labels=zone_df["zone"],
                values=zone_df["value"],
                hole=0.42,
                textinfo="percent+label",
                marker=dict(colors=["#F5D2D8", "#F1A4B5", "#E97A93", "#D92B4D", "#6E1222"]),
            )
        ]
    )
    fig.update_layout(
        title=dict(text="Distance Zone Share", x=0.02, xanchor="left", font=dict(size=20, color=MVV_TEXT)),
        height=340,
        margin=dict(l=18, r=18, t=56, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MVV_TEXT, size=12),
        legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left", yanchor="middle"),
    )
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
    wellness_cards = [
        (
            label,
            _format_decimal(monitoring_summary[column], 1),
            f"Gemiddelde {label.lower()} in deze maand",
        )
        for column, label in WELLNESS_PARAMETER_SPECS
    ]
    cards = [
        ("Active Players", _format_int(summary["active_players"]), "Unieke GPS-spelers in deze maand"),
        ("Player Sessions", _format_int(summary["player_sessions"]), "Totaal aantal Summary-sessies"),
        ("Total Distance", _format_distance(summary["total_distance"]), "Opgetelde teamload binnen de maand"),
        ("HSR / HSD", _format_distance(summary["hsr_hsd"]), "Sprint + high sprint distance"),
        ("Sprints", _format_int(summary["sprints"]), "Totale sprintacties in deze maand"),
        ("Speed Exposures", _format_int(summary["speed_exposures"]), "Sessies >= 90% van individuele seizoensmax"),
        ("Dist / Player", _format_distance(summary["dist_per_player"]), "Team totaal gedeeld door actieve spelers"),
        ("Top Speed", _format_speed(summary["top_speed"]), "Hoogste topsnelheid in de gekozen maand"),
        *wellness_cards,
        ("Avg RPE", _format_decimal(monitoring_summary["avg_rpe"], 1), "Gemiddelde team-RPE in deze maand"),
    ]
    html_blocks = []
    for label, value, foot in cards:
        html_blocks.append(
            '<div class="month-report-card">'
            f'<div class="month-report-card-label">{escape(label)}</div>'
            f'<div class="month-report-card-value">{escape(value)}</div>'
            f'<div class="month-report-card-foot">{escape(foot)}</div>'
            "</div>"
        )
    return f'<div class="month-report-card-grid">{"".join(html_blocks)}</div>'


def build_table_html(df: pd.DataFrame, columns: list[tuple[str, str, Callable[[object], str] | None]]) -> str:
    if df.empty:
        return '<div class="month-report-panel-subtitle">Geen data beschikbaar voor deze selectie.</div>'

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
    <div class="month-report-table-wrap">
      <table class="month-report-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{''.join(row_html)}</tbody>
      </table>
    </div>
    """


def render_panel_header(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f'<div class="month-report-panel-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="month-report-panel-anchor"></div>
        <div class="month-report-panel-title">{escape(title)}</div>
        {subtitle_html}
        """,
        unsafe_allow_html=True,
    )


def render_plot_panel(title: str, fig: go.Figure, subtitle: str | None = None) -> None:
    with st.container():
        render_panel_header(title, subtitle)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_html_panel(title: str, html_content: str, subtitle: str | None = None) -> None:
    with st.container():
        render_panel_header(title, subtitle)
        st.markdown(html_content, unsafe_allow_html=True)


def main() -> None:
    render_css()
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

    with st.spinner("Month report data laden..."):
        all_df = fetch_summary_history_cached(access_token)

    if all_df.empty:
        st.info("Geen Summary GPS-data gevonden voor de maandrapportage.")
        st.stop()

    history_df = build_month_history(all_df)
    month_options = history_df["month_start"].sort_values(ascending=False).tolist()
    if not month_options:
        st.info("Geen maanden beschikbaar in de Summary-data.")
        st.stop()

    logo_markup = (
        f'<img src="{TEAM_LOGO_URI}" alt="MVV Maastricht" class="month-report-logo" />'
        if TEAM_LOGO_URI
        else ""
    )

    with st.container():
        st.markdown(
            f"""
            <div class="month-report-hero-anchor"></div>
            <div class="month-report-head">
              {logo_markup}
              <div class="month-report-copyhead">
                <h1 class="month-report-title">Month Report</h1>
                <div class="month-report-kicker">MVV Maastricht | Reports | Month Report</div>
              </div>
            </div>
            <div class="month-report-copy">
              Compacte maandrapportage voor staff met dagelijkse teamload, weekblokken binnen de maand, squad spread en leaders op basis van dezelfde Summary GPS-data als de andere rapporten.
            </div>
            """,
            unsafe_allow_html=True,
        )

        back_col, meta_col = st.columns([0.34, 1.66], gap="large")
        with back_col:
            if st.button("Open Reports", key="month_report_back", use_container_width=True):
                st.switch_page("pages/03_Reports_Page.py")
        with meta_col:
            st.markdown(
                f'<div class="month-report-filter-note">{len(month_options)} maanden beschikbaar in totaal</div>',
                unsafe_allow_html=True,
            )

        filter_col, detail_col = st.columns([1.2, 0.8], gap="large")
        with filter_col:
            st.markdown('<div class="month-report-filter-label">Maand</div>', unsafe_allow_html=True)
            selected_month = st.selectbox(
                "Maand",
                options=month_options,
                index=0,
                format_func=_month_label,
                label_visibility="collapsed",
                key="month_report_selected_month",
            )
        with detail_col:
            selected_month = pd.Timestamp(selected_month).normalize()
            month_end = _month_end(selected_month)
            st.markdown(
                f'<div class="month-report-filter-note">Periode: {selected_month:%d/%m/%Y} - {month_end:%d/%m/%Y}</div>',
                unsafe_allow_html=True,
            )

    selected_month = pd.Timestamp(selected_month).normalize()
    month_end = _month_end(selected_month)
    month_df = all_df[all_df["month_start"] == selected_month].copy()
    if month_df.empty:
        st.info("Geen data gevonden voor deze maand.")
        st.stop()

    player_lookup = (
        month_df.assign(player_id=month_df["player_id"].astype(str), player_name=month_df["player_name"].fillna("Onbekend").astype(str))
        .drop_duplicates(subset=["player_id"])
        .set_index("player_id")["player_name"]
        .to_dict()
    )
    monitoring_df = build_monitoring_dataset(
        SUPABASE_URL or "default",
        sb,
        selected_month.date(),
        month_end.date(),
        player_ids=month_df["player_id"].astype(str).tolist(),
        player_lookup=player_lookup,
    )
    monitoring_summary = summarize_monitoring_dataset(monitoring_df)
    monitoring_day_table = build_monitoring_grouped_summary(monitoring_df, "day")
    monitoring_player_table = build_monitoring_player_summary(monitoring_df)

    day_table = build_month_day_table(month_df)
    day_stats = build_month_day_stats(month_df)
    week_table = build_month_week_table(month_df)
    player_table = build_month_player_table(month_df)
    type_table = build_month_type_table(month_df)
    zone_df = build_zone_totals(month_df)
    history_row = history_df.loc[history_df["month_start"] == selected_month]
    summary = build_month_summary(month_df, history_row.iloc[0] if not history_row.empty else None)
    notes = build_month_notes(summary, day_table, week_table, player_table)
    if monitoring_summary["wellness_entries"]:
        wellness_note = ", ".join(
            f"{label.lower()} {_format_decimal(monitoring_summary[column], 1)}"
            for column, label in WELLNESS_PARAMETER_SPECS
        )
        notes.append(
            f"Wellness gemiddeld: {wellness_note} op basis van {_format_int(monitoring_summary['wellness_entries'])} entries."
        )
    if monitoring_summary["rpe_entries"]:
        notes.append(
            f"RPE gemiddeld: {_format_decimal(monitoring_summary['avg_rpe'], 1)} op basis van {_format_int(monitoring_summary['rpe_entries'])} entries."
        )

    badges = [
        f"{summary['active_days']} actieve dagen",
        f"{summary['weeks_in_month']} weekblokken",
        f"{summary['training_sessions']} training sessions",
        f"{summary['match_sessions']} match sessions",
    ]
    if pd.notna(summary["td_vs_prev"]):
        badges.append(f"TD vs vorige 3 maanden: {_format_signed_pct(summary['td_vs_prev'])}")
    if pd.notna(summary["hsr_vs_prev"]):
        badges.append(f"HSR vs vorige 3 maanden: {_format_signed_pct(summary['hsr_vs_prev'])}")
    st.markdown(
        '<div class="month-report-badge-row">' +
        "".join(f'<span class="month-report-badge">{escape(badge)}</span>' for badge in badges) +
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(build_cards_html(summary, monitoring_summary), unsafe_allow_html=True)

    tab_overview, tab_spread, tab_monitoring, tab_leaders = st.tabs(["Overview", "Squad Spread", "Wellness & RPE", "Leaders & Notes"])

    with tab_overview:
        row_one = st.columns(2, gap="large")
        with row_one[0]:
            render_plot_panel(
                "Daily Team Distance",
                build_bar_chart(
                    day_table,
                    "label",
                    "total_distance",
                    "Daily Team Total Distance",
                    MVV_RED_DEEP,
                    _format_distance,
                ),
                f"Maand {selected_month:%d/%m/%Y} - {month_end:%d/%m/%Y}",
            )
        with row_one[1]:
            render_plot_panel(
                "Weeks in Month",
                build_bar_chart(
                    week_table,
                    "label",
                    "total_distance",
                    "Weekly Team Total Distance within Month",
                    MVV_RED_BRIGHT,
                    _format_distance,
                ),
                "Weekblokken binnen de gekozen maand",
            )

        row_two = st.columns(2, gap="large")
        with row_two[0]:
            render_plot_panel(
                "Distance Zone Share",
                build_zone_share_chart(zone_df),
                "Verdeling over walking, jogging, running, sprint en high sprint",
            )
        with row_two[1]:
            render_html_panel(
                "Training vs Match",
                build_table_html(
                    type_table,
                    [
                        ("session_category", "Type", None),
                        ("active_players", "Players", _format_int),
                        ("player_sessions", "Sessions", _format_int),
                        ("total_distance", "Distance", _format_distance),
                        ("hsr_hsd", "HSR/HSD", _format_distance),
                        ("sprints", "Sprints", _format_int),
                        ("max_speed", "Top Speed", _format_speed),
                    ],
                ),
                "Maandsamenvatting per sessiecategorie",
            )

        render_html_panel(
            "Weeks in Table",
            build_table_html(
                week_table,
                [
                    ("label", "Week", None),
                    ("active_players", "Players", _format_int),
                    ("player_sessions", "Sessions", _format_int),
                    ("total_distance", "Distance", _format_distance),
                    ("hsr_hsd", "HSR/HSD", _format_distance),
                    ("sprints", "Sprints", _format_int),
                    ("distance_per_player", "Dist / Player", _format_distance),
                ],
            ),
            "Wekelijkse samenvatting binnen de gekozen maand",
        )

        render_plot_panel(
            "Daily Team HSR / HSD",
            build_bar_chart(
                day_table,
                "label",
                "hsr_hsd",
                "Daily Team HSR / HSD",
                MVV_RED_BRIGHT,
                _format_distance,
            ),
            "Sprint plus high sprint per dag",
        )

        render_html_panel(
            "Monthdays",
            build_table_html(
                day_table,
                [
                    ("label", "Dag", None),
                    ("active_players", "Players", _format_int),
                    ("player_sessions", "Sessions", _format_int),
                    ("total_distance", "Distance", _format_distance),
                    ("hsr_hsd", "HSR/HSD", _format_distance),
                    ("sprints", "Sprints", _format_int),
                    ("distance_per_player", "Dist / Player", _format_distance),
                ],
            ),
            "Dagselectie voor teamload binnen de gekozen maand",
        )

    with tab_spread:
        spread_row_one = st.columns(2, gap="large")
        with spread_row_one[0]:
            render_plot_panel(
                "Player Avg Distance +/- SD",
                build_error_bar_chart(
                    day_stats,
                    "total_distance_mean",
                    "total_distance_std",
                    "Daily Player Average Total Distance +/- SD",
                    MVV_RED_DEEP,
                    _format_distance,
                ),
                "Per dag gemiddelde spelerload met spreiding",
            )
        with spread_row_one[1]:
            render_plot_panel(
                "Player Avg HSR / HSD +/- SD",
                build_error_bar_chart(
                    day_stats,
                    "hsr_hsd_mean",
                    "hsr_hsd_std",
                    "Daily Player Average HSR / HSD +/- SD",
                    MVV_RED_BRIGHT,
                    _format_distance,
                ),
                "Per dag gemiddelde high-speed distance met spreiding",
            )

        spread_row_two = st.columns(2, gap="large")
        with spread_row_two[0]:
            render_plot_panel(
                "Player Avg Accel / Decel +/- SD",
                build_grouped_error_chart(day_stats),
                "Accelerations en decelerations als daggemiddelde per speler",
            )
        with spread_row_two[1]:
            render_plot_panel(
                "Player Avg Sprints +/- SD",
                build_error_bar_chart(
                    day_stats,
                    "sprints_mean",
                    "sprints_std",
                    "Daily Player Average Sprints +/- SD",
                    MVV_RED_DEEP,
                    _format_int,
                ),
                "Sprints per speler per dag met standaarddeviatie",
            )

    with tab_monitoring:
        if monitoring_df.empty:
            st.info("Geen wellness- of RPE-data beschikbaar voor deze maand.")
        else:
            monitoring_specs = [
                ("muscle_soreness", "Muscle Soreness", "Gemiddelde muscle soreness per dag", MVV_RED_DEEP, _format_decimal, ":.1f", (0, 10)),
                ("fatigue", "Fatigue", "Gemiddelde fatigue per dag", MVV_RED_BRIGHT, _format_decimal, ":.1f", (0, 10)),
                ("sleep_quality", "Sleep Quality", "Gemiddelde sleep quality per dag", MVV_RED_DEEP, _format_decimal, ":.1f", (0, 10)),
                ("stress", "Stress", "Gemiddelde stress per dag", MVV_RED_BRIGHT, _format_decimal, ":.1f", (0, 10)),
                ("mood", "Mood", "Gemiddelde mood per dag", MVV_RED_DEEP, _format_decimal, ":.1f", (0, 10)),
                ("avg_rpe", "Avg RPE", "Gemiddelde team-RPE per dag", MVV_RED_BRIGHT, _format_decimal, ":.1f", (0, 10)),
            ]
            for idx in range(0, len(monitoring_specs), 2):
                cols = st.columns(2, gap="large")
                for col_container, spec in zip(cols, monitoring_specs[idx : idx + 2]):
                    column, label, subtitle, color, formatter, hover_format, y_range = spec
                    with col_container:
                        render_plot_panel(
                            f"Daily {label} +/- SD",
                            build_bar_chart(
                                monitoring_day_table,
                                "label",
                                column,
                                f"Daily Team {label}",
                                color,
                                formatter,
                                hover_format=hover_format,
                                y_range=y_range,
                                error_column=f"{column}_std",
                            ),
                            subtitle,
                        )

            render_html_panel(
                "Monitoring by Day",
                build_table_html(
                    monitoring_day_table,
                    [
                        ("label", "Dag", None),
                        ("wellness_players", "Wellness Players", _format_int),
                        ("rpe_players", "RPE Players", _format_int),
                        ("muscle_soreness", "Muscle", _format_decimal),
                        ("fatigue", "Fatigue", _format_decimal),
                        ("sleep_quality", "Sleep", _format_decimal),
                        ("stress", "Stress", _format_decimal),
                        ("mood", "Mood", _format_decimal),
                        ("readiness_score", "Readiness", _format_decimal),
                        ("avg_rpe", "Avg RPE", _format_decimal),
                    ],
                ),
                "Dagoverzicht van alle wellness-parameters, readiness en RPE",
            )

            render_html_panel(
                "Monitoring Players",
                build_table_html(
                    monitoring_player_table.head(12),
                    [
                        ("player_name", "Speler", None),
                        ("wellness_days", "Wellness Days", _format_int),
                        ("rpe_days", "RPE Days", _format_int),
                        ("muscle_soreness", "Muscle", _format_decimal),
                        ("fatigue", "Fatigue", _format_decimal),
                        ("sleep_quality", "Sleep", _format_decimal),
                        ("stress", "Stress", _format_decimal),
                        ("mood", "Mood", _format_decimal),
                        ("readiness_score", "Readiness", _format_decimal),
                        ("avg_rpe", "Avg RPE", _format_decimal),
                    ],
                ),
                "Top 12 spelers op basis van monitoringvolume in deze maand",
            )

    with tab_leaders:
        leader_row = st.columns(3, gap="large")
        with leader_row[0]:
            render_plot_panel(
                "Top 10 Total Distance",
                build_leaderboard_chart(player_table, "total_distance", "Top 10 Players by Total Distance", _format_distance),
                "Maandranking op totaal afgelegde afstand",
            )
        with leader_row[1]:
            render_plot_panel(
                "Top 10 HSR / HSD",
                build_leaderboard_chart(player_table, "hsr_hsd", "Top 10 Players by HSR / HSD", _format_distance),
                "Maandranking op high-speed distance",
            )
        with leader_row[2]:
            render_plot_panel(
                "Top 10 Sprints",
                build_leaderboard_chart(player_table, "sprints", "Top 10 Players by Sprints", _format_int),
                "Maandranking op sprintacties",
            )

        render_html_panel(
            "Player Summary",
            build_table_html(
                player_table.head(12),
                [
                    ("player_name", "Speler", None),
                    ("sessions", "Sessies", _format_int),
                    ("total_distance", "Distance", _format_distance),
                    ("hsr_hsd", "HSR/HSD", _format_distance),
                    ("sprints", "Sprints", _format_int),
                    ("distance_per_min", "Dist / Min", _format_decimal),
                    ("max_speed", "Top Speed", _format_speed),
                ],
            ),
            "Top 12 spelers binnen deze maand op totaal volume",
        )

        notes_html = (
            '<ul class="month-report-note-list">'
            + "".join(f"<li>{escape(note)}</li>" for note in notes)
            + "</ul>"
            + '<div class="month-report-note-foot">Analyse is gebaseerd op Summary-sessies wanneer beschikbaar; ontbrekende metrics worden niet geschat.</div>'
        )
        render_html_panel(
            "Team Analyst Notes",
            notes_html,
            "Compacte maandlezing voor staffbespreking en opvolging",
        )

    render_sidebar_footer(profile)


if __name__ == "__main__":
    main()
