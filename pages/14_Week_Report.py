from __future__ import annotations

from html import escape
from typing import Callable

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from auth_session import ensure_auth_restored, get_sb_client
from pages.Subscripts.mvv_branding import TEAM_HERO_BG, TEAM_LOGO, build_data_uri
from roles import get_profile, is_staff_user, render_sidebar_footer, render_sidebar_navigation, require_auth
from utils.streamlit_ui import apply_streamlit_chrome


st.set_page_config(page_title="Week Report", layout="wide", initial_sidebar_state="expanded")
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

        [class*="st-key-week_report_back"] button {
          min-height: 2.65rem !important;
          border-radius: 10px !important;
          border: 1px solid rgba(234, 51, 81, 0.22) !important;
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96)) !important;
          color: #ffffff !important;
          font-weight: 800 !important;
          box-shadow: 0 10px 22px rgba(0, 0, 0, 0.18) !important;
        }

        [class*="st-key-week_report_back"] button:hover {
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
    season_max_speed = df.groupby("player_name")["max_speed"].transform("max")
    df["speed_exposure_flag"] = season_max_speed.gt(0) & df["max_speed"].ge(season_max_speed * 0.9)
    return df


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
            max_speed=("max_speed", "max"),
        )
        .reset_index()
    )
    order = pd.Categorical(grouped["session_category"], categories=["Training", "Match"], ordered=True)
    grouped["session_category"] = order
    grouped = grouped.sort_values("session_category").reset_index(drop=True)
    grouped["session_category"] = grouped["session_category"].astype(str)
    return grouped


def build_zone_totals(week_df: pd.DataFrame) -> pd.DataFrame:
    zone_rows = [
        ("Walking", float(week_df["walking"].sum())),
        ("Jogging", float(week_df["jogging"].sum())),
        ("Running", float(week_df["running"].sum())),
        ("Sprint", float(week_df["sprint"].sum())),
        ("High Sprint", float(week_df["high_sprint"].sum())),
    ]
    zone_df = pd.DataFrame(zone_rows, columns=["zone", "value"])
    return zone_df[zone_df["value"] > 0].reset_index(drop=True)


def build_week_notes(summary: dict[str, object], day_table: pd.DataFrame, player_table: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    week_start = summary["week_start"]
    notes.append(
        f"Week {week_start:%d/%m/%Y}: {_format_int(summary['active_players'])} actieve GPS-spelers en {_format_int(summary['player_sessions'])} player-sessies."
    )
    notes.append(
        f"Teamload: {_format_int(summary['total_distance'])} m total distance, {_format_int(summary['hsr_hsd'])} m HSR/HSD en {_format_int(summary['sprints'])} sprints."
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
            f"Speed exposure: {_format_int(summary['speed_exposures'])} sessies bereikten >=90% van de individuele seizoensmax."
        )
    return notes


def build_week_summary(all_df: pd.DataFrame, week_df: pd.DataFrame, history_row: pd.Series | None) -> dict[str, object]:
    active_players = week_df["player_name"].nunique() if not week_df.empty else 0
    player_sessions = len(week_df.index)
    total_distance = float(week_df["total_distance"].sum()) if not week_df.empty else 0.0
    hsr_hsd = float(week_df["hsr_hsd"].sum()) if not week_df.empty else 0.0
    sprints = float(week_df["number_of_sprints"].sum()) if not week_df.empty else 0.0
    top_speed = float(week_df["max_speed"].max()) if not week_df["max_speed"].dropna().empty else float("nan")
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


def build_daily_bar_chart(day_table: pd.DataFrame, column: str, title: str, color: str, value_formatter: Callable[[object], str]) -> go.Figure:
    fig = base_figure(title, height=350)
    if day_table.empty or column not in day_table.columns:
        return fig
    fig.add_trace(
        go.Bar(
            x=day_table["label"],
            y=day_table[column],
            marker_color=color,
            text=[value_formatter(value) for value in day_table[column]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
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


def build_cards_html(summary: dict[str, object]) -> str:
    cards = [
        ("Active Players", _format_int(summary["active_players"]), "Unieke GPS-spelers in deze week"),
        ("Player Sessions", _format_int(summary["player_sessions"]), "Totaal aantal Summary-sessies"),
        ("Total Distance", _format_distance(summary["total_distance"]), "Opgetelde teamload binnen de week"),
        ("HSR / HSD", _format_distance(summary["hsr_hsd"]), "Sprint + high sprint distance"),
        ("Sprints", _format_int(summary["sprints"]), "Totale sprintacties in deze week"),
        ("Speed Exposures", _format_int(summary["speed_exposures"]), "Sessies >= 90% van individuele seizoensmax"),
        ("Dist / Player", _format_distance(summary["dist_per_player"]), "Team totaal gedeeld door actieve spelers"),
        ("Top Speed", _format_speed(summary["top_speed"]), "Hoogste topsnelheid in de gekozen week"),
    ]
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

    with st.spinner("Week report data laden..."):
        all_df = fetch_summary_history_cached(access_token)

    if all_df.empty:
        st.info("Geen Summary GPS-data gevonden voor de weekrapportage.")
        st.stop()

    history_df = build_week_history(all_df)
    week_options = history_df["week_start"].tolist()
    if not week_options:
        st.info("Geen weken beschikbaar in de Summary-data.")
        st.stop()

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
                <h1 class="week-report-title">Week Report</h1>
                <div class="week-report-kicker">MVV Maastricht | Reports | Week Report</div>
              </div>
            </div>
            <div class="week-report-copy">
              Webversie van de team weekrapportage op basis van dezelfde GPS-weekstructuur als in de losse rapportagemap, maar nu compact en direct bruikbaar in het dashboard.
            </div>
            """,
            unsafe_allow_html=True,
        )

        back_col, meta_col = st.columns([0.34, 1.66], gap="large")
        with back_col:
            if st.button("Open Reports", key="week_report_back", use_container_width=True):
                st.switch_page("pages/03_Reports_Page.py")
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
                index=0,
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
    week_df = all_df[(all_df["week_start"] == selected_week)].copy()
    if week_df.empty:
        st.info("Geen data gevonden voor deze week.")
        st.stop()

    day_table = build_week_day_table(week_df)
    day_stats = build_week_day_stats(week_df)
    player_table = build_week_player_table(week_df)
    type_table = build_week_type_table(week_df)
    zone_df = build_zone_totals(week_df)
    history_row = history_df.loc[history_df["week_start"] == selected_week]
    summary = build_week_summary(all_df, week_df, history_row.iloc[0] if not history_row.empty else None)
    notes = build_week_notes(summary, day_table, player_table)

    badges = [
        f"{summary['active_days']} actieve dagen",
        f"{summary['training_sessions']} training sessions",
        f"{summary['match_sessions']} match sessions",
    ]
    if pd.notna(summary["td_vs_prev"]):
        badges.append(f"TD vs vorige 4 weken: {_format_signed_pct(summary['td_vs_prev'])}")
    if pd.notna(summary["hsr_vs_prev"]):
        badges.append(f"HSR vs vorige 4 weken: {_format_signed_pct(summary['hsr_vs_prev'])}")
    st.markdown(
        '<div class="week-report-badge-row">' +
        "".join(f'<span class="week-report-badge">{escape(badge)}</span>' for badge in badges) +
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(build_cards_html(summary), unsafe_allow_html=True)

    tab_overview, tab_spread, tab_leaders = st.tabs(["Overview", "Squad Spread", "Leaders & Notes"])

    with tab_overview:
        top_left, top_right = st.columns(2, gap="large")
        with top_left:
            render_plot_panel(
                "Daily Team Distance",
                build_daily_bar_chart(day_table, "total_distance", "Daily Team Total Distance", MVV_RED_DEEP, _format_distance),
                f"Week {selected_week:%d/%m/%Y} - {week_end:%d/%m/%Y}",
            )
        with top_right:
            render_plot_panel(
                "Daily Team HSR / HSD",
                build_daily_bar_chart(day_table, "hsr_hsd", "Daily Team HSR / HSD", MVV_RED_BRIGHT, _format_distance),
                "Sprint plus high sprint per dag",
            )

        bottom_left, bottom_right = st.columns(2, gap="large")
        with bottom_left:
            render_plot_panel(
                "Distance Zone Share",
                build_zone_share_chart(zone_df),
                "Verdeling over walking, jogging, running, sprint en high sprint",
            )
        with bottom_right:
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
                "Weeksamenvatting per sessiecategorie",
            )

        render_html_panel(
            "Weekdays",
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
            "Dagselectie voor teamload binnen de gekozen week",
        )

    with tab_spread:
        spread_row_one = st.columns(2, gap="large")
        with spread_row_one[0]:
            render_plot_panel(
                "Player Avg Distance +/- SD",
                build_error_bar_chart(day_stats, "total_distance_mean", "total_distance_std", "Daily Player Average Total Distance +/- SD", MVV_RED_DEEP, _format_distance),
                "Per dag gemiddelde spelerload met spreiding",
            )
        with spread_row_one[1]:
            render_plot_panel(
                "Player Avg HSR / HSD +/- SD",
                build_error_bar_chart(day_stats, "hsr_hsd_mean", "hsr_hsd_std", "Daily Player Average HSR / HSD +/- SD", MVV_RED_BRIGHT, _format_distance),
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
                build_error_bar_chart(day_stats, "sprints_mean", "sprints_std", "Daily Player Average Sprints +/- SD", MVV_RED_DEEP, _format_int),
                "Sprints per speler per dag met standaarddeviatie",
            )

    with tab_leaders:
        leader_row = st.columns(3, gap="large")
        with leader_row[0]:
            render_plot_panel(
                "Top 10 Total Distance",
                build_leaderboard_chart(player_table, "total_distance", "Top 10 Players by Total Distance", _format_distance),
                "Weekranking op totaal afgelegde afstand",
            )
        with leader_row[1]:
            render_plot_panel(
                "Top 10 HSR / HSD",
                build_leaderboard_chart(player_table, "hsr_hsd", "Top 10 Players by HSR / HSD", _format_distance),
                "Weekranking op high-speed distance",
            )
        with leader_row[2]:
            render_plot_panel(
                "Top 10 Sprints",
                build_leaderboard_chart(player_table, "sprints", "Top 10 Players by Sprints", _format_int),
                "Weekranking op sprintacties",
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
            "Top 12 spelers binnen deze week op totaal volume",
        )

        notes_html = (
            '<ul class="week-report-note-list">'
            + "".join(f"<li>{escape(note)}</li>" for note in notes)
            + "</ul>"
            + '<div class="week-report-note-foot">Analyse is gebaseerd op Summary-sessies wanneer beschikbaar; ontbrekende metrics worden niet geschat.</div>'
        )
        render_html_panel(
            "Week Notes",
            notes_html,
            "Korte staffsamenvatting van de geselecteerde week",
        )

    render_sidebar_footer(profile)


if __name__ == "__main__":
    main()
