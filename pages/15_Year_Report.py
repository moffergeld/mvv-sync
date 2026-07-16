from __future__ import annotations

from html import escape
from typing import Callable

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

from acwr_settings import compute_chronic_series, get_acwr_mode_meta
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


st.set_page_config(page_title="Year Report", layout="wide", initial_sidebar_state="expanded")
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
          background: __YEAR_REPORT_BG__;
          background-size: cover;
          background-position: center top;
          background-attachment: fixed;
        }

        .block-container {
          max-width: 1380px;
          padding-top: 1.25rem;
          padding-bottom: 2.4rem;
        }

        div[data-testid="stVerticalBlock"]:has(.year-report-hero-anchor) {
          padding: 1.75rem 1.6rem 1.3rem 1.6rem;
          border-radius: 10px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(135deg, rgba(18, 25, 42, 0.88), rgba(10, 15, 27, 0.84));
          box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22);
          margin-bottom: 1.1rem;
        }

        div[data-testid="stVerticalBlock"]:has(.year-report-panel-anchor) {
          padding: 1rem 1rem 0.8rem 1rem;
          border-radius: 10px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
          box-shadow: 0 14px 26px rgba(0, 0, 0, 0.18);
          margin-bottom: 1rem;
        }

        .year-report-hero-anchor,
        .year-report-panel-anchor {
          height: 0;
        }

        .year-report-head {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 1rem;
          margin-bottom: 1rem;
        }

        .year-report-logo {
          width: 78px;
          height: 78px;
          object-fit: contain;
          flex-shrink: 0;
          filter: drop-shadow(0 8px 22px rgba(0,0,0,0.28));
        }

        .year-report-copyhead {
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 0.12rem;
          text-align: left;
        }

        .year-report-kicker {
          color: rgba(255,255,255,0.76);
          font-size: 0.74rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.18em;
          margin-bottom: 0;
        }

        .year-report-title {
          margin: 0;
          font-size: 2.45rem;
          line-height: 1;
          font-weight: 800;
          color: #ffffff;
        }

        .year-report-copy {
          margin-top: 0.85rem;
          max-width: 80ch;
          color: rgba(255,255,255,0.84);
          line-height: 1.6;
        }

        .year-report-filter-label {
          color: rgba(255,255,255,0.92);
          font-size: 0.92rem;
          font-weight: 700;
          margin-bottom: 0.35rem;
        }

        .year-report-filter-note {
          color: rgba(255,255,255,0.80);
          font-size: 0.88rem;
          font-weight: 700;
          text-align: right;
          margin-top: 2rem;
        }

        .year-report-badge-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
          margin-top: 1rem;
        }

        .year-report-badge {
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

        [class*="st-key-year_report_back"] button {
          min-height: 2.65rem !important;
          border-radius: 10px !important;
          border: 1px solid rgba(234, 51, 81, 0.22) !important;
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96)) !important;
          color: #ffffff !important;
          font-weight: 800 !important;
          box-shadow: 0 10px 22px rgba(0, 0, 0, 0.18) !important;
        }

        [class*="st-key-year_report_back"] button:hover {
          border-color: rgba(234, 51, 81, 0.36) !important;
          color: #ffffff !important;
        }

        .year-report-card-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 1rem;
          margin: 0.2rem 0 1.15rem 0;
        }

        .year-report-card {
          border-radius: 10px;
          border: 1px solid rgba(234, 51, 81, 0.14);
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
          box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
          padding: 1rem 1rem 0.9rem 1rem;
          min-height: 132px;
        }

        .year-report-card-label {
          color: rgba(255,255,255,0.62);
          font-size: 0.72rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }

        .year-report-card-value {
          margin-top: 0.55rem;
          color: #ffffff;
          font-size: 2rem;
          line-height: 1;
          font-weight: 800;
        }

        .year-report-card-foot {
          margin-top: 0.72rem;
          color: rgba(255,255,255,0.76);
          line-height: 1.45;
          font-size: 0.84rem;
        }

        .year-report-panel-title {
          color: #ffffff;
          font-size: 1.08rem;
          line-height: 1.2;
          font-weight: 800;
          margin-bottom: 0.2rem;
        }

        .year-report-panel-subtitle {
          color: rgba(255,255,255,0.70);
          font-size: 0.84rem;
          margin-bottom: 0.85rem;
        }

        .year-report-table-wrap {
          overflow-x: auto;
        }

        .year-report-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.88rem;
        }

        .year-report-table thead th {
          text-align: left;
          padding: 0.8rem 0.8rem;
          color: rgba(255,255,255,0.68);
          font-size: 0.73rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          border-bottom: 1px solid rgba(255,255,255,0.10);
        }

        .year-report-table tbody td {
          padding: 0.76rem 0.8rem;
          color: rgba(255,255,255,0.90);
          border-bottom: 1px solid rgba(255,255,255,0.06);
          white-space: nowrap;
        }

        .year-report-table tbody tr:last-child td {
          border-bottom: none;
        }

        .year-report-note-list {
          margin: 0.25rem 0 0 0;
          padding-left: 1.1rem;
          color: rgba(255,255,255,0.90);
          line-height: 1.65;
        }

        .year-report-note-foot {
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
          .year-report-card-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
        }

        @media (max-width: 768px) {
          div[data-testid="stVerticalBlock"]:has(.year-report-hero-anchor) {
            padding: 1.35rem 1rem 1rem 1rem;
          }

          .year-report-head {
            flex-direction: column;
            gap: 0.8rem;
          }

          .year-report-copyhead {
            text-align: center;
          }

          .year-report-title {
            font-size: 2rem;
          }

          .year-report-card-grid {
            grid-template-columns: repeat(1, minmax(0, 1fr));
          }

          .year-report-filter-note {
            text-align: left;
            margin-top: 0.25rem;
          }
        }
        </style>
        """.replace("__YEAR_REPORT_BG__", background),
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


def _classify_acwr(value: object) -> str:
    if pd.isna(value):
        return "--"
    val = float(value)
    if val < 0.8:
        return "Low"
    if val <= 1.3:
        return "OK"
    if val <= 1.5:
        return "Amber"
    return "High"


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


def _format_percent(value: object) -> str:
    base = _format_decimal(value, 1)
    return "--" if base == "--" else f"{base}%"


def _format_signed_percent(value: object) -> str:
    if pd.isna(value):
        return "--"
    prefix = "+" if float(value) >= 0 else ""
    return f"{prefix}{_format_decimal(value, 1)}%"


def _format_hours(value: object) -> str:
    if pd.isna(value):
        return "--"
    return f"{_format_decimal(value, 1)} h"


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
    df["season_start_year"] = df["datum"].dt.year.where(df["datum"].dt.month >= 7, df["datum"].dt.year - 1)
    df["season_label"] = df["season_start_year"].astype(int).astype(str) + "/" + (df["season_start_year"] + 1).astype(int).astype(str)
    season_max_speed = df.groupby("player_name")["max_speed"].transform("max")
    df["speed_exposure_flag"] = season_max_speed.gt(0) & df["max_speed"].ge(season_max_speed * 0.9)
    return df


def build_season_dataset(season_df: pd.DataFrame, acwr_mode: str) -> pd.DataFrame:
    if season_df.empty:
        return pd.DataFrame()
    weekly_df = (
        season_df.groupby("week_start", dropna=False)
        .agg(
            active_players=("player_name", "nunique"),
            sessions=("datum", "size"),
            total_distance=("total_distance", "sum"),
            walking=("walking", "sum"),
            jogging=("jogging", "sum"),
            running=("running", "sum"),
            sprint=("sprint", "sum"),
            high_sprint=("high_sprint", "sum"),
            hsr_hsd=("hsr_hsd", "sum"),
            duration=("duration", "sum"),
            number_of_sprints=("number_of_sprints", "sum"),
            total_accelerations=("total_accelerations", "sum"),
            total_decelerations=("total_decelerations", "sum"),
            playerload2d=("playerload2d", "sum"),
            playerload3d=("playerload3d", "sum"),
            hrtrimp=("hrtrimp", "sum"),
            speed_exposures=("speed_exposure_flag", "sum"),
            max_speed=("max_speed", "max"),
        )
        .reset_index()
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    weekly_df["week_label"] = weekly_df["week_start"].dt.strftime("%d/%m")
    weekly_df["distance_per_player"] = _safe_divide(weekly_df["total_distance"], weekly_df["active_players"])
    weekly_df["sprints_per_player"] = _safe_divide(weekly_df["number_of_sprints"], weekly_df["active_players"])
    weekly_df["accel_density"] = _safe_divide(weekly_df["total_accelerations"], weekly_df["duration"], multiplier=10)
    weekly_df["total_distance_rolling4"] = pd.to_numeric(weekly_df["total_distance"], errors="coerce").rolling(4, min_periods=1).mean()
    weekly_df["total_distance_acwr"] = _safe_divide(
        pd.to_numeric(weekly_df["total_distance"], errors="coerce"),
        compute_chronic_series(weekly_df["total_distance"], acwr_mode),
    )
    weekly_df["hsr_hsd_acwr"] = _safe_divide(
        pd.to_numeric(weekly_df["hsr_hsd"], errors="coerce"),
        compute_chronic_series(weekly_df["hsr_hsd"], acwr_mode),
    )
    weekly_df["hsr_hsd_wow_change"] = pd.to_numeric(weekly_df["hsr_hsd"], errors="coerce").pct_change() * 100
    return weekly_df


def build_category_summary(season_df: pd.DataFrame) -> pd.DataFrame:
    if season_df.empty:
        return pd.DataFrame()
    grouped = (
        season_df.groupby("session_category", dropna=False)
        .agg(
            sessions=("datum", "size"),
            active_players=("player_name", "nunique"),
            total_distance=("total_distance", "sum"),
            hsr_hsd=("hsr_hsd", "sum"),
            number_of_sprints=("number_of_sprints", "sum"),
            speed_exposures=("speed_exposure_flag", "sum"),
            max_speed=("max_speed", "max"),
        )
        .reset_index()
    )
    category = pd.Categorical(grouped["session_category"], categories=["Training", "Match"], ordered=True)
    grouped["session_category"] = category
    grouped = grouped.sort_values("session_category").reset_index(drop=True)
    grouped["session_category"] = grouped["session_category"].astype(str)
    return grouped


def calculate_season_kpis(season_df: pd.DataFrame, weekly_df: pd.DataFrame) -> dict[str, object]:
    if season_df.empty:
        return {
            "players": float("nan"),
            "weeks": float("nan"),
            "total_distance": float("nan"),
            "hsr_hsd": float("nan"),
            "sprints": float("nan"),
            "duration_hours": float("nan"),
            "peak_week": float("nan"),
            "top_speed": float("nan"),
            "speed_exposures": float("nan"),
        }
    return {
        "players": float(season_df["player_name"].nunique()),
        "weeks": float(weekly_df["week_start"].nunique()) if not weekly_df.empty else float("nan"),
        "total_distance": float(season_df["total_distance"].sum()),
        "hsr_hsd": float(season_df["hsr_hsd"].sum()),
        "sprints": float(season_df["number_of_sprints"].sum()),
        "duration_hours": float(season_df["duration"].sum() / 60.0),
        "peak_week": float(weekly_df["total_distance"].max()) if not weekly_df.empty else float("nan"),
        "top_speed": float(season_df["max_speed"].max()) if not season_df["max_speed"].dropna().empty else float("nan"),
        "speed_exposures": float(season_df["speed_exposure_flag"].sum()),
    }


def build_team_alerts(weekly_df: pd.DataFrame, acwr_meta: dict[str, object]) -> list[str]:
    if weekly_df.empty:
        return ["Geen geldige teamdata beschikbaar."]

    latest = weekly_df.sort_values("week_start").tail(1).iloc[0]
    lines = [
        f"ACWR-modus actief: {acwr_meta['label']}.",
    ]
    if pd.notna(latest.get("total_distance_acwr")):
        lines.append(
            f"Laatste week total-distance ACWR: {_format_decimal(latest['total_distance_acwr'], 2)} ({_classify_acwr(latest['total_distance_acwr'])})."
        )
    if pd.notna(latest.get("hsr_hsd_acwr")):
        lines.append(
            f"Laatste week HSR/HSD ACWR: {_format_decimal(latest['hsr_hsd_acwr'], 2)} ({_classify_acwr(latest['hsr_hsd_acwr'])})."
        )
    if pd.notna(latest.get("speed_exposures")):
        lines.append(f"Laatste week speed exposures >=90% max: {_format_int(latest['speed_exposures'])}.")

    peak_week = weekly_df.loc[pd.to_numeric(weekly_df["total_distance"], errors="coerce").idxmax()]
    lines.append(f"Hoogste teamvolume: {_format_int(peak_week['total_distance'])} m in week {peak_week['week_start']:%d/%m/%Y}.")

    peak_hsr = weekly_df.loc[pd.to_numeric(weekly_df["hsr_hsd"], errors="coerce").idxmax()]
    lines.append(f"Hoogste team HSR/HSD: {_format_int(peak_hsr['hsr_hsd'])} m in week {peak_hsr['week_start']:%d/%m/%Y}.")

    median_players = pd.to_numeric(weekly_df["active_players"], errors="coerce").median()
    if pd.notna(median_players):
        lines.append(f"Typische weekbezetting: {_format_int(median_players)} spelers met geldige GPS-data.")

    return lines[:7]


def base_figure(title: str, height: int = 335) -> go.Figure:
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
    fig.update_xaxes(showgrid=False, tickfont=dict(color=MVV_TEXT_SOFT), tickangle=-45)
    fig.update_yaxes(gridcolor=MVV_GRID, zeroline=False, tickfont=dict(color=MVV_TEXT_SOFT))
    return fig


def build_bar_line_chart(
    df: pd.DataFrame,
    *,
    title: str,
    bar_column: str,
    bar_label: str,
    bar_color: str,
    line_column: str | None = None,
    line_label: str | None = None,
    line_color: str = MVV_RED_BRIGHT,
    primary_y_range: tuple[float, float] | None = None,
    error_column: str | None = None,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": bool(line_column)}]])
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=20, color=MVV_TEXT)),
        height=340,
        margin=dict(l=18, r=18, t=56, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color=MVV_TEXT, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    if df.empty:
        return fig

    error_values = None
    if error_column and error_column in df.columns:
        error_values = df[error_column].fillna(0)

    fig.add_trace(
        go.Bar(
            name=bar_label,
            x=df["week_label"],
            y=df[bar_column],
            marker_color=bar_color,
            error_y=dict(type="data", array=error_values, color=MVV_RED_BRIGHT, thickness=1.4) if error_values is not None else None,
        ),
        secondary_y=False,
    )
    if line_column:
        fig.add_trace(
            go.Scatter(
                name=line_label or line_column,
                x=df["week_label"],
                y=df[line_column],
                mode="lines+markers",
                marker=dict(size=7, color=line_color),
                line=dict(width=2.4, color=line_color),
            ),
            secondary_y=True,
        )

    fig.update_xaxes(showgrid=False, tickfont=dict(color=MVV_TEXT_SOFT), tickangle=-45)
    fig.update_yaxes(gridcolor=MVV_GRID, zeroline=False, tickfont=dict(color=MVV_TEXT_SOFT), secondary_y=False)
    fig.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(color=MVV_TEXT_SOFT), secondary_y=True)
    if primary_y_range is not None:
        fig.update_yaxes(range=list(primary_y_range), secondary_y=False)
    return fig


def build_grouped_bars_with_line_chart(
    df: pd.DataFrame,
    *,
    title: str,
    left_column: str,
    left_label: str,
    left_color: str,
    right_column: str,
    right_label: str,
    right_color: str,
    line_column: str,
    line_label: str,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=20, color=MVV_TEXT)),
        height=340,
        margin=dict(l=18, r=18, t=56, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color=MVV_TEXT, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        barmode="group",
    )
    if df.empty:
        return fig

    fig.add_trace(
        go.Bar(name=left_label, x=df["week_label"], y=df[left_column], marker_color=left_color),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(name=right_label, x=df["week_label"], y=df[right_column], marker_color=right_color),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name=line_label,
            x=df["week_label"],
            y=df[line_column],
            mode="lines+markers",
            marker=dict(size=7, color=MVV_RED_BRIGHT),
            line=dict(width=2.4, color=MVV_RED_BRIGHT),
        ),
        secondary_y=True,
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(color=MVV_TEXT_SOFT), tickangle=-45)
    fig.update_yaxes(gridcolor=MVV_GRID, zeroline=False, tickfont=dict(color=MVV_TEXT_SOFT), secondary_y=False)
    fig.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(color=MVV_TEXT_SOFT), secondary_y=True)
    return fig


def build_stacked_percentage_chart(df: pd.DataFrame) -> go.Figure:
    fig = base_figure("Team Distance Zone Distribution", height=340)
    if df.empty:
        return fig
    zone_columns = [
        ("walking", "Walking", "#F5D2D8"),
        ("jogging", "Jogging", "#F1A4B5"),
        ("running", "Running", "#E97A93"),
        ("sprint", "Z5", "#D92B4D"),
        ("high_sprint", "Z6", "#6E1222"),
    ]
    totals = pd.to_numeric(df["total_distance"], errors="coerce").replace(0, pd.NA)
    fig = go.Figure()
    for column, label, color in zone_columns:
        pct = pd.to_numeric(df[column], errors="coerce").div(totals).mul(100).fillna(0)
        fig.add_trace(
            go.Bar(
                name=label,
                x=df["week_label"],
                y=pct,
                marker_color=color,
                hovertemplate=f"%{{x}}<br>{label}: %{{y:.1f}}%<extra></extra>",
            )
        )
    fig.update_layout(
        title=dict(text="Team Distance Zone Distribution", x=0.02, xanchor="left", font=dict(size=20, color=MVV_TEXT)),
        height=340,
        margin=dict(l=18, r=18, t=56, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color=MVV_TEXT, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        barmode="stack",
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(color=MVV_TEXT_SOFT), tickangle=-45)
    fig.update_yaxes(gridcolor=MVV_GRID, zeroline=False, tickfont=dict(color=MVV_TEXT_SOFT), title="% van team distance")
    return fig


def build_leaderboard_chart(df: pd.DataFrame, column: str, title: str, formatter: Callable[[object], str]) -> go.Figure:
    fig = base_figure(title, height=360)
    if df.empty or column not in df.columns:
        return fig
    top_df = df.nlargest(8, column).sort_values(column, ascending=True)
    fig = go.Figure(
        data=[
            go.Bar(
                x=top_df[column],
                y=top_df["week_start"].dt.strftime("%d/%m/%Y"),
                orientation="h",
                marker_color=MVV_RED_DEEP,
                text=[formatter(value) for value in top_df[column]],
                textposition="outside",
                cliponaxis=False,
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=20, color=MVV_TEXT)),
        height=360,
        margin=dict(l=18, r=28, t=56, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color=MVV_TEXT, size=12),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=MVV_GRID, zeroline=False, tickfont=dict(color=MVV_TEXT_SOFT))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=MVV_TEXT_SOFT), automargin=True)
    return fig


def build_cards_html(kpis: dict[str, object], monitoring_summary: dict[str, object]) -> str:
    wellness_cards = [
        (
            label,
            _format_decimal(monitoring_summary[column], 1),
            f"Seizoensgemiddelde {label.lower()}",
        )
        for column, label in WELLNESS_PARAMETER_SPECS
    ]
    cards = [
        ("Players", _format_int(kpis["players"]), "Aantal unieke spelers met geldige GPS-data"),
        ("Weeks", _format_int(kpis["weeks"]), "Actieve GPS-weken in deze selectie"),
        ("Total Distance", _format_distance(kpis["total_distance"]), "Totaal teamvolume over de selectie"),
        ("HSR / HSD", _format_distance(kpis["hsr_hsd"]), "Sprint plus high sprint over de selectie"),
        ("Sprints", _format_int(kpis["sprints"]), "Totale sprintacties in de selectie"),
        ("Total Duration", _format_hours(kpis["duration_hours"]), "Opgetelde trainings- en matchduur"),
        ("Peak Week", _format_distance(kpis["peak_week"]), "Hoogste teamweek op total distance"),
        ("Top Speed", _format_speed(kpis["top_speed"]), "Hoogste geregistreerde topsnelheid"),
        *wellness_cards,
        ("Avg RPE", _format_decimal(monitoring_summary["avg_rpe"], 1), "Gewogen teamgemiddelde RPE over de selectie"),
        ("RPE Load", _format_int(monitoring_summary["rpe_load"]), "Opgetelde duration x RPE over de selectie"),
    ]
    return '<div class="year-report-card-grid">' + "".join(
        '<div class="year-report-card">'
        f'<div class="year-report-card-label">{escape(label)}</div>'
        f'<div class="year-report-card-value">{escape(value)}</div>'
        f'<div class="year-report-card-foot">{escape(foot)}</div>'
        "</div>"
        for label, value, foot in cards
    ) + "</div>"


def build_table_html(df: pd.DataFrame, columns: list[tuple[str, str, Callable[[object], str] | None]]) -> str:
    if df.empty:
        return '<div class="year-report-panel-subtitle">Geen data beschikbaar voor deze selectie.</div>'
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
    <div class="year-report-table-wrap">
      <table class="year-report-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{''.join(row_html)}</tbody>
      </table>
    </div>
    """


def render_panel_header(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f'<div class="year-report-panel-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="year-report-panel-anchor"></div>
        <div class="year-report-panel-title">{escape(title)}</div>
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

    with st.spinner("Year report data laden..."):
        all_df = fetch_summary_history_cached(access_token)

    if all_df.empty:
        st.info("Geen Summary GPS-data gevonden voor de jaarrapportage.")
        st.stop()

    season_years = sorted(all_df["season_start_year"].dropna().astype(int).unique().tolist(), reverse=True)
    if not season_years:
        st.info("Geen seizoenen gevonden in de GPS-data.")
        st.stop()

    acwr_meta = get_acwr_mode_meta()
    logo_markup = (
        f'<img src="{TEAM_LOGO_URI}" alt="MVV Maastricht" class="year-report-logo" />'
        if TEAM_LOGO_URI
        else ""
    )

    with st.container():
        st.markdown(
            f"""
            <div class="year-report-hero-anchor"></div>
            <div class="year-report-head">
              {logo_markup}
              <div class="year-report-copyhead">
                <h1 class="year-report-title">Year Report</h1>
                <div class="year-report-kicker">MVV Maastricht | Reports | Year Report</div>
              </div>
            </div>
            <div class="year-report-copy">
              Staff-overzicht van de seizoensrapportage in dashboardvorm, opgebouwd vanuit dezelfde team year-report structuur als in de rapportagemap maar nu direct filterbaar in de app.
            </div>
            """,
            unsafe_allow_html=True,
        )

        back_col, meta_col = st.columns([0.34, 1.66], gap="large")
        with back_col:
            if st.button("Open Reports", key="year_report_back", use_container_width=True):
                st.switch_page("pages/03_Reports_Page.py")
        with meta_col:
            st.markdown(
                f'<div class="year-report-filter-note">{len(season_years)} seizoenen beschikbaar | ACWR: {escape(str(acwr_meta["short_label"]))}</div>',
                unsafe_allow_html=True,
            )

        filter_col, detail_col = st.columns([1.2, 0.8], gap="large")
        with filter_col:
            st.markdown('<div class="year-report-filter-label">Seizoen</div>', unsafe_allow_html=True)
            selected_season_start = st.selectbox(
                "Seizoen",
                options=season_years,
                index=0,
                format_func=lambda value: f"{value}/{value + 1}",
                label_visibility="collapsed",
                key="year_report_selected_season",
            )
        with detail_col:
            st.markdown(
                f'<div class="year-report-filter-note">Periode: 01/07/{selected_season_start} - 30/06/{selected_season_start + 1}</div>',
                unsafe_allow_html=True,
            )

    season_df = all_df[all_df["season_start_year"] == int(selected_season_start)].copy()
    if season_df.empty:
        st.info("Geen data gevonden voor dit seizoen.")
        st.stop()

    season_start_date = season_df["datum"].min()
    season_end_date = season_df["datum"].max()
    player_lookup = (
        season_df.assign(player_id=season_df["player_id"].astype(str), player_name=season_df["player_name"].fillna("Onbekend").astype(str))
        .drop_duplicates(subset=["player_id"])
        .set_index("player_id")["player_name"]
        .to_dict()
    )
    monitoring_df = build_monitoring_dataset(
        SUPABASE_URL or "default",
        sb,
        season_start_date.date(),
        season_end_date.date(),
        player_ids=season_df["player_id"].astype(str).tolist(),
        player_lookup=player_lookup,
    )
    monitoring_summary = summarize_monitoring_dataset(monitoring_df)
    monitoring_weekly = build_monitoring_grouped_summary(monitoring_df, "week").rename(columns={"label": "week_label"})
    monitoring_players = build_monitoring_player_summary(monitoring_df)

    weekly_df = build_season_dataset(season_df, str(acwr_meta["mode"]))
    category_summary = build_category_summary(season_df)
    kpis = calculate_season_kpis(season_df, weekly_df)
    notes = build_team_alerts(weekly_df, acwr_meta)
    if monitoring_summary["wellness_entries"]:
        wellness_note = ", ".join(
            f"{label.lower()} {_format_decimal(monitoring_summary[column], 1)}"
            for column, label in WELLNESS_PARAMETER_SPECS
        )
        notes.append(
            f"Wellness gemiddeld over het seizoen: {wellness_note}."
        )
    if monitoring_summary["rpe_entries"]:
        notes.append(
            f"RPE gemiddeld over het seizoen: {_format_decimal(monitoring_summary['avg_rpe'], 1)} met totale RPE load {_format_int(monitoring_summary['rpe_load'])}."
        )

    badges = [
        f"Data periode: {season_start_date:%d/%m/%Y} - {season_end_date:%d/%m/%Y}",
        f"Actieve weken: {_format_int(kpis['weeks'])}",
        f"Speed exposures: {_format_int(kpis['speed_exposures'])}",
    ]
    if not weekly_df.empty:
        latest_week = weekly_df.iloc[-1]
        if pd.notna(latest_week.get("total_distance_acwr")):
            badges.append(f"Laatste TD ACWR: {_format_decimal(latest_week['total_distance_acwr'], 2)}")
        if pd.notna(latest_week.get("hsr_hsd_acwr")):
            badges.append(f"Laatste HSR ACWR: {_format_decimal(latest_week['hsr_hsd_acwr'], 2)}")
    st.markdown(
        '<div class="year-report-badge-row">' +
        "".join(f'<span class="year-report-badge">{escape(badge)}</span>' for badge in badges) +
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(build_cards_html(kpis, monitoring_summary), unsafe_allow_html=True)

    tab_trend, tab_load, tab_monitoring, tab_risk = st.tabs(["Trend", "Load Profile", "Wellness & RPE", "Risk & Insights"])

    with tab_trend:
        trend_row_one = st.columns(2, gap="large")
        with trend_row_one[0]:
            render_plot_panel(
                "Team Workload Trend",
                build_bar_line_chart(
                    weekly_df,
                    title="Weekly Team Total Distance and 4-Week Trend",
                    bar_column="total_distance",
                    bar_label="Team Distance",
                    bar_color=MVV_RED_DEEP,
                    line_column="total_distance_rolling4",
                    line_label="4-week rolling avg",
                ),
                "Seizoensvolume per week met voortschrijdend gemiddelde",
            )
        with trend_row_one[1]:
            render_plot_panel(
                "Squad Availability",
                build_bar_line_chart(
                    weekly_df,
                    title="Active GPS Players and Team Sessions",
                    bar_column="active_players",
                    bar_label="Active Players",
                    bar_color=MVV_RED_DEEP,
                    line_column="sessions",
                    line_label="Player Sessions",
                ),
                "Beschikbaarheid en aantal geregistreerde sessies per week",
            )

        trend_row_two = st.columns(2, gap="large")
        with trend_row_two[0]:
            render_plot_panel(
                "Distance per Player",
                build_bar_line_chart(
                    weekly_df,
                    title="Team Distance and Distance per Player",
                    bar_column="total_distance",
                    bar_label="Team Distance",
                    bar_color=MVV_RED_DEEP,
                    line_column="distance_per_player",
                    line_label="Distance per Player",
                ),
                "Teamtotaal en gemiddelde afstand per speler per week",
            )
        with trend_row_two[1]:
            render_plot_panel(
                "Top Weeks",
                build_leaderboard_chart(weekly_df, "total_distance", "Top Team Weeks by Total Distance", _format_distance),
                "Sterkste weken op basis van total distance",
            )

    with tab_load:
        load_row_one = st.columns(2, gap="large")
        with load_row_one[0]:
            render_plot_panel(
                "Team Speed Load",
                build_grouped_bars_with_line_chart(
                    weekly_df,
                    title="Weekly Team Speed Load",
                    left_column="sprint",
                    left_label="Dist. Z5",
                    left_color=MVV_RED_BRIGHT,
                    right_column="high_sprint",
                    right_label="Dist. Z6",
                    right_color=MVV_RED_DEEP,
                    line_column="hsr_hsd",
                    line_label="HSR / HSD",
                ),
                "Sprint- en high sprint-volume met gecombineerde high-speed load",
            )
        with load_row_one[1]:
            render_plot_panel(
                "Speed Exposure",
                build_bar_line_chart(
                    weekly_df,
                    title="Team Max Speed and Speed Exposures",
                    bar_column="max_speed",
                    bar_label="Team Max Speed",
                    bar_color=MVV_RED_DEEP,
                    line_column="speed_exposures",
                    line_label=">=90% max exposures",
                ),
                "Topsnelheid en blootstelling aan 90% van individuele max",
            )

        load_row_two = st.columns(2, gap="large")
        with load_row_two[0]:
            render_plot_panel(
                "Team Load Profile",
                build_stacked_percentage_chart(weekly_df),
                "Zoneverdeling van team distance over walking, jogging, running, Z5 en Z6",
            )
        with load_row_two[1]:
            render_plot_panel(
                "Neuromuscular Team Load",
                build_grouped_bars_with_line_chart(
                    weekly_df,
                    title="Accelerations and Decelerations",
                    left_column="total_accelerations",
                    left_label="Accelerations",
                    left_color=MVV_RED_BRIGHT,
                    right_column="total_decelerations",
                    right_label="Decelerations",
                    right_color=MVV_RED_DEEP,
                    line_column="accel_density",
                    line_label="Acc / 10 min",
                ),
                "Versnellingsbelasting en dichtheid van acceleraties",
            )

    with tab_monitoring:
        if monitoring_df.empty:
            st.info("Geen wellness- of RPE-data beschikbaar voor dit seizoen.")
        else:
            monitoring_specs = [
                ("muscle_soreness", "Muscle Soreness", "Wekelijks gemiddelde muscle soreness", MVV_RED_DEEP, (0, 10)),
                ("fatigue", "Fatigue", "Wekelijks gemiddelde fatigue", MVV_RED_BRIGHT, (0, 10)),
                ("sleep_quality", "Sleep Quality", "Wekelijks gemiddelde sleep quality", MVV_RED_DEEP, (0, 10)),
                ("stress", "Stress", "Wekelijks gemiddelde stress", MVV_RED_BRIGHT, (0, 10)),
                ("mood", "Mood", "Wekelijks gemiddelde mood", MVV_RED_DEEP, (0, 10)),
                ("avg_rpe", "Weighted RPE", "Gewogen team-RPE per week", MVV_RED_BRIGHT, (0, 10)),
                ("rpe_load", "RPE Load", "Totale duration x RPE per week met spreiding per speler", MVV_RED_DEEP, None),
            ]
            for idx in range(0, len(monitoring_specs), 2):
                cols = st.columns(2, gap="large")
                for col_container, spec in zip(cols, monitoring_specs[idx : idx + 2]):
                    column, label, subtitle, color, y_range = spec
                    with col_container:
                        render_plot_panel(
                            f"Weekly {label} +/- SD",
                            build_bar_line_chart(
                                monitoring_weekly,
                                title=f"Weekly Team {label}",
                                bar_column=column,
                                bar_label=label,
                                bar_color=color,
                                primary_y_range=y_range,
                                error_column=f"{column}_std",
                            ),
                            subtitle,
                        )

            render_html_panel(
                "Monitoring by Week",
                build_table_html(
                    monitoring_weekly,
                    [
                        ("week_label", "Week", None),
                        ("wellness_players", "Wellness Players", _format_int),
                        ("rpe_players", "RPE Players", _format_int),
                        ("muscle_soreness", "Muscle", _format_decimal),
                        ("fatigue", "Fatigue", _format_decimal),
                        ("sleep_quality", "Sleep", _format_decimal),
                        ("stress", "Stress", _format_decimal),
                        ("mood", "Mood", _format_decimal),
                        ("readiness_score", "Readiness", _format_decimal),
                        ("avg_rpe", "Avg RPE", _format_decimal),
                        ("rpe_load", "RPE Load", _format_int),
                    ],
                ),
                "Weekoverzicht van alle wellness-parameters, readiness en RPE",
            )

            render_html_panel(
                "Monitoring Players",
                build_table_html(
                    monitoring_players.head(16),
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
                        ("rpe_load", "RPE Load", _format_int),
                    ],
                ),
                "Top 16 spelers op basis van monitoringvolume in dit seizoen",
            )

    with tab_risk:
        risk_row_one = st.columns(2, gap="large")
        with risk_row_one[0]:
            render_plot_panel(
                "Team ACWR",
                build_bar_line_chart(
                    weekly_df,
                    title="Team Total Distance and ACWR",
                    bar_column="total_distance",
                    bar_label="Team Distance",
                    bar_color=MVV_RED_DEEP,
                    line_column="total_distance_acwr",
                    line_label="ACWR",
                ),
                f"ACWR volgens ingestelde modus: {acwr_meta['label']}",
            )
        with risk_row_one[1]:
            render_plot_panel(
                "Week-on-Week Change",
                build_bar_line_chart(
                    weekly_df,
                    title="Team HSR / HSD and Week-on-Week Change",
                    bar_column="hsr_hsd",
                    bar_label="HSR/HSD",
                    bar_color=MVV_RED_DEEP,
                    line_column="hsr_hsd_wow_change",
                    line_label="HSR WoW %",
                ),
                "Verandering in high-speed load per week",
            )

        risk_row_two = st.columns(2, gap="large")
        with risk_row_two[0]:
            render_html_panel(
                "Training vs Match",
                build_table_html(
                    category_summary,
                    [
                        ("session_category", "Type", None),
                        ("active_players", "Players", _format_int),
                        ("sessions", "Sessions", _format_int),
                        ("total_distance", "Distance", _format_distance),
                        ("hsr_hsd", "HSR/HSD", _format_distance),
                        ("number_of_sprints", "Sprints", _format_int),
                        ("speed_exposures", "90% Speed", _format_int),
                    ],
                ),
                "Vergelijking tussen trainings- en matchsessies over het seizoen",
            )
        with risk_row_two[1]:
            render_plot_panel(
                "Sprint Frequency",
                build_bar_line_chart(
                    weekly_df,
                    title="Team Sprints and Sprints per Player",
                    bar_column="number_of_sprints",
                    bar_label="Team Sprints",
                    bar_color=MVV_RED_DEEP,
                    line_column="sprints_per_player",
                    line_label="Sprints per Player",
                ),
                "Volume sprints en gemiddelde sprints per speler per week",
            )

        notes_html = (
            '<ul class="year-report-note-list">'
            + "".join(f"<li>{escape(note)}</li>" for note in notes)
            + "</ul>"
            + '<div class="year-report-note-foot">Analyse is gebaseerd op Summary-sessies wanneer beschikbaar; ontbrekende metrics worden niet geschat.</div>'
        )
        render_html_panel(
            "Team Analyst Notes",
            notes_html,
            "Compacte seizoenssamenvatting voor staffbespreking",
        )

    render_sidebar_footer(profile)


if __name__ == "__main__":
    main()
