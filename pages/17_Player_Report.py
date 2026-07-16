from __future__ import annotations

import re
from datetime import date
from html import escape
from pathlib import Path
from typing import Callable

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

from auth_session import ensure_auth_restored, get_sb_client
from pages.Subscripts.mvv_branding import TEAM_HERO_BG, TEAM_LOGO, build_data_uri
from player_report_pdf import build_player_report_pdf_bytes
from report_monitoring import WELLNESS_PARAMETER_SPECS, build_monitoring_dataset, build_monitoring_grouped_summary, summarize_monitoring_dataset
from roles import get_profile, is_staff_user, pick_target_player, render_sidebar_footer, render_sidebar_navigation, require_auth
from utils.streamlit_ui import apply_streamlit_chrome


st.set_page_config(page_title="Player Report", layout="wide", initial_sidebar_state="expanded")
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
    "number_of_repeated_sprints",
    "hrtrimp",
    "avg_hr",
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
    "number_of_repeated_sprints",
    "hrtrimp",
    "avg_hr",
]

RECENT_SESSION_PERIOD_LABELS = {
    "current_scope": "Huidige selectie",
    "last_14_days": "Laatste 14 dagen",
    "last_30_days": "Laatste 30 dagen",
    "last_6_weeks": "Laatste 6 weken",
    "last_3_months": "Laatste 3 maanden",
    "full_history": "Volledige historie",
}


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
          background: __PLAYER_REPORT_BG__;
          background-size: cover;
          background-position: center top;
          background-attachment: fixed;
        }

        .block-container {
          max-width: 1380px;
          padding-top: 1.25rem;
          padding-bottom: 2.4rem;
        }

        div[data-testid="stVerticalBlock"]:has(.player-report-hero-anchor) {
          padding: 1.75rem 1.6rem 1.3rem 1.6rem;
          border-radius: 10px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(135deg, rgba(18, 25, 42, 0.88), rgba(10, 15, 27, 0.84));
          box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22);
          margin-bottom: 1.1rem;
        }

        div[data-testid="stVerticalBlock"]:has(.player-report-panel-anchor) {
          padding: 1rem 1rem 0.8rem 1rem;
          border-radius: 10px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
          box-shadow: 0 14px 26px rgba(0, 0, 0, 0.18);
          margin-bottom: 1rem;
        }

        .player-report-hero-anchor,
        .player-report-panel-anchor {
          height: 0;
        }

        .player-report-head {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 1rem;
          margin-bottom: 1rem;
        }

        .player-report-logo {
          width: 78px;
          height: 78px;
          object-fit: contain;
          flex-shrink: 0;
          filter: drop-shadow(0 8px 22px rgba(0,0,0,0.28));
        }

        .player-report-copyhead {
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 0.12rem;
          text-align: left;
        }

        .player-report-kicker {
          color: rgba(255,255,255,0.76);
          font-size: 0.74rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.18em;
          margin-bottom: 0;
        }

        .player-report-title {
          margin: 0;
          font-size: 2.45rem;
          line-height: 1;
          font-weight: 800;
          color: #ffffff;
        }

        .player-report-copy {
          margin-top: 0.85rem;
          max-width: 82ch;
          color: rgba(255,255,255,0.84);
          line-height: 1.6;
        }

        .player-report-filter-label {
          color: rgba(255,255,255,0.92);
          font-size: 0.92rem;
          font-weight: 700;
          margin-bottom: 0.35rem;
        }

        .player-report-filter-note {
          color: rgba(255,255,255,0.80);
          font-size: 0.88rem;
          font-weight: 700;
          text-align: right;
          margin-top: 2rem;
        }

        .player-report-badge-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
          margin-top: 1rem;
        }

        .player-report-badge {
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

        .player-report-card-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 1rem;
          margin: 0.2rem 0 1.15rem 0;
        }

        .player-report-card {
          border-radius: 10px;
          border: 1px solid rgba(234, 51, 81, 0.14);
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
          box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
          padding: 1rem 1rem 0.9rem 1rem;
          min-height: 132px;
        }

        .player-report-card-label {
          color: rgba(255,255,255,0.62);
          font-size: 0.72rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }

        .player-report-card-value {
          margin-top: 0.55rem;
          color: #ffffff;
          font-size: 2rem;
          line-height: 1;
          font-weight: 800;
        }

        .player-report-card-foot {
          margin-top: 0.72rem;
          color: rgba(255,255,255,0.76);
          line-height: 1.45;
          font-size: 0.84rem;
        }

        .player-report-panel-title {
          color: #ffffff;
          font-size: 1.08rem;
          line-height: 1.2;
          font-weight: 800;
          margin-bottom: 0.2rem;
        }

        .player-report-panel-subtitle {
          color: rgba(255,255,255,0.70);
          font-size: 0.84rem;
          margin-bottom: 0.85rem;
        }

        .player-report-table-wrap {
          overflow-x: auto;
        }

        .player-report-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.88rem;
        }

        .player-report-table thead th {
          text-align: left;
          padding: 0.8rem 0.8rem;
          color: rgba(255,255,255,0.68);
          font-size: 0.73rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          border-bottom: 1px solid rgba(255,255,255,0.10);
        }

        .player-report-table tbody td {
          padding: 0.76rem 0.8rem;
          color: rgba(255,255,255,0.90);
          border-bottom: 1px solid rgba(255,255,255,0.06);
          white-space: nowrap;
        }

        .player-report-note-list {
          margin: 0.25rem 0 0 0;
          padding-left: 1.1rem;
          color: rgba(255,255,255,0.90);
          line-height: 1.65;
        }

        .player-report-note-foot {
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
          .player-report-card-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
        }

        @media (max-width: 768px) {
          div[data-testid="stVerticalBlock"]:has(.player-report-hero-anchor) {
            padding: 1.35rem 1rem 1rem 1rem;
          }

          .player-report-head {
            flex-direction: column;
            gap: 0.8rem;
          }

          .player-report-copyhead {
            text-align: center;
          }

          .player-report-title {
            font-size: 2rem;
          }

          .player-report-card-grid {
            grid-template-columns: repeat(1, minmax(0, 1fr));
          }

          .player-report-filter-note {
            text-align: left;
            margin-top: 0.25rem;
          }
        }
        </style>
        """.replace("__PLAYER_REPORT_BG__", background),
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


def _format_minutes(value: object) -> str:
    base = _format_int(value)
    return "--" if base == "--" else f"{base} min"


def _safe_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return sanitized.strip("_") or "report"


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
    df["player_id"] = df["player_id"].astype(str)
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce").dt.normalize()
    df["player_name"] = df["player_name"].fillna("Onbekend").astype(str).str.strip()
    df["type"] = df["type"].fillna("").astype(str).str.strip()
    df["event"] = df["event"].fillna("").astype(str).str.strip()

    for column in SUM_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df["max_speed"] = pd.to_numeric(df.get("max_speed"), errors="coerce")
    df = df.dropna(subset=["datum"]).copy()
    df["hsr_hsd"] = df["sprint"].fillna(0.0) + df["high_sprint"].fillna(0.0)
    df["session_category"] = df["type"].apply(_session_category)
    df["week_start"] = (df["datum"] - pd.to_timedelta(df["datum"].dt.weekday, unit="D")).dt.normalize()
    df["month_start"] = df["datum"].dt.to_period("M").dt.to_timestamp()
    df["season_start_year"] = df["datum"].dt.year.where(df["datum"].dt.month >= 7, df["datum"].dt.year - 1)
    df["season_label"] = df["season_start_year"].astype(int).astype(str) + "/" + (df["season_start_year"] + 1).astype(int).astype(str)
    season_max_speed = df.groupby("player_id")["max_speed"].transform("max")
    df["speed_exposure_flag"] = season_max_speed.gt(0) & df["max_speed"].ge(season_max_speed * 0.9)
    return df


def _week_label(week_start: pd.Timestamp) -> str:
    iso = week_start.isocalendar()
    week_end = week_start + pd.Timedelta(days=6)
    return f"{iso.year}-W{int(iso.week):02d} | {week_start:%d/%m/%Y} - {week_end:%d/%m/%Y}"


def _month_label(month_start: pd.Timestamp) -> str:
    month_start = pd.Timestamp(month_start).normalize()
    month_end = month_start + pd.offsets.MonthEnd(0)
    return f"{month_start:%Y-%m} | {month_start:%d/%m/%Y} - {month_end:%d/%m/%Y}"


def _season_label(season_start_year: int) -> str:
    return f"{season_start_year}/{season_start_year + 1}"


def build_scope_summary(scope_df: pd.DataFrame) -> dict[str, object]:
    if scope_df.empty:
        return {
            "sessions": float("nan"),
            "active_days": float("nan"),
            "training_sessions": float("nan"),
            "match_sessions": float("nan"),
            "total_distance": float("nan"),
            "hsr_hsd": float("nan"),
            "sprints": float("nan"),
            "duration_min": float("nan"),
            "distance_per_min": float("nan"),
            "top_speed": float("nan"),
            "speed_exposures": float("nan"),
            "accel_density": float("nan"),
        }

    total_duration = float(scope_df["duration"].sum())
    total_distance = float(scope_df["total_distance"].sum())
    total_accelerations = float(scope_df["total_accelerations"].sum())
    return {
        "sessions": float(len(scope_df)),
        "active_days": float(scope_df["datum"].nunique()),
        "training_sessions": float(scope_df["session_category"].eq("Training").sum()),
        "match_sessions": float(scope_df["session_category"].eq("Match").sum()),
        "total_distance": total_distance,
        "hsr_hsd": float(scope_df["hsr_hsd"].sum()),
        "sprints": float(scope_df["number_of_sprints"].sum()),
        "duration_min": total_duration,
        "distance_per_min": (total_distance / total_duration) if total_duration > 0 else float("nan"),
        "top_speed": float(scope_df["max_speed"].max()) if scope_df["max_speed"].notna().any() else float("nan"),
        "speed_exposures": float(scope_df["speed_exposure_flag"].sum()),
        "accel_density": (total_accelerations / total_duration * 10) if total_duration > 0 else float("nan"),
    }


def build_period_table(scope_df: pd.DataFrame, scope_mode: str) -> pd.DataFrame:
    if scope_df.empty:
        return pd.DataFrame()

    if scope_mode == "Year":
        grouped = (
            scope_df.groupby("week_start", dropna=False)
            .agg(
                sessions=("datum", "size"),
                total_distance=("total_distance", "sum"),
                hsr_hsd=("hsr_hsd", "sum"),
                number_of_sprints=("number_of_sprints", "sum"),
                duration=("duration", "sum"),
                max_speed=("max_speed", "max"),
            )
            .reset_index()
            .sort_values("week_start")
            .reset_index(drop=True)
        )
        grouped["label"] = grouped["week_start"].apply(lambda value: f"W{int(pd.Timestamp(value).isocalendar().week):02d} | {pd.Timestamp(value):%d/%m}")
        grouped["total_distance_rolling4"] = pd.to_numeric(grouped["total_distance"], errors="coerce").rolling(4, min_periods=1).mean()
    else:
        grouped = (
            scope_df.groupby("datum", dropna=False)
            .agg(
                sessions=("datum", "size"),
                total_distance=("total_distance", "sum"),
                hsr_hsd=("hsr_hsd", "sum"),
                number_of_sprints=("number_of_sprints", "sum"),
                duration=("duration", "sum"),
                max_speed=("max_speed", "max"),
            )
            .reset_index()
            .sort_values("datum")
            .reset_index(drop=True)
        )
        grouped["label"] = grouped["datum"].dt.strftime("%d/%m")
        grouped["total_distance_rolling4"] = pd.NA

    grouped["distance_per_min"] = _safe_divide(grouped["total_distance"], grouped["duration"])
    return grouped


def build_type_table(scope_df: pd.DataFrame) -> pd.DataFrame:
    if scope_df.empty:
        return pd.DataFrame()
    grouped = (
        scope_df.groupby("session_category", dropna=False)
        .agg(
            sessions=("datum", "size"),
            total_distance=("total_distance", "sum"),
            hsr_hsd=("hsr_hsd", "sum"),
            sprints=("number_of_sprints", "sum"),
            max_speed=("max_speed", "max"),
            duration=("duration", "sum"),
        )
        .reset_index()
        .sort_values("session_category")
        .reset_index(drop=True)
    )
    grouped["distance_per_min"] = _safe_divide(grouped["total_distance"], grouped["duration"])
    return grouped


def build_zone_totals(scope_df: pd.DataFrame) -> pd.DataFrame:
    zone_map = [
        ("walking", "Walking"),
        ("jogging", "Jogging"),
        ("running", "Running"),
        ("sprint", "Sprint"),
        ("high_sprint", "High Sprint"),
    ]
    rows = []
    for column, label in zone_map:
        rows.append({"zone": label, "value": float(pd.to_numeric(scope_df[column], errors="coerce").fillna(0).sum())})
    return pd.DataFrame(rows)


def build_sessions_table(scope_df: pd.DataFrame) -> pd.DataFrame:
    if scope_df.empty:
        return pd.DataFrame()
    out = scope_df.copy().sort_values(["datum", "total_distance"], ascending=[False, False]).reset_index(drop=True)
    out["datum_label"] = out["datum"].dt.strftime("%d/%m/%Y")
    return out[
        [
            "datum_label",
            "type",
            "event",
            "total_distance",
            "hsr_hsd",
            "number_of_sprints",
            "duration",
            "max_speed",
        ]
    ]


def filter_recent_sessions(
    player_df: pd.DataFrame,
    scope_df: pd.DataFrame,
    scope_mode: str,
    scope_end: pd.Timestamp,
    period_key: str,
) -> tuple[pd.DataFrame, str]:
    scope_label = {"Week": "weekselectie", "Month": "maandselectie", "Year": "seizoensselectie"}.get(scope_mode, "selectie")
    if period_key == "current_scope":
        return scope_df.copy(), f"Binnen de huidige {scope_label}"
    if period_key == "full_history":
        return player_df.copy(), "Volledige spelerhistorie"

    window_days = {
        "last_14_days": 14,
        "last_30_days": 30,
        "last_6_weeks": 42,
        "last_3_months": 90,
    }.get(period_key, 30)
    end_point = pd.Timestamp(scope_end).normalize()
    start_point = end_point - pd.Timedelta(days=window_days - 1)
    filtered = player_df[(player_df["datum"] >= start_point) & (player_df["datum"] <= end_point)].copy()
    return filtered, f"{start_point:%d/%m/%Y} t/m {end_point:%d/%m/%Y}"


def build_player_notes(
    scope_df: pd.DataFrame,
    summary: dict[str, object],
    monitoring_summary: dict[str, object],
    scope_mode: str,
) -> list[str]:
    if scope_df.empty:
        return ["Geen GPS-data beschikbaar voor deze selectie."]

    notes: list[str] = []
    scope_label = {"Week": "week", "Month": "maand", "Year": "seizoen"}.get(scope_mode, "periode")
    notes.append(
        f"In deze {scope_label} staan {_format_int(summary['sessions'])} Summary-sessies over {_format_int(summary['active_days'])} actieve dagen."
    )
    notes.append(
        f"Totale belasting: {_format_distance(summary['total_distance'])}, HSR/HSD {_format_distance(summary['hsr_hsd'])}, sprints {_format_int(summary['sprints'])} en gemiddelde intensiteit {_format_decimal(summary['distance_per_min'], 1)} m/min."
    )

    peak_row = scope_df.sort_values("total_distance", ascending=False).head(1)
    if not peak_row.empty:
        row = peak_row.iloc[0]
        notes.append(
            f"Piekmoment: {row['datum']:%d-%m-%Y} ({row.get('type') or 'Sessie'}) met {_format_distance(row.get('total_distance'))} en topsnelheid {_format_speed(row.get('max_speed'))}."
        )

    if monitoring_summary["wellness_entries"]:
        wellness_note = ", ".join(
            f"{label.lower()} {_format_decimal(monitoring_summary[column], 1)}"
            for column, label in WELLNESS_PARAMETER_SPECS
        )
        notes.append(
            f"Welnessmonitoring: {wellness_note} op basis van {_format_int(monitoring_summary['wellness_entries'])} entries."
        )
    else:
        notes.append("Geen wellness-invoer gevonden binnen deze selectie.")

    if monitoring_summary["rpe_entries"]:
        notes.append(
            f"Interne load: gemiddelde RPE {_format_decimal(monitoring_summary['avg_rpe'], 1)} en totale RPE load {_format_int(monitoring_summary['rpe_load'])}."
        )
    else:
        notes.append("Geen RPE-invoer gevonden binnen deze selectie.")

    return notes[:6]


def base_figure(title: str, height: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=20, color=MVV_TEXT)),
        height=height,
        margin=dict(l=18, r=18, t=56, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color=MVV_TEXT, size=12),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(color=MVV_TEXT_SOFT), tickangle=-35)
    fig.update_yaxes(gridcolor=MVV_GRID, zeroline=False, tickfont=dict(color=MVV_TEXT_SOFT))
    return fig


def build_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color: str,
    formatter: Callable[[object], str],
    hover_format: str = ":,.0f",
    y_range: tuple[float, float] | None = None,
    error_col: str | None = None,
    height: int = 340,
) -> go.Figure:
    fig = base_figure(title, height=height)
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return fig
    error_values = None
    if error_col and error_col in df.columns:
        error_values = df[error_col].fillna(0)
    fig.add_trace(
        go.Bar(
            x=df[x_col],
            y=df[y_col],
            marker_color=color,
            error_y=dict(type="data", array=error_values, color=MVV_RED_BRIGHT, thickness=1.4) if error_values is not None else None,
            text=[formatter(value) for value in df[y_col]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"%{{x}}<br>%{{y{hover_format}}}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    return fig


def build_bar_line_chart(period_df: pd.DataFrame, scope_mode: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        title=dict(text="Workload Trend", x=0.02, xanchor="left", font=dict(size=20, color=MVV_TEXT)),
        height=340,
        margin=dict(l=18, r=18, t=56, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color=MVV_TEXT, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    if period_df.empty:
        return fig

    line_column = "total_distance_rolling4" if scope_mode == "Year" else "hsr_hsd"
    line_label = "4-period avg" if scope_mode == "Year" else "HSR / HSD"
    fig.add_trace(
        go.Bar(
            name="Total Distance",
            x=period_df["label"],
            y=period_df["total_distance"],
            marker_color=MVV_RED_DEEP,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name=line_label,
            x=period_df["label"],
            y=period_df[line_column],
            mode="lines+markers",
            marker=dict(size=7, color=MVV_RED_BRIGHT),
            line=dict(width=2.4, color=MVV_RED_BRIGHT),
        ),
        secondary_y=True,
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(color=MVV_TEXT_SOFT), tickangle=-35)
    fig.update_yaxes(gridcolor=MVV_GRID, zeroline=False, tickfont=dict(color=MVV_TEXT_SOFT), secondary_y=False)
    fig.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(color=MVV_TEXT_SOFT), secondary_y=True)
    return fig


def build_zone_share_chart(zone_df: pd.DataFrame) -> go.Figure:
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


def build_leaderboard_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = base_figure(title, height=360)
    if df.empty:
        return fig
    top_df = df.nlargest(10, "total_distance").sort_values("total_distance", ascending=True)
    fig.add_trace(
        go.Bar(
            x=top_df["total_distance"],
            y=top_df["datum_label"] + " | " + top_df["type"].fillna("").astype(str),
            orientation="h",
            marker_color=MVV_RED_DEEP,
            text=[_format_distance(value) for value in top_df["total_distance"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:,.0f} m<extra></extra>",
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
            f"Gemiddelde {label.lower()} in deze selectie",
        )
        for column, label in WELLNESS_PARAMETER_SPECS
    ]
    cards = [
        ("Sessies", _format_int(summary["sessions"]), "Aantal Summary-sessies in de selectie"),
        ("Actieve dagen", _format_int(summary["active_days"]), "Unieke dagen met GPS-data"),
        ("Total Distance", _format_distance(summary["total_distance"]), "Opgeteld volume binnen de selectie"),
        ("HSR / HSD", _format_distance(summary["hsr_hsd"]), "Sprint + high sprint distance"),
        ("Sprints", _format_int(summary["sprints"]), "Totale sprintacties in de selectie"),
        ("Duur", _format_minutes(summary["duration_min"]), "Opgetelde sessieduur"),
        ("Avg Intensity", _format_decimal(summary["distance_per_min"], 1), "Gemiddelde meters per minuut"),
        ("Top Speed", _format_speed(summary["top_speed"]), "Hoogste geregistreerde topsnelheid"),
        *wellness_cards,
        ("Avg RPE", _format_decimal(monitoring_summary["avg_rpe"], 1), "Gewogen gemiddelde RPE"),
        ("RPE Load", _format_int(monitoring_summary["rpe_load"]), "Opgetelde duration x RPE"),
    ]
    return '<div class="player-report-card-grid">' + "".join(
        '<div class="player-report-card">'
        f'<div class="player-report-card-label">{escape(label)}</div>'
        f'<div class="player-report-card-value">{escape(value)}</div>'
        f'<div class="player-report-card-foot">{escape(foot)}</div>'
        "</div>"
        for label, value, foot in cards
    ) + "</div>"


def build_table_html(df: pd.DataFrame, columns: list[tuple[str, str, Callable[[object], str] | None]]) -> str:
    if df.empty:
        return '<div class="player-report-panel-subtitle">Geen data beschikbaar voor deze selectie.</div>'
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
    <div class="player-report-table-wrap">
      <table class="player-report-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{''.join(row_html)}</tbody>
      </table>
    </div>
    """


def render_panel_header(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f'<div class="player-report-panel-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="player-report-panel-anchor"></div>
        <div class="player-report-panel-title">{escape(title)}</div>
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
    if profile and not is_staff_user(profile) and str(profile.get("role", "")).lower() != "player":
        st.error("Geen toegang tot player reports.")
        st.stop()

    render_sidebar_navigation(profile)

    with st.spinner("Player report data laden..."):
        all_df = fetch_summary_history_cached(access_token)

    if all_df.empty:
        st.info("Geen Summary GPS-data gevonden voor player reports.")
        st.stop()

    logo_markup = (
        f'<img src="{TEAM_LOGO_URI}" alt="MVV Maastricht" class="player-report-logo" />'
        if TEAM_LOGO_URI
        else ""
    )

    with st.container():
        st.markdown(
            f"""
            <div class="player-report-hero-anchor"></div>
            <div class="player-report-head">
              {logo_markup}
              <div class="player-report-copyhead">
                <h1 class="player-report-title">Player Report</h1>
                <div class="player-report-kicker">MVV Maastricht | Reports | Player Report</div>
              </div>
            </div>
            <div class="player-report-copy">
              Individuele rapportage voor dezelfde week-, maand- en seizoenslogica als de teamreports, maar nu per speler met wellness, RPE, GPS-trend en PDF-export.
            </div>
            """,
            unsafe_allow_html=True,
        )

        player_col, scope_col = st.columns([1.25, 0.75], gap="large")
        with player_col:
            st.markdown('<div class="player-report-filter-label">Speler</div>', unsafe_allow_html=True)
            target_player_id, target_player_name, _ = pick_target_player(
                sb,
                profile,
                label="Speler",
                key="player_report_target_player",
            )
        with scope_col:
            st.markdown('<div class="player-report-filter-label">Rapporttype</div>', unsafe_allow_html=True)
            scope_mode = st.selectbox(
                "Rapporttype",
                options=["Week", "Month", "Year"],
                index=0,
                label_visibility="collapsed",
                key="player_report_scope_mode",
            )

    if not target_player_id or not target_player_name:
        st.info("Geen speler beschikbaar voor deze rapportage.")
        st.stop()

    player_df = all_df[all_df["player_id"].astype(str) == str(target_player_id)].copy()
    if player_df.empty:
        st.info("Geen GPS Summary-data gevonden voor deze speler.")
        st.stop()

    period_col, note_col = st.columns([1.25, 0.75], gap="large")
    scope_start: pd.Timestamp
    scope_end: pd.Timestamp
    period_label: str

    if scope_mode == "Week":
        week_options = sorted(player_df["week_start"].dropna().unique().tolist(), reverse=True)
        with period_col:
            st.markdown('<div class="player-report-filter-label">Week</div>', unsafe_allow_html=True)
            selected_period = st.selectbox(
                "Week",
                options=week_options,
                index=0,
                format_func=lambda value: _week_label(pd.Timestamp(value)),
                label_visibility="collapsed",
                key="player_report_week",
            )
        scope_start = pd.Timestamp(selected_period).normalize()
        scope_end = scope_start + pd.Timedelta(days=6)
        period_label = _week_label(scope_start)
        scope_df = player_df[player_df["week_start"] == scope_start].copy()
        monitoring_period = "day"
    elif scope_mode == "Month":
        month_options = sorted(player_df["month_start"].dropna().unique().tolist(), reverse=True)
        with period_col:
            st.markdown('<div class="player-report-filter-label">Maand</div>', unsafe_allow_html=True)
            selected_period = st.selectbox(
                "Maand",
                options=month_options,
                index=0,
                format_func=lambda value: _month_label(pd.Timestamp(value)),
                label_visibility="collapsed",
                key="player_report_month",
            )
        scope_start = pd.Timestamp(selected_period).normalize()
        scope_end = scope_start + pd.offsets.MonthEnd(0)
        period_label = _month_label(scope_start)
        scope_df = player_df[player_df["month_start"] == scope_start].copy()
        monitoring_period = "day"
    else:
        year_options = sorted(player_df["season_start_year"].dropna().astype(int).unique().tolist(), reverse=True)
        with period_col:
            st.markdown('<div class="player-report-filter-label">Seizoen</div>', unsafe_allow_html=True)
            selected_period = st.selectbox(
                "Seizoen",
                options=year_options,
                index=0,
                format_func=_season_label,
                label_visibility="collapsed",
                key="player_report_year",
            )
        scope_df = player_df[player_df["season_start_year"] == int(selected_period)].copy()
        scope_start = scope_df["datum"].min()
        scope_end = scope_df["datum"].max()
        period_label = _season_label(int(selected_period))
        monitoring_period = "week"

    if scope_df.empty:
        st.info("Geen data gevonden voor deze selectie.")
        st.stop()

    with note_col:
        st.markdown(
            f'<div class="player-report-filter-note">{target_player_name} | {escape(period_label)}</div>',
            unsafe_allow_html=True,
        )

    monitoring_df = build_monitoring_dataset(
        SUPABASE_URL or "default",
        sb,
        scope_start.date(),
        scope_end.date(),
        player_ids=[str(target_player_id)],
        player_lookup={str(target_player_id): str(target_player_name)},
    )
    monitoring_summary = summarize_monitoring_dataset(monitoring_df)
    monitoring_group_df = build_monitoring_grouped_summary(monitoring_df, monitoring_period)

    summary = build_scope_summary(scope_df)
    period_df = build_period_table(scope_df, scope_mode)
    type_table = build_type_table(scope_df)
    zone_df = build_zone_totals(scope_df)
    sessions_table = build_sessions_table(scope_df)
    notes = build_player_notes(scope_df, summary, monitoring_summary, scope_mode)

    max_recent_session_count = max(1, len(player_df.index))
    recent_session_count = int(st.session_state.get("player_report_recent_session_count", min(15, max_recent_session_count)))
    recent_session_count = max(1, min(recent_session_count, max_recent_session_count))
    recent_session_period_key = str(st.session_state.get("player_report_recent_session_period", "current_scope"))
    if recent_session_period_key not in RECENT_SESSION_PERIOD_LABELS:
        recent_session_period_key = "current_scope"

    recent_sessions_source_df, recent_sessions_period_label = filter_recent_sessions(
        player_df,
        scope_df,
        scope_mode,
        scope_end,
        recent_session_period_key,
    )
    recent_sessions_table = build_sessions_table(recent_sessions_source_df)
    recent_sessions_preview = recent_sessions_table.head(recent_session_count).copy()
    recent_sessions_visible = min(recent_session_count, len(recent_sessions_table.index))
    recent_sessions_subtitle = (
        f"Toont {_format_int(recent_sessions_visible)} van {_format_int(len(recent_sessions_table.index))} "
        f"Summary-sessies | {recent_sessions_period_label}"
    )

    badges = [
        f"Speler: {target_player_name}",
        f"Scope: {scope_mode}",
        f"{_format_int(summary['sessions'])} sessies",
        f"{_format_int(summary['active_days'])} actieve dagen",
    ]
    if pd.notna(summary["distance_per_min"]):
        badges.append(f"Intensiteit: {_format_decimal(summary['distance_per_min'], 1)} m/min")
    if pd.notna(summary["top_speed"]):
        badges.append(f"Top speed: {_format_speed(summary['top_speed'])}")
    st.markdown(
        '<div class="player-report-badge-row">' +
        "".join(f'<span class="player-report-badge">{escape(badge)}</span>' for badge in badges) +
        "</div>",
        unsafe_allow_html=True,
    )

    pdf_error = None
    pdf_bytes: bytes | None = None
    try:
        pdf_bytes = build_player_report_pdf_bytes(
            player_name=str(target_player_name),
            scope_label=scope_mode,
            period_label=period_label,
            summary=summary,
            monitoring_summary=monitoring_summary,
            sessions_df=recent_sessions_preview,
            monitoring_group_df=monitoring_group_df,
            notes=notes,
        )
    except Exception as exc:
        pdf_error = str(exc)

    action_cols = st.columns([0.42, 0.42, 1.16], gap="large")
    with action_cols[0]:
        if st.button("Open Reports", key="player_report_back", use_container_width=True):
            st.switch_page("pages/03_Reports_Page.py")
    with action_cols[1]:
        if pdf_bytes:
            file_name = f"{_safe_filename(target_player_name)}_{scope_mode.lower()}_report.pdf"
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=file_name,
                mime="application/pdf",
                use_container_width=True,
                key="player_report_pdf_download",
            )
    with action_cols[2]:
        if pdf_error:
            st.warning(f"PDF-export is nog niet beschikbaar: {pdf_error}")

    st.markdown(build_cards_html(summary, monitoring_summary), unsafe_allow_html=True)

    tab_overview, tab_load, tab_monitoring, tab_sessions = st.tabs(["Overview", "Load Profile", "Wellness & RPE", "Sessions & Notes"])

    with tab_overview:
        overview_left, overview_right = st.columns(2, gap="large")
        with overview_left:
            render_plot_panel(
                "Workload Trend",
                build_bar_line_chart(period_df, scope_mode),
                f"{target_player_name} | {period_label}",
            )
        with overview_right:
            render_plot_panel(
                "Intensity Trend",
                build_bar_chart(
                    period_df,
                    "label",
                    "distance_per_min",
                    "Distance per Minute",
                    MVV_RED_BRIGHT,
                    _format_decimal,
                    hover_format=":.1f",
                ),
                "Intensiteit per dag of week binnen de gekozen scope",
            )

        render_html_panel(
            "Period Table",
            build_table_html(
                period_df,
                [
                    ("label", "Periode", None),
                    ("sessions", "Sessies", _format_int),
                    ("total_distance", "Distance", _format_distance),
                    ("hsr_hsd", "HSR/HSD", _format_distance),
                    ("number_of_sprints", "Sprints", _format_int),
                    ("distance_per_min", "m/min", _format_decimal),
                    ("max_speed", "Top Speed", _format_speed),
                ],
            ),
            "Belangrijkste GPS-uitkomsten per dag of week",
        )

    with tab_load:
        load_left, load_right = st.columns(2, gap="large")
        with load_left:
            render_plot_panel(
                "Distance Zone Share",
                build_zone_share_chart(zone_df),
                "Verdeling van walking, jogging, running, sprint en high sprint",
            )
        with load_right:
            render_html_panel(
                "Training vs Match",
                build_table_html(
                    type_table,
                    [
                        ("session_category", "Type", None),
                        ("sessions", "Sessies", _format_int),
                        ("total_distance", "Distance", _format_distance),
                        ("hsr_hsd", "HSR/HSD", _format_distance),
                        ("sprints", "Sprints", _format_int),
                        ("distance_per_min", "m/min", _format_decimal),
                        ("max_speed", "Top Speed", _format_speed),
                    ],
                ),
                "Vergelijking tussen trainings- en wedstrijdbelasting voor deze speler",
            )

        render_plot_panel(
            "Top Sessions by Distance",
            build_leaderboard_chart(sessions_table, "Top 10 Sessions by Distance"),
            "Piekbelastingen binnen de huidige scope",
        )

    with tab_monitoring:
        if monitoring_df.empty:
            st.info("Geen wellness- of RPE-data beschikbaar voor deze selectie.")
        else:
            monitoring_specs = [
                ("muscle_soreness", "Muscle Soreness", "Gemiddelde muscle soreness", MVV_RED_DEEP, _format_decimal, ":.1f", (0, 10)),
                ("fatigue", "Fatigue", "Gemiddelde fatigue", MVV_RED_BRIGHT, _format_decimal, ":.1f", (0, 10)),
                ("sleep_quality", "Sleep Quality", "Gemiddelde sleep quality", MVV_RED_DEEP, _format_decimal, ":.1f", (0, 10)),
                ("stress", "Stress", "Gemiddelde stress", MVV_RED_BRIGHT, _format_decimal, ":.1f", (0, 10)),
                ("mood", "Mood", "Gemiddelde mood", MVV_RED_DEEP, _format_decimal, ":.1f", (0, 10)),
                ("avg_rpe", "Weighted RPE", "Gewogen gemiddelde RPE", MVV_RED_BRIGHT, _format_decimal, ":.1f", (0, 10)),
                ("rpe_load", "RPE Load", "Totale duration x RPE", MVV_RED_DEEP, _format_int, ":,.0f", None),
            ]
            prefix = "Weekly" if scope_mode == "Year" else "Daily"
            for idx in range(0, len(monitoring_specs), 2):
                cols = st.columns(2, gap="large")
                for col_container, spec in zip(cols, monitoring_specs[idx : idx + 2]):
                    column, label, subtitle, color, formatter, hover_format, y_range = spec
                    with col_container:
                        render_plot_panel(
                            f"{prefix} {label} +/- SD",
                            build_bar_chart(
                                monitoring_group_df,
                                "label",
                                column,
                                f"{prefix} {label}",
                                color,
                                formatter,
                                hover_format=hover_format,
                                y_range=y_range,
                                error_col=f"{column}_std",
                            ),
                            subtitle,
                        )

            render_html_panel(
                "Monitoring Timeline",
                build_table_html(
                    monitoring_group_df,
                    [
                        ("label", "Periode", None),
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
                "Alle wellness-parameters, readiness en RPE door de tijd",
            )

    with tab_sessions:
        session_filter_count_col, session_filter_period_col = st.columns([0.34, 0.66], gap="large")
        with session_filter_count_col:
            st.markdown('<div class="player-report-filter-label">Aantal recente sessies</div>', unsafe_allow_html=True)
            st.number_input(
                "Aantal recente sessies",
                min_value=1,
                max_value=max_recent_session_count,
                value=recent_session_count,
                step=1,
                label_visibility="collapsed",
                key="player_report_recent_session_count",
            )
        with session_filter_period_col:
            st.markdown('<div class="player-report-filter-label">Periode recente sessies</div>', unsafe_allow_html=True)
            period_options = list(RECENT_SESSION_PERIOD_LABELS.keys())
            st.selectbox(
                "Periode recente sessies",
                options=period_options,
                index=period_options.index(recent_session_period_key),
                format_func=lambda value: RECENT_SESSION_PERIOD_LABELS.get(value, value),
                label_visibility="collapsed",
                key="player_report_recent_session_period",
            )

        sessions_left, sessions_right = st.columns(2, gap="large")
        with sessions_left:
            render_html_panel(
                "Recent Sessions",
                build_table_html(
                    recent_sessions_preview,
                    [
                        ("datum_label", "Datum", None),
                        ("type", "Type", None),
                        ("event", "Event", None),
                        ("total_distance", "Distance", _format_distance),
                        ("hsr_hsd", "HSR/HSD", _format_distance),
                        ("number_of_sprints", "Sprints", _format_int),
                        ("duration", "Duur", _format_minutes),
                        ("max_speed", "Top Speed", _format_speed),
                    ],
                ),
                recent_sessions_subtitle,
            )
        with sessions_right:
            notes_html = (
                '<ul class="player-report-note-list">'
                + "".join(f"<li>{escape(note)}</li>" for note in notes)
                + "</ul>"
                + '<div class="player-report-note-foot">PDF-export gebruikt dezelfde scope, spelerselectie en kernsamenvatting als de dashboardweergave.</div>'
            )
            render_html_panel(
                "Analyst Notes",
                notes_html,
                "Compacte samenvatting voor staff of individuele follow-up",
            )

    render_sidebar_footer(profile)


if __name__ == "__main__":
    main()
