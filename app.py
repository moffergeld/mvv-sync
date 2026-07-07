import base64
import html
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import extra_streamlit_components as stx  # noqa: F401
import pandas as pd
import streamlit as st
import roles as roles_mod

from acwr_settings import compute_chronic_series, get_acwr_mode_meta
from readiness_utils import build_wellness_snapshot_lookup, enrich_wellness_scores
from roles import (
    clear_tokens_in_cookie,
    cookie_mgr,
    get_profile,
    get_sb,
    render_sidebar_footer,
    render_sidebar_navigation,
    set_tokens_in_cookie,
    try_restore_or_refresh_session,
)
from utils.streamlit_ui import apply_streamlit_chrome


st.set_page_config(page_title="MVV Dashboard", layout="wide")
apply_streamlit_chrome()

DIAG_MODE = st.query_params.get("diag") == "1"
SAFE_MODE = st.query_params.get("safe") == "1"

MAINTENANCE_MODE = False
MAINTENANCE_TITLE = "Onderhoud"
MAINTENANCE_TEXT = "Er wordt onderhoud uitgevoerd. Je kunt mogelijk tijdelijk uitgelogd worden."

ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "Assets" / "Afbeeldingen"
TEAM_LOGO = ASSETS_DIR / "Team_Logos" / "MVV Maastricht.png"
HOME_BG = ASSETS_DIR / "Backgrounds" / "team_page_hero.png"

ACWR_HOME_METRICS = [("total_distance", "ACWR TD")]


def _fallback_consume_login_notice() -> Optional[str]:
    notice = st.session_state.pop("_login_notice", None)
    if notice is None:
        return None
    return str(notice)


def _fallback_clear_auth_state(clear_cookies: bool = False) -> None:
    for key in (
        "access_token",
        "sb_session",
        "_profile_cache",
        "role",
        "player_id",
        "user_id",
        "_sb_client",
        "auth_err",
    ):
        st.session_state.pop(key, None)
    if clear_cookies:
        clear_tokens_in_cookie()


def _fallback_ensure_valid_session(sb=None) -> bool:
    token = st.session_state.get("access_token")
    if not token:
        sess = st.session_state.get("sb_session")
        token = getattr(sess, "access_token", None) if sess is not None else None
    if token:
        st.session_state["access_token"] = str(token)
        return True

    restored = try_restore_or_refresh_session(sb)
    if not restored:
        return False

    token = st.session_state.get("access_token")
    if not token:
        sess = st.session_state.get("sb_session")
        token = getattr(sess, "access_token", None) if sess is not None else None
    if token:
        st.session_state["access_token"] = str(token)
        return True
    return False


consume_login_notice = getattr(roles_mod, "consume_login_notice", _fallback_consume_login_notice)
clear_auth_state = getattr(roles_mod, "clear_auth_state", _fallback_clear_auth_state)
ensure_valid_session = getattr(roles_mod, "ensure_valid_session", _fallback_ensure_valid_session)


def build_image_data_uri(path_like: str | Path) -> str:
    path = Path(path_like)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        return ""

    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def maintenance_banner() -> None:
    if not MAINTENANCE_MODE:
        return
    st.markdown(
        f"""
        <div style="
            padding:12px 14px;
            border-radius:8px;
            border:1px solid rgba(234,51,81,.42);
            background:rgba(200,16,46,.16);
            font-weight:800;
            color:#ffffff;">
            {MAINTENANCE_TITLE}
            <div style="font-weight:600;opacity:.9;margin-top:6px">{MAINTENANCE_TEXT}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home_css() -> None:
    page_bg_uri = "" if SAFE_MODE else build_image_data_uri(HOME_BG)
    app_background = (
        f"linear-gradient(180deg, rgba(6, 10, 20, 0.82) 0%, rgba(6, 10, 20, 0.80) 100%), "
        f"radial-gradient(circle at top left, rgba(200, 16, 46, 0.16), rgba(200, 16, 46, 0.02) 24%, transparent 46%), "
        f"radial-gradient(circle at top right, rgba(234, 51, 81, 0.10), rgba(234, 51, 81, 0.02) 18%, transparent 42%), "
        f"url('{page_bg_uri}')"
        if page_bg_uri
        else "radial-gradient(circle at top left, rgba(200, 16, 46, 0.28), rgba(200, 16, 46, 0.03) 26%, transparent 48%), radial-gradient(circle at top right, rgba(234, 51, 81, 0.18), rgba(234, 51, 81, 0.03) 18%, transparent 44%), linear-gradient(180deg, #070c18 0%, #0a1020 100%)"
    )
    st.markdown(
        """
        <style>
          :root {
            --mvv-red: #c8102e;
            --mvv-red-bright: #ea3351;
            --mvv-navy: #0b1020;
            --mvv-panel: #12192a;
            --mvv-text: #f8fafc;
            --mvv-muted: rgba(226, 232, 240, 0.74);
          }

          .stApp {
            background: __APP_BACKGROUND__;
            background-size: cover;
            background-position: center top;
            background-attachment: fixed;
            color: var(--mvv-text);
          }

          [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(16, 23, 38, 0.98), rgba(9, 13, 23, 0.98));
            border-right: 1px solid rgba(255,255,255,0.06);
          }

          [data-testid="stSidebar"] .stMarkdown,
          [data-testid="stSidebar"] label,
          [data-testid="stSidebar"] div {
            color: var(--mvv-text);
          }

          .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1380px;
          }

          .home-brand-shell {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            margin-bottom: 1.2rem;
          }

          .home-brand-logo {
            width: 78px;
            height: 78px;
            object-fit: contain;
            flex-shrink: 0;
            filter: drop-shadow(0 8px 22px rgba(0,0,0,0.28));
          }

          .home-brand-copy {
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 0.12rem;
            text-align: left;
          }

          .home-brand-kicker {
            color: rgba(255,255,255,0.76);
            font-size: 0.74rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            margin-bottom: 0;
          }

          .home-brand-title {
            margin: 0;
            font-size: 2.3rem;
            line-height: 1;
            font-weight: 800;
            color: #ffffff;
          }

          .home-summary-wrap {
            margin-bottom: 1.55rem;
          }

          .home-summary-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
          }

          .home-summary-card {
            min-height: 120px;
            padding: 1rem 1.05rem 0.95rem 1.05rem;
            border-radius: 8px;
            border: 1px solid rgba(234, 51, 81, 0.14);
            background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
          }

          .home-summary-label {
            color: rgba(255,255,255,0.68);
            font-size: 0.8rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
          }

          .home-summary-value {
            margin-top: 0.55rem;
            font-size: 1.95rem;
            line-height: 1.1;
            font-weight: 800;
            color: #ffffff;
            word-break: break-word;
          }

          .home-summary-foot {
            margin-top: 0.65rem;
            color: rgba(255,255,255,0.8);
            font-size: 0.86rem;
            line-height: 1.4;
          }

          .home-section-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 1rem;
            margin-bottom: 0.95rem;
          }

          .home-section-kicker {
            color: rgba(255,255,255,0.62);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
          }

          .home-section-title {
            margin-top: 0.25rem;
            color: #ffffff;
            font-size: 1.1rem;
            font-weight: 700;
          }

          .home-section-note {
            color: rgba(255,255,255,0.8);
            font-size: 0.88rem;
            font-weight: 700;
            text-align: right;
          }

          .home-kpi-board {
            border-radius: 8px;
            border: 1px solid rgba(234, 51, 81, 0.14);
            background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
            overflow: hidden;
          }

          .home-kpi-head,
          .home-kpi-row {
            display: grid;
            grid-template-columns: minmax(220px, 2fr) 0.95fr 0.8fr 0.8fr 0.8fr 1fr 0.9fr;
            gap: 0.85rem;
            align-items: center;
          }

          .home-kpi-head {
            padding: 0.9rem 1.05rem 0.8rem 1.05rem;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
            color: rgba(255,255,255,0.64);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
          }

          .home-kpi-row {
            padding: 0.95rem 1.05rem;
            border-bottom: 1px solid rgba(255,255,255,0.06);
          }

          .home-kpi-row:last-child {
            border-bottom: none;
          }

          .home-kpi-row[data-readiness="Alert"] {
            background: linear-gradient(90deg, rgba(185, 28, 28, 0.14), rgba(185, 28, 28, 0.02));
          }

          .home-kpi-row[data-readiness="Watch"] {
            background: linear-gradient(90deg, rgba(217, 119, 6, 0.10), rgba(217, 119, 6, 0.02));
          }

          .home-player-cell {
            display: flex;
            flex-direction: column;
            gap: 0.22rem;
            min-width: 0;
          }

          .home-player-name {
            color: #ffffff;
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.15;
          }

          .home-player-meta {
            color: rgba(255,255,255,0.68);
            font-size: 0.8rem;
            line-height: 1.4;
          }

          .home-kpi-cell {
            color: #ffffff;
            font-size: 0.95rem;
            font-weight: 700;
            line-height: 1.15;
          }

          .home-kpi-cell-muted {
            color: rgba(255,255,255,0.7);
          }

          .home-ready-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: fit-content;
            min-width: 78px;
            padding: 0.38rem 0.62rem;
            border-radius: 999px;
            color: #ffffff;
            font-size: 0.78rem;
            font-weight: 800;
            border: 1px solid rgba(255,255,255,0.14);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.12);
          }

          .home-kpi-mobile-label {
            display: none;
          }

          .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(234, 51, 81, 0.24);
            background: linear-gradient(135deg, rgba(234, 51, 81, 0.18), rgba(200, 16, 46, 0.28));
            color: #ffffff;
            font-weight: 800;
            min-height: 2.8rem;
            box-shadow: 0 10px 22px rgba(0,0,0,0.14);
          }

          .stButton > button:hover {
            border-color: rgba(234, 51, 81, 0.38);
            background: linear-gradient(135deg, rgba(234, 51, 81, 0.24), rgba(200, 16, 46, 0.36));
          }

          .stButton > button:disabled {
            background: rgba(255,255,255,0.04);
            border-color: rgba(255,255,255,0.08);
            color: rgba(255,255,255,0.48);
          }

          [data-testid="stTextInputRootElement"] {
            background: rgba(11, 16, 29, 0.88);
            border-radius: 8px;
          }

          @media (max-width: 1100px) {
            .home-brand-shell {
              align-items: center;
            }

            .home-summary-grid {
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }
          }

          @media (max-width: 768px) {
            .home-brand-shell {
              flex-direction: column;
              align-items: center;
              gap: 0.8rem;
            }

            .home-brand-copy {
              text-align: center;
            }

            .home-brand-title {
              font-size: 1.9rem;
            }

            .home-summary-grid {
              grid-template-columns: 1fr;
            }

            .home-section-head {
              flex-direction: column;
              align-items: flex-start;
            }

            .home-section-note {
              text-align: left;
            }

            .home-kpi-head {
              display: none;
            }

            .home-kpi-row {
              grid-template-columns: repeat(2, minmax(0, 1fr));
              gap: 0.7rem 1rem;
            }

            .home-player-cell {
              grid-column: 1 / -1;
            }

            .home-kpi-cell {
              display: flex;
              flex-direction: column;
              gap: 0.18rem;
            }

            .home-kpi-mobile-label {
              display: block;
              color: rgba(255,255,255,0.56);
              font-size: 0.7rem;
              font-weight: 800;
              letter-spacing: 0.12em;
              text-transform: uppercase;
            }
          }
        </style>
        """.replace("__APP_BACKGROUND__", app_background),
        unsafe_allow_html=True,
    )


def login_ui() -> None:
    maintenance_banner()
    st.markdown("## Inloggen")

    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pw")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if not submitted:
        return

    try:
        res = get_sb().auth.sign_in_with_password({"email": email, "password": password})
        sess = getattr(res, "session", None)
        token = getattr(sess, "access_token", None)
        refresh_token = getattr(sess, "refresh_token", None)

        if not token or not refresh_token:
            st.error("Login mislukt: geen geldige sessie ontvangen.")
            return

        st.session_state["access_token"] = token
        st.session_state["user_email"] = email
        st.session_state["sb_session"] = sess

        set_tokens_in_cookie(token, refresh_token, email)
        st.session_state.pop("_profile_cache", None)
        st.session_state.pop("role", None)
        st.session_state.pop("player_id", None)
        st.rerun()

    except Exception as exc:
        st.error(f"Sign in mislukt: {exc}")


def role_label_for_home(role: str) -> str:
    return "Staff" if (role or "").lower() != "player" else "Speler"


def build_status(score: Optional[float]) -> tuple[str, str]:
    if score is None or pd.isna(score):
        return "No data", "#6b7280"
    if score <= 4.5:
        return "Ready", "#14803c"
    if score <= 7.5:
        return "Watch", "#d97706"
    return "Alert", "#b91c1c"


def format_metric_value(value: Optional[float], suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "--"
    value = float(value)
    if not math.isfinite(value):
        return "--"
    if value.is_integer():
        return f"{int(value)}{suffix}"
    return f"{value:.1f}{suffix}"


def format_acwr_value(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "--"
    value = float(value)
    if not math.isfinite(value):
        return "--"
    return f"{value:.2f}"


def current_week_context() -> tuple[int, str]:
    iso = date.today().isocalendar()
    week_key = int(iso.year) * 100 + int(iso.week)
    week_label = f"{int(iso.year):04d}-W{int(iso.week):02d}"
    return week_key, week_label


@st.cache_data(show_spinner=False, ttl=300)
def fetch_active_players_cached(_sb, cache_scope: str = "default") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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

    out: List[Dict[str, Any]] = []
    for row in rows:
        player_id = row.get("player_id")
        full_name = str(row.get("full_name") or "").strip()
        position_value = row.get("Position")
        if position_value is None:
            position_value = row.get("position")
        if player_id and full_name:
            out.append(
                {
                    "player_id": str(player_id),
                    "full_name": full_name,
                    "position": str(position_value or "").strip(),
                }
            )
    return out


@st.cache_data(show_spinner=False, ttl=120)
def fetch_wellness_snapshot(_sb, access_scope: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    try:
        rows = (
            _sb.table("asrm_entries")
            .select("player_id,entry_date,muscle_soreness,fatigue,sleep_quality,stress,mood")
            .gte("entry_date", start_iso)
            .lte("entry_date", end_iso)
            .execute()
            .data
            or []
        )
    except Exception:
        rows = []

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce").dt.date
    return enrich_wellness_scores(df)


@st.cache_data(show_spinner=False, ttl=120)
def fetch_rpe_snapshot(_sb, access_scope: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    try:
        headers = (
            _sb.table("rpe_entries")
            .select("id,player_id,entry_date")
            .gte("entry_date", start_iso)
            .lte("entry_date", end_iso)
            .execute()
            .data
            or []
        )
    except Exception:
        headers = []

    headers_df = pd.DataFrame(headers)
    if headers_df.empty:
        return headers_df

    headers_df["entry_date"] = pd.to_datetime(headers_df["entry_date"], errors="coerce").dt.date
    entry_ids = [str(item) for item in headers_df["id"].dropna().tolist()]
    session_rows: List[Dict[str, Any]] = []

    for idx in range(0, len(entry_ids), 100):
        chunk = entry_ids[idx: idx + 100]
        try:
            rows = (
                _sb.table("rpe_sessions")
                .select("rpe_entry_id,rpe,duration_min")
                .in_("rpe_entry_id", chunk)
                .execute()
                .data
                or []
            )
        except Exception:
            rows = []
        session_rows.extend(rows)

    sessions_df = pd.DataFrame(session_rows)
    if sessions_df.empty:
        return pd.DataFrame(columns=["player_id", "entry_date", "rpe_avg"])

    sessions_df["rpe"] = pd.to_numeric(sessions_df["rpe"], errors="coerce")
    merged = headers_df.merge(sessions_df, left_on="id", right_on="rpe_entry_id", how="left")
    daily = (
        merged.groupby(["player_id", "entry_date"], as_index=False)
        .agg(rpe_avg=("rpe", "mean"))
        .sort_values(["player_id", "entry_date"])
    )
    return daily


@st.cache_data(show_spinner=False, ttl=120)
def fetch_gps_snapshot(_sb, access_scope: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    try:
        rows = (
            _sb.table("v_gps_summary")
            .select("player_id,datum,total_distance")
            .gte("datum", start_iso)
            .lte("datum", end_iso)
            .execute()
            .data
            or []
        )
    except Exception:
        rows = []

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["datum"] = pd.to_datetime(df["datum"], errors="coerce").dt.date
    df["total_distance"] = pd.to_numeric(df["total_distance"], errors="coerce").fillna(0.0)
    daily = (
        df.groupby(["player_id", "datum"], as_index=False)
        .agg(total_distance=("total_distance", "sum"))
        .sort_values(["player_id", "datum"])
    )
    return daily


@st.cache_data(show_spinner=False, ttl=120)
def fetch_gps_weekly_acwr(_sb, access_scope: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    try:
        rows = (
            _sb.table("v_gps_summary")
            .select("player_id,datum,total_distance,running,sprint,high_sprint")
            .gte("datum", start_iso)
            .lte("datum", end_iso)
            .execute()
            .data
            or []
        )
    except Exception:
        rows = []

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df = df.dropna(subset=["player_id", "datum"]).copy()
    if df.empty:
        return df

    metric_cols = [metric for metric, _ in ACWR_HOME_METRICS]
    for metric in metric_cols:
        if metric not in df.columns:
            df[metric] = 0.0
        df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)

    iso = df["datum"].dt.isocalendar()
    df["iso_year"] = iso["year"].astype("Int64")
    df["iso_week"] = iso["week"].astype("Int64")
    df["week_key"] = (df["iso_year"] * 100 + df["iso_week"]).astype("Int64")
    df["week_label"] = df.apply(
        lambda row: f"{int(row['iso_year']):04d}-W{int(row['iso_week']):02d}"
        if pd.notna(row["iso_year"]) and pd.notna(row["iso_week"])
        else None,
        axis=1,
    )

    weekly = (
        df.groupby(["player_id", "week_key", "week_label"], as_index=False)[metric_cols]
        .sum()
        .sort_values(["player_id", "week_key"])
    )
    return weekly


def build_snapshot_lookup(df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Dict[str, Any]]:
    if df.empty:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    today_value = date.today()
    for player_id, grp in df.groupby("player_id"):
        grp = grp.sort_values(date_col)
        latest = grp.tail(1).iloc[0]
        today_mask = grp[date_col] == today_value
        row = grp[today_mask].tail(1).iloc[0] if today_mask.any() else latest
        out[str(player_id)] = {
            "value": None if pd.isna(row[value_col]) else float(row[value_col]),
            "date": row[date_col],
            "is_today": bool(row[date_col] == today_value),
        }
    return out


def build_current_week_acwr_lookup(weekly_df: pd.DataFrame) -> tuple[str, Dict[str, Dict[str, Any]]]:
    current_week_key, current_week_label = current_week_context()
    if weekly_df.empty:
        return current_week_label, {}

    acwr_meta = get_acwr_mode_meta()
    df = weekly_df.copy()
    df["week_key"] = pd.to_numeric(df["week_key"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["player_id", "week_key"]).sort_values(["player_id", "week_key"]).copy()
    if df.empty:
        return current_week_label, {}

    metric_cols = [metric for metric, _ in ACWR_HOME_METRICS]
    for metric in metric_cols:
        chronic = df.groupby("player_id")[metric].transform(lambda series: compute_chronic_series(series, acwr_meta["mode"]))
        df[f"{metric}_acwr"] = df[metric].div(chronic.where(chronic != 0))

    current_df = df[df["week_key"] == current_week_key].copy()
    lookup: Dict[str, Dict[str, Any]] = {}
    for _, row in current_df.iterrows():
        player_payload: Dict[str, Any] = {"week_label": current_week_label}
        for metric in metric_cols:
            value = row.get(f"{metric}_acwr")
            if value is None or pd.isna(value):
                player_payload[f"{metric}_acwr"] = None
                continue
            value = float(value)
            player_payload[f"{metric}_acwr"] = value if math.isfinite(value) else None
        lookup[str(row["player_id"])] = player_payload

    return current_week_label, lookup


def assemble_home_rows(sb, access_scope: str) -> pd.DataFrame:
    players = fetch_active_players_cached(sb, access_scope)
    players_df = pd.DataFrame(players)
    if players_df.empty:
        return players_df

    today_value = date.today()
    start_wellness = today_value - timedelta(days=13)
    start_rpe = today_value - timedelta(days=6)
    start_gps = today_value - timedelta(days=13)
    start_acwr = today_value - timedelta(days=84)

    wellness_df = fetch_wellness_snapshot(sb, access_scope, start_wellness.isoformat(), today_value.isoformat())
    rpe_df = fetch_rpe_snapshot(sb, access_scope, start_rpe.isoformat(), today_value.isoformat())
    gps_df = fetch_gps_snapshot(sb, access_scope, start_gps.isoformat(), today_value.isoformat())
    acwr_weekly_df = fetch_gps_weekly_acwr(sb, access_scope, start_acwr.isoformat(), today_value.isoformat())

    wellness_lookup = build_wellness_snapshot_lookup(wellness_df, "entry_date")
    rpe_lookup = build_snapshot_lookup(rpe_df, "entry_date", "rpe_avg")
    gps_lookup = build_snapshot_lookup(gps_df, "datum", "total_distance")
    current_week_label, acwr_lookup = build_current_week_acwr_lookup(acwr_weekly_df)

    rows: List[Dict[str, Any]] = []
    for player in players:
        player_id = player["player_id"]
        wellness = wellness_lookup.get(player_id, {})
        rpe = rpe_lookup.get(player_id, {})
        gps = gps_lookup.get(player_id, {})
        acwr = acwr_lookup.get(player_id, {})
        readiness_score = wellness.get("readiness_score")
        readiness_label, readiness_color = build_status(readiness_score)

        rows.append(
            {
                "player_id": player_id,
                "full_name": player["full_name"],
                "position": player.get("position"),
                "wellness_value": wellness.get("overall"),
                "wellness_physical": wellness.get("physical"),
                "wellness_mental": wellness.get("mental"),
                "wellness_date": wellness.get("date"),
                "wellness_today": bool(wellness.get("is_today", False)),
                "rpe_value": rpe.get("value"),
                "rpe_date": rpe.get("date"),
                "rpe_today": bool(rpe.get("is_today", False)),
                "gps_value": gps.get("value"),
                "gps_date": gps.get("date"),
                "acwr_week_label": acwr.get("week_label", current_week_label),
                "total_distance_acwr": acwr.get("total_distance_acwr"),
                "readiness_score": readiness_score,
                "readiness_label": readiness_label,
                "readiness_color": readiness_color,
                "readiness_rank": {"Alert": 0, "Watch": 1, "Ready": 2, "No data": 3}.get(readiness_label, 3),
            }
        )

    return pd.DataFrame(rows)


def filter_home_rows_for_profile(df: pd.DataFrame, role: str, profile: dict) -> pd.DataFrame:
    if df.empty or (role or "").lower() != "player":
        return df

    player_id = str(profile.get("player_id") or "").strip()
    if not player_id:
        return df.iloc[0:0].copy()

    filtered = df[df["player_id"].astype(str) == player_id].copy()
    return filtered.reset_index(drop=True)


def render_home_brand() -> None:
    logo_uri = build_image_data_uri(TEAM_LOGO) if TEAM_LOGO.exists() else ""
    logo_markup = f'<img src="{logo_uri}" alt="MVV Maastricht" class="home-brand-logo" />' if logo_uri else ""

    st.markdown(
        f"""
        <div class="home-brand-shell">
          {logo_markup}
          <div class="home-brand-copy">
            <h1 class="home-brand-title">Selectie KPI's</h1>
            <div class="home-brand-kicker">MVV Maastricht | Dashboard | Readiness overview</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home_summary(df: pd.DataFrame, role: str) -> None:
    role_label = role_label_for_home(role)
    ready_count = int((df["readiness_label"] == "Ready").sum()) if not df.empty else 0
    watch_count = int((df["readiness_label"] == "Watch").sum()) if not df.empty else 0
    alert_count = int((df["readiness_label"] == "Alert").sum()) if not df.empty else 0
    wellness_today_count = int(df["wellness_today"].sum()) if not df.empty else 0
    rpe_today_count = int(df["rpe_today"].sum()) if not df.empty else 0

    summary_cards = [
        ("Rol", role_label, "Actieve toegangslaag voor deze sessie"),
        ("Spelers", str(len(df)), "Compact overzicht op basis van de actuele selectie"),
        ("Wellness vandaag", str(wellness_today_count), "Spelers met wellness-invoer vandaag"),
        ("RPE vandaag", str(rpe_today_count), f"Ready: {ready_count} | Watch: {watch_count} | Alert: {alert_count}"),
    ]
    summary_markup = "".join(
        f"""<div class="home-summary-card">
<div class="home-summary-label">{label}</div>
<div class="home-summary-value">{value}</div>
<div class="home-summary-foot">{foot}</div>
</div>"""
        for label, value, foot in summary_cards
    )

    st.markdown(
        f"""
        <div class="home-summary-wrap">
          <div class="home-summary-grid">
            {summary_markup}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_player_meta(row: dict) -> str:
    parts: list[str] = []
    position_value = str(row.get("position") or "").strip()
    if position_value:
        parts.append(position_value)
    if row.get("gps_date"):
        parts.append(f"GPS {row['gps_date'].strftime('%d-%m')}")
    parts.append(f"Wellness vandaag {'Ja' if row.get('wellness_today') else 'Nee'}")
    parts.append(f"RPE vandaag {'Ja' if row.get('rpe_today') else 'Nee'}")
    return " | ".join(parts)


def render_home_kpi_board(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Nog geen spelerdata beschikbaar voor deze startpagina.")
        return

    acwr_meta = get_acwr_mode_meta()
    current_week_label = (
        str(df["acwr_week_label"].dropna().iloc[0])
        if "acwr_week_label" in df.columns and df["acwr_week_label"].notna().any()
        else current_week_context()[1]
    )
    st.markdown(
        f"""
        <div class="home-section-head">
          <div>
            <div class="home-section-kicker">Selectie</div>
            <div class="home-section-title">Compact readiness-overzicht per speler</div>
          </div>
          <div class="home-section-note">Gesorteerd op aandacht | ACWR week {current_week_label} | {acwr_meta['short_label']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows_df = df.sort_values(["readiness_rank", "full_name"], ascending=[True, True]).reset_index(drop=True)
    row_markup = "".join(
        f"""<div class="home-kpi-row" data-readiness="{html.escape(str(row['readiness_label']))}">
<div class="home-player-cell">
<div class="home-player-name">{html.escape(str(row['full_name']))}</div>
<div class="home-player-meta">{html.escape(build_player_meta(row))}</div>
</div>
<div class="home-kpi-cell">
<span class="home-kpi-mobile-label">Readiness</span>
<span class="home-ready-badge" style="background:{html.escape(str(row['readiness_color']))};">
{html.escape(str(row['readiness_label']))}
</span>
</div>
<div class="home-kpi-cell">
<span class="home-kpi-mobile-label">Physical</span>
{html.escape(format_metric_value(row.get('wellness_physical')))}
</div>
<div class="home-kpi-cell">
<span class="home-kpi-mobile-label">Mental</span>
{html.escape(format_metric_value(row.get('wellness_mental')))}
</div>
<div class="home-kpi-cell">
<span class="home-kpi-mobile-label">RPE</span>
{html.escape(format_metric_value(row.get('rpe_value')))}
</div>
<div class="home-kpi-cell">
<span class="home-kpi-mobile-label">GPS</span>
{html.escape(format_metric_value(row.get('gps_value'), ' m'))}
</div>
<div class="home-kpi-cell home-kpi-cell-muted">
<span class="home-kpi-mobile-label">ACWR TD</span>
{html.escape(format_acwr_value(row.get('total_distance_acwr')))}
</div>
</div>"""
        for row in rows_df.to_dict(orient="records")
    )
    st.markdown(
        f"""
        <div class="home-kpi-board">
          <div class="home-kpi-head">
            <div>Speler</div>
            <div>Readiness</div>
            <div>Physical</div>
            <div>Mental</div>
            <div>RPE</div>
            <div>GPS</div>
            <div>ACWR TD</div>
          </div>
          {row_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


if DIAG_MODE:
    st.title("DIAG OK")
    st.write("Als je dit ziet, werkt Streamlit op dit toestel/netwerk.")
    st.write("Zet diag uit door ?diag=0 of verwijder de query parameter.")
    st.stop()


render_home_css()

sb = get_sb()
if sb is None:
    st.error("Supabase client niet beschikbaar (secrets ontbreken of create_client faalt).")
    st.stop()


try:
    cm = cookie_mgr()
    _ = cm.get("sb_refresh")
except Exception:
    pass

login_notice = consume_login_notice()
if login_notice:
    st.warning(login_notice)

if not ensure_valid_session(sb):
    login_ui()
    st.stop()


profile = get_profile(sb)
if not profile:
    clear_auth_state(clear_cookies=True)
    st.warning("Geen profiel gevonden of geen rechten. Log opnieuw in.")
    login_ui()
    st.stop()


role = str(st.session_state.get("role") or profile.get("role") or "").lower()
render_sidebar_navigation(profile)

if DIAG_MODE:
    render_sidebar_footer(profile, show_debug=True)
else:
    render_sidebar_footer(profile)

maintenance_banner()

if SAFE_MODE:
    st.warning("Safe mode actief (minimale UI). Zet uit door ?safe=0 te gebruiken.")

access_scope = f"{role}:{profile.get('user_id', 'anon')}"
home_df = assemble_home_rows(sb, access_scope)
home_df = filter_home_rows_for_profile(home_df, role, profile)
if home_df.empty:
    st.warning("Geen readiness-KPI's beschikbaar voor deze gebruiker.")
    st.stop()

render_home_brand()
render_home_summary(home_df, role)
render_home_kpi_board(home_df)
