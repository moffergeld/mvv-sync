from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
ASSETS_DIR = ROOT_DIR / "Assets" / "Afbeeldingen"
PLAYER_IMG_DIR = ASSETS_DIR / "Spelers"
TEAM_LOGO = ASSETS_DIR / "Team_Logos" / "MVV Maastricht.png"

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from roles import get_profile, get_sb, require_auth  # noqa: E402
from Subscripts.player_tab_data import (  # noqa: E402
    ASRM_COLS,
    fetch_asrm_14d,
    fetch_gps_14d,
    fetch_rpe_for_date,
    fetch_rpe_over_time_7d,
    gps_daily_aggregate,
    load_asrm,
    render_data_tab,
)
from Subscripts.player_tab_forms import render_forms_tab  # noqa: E402


st.set_page_config(page_title="Player Page Beta", layout="wide")


@st.cache_data(show_spinner=False, ttl=300)
def fetch_active_players_cached(_sb, cache_scope: str = "default"):
    try:
        rows = (
            _sb.table("players")
            .select("player_id,full_name,is_active")
            .eq("is_active", True)
            .order("full_name")
            .execute()
            .data
            or []
        )
        return rows
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=300)
def fetch_player_name_cached(_sb, player_id: str) -> str:
    try:
        row = (
            _sb.table("players")
            .select("full_name")
            .eq("player_id", player_id)
            .maybe_single()
            .execute()
            .data
        )
        return (row or {}).get("full_name") or "Player"
    except Exception:
        return "Player"


@st.cache_data(show_spinner=False, ttl=300)
def build_player_image_index() -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not PLAYER_IMG_DIR.exists():
        return out

    for path in PLAYER_IMG_DIR.glob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        out[path.stem.strip().lower()] = str(path)
    return out


def resolve_player_image(player_name: str) -> Optional[str]:
    index = build_player_image_index()
    exact = index.get(player_name.strip().lower())
    if exact:
        return exact

    normalized_name = "".join(ch for ch in player_name.lower() if ch.isalnum())
    for stem, path in index.items():
        normalized_stem = "".join(ch for ch in stem.lower() if ch.isalnum())
        if normalized_stem == normalized_name:
            return path
    return None


def pick_active_player_dropdown(sb, cache_scope: str, key: str = "pp_beta_player_select"):
    rows = fetch_active_players_cached(sb, cache_scope)
    if not rows:
        return None, None

    pairs = []
    for row in rows:
        pid = row.get("player_id")
        name = (row.get("full_name") or "").strip()
        if pid and name:
            pairs.append((name, str(pid)))

    if not pairs:
        return None, None

    options = [name for name, _ in pairs]
    selected_name = st.selectbox("Speler", options=options, key=key)
    name_to_id = dict(pairs)
    return name_to_id.get(selected_name), selected_name


def build_status(score: Optional[float]) -> tuple[str, str]:
    if score is None:
        return "No data", "#6b7280"
    if score <= 4.5:
        return "Ready", "#14803c"
    if score <= 7.5:
        return "Watch", "#d97706"
    return "Alert", "#b91c1c"


def format_metric_value(value: Optional[float], suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "--"
    if float(value).is_integer():
        return f"{int(value)}{suffix}"
    return f"{value:.1f}{suffix}"


def latest_wellness_score(sb, player_id: str) -> tuple[Optional[float], Optional[date], bool]:
    today_row = load_asrm(sb, player_id, date.today())
    if today_row:
        vals = [float(today_row.get(col, 0) or 0) for _, col in ASRM_COLS]
        return sum(vals) / len(vals), date.today(), True

    df = fetch_asrm_14d(sb, player_id, days=14)
    if df.empty:
        return None, None, False

    latest_date = df["entry_date"].max()
    latest_row = df[df["entry_date"] == latest_date].tail(1)
    if latest_row.empty:
        return None, None, False

    cols = [col for _, col in ASRM_COLS]
    score = latest_row[cols].astype(float).mean(axis=1).iloc[0]
    return float(score), latest_date, False


def latest_rpe_score(sb, player_id: str) -> tuple[Optional[float], Optional[date], bool]:
    today_sessions = fetch_rpe_for_date(sb, player_id, date.today())
    if not today_sessions.empty:
        return float(today_sessions["rpe"].mean()), date.today(), True

    df = fetch_rpe_over_time_7d(sb, player_id)
    if df.empty:
        return None, None, False

    latest_row = df.sort_values("entry_date").tail(1)
    return float(latest_row["rpe"].iloc[0]), latest_row["entry_date"].iloc[0], False


def latest_gps_summary(sb, player_id: str) -> tuple[Optional[float], Optional[date]]:
    df = fetch_gps_14d(sb, player_id, days=14)
    if df.empty:
        return None, None

    daily = gps_daily_aggregate(df, "total_distance")
    if daily.empty:
        return None, None

    latest_row = daily.sort_values("datum").tail(1)
    return float(latest_row["total_distance"].iloc[0]), latest_row["datum"].iloc[0]


def render_css() -> None:
    st.markdown(
        """
        <style>
          .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 1320px;
          }

          .mvv-beta-shell {
            background:
              radial-gradient(circle at top right, rgba(228, 8, 36, 0.18), transparent 30%),
              linear-gradient(180deg, rgba(24, 24, 27, 0.97), rgba(11, 11, 15, 0.97));
            color: #f9fafb;
            border-radius: 8px;
            padding: 1.25rem 1.4rem 1.35rem 1.4rem;
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1rem;
          }

          .mvv-beta-kicker {
            color: rgba(255,255,255,0.72);
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.4rem;
          }

          .mvv-beta-title {
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.02;
            margin: 0;
          }

          .mvv-beta-sub {
            margin-top: 0.55rem;
            max-width: 62ch;
            color: rgba(255,255,255,0.8);
            font-size: 1rem;
            line-height: 1.5;
          }

          .mvv-band {
            background: linear-gradient(90deg, rgba(228,8,36,0.14), rgba(255,255,255,0.02));
            border: 1px solid rgba(228,8,36,0.18);
            border-radius: 8px;
            padding: 0.95rem 1rem;
            margin: 0.85rem 0 1rem 0;
          }

          .mvv-pill {
            display: inline-block;
            padding: 0.36rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-right: 0.4rem;
            margin-bottom: 0.3rem;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.05);
          }

          .mvv-section {
            padding: 1rem 0 0.25rem 0;
          }

          .mvv-section h3 {
            margin: 0;
            font-size: 1.18rem;
            font-weight: 800;
            color: #111827;
          }

          .mvv-section p {
            margin: 0.35rem 0 0 0;
            color: #4b5563;
          }

          .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            margin-bottom: 0.75rem;
          }

          .stTabs [data-baseweb="tab"] {
            height: 48px;
            border-radius: 8px;
            padding: 0 1rem;
            background: #f3f4f6;
            color: #111827;
            font-weight: 700;
          }

          .stTabs [aria-selected="true"] {
            background: #e40824 !important;
            color: white !important;
          }

          [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid rgba(17,24,39,0.08);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            height: 100%;
          }

          [data-testid="stMetricLabel"] {
            font-weight: 700;
            color: #6b7280;
          }

          [data-testid="stMetricValue"] {
            font-size: 2rem;
            color: #111827;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(
    player_name: str,
    player_image: Optional[str],
    wellness_score: Optional[float],
    wellness_date: Optional[date],
    has_wellness_today: bool,
    rpe_score: Optional[float],
    rpe_date: Optional[date],
    has_rpe_today: bool,
    gps_distance: Optional[float],
    gps_date: Optional[date],
) -> None:
    status_label, status_color = build_status(wellness_score)

    hero_left, hero_right = st.columns([1.05, 1.95], gap="large")

    with hero_left:
        if player_image:
            st.image(player_image, use_container_width=True)
        elif TEAM_LOGO.exists():
            st.image(str(TEAM_LOGO), width=220)

    with hero_right:
        st.markdown(
            f"""
            <div class="mvv-beta-shell">
              <div class="mvv-beta-kicker">MVV Maastricht | Player Experience | Beta</div>
              <h1 class="mvv-beta-title">{player_name}</h1>
              <div class="mvv-beta-sub">
                Nieuwe branded playeromgeving met snellere focus op readiness, laatste belasting en check-in.
              </div>
              <div class="mvv-band">
                <span class="mvv-pill" style="background:{status_color};color:#fff;border-color:{status_color};">
                  Wellness status | {status_label}
                </span>
                <span class="mvv-pill">Vandaag wellness | {"Ingevuld" if has_wellness_today else "Open"}</span>
                <span class="mvv-pill">Vandaag RPE | {"Ingevuld" if has_rpe_today else "Open"}</span>
                <span class="mvv-pill">Laatste GPS | {gps_date.strftime("%d-%m-%Y") if gps_date else "--"}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                "Wellness avg",
                format_metric_value(wellness_score),
                delta=("Vandaag" if has_wellness_today else wellness_date.strftime("%d-%m") if wellness_date else None),
            )
        with m2:
            st.metric(
                "RPE avg",
                format_metric_value(rpe_score),
                delta=("Vandaag" if has_rpe_today else rpe_date.strftime("%d-%m") if rpe_date else None),
            )
        with m3:
            st.metric(
                "Latest GPS",
                format_metric_value(gps_distance, " m"),
                delta=gps_date.strftime("%d-%m") if gps_date else None,
            )


def render_overview(sb, player_id: str) -> None:
    st.markdown(
        """
        <div class="mvv-section">
          <h3>Overview</h3>
          <p>Snelle kijk op de laatste check-ins en trainingsbelasting van deze speler.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    today = date.today()
    today_asrm = load_asrm(sb, player_id, today)
    today_rpe = fetch_rpe_for_date(sb, player_id, today)
    wellness_df = fetch_asrm_14d(sb, player_id, days=14)
    rpe_df = fetch_rpe_over_time_7d(sb, player_id)

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        recent_rows = []
        if not wellness_df.empty:
            latest_five = wellness_df.sort_values("entry_date", ascending=False).head(5).copy()
            latest_five["wellness_avg"] = latest_five[[col for _, col in ASRM_COLS]].astype(float).mean(axis=1)
            for _, row in latest_five.iterrows():
                recent_rows.append(
                    {
                        "Date": row["entry_date"].strftime("%d-%m-%Y"),
                        "Wellness avg": round(float(row["wellness_avg"]), 1),
                    }
                )
        if recent_rows:
            st.dataframe(pd.DataFrame(recent_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Nog geen wellnesshistorie beschikbaar.")

    with right:
        st.markdown("#### Vandaag")
        if today_asrm:
            avg_today = sum(float(today_asrm.get(col, 0) or 0) for _, col in ASRM_COLS) / len(ASRM_COLS)
            label, color = build_status(avg_today)
            st.markdown(
                f"""
                <div class="mvv-band">
                  <span class="mvv-pill" style="background:{color};color:#fff;border-color:{color};">
                    Wellness | {label} | {avg_today:.1f}
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Wellness voor vandaag is nog niet ingevuld.")

        if not today_rpe.empty:
            st.success(f"RPE vandaag ingevuld | gemiddeld {today_rpe['rpe'].mean():.1f}")
        else:
            st.info("RPE voor vandaag is nog niet ingevuld.")

        if not rpe_df.empty:
            latest_rpe = rpe_df.sort_values("entry_date", ascending=False).head(3).copy()
            latest_rpe["entry_date"] = latest_rpe["entry_date"].apply(lambda x: x.strftime("%d-%m-%Y"))
            latest_rpe["rpe"] = latest_rpe["rpe"].round(1)
            st.dataframe(
                latest_rpe.rename(columns={"entry_date": "Date", "rpe": "RPE avg"}),
                use_container_width=True,
                hide_index=True,
            )


def main() -> None:
    render_css()

    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    profile = get_profile(sb) or {}
    role = str(profile.get("role") or "").lower()
    my_player_id = profile.get("player_id")
    cache_scope = f"{role}:{profile.get('user_id', 'anon')}"

    if role == "player":
        if not my_player_id:
            st.error("Je profiel is niet gekoppeld aan een speler (player_id ontbreekt).")
            st.stop()
        target_player_id = str(my_player_id)
        target_player_name = fetch_player_name_cached(sb, target_player_id)
    else:
        st.markdown(
            """
            <div class="mvv-section">
              <h3>Player Selection</h3>
              <p>Kies een speler om de beta-ervaring te bekijken.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        target_player_id, target_player_name = pick_active_player_dropdown(
            sb,
            cache_scope=cache_scope,
            key="pp_beta_player_select",
        )
        if not target_player_id:
            st.error("Geen speler beschikbaar.")
            st.stop()

    player_image = resolve_player_image(target_player_name)
    wellness_score, wellness_date, has_wellness_today = latest_wellness_score(sb, target_player_id)
    rpe_score, rpe_date, has_rpe_today = latest_rpe_score(sb, target_player_id)
    gps_distance, gps_date = latest_gps_summary(sb, target_player_id)

    render_hero(
        player_name=target_player_name,
        player_image=player_image,
        wellness_score=wellness_score,
        wellness_date=wellness_date,
        has_wellness_today=has_wellness_today,
        rpe_score=rpe_score,
        rpe_date=rpe_date,
        has_rpe_today=has_rpe_today,
        gps_distance=gps_distance,
        gps_date=gps_date,
    )

    tabs = st.tabs(["Overview", "Performance", "Check-in"])

    with tabs[0]:
        render_overview(sb, target_player_id)

    with tabs[1]:
        render_data_tab(sb, target_player_id)

    with tabs[2]:
        render_forms_tab(sb, target_player_id)


if __name__ == "__main__":
    main()
