from __future__ import annotations

import math
import sys
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

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


st.set_page_config(page_title="Team Page Beta", layout="wide")


GROUP_ORDER = [
    ("Doelmannen", "Goalkeepers"),
    ("Verdedigers", "Defenders"),
    ("Middenvelders", "Midfielders"),
    ("Aanvallers", "Attackers"),
    ("Selectie", "Squad"),
]


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    return "".join(ch for ch in text if ch.isalnum())


POSITION_GROUP_BY_NAME = {
    normalize_name("Sem Westerveld"): "Doelmannen",
    normalize_name("Tom Poitoux"): "Doelmannen",
    normalize_name("Sep van der Heijden"): "Doelmannen",
    normalize_name("Ruud Geerinck"): "Doelmannen",
    normalize_name("Nicola Rijnbout"): "Doelmannen",
    normalize_name("Simon Francis"): "Verdedigers",
    normalize_name("Finn Dicke"): "Verdedigers",
    normalize_name("Wout Coomans"): "Verdedigers",
    normalize_name("Ilias Breugelmans"): "Verdedigers",
    normalize_name("Adam Zaian"): "Verdedigers",
    normalize_name("Djairo Tehubijuluw"): "Verdedigers",
    normalize_name("Mitch van Kempen"): "Verdedigers",
    normalize_name("Lenn-Minh Tran"): "Verdedigers",
    normalize_name("Lars Schenk"): "Verdedigers",
    normalize_name("Kanou Sy"): "Verdedigers",
    normalize_name("Andrea Librici"): "Verdedigers",
    normalize_name("Andrea Librici"): "Verdedigers",
    normalize_name("Nabil El Basri"): "Middenvelders",
    normalize_name("Stan van Dessel"): "Middenvelders",
    normalize_name("Amine Amgar"): "Middenvelders",
    normalize_name("Marko Kleinen"): "Middenvelders",
    normalize_name("Adriano Mpudi"): "Middenvelders",
    normalize_name("Robert Klaasen"): "Middenvelders",
    normalize_name("Bryan Smeets"): "Middenvelders",
    normalize_name("Jashari"): "Middenvelders",
    normalize_name("Lirim Jashari"): "Middenvelders",
    normalize_name("Travis de Jong"): "Aanvallers",
    normalize_name("Ayman Kassimi"): "Aanvallers",
    normalize_name("Sven Braken"): "Aanvallers",
    normalize_name("Ilano Silva Timas"): "Aanvallers",
    normalize_name("Thijme Verheijen"): "Aanvallers",
    normalize_name("Mats Kuipers"): "Aanvallers",
    normalize_name("Jael Pawirodihardjo"): "Aanvallers",
    normalize_name("Jael"): "Aanvallers",
    normalize_name("Jael Pawirodihardjo"): "Aanvallers",
    normalize_name("Delano Asante"): "Aanvallers",
    normalize_name("Luca Foubert"): "Aanvallers",
    normalize_name("Camil Mmaee"): "Aanvallers",
}


@st.cache_data(show_spinner=False, ttl=300)
def fetch_active_players_cached(_sb, cache_scope: str = "default") -> List[Dict[str, str]]:
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
    except Exception:
        rows = []

    out: List[Dict[str, str]] = []
    for row in rows:
        player_id = row.get("player_id")
        full_name = str(row.get("full_name") or "").strip()
        if player_id and full_name:
            out.append({"player_id": str(player_id), "full_name": full_name})
    return out


@st.cache_data(show_spinner=False, ttl=300)
def build_player_image_index() -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not PLAYER_IMG_DIR.exists():
        return out

    for path in PLAYER_IMG_DIR.glob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        out[normalize_name(path.stem)] = str(path)
    return out


def resolve_player_image(player_name: str) -> Optional[str]:
    index = build_player_image_index()
    normalized = normalize_name(player_name)
    exact = index.get(normalized)
    if exact:
        return exact

    for stem, path in index.items():
        if normalized and (normalized in stem or stem in normalized):
            return path
    return None


def resolve_group(player_name: str) -> str:
    return POSITION_GROUP_BY_NAME.get(normalize_name(player_name), "Selectie")


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
    if float(value).is_integer():
        return f"{int(value)}{suffix}"
    return f"{value:.1f}{suffix}"


def initials_for_name(name: str) -> str:
    parts = [part for part in str(name).split() if part]
    if not parts:
        return "MV"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def render_css() -> None:
    st.markdown(
        """
        <style>
          .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 1380px;
          }

          .team-hero {
            background:
              radial-gradient(circle at top right, rgba(228, 8, 36, 0.2), transparent 28%),
              linear-gradient(180deg, rgba(24, 24, 27, 0.97), rgba(11, 11, 15, 0.97));
            color: #f9fafb;
            border-radius: 8px;
            padding: 1.25rem 1.35rem 1.4rem 1.35rem;
            border: 1px solid rgba(255,255,255,0.08);
          }

          .team-kicker {
            color: rgba(255,255,255,0.72);
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.35rem;
          }

          .team-title {
            margin: 0;
            font-size: 2.25rem;
            line-height: 1.02;
            font-weight: 800;
          }

          .team-sub {
            margin-top: 0.55rem;
            max-width: 68ch;
            color: rgba(255,255,255,0.82);
            line-height: 1.5;
          }

          .team-pill {
            display: inline-block;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            margin-right: 0.45rem;
            margin-top: 0.75rem;
            font-size: 0.82rem;
            font-weight: 700;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.05);
          }

          .team-section {
            padding-top: 1.1rem;
          }

          .team-section-head {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 1rem;
            margin-bottom: 0.85rem;
            padding-bottom: 0.45rem;
            border-bottom: 1px solid rgba(17, 24, 39, 0.08);
          }

          .team-section-title {
            margin: 0;
            font-size: 1.22rem;
            font-weight: 800;
            color: #111827;
          }

          .team-section-copy {
            margin: 0.15rem 0 0 0;
            color: #4b5563;
          }

          .team-count {
            font-size: 0.88rem;
            font-weight: 700;
            color: #6b7280;
            white-space: nowrap;
          }

          .team-card {
            border: 1px solid rgba(17,24,39,0.08);
            border-radius: 8px;
            padding: 0.85rem 0.9rem 0.95rem 0.9rem;
            background: #ffffff;
            min-height: 214px;
          }

          .team-card-title {
            margin: 0.7rem 0 0 0;
            font-size: 1rem;
            font-weight: 800;
            color: #111827;
          }

          .team-card-meta {
            margin-top: 0.15rem;
            color: #6b7280;
            font-size: 0.84rem;
          }

          .team-status {
            display: inline-block;
            margin-top: 0.65rem;
            padding: 0.34rem 0.6rem;
            border-radius: 999px;
            color: #ffffff;
            font-size: 0.8rem;
            font-weight: 800;
          }

          .team-grid-gap {
            height: 0.85rem;
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
            font-size: 1.95rem;
            color: #111827;
          }

          .team-initials {
            width: 100%;
            aspect-ratio: 4 / 5;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(180deg, #1f2937, #111827);
            color: #ffffff;
            font-size: 2rem;
            font-weight: 800;
            border: 1px solid rgba(255,255,255,0.08);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    metric_cols = ["muscle_soreness", "fatigue", "sleep_quality", "stress", "mood"]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["wellness_avg"] = df[metric_cols].mean(axis=1)
    return df


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


def assemble_team_rows(sb, access_scope: str) -> pd.DataFrame:
    players = fetch_active_players_cached(sb, access_scope)
    players_df = pd.DataFrame(players)
    if players_df.empty:
        return players_df

    today_value = date.today()
    start_wellness = today_value - timedelta(days=13)
    start_rpe = today_value - timedelta(days=6)
    start_gps = today_value - timedelta(days=13)

    wellness_df = fetch_wellness_snapshot(sb, access_scope, start_wellness.isoformat(), today_value.isoformat())
    rpe_df = fetch_rpe_snapshot(sb, access_scope, start_rpe.isoformat(), today_value.isoformat())
    gps_df = fetch_gps_snapshot(sb, access_scope, start_gps.isoformat(), today_value.isoformat())

    wellness_lookup = build_snapshot_lookup(wellness_df, "entry_date", "wellness_avg")
    rpe_lookup = build_snapshot_lookup(rpe_df, "entry_date", "rpe_avg")
    gps_lookup = build_snapshot_lookup(gps_df, "datum", "total_distance")

    rows: List[Dict[str, Any]] = []
    for player in players:
        player_id = player["player_id"]
        player_name = player["full_name"]
        wellness = wellness_lookup.get(player_id, {})
        rpe = rpe_lookup.get(player_id, {})
        gps = gps_lookup.get(player_id, {})
        readiness_label, readiness_color = build_status(wellness.get("value"))

        rows.append(
            {
                "player_id": player_id,
                "full_name": player_name,
                "group": resolve_group(player_name),
                "photo_path": resolve_player_image(player_name),
                "wellness_value": wellness.get("value"),
                "wellness_date": wellness.get("date"),
                "wellness_today": bool(wellness.get("is_today", False)),
                "rpe_value": rpe.get("value"),
                "rpe_date": rpe.get("date"),
                "rpe_today": bool(rpe.get("is_today", False)),
                "gps_value": gps.get("value"),
                "gps_date": gps.get("date"),
                "readiness_label": readiness_label,
                "readiness_color": readiness_color,
                "readiness_rank": {"Alert": 0, "Watch": 1, "Ready": 2, "No data": 3}.get(readiness_label, 3),
            }
        )

    return pd.DataFrame(rows)


def render_hero(df: pd.DataFrame) -> None:
    left, right = st.columns([0.28, 1.72], gap="large")
    with left:
        if TEAM_LOGO.exists():
            st.image(str(TEAM_LOGO), width=120)

    with right:
        st.markdown(
            """
            <div class="team-hero">
              <div class="team-kicker">MVV Maastricht | Team Readiness | Beta</div>
              <h1 class="team-title">Team Page</h1>
              <div class="team-sub">
                Overzicht van de selectie met per linie de foto, naam en actuele readiness op basis van de laatste wellnesscheck,
                aangevuld met RPE en laatste GPS-belasting.
              </div>
              <span class="team-pill">Gegroepeerd op doelmannen, verdedigers, middenvelders en aanvallers</span>
              <span class="team-pill">Readiness gebaseerd op laatste wellness-score in de app</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    ready_count = int((df["readiness_label"] == "Ready").sum()) if not df.empty else 0
    alert_count = int((df["readiness_label"] == "Alert").sum()) if not df.empty else 0
    wellness_today_count = int(df["wellness_today"].sum()) if not df.empty else 0
    rpe_today_count = int(df["rpe_today"].sum()) if not df.empty else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Spelers", len(df))
    with m2:
        st.metric("Wellness vandaag", wellness_today_count)
    with m3:
        st.metric("RPE vandaag", rpe_today_count)
    with m4:
        st.metric("Alerts", alert_count, delta=f"Ready {ready_count}")


def render_player_card(player: Dict[str, Any]) -> None:
    if player.get("photo_path"):
        st.image(player["photo_path"], use_container_width=True)
    else:
        st.markdown(
            f"<div class='team-initials'>{initials_for_name(player['full_name'])}</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="team-card">
          <div class="team-card-title">{player['full_name']}</div>
          <div class="team-card-meta">
            Wellness {format_metric_value(player.get('wellness_value'))} | RPE {format_metric_value(player.get('rpe_value'))}
          </div>
          <div class="team-card-meta">
            GPS {format_metric_value(player.get('gps_value'), ' m')} | Laatste update {player['wellness_date'].strftime("%d-%m") if player.get('wellness_date') else '--'}
          </div>
          <span class="team-status" style="background:{player['readiness_color']};">
            {player['readiness_label']}
          </span>
          <div class="team-card-meta" style="margin-top:0.65rem;">
            Vandaag wellness: {"Ja" if player.get('wellness_today') else "Nee"} | Vandaag RPE: {"Ja" if player.get('rpe_today') else "Nee"}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_group_section(df: pd.DataFrame, group_name: str, label_en: str) -> None:
    group_df = df[df["group"] == group_name].copy()
    if group_df.empty:
        return

    group_df = group_df.sort_values(["readiness_rank", "full_name"], ascending=[True, True]).reset_index(drop=True)

    st.markdown('<div class="team-section">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="team-section-head">
          <div>
            <div class="team-section-title">{group_name}</div>
            <div class="team-section-copy">{label_en}</div>
          </div>
          <div class="team-count">{len(group_df)} spelers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols_per_row = 4
    rows_needed = int(math.ceil(len(group_df) / cols_per_row))
    for row_idx in range(rows_needed):
        cols = st.columns(cols_per_row, gap="large")
        slice_df = group_df.iloc[row_idx * cols_per_row: (row_idx + 1) * cols_per_row]
        for col, (_, player_row) in zip(cols, slice_df.iterrows()):
            with col:
                render_player_card(player_row.to_dict())
        if row_idx < rows_needed - 1:
            st.markdown('<div class="team-grid-gap"></div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    render_css()

    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    profile = get_profile(sb) or {}
    role = str(profile.get("role") or "").lower()
    access_scope = f"{role}:{profile.get('user_id', 'anon')}"
    if role == "player":
        st.info("Deze teamweergave toont alleen spelers waarvoor je app-toegang hebt.")

    squad_df = assemble_team_rows(sb, access_scope)
    if squad_df.empty:
        st.warning("Geen actieve spelers gevonden.")
        st.stop()

    render_hero(squad_df)

    for group_name, label_en in GROUP_ORDER:
        render_group_section(squad_df, group_name, label_en)


if __name__ == "__main__":
    main()
