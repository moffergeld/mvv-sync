# pages/03_Match_Reports.py
# ============================================================
# Match Reports (Streamlit)
# - Opponent dropdown (A-Z) + Date dropdown (YYYY-MM-DD (H/A))
# - Header: 2 logos same size + centered, nicer multi-line match info
# - Uses Plotly (zelfde stijl als Player Pages)
# - Uses v_gps_match_events (event = First Half / Second Half)
# - Tables:
#   * Radio: Full match / First half / Second half
#   * 1 rij met 5 losse tabellen (TD, 14.4–19.7, 19.8–25.1, 25.2+, Max Speed)
#   * Geen duration-kolom, geen index-kolom
#   * Getallen: 0 decimaal (absolute), /min: 2 decimaal
#   * Numerieke kolommen gecentreerd; Player links
#   * Kleuren op percentielen van ABSOLUTE waardes
#   * Sorteren op gekozen "/min" metric
# - Geen caching met sb (voorkomt UnhashableParamError)
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from roles import get_sb, require_auth, get_profile

# ---------- Config ----------
MATCHES_TABLE = "matches"
MATCH_EVENTS_VIEW = "v_gps_match_events"  # verwacht: event = 'First Half'/'Second Half'
TEAM_LOGOS_DIR = Path("Assets/Afbeeldingen/Team_Logos")

MVV_TEAM_NAME = "MVV Maastricht"
MVV_LOGO_FILENAME_CANDIDATES = [
    "MVV Maastricht.png",
    "MVV.png",
    "MVV Maastricht FC.png",
]

EVENT_FULL = "Full match"
EVENT_FIRST = "First half"
EVENT_SECOND = "Second half"

EVENT_MAP = {
    EVENT_FULL: ["First Half", "Second Half"],
    EVENT_FIRST: ["First Half"],
    EVENT_SECOND: ["Second Half"],
}

# We tonen exact deze metrics in tabellen + grafieken
METRICS = [
    ("TD", "total_distance"),          # meters
    ("14.4–19.7", "running"),
    ("19.8–25.1", "sprint"),
    ("25.2+", "high_sprint"),
    ("Max Speed", "max_speed"),        # km/u
]

SORT_OPTIONS = [
    ("Total Distance/min", "total_distance"),
    ("14.4–19.7/min", "running"),
    ("19.8–25.1/min", "sprint"),
    ("25.2+/min", "high_sprint"),
]

# ---------- Helpers ----------
def _df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _safe_int(x, default=0) -> int:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return int(float(x))
    except Exception:
        return default


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def _as_date(x) -> Optional[date]:
    try:
        d = pd.to_datetime(x, errors="coerce").date()
        return d
    except Exception:
        return None


def _fmt_score(gf: Any, ga: Any) -> str:
    if gf is None or ga is None:
        return "—"
    try:
        return f"{int(gf)} - {int(ga)}"
    except Exception:
        return "—"


def _home_away_short(v: Any) -> str:
    s = str(v or "").strip().lower()
    if s.startswith("h"):
        return "H"
    if s.startswith("a"):
        return "A"
    return "?"


def _logo_path_for_team(team: str) -> Optional[Path]:
    # Exact match
    p = TEAM_LOGOS_DIR / f"{team}.png"
    if p.exists():
        return p

    # Common alternatives / minor normalizations
    alt = team.replace("/", "-").replace(":", "").strip()
    p2 = TEAM_LOGOS_DIR / f"{alt}.png"
    if p2.exists():
        return p2

    # Try case-insensitive search
    if TEAM_LOGOS_DIR.exists():
        for f in TEAM_LOGOS_DIR.glob("*.png"):
            if f.stem.lower() == team.lower():
                return f
            if f.stem.lower() == alt.lower():
                return f

    return None


def _mvv_logo_path() -> Optional[Path]:
    for fn in MVV_LOGO_FILENAME_CANDIDATES:
        p = TEAM_LOGOS_DIR / fn
        if p.exists():
            return p
    return _logo_path_for_team(MVV_TEAM_NAME)


def _center_image(path: Optional[Path], width: int):
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    if path and path.exists():
        st.image(str(path), width=width)
    else:
        st.write("")  # fallback leeg
    st.markdown("</div>", unsafe_allow_html=True)


def _header_block(
    match_date: date,
    home_away: str,
    opponent: str,
    match_type: str,
    season: str,
    fixture: str,
    gf: Any,
    ga: Any,
):
    # CSS spacing (klein beetje zoals voorbeeld)
    st.markdown(
        """
        <style>
        .mr-title { font-size: 28px; font-weight: 800; margin: 0 0 4px 0; }
        .mr-sub   { font-size: 15px; font-weight: 600; margin: 0 0 10px 0; opacity: 0.95; }
        .mr-meta  { font-size: 13px; margin: 0; opacity: 0.75; }
        .mr-score { font-size: 30px; font-weight: 900; margin: 10px 0 4px 0; }
        .mr-gap   { height: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    title = f"{match_date.isoformat()} • {MVV_TEAM_NAME} - {opponent}"
    score = _fmt_score(gf, ga)

    # netjes op meerdere regels + wat lucht
    st.markdown(f"<div class='mr-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='mr-score'>Score: {score}</div>", unsafe_allow_html=True)

    parts = []
    if home_away:
        parts.append("Home" if _home_away_short(home_away) == "H" else "Away")
    if match_type:
        parts.append(str(match_type))
    if season:
        parts.append(str(season))
    st.markdown(f"<div class='mr-sub'>{' • '.join(parts) if parts else ''}</div>", unsafe_allow_html=True)

    if fixture:
        st.markdown(f"<p class='mr-meta'>Fixture: {fixture}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='mr-meta'>Opponent: {opponent}</p>", unsafe_allow_html=True)


def fetch_matches(sb, limit: int = 2000) -> pd.DataFrame:
    rows = (
        sb.table(MATCHES_TABLE)
        .select("match_id,match_date,fixture,home_away,opponent,match_type,season,result,goals_for,goals_against")
        .order("match_date", desc=True)
        .limit(limit)
        .execute()
        .data
        or []
    )
    df = _df(rows)
    if df.empty:
        return df
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.date
    df["opponent"] = df["opponent"].astype(str)
    return df


def fetch_match_events(sb, match_id: int, fallback_date: Optional[date] = None) -> pd.DataFrame:
    """
    Liefst filteren op match_id (als aanwezig in view).
    Fallback: filter op datum + type='Match' (als match_id ontbreekt).
    """
    # probeer match_id in view
    try:
        rows = (
            sb.table(MATCH_EVENTS_VIEW)
            .select(
                "gps_id,match_id,player_id,player_name,datum,type,event,duration,"
                "total_distance,running,sprint,high_sprint,max_speed"
            )
            .eq("match_id", match_id)
            .execute()
            .data
            or []
        )
        df = _df(rows)
        if not df.empty:
            df["datum"] = pd.to_datetime(df["datum"], errors="coerce").dt.date
            return df
    except Exception:
        pass

    # fallback (geen match_id in view)
    if fallback_date is None:
        return pd.DataFrame()

    try:
        rows = (
            sb.table(MATCH_EVENTS_VIEW)
            .select(
                "gps_id,player_id,player_name,datum,type,event,duration,"
                "total_distance,running,sprint,high_sprint,max_speed"
            )
            .eq("datum", fallback_date.isoformat())
            .eq("type", "Match")
            .execute()
            .data
            or []
        )
        df = _df(rows)
        if df.empty:
            return df
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce").dt.date
        return df
    except Exception:
        return pd.DataFrame()


def _aggregate_players(events_df: pd.DataFrame, event_choice: str) -> pd.DataFrame:
    if events_df.empty:
        return events_df

    df = events_df.copy()
    allowed = EVENT_MAP[event_choice]
    df = df[df["event"].isin(allowed)].copy()
    if df.empty:
        return df

    # numeric
    for c in ["duration", "total_distance", "running", "sprint", "high_sprint", "max_speed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # sums for distance-like, max for max_speed; duration sum
    agg = (
        df.groupby("player_name", as_index=False)
        .agg(
            duration=("duration", "sum"),
            total_distance=("total_distance", "sum"),
            running=("running", "sum"),
            sprint=("sprint", "sum"),
            high_sprint=("high_sprint", "sum"),
            max_speed=("max_speed", "max"),
        )
        .copy()
    )

    # per minute (avoid /0)
    dur = agg["duration"].astype(float).replace(0, np.nan)
    for _, key in METRICS:
        if key == "max_speed":
            continue
        agg[f"{key}_min"] = agg[key].astype(float) / dur
    agg["duration"] = agg["duration"].fillna(0)

    return agg


def _percentile_colors(values: pd.Series) -> List[str]:
    """
    Discrete percentile -> color (RdYlGn-ish).
    Hoog = groen, laag = rood.
    """
    v = pd.to_numeric(values, errors="coerce")
    if v.notna().sum() < 2:
        return ["transparent"] * len(values)

    qs = v.quantile([0.2, 0.4, 0.6, 0.8]).tolist()

    def pick(x):
        if pd.isna(x):
            return "transparent"
        if x <= qs[0]:
            return "rgba(255,0,0,0.28)"
        if x <= qs[1]:
            return "rgba(255,120,0,0.22)"
        if x <= qs[2]:
            return "rgba(255,200,0,0.18)"
        if x <= qs[3]:
            return "rgba(120,255,120,0.18)"
        return "rgba(0,200,0,0.22)"

    return [pick(x) for x in v.tolist()]


def _styled_table(
    df: pd.DataFrame,
    abs_col: str,
    per_min_col: Optional[str],
    col_label_abs: str,
    col_label_min: Optional[str],
    sort_by_min_key: str,
) -> "pd.io.formats.style.Styler":
    """
    Output table:
    - Player (left)
    - ABS metric (0 dec)
    - /min metric (2 dec) if present
    """
    if df.empty:
        return pd.DataFrame({"Player": []}).style

    out_cols = ["player_name", abs_col]
    rename = {"player_name": "Player", abs_col: col_label_abs}

    if per_min_col:
        out_cols.append(per_min_col)
        rename[per_min_col] = col_label_min or "/min"

    t = df[out_cols].rename(columns=rename).copy()

    # rounding requirements
    if col_label_abs in t.columns:
        t[col_label_abs] = pd.to_numeric(t[col_label_abs], errors="coerce").round(0)
    if col_label_min and col_label_min in t.columns:
        t[col_label_min] = pd.to_numeric(t[col_label_min], errors="coerce").round(2)

    # sort on selected per-min metric (descending)
    # NOTE: sort_by_min_key is e.g. "total_distance" -> column "total_distance_min"
    sort_col = f"{sort_by_min_key}_min"
    if sort_col in df.columns:
        order = df.sort_values(sort_col, ascending=False)["player_name"].tolist()
        t["__order__"] = pd.Categorical(t["Player"], categories=order, ordered=True)
        t = t.sort_values("__order__").drop(columns="__order__")

    # colors based on ABS values only (percentiles)
    colors = _percentile_colors(t[col_label_abs] if col_label_abs in t.columns else pd.Series(dtype=float))

    def _apply_abs_col(s: pd.Series):
        return [f"background-color: {c};" for c in colors]

    styler = t.style

    if col_label_abs in t.columns:
        styler = styler.apply(_apply_abs_col, subset=[col_label_abs])

    # alignment: Player left, numerics center
    num_cols = [c for c in t.columns if c != "Player"]
    styler = styler.set_properties(subset=["Player"], **{"text-align": "left"})
    if num_cols:
        styler = styler.set_properties(subset=num_cols, **{"text-align": "center"})

    # formatting: 0 decimals for abs; 2 for /min
    fmt: Dict[str, str] = {}
    if col_label_abs in t.columns:
        fmt[col_label_abs] = "{:.0f}"
    if col_label_min and col_label_min in t.columns:
        fmt[col_label_min] = "{:.2f}"
    styler = styler.format(fmt, na_rep="")

    # make header nicer
    styler = styler.set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "left")]},
            {"selector": "td", "props": [("padding", "6px 10px")]},
        ]
    )

    return styler


def _render_table_in_column(styler, n_rows: int):
    # st.dataframe scrollt snel; st.table toont alles maar is minder interactief.
    # Hier: st.table met styler => geen scroll + alles zichtbaar.
    st.table(styler)


def _plot_total_distance_bar(df: pd.DataFrame, title: str):
    if df.empty:
        st.info("Geen data voor grafiek.")
        return

    d = df.copy()
    d["total_distance"] = pd.to_numeric(d["total_distance"], errors="coerce").fillna(0)
    d = d.sort_values("total_distance", ascending=False)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=d["player_name"],
            y=d["total_distance"],
            marker=dict(color="#FF0033"),
        )
    )
    fig.update_layout(
        title=title,
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(tickangle=90),
        yaxis_title="m",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_sprint_vs_high_sprint(df: pd.DataFrame, title: str):
    if df.empty:
        st.info("Geen data voor grafiek.")
        return

    d = df.copy()
    for c in ["sprint", "high_sprint"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    d = d.sort_values("sprint", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["player_name"], y=d["sprint"], name="sprint", marker=dict(color="rgba(255,0,51,0.55)")))
    fig.add_trace(go.Bar(x=d["player_name"], y=d["high_sprint"], name="high_sprint", marker=dict(color="#FF0033")))
    fig.update_layout(
        title=title,
        barmode="group",
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(tickangle=90),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="m")
    st.plotly_chart(fig, use_container_width=True)


# ---------- UI ----------
def main():
    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    profile = get_profile(sb)

    st.title("Match Reports")

    matches = fetch_matches(sb, limit=2000)
    if matches.empty:
        st.info("Geen matches gevonden in tabel 'matches'.")
        st.stop()

    # Opponent dropdown (A-Z)
    opponents = sorted(matches["opponent"].dropna().unique().tolist(), key=lambda x: str(x).lower())

    # Layout: opponent + date
    c1, c2 = st.columns([1, 2])
    with c1:
        opp = st.selectbox("Opponent", options=opponents, index=0)
    opp_df = matches[matches["opponent"] == opp].copy().sort_values("match_date", ascending=False)

    # Date dropdown: YYYY-MM-DD (H/A)
    opp_df["ha_short"] = opp_df["home_away"].apply(_home_away_short)
    opp_df["date_label"] = opp_df["match_date"].apply(lambda d: d.isoformat()) + " (" + opp_df["ha_short"] + ")"

    labels = opp_df["date_label"].tolist()

    with c2:
        date_label = st.selectbox("Date", options=labels, index=0)

    row = opp_df[opp_df["date_label"] == date_label].iloc[0].to_dict()
    match_id = _safe_int(row.get("match_id"))
    match_date = row.get("match_date")
    opponent = row.get("opponent") or ""
    home_away = row.get("home_away") or ""
    fixture = row.get("fixture") or ""
    match_type = row.get("match_type") or ""
    season = row.get("season") or ""
    gf = row.get("goals_for")
    ga = row.get("goals_against")

    # Header
    mvv_logo = _mvv_logo_path()
    opp_logo = _logo_path_for_team(opponent)

    left, mid, right = st.columns([1, 2.4, 1], vertical_alignment="center")
    with left:
        _center_image(mvv_logo, width=170)
    with mid:
        _header_block(
            match_date=match_date,
            home_away=home_away,
            opponent=opponent,
            match_type=match_type,
            season=season,
            fixture=fixture,
            gf=gf,
            ga=ga,
        )
    with right:
        _center_image(opp_logo, width=170)

    st.divider()

    # Sorting (per minute)
    sort_label, sort_key = st.selectbox("Sort tables on (per minute)", options=SORT_OPTIONS, index=0)

    # Table selection radio (bolletjes)
    st.markdown("**Tables**")
    event_choice = st.radio(
        "",
        options=[EVENT_FULL, EVENT_FIRST, EVENT_SECOND],
        horizontal=True,
        index=0,
        key="mr_event_choice",
    )

    # Load events and aggregate
    events = fetch_match_events(sb, match_id=match_id, fallback_date=match_date)
    if events.empty:
        st.info("Geen match events gevonden (v_gps_match_events) voor deze selectie.")
        st.stop()

    agg = _aggregate_players(events, event_choice=event_choice)
    if agg.empty:
        st.info("Geen data na filter (First/Second half).")
        st.stop()

    # Graphs (zelfde selectie als radio)
    g1, g2 = st.columns(2)
    with g1:
        _plot_total_distance_bar(agg, title=f"Total Distance ({event_choice})")
    with g2:
        _plot_sprint_vs_high_sprint(agg, title=f"Sprint vs High Sprint ({event_choice})")

    st.markdown("## Tables")

    # 1 rij met 5 losse tabellen
    tcols = st.columns([1.15, 1, 1, 1, 1.05])

    # TD table
    with tcols[0]:
        sty = _styled_table(
            df=agg,
            abs_col="total_distance",
            per_min_col="total_distance_min",
            col_label_abs="TD",
            col_label_min="/min",
            sort_by_min_key=sort_key,
        )
        _render_table_in_column(sty, n_rows=len(agg))

    # 14.4–19.7
    with tcols[1]:
        sty = _styled_table(
            df=agg,
            abs_col="running",
            per_min_col="running_min",
            col_label_abs="14.4–19.7",
            col_label_min="/min",
            sort_by_min_key=sort_key,
        )
        _render_table_in_column(sty, n_rows=len(agg))

    # 19.8–25.1
    with tcols[2]:
        sty = _styled_table(
            df=agg,
            abs_col="sprint",
            per_min_col="sprint_min",
            col_label_abs="19.8–25.1",
            col_label_min="/min",
            sort_by_min_key=sort_key,
        )
        _render_table_in_column(sty, n_rows=len(agg))

    # 25.2+
    with tcols[3]:
        sty = _styled_table(
            df=agg,
            abs_col="high_sprint",
            per_min_col="high_sprint_min",
            col_label_abs="25.2+",
            col_label_min="/min",
            sort_by_min_key=sort_key,
        )
        _render_table_in_column(sty, n_rows=len(agg))

    # Max Speed (geen /min)
    with tcols[4]:
        sty = _styled_table(
            df=agg,
            abs_col="max_speed",
            per_min_col=None,
            col_label_abs="Max Speed",
            col_label_min=None,
            sort_by_min_key=sort_key,
        )
        _render_table_in_column(sty, n_rows=len(agg))


if __name__ == "__main__":
    main()
