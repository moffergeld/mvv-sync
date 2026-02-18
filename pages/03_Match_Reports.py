# pages/03_Match_Reports.py
# ============================================================
# Match Reports (Streamlit)
# - Opponent dropdown (A->Z) + Date dropdown (YYYY-MM-DD (H/A))
# - Header: beide logo's zelfde grootte, gecentreerd, scorecard-achtige tekst (mooier)
# - Data uit public.matches + view public.v_gps_match_events (met match_id)
# - Halve selectie: Full match / First half / Second half
# - 2 Plotly charts (zelfde stijl als player pages): TD bar + Sprint vs High Sprint
# - Tables: 1 rij met 5 aparte tabellen (TD, 14.4–19.7, 19.8–25.1, 25.2+, Max Speed)
# - In tables: géén Duration, géén index/nummering
# - Sort dropdown op per-minute metric
# - Kleur op percentielen van ABSOLUTE waardes (niet /min)
# - Formatting: absolute getallen 0 decimaal, /min 2 decimaal
# - Alignment: numerieke kolommen gecentreerd, Player niet gecentreerd
# ============================================================

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from roles import get_sb, require_auth, get_profile


# -----------------------------
# Config
# -----------------------------
MATCHES_TABLE = "matches"
EVENTS_VIEW = "v_gps_match_events"

TEAM_LOGO_DIR = Path("Assets/Afbeeldingen/Team_Logos")  # repo pad
MVV_TEAM_NAME = "MVV Maastricht"

# Metrics (kolommen in v_gps_match_events)
METRICS = {
    "TD": "total_distance",
    "14.4–19.7": "running",
    "19.8–25.1": "sprint",
    "25.2+": "high_sprint",
    "Max Speed": "max_speed",
}
PER_MIN_SORT_OPTIONS = [
    ("Total Distance/min", "total_distance"),
    ("14.4–19.7 km/h/min", "running"),
    ("19.8–25.1 km/h/min", "sprint"),
    ("25.2+ km/h/min", "high_sprint"),
]


# -----------------------------
# Helpers
# -----------------------------
def _home_away_short(v: Any) -> str:
    s = (str(v or "")).strip().lower()
    if s.startswith("h"):
        return "H"
    if s.startswith("a"):
        return "A"
    return ""


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return int(v)
    except Exception:
        return default


def _fmt_score(gf: Any, ga: Any) -> str:
    return f"{_safe_int(gf)} - {_safe_int(ga)}"


def _df_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _coerce_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _normalize_team_name(name: str) -> str:
    # probeer bestand te matchen met minimale normalisatie
    return (
        name.replace("/", "-")
        .replace("\\", "-")
        .replace(":", "")
        .replace("*", "")
        .replace("?", "")
        .replace('"', "")
        .replace("<", "")
        .replace(">", "")
        .replace("|", "")
        .strip()
    )


def _find_logo_path(team_name: str) -> Optional[Path]:
    if not TEAM_LOGO_DIR.exists():
        return None

    candidates = [
        TEAM_LOGO_DIR / f"{team_name}.png",
        TEAM_LOGO_DIR / f"{_normalize_team_name(team_name)}.png",
    ]

    for p in candidates:
        if p.exists():
            return p

    # fallback: case-insensitive search
    target = _normalize_team_name(team_name).lower()
    for p in TEAM_LOGO_DIR.glob("*.png"):
        if _normalize_team_name(p.stem).lower() == target:
            return p

    return None


def _center_image(path: Optional[Path], width: int = 155):
    if not path or not Path(path).exists():
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        return
    st.image(str(path), width=width, use_container_width=False)


def _header_css():
    st.markdown(
        """
        <style>
        .mr-wrap { text-align:center; }
        .mr-date { font-size:16px; font-weight:700; opacity:0.95; margin: 0 0 6px 0; }
        .mr-fixture { font-size:26px; font-weight:900; margin:0 0 6px 0; }
        .mr-score { font-size:56px; font-weight:950; line-height:1; margin: 6px 0 10px 0; }
        .mr-meta { font-size:14px; opacity:0.85; margin: 4px 0; }
        .mr-pill { display:inline-block; padding: 4px 10px; border-radius: 999px;
                   background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10);
                   font-size: 13px; opacity: 0.9; }
        .mr-subline { display:flex; justify-content:center; gap:10px; flex-wrap:wrap; margin-top: 6px; }
        .mr-teamname { font-size:14px; font-weight:800; margin-top: 8px; opacity: 0.95; text-align:center; }

        /* tables: minder padding zodat alles past, geen "in-table" scroll gevoel */
        div[data-testid="stDataFrame"] { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    ha = "Home" if _home_away_short(home_away) == "H" else "Away"
    score = _fmt_score(gf, ga)

    st.markdown("<div class='mr-wrap'>", unsafe_allow_html=True)

    st.markdown(f"<div class='mr-date'>{match_date.isoformat()}</div>", unsafe_allow_html=True)

    # Fixture regel (mooier en los van score)
    fixture_txt = fixture or f"{MVV_TEAM_NAME} - {opponent}"
    st.markdown(f"<div class='mr-fixture'>{fixture_txt}</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='mr-score'>{score}</div>", unsafe_allow_html=True)

    pills = []
    if ha:
        pills.append(ha)
    if match_type:
        pills.append(str(match_type))
    if season:
        pills.append(str(season))

    st.markdown(
        "<div class='mr-subline'>"
        + "".join([f"<span class='mr-pill'>{p}</span>" for p in pills])
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(f"<div class='mr-meta'>Opponent: {opponent}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Supabase fetch
# -----------------------------
def fetch_matches(sb, limit: int = 1500) -> pd.DataFrame:
    try:
        rows = (
            sb.table(MATCHES_TABLE)
            .select("match_id,match_date,fixture,home_away,opponent,match_type,season,result,goals_for,goals_against")
            .order("match_date", desc=True)
            .limit(limit)
            .execute()
            .data
            or []
        )
        df = _df_from_rows(rows)
        if df.empty:
            return df
        df["match_date"] = _coerce_date_series(df["match_date"])
        df["opponent"] = df["opponent"].fillna("").astype(str)
        df["home_away_short"] = df["home_away"].apply(_home_away_short)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_match_events(sb, match_id: int) -> pd.DataFrame:
    # Verwacht dat v_gps_match_events match_id bevat
    try:
        rows = (
            sb.table(EVENTS_VIEW)
            .select(
                "match_id,player_name,event,duration,total_distance,running,sprint,high_sprint,max_speed"
            )
            .eq("match_id", match_id)
            .in_("event", ["First Half", "Second Half"])
            .execute()
            .data
            or []
        )
        df = _df_from_rows(rows)
        if df.empty:
            return df
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
        for c in ["total_distance", "running", "sprint", "high_sprint", "max_speed"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df["player_name"] = df["player_name"].fillna("").astype(str)
        df["event"] = df["event"].fillna("").astype(str)
        return df
    except Exception:
        return pd.DataFrame()


# -----------------------------
# Aggregation
# -----------------------------
def _aggregate_for_period(df_events: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    period: 'Full match' / 'First half' / 'Second half'
    Returns per player:
      duration_sum, sums for distance-buckets, max for max_speed
    """
    if df_events.empty:
        return df_events

    if period == "First half":
        dff = df_events[df_events["event"] == "First Half"].copy()
    elif period == "Second half":
        dff = df_events[df_events["event"] == "Second Half"].copy()
    else:
        dff = df_events.copy()

    if dff.empty:
        return pd.DataFrame()

    agg = (
        dff.groupby("player_name", as_index=False)
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

    # per minute
    dur = agg["duration"].replace(0, np.nan)
    for k in ["total_distance", "running", "sprint", "high_sprint"]:
        agg[f"{k}_per_min"] = (agg[k] / dur).astype(float)
    agg["max_speed_per_min"] = np.nan  # niet gebruikt, maar handig consistent

    # duration terug naar 0 waar NaN (na deling)
    agg["duration"] = agg["duration"].fillna(0.0)
    for c in [f"{k}_per_min" for k in ["total_distance", "running", "sprint", "high_sprint"]]:
        agg[c] = agg[c].fillna(0.0)

    return agg


# -----------------------------
# Styling (percentile colors + formatting)
# -----------------------------
def _percentile_thresholds(vals: pd.Series) -> Tuple[float, float, float, float]:
    v = pd.to_numeric(vals, errors="coerce").dropna()
    if v.empty:
        return (0, 0, 0, 0)
    return tuple(np.percentile(v, [20, 40, 60, 80]).tolist())  # type: ignore


def _cell_bg_color(val: Any, p20: float, p40: float, p60: float, p80: float) -> str:
    try:
        x = float(val)
    except Exception:
        return ""
    # low = rood, high = groen (zoals jouw screenshot)
    if x <= p20:
        return "background-color: rgba(255, 0, 51, 0.28);"
    if x <= p40:
        return "background-color: rgba(255, 0, 51, 0.18);"
    if x <= p60:
        return "background-color: rgba(255, 204, 0, 0.16);"
    if x <= p80:
        return "background-color: rgba(0, 200, 0, 0.16);"
    return "background-color: rgba(0, 200, 0, 0.26);"


def _style_table(df: pd.DataFrame, abs_col: str, per_min_col: Optional[str] = None) -> "pd.io.formats.style.Styler":
    """
    - abs_col: kleur op percentielen (absolute)
    - per_min_col: 2 decimaal, geen kleur
    - alignment: Player left, numeriek center
    - formatting: abs 0 dec, /min 2 dec
    """
    show = df.copy()

    # formatting helpers
    fmt: Dict[str, Any] = {}
    if abs_col in show.columns:
        fmt[abs_col] = "{:,.0f}"
    if per_min_col and per_min_col in show.columns:
        fmt[per_min_col] = "{:,.2f}"

    # thresholds for coloring (ABSOLUTE)
    p20, p40, p60, p80 = _percentile_thresholds(show[abs_col]) if abs_col in show.columns else (0, 0, 0, 0)

    def _apply_abs_bg(s: pd.Series):
        return [
            _cell_bg_color(v, p20, p40, p60, p80) if s.name == abs_col else ""
            for v in s.values
        ]

    sty = show.style

    # kleur alleen op absolute kolom
    if abs_col in show.columns:
        sty = sty.apply(_apply_abs_bg, axis=0, subset=[abs_col])

    # format
    if fmt:
        sty = sty.format(fmt)

    # alignments
    numeric_cols = [c for c in show.columns if c != "Player"]
    if numeric_cols:
        sty = sty.set_properties(subset=numeric_cols, **{"text-align": "center"})
    sty = sty.set_properties(subset=["Player"], **{"text-align": "left", "font-weight": "700"})

    return sty


def _table_height_from_rows(n_rows: int) -> int:
    # Geen interne scroll: maak de dataframe hoog genoeg voor alle rijen
    # (pagina mag wél scrollen, maar de tabel zelf niet)
    header = 36
    row_h = 28
    pad = 14
    return max(220, header + n_rows * row_h + pad)


# -----------------------------
# Plotly charts
# -----------------------------
def plot_total_distance_bar(df_agg: pd.DataFrame, period_label: str):
    if df_agg.empty:
        st.info("Geen data voor grafiek.")
        return

    dff = df_agg.sort_values("total_distance", ascending=False).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dff["player_name"],
            y=dff["total_distance"],
            marker=dict(color="#FF0033"),
        )
    )
    fig.update_layout(
        title=f"Total Distance ({period_label})",
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
        xaxis_title="",
        yaxis_title="",
        showlegend=False,
    )
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


def plot_sprint_vs_highsprint(df_agg: pd.DataFrame, period_label: str):
    if df_agg.empty:
        st.info("Geen data voor grafiek.")
        return

    dff = df_agg.sort_values("sprint", ascending=False).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dff["player_name"], y=dff["sprint"], name="sprint", marker=dict(color="rgba(255,0,51,0.55)")))
    fig.add_trace(go.Bar(x=dff["player_name"], y=dff["high_sprint"], name="high_sprint", marker=dict(color="#FF0033")))
    fig.update_layout(
        title=f"Sprint vs High Sprint ({period_label})",
        barmode="group",
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
        xaxis_title="",
        yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Main
# -----------------------------
def main():
    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    _header_css()

    st.title("Match Reports")

    # ---- Load matches
    matches = fetch_matches(sb, limit=2000)
    if matches.empty:
        st.info("Geen matches gevonden.")
        st.stop()

    # ---- Opponent dropdown (A->Z)
    opps = sorted([o for o in matches["opponent"].unique().tolist() if str(o).strip() != ""], key=lambda x: str(x).lower())
    if not opps:
        st.info("Geen opponents gevonden in matches.")
        st.stop()

    c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
    with c1:
        opp_sel = st.selectbox("Opponent", options=opps, index=0, key="mr_opp")

    # ---- Date dropdown for selected opponent (with H/A)
    m_opp = matches[matches["opponent"] == opp_sel].copy()
    m_opp = m_opp.sort_values("match_date", ascending=False)

    def _date_label(row) -> str:
        d = row["match_date"]
        ha = row.get("home_away_short", "")
        return f"{d.isoformat()} ({ha})"

    m_opp["date_label"] = m_opp.apply(_date_label, axis=1)

    # map label -> match_id (if duplicates, keep latest row order)
    label_to_match: Dict[str, int] = {}
    for _, r in m_opp.iterrows():
        label_to_match[r["date_label"]] = int(r["match_id"])

    date_labels = list(label_to_match.keys())
    with c2:
        date_sel = st.selectbox("Date", options=date_labels, index=0, key="mr_date")

    match_id = label_to_match[date_sel]
    match_row = m_opp[m_opp["match_id"] == match_id].iloc[0].to_dict()

    # ---- Header: logos same size + centered, nicer text
    match_date = match_row.get("match_date")
    fixture = str(match_row.get("fixture") or "").strip()
    home_away = str(match_row.get("home_away") or "").strip()
    match_type = str(match_row.get("match_type") or "").strip()
    season = str(match_row.get("season") or "").strip()
    gf = match_row.get("goals_for")
    ga = match_row.get("goals_against")

    # logos
    mvv_logo = _find_logo_path(MVV_TEAM_NAME)
    opp_logo = _find_logo_path(opp_sel)

    left, mid, right = st.columns([1.1, 1.8, 1.1], vertical_alignment="center")

    with left:
        _center_image(mvv_logo, width=155)
        st.markdown(f"<div class='mr-teamname'>{MVV_TEAM_NAME}</div>", unsafe_allow_html=True)

    with mid:
        _header_block(
            match_date=match_date,
            home_away=home_away,
            opponent=opp_sel,
            match_type=match_type,
            season=season,
            fixture=fixture,
            gf=gf,
            ga=ga,
        )

    with right:
        _center_image(opp_logo, width=155)
        st.markdown(f"<div class='mr-teamname'>{opp_sel}</div>", unsafe_allow_html=True)

    st.divider()

    # ---- Controls
    sort_label, sort_key = st.selectbox(
        "Sort tables on (per minute)",
        options=PER_MIN_SORT_OPTIONS,
        index=0,
        format_func=lambda x: x[0],
        key="mr_sort",
    )
    sort_metric = sort_key  # like "total_distance"

    st.markdown("**Tables**")
    period = st.radio(" ", ["Full match", "First half", "Second half"], horizontal=True, key="mr_period")

    # ---- Load + aggregate events for this match
    events = fetch_match_events(sb, match_id=match_id)
    if events.empty:
        st.info("Geen match events gevonden (v_gps_match_events).")
        st.stop()

    agg = _aggregate_for_period(events, period=period)
    if agg.empty:
        st.info("Geen data voor deze selectie.")
        st.stop()

    # ---- Sorting (per minute)
    sort_col = f"{sort_metric}_per_min"
    if sort_col in agg.columns:
        agg = agg.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # ---- Charts row (match selection)
    ch_l, ch_r = st.columns(2)
    with ch_l:
        plot_total_distance_bar(agg, period_label=period)
    with ch_r:
        plot_sprint_vs_highsprint(agg, period_label=period)

    st.markdown("## Tables")

    # Build small tables (1 row, 5 columns)
    # Table content: Player + ABS + /min (except Max Speed)
    base = agg.copy()
    base = base.rename(columns={"player_name": "Player"})

    # TD table
    td_df = base[["Player", "total_distance", "total_distance_per_min"]].rename(
        columns={"total_distance": "TD", "total_distance_per_min": "/min"}
    )
    td_sty = _style_table(td_df, abs_col="TD", per_min_col="/min")

    # 14.4–19.7
    r_df = base[["Player", "running", "running_per_min"]].rename(columns={"running": "14.4–19.7", "running_per_min": "/min"})
    r_sty = _style_table(r_df, abs_col="14.4–19.7", per_min_col="/min")

    # 19.8–25.1
    s_df = base[["Player", "sprint", "sprint_per_min"]].rename(columns={"sprint": "19.8–25.1", "sprint_per_min": "/min"})
    s_sty = _style_table(s_df, abs_col="19.8–25.1", per_min_col="/min")

    # 25.2+
    hs_df = base[["Player", "high_sprint", "high_sprint_per_min"]].rename(columns={"high_sprint": "25.2+", "high_sprint_per_min": "/min"})
    hs_sty = _style_table(hs_df, abs_col="25.2+", per_min_col="/min")

    # Max speed (0 decimals)
    ms_df = base[["Player", "max_speed"]].rename(columns={"max_speed": "Max Speed"})
    ms_sty = _style_table(ms_df, abs_col="Max Speed", per_min_col=None).format({"Max Speed": "{:,.0f}"})

    # Heights: force show all players without internal scroll
    n_rows = int(len(base))
    h = _table_height_from_rows(n_rows)

    t1, t2, t3, t4, t5 = st.columns([1.05, 1.0, 1.0, 1.0, 1.0], gap="small")
    with t1:
        st.dataframe(td_sty, use_container_width=True, hide_index=True, height=h)
    with t2:
        st.dataframe(r_sty, use_container_width=True, hide_index=True, height=h)
    with t3:
        st.dataframe(s_sty, use_container_width=True, hide_index=True, height=h)
    with t4:
        st.dataframe(hs_sty, use_container_width=True, hide_index=True, height=h)
    with t5:
        st.dataframe(ms_sty, use_container_width=True, hide_index=True, height=h)


if __name__ == "__main__":
    main()
