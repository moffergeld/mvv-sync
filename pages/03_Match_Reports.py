# pages/03_Match_Reports.py
# ============================================================
# Match Reports (Streamlit)
# Update:
# - Beter voorpagina / header:
#   - Logos kleiner + gecentreerd
#   - Fixture + Home/Away + Season + Match type
#   - Score (goals_for-goals_against) + Result
#   - Nettere layout (3 kolommen: logo - info - logo)
# - Tabellen: altijd volledig zichtbaar (geen interne scroll):
#   - automatische hoogte obv aantal rijen
# - Tables: radio bolletjes (Full match / First half / Second half)
# - Sort: op per-minute keuze (dropdown)
#
# Data:
# - matches table (public.matches)
# - view: public.v_gps_match_events (moet match_id bevatten)
#   kolommen o.a.: match_id, player_name, event, duration, total_distance, running, sprint, high_sprint, max_speed, ...
# ============================================================

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from roles import get_sb, require_auth, get_profile  # same style as other pages


# -----------------------------
# Config
# -----------------------------
MATCHES_TABLE = "matches"
MATCH_EVENTS_VIEW = "v_gps_match_events"  # must include match_id

# Logo folder inside repo (as you showed)
# mvv-sync/Assets/Afbeeldingen/Team_Logos/<Team>.png
LOGO_DIR = "Assets/Afbeeldingen/Team_Logos"
DEFAULT_LOGO = f"{LOGO_DIR}/default.png"  # optional (if you have it). If not, code falls back to None.

MVV_TEAM_NAME = "MVV Maastricht"
MVV_LOGO = f"{LOGO_DIR}/MVV Maastricht.png"

EVENT_FULL = "Full match"
EVENT_1H = "First Half"
EVENT_2H = "Second Half"
EVENT_CHOICES = [EVENT_FULL, EVENT_1H, EVENT_2H]

# GPS columns we need
METRIC_COLS = {
    "TD": "total_distance",
    "14.4–19.7": "running",
    "19.8–25.1": "sprint",
    "25.2+": "high_sprint",
    "Max Speed": "max_speed",
}

PER_MIN_OPTIONS = [
    ("Total Distance/min", "TD"),
    ("14.4–19.7 /min", "14.4–19.7"),
    ("19.8–25.1 /min", "19.8–25.1"),
    ("25.2+ /min", "25.2+"),
]


# -----------------------------
# Utils
# -----------------------------
def _df_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df


def _logo_path_for_team(team: str) -> Optional[str]:
    if not team:
        return None
    # assumes filenames exactly as opponent/team names in your folder
    return f"{LOGO_DIR}/{team}.png"


def _df_height_for_all_rows(df: pd.DataFrame, row_h: int = 32, header_h: int = 38, pad: int = 10) -> int:
    n = 0 if df is None else int(len(df))
    return header_h + (n * row_h) + pad


def _percentile_color(v: float, q20: float, q40: float, q60: float, q80: float) -> str:
    # higher is better -> green at top
    if pd.isna(v):
        return ""
    if v <= q20:
        return "background-color: rgba(255, 0, 51, 0.55);"   # red
    if v <= q40:
        return "background-color: rgba(255, 122, 0, 0.45);"  # orange
    if v <= q60:
        return "background-color: rgba(255, 204, 0, 0.35);"  # yellow
    if v <= q80:
        return "background-color: rgba(140, 200, 0, 0.30);"  # light green
    return "background-color: rgba(0, 160, 70, 0.35);"       # green


def _style_by_percentiles(df: pd.DataFrame, value_col: str) -> pd.io.formats.style.Styler:
    if df.empty or value_col not in df.columns:
        return df.style

    vals = _safe_num(df[value_col])
    qs = vals.quantile([0.2, 0.4, 0.6, 0.8]).tolist()
    q20, q40, q60, q80 = qs[0], qs[1], qs[2], qs[3]

    def _apply(s: pd.Series):
        out = []
        for x in s:
            out.append(_percentile_color(x, q20, q40, q60, q80))
        return out

    sty = df.style.apply(_apply, subset=[value_col])

    # a bit tighter table look
    sty = sty.set_table_styles(
        [
            {"selector": "th", "props": [("font-size", "12px"), ("text-align", "left")]},
            {"selector": "td", "props": [("font-size", "12px")]},
        ]
    )
    return sty


# -----------------------------
# Supabase fetch
# -----------------------------
@st.cache_data(ttl=120)
def fetch_matches(_sb, limit: int = 500) -> pd.DataFrame:
    rows = (
        _sb.table(MATCHES_TABLE)
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
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.date
    return df


@st.cache_data(ttl=120)
def fetch_match_events(_sb, match_id: int) -> pd.DataFrame:
    rows = (
        _sb.table(MATCH_EVENTS_VIEW)
        .select(
            "match_id,player_name,event,duration,total_distance,running,sprint,high_sprint,max_speed"
        )
        .eq("match_id", match_id)
        .execute()
        .data
        or []
    )
    df = _df_from_rows(rows)
    if df.empty:
        return df

    need = ["player_name", "event", "duration"] + list(METRIC_COLS.values())
    df = _ensure_cols(df, need)

    # numeric cleanup
    df["duration"] = _safe_num(df["duration"]).fillna(0.0)
    for c in METRIC_COLS.values():
        df[c] = _safe_num(df[c]).fillna(0.0)
    return df


# -----------------------------
# Aggregation
# -----------------------------
def aggregate_by_player(df: pd.DataFrame, event_filter: str) -> pd.DataFrame:
    if df.empty:
        return df

    if event_filter == EVENT_FULL:
        dff = df[df["event"].isin([EVENT_1H, EVENT_2H])].copy()
        event_label = EVENT_FULL
    else:
        dff = df[df["event"] == event_filter].copy()
        event_label = event_filter

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
    )
    agg["event_label"] = event_label

    # per minute columns (avoid divide by 0)
    mins = agg["duration"].replace(0, pd.NA)
    agg["td_min"] = (agg["total_distance"] / mins).astype(float)
    agg["run_min"] = (agg["running"] / mins).astype(float)
    agg["spr_min"] = (agg["sprint"] / mins).astype(float)
    agg["hs_min"] = (agg["high_sprint"] / mins).astype(float)

    # fill NaN (if duration=0)
    for c in ["td_min", "run_min", "spr_min", "hs_min"]:
        agg[c] = agg[c].fillna(0.0)

    return agg


def apply_global_order(agg: pd.DataFrame, sort_key_min: str, df_table: pd.DataFrame) -> pd.DataFrame:
    # sort_key_min is one of: TD, 14.4–19.7, 19.8–25.1, 25.2+
    key_map = {
        "TD": "td_min",
        "14.4–19.7": "run_min",
        "19.8–25.1": "spr_min",
        "25.2+": "hs_min",
    }
    k = key_map.get(sort_key_min, "td_min")
    order = agg.sort_values(k, ascending=False)["player_name"].tolist()
    out = df_table.copy()
    out["__order__"] = pd.Categorical(out["Player"], categories=order, ordered=True)
    out = out.sort_values("__order__").drop(columns=["__order__"])
    return out


def build_table_td(agg: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Player": agg["player_name"],
            "TD": agg["total_distance"],
            "/min": agg["td_min"],
        }
    )
    return df


def build_table_band(agg: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    per_min_map = {
        "running": "run_min",
        "sprint": "spr_min",
        "high_sprint": "hs_min",
    }
    df = pd.DataFrame(
        {
            "Player": agg["player_name"],
            label: agg[col],
            "/min": agg[per_min_map[col]],
        }
    )
    return df


def build_table_maxspeed(agg: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Player": agg["player_name"],
            "Max Speed": agg["max_speed"],
        }
    )
    return df


# -----------------------------
# Charts
# -----------------------------
def plot_total_distance_bar(agg: pd.DataFrame, title: str):
    d = agg.sort_values("total_distance", ascending=False)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=d["player_name"],
            y=d["total_distance"],
            marker=dict(color="#FF0033"),
            name="Total Distance",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=40, b=70),
        title=title,
        showlegend=False,
    )
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


def plot_sprint_vs_highsprint(agg: pd.DataFrame, title: str):
    d = agg.sort_values("sprint", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["player_name"], y=d["sprint"], name="sprint", marker=dict(color="rgba(255,0,51,0.75)")))
    fig.add_trace(go.Bar(x=d["player_name"], y=d["high_sprint"], name="high_sprint", marker=dict(color="#FF0033")))
    fig.update_layout(
        barmode="group",
        height=320,
        margin=dict(l=10, r=10, t=40, b=70),
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# UI pieces
# -----------------------------
def render_header(match_row: Dict[str, Any], left_logo: Optional[str], right_logo: Optional[str]):
    # Cleaner header: logos smaller + info centered
    c1, c2, c3 = st.columns([1.2, 2.2, 1.2])

    with c1:
        if left_logo:
            st.image(left_logo, use_container_width=True)

    with c2:
        md = match_row.get("match_date")
        opponent = match_row.get("opponent") or ""
        fixture = match_row.get("fixture") or ""
        season = match_row.get("season") or ""
        match_type = match_row.get("match_type") or ""
        home_away = match_row.get("home_away") or ""

        gf = match_row.get("goals_for")
        ga = match_row.get("goals_against")
        result = match_row.get("result") or ""

        score_txt = ""
        if gf is not None and ga is not None:
            score_txt = f"**Score:** {int(gf)} - {int(ga)}"

        st.markdown(
            f"""
            <div style="padding-top:10px">
              <div style="font-size:22px;font-weight:700;margin-bottom:4px">{MVV_TEAM_NAME} vs {opponent}</div>
              <div style="opacity:.85;margin-bottom:6px">{md} • {fixture} • {home_away} • {season} • {match_type}</div>
              <div style="font-size:16px;margin-top:2px">{score_txt} &nbsp; <span style="opacity:.85">{result}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        if right_logo:
            st.image(right_logo, use_container_width=True)


def render_tables_row(agg: pd.DataFrame, sort_key_min: str):
    if agg.empty:
        st.info("Geen data.")
        return

    td = apply_global_order(agg, sort_key_min, build_table_td(agg))
    run = apply_global_order(agg, sort_key_min, build_table_band(agg, "running", "14.4–19.7"))
    spr = apply_global_order(agg, sort_key_min, build_table_band(agg, "sprint", "19.8–25.1"))
    hs = apply_global_order(agg, sort_key_min, build_table_band(agg, "high_sprint", "25.2+"))
    ms = build_table_maxspeed(agg)

    # 1 row with 5 tables
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        sty = _style_by_percentiles(td, "TD").format({"TD": "{:,.0f}", "/min": "{:,.2f}"})
        st.dataframe(sty, use_container_width=True, hide_index=True, height=_df_height_for_all_rows(td))

    with c2:
        sty = _style_by_percentiles(run, "14.4–19.7").format({"14.4–19.7": "{:,.0f}", "/min": "{:,.2f}"})
        st.dataframe(sty, use_container_width=True, hide_index=True, height=_df_height_for_all_rows(run))

    with c3:
        sty = _style_by_percentiles(spr, "19.8–25.1").format({"19.8–25.1": "{:,.0f}", "/min": "{:,.2f}"})
        st.dataframe(sty, use_container_width=True, hide_index=True, height=_df_height_for_all_rows(spr))

    with c4:
        sty = _style_by_percentiles(hs, "25.2+").format({"25.2+": "{:,.0f}", "/min": "{:,.2f}"})
        st.dataframe(sty, use_container_width=True, hide_index=True, height=_df_height_for_all_rows(hs))

    with c5:
        sty = _style_by_percentiles(ms, "Max Speed").format({"Max Speed": "{:.2f}"})
        st.dataframe(sty, use_container_width=True, hide_index=True, height=_df_height_for_all_rows(ms))


# -----------------------------
# Main
# -----------------------------
def main():
    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    _ = get_profile(sb)  # keep consistent with other pages

    st.title("Match Reports")

    matches_df = fetch_matches(sb, limit=500)
    if matches_df.empty:
        st.info("Geen matches gevonden.")
        return

    # Build label list
    def _label(r: pd.Series) -> str:
        md = r.get("match_date")
        fixture = r.get("fixture") or ""
        ha = r.get("home_away") or ""
        opp = r.get("opponent") or ""
        season = r.get("season") or ""
        gf = r.get("goals_for")
        ga = r.get("goals_against")
        score = ""
        if pd.notna(gf) and pd.notna(ga):
            score = f"{int(gf)}-{int(ga)}"
        return f"{md} · {fixture} · {ha} · {opp} · {season} · {score}"

    matches_df = matches_df.copy()
    matches_df["label"] = matches_df.apply(_label, axis=1)

    sel_label = st.selectbox("Select match", options=matches_df["label"].tolist(), index=0)
    match_row = matches_df.loc[matches_df["label"] == sel_label].iloc[0].to_dict()
    match_id = int(match_row["match_id"])

    # Logos (smaller via layout columns)
    opponent = match_row.get("opponent") or ""
    opp_logo = _logo_path_for_team(opponent)

    # Header
    render_header(match_row, left_logo=MVV_LOGO, right_logo=opp_logo)

    st.markdown("")

    # Sort selection
    sort_label = st.selectbox("Sort tables on (per minute)", options=[x[0] for x in PER_MIN_OPTIONS], index=0)
    sort_key_min = dict(PER_MIN_OPTIONS)[sort_label]

    # Table filter (bolletjes)
    table_event = st.radio("Tables", EVENT_CHOICES, horizontal=True)

    # Fetch + aggregate
    df = fetch_match_events(sb, match_id=match_id)
    if df.empty:
        st.info("Geen match events gevonden (v_gps_match_events).")
        return

    agg = aggregate_by_player(df, table_event)
    if agg.empty:
        st.info("Geen data voor gekozen event.")
        return

    # Charts (use same event selection for charts as well)
    chart_title_suffix = table_event
    c1, c2 = st.columns(2)
    with c1:
        plot_total_distance_bar(agg, title=f"Total Distance ({chart_title_suffix})")
    with c2:
        plot_sprint_vs_highsprint(agg, title=f"Sprint vs High Sprint ({chart_title_suffix})")

    st.markdown("## Tables")
    render_tables_row(agg, sort_key_min=sort_key_min)


if __name__ == "__main__":
    main()
