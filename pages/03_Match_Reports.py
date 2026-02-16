# pages/03_Match_Reports.py
# ============================================================
# Update:
# - Tables: géén Duration (min) kolom
# - Nummering/Index links weg (hide_index=True)
# ============================================================

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from roles import get_sb, require_auth, get_profile


# -----------------------------
# Config
# -----------------------------
MVV_RED = "#FF0033"

EVENT_FULL = "Full match (First + Second)"
EVENT_FIRST = "First Half"
EVENT_SECOND = "Second Half"

TEAM_LOGO_DIR = Path("Assets/Afbeeldingen/Team_Logos")
MVV_LOGO_PATHS = [
    Path("Assets/Afbeeldingen/Team_Logos/MVV Maastricht.png"),
    Path("Assets/Afbeeldingen/Team_Logos/MVV.png"),
    Path("Assets/Afbeeldingen/Team_Logos/MVV_Maastricht.png"),
]

MATCH_EVENTS_VIEW = "v_gps_match_events"
MATCHES_TABLE = "matches"

SORT_OPTIONS = [
    ("Total Distance/min", "total_distance"),
    ("14.4–19.7 km/h/min", "running"),
    ("19.8–25.1 km/h/min", "sprint"),
    (">25.1 km/h/min", "high_sprint"),
]

HALF_OPTIONS = [
    ("Full match", EVENT_FULL),
    ("First half", EVENT_FIRST),
    ("Second half", EVENT_SECOND),
]


# -----------------------------
# Helpers
# -----------------------------
def _df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _coerce_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _read_logo_path(team_name: str) -> Optional[Path]:
    if not team_name:
        return None
    p = TEAM_LOGO_DIR / f"{team_name}.png"
    if p.exists():
        return p
    if TEAM_LOGO_DIR.exists():
        low = team_name.lower()
        for fp in TEAM_LOGO_DIR.glob("*.png"):
            if fp.stem.lower() == low:
                return fp
    return None


def _read_mvv_logo() -> Optional[Path]:
    for p in MVV_LOGO_PATHS:
        if p.exists():
            return p
    if TEAM_LOGO_DIR.exists():
        for fp in TEAM_LOGO_DIR.glob("*.png"):
            if "mvv" in fp.stem.lower():
                return fp
    return None


def _style_by_percentiles(df: pd.DataFrame, value_col: str) -> "pd.io.formats.style.Styler":
    if df.empty:
        return df.style

    vals = pd.to_numeric(df[value_col], errors="coerce")
    avals = vals.abs()
    if not np.isfinite(avals).any():
        return df.style

    p20, p40, p60, p80 = np.nanpercentile(avals, [20, 40, 60, 80])

    def cell_style(v):
        try:
            x = abs(float(v))
        except Exception:
            return ""
        if x <= p20:
            return "background-color: rgba(255,0,51,0.35);"
        if x <= p40:
            return "background-color: rgba(255,0,51,0.20);"
        if x <= p60:
            return "background-color: rgba(255,165,0,0.22);"
        if x <= p80:
            return "background-color: rgba(0,200,0,0.20);"
        return "background-color: rgba(0,200,0,0.35);"

    return df.style.applymap(cell_style, subset=[value_col])


# -----------------------------
# Supabase fetch (geen cache met sb-param)
# -----------------------------
def fetch_matches(sb, limit: int = 500) -> pd.DataFrame:
    rows = (
        sb.table(MATCHES_TABLE)
        .select("match_id,match_date,fixture,home_away,opponent,match_type,season,result,goals_for,goals_against")
        .order("match_date", desc=True)
        .limit(limit)
        .execute()
        .data
        or []
    )
    dfm = _df(rows)
    if dfm.empty:
        return dfm
    dfm["match_date"] = _coerce_date(dfm["match_date"])
    return dfm


def fetch_match_events(sb, match_id: int) -> pd.DataFrame:
    rows = (
        sb.table(MATCH_EVENTS_VIEW)
        .select("match_id,player_name,datum,event,duration,total_distance,running,sprint,high_sprint,max_speed")
        .eq("match_id", match_id)
        .in_("event", [EVENT_FIRST, EVENT_SECOND, "First Half", "Second Half"])
        .execute()
        .data
        or []
    )
    df = _df(rows)
    if df.empty:
        return df

    df["datum"] = _coerce_date(df["datum"])
    for c in ["duration", "total_distance", "running", "sprint", "high_sprint", "max_speed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["event"] = df["event"].replace({"First Half": EVENT_FIRST, "Second Half": EVENT_SECOND})
    return df


# -----------------------------
# Aggregation
# -----------------------------
def aggregate_block(df_events: pd.DataFrame, block: str) -> pd.DataFrame:
    if df_events.empty:
        return df_events

    df = df_events.copy()
    if block == EVENT_FIRST:
        df = df[df["event"] == EVENT_FIRST]
    elif block == EVENT_SECOND:
        df = df[df["event"] == EVENT_SECOND]
    else:
        df = df[df["event"].isin([EVENT_FIRST, EVENT_SECOND])]

    if df.empty:
        return df

    agg = (
        df.groupby("player_name", as_index=False)
        .agg(
            duration_min=("duration", "sum"),
            total_distance=("total_distance", "sum"),
            running=("running", "sum"),
            sprint=("sprint", "sum"),
            high_sprint=("high_sprint", "sum"),
            max_speed=("max_speed", "max"),
        )
        .copy()
    )

    dur = agg["duration_min"].replace(0, np.nan)
    agg["total_distance_min"] = agg["total_distance"] / dur
    agg["running_min"] = agg["running"] / dur
    agg["sprint_min"] = agg["sprint"] / dur
    agg["high_sprint_min"] = agg["high_sprint"] / dur
    return agg


# -----------------------------
# Charts
# -----------------------------
def plot_total_distance_bar(agg: pd.DataFrame, title_suffix: str):
    if agg.empty:
        st.info("Geen data voor grafiek.")
        return
    df = agg.sort_values("total_distance", ascending=False).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["player_name"], y=df["total_distance"], marker=dict(color=MVV_RED)))
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"Total Distance ({title_suffix})",
        showlegend=False,
    )
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


def plot_sprint_vs_highsprint_bar(agg: pd.DataFrame, title_suffix: str):
    if agg.empty:
        st.info("Geen data voor grafiek.")
        return
    df = agg.sort_values("sprint", ascending=False).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["player_name"], y=df["sprint"], name="sprint", marker=dict(color="rgba(255,0,51,0.55)")))
    fig.add_trace(go.Bar(x=df["player_name"], y=df["high_sprint"], name="high_sprint", marker=dict(color=MVV_RED)))
    fig.update_layout(
        barmode="group",
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"Sprint vs High Sprint ({title_suffix})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Tables (5 losse) - zonder Duration
# -----------------------------
def build_table_td(agg: pd.DataFrame) -> pd.DataFrame:
    out = agg[["player_name", "total_distance", "total_distance_min"]].copy()
    out.columns = ["Player", "TD", "/min"]
    return out


def build_table_band(agg: pd.DataFrame, col: str, label_short: str) -> pd.DataFrame:
    out = agg[["player_name", col, f"{col}_min"]].copy()
    out.columns = ["Player", label_short, "/min"]
    return out


def build_table_maxspeed(agg: pd.DataFrame) -> pd.DataFrame:
    out = agg[["player_name", "max_speed"]].copy()
    out.columns = ["Player", "Max Speed"]
    return out.sort_values("Max Speed", ascending=False)


def apply_global_order(agg: pd.DataFrame, sort_key_min: str, df_table: pd.DataFrame) -> pd.DataFrame:
    order = (
        agg.assign(_k=agg[f"{sort_key_min}_min"])
        .sort_values("_k", ascending=False)["player_name"]
        .tolist()
    )
    d = df_table.copy()
    d["__ord"] = d["Player"].apply(lambda x: order.index(x) if x in order else 10_000)
    d = d.sort_values("__ord").drop(columns="__ord")
    return d


def render_tables_row(agg: pd.DataFrame, sort_key_min: str):
    if agg.empty:
        st.info("Geen data.")
        return

    td = apply_global_order(agg, sort_key_min, build_table_td(agg))
    run = apply_global_order(agg, sort_key_min, build_table_band(agg, "running", "14.4–19.7"))
    spr = apply_global_order(agg, sort_key_min, build_table_band(agg, "sprint", "19.8–25.1"))
    hs = apply_global_order(agg, sort_key_min, build_table_band(agg, "high_sprint", "25.2+"))
    ms = build_table_maxspeed(agg)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        sty = _style_by_percentiles(td, "TD").format({"TD": "{:,.0f}", "/min": "{:,.2f}"})
        st.dataframe(sty, use_container_width=True, height=380, hide_index=True)

    with c2:
        sty = _style_by_percentiles(run, "14.4–19.7").format({"14.4–19.7": "{:,.0f}", "/min": "{:,.2f}"})
        st.dataframe(sty, use_container_width=True, height=380, hide_index=True)

    with c3:
        sty = _style_by_percentiles(spr, "19.8–25.1").format({"19.8–25.1": "{:,.0f}", "/min": "{:,.2f}"})
        st.dataframe(sty, use_container_width=True, height=380, hide_index=True)

    with c4:
        sty = _style_by_percentiles(hs, "25.2+").format({"25.2+": "{:,.0f}", "/min": "{:,.2f}"})
        st.dataframe(sty, use_container_width=True, height=380, hide_index=True)

    with c5:
        sty = _style_by_percentiles(ms, "Max Speed").format({"Max Speed": "{:.2f}"})
        st.dataframe(sty, use_container_width=True, height=380, hide_index=True)


# -----------------------------
# UI
# -----------------------------
def match_label(r: pd.Series) -> str:
    d = r.get("match_date")
    opp = r.get("opponent") or ""
    ha = r.get("home_away") or ""
    fixture = r.get("fixture") or ""
    season = r.get("season") or ""
    res = r.get("result") or ""
    gf = r.get("goals_for")
    ga = r.get("goals_against")
    score = ""
    if gf is not None and ga is not None and str(gf) != "nan" and str(ga) != "nan":
        score = f"{int(gf)}-{int(ga)}"
    bits = [str(d), fixture, ha, opp, season, res, score]
    return " • ".join([b for b in bits if b])


def main():
    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    _ = get_profile(sb)

    st.title("Match Reports")

    matches_df = fetch_matches(sb, limit=500)
    if matches_df.empty:
        st.info("Geen matches gevonden.")
        return

    options = matches_df.index.tolist()
    sel_idx = st.selectbox(
        "Select match",
        options=options,
        index=0,
        format_func=lambda i: match_label(matches_df.loc[i]),
        key="mr_match_select",
    )
    match_row = matches_df.loc[sel_idx]
    match_id = int(match_row["match_id"])
    match_date = match_row["match_date"]
    opponent = str(match_row.get("opponent") or "")

    lh, mh, rh = st.columns([1, 2, 1], vertical_alignment="center")
    with lh:
        mvv_logo = _read_mvv_logo()
        if mvv_logo:
            st.image(str(mvv_logo), use_container_width=True)
    with mh:
        st.subheader(f"{match_date}  •  {opponent}")
    with rh:
        opp_logo = _read_logo_path(opponent)
        if opp_logo:
            st.image(str(opp_logo), use_container_width=True)

    df_events = fetch_match_events(sb, match_id)
    if df_events.empty:
        st.warning("Geen match events gevonden in v_gps_match_events voor deze match_id.")
        return

    sort_label = st.selectbox("Sort tables on (per minute)", options=[x[0] for x in SORT_OPTIONS], index=0)
    sort_key_min = dict(SORT_OPTIONS)[sort_label]

    half_label = st.radio(
        "Tables",
        options=[x[0] for x in HALF_OPTIONS],
        index=0,
        horizontal=True,
        key="mr_half_radio",
    )
    half_key = dict(HALF_OPTIONS)[half_label]

    agg = aggregate_block(df_events, half_key)
    title_suffix = half_label

    c1, c2 = st.columns(2)
    with c1:
        plot_total_distance_bar(agg, title_suffix)
    with c2:
        plot_sprint_vs_highsprint_bar(agg, title_suffix)

    st.markdown("## Tables")
    render_tables_row(agg, sort_key_min=sort_key_min)


if __name__ == "__main__":
    main()
