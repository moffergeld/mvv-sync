# pages/03_Match_Reports.py
# ============================================================
# Match Reports (Streamlit)
# - Data uit:
#     public.matches
#     public.v_gps_match_events   (met o.a. match_id, event = 'First Half'/'Second Half')
#
# Layout (zoals je PDF):
# 1) Boven: sort dropdown (per minute)
# 2) Daaronder: 2 grafieken naast elkaar (Plotly)
# 3) Daaronder: TABLES
#    - Full match: 5 losse tabellen naast elkaar (TD, 14.4–19.7, 19.8–25.1, >25.1, Max Speed)
#    - First half: 5 losse tabellen naast elkaar
#    - Second half: 5 losse tabellen naast elkaar
#
# Kleur op percentielen van ABSOLUTE waardes (per metric-kolom)
# Sorteren op /min basis
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
MVV_RED = "#FF0033"

EVENT_FULL = "Full match (First + Second)"
EVENT_FIRST = "First Half"
EVENT_SECOND = "Second Half"

TEAM_LOGO_DIR = Path("Assets/Afbeeldingen/Team_Logos")  # repo-pad
MVV_LOGO_PATHS = [
    Path("Assets/Afbeeldingen/Team_Logos/MVV Maastricht.png"),
    Path("Assets/Afbeeldingen/Team_Logos/MVV.png"),
    Path("Assets/Afbeeldingen/Team_Logos/MVV_Maastricht.png"),
]

# v_gps_match_events kolommen (zoals jij beschreef; match_id zit in de view)
MATCH_EVENTS_VIEW = "v_gps_match_events"
MATCHES_TABLE = "matches"

# Metrics (labels voor UI)
METRICS = [
    ("Total Distance (m)", "total_distance"),
    ("14.4–19.7 km/h", "running"),
    ("19.8–25.1 km/h", "sprint"),
    (">25.1 km/h", "high_sprint"),
    ("Max Speed (km/u)", "max_speed"),
]

SORT_OPTIONS = [
    ("Total Distance/min", "total_distance"),
    ("14.4–19.7 km/h/min", "running"),
    ("19.8–25.1 km/h/min", "sprint"),
    (">25.1 km/h/min", "high_sprint"),
]


# -----------------------------
# Helpers
# -----------------------------
def _df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _coerce_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _fmt_num(x: Any, decimals: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        x = float(x)
    except Exception:
        return str(x)
    if decimals == 0:
        return f"{x:,.0f}".replace(",", " ")
    return f"{x:,.{decimals}f}".replace(",", " ")


def _add_zone_background(fig: go.Figure, y_min: float = 0, y_max: float = 10):
    zones = [
        (0, 4, "rgba(0, 200, 0, 0.12)"),
        (5, 7, "rgba(255, 165, 0, 0.14)"),
        (8, 10, "rgba(255, 0, 0, 0.14)"),
    ]
    for y0, y1, color in zones:
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="y",
            x0=0,
            x1=1,
            y0=y0,
            y1=y1,
            fillcolor=color,
            line=dict(width=0),
            layer="below",
        )
    fig.update_yaxes(range=[y_min, y_max], tick0=y_min, dtick=1)


def _read_logo_path(team_name: str) -> Optional[Path]:
    if not team_name:
        return None
    # exacte match
    p = TEAM_LOGO_DIR / f"{team_name}.png"
    if p.exists():
        return p
    # fallback: zoek case-insensitive op bestandsnaam
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
    # fallback: zoek op "mvv"
    if TEAM_LOGO_DIR.exists():
        for fp in TEAM_LOGO_DIR.glob("*.png"):
            if "mvv" in fp.stem.lower():
                return fp
    return None


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
    # v_gps_match_events moet match_id bevatten
    rows = (
        sb.table(MATCH_EVENTS_VIEW)
        .select(
            "match_id,player_id,player_name,datum,event,duration,total_distance,running,sprint,high_sprint,max_speed"
        )
        .eq("match_id", match_id)
        .in_("event", [EVENT_FIRST, EVENT_SECOND, "First Half", "Second Half"])  # tolerant
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

    # normaliseer event labels exact
    df["event"] = df["event"].replace({"First Half": EVENT_FIRST, "Second Half": EVENT_SECOND})
    return df


# -----------------------------
# Build tables (5 losse)
# -----------------------------
def _aggregate_for_block(df_events: pd.DataFrame, which: str) -> pd.DataFrame:
    """
    which in {EVENT_FULL, EVENT_FIRST, EVENT_SECOND}
    Output per player: duration (min), metrics sums, max_speed max
    """
    if df_events.empty:
        return df_events

    df = df_events.copy()

    if which == EVENT_FIRST:
        df = df[df["event"] == EVENT_FIRST]
    elif which == EVENT_SECOND:
        df = df[df["event"] == EVENT_SECOND]
    else:
        # full match: first+second
        df = df[df["event"].isin([EVENT_FIRST, EVENT_SECOND])]

    if df.empty:
        return df

    # per player aggregatie
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

    # per minute
    dur = agg["duration_min"].replace(0, np.nan)
    agg["total_distance_min"] = agg["total_distance"] / dur
    agg["running_min"] = agg["running"] / dur
    agg["sprint_min"] = agg["sprint"] / dur
    agg["high_sprint_min"] = agg["high_sprint"] / dur

    return agg


def _style_by_percentiles(df: pd.DataFrame, value_col: str) -> "pd.io.formats.style.Styler":
    """
    Kleur op percentielen van ABS(value_col).
    """
    if df.empty:
        return df.style

    vals = pd.to_numeric(df[value_col], errors="coerce")
    avals = vals.abs()
    p20, p40, p60, p80 = np.nanpercentile(avals, [20, 40, 60, 80]) if np.isfinite(avals).any() else (0, 0, 0, 0)

    def cell_style(v):
        try:
            x = float(v)
        except Exception:
            return ""
        x = abs(x)
        # 5 bins (rood -> groen)
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


def build_metric_table_block(
    agg: pd.DataFrame,
    metric_key: str,
    metric_label_short: str,
    sort_key_min: str,
) -> pd.DataFrame:
    """
    Return tabel met:
      Player | ABS | /min
    """
    if agg.empty:
        return agg

    metric_abs = metric_key
    if metric_key == "max_speed":
        # max_speed heeft geen /min
        out = agg[["player_name", "max_speed"]].copy()
        out = out.rename(columns={"player_name": "Player", "max_speed": "Max Speed"})
        out["Max Speed"] = pd.to_numeric(out["Max Speed"], errors="coerce")
        out = out.sort_values("Max Speed", ascending=False)
        return out

    min_col = f"{metric_key}_min"
    out = agg[["player_name", metric_abs, min_col, "duration_min"]].copy()
    out = out.rename(
        columns={
            "player_name": "Player",
            metric_abs: metric_label_short,
            min_col: "/min",
            "duration_min": "Duration (min)",
        }
    )

    # sort op gekozen /min metric
    if sort_key_min == metric_key:
        out = out.sort_values("/min", ascending=False)
    else:
        # default: sort op eigen /min
        out = out.sort_values("/min", ascending=False)

    return out


# -----------------------------
# Charts (Plotly, zoals player page)
# -----------------------------
def plot_total_distance_bar(agg_full: pd.DataFrame):
    if agg_full.empty:
        st.info("Geen data voor grafiek.")
        return

    df = agg_full.sort_values("total_distance", ascending=False).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["player_name"],
            y=df["total_distance"],
            marker=dict(color=MVV_RED),
            name="Total Distance",
        )
    )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Total Distance (Full match)",
        showlegend=False,
    )
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


def plot_sprint_vs_highsprint_bar(agg_full: pd.DataFrame):
    if agg_full.empty:
        st.info("Geen data voor grafiek.")
        return

    df = agg_full.sort_values("sprint", ascending=False).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["player_name"], y=df["sprint"], name="sprint", marker=dict(color="rgba(255,0,51,0.55)")))
    fig.add_trace(
        go.Bar(x=df["player_name"], y=df["high_sprint"], name="high_sprint", marker=dict(color=MVV_RED))
    )

    fig.update_layout(
        barmode="group",
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Sprint vs High Sprint (Full match)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# UI
# -----------------------------
def _match_label(r: pd.Series) -> str:
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

    profile = get_profile(sb)

    st.title("Match Reports")

    matches_df = fetch_matches(sb, limit=500)
    if matches_df.empty:
        st.info("Geen matches gevonden in public.matches.")
        return

    # Match select
    idx = 0
    options = matches_df.index.tolist()
    labels = {i: _match_label(matches_df.loc[i]) for i in options}
    sel_idx = st.selectbox(
        "Select match",
        options=options,
        index=idx,
        format_func=lambda i: labels.get(i, str(i)),
        key="mr_match_select",
    )
    match_row = matches_df.loc[sel_idx]
    match_id = int(match_row["match_id"])
    match_date = match_row["match_date"]
    opponent = str(match_row.get("opponent") or "")

    # Header with logos (optioneel)
    left_h, mid_h, right_h = st.columns([1, 2, 1], vertical_alignment="center")
    with left_h:
        mvv_logo = _read_mvv_logo()
        if mvv_logo:
            st.image(str(mvv_logo), use_container_width=True)
    with mid_h:
        st.subheader(f"{match_date}  •  {opponent}")
    with right_h:
        opp_logo = _read_logo_path(opponent)
        if opp_logo:
            st.image(str(opp_logo), use_container_width=True)

    # Fetch events
    df_events = fetch_match_events(sb, match_id=match_id)
    if df_events.empty:
        st.warning("Geen match events gevonden in v_gps_match_events voor deze match_id.")
        return

    # Sort dropdown (per minute)
    sort_label = st.selectbox("Sort tables on (per minute)", options=[x[0] for x in SORT_OPTIONS], index=0)
    sort_key_min = dict(SORT_OPTIONS)[sort_label]

    # Aggregations
    agg_full = _aggregate_for_block(df_events, EVENT_FULL)
    agg_first = _aggregate_for_block(df_events, EVENT_FIRST)
    agg_second = _aggregate_for_block(df_events, EVENT_SECOND)

    # Charts row
    c1, c2 = st.columns(2)
    with c1:
        plot_total_distance_bar(agg_full)
    with c2:
        plot_sprint_vs_highsprint_bar(agg_full)

    st.markdown("## Tables")

    def render_block(title: str, agg: pd.DataFrame):
        st.markdown(f"### {title}")

        if agg.empty:
            st.info("Geen data.")
            return

        # 5 tables naast elkaar (zoals jouw PDF)
        t1, t2, t3, t4, t5 = st.columns(5)

        # TD
        td = build_metric_table_block(
            agg,
            metric_key="total_distance",
            metric_label_short="TD",
            sort_key_min=sort_key_min,
        )
        # Running band
        run = build_metric_table_block(
            agg,
            metric_key="running",
            metric_label_short="14.4–19.7",
            sort_key_min=sort_key_min,
        )
        # Sprint band
        spr = build_metric_table_block(
            agg,
            metric_key="sprint",
            metric_label_short="19.8–25.1",
            sort_key_min=sort_key_min,
        )
        # High sprint band
        hs = build_metric_table_block(
            agg,
            metric_key="high_sprint",
            metric_label_short="25.2+",
            sort_key_min=sort_key_min,
        )
        # Max speed (geen /min)
        ms = build_metric_table_block(
            agg,
            metric_key="max_speed",
            metric_label_short="Max Speed",
            sort_key_min=sort_key_min,
        )

        # Display helpers (format)
        def format_td(df0: pd.DataFrame) -> pd.DataFrame:
            d = df0.copy()
            if "Duration (min)" in d.columns:
                d["Duration (min)"] = pd.to_numeric(d["Duration (min)"], errors="coerce")
            if "TD" in d.columns:
                d["TD"] = pd.to_numeric(d["TD"], errors="coerce")
            if "/min" in d.columns:
                d["/min"] = pd.to_numeric(d["/min"], errors="coerce")
            return d

        def format_band(df0: pd.DataFrame, colname: str) -> pd.DataFrame:
            d = df0.copy()
            if "Duration (min)" in d.columns:
                d["Duration (min)"] = pd.to_numeric(d["Duration (min)"], errors="coerce")
            if colname in d.columns:
                d[colname] = pd.to_numeric(d[colname], errors="coerce")
            if "/min" in d.columns:
                d["/min"] = pd.to_numeric(d["/min"], errors="coerce")
            return d

        def format_ms(df0: pd.DataFrame) -> pd.DataFrame:
            d = df0.copy()
            if "Max Speed" in d.columns:
                d["Max Speed"] = pd.to_numeric(d["Max Speed"], errors="coerce")
            return d

        td = format_td(td)
        run = format_band(run, "14.4–19.7")
        spr = format_band(spr, "19.8–25.1")
        hs = format_band(hs, "25.2+")
        ms = format_ms(ms)

        # Sort logic: sort op gekozen /min (dus in alle tables dezelfde ranking op basis van gekozen metric/min)
        # Hiervoor maken we een ranking lijst uit agg op sort_key_min:
        if sort_key_min in agg.columns:
            rank = (
                agg.assign(_k=agg[f"{sort_key_min}_min"])
                .sort_values("_k", ascending=False)[["player_name"]]
                .rename(columns={"player_name": "Player"})
            )
            order = rank["Player"].tolist()

            def apply_order(df0: pd.DataFrame) -> pd.DataFrame:
                if df0.empty or "Player" not in df0.columns:
                    return df0
                df0 = df0.copy()
                df0["__ord"] = df0["Player"].apply(lambda x: order.index(x) if x in order else 10_000)
                df0 = df0.sort_values("__ord").drop(columns=["__ord"])
                return df0

            td = apply_order(td)
            run = apply_order(run)
            spr = apply_order(spr)
            hs = apply_order(hs)
            ms = apply_order(ms)

        # Show (kleur op ABS metric kolom)
        with t1:
            show = td[["Player", "Duration (min)", "TD", "/min"]].copy()
            sty = _style_by_percentiles(show, "TD").format(
                {"Duration (min)": "{:.2f}", "TD": "{:,.0f}", "/min": "{:,.2f}"}
            )
            st.dataframe(sty, use_container_width=True, height=380)

        with t2:
            show = run[["Player", "Duration (min)", "14.4–19.7", "/min"]].copy()
            sty = _style_by_percentiles(show, "14.4–19.7").format(
                {"Duration (min)": "{:.2f}", "14.4–19.7": "{:,.0f}", "/min": "{:,.2f}"}
            )
            st.dataframe(sty, use_container_width=True, height=380)

        with t3:
            show = spr[["Player", "Duration (min)", "19.8–25.1", "/min"]].copy()
            sty = _style_by_percentiles(show, "19.8–25.1").format(
                {"Duration (min)": "{:.2f}", "19.8–25.1": "{:,.0f}", "/min": "{:,.2f}"}
            )
            st.dataframe(sty, use_container_width=True, height=380)

        with t4:
            show = hs[["Player", "Duration (min)", "25.2+", "/min"]].copy()
            sty = _style_by_percentiles(show, "25.2+").format(
                {"Duration (min)": "{:.2f}", "25.2+": "{:,.0f}", "/min": "{:,.2f}"}
            )
            st.dataframe(sty, use_container_width=True, height=380)

        with t5:
            show = ms[["Player", "Max Speed"]].copy()
            sty = _style_by_percentiles(show, "Max Speed").format({"Max Speed": "{:.2f}"})
            st.dataframe(sty, use_container_width=True, height=380)

    # Render blocks (zoals jouw PDF: Full, First, Second)
    render_block("Full match (First + Second)", agg_full)
    st.divider()
    render_block("First half", agg_first)
    st.divider()
    render_block("Second half", agg_second)


if __name__ == "__main__":
    main()
