# pages/03_Match_Reports.py
# ============================================================
# FIXES:
# - Absolute waardes: 0 decimalen (ook in de dataframe-weergave)
# - /min waardes: 2 decimalen (ook in de dataframe-weergave)
# - Center alignment voor ALLE numerieke kolommen
# - Player kolom NIET gecentreerd (links)
# ============================================================

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from roles import get_sb, require_auth, get_profile

MVV_RED = "#FF0033"

TEAM_LOGOS_DIR = os.path.join("Assets", "Afbeeldingen", "Team_Logos")
MVV_LOGO_CANDIDATES = [
    os.path.join(TEAM_LOGOS_DIR, "MVV Maastricht.png"),
    os.path.join("Assets", "Afbeeldingen", "Script", "MVV.png"),
    os.path.join("Assets", "Afbeeldingen", "Script", "MVV_logo.png"),
]

MATCH_EVENTS_VIEW = "v_gps_match_events"
MATCHES_TABLE = "matches"

SORT_OPTIONS = {
    "Total Distance/min": ("total_distance", "per_min"),
    "14.4–19.7/min": ("running", "per_min"),
    "19.8–25.1/min": ("sprint", "per_min"),
    "25.2+/min": ("high_sprint", "per_min"),
    "Max Speed": ("max_speed", "abs"),
}

# --- display formats ---
ABS_FMT_0 = "{:,.0f}"
PMIN_FMT_2 = "{:,.2f}"


def _df(rows) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def _find_logo_path(team_name: str) -> Optional[str]:
    if not team_name:
        return None
    candidates = [
        os.path.join(TEAM_LOGOS_DIR, f"{team_name}.png"),
        os.path.join(TEAM_LOGOS_DIR, f"{team_name}.PNG"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    try:
        files = os.listdir(TEAM_LOGOS_DIR)
        lookup = {f.lower(): f for f in files}
        key = f"{team_name}.png".lower()
        if key in lookup:
            p = os.path.join(TEAM_LOGOS_DIR, lookup[key])
            if os.path.exists(p):
                return p
    except Exception:
        pass
    return None


def _find_mvv_logo() -> Optional[str]:
    for p in MVV_LOGO_CANDIDATES:
        if os.path.exists(p):
            return p
    p = os.path.join(TEAM_LOGOS_DIR, "MVV.png")
    return p if os.path.exists(p) else None


def _percentile_color(v: float, p33: float, p66: float) -> str:
    if pd.isna(v):
        return ""
    if v <= p33:
        return "background-color: rgba(255,0,0,0.25);"
    if v <= p66:
        return "background-color: rgba(255,165,0,0.22);"
    return "background-color: rgba(0,200,0,0.22);"


def _style_metric_cols(df: pd.DataFrame, metric_cols: list[str]) -> "pd.io.formats.style.Styler":
    dff = df.copy()

    pvals = {}
    for c in metric_cols:
        s = pd.to_numeric(dff[c], errors="coerce")
        p33 = float(s.quantile(0.33)) if s.notna().any() else 0.0
        p66 = float(s.quantile(0.66)) if s.notna().any() else 0.0
        pvals[c] = (p33, p66)

    def _apply_row(row):
        out = [""] * len(row)
        for i, col in enumerate(row.index):
            if col in pvals:
                p33, p66 = pvals[col]
                out[i] = _percentile_color(row[col], p33, p66)
        return out

    return dff.style.apply(_apply_row, axis=1)


def _table_height_for_n(n_rows: int) -> int:
    return int(35 * (n_rows + 1) + 10)


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
    dfm["match_date"] = pd.to_datetime(dfm["match_date"], errors="coerce").dt.date
    return dfm


def fetch_match_events(sb, match_id: int) -> pd.DataFrame:
    rows = (
        sb.table(MATCH_EVENTS_VIEW)
        .select(
            "match_id,player_id,player_name,event,duration,total_distance,running,sprint,high_sprint,max_speed"
        )
        .eq("match_id", match_id)
        .execute()
        .data
        or []
    )
    dfe = _df(rows)
    if dfe.empty:
        return dfe

    num_cols = ["duration", "total_distance", "running", "sprint", "high_sprint", "max_speed"]
    for c in num_cols:
        if c in dfe.columns:
            dfe[c] = pd.to_numeric(dfe[c], errors="coerce")

    dfe["player_name"] = dfe["player_name"].fillna("").astype(str)
    dfe["event"] = dfe["event"].fillna("").astype(str)
    return dfe


def _segment_filter(dfe: pd.DataFrame, segment: str) -> pd.DataFrame:
    if dfe.empty:
        return dfe
    if segment == "Full match":
        return dfe.copy()
    if segment == "First half":
        return dfe[dfe["event"].str.contains("First", case=False, na=False)].copy()
    if segment == "Second half":
        return dfe[dfe["event"].str.contains("Second", case=False, na=False)].copy()
    return dfe.copy()


def agg_for_charts_and_tables(dfe_seg: pd.DataFrame) -> pd.DataFrame:
    if dfe_seg.empty:
        return dfe_seg

    g = dfe_seg.groupby("player_name", as_index=False).agg(
        duration=("duration", "sum"),
        total_distance=("total_distance", "sum"),
        running=("running", "sum"),
        sprint=("sprint", "sum"),
        high_sprint=("high_sprint", "sum"),
        max_speed=("max_speed", "max"),
    )

    dur = pd.to_numeric(g["duration"], errors="coerce").replace(0, pd.NA)

    # /min (2 dec)
    g["total_distance_min"] = (g["total_distance"] / dur).round(2)
    g["running_min"] = (g["running"] / dur).round(2)
    g["sprint_min"] = (g["sprint"] / dur).round(2)
    g["high_sprint_min"] = (g["high_sprint"] / dur).round(2)

    # absolute (0 dec)
    for c in ["duration", "total_distance", "running", "sprint", "high_sprint"]:
        g[c] = pd.to_numeric(g[c], errors="coerce").round(0)

    g["max_speed"] = pd.to_numeric(g["max_speed"], errors="coerce").round(0)

    return g


def sort_players(df_agg: pd.DataFrame, sort_choice: str) -> pd.DataFrame:
    if df_agg.empty:
        return df_agg
    col_key, mode = SORT_OPTIONS[sort_choice]
    sort_col = f"{col_key}_min" if mode == "per_min" else col_key
    if sort_col not in df_agg.columns:
        return df_agg.sort_values("player_name")
    return df_agg.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)


def plot_total_distance_bar(df_sorted: pd.DataFrame, segment: str):
    if df_sorted.empty:
        st.info("Geen data voor grafiek.")
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_sorted["player_name"], y=df_sorted["total_distance"], marker_color=MVV_RED))
    fig.update_layout(title=f"Total Distance ({segment})", height=340, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


def plot_sprint_vs_high(df_sorted: pd.DataFrame, segment: str):
    if df_sorted.empty:
        st.info("Geen data voor grafiek.")
        return
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=df_sorted["player_name"], y=df_sorted["sprint"], marker_color="rgba(255,0,51,0.55)", name="sprint")
    )
    fig.add_trace(go.Bar(x=df_sorted["player_name"], y=df_sorted["high_sprint"], marker_color=MVV_RED, name="high_sprint"))
    fig.update_layout(
        title=f"Sprint vs High Sprint ({segment})",
        barmode="group",
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


def _metric_header(abs_col: str) -> str:
    if abs_col == "total_distance":
        return "TD"
    if abs_col == "running":
        return "14.4–19.7"
    if abs_col == "sprint":
        return "19.8–25.1"
    if abs_col == "high_sprint":
        return "25.2+"
    return abs_col


def _format_and_align_table(sty: "pd.io.formats.style.Styler") -> "pd.io.formats.style.Styler":
    """
    - 0 dec voor absolute kolom(len)
    - 2 dec voor /min
    - numeriek gecentreerd
    - Player links
    """
    df = sty.data
    cols = list(df.columns)

    # detecteer kolommen
    player_col = "Player" if "Player" in cols else cols[0]
    min_col = "/min" if "/min" in cols else None

    # format
    fmt_map = {}
    for c in cols:
        if c == player_col:
            continue
        if c == min_col:
            fmt_map[c] = PMIN_FMT_2
        else:
            fmt_map[c] = ABS_FMT_0

    sty = sty.format(fmt_map)

    # alignment
    sty = sty.set_properties(**{"text-align": "left"}, subset=[player_col])
    num_cols = [c for c in cols if c != player_col]
    if num_cols:
        sty = sty.set_properties(**{"text-align": "center"}, subset=num_cols)

    return sty


def build_table_df(df_sorted: pd.DataFrame, metric_key: str) -> Tuple[pd.DataFrame, list[str]]:
    if df_sorted.empty:
        return pd.DataFrame(), []

    if metric_key == "max_speed":
        out = df_sorted[["player_name", "max_speed"]].copy()
        out.columns = ["Player", "Max Speed"]
        return out, ["Max Speed"]

    abs_col = metric_key
    per_min_col = f"{metric_key}_min"
    out = df_sorted[["player_name", abs_col, per_min_col]].copy()
    out.columns = ["Player", _metric_header(abs_col), "/min"]
    return out, [_metric_header(abs_col)]


def render_tables_row(df_sorted: pd.DataFrame):
    if df_sorted.empty:
        st.info("Geen data voor tabellen.")
        return

    n = int(df_sorted.shape[0])
    h = _table_height_for_n(n)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        tdf, mcols = build_table_df(df_sorted, "total_distance")
        sty = _style_metric_cols(tdf, mcols)
        sty = _format_and_align_table(sty)
        st.dataframe(sty, use_container_width=True, hide_index=True, height=h)

    with c2:
        tdf, mcols = build_table_df(df_sorted, "running")
        sty = _style_metric_cols(tdf, mcols)
        sty = _format_and_align_table(sty)
        st.dataframe(sty, use_container_width=True, hide_index=True, height=h)

    with c3:
        tdf, mcols = build_table_df(df_sorted, "sprint")
        sty = _style_metric_cols(tdf, mcols)
        sty = _format_and_align_table(sty)
        st.dataframe(sty, use_container_width=True, hide_index=True, height=h)

    with c4:
        tdf, mcols = build_table_df(df_sorted, "high_sprint")
        sty = _style_metric_cols(tdf, mcols)
        sty = _format_and_align_table(sty)
        st.dataframe(sty, use_container_width=True, hide_index=True, height=h)

    with c5:
        tdf, mcols = build_table_df(df_sorted, "max_speed")
        sty = _style_metric_cols(tdf, mcols)
        sty = _format_and_align_table(sty)
        st.dataframe(sty, use_container_width=True, hide_index=True, height=h)


def render_header(match_row: Dict[str, Any]):
    mvv_logo = _find_mvv_logo()
    opp = (match_row.get("opponent") or "").strip()
    opp_logo = _find_logo_path(opp)

    match_date = match_row.get("match_date")
    fixture = (match_row.get("fixture") or "").strip()
    home_away = (match_row.get("home_away") or "").strip()
    season = (match_row.get("season") or "").strip()
    match_type = (match_row.get("match_type") or "").strip()

    gf = _safe_int(match_row.get("goals_for"))
    ga = _safe_int(match_row.get("goals_against"))
    score_txt = "Score: –"
    if gf is not None and ga is not None:
        score_txt = f"Score: {gf} - {ga}"

    mid_lines = []
    if match_date:
        mid_lines.append(str(match_date))
    if fixture:
        mid_lines.append(fixture)
    elif opp:
        mid_lines.append(opp)

    sub = []
    if home_away:
        sub.append(home_away)
    if match_type:
        sub.append(match_type)
    if season:
        sub.append(season)
    sub.append(score_txt)

    left, mid, right = st.columns([1, 2, 1])

    with left:
        if mvv_logo and os.path.exists(mvv_logo):
            st.image(mvv_logo, width=160)
    with mid:
        st.markdown(f"### {' • '.join([x for x in mid_lines if x])}")
        st.markdown("**" + " • ".join([x for x in sub if x]) + "**")
        if opp:
            st.caption(f"Tegenstander: {opp}")
    with right:
        if opp_logo and os.path.exists(opp_logo):
            st.image(opp_logo, width=160)


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
        st.info("Geen matches gevonden in public.matches.")
        return

    def _label(r: pd.Series) -> str:
        d = r.get("match_date")
        opp = r.get("opponent") or ""
        fixture = r.get("fixture") or ""
        ha = r.get("home_away") or ""
        season = r.get("season") or ""
        gf = _safe_int(r.get("goals_for"))
        ga = _safe_int(r.get("goals_against"))
        score = f"{gf}-{ga}" if (gf is not None and ga is not None) else "–"
        core = fixture if fixture else opp
        return f"{d} · {core} · {ha} · {season} · {score}"

    options = matches_df["match_id"].tolist()
    labels = {int(r["match_id"]): _label(r) for _, r in matches_df.iterrows()}

    sel_match_id = st.selectbox(
        "Select match",
        options=options,
        format_func=lambda mid: labels.get(int(mid), str(mid)),
        index=0,
        key="mr_match_select",
    )

    match_row = matches_df[matches_df["match_id"] == sel_match_id].iloc[0].to_dict()
    render_header(match_row)

    st.divider()

    dfe = fetch_match_events(sb, int(sel_match_id))
    if dfe.empty:
        st.info("Geen match events gevonden in v_gps_match_events voor deze match.")
        return

    sort_choice = st.selectbox("Sort tables on (per minute)", list(SORT_OPTIONS.keys()), index=0, key="mr_sort")

    segment = st.radio(
        "Tables",
        ["Full match", "First half", "Second half"],
        horizontal=True,
        key="mr_segment",
    )

    dfe_seg = _segment_filter(dfe, segment)
    agg = agg_for_charts_and_tables(dfe_seg)
    if agg.empty:
        st.info("Geen data na filter.")
        return
    agg_sorted = sort_players(agg, sort_choice)

    g1, g2 = st.columns(2)
    with g1:
        plot_total_distance_bar(agg_sorted, segment)
    with g2:
        plot_sprint_vs_high(agg_sorted, segment)

    st.markdown("## Tables")
    render_tables_row(agg_sorted)


if __name__ == "__main__":
    main()
