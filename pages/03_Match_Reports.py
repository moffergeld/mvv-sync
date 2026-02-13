# 03_Match_Reports.py
# ============================================================
# Match Reports (1 pagina/tab):
# - Selecteer match (matches tabel)
# - Haalt events uit: public.v_gps_match_events (moet match_id bevatten)
# - 3 tabellen:
#     1) Full Match (1st + 2nd half samen)
#     2) First Half
#     3) Second Half
# - Sorteren op per-minute basis (default: Total Distance / min)
# - Percentiel-kleuren op ABSOLUTE waardes (per metric kolom)
# - Per-minute kolommen worden berekend en gebruikt voor sortering
# - Grafieken + tabellen op dezelfde pagina
#
# Vereist:
# - roles.py met: get_sb, require_auth, get_profile
# - v_gps_match_events view met match_id in kolommen
# ============================================================

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from roles import get_sb, require_auth, get_profile


# -----------------------------
# Config
# -----------------------------
MATCH_EVENTS_TABLE = "v_gps_match_events"
MATCHES_TABLE = "matches"

# Metrics die je in tabellen/grafieken toont
METRICS = [
    ("Total Distance (m)", "total_distance"),
    ("14.4–19.7 km/h", "running"),
    ("19.8–25.1 km/h", "sprint"),
    (">25,1 km/h", "high_sprint"),
    ("Max Speed (km/u)", "max_speed"),
]

# Kolommen die we nodig hebben uit v_gps_match_events
EVENT_COLS = [
    "match_id",
    "player_id",
    "player_name",
    "datum",
    "type",
    "event",
    "duration",
    "total_distance",
    "running",
    "sprint",
    "high_sprint",
    "max_speed",
]


# -----------------------------
# Helpers
# -----------------------------
def _df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _to_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b2 = b.replace({0: pd.NA})
    return a / b2


def _percentile_colors(series: pd.Series) -> List[str]:
    """
    Percentielkleur per cel (ABSOLUTE waardes):
    - laag = groen, hoog = rood
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() < 2:
        return [""] * len(series)

    # percent_rank 0..1
    pr = s.rank(pct=True)

    def col(p: float) -> str:
        if pd.isna(p):
            return ""
        if p >= 0.90:
            return "background-color: rgba(255,0,0,0.25)"      # rood
        if p >= 0.70:
            return "background-color: rgba(255,165,0,0.22)"    # oranje
        if p >= 0.40:
            return "background-color: rgba(255,215,0,0.18)"    # geel
        return "background-color: rgba(0,200,0,0.14)"          # groen

    return [col(p) for p in pr.tolist()]


def style_percentiles_absolute(df: pd.DataFrame, abs_cols: List[str]) -> "pd.io.formats.style.Styler":
    sty = df.style
    for c in abs_cols:
        if c in df.columns:
            sty = sty.apply(lambda _: _percentile_colors(df[c]), subset=[c])
    return sty


def _format_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nette afronding/kolomvolgorde.
    """
    out = df.copy()

    # afronden
    for _, key in METRICS:
        if key in out.columns:
            if key == "max_speed":
                out[key] = pd.to_numeric(out[key], errors="coerce").round(2)
            else:
                out[key] = pd.to_numeric(out[key], errors="coerce").round(0)

    # duration min (float -> 1 dec)
    if "duration_min" in out.columns:
        out["duration_min"] = pd.to_numeric(out["duration_min"], errors="coerce").round(1)

    # per-min afronden
    for _, key in METRICS:
        kpm = f"{key}_per_min"
        if kpm in out.columns:
            out[kpm] = pd.to_numeric(out[kpm], errors="coerce").round(2)

    # kolommen zetten
    base_cols = ["player_name", "duration_min"]
    abs_cols = [k for _, k in METRICS]
    pm_cols = [f"{k}_per_min" for _, k in METRICS]

    cols = [c for c in base_cols + abs_cols + pm_cols if c in out.columns]
    return out[cols]


# -----------------------------
# Supabase queries
# -----------------------------
def fetch_matches(sb, limit: int = 500) -> pd.DataFrame:
    rows = (
        sb.table(MATCHES_TABLE)
        .select("match_id,match_date,fixture,opponent,home_away,season,result,goals_for,goals_against")
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
    return df


def fetch_match_events(sb, match_id: int) -> pd.DataFrame:
    rows = (
        sb.table(MATCH_EVENTS_TABLE)
        .select(",".join(EVENT_COLS))
        .eq("match_id", match_id)
        .execute()
        .data
        or []
    )
    df = _df(rows)
    if df.empty:
        return df

    df["datum"] = pd.to_datetime(df["datum"], errors="coerce").dt.date
    df = _to_num(df, ["duration", "total_distance", "running", "sprint", "high_sprint", "max_speed"])
    return df


# -----------------------------
# Aggregation (per player)
# -----------------------------
def build_player_table(events_df: pd.DataFrame, event_filter: Optional[List[str]]) -> pd.DataFrame:
    """
    events_df: rows per speler per event (First Half/Second Half)
    event_filter: None -> alles, anders lijst met events
    Output: 1 rij per speler met duration_min + metrics (ABS) + metrics_per_min
    """
    if events_df.empty:
        return pd.DataFrame()

    df = events_df.copy()
    if event_filter:
        df = df[df["event"].isin(event_filter)]

    if df.empty:
        return pd.DataFrame()

    # duration: sommige imports hebben duration numeric (min). We noemen het duration_min.
    df["duration_min"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)

    # Aggregatie:
    # - som voor distances
    # - max voor max_speed
    agg_dict = {
        "duration_min": "sum",
        "total_distance": "sum",
        "running": "sum",
        "sprint": "sum",
        "high_sprint": "sum",
        "max_speed": "max",
    }

    out = (
        df.groupby(["player_id", "player_name"], as_index=False)
        .agg(agg_dict)
        .sort_values("player_name")
        .reset_index(drop=True)
    )

    # Per-minute metrics (sort basis)
    for _, key in METRICS:
        if key in out.columns:
            out[f"{key}_per_min"] = _safe_div(out[key], out["duration_min"])  # per minuut

    return out


# -----------------------------
# Plots
# -----------------------------
def plot_top_players_per_min(df_players: pd.DataFrame, metric_key: str, metric_label: str, top_n: int = 12):
    if df_players.empty:
        st.info("Geen data voor grafiek.")
        return

    kpm = f"{metric_key}_per_min"
    if kpm not in df_players.columns:
        st.info("Onvoldoende data voor grafiek.")
        return

    dff = df_players.copy()
    dff[kpm] = pd.to_numeric(dff[kpm], errors="coerce")
    dff = dff.dropna(subset=[kpm]).sort_values(kpm, ascending=False).head(top_n)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dff[kpm],
            y=dff["player_name"],
            orientation="h",
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_team_halves(events_df: pd.DataFrame, metric_key: str, metric_label: str):
    """
    Team total per half (som), puur ter context.
    """
    if events_df.empty:
        st.info("Geen data voor grafiek.")
        return

    df = events_df.copy()
    df[metric_key] = pd.to_numeric(df[metric_key], errors="coerce").fillna(0.0)

    half_map = {"First Half": "1st Half", "Second Half": "2nd Half"}
    df = df[df["event"].isin(["First Half", "Second Half"])].copy()
    if df.empty:
        st.info("Geen First/Second Half events gevonden.")
        return

    g = df.groupby("event", as_index=False)[metric_key].sum()
    g["half"] = g["event"].map(half_map).fillna(g["event"])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=g["half"],
            y=g[metric_key],
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# UI
# -----------------------------
def main():
    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    _ = get_profile(sb)  # nu geen role gating; alleen auth check

    st.title("Match Reports")

    matches_df = fetch_matches(sb, limit=500)
    if matches_df.empty:
        st.info("Geen matches gevonden in public.matches.")
        st.stop()

    # Selectbox label
    def _match_label(r: pd.Series) -> str:
        d = r.get("match_date")
        opp = r.get("opponent") or r.get("fixture") or "Unknown"
        ha = r.get("home_away") or ""
        season = r.get("season") or ""
        res = r.get("result") or ""
        return f"{d} | {ha} | {opp} | {season} | {res}".strip()

    matches_df = matches_df.sort_values("match_date", ascending=False).reset_index(drop=True)
    opts = matches_df["match_id"].tolist()
    labels = {int(row["match_id"]): _match_label(row) for _, row in matches_df.iterrows()}

    sel_match_id = st.selectbox(
        "Select match",
        options=opts,
        format_func=lambda mid: labels.get(int(mid), str(mid)),
        key="mr_match_select",
    )

    events_df = fetch_match_events(sb, int(sel_match_id))
    if events_df.empty:
        st.info("Geen match events gevonden in v_gps_match_events voor deze match.")
        st.stop()

    # Metric select + sort metric
    metric_label = st.selectbox("Parameter", options=[m[0] for m in METRICS], index=0, key="mr_metric_sel")
    metric_key = dict(METRICS)[metric_label]

    sort_on = st.selectbox(
        "Sorteer op (per minuut)",
        options=[m[0] for m in METRICS],
        index=0,
        key="mr_sort_sel",
    )
    sort_key = dict(METRICS)[sort_on]
    sort_kpm = f"{sort_key}_per_min"

    st.divider()

    # Grafieken bovenaan (zelfde tab/pagina)
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Team (1st vs 2nd half)")
        plot_team_halves(events_df, metric_key=metric_key, metric_label=metric_label)
    with g2:
        st.subheader(f"Top spelers ({sort_on} / min)")
        full_players = build_player_table(events_df, event_filter=["First Half", "Second Half"])
        plot_top_players_per_min(full_players, metric_key=sort_key, metric_label=sort_on)

    st.divider()

    # 3 tabellen (zelfde pagina)
    t1, t2, t3 = st.columns(3)

    def _render_table(title: str, df_players: pd.DataFrame, key: str):
        st.markdown(f"### {title}")
        if df_players.empty:
            st.info("Geen data.")
            return

        df_players = _format_table(df_players)

        # sorteer op per-minute
        if sort_kpm in df_players.columns:
            df_players = df_players.sort_values(sort_kpm, ascending=False)

        # percentiel kleuren op ABS kolommen
        abs_cols = [k for _, k in METRICS if k in df_players.columns]
        sty = style_percentiles_absolute(df_players, abs_cols)

        # toon
        st.dataframe(
            sty,
            use_container_width=True,
            hide_index=True,
            height=520,
        )

    with t1:
        df_full = build_player_table(events_df, event_filter=["First Half", "Second Half"])
        _render_table("Full Match (1st + 2nd)", df_full, key="tbl_full")

    with t2:
        df_h1 = build_player_table(events_df, event_filter=["First Half"])
        _render_table("1st Half", df_h1, key="tbl_h1")

    with t3:
        df_h2 = build_player_table(events_df, event_filter=["Second Half"])
        _render_table("2nd Half", df_h2, key="tbl_h2")


if __name__ == "__main__":
    main()
