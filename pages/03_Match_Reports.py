# pages/03_Match_Reports.py
# ============================================================
# Match Reports (Streamlit)
# Layout gebaseerd op match_analysis.py :contentReference[oaicite:0]{index=0}
#
# Vereisten:
# - Gebruik Plotly (zelfde stijl als Player pages)
# - Data uit:
#     - public.matches
#     - public.v_gps_match_events (First Half / Second Half events per speler)
# - 1 pagina/tab met: header + graphs + 3 tabellen
#     Tabel 1: Full match = First+Second samen
#     Tabel 2: First half
#     Tabel 3: Second half
# - Percentiel-kleuren op ABSOLUTE waardes (niet /min)
# - Sorteren op per-minute basis (kies metric)
# - Team-logo’s uit repo map: Assets/Afbeeldingen/Team_Logos
# - Fix Streamlit cache “UnhashableParamError” (cache functies nemen géén sb als parameter)
# ============================================================

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from roles import get_sb, require_auth, get_profile, pick_target_player  # pick_target_player niet nodig, maar laten staan


# -----------------------------
# Config
# -----------------------------
MATCHES_TABLE = "matches"
MATCH_EVENTS_VIEW = "v_gps_match_events"

# Repo pad (pas aan als jouw structuur anders is)
TEAM_LOGO_DIR = Path("Assets/Afbeeldingen/Team_Logos")

MVV_NAME = "MVV Maastricht"

# Metrics (kolommen in v_gps_match_events)
METRICS_SUM = [
    ("Total Distance (m)", "total_distance"),
    ("14.4–19.7 km/h", "running"),
    ("19.8–25.1 km/h", "sprint"),
    (">25,1 km/h", "high_sprint"),
    ("Duration (min)", "duration"),
]
METRICS_MAX = [
    ("Max Speed (km/u)", "max_speed"),
]

DEFAULT_SORT_PM = "total_distance"  # sort op Total Distance/min


# -----------------------------
# Utils
# -----------------------------
def _df_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _coerce_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _normalize_event(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in {"first half", "1st half", "firsthalf", "h1", "1e helft", "1ste helft"}:
        return "first"
    if s in {"second half", "2nd half", "secondhalf", "h2", "2e helft", "2de helft"}:
        return "second"
    if s == "summary":
        return "summary"
    return s


def _safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col), errors="coerce")


def _logo_path(team: str) -> Optional[Path]:
    if not team:
        return None
    # bestandsnamen in jouw repo lijken exact clubnaam + ".png"
    # (zoals "ADO Den Haag.png", "Jong Ajax.png", etc.)
    p = TEAM_LOGO_DIR / f"{team}.png"
    if p.exists():
        return p
    # fallback: case-insensitive match
    want = team.strip().lower()
    for f in TEAM_LOGO_DIR.glob("*.png"):
        if f.stem.strip().lower() == want:
            return f
    return None


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def _percentile_colors(values: pd.Series) -> List[str]:
    """
    Maak per rij een kleur obv percentiel (rood->oranje->groen) op ABSOLUTE waarde.
    """
    v = pd.to_numeric(values, errors="coerce")
    if v.notna().sum() <= 1:
        return [""] * len(values)

    pct = v.rank(pct=True, method="average")
    # stops: rood (#d73027) -> oranje (#fdae61) -> groen (#1a9850)
    c_red = np.array([0xD7, 0x30, 0x27], dtype=float)
    c_org = np.array([0xFD, 0xAE, 0x61], dtype=float)
    c_grn = np.array([0x1A, 0x98, 0x50], dtype=float)

    out = []
    for p in pct.fillna(np.nan).to_numpy():
        if np.isnan(p):
            out.append("")
            continue
        if p <= 0.5:
            t = p / 0.5
            rgb = (1 - t) * c_red + t * c_org
        else:
            t = (p - 0.5) / 0.5
            rgb = (1 - t) * c_org + t * c_grn
        out.append(f"background-color: {_rgba('%02x%02x%02x' % tuple(rgb.astype(int)), 0.75)};")
    return out


def _style_table(df: pd.DataFrame, abs_cols: List[str]) -> "pd.io.formats.style.Styler":
    """
    Percentiel kleuren op ABSOLUTE kolommen (dus niet op /min).
    """
    sty = df.style
    for c in abs_cols:
        if c in df.columns:
            sty = sty.apply(lambda s: _percentile_colors(s), subset=[c])
    return sty


def _add_zone_background(fig: go.Figure):
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


# -----------------------------
# Fetch (GEEN sb in cache-args)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60)
def fetch_matches_cached(limit: int = 500) -> pd.DataFrame:
    sb = get_sb()
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
    df["match_date"] = _coerce_date(df["match_date"])
    return df


@st.cache_data(show_spinner=False, ttl=60)
def fetch_match_events_cached(match_id: int, limit: int = 5000) -> pd.DataFrame:
    sb = get_sb()
    rows = (
        sb.table(MATCH_EVENTS_VIEW)
        .select(
            "gps_id,match_id,player_id,player_name,datum,event,duration,total_distance,running,sprint,high_sprint,max_speed"
        )
        .eq("match_id", match_id)
        .order("player_name", desc=False)
        .limit(limit)
        .execute()
        .data
        or []
    )
    df = _df_from_rows(rows)
    if df.empty:
        return df

    df["datum"] = _coerce_date(df["datum"])
    df["event_norm"] = df["event"].map(_normalize_event)

    for _, c in METRICS_SUM + METRICS_MAX:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# -----------------------------
# Aggregaties (per speler)
# -----------------------------
def _agg_block(df: pd.DataFrame, event_norm: Optional[str]) -> pd.DataFrame:
    """
    event_norm:
      - None => first+second samen
      - "first" / "second"
    Output: abs + /min (voor sum-metrics) + max-speed als max
    """
    if df.empty:
        return df

    if event_norm is None:
        d = df[df["event_norm"].isin(["first", "second"])].copy()
    else:
        d = df[df["event_norm"].eq(event_norm)].copy()

    if d.empty:
        return pd.DataFrame()

    group_cols = ["player_name"]

    # sums
    sum_cols = [c for _, c in METRICS_SUM if c in d.columns]
    # max
    max_cols = [c for _, c in METRICS_MAX if c in d.columns]

    out_parts = []

    if sum_cols:
        gsum = d.groupby(group_cols, as_index=False)[sum_cols].sum()
        out_parts.append(gsum)

    if max_cols:
        gmax = d.groupby(group_cols, as_index=False)[max_cols].max()
        out_parts.append(gmax)

    out = out_parts[0]
    for p in out_parts[1:]:
        out = out.merge(p, on=group_cols, how="outer")

    # per-minute (op basis van duration)
    if "duration" in out.columns:
        mins = pd.to_numeric(out["duration"], errors="coerce")
    else:
        mins = pd.Series([np.nan] * len(out), index=out.index)

    for _, c in METRICS_SUM:
        if c in out.columns and c != "duration":
            out[f"{c}_pm"] = np.where(mins > 0, out[c] / mins, np.nan)

    return out


def _pretty_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Return: (pretty_df, abs_cols_for_percentile)
    """
    if df.empty:
        return df, []

    rename_abs = {
        "player_name": "Player",
        "total_distance": "Total Distance (m)",
        "running": "14.4–19.7 km/h",
        "sprint": "19.8–25.1 km/h",
        "high_sprint": ">25,1 km/h",
        "duration": "Duration (min)",
        "max_speed": "Max Speed (km/u)",
    }
    rename_pm = {
        "total_distance_pm": "Total Distance/min",
        "running_pm": "14.4–19.7/min",
        "sprint_pm": "19.8–25.1/min",
        "high_sprint_pm": ">25,1/min",
    }

    show = df.copy()

    # ronding / types
    for c in ["total_distance", "running", "sprint", "high_sprint", "duration"]:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce")
    if "max_speed" in show.columns:
        show["max_speed"] = pd.to_numeric(show["max_speed"], errors="coerce")

    for c in ["total_distance_pm", "running_pm", "sprint_pm", "high_sprint_pm"]:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce")

    cols = ["player_name", "duration", "total_distance", "running", "sprint", "high_sprint", "max_speed",
            "total_distance_pm", "running_pm", "sprint_pm", "high_sprint_pm"]
    cols = [c for c in cols if c in show.columns]
    show = show[cols]

    show = show.rename(columns={**rename_abs, **rename_pm})

    # ABS columns voor percentiel-kleur
    abs_cols = [v for k, v in rename_abs.items() if k in df.columns and v != "Player"]

    return show, abs_cols


def _sort_table(df_pretty: pd.DataFrame, sort_metric_key: str) -> pd.DataFrame:
    """
    sort_metric_key is de RAW metric key (bv "total_distance").
    We sorteren op per-minute kolom (zoals gevraagd).
    """
    map_pm = {
        "total_distance": "Total Distance/min",
        "running": "14.4–19.7/min",
        "sprint": "19.8–25.1/min",
        "high_sprint": ">25,1/min",
    }
    sort_col = map_pm.get(sort_metric_key, "Total Distance/min")
    if sort_col in df_pretty.columns:
        return df_pretty.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)
    return df_pretty


# -----------------------------
# Graphs (Plotly)
# -----------------------------
def _bar_topn(df_agg: pd.DataFrame, value_col: str, title: str, n: int = 18) -> go.Figure:
    d = df_agg[["player_name", value_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col]).sort_values(value_col, ascending=False).head(n)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["player_name"], y=d[value_col], marker=dict(color="#FF0033")))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        title=title,
        showlegend=False,
    )
    fig.update_xaxes(tickangle=90)
    return fig


def _bar_duo_topn(df_agg: pd.DataFrame, c1: str, c2: str, title: str, n: int = 18) -> go.Figure:
    d = df_agg[["player_name", c1, c2]].copy()
    d[c1] = pd.to_numeric(d[c1], errors="coerce")
    d[c2] = pd.to_numeric(d[c2], errors="coerce")
    d = d.fillna(0.0)
    d = d.sort_values(c1, ascending=False).head(n)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["player_name"], y=d[c1], name=c1, marker=dict(color="#FF6666")))
    fig.add_trace(go.Bar(x=d["player_name"], y=d[c2], name=c2, marker=dict(color="#FF0033")))
    fig.update_layout(
        barmode="group",
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        title=title,
        legend=dict(orientation="h", y=1.02, x=0),
    )
    fig.update_xaxes(tickangle=90)
    return fig


# -----------------------------
# UI
# -----------------------------
def _match_label(row: pd.Series) -> str:
    dt = pd.to_datetime(row["match_date"]).date() if pd.notna(row.get("match_date")) else None
    opp = str(row.get("opponent") or "").strip()
    ha = str(row.get("home_away") or "").strip().lower()
    ha_short = "H" if ha.startswith("h") else ("A" if ha.startswith("a") else "")
    dts = dt.strftime("%d-%m-%Y") if dt else "—"
    return f"{dts} ({ha_short}) vs {opp}"


def main():
    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    st.title("Match Reports")

    matches = fetch_matches_cached(limit=500)
    if matches.empty:
        st.info("Geen wedstrijden gevonden in public.matches.")
        return

    matches = matches.sort_values("match_date", ascending=False).reset_index(drop=True)
    labels = matches.apply(_match_label, axis=1).tolist()
    pick = st.selectbox("Select match", options=list(range(len(labels))), format_func=lambda i: labels[i], index=0)

    row = matches.iloc[int(pick)]
    match_id = int(row["match_id"])
    match_date = row["match_date"]
    opponent = str(row.get("opponent") or "").strip()
    home_away = str(row.get("home_away") or "").strip()

    # Header card (logos + basic info)
    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        p = _logo_path(MVV_NAME)
        if p:
            st.image(str(p), use_container_width=True)
    with c2:
        st.subheader(_match_label(row))
        meta = []
        if opponent:
            meta.append(f"Opponent: {opponent}")
        if home_away:
            meta.append(f"Home/Away: {home_away}")
        if row.get("season"):
            meta.append(f"Season: {row.get('season')}")
        if row.get("match_type"):
            meta.append(f"Type: {row.get('match_type')}")
        if row.get("result") or (pd.notna(row.get("goals_for")) and pd.notna(row.get("goals_against"))):
            gf = row.get("goals_for")
            ga = row.get("goals_against")
            if pd.notna(gf) and pd.notna(ga):
                meta.append(f"Score: {int(gf)}-{int(ga)}")
            if row.get("result"):
                meta.append(f"Result: {row.get('result')}")
        if meta:
            st.caption(" • ".join(meta))
    with c3:
        p = _logo_path(opponent)
        if p:
            st.image(str(p), use_container_width=True)

    st.divider()

    # Alles op 1 pagina/tab: Graphs + Tables
    df = fetch_match_events_cached(match_id=match_id, limit=8000)
    if df.empty:
        st.info("Geen data gevonden in v_gps_match_events voor deze match_id.")
        return

    # Alleen halves (First/Second) voor reports zoals jouw voorbeeld
    df_halves = df[df["event_norm"].isin(["first", "second"])].copy()
    if df_halves.empty:
        st.info("Geen First/Second half events gevonden.")
        return

    # Agg blocks
    full_agg = _agg_block(df_halves, event_norm=None)
    first_agg = _agg_block(df_halves, event_norm="first")
    second_agg = _agg_block(df_halves, event_norm="second")

    # Sort metric (per-minute)
    sort_label_to_key = {
        "Total Distance/min": "total_distance",
        "14.4–19.7/min": "running",
        "19.8–25.1/min": "sprint",
        ">25,1/min": "high_sprint",
    }
    sort_choice = st.selectbox("Sort tables on (per minute)", options=list(sort_label_to_key.keys()), index=0)
    sort_key = sort_label_to_key[sort_choice]

    # ----- Graphs (boven) -----
    g1, g2 = st.columns(2)
    with g1:
        fig = _bar_topn(full_agg, "total_distance", "Total Distance (Full match)")
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        fig = _bar_duo_topn(full_agg, "sprint", "high_sprint", "Sprint vs High Sprint (Full match)")
        st.plotly_chart(fig, use_container_width=True)

    # ----- Tables (onder): 3 stuks -----
    st.subheader("Tables")

    def render_table(title: str, agg_df: pd.DataFrame):
        st.markdown(f"**{title}**")
        pretty, abs_cols = _pretty_table(agg_df)
        if pretty.empty:
            st.info("Geen data.")
            return
        pretty = _sort_table(pretty, sort_metric_key=sort_key)

        # Percentiel kleuren op ABSOLUTE kolommen
        sty = _style_table(pretty, abs_cols=abs_cols)

        # Toon
        st.dataframe(sty, use_container_width=True, hide_index=True)

    t1, t2, t3 = st.columns(3)
    with t1:
        render_table("Full match (First + Second)", full_agg)
    with t2:
        render_table("First half", first_agg)
    with t3:
        render_table("Second half", second_agg)


if __name__ == "__main__":
    main()
