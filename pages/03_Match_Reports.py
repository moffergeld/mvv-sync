# pages/03_Match_Reports.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from roles import get_sb, require_auth


# ============================================================
# Config
# ============================================================
MVV_NAME = "MVV Maastricht"
TEAM_LOGO_DIR = Path("Assets/Afbeeldingen/Team_Logos")

MATCH_EVENTS_VIEW = "v_gps_match_events"  # must contain match_id
MATCHES_TABLE = "matches"


# ============================================================
# Styling
# ============================================================
def inject_css():
    st.markdown(
        """
        <style>
          .mr-header-wrap{width:100%; padding:8px 0 6px 0;}
          .mr-header{display:flex; align-items:center; justify-content:space-between; gap:24px;}
          .mr-team{width:240px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:10px;}
          .mr-team img{display:block;}
          .mr-team-name{opacity:.9; font-weight:700; text-align:center; line-height:1.1; margin:0;}
          .mr-center{flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; gap:10px; min-height:220px;}
          .mr-date{opacity:.85; font-weight:600; letter-spacing:.2px;}
          .mr-fixture{font-size:28px; font-weight:800; margin:0; line-height:1.1;}
          .mr-score{font-size:56px; font-weight:900; margin:0; line-height:1;}
          .mr-chips{display:flex; gap:10px; flex-wrap:wrap; justify-content:center; margin-top:2px;}
          .mr-chip{border:1px solid rgba(255,255,255,.15); background:rgba(255,255,255,.06);
                   padding:6px 12px; border-radius:999px; font-weight:700; font-size:13px; opacity:.95;}

          .mr-table{width:100%; border-collapse:separate; border-spacing:0; overflow:visible; font-size:12px;}
          .mr-table th{text-align:left; padding:8px 10px; background:rgba(255,255,255,.04);
                       border-top:1px solid rgba(255,255,255,.10); border-bottom:1px solid rgba(255,255,255,.10);}
          .mr-table th:first-child{border-left:1px solid rgba(255,255,255,.10); border-top-left-radius:10px;}
          .mr-table th:last-child{border-right:1px solid rgba(255,255,255,.10); border-top-right-radius:10px;}
          .mr-table td{padding:7px 10px; border-bottom:1px solid rgba(255,255,255,.08);}
          .mr-table td:first-child{border-left:1px solid rgba(255,255,255,.10); font-weight:700; text-align:left; white-space:nowrap;}
          .mr-table td:last-child{border-right:1px solid rgba(255,255,255,.10);}
          .mr-table tr:last-child td:first-child{border-bottom-left-radius:10px;}
          .mr-table tr:last-child td:last-child{border-bottom-right-radius:10px;}

          .mr-num{text-align:center; font-variant-numeric:tabular-nums;}
          .mr-table-card{border:1px solid rgba(255,255,255,.10); border-radius:12px; background:rgba(255,255,255,.02); padding:10px;}
          .mr-table-title{font-weight:800; margin:0 0 8px 0; opacity:.95;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Helpers
# ============================================================
def _df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def fmt0(x) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.0f}"


def fmt2(x) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.2f}"


# ============================================================
# Supabase fetchers (with visible errors)
# ============================================================
def fetch_matches(sb, limit: int = 800) -> pd.DataFrame:
    # 1) minimal test query -> direct error if RLS/permission/table issue
    try:
        test = sb.table(MATCHES_TABLE).select("match_id").limit(1).execute()
        _ = test.data or []
    except Exception as e:
        st.error(f"Kan {MATCHES_TABLE} niet lezen (test query). Waarschijnlijk RLS/permissions/naam. Error: {e}")
        return pd.DataFrame()

    # 2) full query
    try:
        rows = (
            sb.table(MATCHES_TABLE)
            .select("match_id,match_date,fixture,home_away,opponent,match_type,season,goals_for,goals_against,result")
            .order("match_date", desc=True)
            .limit(limit)
            .execute()
            .data
            or []
        )
    except Exception as e:
        st.error(f"Fout bij ophalen matches: {e}")
        return pd.DataFrame()

    df = _df(rows)
    if df.empty:
        return df

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.date
    for c in ["opponent", "home_away", "fixture", "match_type", "season"]:
        df[c] = df[c].fillna("").astype(str)

    df["goals_for"] = pd.to_numeric(df["goals_for"], errors="coerce")
    df["goals_against"] = pd.to_numeric(df["goals_against"], errors="coerce")
    return df


def fetch_match_events(sb, match_id: int) -> pd.DataFrame:
    try:
        rows = (
            sb.table(MATCH_EVENTS_VIEW)
            .select("match_id,player_name,event,duration,total_distance,running,sprint,high_sprint,max_speed")
            .eq("match_id", match_id)
            .execute()
            .data
            or []
        )
    except Exception as e:
        st.error(f"Fout bij ophalen events ({MATCH_EVENTS_VIEW}): {e}")
        return pd.DataFrame()

    df = _df(rows)
    if df.empty:
        return df

    df["player_name"] = df["player_name"].fillna("").astype(str)
    df["event"] = df["event"].fillna("").astype(str)

    for c in ["duration", "total_distance", "running", "sprint", "high_sprint", "max_speed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ============================================================
# Logos
# ============================================================
def find_logo_path(team_name: str) -> Optional[Path]:
    if not TEAM_LOGO_DIR.exists():
        return None

    candidates = [
        TEAM_LOGO_DIR / f"{team_name}.png",
        TEAM_LOGO_DIR / f"{team_name}.PNG",
        TEAM_LOGO_DIR / f"{team_name}.jpg",
        TEAM_LOGO_DIR / f"{team_name}.jpeg",
    ]
    for p in candidates:
        if p.exists():
            return p

    target = team_name.strip().lower()
    for p in TEAM_LOGO_DIR.glob("*"):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            if p.stem.strip().lower() == target:
                return p
    return None


# ============================================================
# Processing
# ============================================================
def filter_event(df: pd.DataFrame, which: str) -> pd.DataFrame:
    if df.empty:
        return df
    w = which.lower()
    if w == "full match":
        return df[df["event"].isin(["First Half", "Second Half"])]
    if w == "first half":
        return df[df["event"] == "First Half"]
    if w == "second half":
        return df[df["event"] == "Second Half"]
    return df


def agg_players(df_events: pd.DataFrame) -> pd.DataFrame:
    if df_events.empty:
        return df_events

    g = df_events.groupby("player_name", as_index=False).agg(
        duration=("duration", "sum"),
        total_distance=("total_distance", "sum"),
        running=("running", "sum"),
        sprint=("sprint", "sum"),
        high_sprint=("high_sprint", "sum"),
        max_speed=("max_speed", "max"),
    )
    g["duration"] = g["duration"].fillna(0.0)

    for col in ["total_distance", "running", "sprint", "high_sprint"]:
        g[f"{col}_pm"] = (g[col] / g["duration"]).where(g["duration"] > 0, 0.0)

    return g


def pct_color(val: float, series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "rgba(255,255,255,0.02)"
    p20, p40, p60, p80 = s.quantile([0.2, 0.4, 0.6, 0.8]).tolist()
    if val <= p20:
        return "rgba(255, 0, 51, 0.35)"
    if val <= p40:
        return "rgba(255, 122, 0, 0.30)"
    if val <= p60:
        return "rgba(255, 196, 0, 0.26)"
    if val <= p80:
        return "rgba(60, 220, 120, 0.22)"
    return "rgba(0, 180, 90, 0.30)"


def render_metric_table(df_players: pd.DataFrame, title: str, abs_col: str, pm_col: Optional[str]):
    if df_players.empty:
        st.info("Geen data.")
        return

    df = df_players.copy()

    # sort per table by its own /min (if exists) else abs
    if pm_col and pm_col in df.columns:
        df = df.sort_values(pm_col, ascending=False)
    else:
        df = df.sort_values(abs_col, ascending=False)

    abs_series = df[abs_col]

    rows_html = []
    for _, r in df.iterrows():
        player = str(r["player_name"])
        abs_val = r.get(abs_col)
        bg = pct_color(float(abs_val) if pd.notna(abs_val) else 0.0, abs_series)

        if pm_col:
            rows_html.append(
                f"""
                <tr>
                  <td>{player}</td>
                  <td class="mr-num" style="background:{bg};">{fmt0(abs_val)}</td>
                  <td class="mr-num">{fmt2(r.get(pm_col))}</td>
                </tr>
                """
            )
        else:
            rows_html.append(
                f"""
                <tr>
                  <td>{player}</td>
                  <td class="mr-num" style="background:{bg};">{fmt0(abs_val)}</td>
                </tr>
                """
            )

    if pm_col:
        headers = f"""
          <tr>
            <th>Player</th>
            <th class="mr-num">{title}</th>
            <th class="mr-num">/min</th>
          </tr>
        """
    else:
        headers = f"""
          <tr>
            <th>Player</th>
            <th class="mr-num">{title}</th>
          </tr>
        """

    st.markdown(
        f"""
        <div class="mr-table-card">
          <div class="mr-table-title">{title}</div>
          <table class="mr-table">
            <thead>{headers}</thead>
            <tbody>
              {''.join(rows_html)}
            </tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_total_distance(df_players: pd.DataFrame, title_suffix: str):
    if df_players.empty:
        st.info("Geen data.")
        return
    d = df_players.sort_values("total_distance", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["player_name"], y=d["total_distance"], marker=dict(color="#FF0033")))
    fig.update_layout(
        title=f"Total Distance ({title_suffix})",
        margin=dict(l=10, r=10, t=35, b=10),
        height=330,
        showlegend=False,
    )
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


def plot_sprint_vs_high(df_players: pd.DataFrame, title_suffix: str):
    if df_players.empty:
        st.info("Geen data.")
        return

    d = df_players.copy()
    d["sprint_total"] = pd.to_numeric(d["sprint"], errors="coerce").fillna(0) + pd.to_numeric(
        d["high_sprint"], errors="coerce"
    ).fillna(0)
    d = d.sort_values("sprint_total", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["player_name"], y=d["sprint"], marker=dict(color="rgba(255,0,51,0.55)"), name="sprint"))
    fig.add_trace(
        go.Bar(x=d["player_name"], y=d["high_sprint"], marker=dict(color="rgba(255,0,51,1.0)"), name="high_sprint")
    )
    fig.update_layout(
        barmode="group",
        title=f"Sprint vs High Sprint ({title_suffix})",
        margin=dict(l=10, r=10, t=35, b=10),
        height=330,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


def render_header(match_row: Dict[str, Any], left_team: str, right_team: str):
    match_date = match_row.get("match_date")
    fixture = (match_row.get("fixture") or "").strip()
    match_type = (match_row.get("match_type") or "").strip()
    season = (match_row.get("season") or "").strip()
    home_away = (match_row.get("home_away") or "").strip()

    gf = match_row.get("goals_for")
    ga = match_row.get("goals_against")
    try:
        gf_i = int(gf) if pd.notna(gf) else None
    except Exception:
        gf_i = None
    try:
        ga_i = int(ga) if pd.notna(ga) else None
    except Exception:
        ga_i = None

    if gf_i is None or ga_i is None:
        score_txt = "—"
    else:
        if home_away.lower() == "away":
            score_txt = f"{ga_i} - {gf_i}"
        else:
            score_txt = f"{gf_i} - {ga_i}"

    fixture_line = fixture if fixture else f"{left_team} - {right_team}"

    left_logo = find_logo_path(left_team)
    right_logo = find_logo_path(right_team)

    st.markdown('<div class="mr-header-wrap"><div class="mr-header">', unsafe_allow_html=True)

    st.markdown('<div class="mr-team">', unsafe_allow_html=True)
    if left_logo and left_logo.exists():
        st.image(str(left_logo), width=200)
    else:
        st.markdown(f"**{left_team}**")
    st.markdown(f'<div class="mr-team-name">{left_team}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="mr-center">', unsafe_allow_html=True)
    st.markdown(f'<div class="mr-date">{match_date}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mr-fixture">{fixture_line}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mr-score">{score_txt}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="mr-chips">
          <div class="mr-chip">{home_away if home_away else "-"}</div>
          <div class="mr-chip">{match_type if match_type else "-"}</div>
          <div class="mr-chip">{season if season else "-"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="mr-team">', unsafe_allow_html=True)
    if right_logo and right_logo.exists():
        st.image(str(right_logo), width=200)
    else:
        st.markdown(f"**{right_team}**")
    st.markdown(f'<div class="mr-team-name">{right_team}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


# ============================================================
# Main
# ============================================================
def main():
    require_auth()
    inject_css()

    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    st.title("Match Reports")

    # Optional: quick debug toggle
    debug = st.toggle("Debug", value=False, key="mr_debug")

    matches_df = fetch_matches(sb, limit=800)

    if debug:
        st.write("matches_df rows:", 0 if matches_df is None else len(matches_df))
        if matches_df is not None and not matches_df.empty:
            st.dataframe(matches_df.head(20), use_container_width=True, hide_index=True)

    if matches_df is None or matches_df.empty:
        st.info("Geen matches gevonden.")
        return

    # Opponent dropdown A->Z
    opponents = sorted([o for o in matches_df["opponent"].unique().tolist() if str(o).strip()])

    c1, c2 = st.columns([1, 2])
    with c1:
        opp = st.selectbox("Opponent", options=opponents, index=0, key="mr_opp")

    df_opp = matches_df[matches_df["opponent"] == opp].copy().sort_values("match_date", ascending=False)

    def ha_tag(x: str) -> str:
        x = (x or "").strip().lower()
        return "H" if x == "home" else "A"

    date_labels = []
    id_by_label: Dict[str, int] = {}
    for _, r in df_opp.iterrows():
        d = r["match_date"]
        ha = ha_tag(r.get("home_away", ""))
        label = f"{d} ({ha})"
        date_labels.append(label)
        id_by_label[label] = int(r["match_id"])

    with c2:
        date_label = st.selectbox("Date", options=date_labels, index=0, key="mr_date")

    match_id = id_by_label[date_label]
    match_row = df_opp[df_opp["match_id"] == match_id].iloc[0].to_dict()

    home_away = (match_row.get("home_away") or "").strip().lower()
    if home_away == "away":
        left_team, right_team = opp, MVV_NAME
    else:
        left_team, right_team = MVV_NAME, opp

    render_header(match_row, left_team=left_team, right_team=right_team)

    st.divider()

    which = st.radio("Tables", ["Full match", "First Half", "Second Half"], horizontal=True, key="mr_half")

    events_df = fetch_match_events(sb, match_id=match_id)
    if debug:
        st.write("events rows:", 0 if events_df is None else len(events_df))
        if events_df is not None and not events_df.empty:
            st.dataframe(events_df.head(25), use_container_width=True, hide_index=True)

    if events_df is None or events_df.empty:
        st.info("Geen match events gevonden (v_gps_match_events).")
        return

    events_df = filter_event(events_df, which=which)
    players_df = agg_players(events_df)

    # Plots
    p1, p2 = st.columns(2)
    with p1:
        plot_total_distance(players_df, title_suffix=which)
    with p2:
        plot_sprint_vs_high(players_df, title_suffix=which)

    st.markdown("## Tables")

    cols = st.columns(5)
    with cols[0]:
        render_metric_table(players_df, title="TD", abs_col="total_distance", pm_col="total_distance_pm")
    with cols[1]:
        render_metric_table(players_df, title="14.4–19.7", abs_col="running", pm_col="running_pm")
    with cols[2]:
        render_metric_table(players_df, title="19.8–25.1", abs_col="sprint", pm_col="sprint_pm")
    with cols[3]:
        render_metric_table(players_df, title="25.2+", abs_col="high_sprint", pm_col="high_sprint_pm")
    with cols[4]:
        render_metric_table(players_df, title="Max Speed", abs_col="max_speed", pm_col=None)


if __name__ == "__main__":
    main()
