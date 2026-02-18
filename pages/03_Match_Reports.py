# pages/03_Match_Reports.py
# ============================================================
# Match Reports (Streamlit)
# - Opponent dropdown (A-Z) + Date dropdown (YYYY-MM-DD (H/A))
# - Header: logos zelfde grootte, horizontaal gecentreerd, wisselen bij Away:
#     Away  -> opponent links, MVV rechts
#     Home  -> MVV links, opponent rechts
# - Data uit:
#     public.matches
#     public.v_gps_match_events  (events: First Half / Second Half)
# - Tabellen:
#     1 rij met 5 tabellen (geen duration, geen index)
#     per tabel sorteren op eigen /min (hoog -> laag)
#     waarden: 0 decimalen, /min: 2 decimalen
#     kleuren op percentielen van ABSOLUTE kolom (niet op /min)
#     GEEN scroll: hoogte dynamisch op basis van aantal spelers
# - Grafieken: Plotly, rood, gesorteerd op ABS (hoog -> laag)
# - Debug toggle: laat zien waarom je "geen matches" krijgt (RLS/auth)
# ============================================================

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from roles import get_sb, require_auth, get_profile

# -----------------------------
# Config
# -----------------------------
LOGO_DIR = Path("Assets/Afbeeldingen/Team_Logos")  # repo pad
MVV_TEAM_NAME = "MVV Maastricht"

EVENT_FULL = "Full match"
EVENT_FIRST = "First Half"
EVENT_SECOND = "Second Half"

EVENT_MAP = {
    EVENT_FULL: ["First Half", "Second Half"],
    EVENT_FIRST: ["First Half"],
    EVENT_SECOND: ["Second Half"],
}

# Tabellen (kolommen in v_gps_match_events)
COL_PLAYER = "player_name"
COL_TD = "total_distance"
COL_RUN = "running"
COL_SPR = "sprint"
COL_HSPR = "high_sprint"
COL_MAX = "max_speed"
COL_DUR = "duration"

# Labels in UI
LABEL_PLAYER = "Player"
LABEL_TD = "TD"
LABEL_RUN = "14.4–19.7"
LABEL_SPR = "19.8–25.1"
LABEL_HSPR = "25.2+"
LABEL_MAX = "Max Speed"
LABEL_PERMIN = "/min"

# Dataframe hoogte zodat geen scroll
ROW_HEIGHT = 32
HEADER_HEIGHT = 34
PAD_HEIGHT = 22

# Logo sizing
LOGO_W = 170

# -----------------------------
# Global CSS (voorkom interne scroll waar mogelijk)
# -----------------------------
st.markdown(
    """
<style>
/* iets strakker */
.block-container { padding-top: 1.3rem; }

/* dataframe: probeer overflow te minimaliseren */
[data-testid="stDataFrame"] div { overflow: visible !important; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
def _df_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _calc_height(n_rows: int) -> int:
    return int(HEADER_HEIGHT + PAD_HEIGHT + max(1, n_rows) * ROW_HEIGHT)


def _build_logo_index() -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    if LOGO_DIR.exists():
        for p in LOGO_DIR.glob("*.png"):
            idx[p.stem.strip().lower()] = p
    return idx


_LOGO_INDEX = _build_logo_index()


def _logo_path_for_team(team: str) -> Optional[Path]:
    if not team:
        return None
    key = team.strip().lower()
    if key in _LOGO_INDEX:
        return _LOGO_INDEX[key]

    key2 = (
        key.replace("  ", " ")
        .replace("-", " ")
        .replace("_", " ")
        .replace(".", "")
        .strip()
    )
    for k, p in _LOGO_INDEX.items():
        if k == key2:
            return p
    for k, p in _LOGO_INDEX.items():
        if key2 in k or k in key2:
            return p
    return None


def _read_image_bytes(p: Path) -> bytes:
    return p.read_bytes()


def _fmt_int0(x: Any) -> str:
    if pd.isna(x):
        return ""
    try:
        return f"{int(round(float(x), 0)):,}"
    except Exception:
        return ""


def _fmt_min2(x: Any) -> str:
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.2f}"
    except Exception:
        return ""


def _percentile_color(val: float, q25: float, q50: float, q75: float) -> str:
    if np.isnan(val):
        return ""
    red = "rgba(180, 20, 40, 0.55)"
    red2 = "rgba(180, 80, 40, 0.45)"
    amber = "rgba(190, 140, 20, 0.45)"
    green = "rgba(20, 140, 60, 0.55)"

    if val <= q25:
        return red
    if val <= q50:
        return red2
    if val <= q75:
        return amber
    return green


def _style_table(
    df: pd.DataFrame,
    abs_col: str,
    per_min_col: Optional[str],
) -> "pd.io.formats.style.Styler":
    dff = df.copy()

    abs_vals = _safe_num(dff[abs_col]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(abs_vals) >= 2:
        q25, q50, q75 = abs_vals.quantile([0.25, 0.50, 0.75]).tolist()
    elif len(abs_vals) == 1:
        q25 = q50 = q75 = float(abs_vals.iloc[0])
    else:
        q25 = q50 = q75 = 0.0

    def _bg_abs(s: pd.Series) -> List[str]:
        out = []
        for v in _safe_num(s).tolist():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out.append("")
            else:
                out.append(f"background-color: {_percentile_color(float(v), q25, q50, q75)};")
        return out

    fmt: Dict[str, Any] = {abs_col: _fmt_int0}
    if per_min_col:
        fmt[per_min_col] = _fmt_min2

    sty = (
        dff.style.format(fmt)
        .apply(_bg_abs, subset=[abs_col])
        .set_properties(subset=[LABEL_PLAYER], **{"text-align": "left"})
    )

    numeric_cols = [c for c in dff.columns if c != LABEL_PLAYER]
    sty = sty.set_properties(subset=numeric_cols, **{"text-align": "center"})

    sty = sty.set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "left"), ("font-weight", "600")]},
            {"selector": "td", "props": [("border-color", "rgba(255,255,255,0.06)")]},
        ]
    )
    return sty


# -----------------------------
# Supabase fetch
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60)
def fetch_matches_rows(limit: int = 1000) -> pd.DataFrame:
    sb = get_sb()
    res = (
        sb.table("matches")
        .select("match_id,match_date,fixture,home_away,opponent,match_type,season,goals_for,goals_against")
        .order("match_date", desc=True)
        .limit(limit)
        .execute()
    )
    rows = res.data or []
    df = _df_from_rows(rows)
    if df.empty:
        return df
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.date
    return df


@st.cache_data(show_spinner=False, ttl=60)
def fetch_match_events_for_date(match_date: date) -> pd.DataFrame:
    sb = get_sb()
    res = (
        sb.table("v_gps_match_events")
        .select(
            "gps_id,player_id,player_name,datum,type,event,duration,total_distance,running,sprint,high_sprint,max_speed"
        )
        .eq("datum", match_date.isoformat())
        .execute()
    )
    rows = res.data or []
    df = _df_from_rows(rows)
    if df.empty:
        return df

    df["datum"] = pd.to_datetime(df["datum"], errors="coerce").dt.date
    for c in [COL_DUR, COL_TD, COL_RUN, COL_SPR, COL_HSPR, COL_MAX]:
        if c in df.columns:
            df[c] = _safe_num(df[c]).fillna(0.0)
    return df


# -----------------------------
# Data prep (per phase)
# -----------------------------
def build_phase_df(df_events: pd.DataFrame, phase: str) -> pd.DataFrame:
    if df_events.empty:
        return df_events

    events_keep = EVENT_MAP.get(phase, ["First Half", "Second Half"])

    dff = df_events.copy()
    if "type" in dff.columns:
        dff = dff[dff["type"].astype(str).str.lower().eq("match")]
    dff = dff[dff["event"].isin(events_keep)]
    if dff.empty:
        return dff

    g = dff.groupby(COL_PLAYER, as_index=False).agg(
        duration=(COL_DUR, "sum"),
        total_distance=(COL_TD, "sum"),
        running=(COL_RUN, "sum"),
        sprint=(COL_SPR, "sum"),
        high_sprint=(COL_HSPR, "sum"),
        max_speed=(COL_MAX, "max"),
    )

    dur_min = g["duration"].replace(0, np.nan)
    g["td_per_min"] = g["total_distance"] / dur_min
    g["run_per_min"] = g["running"] / dur_min
    g["spr_per_min"] = g["sprint"] / dur_min
    g["hspr_per_min"] = g["high_sprint"] / dur_min

    g = g.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return g


# -----------------------------
# Plotly charts (gesorteerd op ABS, niet per/min)
# -----------------------------
def plot_td_bar(df: pd.DataFrame, title: str):
    if df.empty:
        st.info("Geen data voor grafiek.")
        return
    dff = df.sort_values(COL_TD, ascending=False).copy()
    fig = go.Figure(
        data=[
            go.Bar(
                x=dff[COL_PLAYER],
                y=dff[COL_TD],
                marker=dict(color="#FF0033"),
                name="TD",
            )
        ]
    )
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10), title=title, showlegend=False)
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


def plot_sprint_vs_high(df: pd.DataFrame, title: str):
    if df.empty:
        st.info("Geen data voor grafiek.")
        return
    dff = df.sort_values(COL_SPR, ascending=False).copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=dff[COL_PLAYER], y=dff[COL_SPR], name="sprint", marker=dict(color="rgba(255,0,51,0.85)")))
    fig.add_trace(
        go.Bar(x=dff[COL_PLAYER], y=dff[COL_HSPR], name="high_sprint", marker=dict(color="rgba(255,0,51,0.45)"))
    )
    fig.update_layout(
        barmode="group",
        height=330,
        margin=dict(l=10, r=10, t=40, b=10),
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Tables (5 in one row)
# -----------------------------
def render_tables_row(df_phase: pd.DataFrame):
    if df_phase.empty:
        st.info("Geen data voor tabellen.")
        return

    base = df_phase[
        [
            COL_PLAYER,
            COL_TD,
            COL_RUN,
            COL_SPR,
            COL_HSPR,
            COL_MAX,
            "td_per_min",
            "run_per_min",
            "spr_per_min",
            "hspr_per_min",
        ]
    ].copy()

    def _make_tbl(abs_col: str, per_col: Optional[str], label_abs: str) -> Tuple[pd.DataFrame, str, Optional[str]]:
        cols = [COL_PLAYER, abs_col]
        if per_col:
            cols.append(per_col)
        out = base[cols].copy()
        rename = {COL_PLAYER: LABEL_PLAYER, abs_col: label_abs}
        if per_col:
            rename[per_col] = LABEL_PERMIN
        out = out.rename(columns=rename)
        return out, label_abs, (LABEL_PERMIN if per_col else None)

    # per tabel sort op EIGEN /min (of ABS voor Max Speed)
    t1, abs1, per1 = _make_tbl(COL_TD, "td_per_min", LABEL_TD)
    t1 = t1.sort_values(LABEL_PERMIN, ascending=False)

    t2, abs2, per2 = _make_tbl(COL_RUN, "run_per_min", LABEL_RUN)
    t2 = t2.sort_values(LABEL_PERMIN, ascending=False)

    t3, abs3, per3 = _make_tbl(COL_SPR, "spr_per_min", LABEL_SPR)
    t3 = t3.sort_values(LABEL_PERMIN, ascending=False)

    t4, abs4, per4 = _make_tbl(COL_HSPR, "hspr_per_min", LABEL_HSPR)
    t4 = t4.sort_values(LABEL_PERMIN, ascending=False)

    t5, abs5, per5 = _make_tbl(COL_MAX, None, LABEL_MAX)
    t5 = t5.sort_values(LABEL_MAX, ascending=False)

    tables = [(t1, abs1, per1), (t2, abs2, per2), (t3, abs3, per3), (t4, abs4, per4), (t5, abs5, per5)]

    cols = st.columns(5, gap="large")
    for i, (tbl, abs_label, per_label) in enumerate(tables):
        with cols[i]:
            sty = _style_table(tbl, abs_col=abs_label, per_min_col=per_label)
            st.dataframe(
                sty,
                use_container_width=True,
                hide_index=True,
                height=_calc_height(len(tbl)),
            )


# -----------------------------
# Header UI
# -----------------------------
def render_match_header(match_row: pd.Series):
    match_date: date = match_row["match_date"]
    opponent: str = str(match_row.get("opponent") or "").strip()
    fixture: str = str(match_row.get("fixture") or "").strip()
    home_away: str = str(match_row.get("home_away") or "").strip()  # "Home"/"Away"
    match_type: str = str(match_row.get("match_type") or "").strip()
    season: str = str(match_row.get("season") or "").strip()

    gf = match_row.get("goals_for", None)
    ga = match_row.get("goals_against", None)

    try:
        gf_i = int(gf) if pd.notna(gf) else None
    except Exception:
        gf_i = None
    try:
        ga_i = int(ga) if pd.notna(ga) else None
    except Exception:
        ga_i = None

    score_txt = f"{gf_i} - {ga_i}" if (gf_i is not None and ga_i is not None) else "-"

    is_away = home_away.lower().startswith("a")

    # Away: opponent links, MVV rechts. Home: MVV links, opponent rechts.
    left_team = opponent if is_away else MVV_TEAM_NAME
    right_team = MVV_TEAM_NAME if is_away else opponent

    left_logo = _logo_path_for_team(left_team)
    right_logo = _logo_path_for_team(right_team)

    # Midden titel (fixture als die er is, anders fallback)
    title_line = fixture
    if not title_line:
        title_line = f"{left_team} - {right_team}"

    # 3 kolommen (logo - info - logo)
    c1, c2, c3 = st.columns([1.2, 2.2, 1.2], vertical_alignment="center")

    with c1:
        if left_logo and left_logo.exists():
            st.image(_read_image_bytes(left_logo), width=LOGO_W)
        st.markdown(
            f"<div style='text-align:center; font-weight:700; opacity:.9; margin-top:10px;'>{left_team}</div>",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div style="text-align:center;">
              <div style="opacity:.85; font-weight:650; font-size:14px; margin-bottom:6px;">{match_date.isoformat()}</div>
              <div style="font-weight:850; font-size:28px; margin-bottom:10px;">{title_line}</div>
              <div style="font-weight:900; font-size:52px; line-height:1; margin-bottom:14px;">{score_txt}</div>

              <div style="display:flex; justify-content:center; gap:10px; flex-wrap:wrap; margin-top:6px;">
                <span style="padding:6px 12px; border-radius:999px; border:1px solid rgba(255,255,255,0.10); background:rgba(255,255,255,0.04); font-weight:650;">{home_away}</span>
                <span style="padding:6px 12px; border-radius:999px; border:1px solid rgba(255,255,255,0.10); background:rgba(255,255,255,0.04); font-weight:650;">{match_type}</span>
                <span style="padding:6px 12px; border-radius:999px; border:1px solid rgba(255,255,255,0.10); background:rgba(255,255,255,0.04); font-weight:650;">{season}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        if right_logo and right_logo.exists():
            st.image(_read_image_bytes(right_logo), width=LOGO_W)
        st.markdown(
            f"<div style='text-align:center; font-weight:700; opacity:.9; margin-top:10px;'>{right_team}</div>",
            unsafe_allow_html=True,
        )


# -----------------------------
# Debug: waarom geen matches?
# -----------------------------
def debug_probe_matches(sb):
    try:
        u = sb.auth.get_user()
    except Exception as e:
        u = f"auth.get_user() error: {e}"

    st.write("sb.auth.get_user():", u)

    try:
        probe = sb.table("matches").select("match_id,match_date,opponent").limit(5).execute()
        st.write("probe error:", getattr(probe, "error", None))
        st.write("probe rows:", probe.data)
    except Exception as e:
        st.write("probe exception:", e)


# -----------------------------
# Main
# -----------------------------
def main():
    require_auth()

    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    _ = get_profile(sb)

    st.title("Match Reports")

    debug = st.toggle("Debug", value=False)
    if debug:
        debug_probe_matches(sb)

    matches_df = fetch_matches_rows(limit=1000)

    if debug:
        st.write("matches_df rows:", int(len(matches_df)))
        if not matches_df.empty:
            st.write("matches_df columns:", list(matches_df.columns))
            st.dataframe(matches_df.head(10), use_container_width=True)

    if matches_df.empty:
        st.info("Geen matches gevonden.")
        st.stop()

    # Opponent dropdown (A->Z)
    opponents = sorted(
        [o for o in matches_df["opponent"].dropna().astype(str).unique().tolist() if o.strip()],
        key=lambda x: x.lower(),
    )
    top_l, top_r = st.columns([1.2, 2.0])

    with top_l:
        sel_opp = st.selectbox("Opponent", options=opponents, index=0, key="mr_opp")

    df_opp = matches_df[matches_df["opponent"].astype(str) == str(sel_opp)].copy()
    df_opp = df_opp.sort_values("match_date", ascending=False)

    def _ha_tag(x: str) -> str:
        if not isinstance(x, str):
            return ""
        x2 = x.strip().lower()
        return "A" if x2.startswith("a") else "H"

    date_options: List[str] = []
    for _, r in df_opp.iterrows():
        ha = _ha_tag(str(r.get("home_away") or ""))
        date_options.append(f"{r['match_date'].isoformat()} ({ha})")

    with top_r:
        sel_date_label = st.selectbox("Date", options=date_options, index=0, key="mr_date")

    sel_date = date.fromisoformat(sel_date_label.split(" ")[0])

    # als er meerdere matches op 1 datum staan (zeldzaam): pak eerste
    match_row = df_opp[df_opp["match_date"] == sel_date].iloc[0]

    render_match_header(match_row)

    st.divider()

    # Phase radio
    phase = st.radio("Tables", [EVENT_FULL, EVENT_FIRST, EVENT_SECOND], horizontal=True, key="mr_phase")

    df_events = fetch_match_events_for_date(sel_date)
    if debug:
        st.write("events rows:", int(len(df_events)))
        if not df_events.empty:
            st.dataframe(df_events.head(10), use_container_width=True)

    if df_events.empty:
        st.info("Geen match events gevonden in v_gps_match_events voor deze datum.")
        st.stop()

    df_phase = build_phase_df(df_events, phase)

    # Charts + tables op 1 pagina
    left, right = st.columns(2)
    with left:
        plot_td_bar(df_phase, title=f"Total Distance ({phase})")
    with right:
        plot_sprint_vs_high(df_phase, title=f"Sprint vs High Sprint ({phase})")

    st.markdown("## Tables")
    render_tables_row(df_phase)


if __name__ == "__main__":
    main()
