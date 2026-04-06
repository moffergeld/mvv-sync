# pages/02_Match_Reports.py
# ============================================================
# Match Reports (Streamlit) - Redesigned
# - Opponent dropdown (A-Z) + Date dropdown (YYYY-MM-DD (H/A))
# - Header: logos zelfde grootte, horizontaal gecentreerd, wisselen bij Away:
#     Away  -> opponent links, MVV rechts
#     Home  -> MVV links, opponent rechts
# - Data uit:
#     public.matches
#     public.v_gps_match_events
# - Full match:
#     gebruikt eerst Summary
#     valt anders terug op First Half + Second Half
# - Koppeling GPS-data:
#     via match_id i.p.v. alleen datum
# ============================================================

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Project root / imports
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roles import get_sb, require_auth, get_profile  # noqa: E402

# -----------------------------
# Config
# -----------------------------
LOGO_DIR = ROOT / "Assets" / "Afbeeldingen" / "Team_Logos"
MVV_TEAM_NAME = "MVV Maastricht"

EVENT_FULL = "Full match"
EVENT_FIRST = "First Half"
EVENT_SECOND = "Second Half"

COL_PLAYER = "player_name"
COL_TD = "total_distance"
COL_RUN = "running"
COL_SPR = "sprint"
COL_HSPR = "high_sprint"
COL_MAX = "max_speed"
COL_DUR = "duration"

LABEL_PLAYER = "Player"
LABEL_TD = "TD"
LABEL_RUN = "14.4–19.7"
LABEL_SPR = "19.8–25.1"
LABEL_HSPR = "25.2+"
LABEL_MAX = "Max Speed"
LABEL_PERMIN = "/min"

ROW_HEIGHT = 32
HEADER_HEIGHT = 34
PAD_HEIGHT = 22
LOGO_W = 170

MVV_RED = "#C8102E"
MVV_RED_LIGHT = "#E8213F"
MVV_RED_DARK = "#8B0A1F"
MVV_TEXT = "#F5F7FB"
MVV_TEXT_SOFT = "rgba(245,247,251,0.76)"
MVV_GRID = "rgba(255,255,255,0.08)"

# -----------------------------
# Global CSS
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stDataFrame"] div {
        overflow: visible !important;
    }

    .mr-kicker {
        font-size: 11px;
        letter-spacing: .24em;
        text-transform: uppercase;
        font-weight: 800;
        color: rgba(255,255,255,.72);
        margin-bottom: 6px;
    }

    .mr-sub {
        color: rgba(255,255,255,.80);
        margin-bottom: 14px;
        line-height: 1.5;
    }

    .mr-hero {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 18px 20px 14px 20px;
        margin-bottom: 18px;
        background:
            radial-gradient(circle at top left, rgba(200,16,46,0.20) 0%, rgba(200,16,46,0.07) 24%, rgba(0,0,0,0) 58%),
            linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 45%, rgba(255,255,255,0.015) 100%);
        box-shadow:
            0 16px 40px rgba(0,0,0,0.22),
            inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .mr-panel {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 14px 16px;
        background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.018) 100%);
        box-shadow: 0 10px 24px rgba(0,0,0,.14);
    }

    .mr-chip-row {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 8px;
    }

    .mr-chip {
        padding: 7px 12px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.04);
        font-weight: 700;
        color: white;
        font-size: 12px;
    }

    .mr-score {
        font-weight: 900;
        font-size: 54px;
        line-height: 1;
        margin-bottom: 12px;
        color: #FFFFFF;
    }

    .mr-title {
        font-weight: 850;
        font-size: 30px;
        margin-bottom: 10px;
        color: #FFFFFF;
    }

    .mr-date {
        opacity: .84;
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 6px;
        color: rgba(255,255,255,.82);
    }

    .mr-kpi-card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 14px 16px;
        background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.018) 100%);
        box-shadow: 0 10px 24px rgba(0,0,0,.14);
        min-height: 96px;
    }

    .mr-kpi-label {
        font-size: 11px;
        letter-spacing: .22em;
        text-transform: uppercase;
        font-weight: 800;
        color: rgba(255,255,255,.62);
        margin-bottom: 10px;
    }

    .mr-kpi-value {
        font-size: 20px;
        line-height: 1.1;
        font-weight: 850;
        color: #FFFFFF;
    }

    div[data-testid="stTabs"] button {
        border-radius: 999px !important;
        padding: 10px 16px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        background: rgba(255,255,255,0.02) !important;
        color: rgba(255,255,255,0.80) !important;
        font-weight: 700 !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: linear-gradient(180deg, rgba(200,16,46,0.30) 0%, rgba(200,16,46,0.16) 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(232,33,63,0.40) !important;
        box-shadow: 0 8px 18px rgba(200,16,46,0.18);
    }

    .mr-section-label {
        font-size: 11px;
        letter-spacing: .22em;
        font-weight: 800;
        text-transform: uppercase;
        color: rgba(255,255,255,.72);
        margin-bottom: 8px;
    }
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


def _norm_text(x: Any) -> str:
    return str(x or "").strip().lower()


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

    if key2 in _LOGO_INDEX:
        return _LOGO_INDEX[key2]

    for k, p in _LOGO_INDEX.items():
        if k == key2:
            return p

    for k, p in _LOGO_INDEX.items():
        if key2 in k or k in key2:
            return p

    return None


def _fmt_int0(x: Any) -> str:
    if pd.isna(x):
        return ""
    try:
        return f"{int(round(float(x), 0)):,}".replace(",", " ")
    except Exception:
        return ""


def _fmt_min2(x: Any) -> str:
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.2f}"
    except Exception:
        return ""


def _fmt_max_speed2(x: Any) -> str:
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


def _kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="mr-kpi-card">
            <div class="mr-kpi-label">{label}</div>
            <div class="mr-kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _ha_tag(x: str) -> str:
    x2 = _norm_text(x)
    return "A" if x2.startswith("a") else "H"


def _series_rank_colors(n: int) -> List[str]:
    palette = [
        "#FF335C",
        "#F42B56",
        "#E1224C",
        "#CB1A42",
        "#B7143A",
        "#A11134",
        "#8B0F2E",
    ]
    if n <= len(palette):
        return palette[:n]
    return [palette[min(i, len(palette) - 1)] for i in range(n)]


def _base_plot_layout(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=20, color=MVV_TEXT)),
        height=430,
        margin=dict(l=20, r=20, t=60, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.015)",
        font=dict(color=MVV_TEXT, size=12),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
        ),
        bargap=0.18,
    )
    fig.update_xaxes(
        tickangle=-35,
        showgrid=False,
        tickfont=dict(size=11),
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor=MVV_GRID,
        zeroline=False,
        tickfont=dict(size=11),
    )


# -----------------------------
# Supabase fetch
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60)
def fetch_matches_rows(limit: int = 1000) -> pd.DataFrame:
    sb = get_sb()
    res = (
        sb.table("matches")
        .select(
            "match_id,match_date,fixture,home_away,opponent,match_type,season,goals_for,goals_against"
        )
        .order("match_date", desc=True)
        .limit(limit)
        .execute()
    )

    df = _df_from_rows(res.data or [])
    if df.empty:
        return df

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.date
    df = df.dropna(subset=["match_id", "match_date"]).copy()

    df["date_label"] = df.apply(
        lambda r: f"{r['match_date'].isoformat()} ({_ha_tag(r.get('home_away'))})",
        axis=1,
    )

    return df


@st.cache_data(show_spinner=False, ttl=60)
def fetch_match_events_for_match(match_id: int) -> pd.DataFrame:
    sb = get_sb()
    res = (
        sb.table("v_gps_match_events")
        .select(
            "gps_id,match_id,player_id,player_name,datum,type,event,duration,total_distance,running,sprint,high_sprint,max_speed"
        )
        .eq("match_id", match_id)
        .execute()
    )

    df = _df_from_rows(res.data or [])
    if df.empty:
        return df

    df["datum"] = pd.to_datetime(df["datum"], errors="coerce").dt.date

    for c in [COL_DUR, COL_TD, COL_RUN, COL_SPR, COL_HSPR, COL_MAX]:
        if c in df.columns:
            df[c] = _safe_num(df[c]).fillna(0.0)

    if "event" in df.columns:
        df["event"] = df["event"].astype(str).str.strip()

    return df


# -----------------------------
# Data prep
# -----------------------------
def build_phase_df(df_events: pd.DataFrame, phase: str) -> pd.DataFrame:
    if df_events.empty:
        return df_events

    dff = df_events.copy()
    dff["event_norm"] = dff["event"].astype(str).str.strip().str.lower()

    if phase == EVENT_FULL:
        summary_df = dff[dff["event_norm"].eq("summary")].copy()
        if not summary_df.empty:
            dff = summary_df
        else:
            dff = dff[dff["event_norm"].isin(["first half", "second half"])].copy()

    elif phase == EVENT_FIRST:
        dff = dff[dff["event_norm"].eq("first half")].copy()

    elif phase == EVENT_SECOND:
        dff = dff[dff["event_norm"].eq("second half")].copy()

    else:
        dff = dff[dff["event_norm"].isin(["summary", "first half", "second half"])].copy()

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
# KPI row
# -----------------------------
def render_kpi_row(df_phase: pd.DataFrame) -> None:
    if df_phase.empty:
        return

    n_players = len(df_phase)
    med_td = float(df_phase[COL_TD].median()) if COL_TD in df_phase.columns else 0.0
    med_spr = float(df_phase[COL_SPR].median()) if COL_SPR in df_phase.columns else 0.0
    peak_speed = float(df_phase[COL_MAX].max()) if COL_MAX in df_phase.columns else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi_card("Spelers", str(n_players))
    with c2:
        _kpi_card("Mediaan TD", f"{_fmt_int0(med_td)} m")
    with c3:
        _kpi_card("Mediaan Sprint", f"{_fmt_int0(med_spr)} m")
    with c4:
        _kpi_card("Peak Speed", f"{_fmt_max_speed2(peak_speed)}")


# -----------------------------
# Charts
# -----------------------------
def plot_td_bar(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        st.info("Geen data voor grafiek.")
        return

    dff = df.sort_values(COL_TD, ascending=False).reset_index(drop=True).copy()
    colors = _series_rank_colors(len(dff))
    median_td = float(dff[COL_TD].median())

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dff[COL_PLAYER],
            y=dff[COL_TD],
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.18)", width=1.0),
            ),
            text=[_fmt_int0(v) for v in dff[COL_TD]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>Total Distance: <b>%{y:,.0f} m</b><extra></extra>",
            name="Total Distance",
        )
    )

    fig.add_hline(
        y=median_td,
        line_width=1.5,
        line_dash="dash",
        line_color="rgba(255,255,255,0.45)",
        annotation_text=f"Mediaan: {_fmt_int0(median_td)} m",
        annotation_position="top left",
        annotation_font=dict(size=11, color="rgba(255,255,255,0.72)"),
    )

    _base_plot_layout(fig, title)
    fig.update_yaxes(title_text="Meters")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def plot_sprint_vs_high(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        st.info("Geen data voor grafiek.")
        return

    dff = df.sort_values(COL_SPR, ascending=False).reset_index(drop=True).copy()
    sprint_med = float(dff[COL_SPR].median())
    hs_med = float(dff[COL_HSPR].median())

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dff[COL_PLAYER],
            y=dff[COL_SPR],
            name="Sprint",
            marker=dict(
                color="rgba(232,33,63,0.92)",
                line=dict(color="rgba(255,255,255,0.14)", width=1),
            ),
            hovertemplate="<b>%{x}</b><br>Sprint: <b>%{y:,.0f} m</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=dff[COL_PLAYER],
            y=dff[COL_HSPR],
            name="High Sprint",
            marker=dict(
                color="rgba(255,110,130,0.60)",
                line=dict(color="rgba(255,255,255,0.12)", width=1),
            ),
            hovertemplate="<b>%{x}</b><br>High Sprint: <b>%{y:,.0f} m</b><extra></extra>",
        )
    )

    fig.add_hline(
        y=sprint_med,
        line_width=1.2,
        line_dash="dot",
        line_color="rgba(255,255,255,0.36)",
        annotation_text=f"Mediaan Sprint: {_fmt_int0(sprint_med)} m",
        annotation_position="top left",
        annotation_font=dict(size=10, color="rgba(255,255,255,0.68)"),
    )
    fig.add_hline(
        y=hs_med,
        line_width=1.2,
        line_dash="dash",
        line_color="rgba(255,180,190,0.40)",
        annotation_text=f"Mediaan High Sprint: {_fmt_int0(hs_med)} m",
        annotation_position="top right",
        annotation_font=dict(size=10, color="rgba(255,255,255,0.68)"),
    )

    _base_plot_layout(fig, title)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Meters")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


# -----------------------------
# Tables
# -----------------------------
def _style_table(df: pd.DataFrame, abs_col: str, per_min_col: Optional[str]) -> "pd.io.formats.style.Styler":
    dff = df.copy()

    abs_vals = _safe_num(dff[abs_col]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(abs_vals) >= 2:
        q25, q50, q75 = abs_vals.quantile([0.25, 0.50, 0.75]).tolist()
    elif len(abs_vals) == 1:
        q25 = q50 = q75 = float(abs_vals.iloc[0])
    else:
        q25 = q50 = q75 = 0.0

    def _bg_abs(s: pd.Series) -> List[str]:
        out: List[str] = []
        for v in _safe_num(s).tolist():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out.append("")
            else:
                out.append(f"background-color: {_percentile_color(float(v), q25, q50, q75)};")
        return out

    fmt: Dict[str, Any] = {}

    if abs_col == LABEL_MAX:
        fmt[abs_col] = _fmt_max_speed2
    else:
        fmt[abs_col] = _fmt_int0

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
            {
                "selector": "th",
                "props": [
                    ("text-align", "left"),
                    ("font-weight", "700"),
                    ("background-color", "rgba(255,255,255,0.04)"),
                    ("color", "#F5F7FB"),
                    ("border-color", "rgba(255,255,255,0.08)"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border-color", "rgba(255,255,255,0.06)"),
                    ("color", "#F5F7FB"),
                ],
            },
        ]
    )
    return sty


def render_tables_row(df_phase: pd.DataFrame) -> None:
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

        rename = {
            COL_PLAYER: LABEL_PLAYER,
            abs_col: label_abs,
        }
        if per_col:
            rename[per_col] = LABEL_PERMIN

        return out.rename(columns=rename), label_abs, (LABEL_PERMIN if per_col else None)

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

    tab_td, tab_run, tab_spr, tab_hspr, tab_max = st.tabs(
        ["TD", "Running", "Sprint", "High Sprint", "Max Speed"]
    )

    with tab_td:
        st.dataframe(
            _style_table(t1, abs_col=abs1, per_min_col=per1),
            use_container_width=True,
            hide_index=True,
            height=_calc_height(len(t1)),
        )
    with tab_run:
        st.dataframe(
            _style_table(t2, abs_col=abs2, per_min_col=per2),
            use_container_width=True,
            hide_index=True,
            height=_calc_height(len(t2)),
        )
    with tab_spr:
        st.dataframe(
            _style_table(t3, abs_col=abs3, per_min_col=per3),
            use_container_width=True,
            hide_index=True,
            height=_calc_height(len(t3)),
        )
    with tab_hspr:
        st.dataframe(
            _style_table(t4, abs_col=abs4, per_min_col=per4),
            use_container_width=True,
            hide_index=True,
            height=_calc_height(len(t4)),
        )
    with tab_max:
        st.dataframe(
            _style_table(t5, abs_col=abs5, per_min_col=per5),
            use_container_width=True,
            hide_index=True,
            height=_calc_height(len(t5)),
        )


# -----------------------------
# Header
# -----------------------------
def render_match_header(match_row: pd.Series) -> None:
    match_date: date = match_row["match_date"]
    opponent: str = str(match_row.get("opponent") or "").strip()
    fixture: str = str(match_row.get("fixture") or "").strip()
    home_away: str = str(match_row.get("home_away") or "").strip()
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

    is_away = _norm_text(home_away).startswith("a")

    if gf_i is not None and ga_i is not None:
        score_txt = f"{ga_i} - {gf_i}" if is_away else f"{gf_i} - {ga_i}"
    else:
        score_txt = "-"

    left_team = opponent if is_away else MVV_TEAM_NAME
    right_team = MVV_TEAM_NAME if is_away else opponent

    left_logo = _logo_path_for_team(left_team)
    right_logo = _logo_path_for_team(right_team)

    title_line = fixture or f"{left_team} - {right_team}"

    st.markdown('<div class="mr-hero">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.1, 2.2, 1.1], vertical_alignment="center")

    with c1:
        if left_logo and left_logo.exists():
            st.image(str(left_logo), width=LOGO_W)

    with c2:
        st.markdown(
            f"""
            <div style="text-align:center;">
              <div class="mr-date">{match_date.isoformat()}</div>
              <div class="mr-title">{title_line}</div>
              <div class="mr-score">{score_txt}</div>
              <div class="mr-chip-row">
                <span class="mr-chip">{home_away}</span>
                <span class="mr-chip">{match_type}</span>
                <span class="mr-chip">{season}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        if right_logo and right_logo.exists():
            st.image(str(right_logo), width=LOGO_W)

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    require_auth()

    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    _ = get_profile(sb)

    st.markdown('<div class="mr-kicker">Match Reports</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mr-sub">Professionele wedstrijdrapportage met matchheader, grafieken en per-speler tabellen op basis van GPS-match events.</div>',
        unsafe_allow_html=True,
    )

    matches_df = fetch_matches_rows(limit=1000)
    if matches_df.empty:
        st.info("Geen matches gevonden.")
        st.stop()

    opponents = sorted(
        [
            o
            for o in matches_df["opponent"].dropna().astype(str).unique().tolist()
            if o.strip()
        ],
        key=lambda x: x.lower(),
    )

    select_l, select_r = st.columns([1.2, 2.0])

    with select_l:
        sel_opp = st.selectbox("Opponent", options=opponents, index=0, key="mr_opp")

    df_opp = (
        matches_df[matches_df["opponent"].astype(str) == str(sel_opp)]
        .copy()
        .sort_values("match_date", ascending=False)
    )

    date_options = df_opp["date_label"].tolist()

    with select_r:
        sel_date_label = st.selectbox("Date", options=date_options, index=0, key="mr_date")

    match_row = df_opp[df_opp["date_label"] == sel_date_label].iloc[0]
    match_id = int(match_row["match_id"])

    render_match_header(match_row)

    st.markdown('<div class="mr-section-label">Fase</div>', unsafe_allow_html=True)
    phase = st.radio(
        "Fase",
        [EVENT_FULL, EVENT_FIRST, EVENT_SECOND],
        horizontal=True,
        key="mr_phase",
        label_visibility="collapsed",
    )

    df_events = fetch_match_events_for_match(match_id)
    if df_events.empty:
        st.info("Geen match events gevonden in v_gps_match_events voor deze match.")
        st.stop()

    df_phase = build_phase_df(df_events, phase)
    if df_phase.empty:
        st.info("Geen data voor deze fase.")
        st.stop()

    render_kpi_row(df_phase)

    chart_l, chart_r = st.columns(2)
    with chart_l:
        plot_td_bar(df_phase, title=f"Total Distance ({phase})")
    with chart_r:
        plot_sprint_vs_high(df_phase, title=f"Sprint vs High Sprint ({phase})")

    st.markdown("### Tables")
    render_tables_row(df_phase)


if __name__ == "__main__":
    main()
