from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Config / constants
# -----------------------------
CHART_H = 420

ASRM_PARAMS = [
    ("Muscle soreness", "muscle_soreness"),
    ("Fatigue", "fatigue"),
    ("Sleep quality", "sleep_quality"),
    ("Stress", "stress"),
    ("Mood", "mood"),
]

RPE_PARAMS = [
    ("RPE (weighted avg)", "avg_rpe"),
    ("RPE Load (sum dur*rpe)", "rpe_load"),
    ("Total duration (min)", "duration_min"),
]

ZONE_GREEN_MAX = 4.5
ZONE_ORANGE_MAX = 7.5

ASRM_RED_THRESHOLD = 7.5
RPE_RED_THRESHOLD = 7.5

MVV_COLORS = {
    "primary": "#C8102E",
    "light": "#E8213F",
    "dark": "#8B0A1F",
    "background": "#0D0E13",
    "card": "rgba(255, 255, 255, 0.04)",
    "text_primary": "#F0F0F0",
    "text_muted": "rgba(240, 240, 240, 0.45)",
    "grid": "rgba(255, 255, 255, 0.09)",
    "zone_green": "rgba(0, 200, 0, 0.12)",
    "zone_orange": "rgba(255, 165, 0, 0.14)",
    "zone_red": "rgba(255, 0, 0, 0.14)",
}


def _df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _coerce_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _bool_mask(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=bool)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def iso_week_start_end(year: int, week: int) -> Tuple[date, date]:
    d0 = date.fromisocalendar(year, week, 1)
    d1 = d0 + timedelta(days=6)
    return d0, d1


def add_zone_background(fig: go.Figure, y_min: float = 0, y_max: float = 10) -> None:
    zones = [
        (0, ZONE_GREEN_MAX, MVV_COLORS["zone_green"]),
        (ZONE_GREEN_MAX, ZONE_ORANGE_MAX, MVV_COLORS["zone_orange"]),
        (ZONE_ORANGE_MAX, y_max, MVV_COLORS["zone_red"]),
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
    fig.update_yaxes(range=[y_min, y_max], tick0=0, dtick=1)


# -----------------------------
# Chart helpers
# -----------------------------
def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _blend_hex(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return _rgb_to_hex((r, g, b))


def _seq_colors(n: int, start_hex: str, end_hex: str) -> list[str]:
    if n <= 1:
        return [start_hex]
    return [_blend_hex(start_hex, end_hex, i / (n - 1)) for i in range(n)]


def _zone_bar_colors(values: pd.Series) -> list[str]:
    colors = []
    for v in values:
        if pd.isna(v):
            colors.append(MVV_COLORS["primary"])
        elif float(v) < ZONE_GREEN_MAX:
            colors.append("#00C46A")
        elif float(v) < ZONE_ORANGE_MAX:
            colors.append("#F5A623")
        else:
            colors.append("#E8213F")
    return colors


def _fmt_bar_label(v: float) -> str:
    if pd.isna(v):
        return ""
    if abs(float(v) - round(float(v))) < 1e-9:
        return str(int(round(float(v))))
    return f"{float(v):.1f}"


# -----------------------------
# MVV Styled Charts
# -----------------------------
def create_mvv_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    show_zones: bool = False,
    y_range: Optional[Tuple[float, float]] = (0, 10),
) -> go.Figure:
    plot_df = df.copy()

    if plot_df.empty:
        return go.Figure()

    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col, y_col]).copy()

    if plot_df.empty:
        return go.Figure()

    n_bars = len(plot_df)
    dynamic_height = max(460, min(760, 360 + (n_bars * 22)))

    if show_zones:
        bar_colors = _zone_bar_colors(plot_df[y_col])
    else:
        bar_colors = _seq_colors(n_bars, MVV_COLORS["primary"], MVV_COLORS["light"])

    value_labels = [_fmt_bar_label(v) for v in plot_df[y_col]]

    fig = go.Figure()

    has_std = "std" in plot_df.columns and not plot_df["std"].isna().all()

    bar_kwargs = dict(
        x=plot_df[x_col],
        y=plot_df[y_col],
        text=value_labels,
        textposition="outside",
        textfont=dict(
            family="DM Sans",
            size=12,
            color=MVV_COLORS["text_primary"],
        ),
        cliponaxis=False,
        marker=dict(
            color=bar_colors,
            line=dict(color="rgba(255,255,255,0.18)", width=1.5),
        ),
        opacity=0.96,
        hovertemplate="<b>%{x}</b><br>Waarde: <b>%{y:.1f}</b><extra></extra>",
    )

    if has_std:
        bar_kwargs["error_y"] = dict(
            type="data",
            array=plot_df["std"],
            color="rgba(255,255,255,0.45)",
            thickness=1.6,
            width=4,
        )

    fig.add_trace(go.Bar(**bar_kwargs))

    if show_zones:
        zone_top = y_range[1] if y_range is not None else max(10, float(plot_df[y_col].max()) + 1)
        zones = [
            (0, ZONE_GREEN_MAX, "rgba(0, 196, 106, 0.10)"),
            (ZONE_GREEN_MAX, ZONE_ORANGE_MAX, "rgba(245, 166, 35, 0.11)"),
            (ZONE_ORANGE_MAX, zone_top, "rgba(232, 33, 63, 0.11)"),
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

    mean_val = float(plot_df[y_col].mean())
    fig.add_hline(
        y=mean_val,
        line_width=1.5,
        line_dash="dash",
        line_color="rgba(255,255,255,0.45)",
        annotation_text=f"Team avg: {mean_val:.1f}",
        annotation_position="top left",
        annotation_font=dict(size=11, color="rgba(255,255,255,0.70)"),
    )

    yaxis_cfg = dict(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(255,255,255,0.07)",
        zeroline=False,
        tickfont=dict(color=MVV_COLORS["text_primary"], size=11),
        title_font=dict(color=MVV_COLORS["text_primary"]),
        fixedrange=True,
    )

    if y_range is not None:
        yaxis_cfg["range"] = list(y_range)
    else:
        y_max = float(plot_df[y_col].max()) if not plot_df.empty else 10
        yaxis_cfg["range"] = [0, y_max * 1.22 if y_max > 0 else 10]

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                family="DM Sans",
                size=20,
                color=MVV_COLORS["text_primary"],
            ),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.015)",
        font=dict(
            family="DM Sans",
            size=12,
            color=MVV_COLORS["text_primary"],
        ),
        xaxis=dict(
            showgrid=False,
            tickangle=-35,
            tickfont=dict(color=MVV_COLORS["text_primary"], size=11),
            automargin=True,
            fixedrange=True,
        ),
        yaxis=yaxis_cfg,
        bargap=0.22,
        margin=dict(l=40, r=20, t=80, b=130),
        showlegend=False,
        height=dynamic_height,
        hovermode="closest",
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )

    return fig


# -----------------------------
# Caching layer (common calls)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60)
def fetch_active_players_cached(sb_url_key: str, _sb, ttl_salt: str = "") -> pd.DataFrame:
    attempts = [
        lambda: (
            _sb.table("players")
            .select("player_id,full_name,is_active")
            .eq("is_active", True)
            .order("full_name")
            .execute()
        ),
        lambda: (
            _sb.table("players")
            .select("player_id,full_name,is_active")
            .order("full_name")
            .execute()
        ),
        lambda: (
            _sb.table("players")
            .select("player_id,full_name")
            .order("full_name")
            .execute()
        ),
        lambda: (
            _sb.table("players")
            .select("*")
            .execute()
        ),
    ]

    last_err = None
    result_df = None

    for attempt in attempts:
        try:
            res = attempt()
            rows = res.data or []
            result_df = _df(rows)
            break
        except Exception as e:
            last_err = e

    if result_df is None:
        raise RuntimeError(
            "Kon spelers niet ophalen uit Supabase. Controleer de tabel 'players', "
            "de kolommen en je RLS/SELECT policy."
        ) from last_err

    if result_df.empty:
        return pd.DataFrame(columns=["player_id", "full_name"])

    id_col = _first_existing_col(result_df, ["player_id", "id", "uuid"])
    name_col = _first_existing_col(result_df, ["full_name", "name", "player_name"])

    if id_col is None or name_col is None:
        raise RuntimeError(f"Onverwachte kolommen in players-tabel: {list(result_df.columns)}")

    df = result_df.rename(columns={id_col: "player_id", name_col: "full_name"}).copy()

    if "is_active" in df.columns:
        active_mask = _bool_mask(df["is_active"])
        if active_mask.any():
            df = df.loc[active_mask].copy()

    df["player_id"] = df["player_id"].astype(str)
    df["full_name"] = df["full_name"].astype(str)

    df = (
        df[["player_id", "full_name"]]
        .dropna(subset=["player_id", "full_name"])
        .drop_duplicates()
        .sort_values("full_name")
        .reset_index(drop=True)
    )
    return df


@st.cache_data(show_spinner=False, ttl=60)
def fetch_asrm_range_cached(sb_url_key: str, _sb, d0_iso: str, d1_iso: str) -> pd.DataFrame:
    rows = (
        _sb.table("asrm_entries")
        .select("player_id,entry_date,muscle_soreness,fatigue,sleep_quality,stress,mood")
        .gte("entry_date", d0_iso)
        .lte("entry_date", d1_iso)
        .execute()
        .data
        or []
    )
    df = _df(rows)
    if df.empty:
        return df
    df["player_id"] = df["player_id"].astype(str)
    df["entry_date"] = _coerce_date(df["entry_date"])
    return df


@st.cache_data(show_spinner=False, ttl=60)
def fetch_asrm_date_cached(sb_url_key: str, _sb, d_iso: str) -> pd.DataFrame:
    rows = (
        _sb.table("asrm_entries")
        .select("player_id,entry_date,muscle_soreness,fatigue,sleep_quality,stress,mood")
        .eq("entry_date", d_iso)
        .execute()
        .data
        or []
    )
    df = _df(rows)
    if df.empty:
        return df
    df["player_id"] = df["player_id"].astype(str)
    df["entry_date"] = _coerce_date(df["entry_date"])
    return df


@st.cache_data(show_spinner=False, ttl=60)
def fetch_rpe_headers_range_cached(sb_url_key: str, _sb, d0_iso: str, d1_iso: str) -> pd.DataFrame:
    rows = (
        _sb.table("rpe_entries")
        .select("id,player_id,entry_date")
        .gte("entry_date", d0_iso)
        .lte("entry_date", d1_iso)
        .execute()
        .data
        or []
    )
    df = _df(rows)
    if df.empty:
        return df
    df["id"] = df["id"].astype(str)
    df["player_id"] = df["player_id"].astype(str)
    df["entry_date"] = _coerce_date(df["entry_date"])
    return df


@st.cache_data(show_spinner=False, ttl=60)
def fetch_rpe_headers_date_cached(sb_url_key: str, _sb, d_iso: str) -> pd.DataFrame:
    rows = (
        _sb.table("rpe_entries")
        .select("id,player_id,entry_date")
        .eq("entry_date", d_iso)
        .execute()
        .data
        or []
    )
    df = _df(rows)
    if df.empty:
        return df
    df["id"] = df["id"].astype(str)
    df["player_id"] = df["player_id"].astype(str)
    df["entry_date"] = _coerce_date(df["entry_date"])
    return df


@st.cache_data(show_spinner=False, ttl=60)
def fetch_rpe_sessions_for_ids_cached(
    sb_url_key: str,
    _sb,
    entry_ids_tuple: Tuple[str, ...],
) -> pd.DataFrame:
    entry_ids = list(entry_ids_tuple)
    if not entry_ids:
        return pd.DataFrame()

    rows = (
        _sb.table("rpe_sessions")
        .select("rpe_entry_id,session_index,duration_min,rpe")
        .in_("rpe_entry_id", entry_ids)
        .execute()
        .data
        or []
    )
    df = _df(rows)
    if df.empty:
        return df

    df["rpe_entry_id"] = df["rpe_entry_id"].astype(str)
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce").fillna(0.0)
    df["rpe"] = pd.to_numeric(df["rpe"], errors="coerce").fillna(0.0)
    df["load"] = df["duration_min"] * df["rpe"]
    return df


# -----------------------------
# Injury fetch
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60)
def fetch_rpe_injuries_range_cached(sb_url_key: str, _sb, d0_iso: str, d1_iso: str) -> pd.DataFrame:
    rows = (
        _sb.table("rpe_entries")
        .select("id,player_id,entry_date,injury,injury_type,injury_pain,attachment_url,notes,created_at,updated_at")
        .gte("entry_date", d0_iso)
        .lte("entry_date", d1_iso)
        .execute()
        .data
        or []
    )
    df = _df(rows)
    if df.empty:
        return df

    df["id"] = df["id"].astype(str)
    df["player_id"] = df["player_id"].astype(str)
    df["entry_date"] = _coerce_date(df["entry_date"])
    df["injury_pain"] = pd.to_numeric(df["injury_pain"], errors="coerce")
    if "injury" in df.columns:
        df["injury"] = _bool_mask(df["injury"])
    else:
        df["injury"] = False
    return df


# -----------------------------
# RPE computations
# -----------------------------
def build_rpe_player_daily(sb_url_key: str, sb, headers_df: pd.DataFrame) -> pd.DataFrame:
    if headers_df is None or headers_df.empty:
        return pd.DataFrame(columns=["entry_date", "player_id", "avg_rpe", "rpe_load", "duration_min"])

    entry_ids = headers_df["id"].astype(str).tolist()
    sess = fetch_rpe_sessions_for_ids_cached(sb_url_key, sb, tuple(entry_ids))
    if sess.empty:
        return pd.DataFrame(columns=["entry_date", "player_id", "avg_rpe", "rpe_load", "duration_min"])

    merged = sess.merge(
        headers_df[["id", "player_id", "entry_date"]],
        left_on="rpe_entry_id",
        right_on="id",
        how="left",
    ).dropna(subset=["player_id", "entry_date"])

    out = (
        merged.groupby(["entry_date", "player_id"], as_index=False)
        .agg(
            rpe_load=("load", "sum"),
            duration_min=("duration_min", "sum"),
            mean_rpe=("rpe", "mean"),
        )
        .reset_index(drop=True)
    )

    out["avg_rpe"] = out["mean_rpe"]
    dur_mask = out["duration_min"] > 0
    out.loc[dur_mask, "avg_rpe"] = out.loc[dur_mask, "rpe_load"] / out.loc[dur_mask, "duration_min"]

    out = out.drop(columns=["mean_rpe"])
    out["entry_date"] = _coerce_date(out["entry_date"])
    out["player_id"] = out["player_id"].astype(str)

    return out[["entry_date", "player_id", "avg_rpe", "rpe_load", "duration_min"]]


def agg_week_player_mean_std(df_long: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df_long is None or df_long.empty:
        return pd.DataFrame(columns=["player_id", "mean", "std", "n"])

    tmp = df_long.copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=["player_id", value_col])

    out = (
        tmp.groupby("player_id", as_index=False)
        .agg(mean=(value_col, "mean"), std=(value_col, "std"), n=(value_col, "count"))
    )
    out["std"] = out["std"].fillna(0.0)
    out["player_id"] = out["player_id"].astype(str)
    return out


# -----------------------------
# Plotting helpers
# -----------------------------
def plot_day_bars(df: pd.DataFrame, x_col: str, y_col: str, y_title: str, zone_0_10: bool) -> None:
    if df.empty:
        st.info("Geen data voor deze selectie.")
        return

    df = df.sort_values(y_col, ascending=False)

    fig = create_mvv_bar_chart(
        df=df,
        x_col=x_col,
        y_col=y_col,
        title=y_title,
        show_zones=zone_0_10,
        y_range=(0, 10) if zone_0_10 else None,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def plot_week_player_mean_std_bars(
    df_stats: pd.DataFrame,
    player_name_col: str,
    mean_col: str = "mean",
    std_col: str = "std",
    y_title: str = "",
    zone_0_10: bool = False,
) -> None:
    if df_stats.empty:
        st.info("Geen data voor deze week/selectie.")
        return

    df_stats = df_stats.sort_values(mean_col, ascending=False)

    plot_df = df_stats.copy()
    if std_col in plot_df.columns and std_col != "std":
        plot_df["std"] = pd.to_numeric(plot_df[std_col], errors="coerce").fillna(0.0)

    fig = create_mvv_bar_chart(
        df=plot_df,
        x_col=player_name_col,
        y_col=mean_col,
        title=y_title,
        show_zones=zone_0_10,
        y_range=(0, 10) if zone_0_10 else None,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
