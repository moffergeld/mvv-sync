# acwr_pages.py
# ============================================================
# ACWR-dashboard + threshold-planner + Targets vs Workload
#
# Data:
# - GPS-database (per sessie) met kolommen: 'Week', 'Year', 'Speler', 'Event' + load-parameters
# - Alleen Event == 'Summary' wordt gebruikt.
#
# FIX (jaarwisseling):
# - Gebruik Year + Week als leidend (week_key = Year*100 + Week)
# - Sorteer/bereken ACWR op week_key (bv. 202552, 202601)
# - Toon labels "2026-W01"
# - Datum wordt alleen gebruikt als fallback (als Year ontbreekt)
#
# NIEUW (thresholds opslaan per week -> Supabase):
# - Threshold planner kan ratio_low/ratio_high per (team, week_key, metric) opslaan
# - Deze opgeslagen thresholds worden gebruikt in Targets vs Workload
# - In Targets vs Workload: ondergrens wordt getoond als stippellijn (min target in % van max target)
#
# Vereist Supabase tabel:
#   public.acwr_week_thresholds
# Unique index:
#   (team, week_key, metric)
#
# Let op:
# - Deze module maakt zelf een Supabase client aan via st.secrets.
# - Na login moet je access_token in st.session_state["access_token"] staan (zoals in jouw app),
#   dan wordt sb.postgrest.auth(token) gezet zodat RLS policies voor authenticated werken.
# ============================================================

from __future__ import annotations

from datetime import date, timedelta, datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Supabase client (python package: supabase)
try:
    from supabase import create_client  # type: ignore
except Exception:
    create_client = None  # module blijft werken zonder Supabase, maar opslaan niet


# ------------------------------------------------------------
# CONSTANTEN / CONFIG
# ------------------------------------------------------------

# ACWR sweet-spot grenzen
SWEET_SPOT_LOW = 0.80
SWEET_SPOT_HIGH = 1.30

# GPS kolomnamen in jouw Database.xlsx
COL_WEEK = "Week"
COL_YEAR = "Year"    # <-- jij hebt "Year" gemaakt
COL_DATE = "Datum"   # fallback (optioneel)
COL_PLAYER = "Speler"
COL_EVENT = "Event"

# Supabase thresholds opslag
THRESH_TABLE = "acwr_week_thresholds"
DEFAULT_TEAM = "MVV"  # pas aan als je meerdere squads wil onderscheiden

# Kolommen die NIET als ACWR-parameter gebruikt mogen worden
EXCLUDE_METRICS = {"Max Speed", "Avg Speed", "Avg HR", "Max HR"}
EXCLUDE_SUFFIXES = ("/min",)

# Voorkeursmetrics als default-selectie
DEFAULT_PREF_METRICS = ["Total Distance", "Sprint", "High Sprint", "playerload2D"]


# ------------------------------------------------------------
# SUPABASE – INIT / HELPERS
# ------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_supabase_client():
    """
    Maak 1 client per Streamlit sessie.
    Vereist in st.secrets:
      SUPABASE_URL
      SUPABASE_ANON_KEY
    """
    if create_client is None:
        return None

    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_ANON_KEY"]
    except Exception:
        return None

    try:
        return create_client(url, key)
    except Exception:
        return None


def _sb_auth_if_possible(sb):
    """
    Zet PostgREST auth header op basis van token uit session_state.
    Dit is nodig als RLS policies op authenticated staan.
    """
    if sb is None:
        return
    token = st.session_state.get("access_token")
    if token:
        try:
            sb.postgrest.auth(token)
        except Exception:
            pass


@st.cache_data(show_spinner=False, ttl=60)
def sb_get_thresholds_cached(team: str, week_key: int) -> pd.DataFrame:
    """
    Cached wrapper (ttl 60s) – wordt invalidated door st.cache_data.clear() na upsert.
    """
    sb = _get_supabase_client()
    _sb_auth_if_possible(sb)

    cols = ["team", "week_key", "week_label", "metric", "ratio_low", "ratio_high", "updated_at", "created_by", "note"]
    if sb is None:
        return pd.DataFrame(columns=cols)

    try:
        resp = (
            sb.table(THRESH_TABLE)
              .select(",".join(cols))
              .eq("team", team)
              .eq("week_key", int(week_key))
              .execute()
        )
        rows = resp.data or []
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=cols)


def sb_upsert_thresholds(team: str, week_key: int, week_label: str, df_ratios: pd.DataFrame, note: str = "") -> Tuple[bool, str]:
    """
    Upsert thresholds per (team, week_key, metric).
    Vereist unique index/constraint op team, week_key, metric.
    """
    sb = _get_supabase_client()
    _sb_auth_if_possible(sb)

    if sb is None:
        return False, "Supabase client niet beschikbaar (secrets/package ontbreken)."

    user_email = st.session_state.get("user_email")

    payload = []
    for _, r in df_ratios.iterrows():
        payload.append({
            "team": team,
            "week_key": int(week_key),
            "week_label": str(week_label),
            "metric": str(r["metric"]),
            "ratio_low": float(r["ratio_low"]),
            "ratio_high": float(r["ratio_high"]),
            "note": note,
            "created_by": user_email,
        })

    try:
        sb.table(THRESH_TABLE).upsert(payload, on_conflict="team,week_key,metric").execute()
        st.cache_data.clear()  # refresh cached reads
        return True, "OK"
    except Exception as e:
        return False, f"Upsert faalde: {e}"


def ratios_from_threshold_df(
    df_thr: pd.DataFrame,
    metrics: List[str],
    fallback_low: float,
    fallback_high: float,
) -> Dict[str, Tuple[float, float]]:
    """
    Maak dict metric -> (ratio_low, ratio_high) met fallback als metric ontbreekt.
    """
    out: Dict[str, Tuple[float, float]] = {}
    if df_thr is None or df_thr.empty:
        return {m: (float(fallback_low), float(fallback_high)) for m in metrics}

    df_thr = df_thr.copy()
    df_thr["metric"] = df_thr["metric"].astype(str)

    for m in metrics:
        hit = df_thr[df_thr["metric"] == str(m)]
        if hit.empty:
            out[m] = (float(fallback_low), float(fallback_high))
        else:
            lo = hit["ratio_low"].iloc[0]
            hi = hit["ratio_high"].iloc[0]
            lo = float(lo) if pd.notna(lo) else float(fallback_low)
            hi = float(hi) if pd.notna(hi) else float(fallback_high)
            out[m] = (lo, hi)

    return out


# ------------------------------------------------------------
# HELPER-FUNCTIES – DATA
# ------------------------------------------------------------

def _normalize_event(e: str) -> str:
    s = str(e).strip().lower()
    if s == "summary":
        return "summary"
    return s


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns:
        return out
    s = out[col]
    if np.issubdtype(s.dtype, np.datetime64):
        return out
    out[col] = pd.to_datetime(s, errors="coerce", dayfirst=True, utc=False)
    return out


def _add_week_key_from_year_week(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if COL_YEAR not in out.columns or COL_WEEK not in out.columns:
        out["week_key"] = pd.NA
        out["week_label"] = pd.NA
        return out

    out[COL_YEAR] = pd.to_numeric(out[COL_YEAR], errors="coerce").astype("Int64")
    out[COL_WEEK] = pd.to_numeric(out[COL_WEEK], errors="coerce").astype("Int64")

    out["week_key"] = (out[COL_YEAR] * 100 + out[COL_WEEK]).astype("Int64")
    out["week_label"] = out.apply(
        lambda r: f"{int(r[COL_YEAR]):04d}-W{int(r[COL_WEEK]):02d}"
        if pd.notna(r[COL_YEAR]) and pd.notna(r[COL_WEEK]) else None,
        axis=1,
    )
    return out


def _add_iso_week_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _ensure_datetime(out, COL_DATE)
    if COL_DATE not in out.columns:
        out["week_key"] = pd.NA
        out["week_label"] = pd.NA
        return out

    iso = out[COL_DATE].dt.isocalendar()
    out["iso_year"] = iso["year"].astype("Int64")
    out["iso_week"] = iso["week"].astype("Int64")
    out["week_key"] = (out["iso_year"] * 100 + out["iso_week"]).astype("Int64")
    out["week_label"] = out.apply(
        lambda r: f"{int(r['iso_year']):04d}-W{int(r['iso_week']):02d}"
        if pd.notna(r.get("iso_year")) and pd.notna(r.get("iso_week")) else None,
        axis=1,
    )
    return out


def detect_metrics_from_gps(df_gps: pd.DataFrame) -> List[str]:
    base_cols = {
        COL_WEEK, COL_YEAR, COL_DATE, COL_PLAYER, "Type", COL_EVENT,
        "Hoofdpositie", "Subpositie", "Subpositie ", "Wedstrijd", "Opponent",
        "EVENT_NORM", "iso_year", "iso_week", "week_key", "week_label",
    }

    candidates: List[str] = []
    for c in df_gps.columns:
        if c in base_cols:
            continue
        if any(str(c).endswith(suf) for suf in EXCLUDE_SUFFIXES):
            continue
        if c in EXCLUDE_METRICS:
            continue
        if np.issubdtype(df_gps[c].dtype, np.number):
            candidates.append(c)

    return candidates


def make_weekly_from_gps(df_gps: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    df = df_gps.copy()

    # Summary-only
    if COL_EVENT in df.columns:
        df["EVENT_NORM"] = df[COL_EVENT].map(_normalize_event)
        df = df[df["EVENT_NORM"] == "summary"].copy()

    # 1) Prefer Year+Week
    df = _add_week_key_from_year_week(df)

    # 2) Fallback ISO-week via Datum
    if "week_key" not in df.columns or df["week_key"].isna().mean() > 0.50:
        if COL_DATE in df.columns:
            df = _add_iso_week_fields(df)

    # 3) Last fallback: only Week
    if "week_key" not in df.columns or df["week_key"].isna().all():
        if COL_WEEK not in df.columns:
            raise ValueError(f"Kan geen week maken: '{COL_YEAR}'+ '{COL_WEEK}' ontbreekt en ook geen '{COL_DATE}'.")
        wk = pd.to_numeric(df[COL_WEEK], errors="coerce").astype("Int64")
        df["week_key"] = wk
        df["week_label"] = wk.apply(lambda x: f"W{int(x):02d}" if pd.notna(x) else None)

    df = df.dropna(subset=["week_key", COL_PLAYER]).copy()
    grp = df.groupby(["week_key", "week_label", COL_PLAYER], as_index=False)[metrics].sum()
    grp = grp.rename(columns={COL_PLAYER: "player"})
    return grp


def compute_acwr(df: pd.DataFrame, metrics: List[str], group_col: str = "player", week_col: str = "week_key") -> pd.DataFrame:
    df = df.copy()
    df[week_col] = pd.to_numeric(df[week_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[week_col]).copy()
    df = df.sort_values([group_col, week_col])

    for m in metrics:
        grp = df.groupby(group_col)[m]
        chronic = grp.shift(1).rolling(window=4, min_periods=2).mean()
        df[f"{m}_ACWR"] = df[m] / chronic

    return df


def make_team_level(df: pd.DataFrame, metrics: List[str], week_col: str = "week_key") -> pd.DataFrame:
    team = df.groupby([week_col, "week_label"])[metrics].sum().reset_index()
    team["player"] = "Team"
    cols = [week_col, "week_label", "player"] + metrics
    return team[cols]


def compute_chronic_last4weeks(df: pd.DataFrame, metrics: List[str], group_col: str = "player", week_col: str = "week_key") -> pd.DataFrame:
    rows = []
    for group_value, df_g in df.groupby(group_col):
        df_g = df_g.copy()
        df_g[week_col] = pd.to_numeric(df_g[week_col], errors="coerce").astype("Int64")
        df_g = df_g.dropna(subset=[week_col]).sort_values(week_col)
        last4 = df_g.tail(4)
        if len(last4) == 0:
            continue
        chronic = last4[metrics].mean()
        row = {group_col: group_value}
        row.update(chronic.to_dict())
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[group_col] + metrics)

    return pd.DataFrame(rows)


def _compute_next_week_from_weekkey(max_week_key: int) -> Tuple[int, str]:
    y = int(max_week_key // 100)
    w = int(max_week_key % 100)
    d0 = date.fromisocalendar(y, w, 1)
    d1 = d0 + timedelta(days=7)
    iso = d1.isocalendar()
    next_y = int(iso.year)
    next_w = int(iso.week)
    wk = next_y * 100 + next_w
    lbl = f"{next_y:04d}-W{next_w:02d}"
    return wk, lbl


# ------------------------------------------------------------
# GRAFIEK – ACWR (Plotly)
# ------------------------------------------------------------

def line_chart_acwr(
    df_view: pd.DataFrame,
    param: str,
    week_col: str = "week_key",
    label_col: str = "week_label",
    group_label: Optional[str] = None,
    highlight_week: Optional[int] = None,
):
    acwr_col = f"{param}_ACWR"
    if acwr_col not in df_view.columns:
        st.warning(f"Geen ACWR gevonden voor parameter '{param}'.")
        return

    df_plot = df_view[[week_col, label_col, acwr_col]].dropna().copy()
    if df_plot.empty:
        st.warning(f"Geen ACWR-data om te tonen voor '{param}'.")
        return

    df_plot[week_col] = pd.to_numeric(df_plot[week_col], errors="coerce").astype("Int64")
    df_plot = df_plot.dropna(subset=[week_col]).sort_values(week_col)

    max_val = float(df_plot[acwr_col].max())
    max_y = max(1.6, max_val * 1.10)

    x_labels = df_plot[label_col].astype(str).tolist()
    y_vals = df_plot[acwr_col].astype(float).tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=y_vals,
            mode="lines+markers",
            line=dict(width=2, shape="spline"),
            marker=dict(size=6),
            showlegend=False,
        )
    )

    if group_label == "Team (globaal)":
        y = np.array(y_vals, dtype=float)
        sd = float(np.nanstd(y))
        upper = y + sd
        lower = np.maximum(y - sd, 0.0)

        fig.add_trace(
            go.Scatter(
                x=x_labels, y=upper,
                mode="lines",
                line=dict(color="rgba(0,150,255,0)", shape="spline"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_labels, y=lower,
                mode="lines",
                line=dict(color="rgba(0,150,255,0)", shape="spline"),
                fill="tonexty",
                fillcolor="rgba(0,150,255,0.22)",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_hrect(y0=0.0, y1=SWEET_SPOT_LOW, line_width=0, fillcolor="#8B0000", opacity=0.25, layer="below")
    fig.add_hrect(y0=SWEET_SPOT_LOW, y1=SWEET_SPOT_HIGH, line_width=0, fillcolor="#006400", opacity=0.30, layer="below")
    fig.add_hrect(y0=SWEET_SPOT_HIGH, y1=max_y, line_width=0, fillcolor="#8B0000", opacity=0.25, layer="below")
    fig.add_hline(y=1.0, line_dash="dot", line_width=1)

    if highlight_week is not None:
        m = df_plot[df_plot[week_col] == highlight_week]
        if not m.empty:
            highlight_label = str(m[label_col].iloc[0])
            fig.add_vline(x=highlight_label, line_dash="dash", line_width=1.5)

    fig.update_layout(
        title=f"ACWR - {param}" + (f" ({group_label})" if group_label else ""),
        xaxis_title="Week",
        yaxis_title="ACWR",
        yaxis_range=[0.0, max_y],
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


# ------------------------------------------------------------
# TARGETS vs WORKLOAD – HELPERS
# ------------------------------------------------------------

def _compute_player_targets(
    df_weekly: pd.DataFrame,
    metrics: List[str],
    ratios_by_metric: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    chronic_players = compute_chronic_last4weeks(
        df=df_weekly,
        metrics=metrics,
        group_col="player",
        week_col="week_key",
    )
    rows = []
    for _, row in chronic_players.iterrows():
        p = row["player"]
        for m in metrics:
            chronic_val = float(row[m])
            rlow, rhigh = ratios_by_metric.get(m, (0.8, 1.0))
            rows.append({
                "player": p,
                "metric": m,
                "chronic": chronic_val,
                "ratio_low": float(rlow),
                "ratio_high": float(rhigh),
                "target_low": float(rlow) * chronic_val,
                "target_high": float(rhigh) * chronic_val,
            })
    if not rows:
        return pd.DataFrame(columns=["player", "metric", "chronic", "ratio_low", "ratio_high", "target_low", "target_high"])
    return pd.DataFrame(rows)


def _compute_team_targets(
    df_weekly: pd.DataFrame,
    metrics: List[str],
    ratios_by_metric: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    team_full = make_team_level(df_weekly, metrics, week_col="week_key")
    chronic_team = compute_chronic_last4weeks(
        df=team_full,
        metrics=metrics,
        group_col="player",
        week_col="week_key",
    )
    rows = []
    for _, row in chronic_team.iterrows():
        for m in metrics:
            chronic_val = float(row[m])
            rlow, rhigh = ratios_by_metric.get(m, (0.8, 1.0))
            rows.append({
                "group": "Team",
                "metric": m,
                "chronic": chronic_val,
                "ratio_low": float(rlow),
                "ratio_high": float(rhigh),
                "target_low": float(rlow) * chronic_val,
                "target_high": float(rhigh) * chronic_val,
            })
    if not rows:
        return pd.DataFrame(columns=["group", "metric", "chronic", "ratio_low", "ratio_high", "target_low", "target_high"])
    return pd.DataFrame(rows)


def _compute_target_bar_data(
    weekly_df: pd.DataFrame,
    targets_df_players: pd.DataFrame,
    targets_df_team: pd.DataFrame,
    metric: str,
    week_key_val: int,
    target_level: str,
    target_player: str,
) -> pd.DataFrame:
    if week_key_val is None:
        return pd.DataFrame()

    week_sel = weekly_df[weekly_df["week_key"] == week_key_val].copy()
    if week_sel.empty:
        return pd.DataFrame()

    if target_level == "Per speler":
        t_players = targets_df_players[targets_df_players["metric"] == metric].copy()
        if t_players.empty:
            return pd.DataFrame()

        if target_player != "Alle spelers":
            t_players = t_players[t_players["player"] == target_player].copy()
            if t_players.empty:
                return pd.DataFrame()

        week_metric = week_sel[["player", metric]].rename(columns={metric: "actual_abs"})
        df = t_players.merge(week_metric, on="player", how="left")

    else:
        t_team = targets_df_team[targets_df_team["metric"] == metric].copy()
        if t_team.empty:
            return pd.DataFrame()

        team_week = (
            week_sel.groupby("week_key", as_index=False)[metric]
            .sum()
            .rename(columns={metric: "actual_abs"})
        )
        if team_week.empty:
            return pd.DataFrame()

        actual_val = float(team_week["actual_abs"].iloc[0])
        df = t_team.copy()
        df["player"] = "Team"
        df["actual_abs"] = actual_val

    df["actual_abs"] = df["actual_abs"].fillna(0.0)
    df["target_low_abs"] = df["target_low"].fillna(0.0)
    df["target_high_abs"] = df["target_high"].fillna(0.0)

    # ratio to MAX target (target_high)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            df["target_high_abs"] > 0,
            df["actual_abs"] / df["target_high_abs"],
            np.nan,
        )
        min_ratio = np.where(
            df["target_high_abs"] > 0,
            df["target_low_abs"] / df["target_high_abs"],
            np.nan,
        )

    df["ratio"] = ratio
    df["min_ratio"] = min_ratio  # <-- voor ondergrens-lijn

    ratio_clamped = np.clip(ratio, 0.0, 2.0)
    green = np.minimum(ratio_clamped, 1.0)
    red_missing = np.maximum(1.0 - ratio_clamped, 0.0)
    red_excess = np.maximum(ratio_clamped - 1.0, 0.0)

    df["green"] = green
    df["red_missing"] = red_missing
    df["red_excess"] = red_excess

    df["remaining_to_min_abs"] = np.maximum(df["target_low_abs"] - df["actual_abs"], 0.0)
    df["remaining_to_max_abs"] = np.maximum(df["target_high_abs"] - df["actual_abs"], 0.0)

    return df[[
        "player",
        "target_low_abs",
        "target_high_abs",
        "actual_abs",
        "remaining_to_min_abs",
        "remaining_to_max_abs",
        "ratio",
        "min_ratio",
        "green",
        "red_missing",
        "red_excess",
    ]].copy()


def _build_target_bar_figure(df_bar: pd.DataFrame, metric: str, week_label: str, title_prefix: str = "Week target"):
    if df_bar.empty:
        return go.Figure()

    players = df_bar["player"].tolist()
    red_missing = df_bar["red_missing"].to_numpy()
    green = df_bar["green"].to_numpy()
    red_excess = df_bar["red_excess"].to_numpy()
    ratio = df_bar["ratio"].to_numpy()

    total_height = red_missing + green + red_excess
    max_total = float(np.nanmax(total_height)) if len(total_height) else 1.0
    max_total = max(max_total, 1.0)

    fig = go.Figure()
    fig.add_bar(x=players, y=green, name="Load", marker_color="#00CC00")
    fig.add_bar(x=players, y=red_missing, name="Remaining", marker_color="#CC0000")

    if np.nanmax(red_excess) > 0:
        fig.add_bar(x=players, y=red_excess, name="Above target", marker_color="#990000")

    # % label boven balk
    perc_labels = [f"{r * 100:.0f}%" if np.isfinite(r) else "–" for r in ratio]
    fig.add_scatter(
        x=players,
        y=total_height + 0.02 * max_total,
        mode="text",
        text=perc_labels,
        textposition="top center",
        textfont=dict(color="white", size=10),
        showlegend=False,
        hoverinfo="skip",
    )

    # Ondergrens (min target) als lijn (in % t.o.v. max target)
    min_ratio_vals = pd.to_numeric(df_bar["min_ratio"], errors="coerce")
    min_ratio_vals = min_ratio_vals[np.isfinite(min_ratio_vals)]
    if len(min_ratio_vals) > 0:
        y_min_line = float(np.nanmedian(min_ratio_vals))
        if np.isfinite(y_min_line):
            fig.add_hline(y=y_min_line, line_width=2, line_dash="dash")

    fig.update_layout(
        barmode="stack",
        title=f"{title_prefix}: {metric} ({week_label})",
        margin=dict(l=10, r=10, t=40, b=40),
        showlegend=False,
    )
    fig.update_yaxes(title="%", tickformat=".0%", range=[0, max_total * 1.08])
    fig.update_xaxes(tickangle=-60)
    return fig


def _build_target_table(dfs_for_table: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for metric, df_bar in dfs_for_table:
        for _, r in df_bar.iterrows():
            rows.append({
                "Player": r["player"],
                "Parameter": metric,
                "Actual": r["actual_abs"],
                "Min target": r["target_low_abs"],
                "Max target": r["target_high_abs"],
                "Remaining to min": r["remaining_to_min_abs"],
                "Remaining to max": r["remaining_to_max_abs"],
            })
    if not rows:
        return pd.DataFrame(columns=[
            "Player", "Parameter", "Actual", "Min target", "Max target",
            "Remaining to min", "Remaining to max"
        ])
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# HOOFDFUNCTIE VOOR STREAMLIT-APP
# ------------------------------------------------------------

def acwr_pages_main(df_gps: pd.DataFrame):
    """
    Entry point vanuit app.py
    df_gps: GPS-database ingelezen uit werkblad 'GPS'.
    """

    # Supabase status
    sb = _get_supabase_client()
    _sb_auth_if_possible(sb)

    # --------------------------------------------------------
    # CHECK: Year/Week beschikbaar?
    # --------------------------------------------------------
    if COL_YEAR not in df_gps.columns:
        st.warning(f"Kolom '{COL_YEAR}' ontbreekt. Voeg 'Year' toe voor correcte weekvolgorde over jaarwisseling.")

    # --------------------------------------------------------
    # DATA: metrics detecteren + weekdataset
    # --------------------------------------------------------
    metrics = detect_metrics_from_gps(df_gps)
    if not metrics:
        st.error("Geen geschikte load-parameters gevonden in de GPS-data.")
        return

    df_weekly = make_weekly_from_gps(df_gps, metrics)

    # Per speler + Team
    df_acwr_players = compute_acwr(df_weekly, metrics, group_col="player", week_col="week_key")
    df_team = make_team_level(df_weekly, metrics, week_col="week_key")
    df_acwr_team = compute_acwr(df_team, metrics, group_col="player", week_col="week_key")

    # UI opties: week_key gesorteerd + labels
    df_weeks = (
        df_weekly[["week_key", "week_label"]]
        .dropna()
        .drop_duplicates()
        .copy()
    )
    df_weeks["week_key"] = pd.to_numeric(df_weeks["week_key"], errors="coerce").astype("Int64")
    df_weeks = df_weeks.dropna(subset=["week_key"]).sort_values("week_key")

    weeks_keys = df_weeks["week_key"].astype(int).tolist()
    weekkey_to_label = dict(zip(df_weeks["week_key"].astype(int), df_weeks["week_label"].astype(str)))
    players_sorted = sorted(df_weekly["player"].dropna().unique().tolist())

    # Default metrics
    default_metrics = [m for m in DEFAULT_PREF_METRICS if m in metrics]
    if not default_metrics:
        default_metrics = metrics[: min(4, len(metrics))]

    # --------------------------------------------------------
    # LAYOUT: TABS
    # --------------------------------------------------------
    tab_dashboard, tab_thresholds, tab_targets = st.tabs(
        ["ACWR Dashboard", "Threshold planner", "Targets vs Workload"]
    )

    # ========================================================
    # TAB 1: ACWR DASHBOARD
    # ========================================================
    with tab_dashboard:
        st.header("ACWR Dashboard (per week)")

        col_sel1, col_sel2, col_sel3 = st.columns([1.2, 1.2, 1])

        with col_sel1:
            view_mode = st.radio("Niveau", ["Per speler", "Team (globaal)"], key="acwr_dashboard_level")

        with col_sel2:
            if view_mode == "Per speler":
                selected_player = st.selectbox("Kies speler", players_sorted, key="acwr_dashboard_player")
            else:
                selected_player = "Team"

        with col_sel3:
            if weeks_keys:
                label_opts = [weekkey_to_label[wk] for wk in weeks_keys]
                idx_default = len(weeks_keys) - 1
                sel_label = st.selectbox("Week highlight", options=label_opts, index=idx_default, key="acwr_dashboard_week_label")
                label_to_key = {v: k for k, v in weekkey_to_label.items()}
                selected_week_key = label_to_key.get(sel_label)
            else:
                selected_week_key = None

        selected_params = st.multiselect(
            "Kies parameters (max 4 tegelijk)",
            options=metrics,
            default=default_metrics,
            key="acwr_dashboard_params",
        )[:4]

        if not selected_params:
            st.warning("Selecteer minstens één parameter.")
        else:
            if view_mode == "Per speler":
                df_view = df_acwr_players[df_acwr_players["player"] == selected_player].copy()
                group_label = selected_player
            else:
                df_view = df_acwr_team.copy()
                group_label = "Team (globaal)"

            if df_view.empty:
                st.warning("Geen data voor deze selectie.")
            else:
                cols_plot = st.columns(2)
                for i, param in enumerate(selected_params):
                    with cols_plot[i % 2]:
                        line_chart_acwr(
                            df_view=df_view,
                            param=param,
                            week_col="week_key",
                            label_col="week_label",
                            group_label=group_label,
                            highlight_week=selected_week_key,
                        )

    # ========================================================
    # TAB 2: THRESHOLD PLANNER (OPSLAAN PER WEEK in Supabase)
    # ========================================================
    with tab_thresholds:
        st.header("Threshold planner (opslaan per week)")

        if sb is None:
            st.error(
                "Supabase client niet beschikbaar. Controleer of je `supabase` package geïnstalleerd is "
                "en `SUPABASE_URL` + `SUPABASE_ANON_KEY` in `st.secrets` staan."
            )
            st.stop()

        if "access_token" not in st.session_state:
            st.warning("Niet ingelogd (geen access_token). Opslaan/lezen kan falen door RLS.")
        st.markdown(
            "Sla **ratio_low** en **ratio_high** op per week en per parameter. "
            "Deze worden automatisch gebruikt in **Targets vs Workload**."
        )

        if not weeks_keys:
            st.warning("Geen weken gevonden in de data.")
            st.stop()

        # team selector (als je later meerdere squads wilt)
        team = st.text_input("Team sleutel", value=DEFAULT_TEAM)

        # plan week: default = volgende week na laatste week in data
        max_wk = int(max(weeks_keys))
        next_wk_key, next_wk_label = _compute_next_week_from_weekkey(max_wk)

        plan_options = [(next_wk_key, f"{next_wk_label} (volgende week)")] + [(wk, weekkey_to_label[wk]) for wk in weeks_keys]
        plan_labels = [lbl for _, lbl in plan_options]
        plan_label_to_key = {lbl: wk for wk, lbl in plan_options}

        col_p1, col_p2 = st.columns([1.2, 2.0])
        with col_p1:
            sel_plan_label = st.selectbox("Plan thresholds voor week", options=plan_labels, index=0, key="thr_plan_week")
            plan_week_key = int(plan_label_to_key[sel_plan_label])
            plan_week_label = sel_plan_label.split(" ")[0] if "(volgende" in sel_plan_label else sel_plan_label

        with col_p2:
            params_thr = st.multiselect(
                "Kies parameters voor thresholds",
                options=metrics,
                default=default_metrics,
                key="thr_params",
            )
            if not params_thr:
                st.warning("Kies minstens één parameter.")
                st.stop()

        col_thr1, col_thr2 = st.columns(2)
        with col_thr1:
            fallback_low = st.number_input("Default ondergrens ratio (fallback)", value=0.80, step=0.05, key="thr_fallback_low")
        with col_thr2:
            fallback_high = st.number_input("Default bovengrens ratio (fallback)", value=1.00, step=0.05, key="thr_fallback_high")

        if fallback_low <= 0 or fallback_high <= 0:
            st.error("Ratio-grenzen moeten > 0 zijn.")
            st.stop()
        if fallback_low > fallback_high:
            st.error("Ondergrens mag niet groter zijn dan bovengrens.")
            st.stop()

        # Load existing thresholds from Supabase for this week
        df_thr_existing = sb_get_thresholds_cached(team=team, week_key=plan_week_key)
        ratios_existing = ratios_from_threshold_df(
            df_thr=df_thr_existing,
            metrics=params_thr,
            fallback_low=float(fallback_low),
            fallback_high=float(fallback_high),
        )

        st.markdown("### Ratio's per parameter (aanpasbaar)")
        df_edit = pd.DataFrame([{
            "metric": m,
            "ratio_low": ratios_existing[m][0],
            "ratio_high": ratios_existing[m][1],
        } for m in params_thr])

        df_edit = st.data_editor(
            df_edit,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "metric": st.column_config.TextColumn("Parameter", disabled=True),
                "ratio_low": st.column_config.NumberColumn("Ratio low", min_value=0.01, step=0.05),
                "ratio_high": st.column_config.NumberColumn("Ratio high", min_value=0.01, step=0.05),
            },
            key="thr_ratios_editor",
        )

        bad = df_edit[(df_edit["ratio_low"] <= 0) | (df_edit["ratio_high"] <= 0) | (df_edit["ratio_low"] > df_edit["ratio_high"])]
        if not bad.empty:
            st.error("In de tabel: ratio_low en ratio_high moeten > 0 zijn en ratio_low <= ratio_high.")
            st.stop()

        note = st.text_input("Notitie (optioneel)", value="", key="thr_note")

        col_s1, col_s2 = st.columns([1.0, 2.0])
        with col_s1:
            if st.button("Opslaan voor deze week", type="primary"):
                ok, msg = sb_upsert_thresholds(
                    team=team,
                    week_key=plan_week_key,
                    week_label=plan_week_label,
                    df_ratios=df_edit[["metric", "ratio_low", "ratio_high"]],
                    note=note,
                )
                if ok:
                    st.success(f"Thresholds opgeslagen voor {plan_week_label}.")
                else:
                    st.error(msg)

        with col_s2:
            if not df_thr_existing.empty:
                last_upd = df_thr_existing.get("updated_at", pd.Series([], dtype=str))
                st.caption(f"Bestaande thresholds gevonden: {len(df_thr_existing)} rijen.")

        st.markdown("### Overzicht opgeslagen thresholds (deze week)")
        df_thr_show = sb_get_thresholds_cached(team=team, week_key=plan_week_key)
        if df_thr_show.empty:
            st.info("Nog geen thresholds opgeslagen voor deze week.")
        else:
            df_thr_show = df_thr_show.sort_values(["metric"]).reset_index(drop=True)
            st.dataframe(df_thr_show, use_container_width=True)

    # ========================================================
    # TAB 3: TARGETS vs WORKLOAD (gebruikt opgeslagen thresholds)
    # ========================================================
    with tab_targets:
        st.header("Targets vs Workload")

        st.markdown(
            "Barcharts tonen % t.o.v. **max target**. "
            "De **ondergrens** (min target) staat als stippellijn."
        )

        if not weeks_keys:
            st.warning("Geen weken gevonden in de data.")
            return

        team = st.text_input("Team sleutel (moet matchen met planner)", value=DEFAULT_TEAM, key="targets_team")

        col_t1, col_t2, col_t3 = st.columns([1.2, 1.2, 1.4])
        with col_t1:
            target_level = st.radio("Niveau", ["Per speler", "Team (globaal)"], key="targets_level")
        with col_t2:
            if target_level == "Per speler":
                player_opts = ["Alle spelers"] + players_sorted
                target_player = st.selectbox("Kies speler", player_opts, key="targets_player")
            else:
                target_player = "Team"
                st.markdown("**Team (globaal)** geselecteerd.")
        with col_t3:
            label_opts = [weekkey_to_label[wk] for wk in weeks_keys]
            idx_default = len(weeks_keys) - 1
            sel_label = st.selectbox("Week (werkelijk load voor vergelijking)", options=label_opts, index=idx_default, key="targets_week_label")
            label_to_key = {v: k for k, v in weekkey_to_label.items()}
            target_week_key = int(label_to_key.get(sel_label))
            target_week_label = sel_label

        st.markdown("---")

        params_target = st.multiselect(
            "Kies parameters (max 4 tegelijk)",
            options=metrics,
            default=default_metrics,
            key="targets_params",
        )[:4]

        if not params_target:
            st.warning("Kies minstens één parameter.")
            return

        col_use1, col_use2 = st.columns([1.2, 1.2])
        with col_use1:
            use_saved = st.checkbox("Gebruik opgeslagen thresholds voor deze week", value=True, key="targets_use_saved_thresholds")
        with col_use2:
            fallback_low_t = st.number_input("Fallback ratio low", value=0.80, step=0.05, key="targets_fallback_low")
            fallback_high_t = st.number_input("Fallback ratio high", value=1.00, step=0.05, key="targets_fallback_high")

        if fallback_low_t <= 0 or fallback_high_t <= 0:
            st.error("Fallback ratio's moeten > 0 zijn.")
            return
        if fallback_low_t > fallback_high_t:
            st.error("Fallback ondergrens mag niet groter zijn dan bovengrens.")
            return

        if use_saved and _get_supabase_client() is not None:
            df_thr_week = sb_get_thresholds_cached(team=team, week_key=target_week_key)
            ratios_by_metric = ratios_from_threshold_df(
                df_thr=df_thr_week,
                metrics=params_target,
                fallback_low=float(fallback_low_t),
                fallback_high=float(fallback_high_t),
            )
        else:
            ratios_by_metric = {m: (float(fallback_low_t), float(fallback_high_t)) for m in params_target}

        # Targets (alleen voor geselecteerde metrics)
        targets_players = _compute_player_targets(df_weekly, params_target, ratios_by_metric)
        targets_team = _compute_team_targets(df_weekly, params_target, ratios_by_metric)

        if targets_players.empty and targets_team.empty:
            st.warning("Geen targets beschikbaar (onvoldoende data voor laatste 4 weken).")
            return

        dfs_for_table: List[Tuple[str, pd.DataFrame]] = []
        cols_grid = st.columns(2)

        for i, p in enumerate(params_target):
            df_bar = _compute_target_bar_data(
                weekly_df=df_weekly,
                targets_df_players=targets_players,
                targets_df_team=targets_team,
                metric=p,
                week_key_val=target_week_key,
                target_level=target_level,
                target_player=target_player,
            )
            dfs_for_table.append((p, df_bar))

            with cols_grid[i % 2]:
                if df_bar.empty:
                    st.info(f"Geen data voor {p} in {target_week_label}.")
                else:
                    rlow, rhigh = ratios_by_metric.get(p, (fallback_low_t, fallback_high_t))
                    fig_bar = _build_target_bar_figure(
                        df_bar,
                        p,
                        target_week_label,
                        title_prefix=f"Week target (ratio {rlow:.2f}–{rhigh:.2f})",
                    )
                    st.plotly_chart(fig_bar, width="stretch")

        st.markdown("----")
        st.subheader("Absolute waardes t.o.v. targets")

        table_df = _build_target_table(dfs_for_table)
        if table_df.empty:
            st.info("Geen data om in de tabel weer te geven.")
        else:
            def highlight_remaining_to_min(col):
                return ["background-color: #FF3333; color: white;" if v > 0 else "" for v in col]

            styled = (
                table_df.style
                .format({
                    "Actual": "{:.0f}",
                    "Min target": "{:.0f}",
                    "Max target": "{:.0f}",
                    "Remaining to min": "{:.0f}",
                    "Remaining to max": "{:.0f}",
                })
                .apply(highlight_remaining_to_min, subset=["Remaining to min"])
            )
            st.dataframe(styled, width="stretch")
