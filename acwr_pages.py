# acwr_pages.py
# ============================================================
# ACWR + Threshold planner (Supabase save per week) + Targets vs Workload
#
# Belangrijk:
# - Alleen Event == 'Summary'
# - Jaarwisseling-fix via week_key = Year*100 + Week (YYYYWW)
# - Thresholds (ratio_low/high) worden opgeslagen in Supabase per (team, week_key, metric)
# - Eén dropdown met alle weken + status-icoon:
#     ✅ = thresholds bestaan in Supabase voor die week
#     ⬜ = nog niet gesaved
# - Targets vs Workload gebruikt AUTOMATISCH de thresholds van de geselecteerde week
# - In Targets vs Workload wordt ondergrens (min target) weergegeven als stippellijn (% van max target)
#
# Vereist Supabase tabel:
#   public.acwr_week_thresholds(team, week_key, metric) unique
#
# Vereist secrets:
#   SUPABASE_URL
#   SUPABASE_ANON_KEY
#
# Vereist login flow (in jouw app):
#   st.session_state["access_token"]
#   st.session_state["user_email"] (optioneel)
# ============================================================

from __future__ import annotations

from datetime import date, timedelta, datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    from supabase import create_client  # type: ignore
except Exception:
    create_client = None


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
SWEET_SPOT_LOW = 0.80
SWEET_SPOT_HIGH = 1.30

COL_WEEK = "Week"
COL_YEAR = "Year"
COL_DATE = "Datum"
COL_PLAYER = "Speler"
COL_EVENT = "Event"

THRESH_TABLE = "acwr_week_thresholds"
DEFAULT_TEAM = "MVV"

EXCLUDE_METRICS = {"Max Speed", "Avg Speed", "Avg HR", "Max HR"}
EXCLUDE_SUFFIXES = ("/min",)

DEFAULT_PREF_METRICS = ["Total Distance", "Sprint", "High Sprint", "playerload2D"]


# ------------------------------------------------------------
# SUPABASE
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_supabase_client():
    if create_client is None:
        return None
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_ANON_KEY"]
        return create_client(url, key)
    except Exception:
        return None


def _sb_auth_if_possible(sb):
    if sb is None:
        return
    token = st.session_state.get("access_token")
    if token:
        try:
            sb.postgrest.auth(token)
        except Exception:
            pass


@st.cache_data(show_spinner=False, ttl=30)
def sb_get_thresholds_cached(team: str, week_key: int) -> pd.DataFrame:
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
        return pd.DataFrame(resp.data or [])
    except Exception:
        return pd.DataFrame(columns=cols)


@st.cache_data(show_spinner=False, ttl=30)
def sb_saved_week_keys_cached(team: str) -> set[int]:
    sb = _get_supabase_client()
    _sb_auth_if_possible(sb)
    if sb is None:
        return set()
    try:
        resp = (
            sb.table(THRESH_TABLE)
              .select("week_key")
              .eq("team", team)
              .execute()
        )
        rows = resp.data or []
        keys = set()
        for r in rows:
            try:
                keys.add(int(r["week_key"]))
            except Exception:
                pass
        return keys
    except Exception:
        return set()


def sb_upsert_thresholds(team: str, week_key: int, week_label: str, df_ratios: pd.DataFrame, note: str = "") -> Tuple[bool, str]:
    sb = _get_supabase_client()
    _sb_auth_if_possible(sb)

    if sb is None:
        return False, "Supabase client niet beschikbaar (package/secrets)."

    created_by = st.session_state.get("user_email")

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
            "created_by": created_by,
        })

    try:
        sb.table(THRESH_TABLE).upsert(payload, on_conflict="team,week_key,metric").execute()

        # refresh cached reads so ✅ appears immediately and Targets-tab sees new values
        st.cache_data.clear()
        return True, "OK"
    except Exception as e:
        return False, f"Upsert faalde: {e}"


def ratios_from_threshold_df(df_thr: pd.DataFrame, metrics: List[str], fallback_low: float, fallback_high: float) -> Dict[str, Tuple[float, float]]:
    if df_thr is None or df_thr.empty:
        return {m: (float(fallback_low), float(fallback_high)) for m in metrics}

    df_thr = df_thr.copy()
    df_thr["metric"] = df_thr["metric"].astype(str)

    out: Dict[str, Tuple[float, float]] = {}
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
# DATA HELPERS
# ------------------------------------------------------------
def _normalize_event(e: str) -> str:
    s = str(e).strip().lower()
    return "summary" if s == "summary" else s


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns:
        return out
    if np.issubdtype(out[col].dtype, np.datetime64):
        return out
    out[col] = pd.to_datetime(out[col], errors="coerce", dayfirst=True, utc=False)
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
    out = _ensure_datetime(df, COL_DATE)
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

    # 2) Fallback via Datum (ISO)
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
    df = df.dropna(subset=[week_col]).sort_values([group_col, week_col])

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
    for g, df_g in df.groupby(group_col):
        df_g = df_g.copy()
        df_g[week_col] = pd.to_numeric(df_g[week_col], errors="coerce").astype("Int64")
        df_g = df_g.dropna(subset=[week_col]).sort_values(week_col)
        last4 = df_g.tail(4)
        if len(last4) == 0:
            continue
        chronic = last4[metrics].mean()
        row = {group_col: g}
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
    wk = int(iso.year) * 100 + int(iso.week)
    lbl = f"{int(iso.year):04d}-W{int(iso.week):02d}"
    return wk, lbl


# ------------------------------------------------------------
# UI HELPERS – ONE DROPDOWN WITH ✅/⬜
# ------------------------------------------------------------
def build_week_options(df_weeks: pd.DataFrame, saved_keys: set[int]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    df = df_weeks.copy()
    df["week_key"] = pd.to_numeric(df["week_key"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["week_key"]).sort_values("week_key").copy()
    df["week_key"] = df["week_key"].astype(int)
    df["week_label"] = df["week_label"].astype(str)

    options: List[str] = []
    opt_to_key: Dict[str, int] = {}
    key_to_label: Dict[int, str] = {}

    for _, r in df.iterrows():
        wk = int(r["week_key"])
        lbl = str(r["week_label"])
        mark = "✅" if wk in saved_keys else "⬜"
        opt = f"{lbl} {mark}"
        options.append(opt)
        opt_to_key[opt] = wk
        key_to_label[wk] = lbl

    return options, opt_to_key, key_to_label


# ------------------------------------------------------------
# ACWR CHART
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
        st.warning(f"Geen ACWR gevonden voor '{param}'.")
        return

    df_plot = df_view[[week_col, label_col, acwr_col]].dropna().copy()
    if df_plot.empty:
        st.warning(f"Geen data voor '{param}'.")
        return

    df_plot[week_col] = pd.to_numeric(df_plot[week_col], errors="coerce").astype("Int64")
    df_plot = df_plot.dropna(subset=[week_col]).sort_values(week_col)

    x_labels = df_plot[label_col].astype(str).tolist()
    y_vals = df_plot[acwr_col].astype(float).tolist()

    max_val = float(np.nanmax(y_vals)) if len(y_vals) else 1.0
    max_y = max(1.6, max_val * 1.10)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_labels, y=y_vals, mode="lines+markers", line=dict(width=2, shape="spline"), marker=dict(size=6)))

    fig.add_hrect(y0=0.0, y1=SWEET_SPOT_LOW, line_width=0, fillcolor="#8B0000", opacity=0.25, layer="below")
    fig.add_hrect(y0=SWEET_SPOT_LOW, y1=SWEET_SPOT_HIGH, line_width=0, fillcolor="#006400", opacity=0.30, layer="below")
    fig.add_hrect(y0=SWEET_SPOT_HIGH, y1=max_y, line_width=0, fillcolor="#8B0000", opacity=0.25, layer="below")
    fig.add_hline(y=1.0, line_dash="dot", line_width=1)

    if highlight_week is not None:
        m = df_plot[df_plot[week_col] == highlight_week]
        if not m.empty:
            fig.add_vline(x=str(m[label_col].iloc[0]), line_dash="dash", line_width=1.5)

    fig.update_layout(
        title=f"ACWR - {param}" + (f" ({group_label})" if group_label else ""),
        xaxis_title="Week",
        yaxis_title="ACWR",
        yaxis_range=[0.0, max_y],
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# TARGETS vs WORKLOAD HELPERS
# ------------------------------------------------------------
def _compute_player_targets(df_weekly: pd.DataFrame, metrics: List[str], ratios_by_metric: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    chronic_players = compute_chronic_last4weeks(df_weekly, metrics, group_col="player", week_col="week_key")
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
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["player", "metric", "chronic", "ratio_low", "ratio_high", "target_low", "target_high"])


def _compute_team_targets(df_weekly: pd.DataFrame, metrics: List[str], ratios_by_metric: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    team_full = make_team_level(df_weekly, metrics, week_col="week_key")
    chronic_team = compute_chronic_last4weeks(team_full, metrics, group_col="player", week_col="week_key")
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
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["group", "metric", "chronic", "ratio_low", "ratio_high", "target_low", "target_high"])


def _compute_target_bar_data(
    weekly_df: pd.DataFrame,
    targets_df_players: pd.DataFrame,
    targets_df_team: pd.DataFrame,
    metric: str,
    week_key_val: int,
    target_level: str,
    target_player: str,
) -> pd.DataFrame:
    week_sel = weekly_df[weekly_df["week_key"] == int(week_key_val)].copy()
    if week_sel.empty:
        return pd.DataFrame()

    if target_level == "Per speler":
        t = targets_df_players[targets_df_players["metric"] == metric].copy()
        if t.empty:
            return pd.DataFrame()

        if target_player != "Alle spelers":
            t = t[t["player"] == target_player].copy()
            if t.empty:
                return pd.DataFrame()

        week_metric = week_sel[["player", metric]].rename(columns={metric: "actual_abs"})
        df = t.merge(week_metric, on="player", how="left")

    else:
        t = targets_df_team[targets_df_team["metric"] == metric].copy()
        if t.empty:
            return pd.DataFrame()

        team_week = week_sel.groupby("week_key", as_index=False)[metric].sum().rename(columns={metric: "actual_abs"})
        if team_week.empty:
            return pd.DataFrame()

        df = t.copy()
        df["player"] = "Team"
        df["actual_abs"] = float(team_week["actual_abs"].iloc[0])

    df["actual_abs"] = pd.to_numeric(df["actual_abs"], errors="coerce").fillna(0.0)
    df["target_low_abs"] = pd.to_numeric(df["target_low"], errors="coerce").fillna(0.0)
    df["target_high_abs"] = pd.to_numeric(df["target_high"], errors="coerce").fillna(0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(df["target_high_abs"] > 0, df["actual_abs"] / df["target_high_abs"], np.nan)
        min_ratio = np.where(df["target_high_abs"] > 0, df["target_low_abs"] / df["target_high_abs"], np.nan)

    ratio_clamped = np.clip(ratio, 0.0, 2.0)
    green = np.minimum(ratio_clamped, 1.0)
    red_missing = np.maximum(1.0 - ratio_clamped, 0.0)
    red_excess = np.maximum(ratio_clamped - 1.0, 0.0)

    out = pd.DataFrame({
        "player": df["player"],
        "ratio": ratio,
        "min_ratio": min_ratio,
        "green": green,
        "red_missing": red_missing,
        "red_excess": red_excess,
        "actual_abs": df["actual_abs"],
        "target_low_abs": df["target_low_abs"],
        "target_high_abs": df["target_high_abs"],
        "remaining_to_min_abs": np.maximum(df["target_low_abs"] - df["actual_abs"], 0.0),
        "remaining_to_max_abs": np.maximum(df["target_high_abs"] - df["actual_abs"], 0.0),
    })
    return out


def _build_target_bar_figure(df_bar: pd.DataFrame, metric: str, week_label: str, title_prefix: str):
    if df_bar.empty:
        return go.Figure()

    players = df_bar["player"].astype(str).tolist()
    green = df_bar["green"].to_numpy()
    red_missing = df_bar["red_missing"].to_numpy()
    red_excess = df_bar["red_excess"].to_numpy()
    ratio = df_bar["ratio"].to_numpy()

    total = green + red_missing + red_excess
    max_total = float(np.nanmax(total)) if len(total) else 1.0
    max_total = max(max_total, 1.0)

    fig = go.Figure()
    fig.add_bar(x=players, y=green, name="Load", marker_color="#00CC00")
    fig.add_bar(x=players, y=red_missing, name="Remaining", marker_color="#CC0000")
    if np.nanmax(red_excess) > 0:
        fig.add_bar(x=players, y=red_excess, name="Above target", marker_color="#990000")

    # labels
    perc_labels = [f"{r*100:.0f}%" if np.isfinite(r) else "–" for r in ratio]
    fig.add_scatter(
        x=players,
        y=total + 0.02 * max_total,
        mode="text",
        text=perc_labels,
        textposition="top center",
        showlegend=False,
        hoverinfo="skip",
        textfont=dict(color="white", size=10),
    )

    # min-target line (percentage t.o.v. max target)
    min_ratio_vals = pd.to_numeric(df_bar["min_ratio"], errors="coerce")
    min_ratio_vals = min_ratio_vals[np.isfinite(min_ratio_vals)]
    if len(min_ratio_vals) > 0:
        y_line = float(np.nanmedian(min_ratio_vals))
        if np.isfinite(y_line):
            fig.add_hline(y=y_line, line_width=2, line_dash="dash")

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
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "Player", "Parameter", "Actual", "Min target", "Max target", "Remaining to min", "Remaining to max"
    ])


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def acwr_pages_main(df_gps: pd.DataFrame):
    sb = _get_supabase_client()
    _sb_auth_if_possible(sb)

    metrics = detect_metrics_from_gps(df_gps)
    if not metrics:
        st.error("Geen geschikte load-parameters gevonden in de GPS-data.")
        return

    df_weekly = make_weekly_from_gps(df_gps, metrics)
    if df_weekly.empty:
        st.warning("Geen Summary-data gevonden.")
        return

    df_acwr_players = compute_acwr(df_weekly, metrics, group_col="player", week_col="week_key")
    df_team = make_team_level(df_weekly, metrics, week_col="week_key")
    df_acwr_team = compute_acwr(df_team, metrics, group_col="player", week_col="week_key")

    df_weeks = df_weekly[["week_key", "week_label"]].dropna().drop_duplicates().copy()
    df_weeks["week_key"] = pd.to_numeric(df_weeks["week_key"], errors="coerce").astype("Int64")
    df_weeks = df_weeks.dropna(subset=["week_key"]).sort_values("week_key")
    weeks_keys_sorted = df_weeks["week_key"].astype(int).tolist()

    players_sorted = sorted(df_weekly["player"].dropna().unique().tolist())

    default_metrics = [m for m in DEFAULT_PREF_METRICS if m in metrics]
    if not default_metrics:
        default_metrics = metrics[: min(4, len(metrics))]

    tab_dashboard, tab_thresholds, tab_targets = st.tabs(["ACWR Dashboard", "Threshold planner", "Targets vs Workload"])

    # ========================================================
    # TAB 1: Dashboard
    # ========================================================
    with tab_dashboard:
        st.header("ACWR Dashboard")

        c1, c2, c3 = st.columns([1.2, 1.2, 1.0])
        with c1:
            level = st.radio("Niveau", ["Per speler", "Team (globaal)"], key="acwr_level")
        with c2:
            selected_player = st.selectbox("Speler", players_sorted, key="acwr_player") if level == "Per speler" else "Team"
        with c3:
            # simple highlight: last week
            if weeks_keys_sorted:
                last_key = int(weeks_keys_sorted[-1])
                last_lbl = str(df_weeks[df_weeks["week_key"].astype(int) == last_key]["week_label"].iloc[0])
                highlight_label = st.selectbox("Highlight week", options=[str(x) for x in df_weeks["week_label"].astype(str)], index=len(df_weeks) - 1)
                label_to_key = dict(zip(df_weeks["week_label"].astype(str), df_weeks["week_key"].astype(int)))
                highlight_key = label_to_key.get(highlight_label, last_key)
            else:
                highlight_key = None

        params = st.multiselect("Parameters (max 4)", options=metrics, default=default_metrics, key="acwr_params")[:4]
        if not params:
            st.warning("Selecteer minimaal 1 parameter.")
        else:
            if level == "Per speler":
                df_view = df_acwr_players[df_acwr_players["player"] == selected_player].copy()
                group_label = selected_player
            else:
                df_view = df_acwr_team.copy()
                group_label = "Team (globaal)"

            cols = st.columns(2)
            for i, p in enumerate(params):
                with cols[i % 2]:
                    line_chart_acwr(df_view, p, group_label=group_label, highlight_week=highlight_key)

    # ========================================================
    # TAB 2: Threshold planner (1 dropdown with ✅/⬜)
    # ========================================================
    with tab_thresholds:
        st.header("Threshold planner")

        if sb is None:
            st.error("Supabase client niet beschikbaar. Controleer secrets + package.")
            st.stop()

        team = st.text_input("Team sleutel", value=DEFAULT_TEAM, key="thr_team")

        # saved status
        saved_keys = sb_saved_week_keys_cached(team)

        # include "next week" option as well (always ⬜)
        max_wk = int(max(weeks_keys_sorted)) if weeks_keys_sorted else None
        if max_wk is not None:
            next_wk_key, next_wk_label = _compute_next_week_from_weekkey(max_wk)
            df_next = pd.DataFrame([{"week_key": next_wk_key, "week_label": next_wk_label}])
            df_weeks_all = pd.concat([df_weeks, df_next], ignore_index=True)
        else:
            df_weeks_all = df_weeks.copy()

        options, opt_to_key, key_to_label = build_week_options(df_weeks_all, saved_keys)

        if not options:
            st.warning("Geen weken gevonden.")
            st.stop()

        # default = next week if present else last
        default_idx = len(options) - 1
        if max_wk is not None:
            next_opt = f"{next_wk_label} {'✅' if next_wk_key in saved_keys else '⬜'}"
            if next_opt in opt_to_key:
                default_idx = options.index(next_opt)

        sel_opt = st.selectbox("Week (✅ = gesaved)", options=options, index=default_idx, key="thr_week_opt")
        plan_week_key = int(opt_to_key[sel_opt])
        plan_week_label = key_to_label.get(plan_week_key, sel_opt.split(" ")[0])

        params_thr = st.multiselect("Parameters", options=metrics, default=default_metrics, key="thr_metrics")
        if not params_thr:
            st.warning("Kies minimaal 1 parameter.")
            st.stop()

        cA, cB = st.columns(2)
        with cA:
            fallback_low = st.number_input("Default ratio low (fallback)", value=0.80, step=0.05, key="thr_low")
        with cB:
            fallback_high = st.number_input("Default ratio high (fallback)", value=1.00, step=0.05, key="thr_high")

        if fallback_low <= 0 or fallback_high <= 0:
            st.error("Ratio's moeten > 0 zijn.")
            st.stop()
        if fallback_low > fallback_high:
            st.error("ratio_low mag niet groter zijn dan ratio_high.")
            st.stop()

        # load existing for selected week -> editor values
        df_thr_existing = sb_get_thresholds_cached(team=team, week_key=plan_week_key)
        ratios_existing = ratios_from_threshold_df(df_thr_existing, params_thr, fallback_low, fallback_high)

        df_edit = pd.DataFrame([{
            "metric": m,
            "ratio_low": ratios_existing[m][0],
            "ratio_high": ratios_existing[m][1],
        } for m in params_thr])

        df_edit = st.data_editor(
            df_edit,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "metric": st.column_config.TextColumn("Parameter", disabled=True),
                "ratio_low": st.column_config.NumberColumn("Ratio low", min_value=0.01, step=0.05),
                "ratio_high": st.column_config.NumberColumn("Ratio high", min_value=0.01, step=0.05),
            },
            key="thr_editor",
        )

        bad = df_edit[
            (pd.to_numeric(df_edit["ratio_low"], errors="coerce") <= 0)
            | (pd.to_numeric(df_edit["ratio_high"], errors="coerce") <= 0)
            | (pd.to_numeric(df_edit["ratio_low"], errors="coerce") > pd.to_numeric(df_edit["ratio_high"], errors="coerce"))
        ]
        if not bad.empty:
            st.error("Ongeldige waarden: ratio_low en ratio_high > 0 en ratio_low <= ratio_high.")
            st.stop()

        note = st.text_input("Notitie (optioneel)", value="", key="thr_note")

        if st.button("Opslaan", type="primary", use_container_width=True):
            ok, msg = sb_upsert_thresholds(
                team=team,
                week_key=plan_week_key,
                week_label=plan_week_label,
                df_ratios=df_edit[["metric", "ratio_low", "ratio_high"]],
                note=note,
            )
            if ok:
                st.success(f"Opgeslagen voor {plan_week_label}.")
                st.rerun()
            else:
                st.error(msg)

        st.caption("Bestaande thresholds voor deze week (Supabase):")
        df_show = sb_get_thresholds_cached(team=team, week_key=plan_week_key)
        if df_show.empty:
            st.info("Geen thresholds opgeslagen voor deze week.")
        else:
            st.dataframe(df_show.sort_values("metric"), use_container_width=True)

    # ========================================================
    # TAB 3: Targets vs Workload (auto uses thresholds for selected week)
    # ========================================================
    with tab_targets:
        st.header("Targets vs Workload")
    
        if sb is None:
            st.error("Supabase client niet beschikbaar. Controleer secrets + package.")
            st.stop()
    
        # =============== RIJ 1 ===============
        r1c1, r1c2, r1c3 = st.columns([1.2, 1.0, 1.8])
        with r1c1:
            team = st.text_input("Team sleutel", value=DEFAULT_TEAM, key="tvw_team")
        with r1c2:
            target_level = st.radio("Niveau", ["Per speler", "Team (globaal)"], key="tvw_level")
        with r1c3:
            if target_level == "Per speler":
                target_player = st.selectbox("Speler", ["Alle spelers"] + players_sorted, key="tvw_player")
            else:
                target_player = "Team"
                st.selectbox("Speler", ["Team"], index=0, disabled=True, key="tvw_player_disabled")
    
        # saved weeks status (✅/⬜ in week dropdown)
        saved_keys = sb_saved_week_keys_cached(team)
        options, opt_to_key, key_to_label = build_week_options(df_weeks, saved_keys)
        if not options:
            st.warning("Geen weken gevonden.")
            st.stop()
    
        # =============== RIJ 2 ===============
        r2c1, r2c2, r2c3, r2c4 = st.columns([1.0, 1.0, 1.1, 2.2])
        with r2c1:
            fallback_low = st.number_input("Fallback ratio low", value=0.80, step=0.05, key="tvw_low")
        with r2c2:
            fallback_high = st.number_input("Fallback ratio high", value=1.00, step=0.05, key="tvw_high")
        with r2c3:
            sel_opt = st.selectbox("Week (✅ = thresholds)", options=options, index=len(options) - 1, key="tvw_week_opt")
            selected_week_key = int(opt_to_key[sel_opt])
            selected_week_label = key_to_label.get(selected_week_key, sel_opt.split(" ")[0])
        with r2c4:
            params_target = st.multiselect(
                "Parameters (max 4)",
                options=metrics,
                default=default_metrics,
                key="tvw_metrics",
            )[:4]
    
        # -------- validations ----------
        if fallback_low <= 0 or fallback_high <= 0:
            st.error("Fallback ratio's moeten > 0 zijn.")
            st.stop()
        if fallback_low > fallback_high:
            st.error("Fallback low mag niet groter zijn dan fallback high.")
            st.stop()
        if not params_target:
            st.warning("Kies minimaal 1 parameter.")
            st.stop()
    
        # -------- AUTO thresholds voor gekozen week ----------
        df_thr_week = sb_get_thresholds_cached(team=team, week_key=selected_week_key)
        ratios_by_metric = ratios_from_threshold_df(df_thr_week, params_target, fallback_low, fallback_high)
    
        # -------- targets berekenen ----------
        targets_players = _compute_player_targets(df_weekly, params_target, ratios_by_metric)
        targets_team = _compute_team_targets(df_weekly, params_target, ratios_by_metric)
    
        if targets_players.empty and targets_team.empty:
            st.warning("Onvoldoende data om chronic (laatste 4 weken) te berekenen.")
            st.stop()
    
        # -------- grafieken ----------
        cols = st.columns(2)
        dfs_for_table: List[Tuple[str, pd.DataFrame]] = []
    
        for i, m in enumerate(params_target):
            df_bar = _compute_target_bar_data(
                weekly_df=df_weekly,
                targets_df_players=targets_players,
                targets_df_team=targets_team,
                metric=m,
                week_key_val=selected_week_key,
                target_level=target_level,
                target_player=target_player,
            )
            dfs_for_table.append((m, df_bar))
    
            with cols[i % 2]:
                if df_bar.empty:
                    st.info(f"Geen data voor {m} in {selected_week_label}.")
                else:
                    rlow, rhigh = ratios_by_metric.get(m, (fallback_low, fallback_high))
                    fig = _build_target_bar_figure(
                        df_bar,
                        metric=m,
                        week_label=selected_week_label,
                        title_prefix=f"Week target (ratio {rlow:.2f}–{rhigh:.2f})",
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
        # -------- tabel ----------
        st.subheader("Absolute waardes t.o.v. targets")
        table_df = _build_target_table(dfs_for_table)
        if table_df.empty:
            st.info("Geen tabeldata.")
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
            st.dataframe(styled, use_container_width=True)
