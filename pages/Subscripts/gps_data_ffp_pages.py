# pages/Subscripts/gps_data_ffp_pages.py
# ============================================
# Fitness–Fatigue–Performance per speler
# - Week-modus: per week (Year + Week -> correcte sortering over jaarwisseling)
# - Extra: Dag-modus (per datum)
# - Alleen Event == 'Summary'
#
# Aangepast:
# - Default metric = playerload2D
# - gps_id wordt uitgesloten als load-metric
# - Modelinstellingen staan in de sidebar
# ============================================

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DEFAULT_FFP_METRIC = "playerload2D"

# ------------------------------------------------------------
# Kolomnamen
# ------------------------------------------------------------
COL_PLAYER = "Speler"
COL_DATE = "Datum"
COL_EVENT = "Event"
COL_WEEK = "Week"
COL_YEAR = "Year"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _find_week_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).strip().lower() == "week":
            return c
    return None


def _find_year_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).strip().lower() == "year":
            return c
    return None


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    if date_col not in out.columns:
        return out
    if not np.issubdtype(out[date_col].dtype, np.datetime64):
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce", dayfirst=True)
    return out


def _filter_summary_only(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if COL_EVENT in out.columns:
        ev = out[COL_EVENT].astype(str).str.strip().str.lower()
        if (ev == "summary").any():
            out = out[ev == "summary"].copy()
    return out


def _detect_load_metrics(df: pd.DataFrame) -> list[str]:
    exclude_id = {
        "gps_id",
        "Speler",
        "Hoofdpositie",
        "Subpositie",
        "Datum",
        "Week",
        "Year",
        "Type",
        "Event",
        "Wedstrijd",
        "Opponent",
        "Tegenstander",
        "EVENT_NORM",
        "iso_year",
        "iso_week",
        "week_key",
        "week_label",
    }
    exclude_explicit = {"Max Speed", "Avg Speed", "Avg HR", "Max HR"}

    metrics = []
    for col in df.columns:
        if col in exclude_id:
            continue
        if col in exclude_explicit:
            continue
        if isinstance(col, str) and col.endswith("/min"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            metrics.append(col)

    return metrics


def _add_week_key_and_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    week_col = _find_week_col(out)
    year_col = _find_year_col(out)

    if year_col is not None and week_col is not None:
        out[year_col] = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")
        out[week_col] = pd.to_numeric(out[week_col], errors="coerce").astype("Int64")

        out["week_key"] = (out[year_col] * 100 + out[week_col]).astype("Int64")
        out["week_label"] = out.apply(
            lambda r: f"{int(r[year_col]):04d}-W{int(r[week_col]):02d}"
            if pd.notna(r[year_col]) and pd.notna(r[week_col]) else None,
            axis=1,
        )

    if ("week_key" not in out.columns) or (out["week_key"].isna().mean() > 0.50):
        if COL_DATE in out.columns:
            out = _ensure_datetime(out, COL_DATE)
            if COL_DATE in out.columns:
                iso = out[COL_DATE].dt.isocalendar()
                out["iso_year"] = iso["year"].astype("Int64")
                out["iso_week"] = iso["week"].astype("Int64")
                out["week_key"] = (out["iso_year"] * 100 + out["iso_week"]).astype("Int64")
                out["week_label"] = out.apply(
                    lambda r: f"{int(r['iso_year']):04d}-W{int(r['iso_week']):02d}"
                    if pd.notna(r.get("iso_year")) and pd.notna(r.get("iso_week")) else None,
                    axis=1,
                )

    if ("week_key" not in out.columns) or (out["week_key"].isna().all()):
        if week_col is None:
            raise ValueError("Geen 'Week'- of 'Datum'-kolom gevonden in GPS-data.")
        wk = pd.to_numeric(out[week_col], errors="coerce").astype("Int64")
        out["week_key"] = wk
        out["week_label"] = wk.apply(lambda x: f"W{int(x):02d}" if pd.notna(x) else None)

    return out


# ------------------------------------------------------------
# Weekly / Daily load
# ------------------------------------------------------------
def _weekly_load_for_player(df_gps: pd.DataFrame, player: str, metric: str) -> pd.DataFrame:
    df = df_gps.copy()
    df = df[df[COL_PLAYER] == player].copy()
    if df.empty:
        return pd.DataFrame(columns=["week_key", "week_label", "load"])

    df = _filter_summary_only(df)
    df = _add_week_key_and_label(df)

    df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)

    weekly = (
        df.dropna(subset=["week_key"])
        .groupby(["week_key", "week_label"], as_index=False)[metric]
        .sum()
        .rename(columns={metric: "load"})
    )

    weekly["week_key"] = pd.to_numeric(weekly["week_key"], errors="coerce").astype("Int64")
    weekly = weekly.dropna(subset=["week_key"]).sort_values("week_key").reset_index(drop=True)
    weekly["week_key"] = weekly["week_key"].astype(int)
    return weekly


def _daily_load_for_player(df_gps: pd.DataFrame, player: str, metric: str) -> pd.DataFrame:
    df = df_gps.copy()
    df = df[df[COL_PLAYER] == player].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "load"])

    df = _filter_summary_only(df)

    if COL_DATE not in df.columns:
        raise ValueError("Geen 'Datum'-kolom gevonden in GPS-data.")
    df = _ensure_datetime(df, COL_DATE)
    df = df[df[COL_DATE].notna()].copy()

    df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)

    df["_DATE_ONLY"] = df[COL_DATE].dt.date
    daily = (
        df.groupby("_DATE_ONLY", as_index=False)[metric]
        .sum()
        .rename(columns={"_DATE_ONLY": "date", metric: "load"})
    )
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


# ------------------------------------------------------------
# Banister model
# ------------------------------------------------------------
def _impulse_response_model(
    load: np.ndarray,
    tau_fit: float = 4.0,
    tau_fat: float = 1.5,
    k_fit: float = 1.0,
    k_fat: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    load = np.asarray(load, dtype=float)
    n = load.size
    F = np.zeros(n, dtype=float)
    D = np.zeros(n, dtype=float)

    alpha = np.exp(-1.0 / tau_fit)
    beta = np.exp(-1.0 / tau_fat)

    for t in range(n):
        if t == 0:
            F[t] = k_fit * load[t]
            D[t] = k_fat * load[t]
        else:
            F[t] = F[t - 1] * alpha + k_fit * load[t]
            D[t] = D[t - 1] * beta + k_fat * load[t]

    P = F - D
    return F, D, P


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
def _plot_ffp_week(weekly_df: pd.DataFrame, metric: str, player: str):
    x = weekly_df["week_label"].astype(str).to_numpy()
    load = weekly_df["load"].to_numpy()
    fit = weekly_df["fitness"].to_numpy()
    fat = weekly_df["fatigue"].to_numpy()
    perf = weekly_df["performance"].to_numpy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=x,
            y=load,
            name=f"Load ({metric})",
            opacity=0.35,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=x, y=fit, name="Fitness",
            mode="lines+markers",
            line=dict(color="#00FF88", width=2, shape="spline"),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=fat, name="Fatigue",
            mode="lines+markers",
            line=dict(color="#FF5555", width=2, shape="spline"),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=perf, name="Performance",
            mode="lines+markers",
            line=dict(color="#66CCFF", width=3, shape="spline"),
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Week")
    fig.update_yaxes(title_text=f"Load ({metric})", secondary_y=False)
    fig.update_yaxes(title_text="Fitness / Fatigue / Performance (relatief)", secondary_y=True)

    fig.update_layout(
        title=f"FFP – {player}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=10, r=10, t=50, b=10),
        bargap=0.15,
    )

    st.plotly_chart(fig, use_container_width=True)


def _plot_ffp_day(daily_df: pd.DataFrame, metric: str, player: str):
    dates = daily_df["date"].to_numpy()
    load = daily_df["load"].to_numpy()
    fit = daily_df["fitness"].to_numpy()
    fat = daily_df["fatigue"].to_numpy()
    perf = daily_df["performance"].to_numpy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=dates,
            y=load,
            name=f"Load ({metric})",
            opacity=0.35,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=dates, y=fit, name="Fitness",
            mode="lines+markers",
            line=dict(color="#00FF88", width=2, shape="spline"),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=fat, name="Fatigue",
            mode="lines+markers",
            line=dict(color="#FF5555", width=2, shape="spline"),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=perf, name="Performance",
            mode="lines+markers",
            line=dict(color="#66CCFF", width=3, shape="spline"),
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Datum")
    fig.update_yaxes(title_text=f"Load ({metric})", secondary_y=False)
    fig.update_yaxes(title_text="Fitness / Fatigue / Performance (relatief)", secondary_y=True)

    fig.update_layout(
        title=f"FFP – {player}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=10, r=10, t=50, b=10),
        bargap=0.15,
    )

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def ffp_pages_main(df_gps: pd.DataFrame):
    st.header("Fitness–Fatigue–Performance")

    required_cols = {COL_PLAYER}
    missing = required_cols - set(df_gps.columns)
    if missing:
        st.error(f"Ontbrekende verplichte kolommen in GPS-sheet: {missing}")
        return

    df_summary = _filter_summary_only(df_gps)

    metrics = _detect_load_metrics(df_summary)
    if not metrics:
        st.error("Geen geschikte load-parameters gevonden in de GPS-data.")
        return

    players = sorted(df_summary[COL_PLAYER].dropna().astype(str).unique().tolist())
    if not players:
        st.warning("Geen spelers gevonden in Summary-data.")
        return

    default_metric_idx = metrics.index(DEFAULT_FFP_METRIC) if DEFAULT_FFP_METRIC in metrics else 0

    col_sel1, col_sel2, col_sel3 = st.columns([1.4, 1.4, 1.0])
    with col_sel1:
        player = st.selectbox("Kies speler", players)

    with col_sel2:
        metric = st.selectbox(
            "Kies load-parameter",
            metrics,
            index=default_metric_idx,
            help=f"Default staat op '{DEFAULT_FFP_METRIC}'.",
        )

    with col_sel3:
        x_mode = st.radio(
            "X-as",
            options=["Week", "Dag"],
            index=0,
            help="Kies of je het model per week of per dag wilt bekijken.",
        )

    with st.sidebar:
        with st.expander("FFP modelinstellingen", expanded=False):
            if x_mode == "Week":
                tau_help_unit = "weken"
                default_tau_fit = 4.0
                default_tau_fat = 1.5
            else:
                tau_help_unit = "dagen"
                default_tau_fit = 28.0
                default_tau_fat = 10.5

            tau_fit = st.number_input(
                f"Tau fitness ({tau_help_unit})",
                min_value=0.5,
                max_value=120.0,
                value=float(default_tau_fit),
                step=0.5,
                key="ffp_tau_fit_sidebar",
            )
            tau_fat = st.number_input(
                f"Tau fatigue ({tau_help_unit})",
                min_value=0.5,
                max_value=120.0,
                value=float(default_tau_fat),
                step=0.5,
                key="ffp_tau_fat_sidebar",
            )
            k_fit = st.number_input(
                "Gain fitness (k_f)",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                key="ffp_k_fit_sidebar",
            )
            k_fat = st.number_input(
                "Gain fatigue (k_d)",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
                key="ffp_k_fat_sidebar",
            )

            st.caption("Grotere tau = effect blijft langer hangen. Fatigue heeft meestal kortere tau en hogere gain.")

    if x_mode == "Week":
        time_df = _weekly_load_for_player(df_summary, player, metric)
        if time_df.empty:
            st.warning("Geen wekelijkse load gevonden voor deze selectie.")
            return

        load = time_df["load"].to_numpy()
        F, D, P = _impulse_response_model(
            load,
            tau_fit=float(tau_fit),
            tau_fat=float(tau_fat),
            k_fit=float(k_fit),
            k_fat=float(k_fat),
        )

        time_df["fitness"] = F
        time_df["fatigue"] = D
        time_df["performance"] = P

        _plot_ffp_week(time_df, metric, player)

        with st.expander("Data – wekelijkse waarden", expanded=False):
            st.dataframe(time_df, use_container_width=True)

    else:
        time_df = _daily_load_for_player(df_summary, player, metric)
        if time_df.empty:
            st.warning("Geen dagelijkse load gevonden voor deze selectie.")
            return

        load = time_df["load"].to_numpy()
        F, D, P = _impulse_response_model(
            load,
            tau_fit=float(tau_fit),
            tau_fat=float(tau_fat),
            k_fit=float(k_fit),
            k_fat=float(k_fat),
        )

        time_df["fitness"] = F
        time_df["fatigue"] = D
        time_df["performance"] = P

        _plot_ffp_day(time_df, metric, player)

        with st.expander("Data – dagelijkse waarden", expanded=False):
            st.dataframe(time_df, use_container_width=True)
