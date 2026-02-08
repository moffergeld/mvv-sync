# ffp_pages.py
# ============================================
# Fitness–Fatigue–Performance per speler
# - Week-modus: identiek aan oude implementatie
# - Extra: Dag-modus (per datum)
# ============================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COL_PLAYER = "Speler"
COL_DATE   = "Datum"
COL_EVENT  = "Event"


# ------------------------------------------------------------
# Helpers om kolommen te vinden
# ------------------------------------------------------------

def _find_week_col(df: pd.DataFrame) -> str | None:
    """Zoek naar een kolom die 'week' heet (case-insensitive)."""
    for c in df.columns:
        if str(c).strip().lower() == "week":
            return c
    return None


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    return df


def _detect_load_metrics(df: pd.DataFrame) -> list[str]:
    """
    Zoek alle numerieke load-kolommen, waarbij we hetzelfde filter gebruiken
    als bij ACWR: geen Max/Avg speed/HR, geen /min-kolommen, geen ID-kolommen.
    """
    exclude_id = {
        "Speler", "Hoofdpositie", "Subpositie", "Datum", "Week",
        "Type", "Event", "Wedstrijd", "Opponent", "Tegenstander"
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


# ------------------------------------------------------------
# Weekly load (identiek aan je oude script)
# ------------------------------------------------------------

def _weekly_load_for_player(
    df_gps: pd.DataFrame,
    player: str,
    metric: str,
) -> pd.DataFrame:
    """
    Aggregeer load per week voor één speler.
    - Filtert op Event == 'Summary' indien beschikbaar.
    - Gebruikt kolom 'Week' indien aanwezig, anders ISO-week van Datum.
    Retourneert DataFrame met kolommen: week (int), load (float).
    """
    df = df_gps.copy()

    # Filter op speler
    df = df[df[COL_PLAYER] == player].copy()
    if df.empty:
        return pd.DataFrame(columns=["week", "load"])

    # Event 'Summary' gebruiken als die er is
    if COL_EVENT in df.columns:
        ev = df[COL_EVENT].astype(str).str.strip().str.lower()
        if (ev == "summary").any():
            df = df[ev == "summary"].copy()

    # Week-kolom bepalen
    week_col = _find_week_col(df)
    if week_col is None:
        # terugvallen op Datum → ISO-week
        if COL_DATE not in df.columns:
            raise ValueError("Geen 'Week'- of 'Datum'-kolom gevonden in GPS-data.")
        df = _ensure_datetime(df, COL_DATE)
        df["week"] = df[COL_DATE].dt.isocalendar().week.astype(int)
        week_col = "week"

    # Metric numeriek maken
    df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)

    weekly = (
        df.groupby(week_col, as_index=False)[metric]
          .sum()
          .rename(columns={week_col: "week", metric: "load"})
    )
    weekly["week"] = weekly["week"].astype(int)
    weekly = weekly.sort_values("week").reset_index(drop=True)
    return weekly


# ------------------------------------------------------------
# Daily load (nieuwe functionaliteit)
# ------------------------------------------------------------

def _daily_load_for_player(
    df_gps: pd.DataFrame,
    player: str,
    metric: str,
) -> pd.DataFrame:
    """
    Aggregeer load per dag voor één speler.
    - Filtert op Event == 'Summary' indien beschikbaar.
    - Gebruikt kolom 'Datum' (datetime).
    Retourneert DataFrame met kolommen: date (datetime.date), load (float).
    """
    df = df_gps.copy()

    # Filter op speler
    df = df[df[COL_PLAYER] == player].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "load"])

    # Event 'Summary' gebruiken als die er is
    if COL_EVENT in df.columns:
        ev = df[COL_EVENT].astype(str).str.strip().str.lower()
        if (ev == "summary").any():
            df = df[ev == "summary"].copy()

    if COL_DATE not in df.columns:
        raise ValueError("Geen 'Datum'-kolom gevonden in GPS-data.")
    df = _ensure_datetime(df, COL_DATE)
    df = df[df[COL_DATE].notna()].copy()

    # Metric numeriek maken
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
# Banister-achtig Fitness–Fatigue model
# (zelfde als in jouw oude script)
# ------------------------------------------------------------

def _impulse_response_model(
    load: np.ndarray,
    tau_fit: float = 4.0,
    tau_fat: float = 1.5,
    k_fit: float = 1.0,
    k_fat: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Eenvoudig Fitness–Fatigue model met één stap per tijdseenheid
    (week of dag, afhankelijk van de input-load).
    F_t = F_{t-1} * exp(-1/tau_fit) + k_fit * load_t
    D_t = D_{t-1} * exp(-1/tau_fat) + k_fat * load_t
    Performance_t = F_t - D_t
    """
    load = np.asarray(load, dtype=float)
    n = load.size
    F = np.zeros(n, dtype=float)
    D = np.zeros(n, dtype=float)

    alpha = np.exp(-1.0 / tau_fit)
    beta  = np.exp(-1.0 / tau_fat)

    for t in range(n):
        if t == 0:
            F[t] = k_fit * load[t]
            D[t] = k_fat * load[t]
        else:
            F[t] = F[t - 1] * alpha + k_fit * load[t]
            D[t] = D[t - 1] * beta  + k_fat * load[t]

    P = F - D
    return F, D, P


# ------------------------------------------------------------
# Plot: week-modus
# ------------------------------------------------------------

def _plot_ffp_week(weekly_df: pd.DataFrame, metric: str, player: str):
    """
    Eén figuur met:
    - bar: weekly load
    - lijn: Fitness
    - lijn: Fatigue
    - lijn: Performance
    Met een secundaire y-as voor load.
    """
    weeks = weekly_df["week"].to_numpy()
    load  = weekly_df["load"].to_numpy()
    fit   = weekly_df["fitness"].to_numpy()
    fat   = weekly_df["fatigue"].to_numpy()
    perf  = weekly_df["performance"].to_numpy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Load als bar op linker as
    fig.add_trace(
        go.Bar(
            x=weeks,
            y=load,
            name=f"Load ({metric})",
            opacity=0.35,
        ),
        secondary_y=False,
    )

    # Fitness / Fatigue / Performance op rechter as (spline-lijnen)
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=fit,
            name="Fitness",
            mode="lines+markers",
            line=dict(color="#00FF88", width=2, shape="spline"),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=fat,
            name="Fatigue",
            mode="lines+markers",
            line=dict(color="#FF5555", width=2, shape="spline"),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=perf,
            name="Performance",
            mode="lines+markers",
            line=dict(color="#66CCFF", width=3, shape="spline"),
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Week")

    fig.update_yaxes(title_text=f"Load ({metric})", secondary_y=False)
    fig.update_yaxes(title_text="Fitness / Fatigue / Performance (relatief)", secondary_y=True)

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=10, r=10, t=50, b=10),
        bargap=0.15,
    )

    st.plotly_chart(fig, width='stretch')


# ------------------------------------------------------------
# Plot: dag-modus
# ------------------------------------------------------------

def _plot_ffp_day(daily_df: pd.DataFrame, metric: str, player: str):
    """
    Zelfde idee als week-plot, maar dan per dag.
    """
    dates = daily_df["date"].to_numpy()
    load  = daily_df["load"].to_numpy()
    fit   = daily_df["fitness"].to_numpy()
    fat   = daily_df["fatigue"].to_numpy()
    perf  = daily_df["performance"].to_numpy()

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
            x=dates,
            y=fit,
            name="Fitness",
            mode="lines+markers",
            line=dict(color="#00FF88", width=2, shape="spline"),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=fat,
            name="Fatigue",
            mode="lines+markers",
            line=dict(color="#FF5555", width=2, shape="spline"),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=perf,
            name="Performance",
            mode="lines+markers",
            line=dict(color="#66CCFF", width=3, shape="spline"),
        ),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Datum")

    fig.update_yaxes(title_text=f"Load ({metric})", secondary_y=False)
    fig.update_yaxes(title_text="Fitness / Fatigue / Performance (relatief)", secondary_y=True)

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=10, r=10, t=50, b=10),
        bargap=0.15,
    )

    st.plotly_chart(fig, width='stretch')


# ------------------------------------------------------------
# Hoofdfunctie voor Streamlit
# ------------------------------------------------------------

def ffp_pages_main(df_gps: pd.DataFrame):
    """
    Hoofdpagina Fitness/Fatigue in de app.
    Verwacht de ruwe GPS-sheet als input (zoals acwr_pages_main).
    """
    st.header("Fitness–Fatigue–Performance")

    required_cols = {COL_PLAYER}
    missing = required_cols - set(df_gps.columns)
    if missing:
        st.error(f"Ontbrekende verplichte kolommen in GPS-sheet: {missing}")
        return

    metrics = _detect_load_metrics(df_gps)
    if not metrics:
        st.error("Geen geschikte load-parameters gevonden in de GPS-data.")
        return

    players = sorted(df_gps[COL_PLAYER].dropna().astype(str).unique().tolist())

    # -----------------------------
    # Selecties
    # -----------------------------
    col_sel1, col_sel2, col_sel3 = st.columns([1.4, 1.4, 1.0])
    with col_sel1:
        player = st.selectbox("Kies speler", players)

    with col_sel2:
        metric = st.selectbox("Kies load-parameter", metrics)

    with col_sel3:
        x_mode = st.radio(
            "X-as",
            options=["Week", "Dag"],
            index=0,
            help="Kies of je het model per week of per dag wilt bekijken.",
        )

    # -----------------------------
    # Modelinstellingen
    # -----------------------------
    with st.expander("Modelinstellingen (Fitness/Fatigue)", expanded=False):
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        if x_mode == "Week":
            tau_help_unit = "weken"
            default_tau_fit = 4.0
            default_tau_fat = 1.5
        else:
            tau_help_unit = "dagen"
            # ruwe equivalent van 4 en 1.5 weken → 28 & ~10.5 dagen
            default_tau_fit = 28.0
            default_tau_fat = 10.5

        with col_m1:
            tau_fit = st.number_input(
                f"Tau fitness ({tau_help_unit})",
                min_value=0.5, max_value=120.0,
                value=default_tau_fit,
                step=0.5,
            )
        with col_m2:
            tau_fat = st.number_input(
                f"Tau fatigue ({tau_help_unit})",
                min_value=0.5, max_value=120.0,
                value=default_tau_fat,
                step=0.5,
            )
        with col_m3:
            k_fit = st.number_input("Gain fitness (k_f)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        with col_m4:
            k_fat = st.number_input("Gain fatigue (k_d)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)

        st.markdown(
            "Bij grotere **tau** blijft het effect langer hangen. "
            "Fatigue heeft meestal een kortere tau en hogere gain dan fitness."
        )

    # -----------------------------
    # Data en model
    # -----------------------------
    if x_mode == "Week":
        time_df = _weekly_load_for_player(df_gps, player, metric)
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

        time_df["fitness"]     = F
        time_df["fatigue"]     = D
        time_df["performance"] = P

        # Plot + tabel
        _plot_ffp_week(time_df, metric, player)

        with st.expander("Data – wekelijkse waarden", expanded=False):
            st.dataframe(time_df, width='stretch')

    else:  # Dag
        time_df = _daily_load_for_player(df_gps, player, metric)
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

        time_df["fitness"]     = F
        time_df["fatigue"]     = D
        time_df["performance"] = P

        _plot_ffp_day(time_df, metric, player)

        with st.expander("Data – dagelijkse waarden", expanded=False):
            st.dataframe(time_df, width='stretch')
