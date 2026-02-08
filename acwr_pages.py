# acwr_pages.py
# ============================================================
# ACWR-dashboard + threshold-planner + Targets vs Workload
# Gebruikt GPS-database (per sessie) met kolommen:
#   'Week', 'Year', 'Speler', 'Event' + load-parameters
#   Alleen Event == 'Summary' wordt gebruikt.
#
# FIX (jaarwisseling):
# - Gebruik Year + Week als leidend (week_key = Year*100 + Week)
# - Sorteer/bereken ACWR op week_key (bv. 202552, 202601)
# - Toon labels "2026-W01"
# - Datum wordt alleen nog gebruikt als fallback (als Year ontbreekt)
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# CONSTANTEN / CONFIG
# ------------------------------------------------------------

# ACWR sweet-spot grenzen
SWEET_SPOT_LOW  = 0.80
SWEET_SPOT_HIGH = 1.30

# GPS kolomnamen in jouw Database.xlsx
COL_WEEK   = "Week"
COL_YEAR   = "Year"   # <-- jij hebt "Year" gemaakt
COL_DATE   = "Datum"  # fallback (optioneel)
COL_PLAYER = "Speler"
COL_EVENT  = "Event"

# Kolommen die NIET als ACWR-parameter gebruikt mogen worden
EXCLUDE_METRICS = {
    "Max Speed",
    "Avg Speed",
    "Avg HR",
    "Max HR",
}
# Geen /min-kolommen in ACWR
EXCLUDE_SUFFIXES = ("/min",)

# Voorkeursmetrics als default-selectie
DEFAULT_PREF_METRICS = [
    "Total Distance",
    "Sprint",
    "High Sprint",
    "playerload2D",
]

# ------------------------------------------------------------
# HELPER-FUNCTIES – DATA
# ------------------------------------------------------------

def _normalize_event(e: str) -> str:
    s = str(e).strip().lower()
    if s == "summary":
        return "summary"
    return s


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Fallback-only: parse Datum (als Year ontbreekt).
    NL-format dd-mm-YYYY / d-m-YYYY werkt met dayfirst=True.
    """
    out = df.copy()
    if col not in out.columns:
        return out
    s = out[col]
    if np.issubdtype(s.dtype, np.datetime64):
        return out
    out[col] = pd.to_datetime(s, errors="coerce", dayfirst=True, utc=False)
    return out


def _add_week_key_from_year_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bouw week_key uit Year + Week:
      week_key = Year*100 + Week
      week_label = YYYY-WWW
    """
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
        if pd.notna(r[COL_YEAR]) and pd.notna(r[COL_WEEK])
        else None,
        axis=1,
    )
    return out


def _add_iso_week_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback: ISO-jaar/week uit Datum.
    Maakt:
      iso_year, iso_week, week_key (YYYYWW), week_label (YYYY-WWW)
    """
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
        if pd.notna(r.get("iso_year")) and pd.notna(r.get("iso_week"))
        else None,
        axis=1,
    )
    return out


def detect_metrics_from_gps(df_gps: pd.DataFrame):
    """
    Zoek automatisch alle load-parameters in de GPS-database die we
    voor ACWR willen gebruiken.
    - Negeert kolommen voor week/speler.
    - Negeert Max Speed, Avg Speed, HR, en alle /min-kolommen.
    - Pakt alleen numerieke kolommen.
    """
    base_cols = {
        COL_WEEK, COL_YEAR, COL_DATE, COL_PLAYER, "Type", COL_EVENT,
        "Hoofdpositie", "Subpositie", "Subpositie ", "Wedstrijd", "Opponent",
        # interne/afgeleide kolommen
        "EVENT_NORM", "iso_year", "iso_week", "week_key", "week_label",
    }

    candidates = []
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


def make_weekly_from_gps(df_gps: pd.DataFrame, metrics):
    """
    Maak week-niveau dataset:
      week_key, week_label, player, metrics...
    Weekload = som over alle sessies in die week.
    Alleen Event == 'Summary' wordt gebruikt als Event-kolom aanwezig is.

    Leidende logic:
    1) Year + Week -> week_key (beste, lost jaarwisseling op)
    2) Fallback: ISO-week uit Datum (als Year ontbreekt)
    3) Laatste fallback: alleen Week (zonder jaar, kan fout rond jaarwisseling)
    """
    df = df_gps.copy()

    # Filter op Event == Summary (indien aanwezig)
    if COL_EVENT in df.columns:
        df["EVENT_NORM"] = df[COL_EVENT].map(_normalize_event)
        df = df[df["EVENT_NORM"] == "summary"].copy()

    # 1) Prefer: Year+Week
    df = _add_week_key_from_year_week(df)

    # 2) Fallback: ISO-week uit Datum als week_key grotendeels leeg is
    if "week_key" not in df.columns or df["week_key"].isna().mean() > 0.50:
        if COL_DATE in df.columns:
            df = _add_iso_week_fields(df)

    # 3) Laatste fallback: alleen Week (zonder jaar)
    if "week_key" not in df.columns or df["week_key"].isna().all():
        if COL_WEEK not in df.columns:
            raise ValueError(
                f"Kan geen week maken: '{COL_YEAR}'+ '{COL_WEEK}' ontbreekt en ook geen '{COL_DATE}'."
            )
        wk = pd.to_numeric(df[COL_WEEK], errors="coerce").astype("Int64")
        df["week_key"] = wk
        df["week_label"] = wk.apply(lambda x: f"W{int(x):02d}" if pd.notna(x) else None)

    # Drop rows zonder week_key of speler
    df = df.dropna(subset=["week_key", COL_PLAYER]).copy()

    # Groepeer per speler per week_key
    grp = df.groupby(["week_key", "week_label", COL_PLAYER], as_index=False)[metrics].sum()

    # Hernoem
    grp = grp.rename(columns={COL_PLAYER: "player"})
    return grp


def compute_acwr(df: pd.DataFrame, metrics, group_col="player", week_col="week_key"):
    """
    Bereken ACWR per metric per group (speler of 'Team').
    ACWR = week_load / (gemiddelde van de vorige 4 weken).
    Sorteert op week_key (YYYYWW) zodat jaarwisseling klopt.
    """
    df = df.copy()
    df[week_col] = pd.to_numeric(df[week_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[week_col]).copy()
    df = df.sort_values([group_col, week_col])

    for m in metrics:
        grp = df.groupby(group_col)[m]
        chronic = grp.shift(1).rolling(window=4, min_periods=2).mean()
        df[f"{m}_ACWR"] = df[m] / chronic

    return df


def make_team_level(df: pd.DataFrame, metrics, week_col="week_key"):
    """
    Maak team-level aggregatie (som per week over alle spelers).
    """
    team = (
        df.groupby([week_col, "week_label"])[metrics]
        .sum()
        .reset_index()
    )
    team["player"] = "Team"
    cols = [week_col, "week_label", "player"] + metrics
    team = team[cols]
    return team


def compute_chronic_last4weeks(df: pd.DataFrame, metrics, group_col="player", week_col="week_key"):
    """
    Bereken chronic load = gemiddelde van de laatste 4 beschikbare weken
    voor elke group (speler of team).
    """
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

# ------------------------------------------------------------
# HELPER-FUNCTIE – ACWR GRAFIEK (Plotly)
# ------------------------------------------------------------

def line_chart_acwr(df_view, param, week_col="week_key", label_col="week_label", group_label=None, highlight_week=None):
    """
    ACWR-grafiek met:
    - spline-lijn + markers
    - rode/groene zones
    - verticale highlight-week (op week_key)
    - voor Team (globaal): blauwe SD-band rond de lijn (spline)
    X-as: week_label (tekst) maar gesorteerd via week_key.
    """
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
                x=x_labels,
                y=upper,
                mode="lines",
                line=dict(color="rgba(0,150,255,0)", shape="spline"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=lower,
                mode="lines",
                line=dict(color="rgba(0,150,255,0)", shape="spline"),
                fill="tonexty",
                fillcolor="rgba(0,150,255,0.22)",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_hrect(y0=0.0, y1=SWEET_SPOT_LOW,  line_width=0, fillcolor="#8B0000", opacity=0.25, layer="below")
    fig.add_hrect(y0=SWEET_SPOT_LOW, y1=SWEET_SPOT_HIGH, line_width=0, fillcolor="#006400", opacity=0.30, layer="below")
    fig.add_hrect(y0=SWEET_SPOT_HIGH, y1=max_y,           line_width=0, fillcolor="#8B0000", opacity=0.25, layer="below")
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

def _compute_player_targets(df_weekly: pd.DataFrame, metrics, ratio_low: float, ratio_high: float) -> pd.DataFrame:
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
            rows.append({
                "player": p,
                "metric": m,
                "chronic": chronic_val,
                "target_low": ratio_low * chronic_val,
                "target_high": ratio_high * chronic_val,
            })
    if not rows:
        return pd.DataFrame(columns=["player", "metric", "chronic", "target_low", "target_high"])
    return pd.DataFrame(rows)


def _compute_team_targets(df_weekly: pd.DataFrame, metrics, ratio_low: float, ratio_high: float) -> pd.DataFrame:
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
            rows.append({
                "group": "Team",
                "metric": m,
                "chronic": chronic_val,
                "target_low": ratio_low * chronic_val,
                "target_high": ratio_high * chronic_val,
            })
    if not rows:
        return pd.DataFrame(columns=["group", "metric", "chronic", "target_low", "target_high"])
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

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            df["target_high_abs"] > 0,
            df["actual_abs"] / df["target_high_abs"],
            np.nan,
        )
    df["ratio"] = ratio

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
        "green",
        "red_missing",
        "red_excess",
    ]].copy()


def _build_target_bar_figure(df_bar: pd.DataFrame, metric: str, week_label: str, title_prefix="Week target"):
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

    fig.update_layout(
        barmode="stack",
        title=f"{title_prefix}: {metric} ({week_label})",
        margin=dict(l=10, r=10, t=40, b=40),
        showlegend=False,
    )
    fig.update_yaxes(title="%", tickformat=".0%", range=[0, max_total * 1.08])
    fig.update_xaxes(tickangle=-60)

    return fig


def _build_target_table(dfs_for_table: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
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
        return pd.DataFrame(
            columns=["Player", "Parameter", "Actual", "Min target", "Max target",
                     "Remaining to min", "Remaining to max"]
        )
    return pd.DataFrame(rows)

# ------------------------------------------------------------
# HOOFDFUNCTIE VOOR STREAMLIT-APP
# ------------------------------------------------------------

def acwr_pages_main(df_gps: pd.DataFrame):
    """
    Entry point vanuit app.py
    df_gps: GPS-database ingelezen uit werkblad 'GPS'.
    """

    # --------------------------------------------------------
    # CHECK: Year/Week beschikbaar?
    # --------------------------------------------------------
    if COL_YEAR not in df_gps.columns:
        st.warning(
            f"Kolom '{COL_YEAR}' ontbreekt. Voeg 'Year' toe voor correcte weekvolgorde over jaarwisseling."
        )

    # --------------------------------------------------------
    # DATA: metrics detecteren + weekdataset
    # --------------------------------------------------------
    metrics = detect_metrics_from_gps(df_gps)
    if not metrics:
        st.error("Geen geschikte load-parameters gevonden in de GPS-data.")
        return

    df_weekly = make_weekly_from_gps(df_gps, metrics)

    # Per speler
    df_acwr_players = compute_acwr(df_weekly, metrics, group_col="player", week_col="week_key")

    # Team-niveau
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

    # Standaardselectie metrics: voorkeur, anders eerste 4
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
            view_mode = st.radio(
                "Niveau",
                ["Per speler", "Team (globaal)"],
                key="acwr_dashboard_level",
            )

        with col_sel2:
            if view_mode == "Per speler":
                selected_player = st.selectbox(
                    "Kies speler",
                    players_sorted,
                    key="acwr_dashboard_player",
                )
            else:
                selected_player = "Team"

        with col_sel3:
            if weeks_keys:
                label_opts = [weekkey_to_label[wk] for wk in weeks_keys]
                idx_default = len(weeks_keys) - 1
                sel_label = st.selectbox(
                    "Week highlight",
                    options=label_opts,
                    index=idx_default,
                    key="acwr_dashboard_week_label",
                )
                label_to_key = {v: k for k, v in weekkey_to_label.items()}
                selected_week_key = label_to_key.get(sel_label)
            else:
                selected_week_key = None

        selected_params = st.multiselect(
            "Kies parameters (max 4 tegelijk)",
            options=metrics,
            default=default_metrics,
            key="acwr_dashboard_params",
        )
        selected_params = selected_params[:4]

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
    # TAB 2: THRESHOLD PLANNER VOOR VOLGENDE WEEK
    # ========================================================
    with tab_thresholds:
        st.header("Threshold planner voor volgende week")

        st.markdown(
            "Bereken op basis van chronische load (gemiddelde van **laatste 4 weken**) "
            "welke absolute load-range hoort bij een bepaalde ACWR-ratio voor de **volgende week**."
        )

        col_thr1, col_thr2, col_thr3 = st.columns([1, 1, 2])

        with col_thr1:
            ratio_low = st.number_input(
                "Ondergrens ratio (bijv. 0.8)",
                value=0.8,
                step=0.05,
                key="thr_ratio_low",
            )
        with col_thr2:
            ratio_high = st.number_input(
                "Bovengrens ratio (bijv. 1.0)",
                value=1.0,
                step=0.05,
                key="thr_ratio_high",
            )

        if ratio_low <= 0 or ratio_high <= 0:
            st.error("Ratio-grenzen moeten > 0 zijn.")
            return
        if ratio_low > ratio_high:
            st.error("Ondergrens mag niet groter zijn dan bovengrens.")
            return

        with col_thr3:
            params_thr = st.multiselect(
                "Kies parameters voor thresholds",
                options=metrics,
                default=default_metrics,
                key="thr_params",
            )

        if not params_thr:
            st.warning("Kies minstens één parameter voor de threshold-berekening.")
            return

        # ---------- Per speler ----------
        st.subheader("Per speler")

        chronic_players = compute_chronic_last4weeks(
            df=df_weekly,
            metrics=params_thr,
            group_col="player",
            week_col="week_key",
        )

        if chronic_players.empty:
            st.warning("Onvoldoende data om chronic load per speler te berekenen.")
        else:
            thr_rows = []
            for _, row in chronic_players.iterrows():
                player_name = row["player"]
                for p in params_thr:
                    chronic_val = row[p]
                    thr_rows.append({
                        "player": player_name,
                        "parameter": p,
                        "chronic_mean_last4w": chronic_val,
                        f"target_low (ACWR={ratio_low})": ratio_low * chronic_val,
                        f"target_high (ACWR={ratio_high})": ratio_high * chronic_val,
                    })

            df_thr_players = pd.DataFrame(thr_rows)
            st.dataframe(df_thr_players, width="stretch")

        # ---------- Team ----------
        st.subheader("Team (globaal)")

        team_full = make_team_level(df_weekly, params_thr, week_col="week_key")
        chronic_team = compute_chronic_last4weeks(
            df=team_full,
            metrics=params_thr,
            group_col="player",
            week_col="week_key",
        )

        if chronic_team.empty:
            st.warning("Onvoldoende data om chronic load voor het team te berekenen.")
        else:
            thr_rows_team = []
            for _, row in chronic_team.iterrows():
                for p in params_thr:
                    chronic_val = row[p]
                    thr_rows_team.append({
                        "group": "Team",
                        "parameter": p,
                        "chronic_mean_last4w": chronic_val,
                        f"target_low (ACWR={ratio_low})": ratio_low * chronic_val,
                        f"target_high (ACWR={ratio_high})": ratio_high * chronic_val,
                    })

            df_thr_team = pd.DataFrame(thr_rows_team)
            st.dataframe(df_thr_team, width="stretch")

    # ========================================================
    # TAB 3: TARGETS vs WORKLOAD
    # ========================================================
    with tab_targets:
        st.header("Targets vs Workload")

        st.markdown(
            "Vergelijk de ingestelde targets (op basis van chronische load × ratio) "
            "met de **werkelijk behaalde weekload**. "
            "De barcharts tonen het percentage van de target, "
            "de tabel daaronder toont de **absolute** waarden."
        )

        if not weeks_keys:
            st.warning("Geen weken gevonden in de data.")
            return

        col_t1, col_t2, col_t3 = st.columns([1.2, 1.2, 1.4])
        with col_t1:
            target_level = st.radio(
                "Niveau",
                ["Per speler", "Team (globaal)"],
                key="targets_level",
            )
        with col_t2:
            if target_level == "Per speler":
                player_opts = ["Alle spelers"] + players_sorted
                target_player = st.selectbox(
                    "Kies speler",
                    player_opts,
                    key="targets_player",
                )
            else:
                target_player = "Team"
                st.markdown("**Team (globaal)** geselecteerd.")
        with col_t3:
            label_opts = [weekkey_to_label[wk] for wk in weeks_keys]
            idx_default = len(weeks_keys) - 1
            sel_label = st.selectbox(
                "Week (werkelijk load voor vergelijking)",
                options=label_opts,
                index=idx_default,
                key="targets_week_label",
            )
            label_to_key = {v: k for k, v in weekkey_to_label.items()}
            target_week_key = label_to_key.get(sel_label)
            target_week_label = sel_label

        st.markdown("---")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            ratio_low_t = st.number_input(
                "Ondergrens ratio (bijv. 0.8)",
                value=0.8,
                step=0.05,
                key="targets_ratio_low",
            )
        with col_r2:
            ratio_high_t = st.number_input(
                "Bovengrens ratio (bijv. 1.0)",
                value=1.0,
                step=0.05,
                key="targets_ratio_high",
            )

        if ratio_low_t <= 0 or ratio_high_t <= 0:
            st.error("Ratio-grenzen moeten > 0 zijn.")
            return
        if ratio_low_t > ratio_high_t:
            st.error("Ondergrens mag niet groter zijn dan bovengrens.")
            return

        params_target = st.multiselect(
            "Kies parameters (max 4 tegelijk)",
            options=metrics,
            default=default_metrics,
            key="targets_params",
        )
        params_target = params_target[:4]

        if not params_target:
            st.warning("Kies minstens één parameter.")
            return

        targets_players = _compute_player_targets(df_weekly, metrics, ratio_low_t, ratio_high_t)
        targets_team = _compute_team_targets(df_weekly, metrics, ratio_low_t, ratio_high_t)

        if targets_players.empty and targets_team.empty:
            st.warning("Geen targets beschikbaar (onvoldoende data voor laatste 4 weken).")
            return

        dfs_for_table = []
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
                    fig_bar = _build_target_bar_figure(df_bar, p, target_week_label)
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
