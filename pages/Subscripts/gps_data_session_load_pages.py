from __future__ import annotations

from datetime import date
from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

COL_DATE = "Datum"
COL_PLAYER = "Speler"
COL_EVENT = "Event"
COL_TYPE = "Type"

COL_TD = "Total Distance"
COL_SPRINT = "Sprint"
COL_HS = "High Sprint"
COL_ACC_TOT = "Total Accelerations"
COL_ACC_HI = "High Accelerations"
COL_DEC_TOT = "Total Decelerations"
COL_DEC_HI = "High Decelerations"

HR_COLS = ["HRzone1", "HRzone2", "HRzone3", "HRzone4", "HRzone5"]
TRIMP_CANDIDATES = ["HRTrimp", "HR Trimp", "HRtrimp", "Trimp", "TRIMP"]

MVV_RED = "#C8102E"
MVV_RED_LIGHT = "#E8213F"
MVV_RED_SOFT = "rgba(232,33,63,0.28)"
MVV_GREEN = "#00C46A"
MVV_ORANGE = "#F5A623"
MVV_BLUE = "#55A8FF"
MVV_BLUE_SOFT = "#A8D0FF"
TEXT = "#F5F7FB"
TEXT_MUTED = "rgba(245,247,251,0.68)"
GRID = "rgba(255,255,255,0.08)"
PLOT_BG = "rgba(255,255,255,0.018)"
GRAY_LINE = "rgba(255,255,255,0.48)"

SELECT_ALL_OPT = "— Select all —"


def _normalize_event(e: str) -> str:
    return "summary" if str(e).strip().lower() == "summary" else str(e).strip().lower()


def _prepare_gps(df_gps: pd.DataFrame) -> pd.DataFrame:
    df = df_gps.copy()
    if COL_DATE not in df.columns or COL_PLAYER not in df.columns:
        return df.iloc[0:0].copy()

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE, COL_PLAYER]).copy()

    if COL_EVENT in df.columns:
        df["_event_norm"] = df[COL_EVENT].map(_normalize_event)
        df = df[df["_event_norm"] == "summary"].copy()

    trimp_col = None
    for c in TRIMP_CANDIDATES:
        if c in df.columns:
            trimp_col = c
            break
    df["TRIMP"] = pd.to_numeric(df[trimp_col], errors="coerce").fillna(0.0) if trimp_col else 0.0

    numeric_cols = [
        COL_TD, COL_SPRINT, COL_HS, COL_ACC_TOT, COL_ACC_HI,
        COL_DEC_TOT, COL_DEC_HI, *HR_COLS, "TRIMP",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in [COL_PLAYER, COL_TYPE]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


def _available_days(df_calendar: pd.DataFrame) -> list[date]:
    if df_calendar is None or df_calendar.empty or COL_DATE not in df_calendar.columns:
        return []
    d = df_calendar.copy()
    d[COL_DATE] = pd.to_datetime(d[COL_DATE], errors="coerce").dt.date
    d = d.dropna(subset=[COL_DATE])
    return sorted(set(d[COL_DATE].tolist()))


def _pick_day_dateinput(df_calendar: pd.DataFrame, key_prefix: str = "sl") -> date:
    days = _available_days(df_calendar)
    if not days:
        return st.date_input("Datum", value=date.today(), key=f"{key_prefix}_date_input_empty")

    sel_key = f"{key_prefix}_selected"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = days[-1]

    picked = st.date_input(
        "Datum",
        value=st.session_state[sel_key],
        min_value=days[0],
        max_value=days[-1],
        key=f"{key_prefix}_date_input",
    )
    st.session_state[sel_key] = picked
    return picked


def _agg_by_player(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metric_cols = [
        COL_TD, COL_SPRINT, COL_HS, COL_ACC_TOT, COL_ACC_HI,
        COL_DEC_TOT, COL_DEC_HI, *HR_COLS, "TRIMP",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    return df.groupby(COL_PLAYER, as_index=False)[metric_cols].sum()


def _median_safe(a: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    return None if a.size == 0 else float(np.median(a))


def _median_for_players(df_agg: pd.DataFrame, players: list[str], col: str) -> float | None:
    if df_agg.empty or col not in df_agg.columns:
        return None
    sub = df_agg[df_agg[COL_PLAYER].astype(str).isin(players)]
    if sub.empty:
        return None
    return _median_safe(sub[col].to_numpy())


def _resolve_select_all(selected: list[str], players_all: list[str]) -> list[str]:
    if any(s == SELECT_ALL_OPT for s in selected):
        return players_all
    return [p for p in selected if p in players_all]


def _card_title(title: str, subtitle: str | None = None) -> None:
    html = f'<div style="margin:0 0 0.55rem 0;"><div style="font-size:0.8rem;letter-spacing:0.16em;text-transform:uppercase;color:{TEXT_MUTED};font-weight:700;">{title}</div>'
    if subtitle:
        html += f'<div style="font-size:0.92rem;color:{TEXT_MUTED};margin-top:0.28rem;">{subtitle}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _style_fig(fig: go.Figure, *, title: str, y_title: str, x_tickangle: int = -35, secondary_y_title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=20, color=TEXT)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT, size=12),
        margin=dict(l=30, r=20, t=70, b=85),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT),
        ),
        hovermode="closest",
        bargap=0.24,
    )
    fig.update_xaxes(
        tickangle=x_tickangle,
        showgrid=False,
        zeroline=False,
        tickfont=dict(color=TEXT, size=11),
        title=None,
        automargin=True,
    )
    fig.update_yaxes(
        title=y_title,
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        tickfont=dict(color=TEXT, size=11),
        title_font=dict(color=TEXT),
    )
    if secondary_y_title is not None:
        fig.update_yaxes(
            title_text=secondary_y_title,
            secondary_y=True,
            showgrid=False,
            zeroline=False,
            tickfont=dict(color=TEXT, size=11),
            title_font=dict(color=TEXT),
        )
    return fig


def _add_median_line(fig: go.Figure, y: float, label: str) -> None:
    fig.add_hline(
        y=y,
        line_dash="dot",
        line_width=1.8,
        line_color=GRAY_LINE,
        annotation_text=label,
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color=TEXT_MUTED,
    )


def _metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div style="
            border:1px solid rgba(255,255,255,0.08);
            border-radius:18px;
            padding:0.85rem 1rem;
            background:rgba(255,255,255,0.035);
            min-height:92px;">
            <div style="font-size:0.78rem;letter-spacing:0.14em;text-transform:uppercase;color:{TEXT_MUTED};font-weight:700;">{label}</div>
            <div style="font-size:1.55rem;color:{TEXT};font-weight:800;margin-top:0.35rem;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _team_selection_ui_inline(players_all: list[str]) -> tuple[bool, list[str], list[str]]:
    with st.expander("Team selectie", expanded=False):
        if "sl_team_sel_on" not in st.session_state:
            st.session_state["sl_team_sel_on"] = False
        if "sl_starters_raw" not in st.session_state:
            st.session_state["sl_starters_raw"] = []
        if "sl_subs_raw" not in st.session_state:
            st.session_state["sl_subs_raw"] = []

        enabled = st.toggle("Team selectie aan", value=st.session_state["sl_team_sel_on"], key="sl_team_sel_on")

        if not enabled:
            st.info("Zet dit aan om vaste selectie en wisselspelers te vergelijken.")
            return False, [], []

        opt_all = [SELECT_ALL_OPT] + players_all

        st.caption("Kies eerst de vaste selectie. De wisselspelers tonen daarna alleen de resterende spelers.")
        c1, c2 = st.columns(2, vertical_alignment="top")

        with c1:
            starters_raw = st.multiselect(
                "Vaste selectie",
                options=opt_all,
                default=[p for p in st.session_state["sl_starters_raw"] if p in opt_all],
                key="sl_starters_raw",
            )

        starters_resolved = _resolve_select_all(starters_raw, players_all)
        subs_pool = [p for p in players_all if p not in set(starters_resolved)]
        opt_all_subs = [SELECT_ALL_OPT] + subs_pool

        with c2:
            subs_raw = st.multiselect(
                "Wisselspelers",
                options=opt_all_subs,
                default=[p for p in st.session_state["sl_subs_raw"] if p in opt_all_subs],
                key="sl_subs_raw",
            )

        subs_resolved = _resolve_select_all(subs_raw, subs_pool)
        starters_final = starters_resolved
        subs_final = [p for p in subs_resolved if p not in set(starters_final)]

        return True, starters_final, subs_final


def _plot_total_distance(df_agg: pd.DataFrame, groups: dict[str, list[str]] | None):
    if COL_TD not in df_agg.columns:
        st.info("Kolom 'Total Distance' niet gevonden.")
        return

    data = df_agg.sort_values(COL_TD, ascending=False).reset_index(drop=True)
    players = data[COL_PLAYER].astype(str).tolist()
    vals = data[COL_TD].to_numpy()

    fig = go.Figure()
    fig.add_bar(
        x=players,
        y=vals,
        name="Total Distance",
        marker=dict(color=MVV_RED, line=dict(color="rgba(255,255,255,0.14)", width=1.4)),
        text=[f"{v:,.0f}".replace(",", " ") for v in vals],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Total Distance: %{y:,.0f} m<extra></extra>",
    )

    med_team = _median_safe(vals)
    if med_team is not None:
        _add_median_line(fig, med_team, f"Mediaan team: {med_team:,.0f} m".replace(",", " "))

    if groups:
        for gname, gplayers in groups.items():
            med = _median_for_players(df_agg, gplayers, COL_TD)
            if med is not None:
                _add_median_line(fig, med, f"{gname}: {med:,.0f} m".replace(",", " "))

    _style_fig(fig, title="Total Distance", y_title="Meters")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def _plot_sprint_hs(df_agg: pd.DataFrame, groups: dict[str, list[str]] | None):
    if COL_SPRINT not in df_agg.columns or COL_HS not in df_agg.columns:
        st.info("Sprint / High Sprint kolommen niet compleet.")
        return

    data = df_agg.sort_values(COL_SPRINT, ascending=False).reset_index(drop=True)
    players = data[COL_PLAYER].astype(str).tolist()
    x = np.arange(len(players))
    sprint_vals = data[COL_SPRINT].to_numpy()
    hs_vals = data[COL_HS].to_numpy()

    fig = go.Figure()
    fig.add_bar(
        x=x - 0.19, y=sprint_vals, width=0.36, name="Sprint",
        marker=dict(color=MVV_RED, line=dict(color="rgba(255,255,255,0.14)", width=1.2)),
        hovertemplate="<b>%{x}</b><br>Sprint: %{y:,.0f} m<extra></extra>",
    )
    fig.add_bar(
        x=x + 0.19, y=hs_vals, width=0.36, name="High Sprint",
        marker=dict(color=MVV_RED_LIGHT, line=dict(color="rgba(255,255,255,0.14)", width=1.2)),
        hovertemplate="<b>%{x}</b><br>High Sprint: %{y:,.0f} m<extra></extra>",
    )

    ms = _median_safe(sprint_vals)
    mh = _median_safe(hs_vals)
    if ms is not None:
        _add_median_line(fig, ms, f"Mediaan Sprint: {ms:,.0f} m".replace(",", " "))
    if mh is not None:
        _add_median_line(fig, mh, f"Mediaan High Sprint: {mh:,.0f} m".replace(",", " "))

    if groups:
        for gname, gplayers in groups.items():
            m1 = _median_for_players(df_agg, gplayers, COL_SPRINT)
            m2 = _median_for_players(df_agg, gplayers, COL_HS)
            if m1 is not None:
                _add_median_line(fig, m1, f"{gname} Sprint: {m1:,.0f} m".replace(",", " "))
            if m2 is not None:
                _add_median_line(fig, m2, f"{gname} HS: {m2:,.0f} m".replace(",", " "))

    fig.update_xaxes(tickvals=x, ticktext=players)
    _style_fig(fig, title="Sprint & High Sprint", y_title="Meters")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def _plot_acc_dec(df_agg: pd.DataFrame):
    have_cols = [c for c in [COL_ACC_TOT, COL_ACC_HI, COL_DEC_TOT, COL_DEC_HI] if c in df_agg.columns]
    if not have_cols:
        st.info("Geen Acceleration/Deceleration kolommen gevonden.")
        return

    sort_col = COL_ACC_TOT if COL_ACC_TOT in df_agg.columns else have_cols[0]
    data = df_agg.sort_values(sort_col, ascending=False).reset_index(drop=True)
    players = data[COL_PLAYER].astype(str).tolist()
    x = np.arange(len(players))
    w = 0.18

    fig = go.Figure()
    if COL_ACC_TOT in data.columns:
        fig.add_bar(x=x - 1.5 * w, y=data[COL_ACC_TOT], width=w, name="Total Acc", marker_color=MVV_RED)
    if COL_ACC_HI in data.columns:
        fig.add_bar(x=x - 0.5 * w, y=data[COL_ACC_HI], width=w, name="High Acc", marker_color=MVV_RED_LIGHT)
    if COL_DEC_TOT in data.columns:
        fig.add_bar(x=x + 0.5 * w, y=data[COL_DEC_TOT], width=w, name="Total Dec", marker_color=MVV_BLUE)
    if COL_DEC_HI in data.columns:
        fig.add_bar(x=x + 1.5 * w, y=data[COL_DEC_HI], width=w, name="High Dec", marker_color=MVV_BLUE_SOFT)

    fig.update_xaxes(tickvals=x, ticktext=players)
    _style_fig(fig, title="Accelerations / Decelerations", y_title="Aantal (N)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def _plot_hr_trimp(df_agg: pd.DataFrame):
    have_hr = [c for c in HR_COLS if c in df_agg.columns]
    has_trimp = "TRIMP" in df_agg.columns
    if not have_hr and not has_trimp:
        st.info("Geen HR zone/TRIMP data gevonden.")
        return

    players = df_agg[COL_PLAYER].astype(str).tolist()
    base_x = np.arange(len(players))

    fig = make_subplots(specs=[[{"secondary_y": has_trimp}]])
    color_map = {
        "HRzone1": "rgba(190,190,190,0.88)",
        "HRzone2": "rgba(84,168,255,0.88)",
        "HRzone3": "rgba(0,196,106,0.88)",
        "HRzone4": "rgba(245,166,35,0.90)",
        "HRzone5": "rgba(232,33,63,0.92)",
    }

    if have_hr:
        n = len(have_hr)
        group_w = 0.8
        bar_w = group_w / max(n, 1)
        start = -group_w / 2 + bar_w / 2
        for idx, z in enumerate(have_hr):
            x = base_x + (start + idx * bar_w)
            fig.add_bar(
                x=x,
                y=df_agg[z],
                name=z,
                marker_color=color_map.get(z, "gray"),
                width=bar_w * 0.95,
                secondary_y=False,
            )

    if has_trimp:
        fig.add_trace(
            go.Scatter(
                x=base_x,
                y=df_agg["TRIMP"],
                mode="lines+markers",
                name="HR Trimp",
                line=dict(color=MVV_GREEN, width=3, shape="spline"),
                marker=dict(size=7),
            ),
            secondary_y=True,
        )

    fig.update_xaxes(tickvals=base_x, ticktext=players)
    _style_fig(fig, title="Time in HR Zone", y_title="Time in zone (min)", secondary_y_title="HR Trimp")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def session_load_pages_main(
    df_gps_scope: pd.DataFrame,
    calendar_df_all: Optional[pd.DataFrame] = None,
    fetch_day_fn: Optional[Callable[[str], pd.DataFrame]] = None,
):
    _card_title("Session Load", "Dagselectie, teamvergelijking en loadverdeling per speler.")

    cal_df = calendar_df_all if calendar_df_all is not None else df_gps_scope

    with st.expander("Kalender", expanded=True):
        selected_day = _pick_day_dateinput(cal_df, key_prefix="sl")

    st.caption(f"Geselecteerd: {selected_day.strftime('%d-%m-%Y')}")

    df_work = df_gps_scope.copy()
    if fetch_day_fn is not None:
        try:
            tmp = df_work.copy()
            tmp[COL_DATE] = pd.to_datetime(tmp[COL_DATE], errors="coerce")
            has_day = bool((tmp[COL_DATE].dt.date == selected_day).any())
        except Exception:
            has_day = False
        if not has_day:
            day_df = fetch_day_fn(selected_day.isoformat())
            if day_df is not None and not day_df.empty:
                df_work = pd.concat([df_work, day_df], ignore_index=True)

    df = _prepare_gps(df_work)
    if df.empty:
        st.warning("Geen bruikbare GPS-data gevonden.")
        return

    df_day = df[df[COL_DATE].dt.date == selected_day].copy()
    if df_day.empty:
        st.info("Geen data op deze datum.")
        return

    df_agg = _agg_by_player(df_day)
    if df_agg.empty:
        st.warning("Geen data om te aggregeren.")
        return

    players_all = sorted(df_agg[COL_PLAYER].astype(str).unique().tolist())
    team_on, starters, subs = _team_selection_ui_inline(players_all)

    groups: dict[str, list[str]] | None = None
    if team_on:
        groups = {}
        if starters:
            groups["Vaste selectie"] = starters
        if subs:
            groups["Wissels"] = subs

    median_td = _median_safe(df_agg[COL_TD].to_numpy()) if COL_TD in df_agg.columns else None
    median_sprint = _median_safe(df_agg[COL_SPRINT].to_numpy()) if COL_SPRINT in df_agg.columns else None
    median_trimp = _median_safe(df_agg["TRIMP"].to_numpy()) if "TRIMP" in df_agg.columns else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric_card("Spelers", str(len(players_all)))
    with c2:
        _metric_card("Mediaan TD", f"{median_td:,.0f} m".replace(",", " ") if median_td is not None else "–")
    with c3:
        _metric_card("Mediaan Sprint", f"{median_sprint:,.0f} m".replace(",", " ") if median_sprint is not None else "–")
    with c4:
        _metric_card("Mediaan TRIMP", f"{median_trimp:,.1f}".replace(",", " ") if median_trimp is not None else "–")

    col_top1, col_top2 = st.columns(2)
    with col_top1:
        _plot_total_distance(df_agg, groups)
    with col_top2:
        _plot_sprint_hs(df_agg, groups)

    col_bot1, col_bot2 = st.columns(2)
    with col_bot1:
        _plot_acc_dec(df_agg)
    with col_bot2:
        _plot_hr_trimp(df_agg)
