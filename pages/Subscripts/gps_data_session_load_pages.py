from __future__ import annotations

from datetime import date
import re
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
COL_SPRINT = "Zone 5"
COL_HS = "Zone 6"
COL_HMLD = "HMLD"
COL_DSL = "Dynamic Stress Load"
COL_HR_EXERTION = "HR Exertion"
COL_FATIGUE = "Fatigue Index"
COL_EXTRA = "Extra Metrics"

HR_COLS = [f"HRzone{index}" for index in range(1, 7)]

MVV_RED = "#C8102E"
MVV_RED_LIGHT = "#E8213F"
MVV_RED_SOFT = "rgba(232,33,63,0.28)"
MVV_GREEN = "#00C46A"
MVV_ORANGE = "#F5A623"
TEXT = "#F5F7FB"
TEXT_MUTED = "rgba(245,247,251,0.68)"
GRID = "rgba(255,255,255,0.08)"
PLOT_BG = "rgba(255,255,255,0.018)"
GRAY_LINE = "rgba(255,255,255,0.48)"

SELECT_ALL_OPT = "— Select all —"


def _normalize_event(e: str) -> str:
    normalized = str(e).strip().lower()
    return "summary" if re.fullmatch(r"summary(?:\s*\(\d+\))?", normalized) else normalized


def _session_family(type_value: str) -> str:
    normalized = str(type_value).strip().lower()
    return "Wedstrijd" if normalized in {"match", "practice match"} else "Training"


def _session_display_label(type_value: str) -> str:
    raw_value = str(type_value).strip()
    if not raw_value:
        return "Onbekende sessie"
    return f"{_session_family(raw_value)} | {raw_value}"


def _extra_value(value: object, *keys: str) -> str:
    if not isinstance(value, dict):
        return ""
    normalized = {str(key).lower().replace(" ", ""): item for key, item in value.items()}
    for key in keys:
        item = normalized.get(key.lower().replace(" ", ""))
        if item is not None and str(item).strip():
            return str(item).strip()
    return ""


def _session_identity(row: pd.Series) -> str:
    extra = row.get(COL_EXTRA)
    session_id = _extra_value(extra, "Session ID", "sessionId")
    start = _extra_value(extra, "Session Start Time", "sessionStartTime")
    title = _extra_value(extra, "Session Title", "sessionTitle")
    session_type = str(row.get(COL_TYPE) or "Sessie").strip()
    return session_id or "|".join((session_type, start, title))


def _session_option_label(row: pd.Series) -> str:
    extra = row.get(COL_EXTRA)
    start = _extra_value(extra, "Session Start Time", "sessionStartTime")
    title = _extra_value(extra, "Session Title", "sessionTitle")
    time_label = ""
    if "T" in start:
        time_label = start.split("T", 1)[1][:5]
    elif len(start) >= 5:
        time_label = start[:5]
    pieces = [_session_display_label(str(row.get(COL_TYPE) or ""))]
    if time_label:
        pieces.append(time_label)
    if title and title.lower() not in str(row.get(COL_TYPE) or "").lower():
        pieces.append(title)
    return " · ".join(pieces)


SELECT_ALL_OPT = "-- Select all --"


def _session_label_map_for_day(df: pd.DataFrame, selected_day: date) -> dict[str, str]:
    if df.empty or COL_DATE not in df.columns or COL_TYPE not in df.columns:
        return {}

    day_df = df.copy()
    day_df[COL_DATE] = pd.to_datetime(day_df[COL_DATE], errors="coerce")
    day_df = day_df[day_df[COL_DATE].dt.date == selected_day].copy()
    if day_df.empty:
        return {}

    if COL_PLAYER in day_df.columns:
        player_counts = day_df.groupby(COL_TYPE, dropna=False)[COL_PLAYER].nunique().to_dict()
    else:
        player_counts = day_df.groupby(COL_TYPE, dropna=False).size().to_dict()

    label_map: dict[str, str] = {}
    for session_type, player_count in player_counts.items():
        session_key = str(session_type or "").strip()
        if not session_key:
            continue
        label_map[session_key] = f"{_session_display_label(session_key)} ({int(player_count)} spelers)"
    return label_map


def _prepare_gps(df_gps: pd.DataFrame) -> pd.DataFrame:
    df = df_gps.copy()
    if COL_DATE not in df.columns or COL_PLAYER not in df.columns:
        return df.iloc[0:0].copy()

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE, COL_PLAYER]).copy()

    if COL_EVENT in df.columns:
        df["_event_norm"] = df[COL_EVENT].map(_normalize_event)
        df = df[df["_event_norm"] == "summary"].copy()

    numeric_cols = [
        COL_TD, COL_SPRINT, COL_HS, COL_HMLD, COL_DSL, COL_HR_EXERTION, COL_FATIGUE,
        *HR_COLS,
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in [COL_PLAYER, COL_TYPE]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df["_session_key"] = df.apply(_session_identity, axis=1)
    df["_session_label"] = df.apply(_session_option_label, axis=1)

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
        return st.date_input(
            "Kalenderdatum",
            value=date.today(),
            key=f"{key_prefix}_date_input_empty",
            label_visibility="collapsed",
        )

    sel_key = f"{key_prefix}_selected"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = days[-1]
    else:
        current_value = st.session_state.get(sel_key)
        if current_value is None or current_value < days[0]:
            st.session_state[sel_key] = days[0]
        elif current_value > days[-1]:
            st.session_state[sel_key] = days[-1]

    picked = st.date_input(
        "Kalenderdatum",
        value=st.session_state[sel_key],
        min_value=days[0],
        max_value=days[-1],
        key=f"{key_prefix}_date_input",
        label_visibility="collapsed",
    )
    st.session_state[sel_key] = picked
    return picked


def pick_day_from_calendar(df_calendar: pd.DataFrame, key_prefix: str = "sl") -> date:
    return _pick_day_dateinput(df_calendar, key_prefix=key_prefix)


def _sessions_for_day(df: pd.DataFrame, selected_day: date) -> pd.DataFrame:
    if df.empty or COL_DATE not in df.columns or COL_TYPE not in df.columns:
        return pd.DataFrame(columns=["_session_key", "_session_label"])

    day_df = df.copy()
    day_df[COL_DATE] = pd.to_datetime(day_df[COL_DATE], errors="coerce")
    day_df = day_df[day_df[COL_DATE].dt.date == selected_day].copy()
    if day_df.empty:
        return pd.DataFrame(columns=["_session_key", "_session_label"])

    return (
        day_df[["_session_key", "_session_label"]]
        .drop_duplicates()
        .sort_values(["_session_label", "_session_key"])
        .reset_index(drop=True)
    )


def _agg_by_player(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metric_cols = [
        COL_TD, COL_SPRINT, COL_HS, COL_HMLD, COL_DSL, COL_HR_EXERTION, COL_FATIGUE, *HR_COLS,
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
        plot_bgcolor="rgba(255,255,255,0.012)",
        font=dict(color=TEXT, size=12),
        margin=dict(l=30, r=20, t=66, b=72),
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
            border:1px solid rgba(255,255,255,0.085);
            border-top-color:rgba(229,43,73,0.34);
            border-radius:14px;
            padding:0.85rem 1rem;
            background:linear-gradient(145deg, rgba(20,30,50,.94), rgba(12,18,31,.96));
            box-shadow:0 10px 28px rgba(0,0,0,.15);
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
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})


def _plot_player_gps_load(df_agg: pd.DataFrame) -> None:
    required = {COL_PLAYER, COL_TD, COL_SPRINT, COL_HS, COL_HMLD}
    if df_agg.empty or not required.issubset(df_agg.columns):
        st.info("Geen complete TD-, Zone 5-, Zone 6- en HMLD-data voor deze sessie.")
        return
    data = df_agg.sort_values(COL_TD, ascending=False).reset_index(drop=True)
    players = data[COL_PLAYER].astype(str)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=.12,
        row_heights=[.58, .42],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )
    fig.add_trace(
        go.Bar(
            x=players,
            y=data[COL_TD],
            name="TD",
            marker_color="#526986",
            text=[f"{float(value):,.0f}".replace(",", ".") for value in data[COL_TD]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>TD %{y:,.0f} m<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=players,
            y=data[COL_HMLD],
            name="HMLD",
            mode="lines+markers",
            line=dict(color="#69D5CB", width=3),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>HMLD %{y:,.0f} m<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    for column, label, color in ((COL_SPRINT, "Zone 5", "#EDB45B"), (COL_HS, "Zone 6", "#E84664")):
        fig.add_trace(
            go.Bar(
                x=players,
                y=data[column],
                name=label,
                marker_color=color,
                hovertemplate=f"<b>%{{x}}</b><br>{label} %{{y:,.0f}} m<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        height=560,
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT),
        legend=dict(orientation="h", y=1.07, x=0),
        margin=dict(l=30, r=30, t=55, b=90),
        hovermode="x unified",
    )
    fig.update_xaxes(tickangle=-45, showgrid=False, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title_text="Meters", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="HMLD (m)", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="Meters", row=2, col=1)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})


def _plot_single_metric(df_agg: pd.DataFrame, column: str, label: str, color: str, decimals: int = 1) -> None:
    if column not in df_agg.columns:
        st.info(f"Geen {label}-data beschikbaar voor deze sessie.")
        return
    data = df_agg[[COL_PLAYER, column]].dropna().sort_values(column, ascending=False)
    if data.empty:
        st.info(f"Geen {label}-data beschikbaar voor deze sessie.")
        return
    mean = float(data[column].mean())
    fig = go.Figure(
        go.Bar(
            x=data[COL_PLAYER],
            y=data[column],
            marker_color=color,
            text=[f"{float(value):.{decimals}f}" for value in data[column]],
            textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{label} %{{y:.{decimals}f}}<extra></extra>",
        )
    )
    fig.add_hline(y=mean, line_dash="dash", line_color="#F8FAFC", annotation_text=f"Gem. {mean:.{decimals}f}")
    _style_fig(fig, title=label, y_title=label)
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})


def _plot_sprint_hs(df_agg: pd.DataFrame, groups: dict[str, list[str]] | None):
    if COL_SPRINT not in df_agg.columns or COL_HS not in df_agg.columns:
        st.info("Zone 5 / Zone 6 kolommen niet compleet.")
        return

    data = df_agg.sort_values(COL_SPRINT, ascending=False).reset_index(drop=True)
    players = data[COL_PLAYER].astype(str).tolist()
    x = np.arange(len(players))
    sprint_vals = data[COL_SPRINT].to_numpy()
    hs_vals = data[COL_HS].to_numpy()

    fig = go.Figure()
    fig.add_bar(
        x=x - 0.19, y=sprint_vals, width=0.36, name="Zone 5",
        marker=dict(color=MVV_RED, line=dict(color="rgba(255,255,255,0.14)", width=1.2)),
        hovertemplate="<b>%{x}</b><br>Zone 5: %{y:,.0f} m<extra></extra>",
    )
    fig.add_bar(
        x=x + 0.19, y=hs_vals, width=0.36, name="Zone 6",
        marker=dict(color=MVV_RED_LIGHT, line=dict(color="rgba(255,255,255,0.14)", width=1.2)),
        hovertemplate="<b>%{x}</b><br>Zone 6: %{y:,.0f} m<extra></extra>",
    )

    ms = _median_safe(sprint_vals)
    mh = _median_safe(hs_vals)
    if ms is not None:
        _add_median_line(fig, ms, f"Mediaan Zone 5: {ms:,.0f} m".replace(",", " "))
    if mh is not None:
        _add_median_line(fig, mh, f"Mediaan Zone 6: {mh:,.0f} m".replace(",", " "))

    if groups:
        for gname, gplayers in groups.items():
            m1 = _median_for_players(df_agg, gplayers, COL_SPRINT)
            m2 = _median_for_players(df_agg, gplayers, COL_HS)
            if m1 is not None:
                _add_median_line(fig, m1, f"{gname} Sprint: {m1:,.0f} m".replace(",", " "))
            if m2 is not None:
                _add_median_line(fig, m2, f"{gname} HS: {m2:,.0f} m".replace(",", " "))

    fig.update_xaxes(tickvals=x, ticktext=players)
    _style_fig(fig, title="Zone 5 & Zone 6", y_title="Meters")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})


def _plot_hr_zones(df_agg: pd.DataFrame):
    have_hr = [c for c in HR_COLS if c in df_agg.columns]
    has_exertion = COL_HR_EXERTION in df_agg.columns
    if not have_hr and not has_exertion:
        st.info("Geen HR-zone- of HR Exertion-data gevonden.")
        return

    players = df_agg[COL_PLAYER].astype(str).tolist()
    fig = make_subplots(specs=[[{"secondary_y": has_exertion}]])
    color_map = {
        "HRzone1": "#526986",
        "HRzone2": "#4F91A5",
        "HRzone3": "#69D5CB",
        "HRzone4": "#EDB45B",
        "HRzone5": "#E9854F",
        "HRzone6": "#E84664",
    }

    for zone in have_hr:
        fig.add_bar(
            x=players,
            y=pd.to_numeric(df_agg[zone], errors="coerce").div(60),
            name=zone.replace("HRzone", "HR Z"),
            marker_color=color_map.get(zone, "gray"),
            secondary_y=False,
            hovertemplate=f"<b>%{{x}}</b><br>{zone.replace('HRzone', 'HR Z')}: %{{y:.1f}} min<extra></extra>",
        )

    if has_exertion:
        fig.add_trace(
            go.Scatter(
                x=players,
                y=df_agg[COL_HR_EXERTION],
                mode="lines+markers",
                name="HR Exertion",
                line=dict(color="#C58AF1", width=3),
                marker=dict(size=7),
            ),
            secondary_y=True,
        )

    fig.update_layout(barmode="stack")
    _style_fig(fig, title="HR-zones & HR Exertion", y_title="Tijd in zone (min)", secondary_y_title="HR Exertion")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})


def session_load_pages_main(
    df_gps_scope: pd.DataFrame,
    calendar_df_all: Optional[pd.DataFrame] = None,
    fetch_day_fn: Optional[Callable[[str], pd.DataFrame]] = None,
    selected_day: Optional[date] = None,
):
    cal_df = calendar_df_all if calendar_df_all is not None else df_gps_scope

    if selected_day is None:
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

    sessions = _sessions_for_day(df, selected_day)
    session_label_map = sessions.set_index("_session_key")["_session_label"].to_dict()
    session_options = [None] + sessions["_session_key"].tolist()
    selected_session = st.selectbox(
        "Sessie op deze dag",
        options=session_options,
        index=0,
        key=f"sl_session_type_{selected_day.isoformat()}",
        format_func=lambda option: (
            "Alle sessies"
            if option is None
            else session_label_map.get(str(option), str(option))
        ),
        help="Sessies worden gescheiden op STATSports-sessie en starttijd, zodat een dubbele training niet wordt samengevoegd.",
    )

    if selected_session is not None:
        df_day = df_day[df_day["_session_key"].astype(str) == str(selected_session)].copy()
        if df_day.empty:
            st.info("Geen data gevonden voor deze sessie op de gekozen datum.")
            return

    filter_caption = (
        f"Actieve sessie: {session_label_map.get(str(selected_session), str(selected_session))}"
        if selected_session is not None
        else "Actieve sessie: alle Entire Session - Live-sessies van deze dag"
    )
    st.caption(filter_caption)

    df_agg = _agg_by_player(df_day)
    if df_agg.empty:
        st.warning("Geen data om te aggregeren.")
        return

    st.markdown("### GPS & belasting per speler")
    _plot_player_gps_load(df_agg)
    st.markdown("### HR-zones & HR Exertion")
    _plot_hr_zones(df_agg)
    load_cols = st.columns(2, gap="large")
    with load_cols[0]:
        _plot_single_metric(df_agg, COL_DSL, "Dynamic Stress Load", MVV_RED, 1)
    with load_cols[1]:
        _plot_single_metric(df_agg, COL_FATIGUE, "Fatigue Index", "#EDB45B", 2)
