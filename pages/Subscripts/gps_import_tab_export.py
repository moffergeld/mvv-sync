# gps_import_tab_export.py
# ============================================================
# Subtab: Export
# ============================================================

from __future__ import annotations

import io
from datetime import date

import pandas as pd
import requests
import streamlit as st

from pages.Subscripts.gps_import_common import (
    GPS_COLS,
    df_to_excel_bytes_single,
    fetch_all_gps_records,
    rest_get,
    safe_sheet_name,
    toast_err,
    toast_ok,
)


# ------------------------------------------------------------
# Helpers for scouting export
# ------------------------------------------------------------
METRIC_ALIASES = {
    "total_distance": [
        "total_distance", "total distance", "distance_total", "distance covered",
        "totaldistance", "distance",
    ],
    "sprint_distance": [
        "sprint_distance", "sprint distance", "sprintdistance", "sprint",
    ],
    "high_sprint_distance": [
        "high_sprint_distance", "high sprint distance", "highsprintdistance",
        "high sprint", "high_speed_running_distance", "high speed running distance",
        "high_sprint",
    ],
    "playerload2d": [
        "playerload2d", "player_load2d", "player load2d", "playerload 2d",
    ],
    "max_speed": [
        "max_speed", "max speed", "maximum_speed", "maximum speed",
        "top_speed", "top speed", "topspeed",
    ],
    "duration": [
        "duration", "duration_min", "minutes", "mins", "minuten",
        "playing_time", "play_time", "total_duration", "effective_duration",
        "session_duration", "duration_minutes",
    ],
    "player_name": [
        "player_name", "player", "athlete_name", "athlete", "name", "speler",
    ],
    "date": [
        "datum", "date", "session_date", "event_date", "match_date",
    ],
    "week": [
        "week", "weeknr", "week_nr", "calendar_week", "iso_week",
    ],
    "event": [
        "event", "event_name", "period", "split_name", "segment", "segment_name",
    ],
}

MATCH_KEY_CANDIDATES = [
    "session_name", "activity_name", "session", "title", "match_name", "game_name",
    "opponent", "against", "team_name", "team", "squad", "label", "description",
]

SESSION_TYPE_CANDIDATES = [
    "session_type", "activity_type", "type", "category", "session_name",
    "activity_name", "title", "description", "opponent",
]


TOP3_METRIC_KEYS = [
    "total_distance",
    "sprint_distance",
    "high_sprint_distance",
    "playerload2d",
]

WEEKLY_METRIC_KEYS = [
    "duration",
    "total_distance",
    "sprint_distance",
    "high_sprint_distance",
    "playerload2d",
]


def _norm(s: str) -> str:
    return str(s).strip().lower().replace("_", " ")


def _find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    normalized = {_norm(c): c for c in df.columns}
    for alias in aliases:
        if _norm(alias) in normalized:
            return normalized[_norm(alias)]
    for alias in aliases:
        a = _norm(alias)
        for nc, original in normalized.items():
            if a in nc:
                return original
    return None


def _display_name(mapping: dict[str, str], metric_key: str, fallback: str | None = None) -> str:
    return mapping.get(metric_key, fallback or metric_key)


def _ensure_numeric(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9.\-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _duration_to_minutes(series: pd.Series) -> pd.Series:
    s = _ensure_numeric(series)
    if s.dropna().empty:
        return s
    median_value = float(s.dropna().median())
    if median_value > 200:
        return s / 60.0
    return s


def _valid_min_from_threshold(series: pd.Series, pct_of_max: float = 0.10):
    s = _ensure_numeric(series).dropna()
    if s.empty:
        return pd.NA
    max_value = s.max()
    if pd.isna(max_value):
        return pd.NA
    if max_value <= 0:
        return s.min()
    threshold = max_value * pct_of_max
    valid = s[s >= threshold]
    if valid.empty:
        return s.max()
    return valid.min()


def _prepare_export_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    if df is None or df.empty:
        return pd.DataFrame(), {}

    mapping: dict[str, str] = {}
    for key, aliases in METRIC_ALIASES.items():
        col = _find_col(df, aliases)
        if col:
            mapping[key] = col

    work = df.copy()

    required = ["player_name", "date", "event"]
    missing_required = [k for k in required if k not in mapping]
    if missing_required:
        raise ValueError(
            "Scouting export mist verplichte kolommen: " + ", ".join(missing_required)
        )

    work["_player_name"] = work[mapping["player_name"]].astype(str).str.strip()
    work["_date"] = pd.to_datetime(work[mapping["date"]], errors="coerce").dt.date
    work["_event"] = work[mapping["event"]].astype(str).str.strip()

    for metric in [
        "total_distance", "sprint_distance", "high_sprint_distance",
        "playerload2d", "max_speed",
    ]:
        if metric in mapping:
            work[f"_{metric}"] = _ensure_numeric(work[mapping[metric]])
        else:
            work[f"_{metric}"] = pd.NA

    if "duration" in mapping:
        work["_duration_minutes"] = _duration_to_minutes(work[mapping["duration"]])
    else:
        work["_duration_minutes"] = pd.NA

    if "week" in mapping:
        week_raw = work[mapping["week"]]
        work["_week"] = week_raw.astype(str).str.strip()
        empty_week = work["_week"].isin(["", "nan", "None"])
        iso = pd.to_datetime(work["_date"], errors="coerce").dt.isocalendar()
        work.loc[empty_week, "_week"] = (
            iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
        )[empty_week]
    else:
        iso = pd.to_datetime(work["_date"], errors="coerce").dt.isocalendar()
        work["_week"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)

    match_parts = [work["_date"].astype(str)]
    used_extra = False
    for candidate in MATCH_KEY_CANDIDATES:
        if candidate in work.columns:
            match_parts.append(work[candidate].astype(str).str.strip())
            used_extra = True
            break
    if not used_extra:
        match_parts.append(work["_player_name"])
    work["_match_key"] = pd.Series(match_parts[0], index=work.index)
    for part in match_parts[1:]:
        work["_match_key"] = work["_match_key"] + " | " + part

    text_cols = [c for c in SESSION_TYPE_CANDIDATES if c in work.columns]
    if text_cols:
        session_text = work[text_cols[0]].astype(str).str.lower().fillna("")
        for extra_col in text_cols[1:]:
            session_text = session_text + " | " + work[extra_col].astype(str).str.lower().fillna("")
    else:
        session_text = work["_event"].astype(str).str.lower().fillna("")

    work["_session_kind"] = "other"
    match_mask = session_text.str.contains(
        r"\b(match|wedstrijd|game|vs\b|v\.?\s|opponent|league|cup)\b",
        regex=True,
        na=False,
    )
    training_mask = session_text.str.contains(
        r"\b(train|training|session|practice)\b",
        regex=True,
        na=False,
    )
    work.loc[training_mask, "_session_kind"] = "training"
    work.loc[match_mask, "_session_kind"] = "match"

    return work, mapping


def _mask_summary_events(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower().str.strip()
    return s.str.contains("summary", na=False)


def _mask_half_events(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower().str.strip()
    first_half = s.str.contains(r"first\s*half|1st\s*half|eerste\s*helft", regex=True, na=False)
    second_half = s.str.contains(r"second\s*half|2nd\s*half|tweede\s*helft", regex=True, na=False)
    return first_half | second_half


def _fetch_selected_players_df(
    access_token: str,
    players: list[str],
    dt_from: date,
    dt_to: date,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in players:
        pname = requests.utils.quote(str(p), safe="")
        q = (
            f"select={','.join(GPS_COLS)}"
            f"&player_name=eq.{pname}"
            f"&datum=gte.{dt_from.isoformat()}"
            f"&datum=lte.{dt_to.isoformat()}"
            f"&order=datum.asc"
            f"&limit=200000"
        )
        dfp = rest_get(access_token, "gps_records", q)
        if not dfp.empty:
            frames.append(dfp)
    if not frames:
        return pd.DataFrame(columns=GPS_COLS)
    return pd.concat(frames, ignore_index=True)


def _build_top3_match_export(
    prepared_df: pd.DataFrame,
    mapping: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    half_df = prepared_df[_mask_half_events(prepared_df["_event"])].copy()
    half_df = half_df.dropna(subset=["_player_name", "_date"])

    if half_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    match_group_cols = ["_player_name", "_date", "_match_key"]
    agg = {
        "_duration_minutes": "sum",
        "_total_distance": "sum",
        "_sprint_distance": "sum",
        "_high_sprint_distance": "sum",
        "_playerload2d": "sum",
        "_max_speed": "max",
    }
    match_df = half_df.groupby(match_group_cols, dropna=False, as_index=False).agg(agg)
    match_df = match_df.rename(columns={
        "_player_name": "player_name",
        "_date": "match_date",
        "_match_key": "match_key",
        "_duration_minutes": "duration_minutes",
        "_total_distance": "total_distance",
        "_sprint_distance": "sprint_distance",
        "_high_sprint_distance": "high_sprint_distance",
        "_playerload2d": "playerload2d",
        "_max_speed": "max_speed",
    })

    match_df = match_df[match_df["duration_minutes"].fillna(0) >= 30].copy()
    if match_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    metric_90_map = {}
    for metric_key in TOP3_METRIC_KEYS:
        metric_90 = f"{metric_key}_90"
        match_df[metric_90] = (match_df[metric_key] / match_df["duration_minutes"]) * 90.0
        metric_90_map[metric_key] = metric_90

    player_summaries = []
    for player, g in match_df.groupby("player_name", dropna=False):
        row: dict[str, object] = {
            "player_name": player,
            "Qualified Matches": len(g),
        }

        valid_speed = g.loc[g["max_speed"].notna() & (g["max_speed"] <= 37), "max_speed"]
        row[_display_name(mapping, "max_speed")] = valid_speed.max() if not valid_speed.empty else pd.NA

        for metric_key in TOP3_METRIC_KEYS:
            metric_90 = metric_90_map[metric_key]
            top3 = g.nlargest(3, metric_90)[metric_90].dropna()
            row[_display_name(mapping, metric_key)] = top3.mean() if not top3.empty else pd.NA

        player_summaries.append(row)

    summary_df = pd.DataFrame(player_summaries).sort_values("player_name").reset_index(drop=True)
    summary_order = [
        "player_name",
        "Qualified Matches",
        _display_name(mapping, "total_distance"),
        _display_name(mapping, "sprint_distance"),
        _display_name(mapping, "high_sprint_distance"),
        _display_name(mapping, "playerload2d"),
        _display_name(mapping, "max_speed"),
    ]
    summary_order = [c for c in summary_order if c in summary_df.columns]
    summary_df = summary_df[summary_order]

    detail_frames = []
    for metric_key in TOP3_METRIC_KEYS:
        metric_90 = metric_90_map[metric_key]
        tmp = (
            match_df.sort_values(["player_name", metric_90], ascending=[True, False])
            .groupby("player_name", as_index=False, group_keys=False)
            .head(3)
            .copy()
        )
        tmp.insert(1, "ranking_metric", _display_name(mapping, metric_key))
        detail_frames.append(tmp)

    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()

    return summary_df, detail_df


def _build_weekly_export(prepared_df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    summary_df = prepared_df[_mask_summary_events(prepared_df["_event"])].copy()
    summary_df = summary_df.dropna(subset=["_player_name", "_date"])
    if summary_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (player, week), g in summary_df.groupby(["_player_name", "_week"], dropna=False):
        row: dict[str, object] = {
            "player_name": player,
            "week": week,
            "sessions_in_week": int(g["_date"].size),
            "trainings_in_week": int((g["_session_kind"] == "training").sum()),
            "matches_in_week": int((g["_session_kind"] == "match").sum()),
        }

        metric_series_map = {
            "duration": g["_duration_minutes"],
            "total_distance": g["_total_distance"],
            "sprint_distance": g["_sprint_distance"],
            "high_sprint_distance": g["_high_sprint_distance"],
            "playerload2d": g["_playerload2d"],
        }

        for metric_key in WEEKLY_METRIC_KEYS:
            metric_name = _display_name(mapping, metric_key)
            s = _ensure_numeric(metric_series_map[metric_key]).dropna()

            row[f"{metric_name}_total"] = s.sum() if not s.empty else pd.NA
            row[f"{metric_name}_sd"] = s.std(ddof=1) if len(s) > 1 else pd.NA
            row[f"{metric_name}_min"] = _valid_min_from_threshold(s, pct_of_max=0.10) if not s.empty else pd.NA
            row[f"{metric_name}_max"] = s.max() if not s.empty else pd.NA

        rows.append(row)

    weekly = pd.DataFrame(rows).sort_values(["player_name", "week"]).reset_index(drop=True)

    ordered_cols = [
        "player_name",
        "week",
        "sessions_in_week",
        "trainings_in_week",
        "matches_in_week",
    ]
    for metric_key in WEEKLY_METRIC_KEYS:
        metric_name = _display_name(mapping, metric_key)
        ordered_cols.extend([
            f"{metric_name}_total",
            f"{metric_name}_sd",
            f"{metric_name}_min",
            f"{metric_name}_max",
        ])

    ordered_cols = [c for c in ordered_cols if c in weekly.columns]
    return weekly[ordered_cols]


def _build_info_sheet(prepared_df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    top3_metric_names = ", ".join(_display_name(mapping, k) for k in TOP3_METRIC_KEYS)
    weekly_metric_names = ", ".join(_display_name(mapping, k) for k in WEEKLY_METRIC_KEYS)
    lines = [
        ["Scouting export", "Top-3 matchgemiddelden + weektotalen"],
        ["Top-3 bron", "Alleen events First Half / Second Half, niet Summary"],
        ["Top-3 drempel", "Alleen wedstrijden met totale duur >= 30 minuten"],
        ["Top-3 output", f"Kolommen tonen parameternamen: {top3_metric_names} + max_speed"],
        ["Normalisatie Top-3", "Top-3 waarden voor load-metrics zijn genormaliseerd naar 90 minuten"],
        ["Max speed", "Hoogste geldige waarde <= 37; waarden boven 37 worden genegeerd"],
        ["Weekexport bron", "Alle Summary events"],
        ["Weekexport output", f"Kolommen voor: {weekly_metric_names}"],
        ["Week minimum", "Minimum wordt berekend op waarden die minimaal 10% van de week-max zijn"],
        ["Weekstatistieken", "total, sd, min en max per speler per week"],
        ["Trainings/wedstrijden", "Gebaseerd op tekstherkenning in type/session-kolommen; onbekende sessies vallen buiten beide tellingen"],
        ["Gebruikte kolommen", "; ".join(f"{k} -> {v}" for k, v in sorted(mapping.items()))],
        ["Beschikbare kolommen in export", "; ".join(map(str, prepared_df.columns.tolist()))],
    ]
    return pd.DataFrame(lines, columns=["item", "value"])


# ------------------------------------------------------------
# Main tab
# ------------------------------------------------------------
def tab_export_main(access_token: str, player_options: list[str]) -> None:
    st.subheader("Export gps_records → Excel")

    st.markdown("### 1) Export alles")
    c1, c2 = st.columns([1.4, 2.6])
    with c1:
        limit = st.number_input(
            "Max rows (veiligheidslimiet)",
            min_value=1,
            max_value=500000,
            value=200000,
            step=10000,
            key="exp_all_limit",
        )
    with c2:
        st.caption("Klik op generate, daarna verschijnt de downloadknop.")

    if st.button("Generate export (ALL)", key="exp_all_btn"):
        try:
            df = fetch_all_gps_records(access_token, limit=int(limit))
            if df.empty:
                st.warning("Geen data gevonden.")
            else:
                ordered = [c for c in GPS_COLS if c in df.columns]
                df = df[ordered]
                xbytes = df_to_excel_bytes_single(df, sheet_name="gps_records")
                st.session_state["export_all_bytes"] = xbytes
                toast_ok(f"Export bevestigd: {len(df)} rijen klaar voor download.")
        except Exception as e:
            toast_err(str(e))

    if st.session_state.get("export_all_bytes"):
        st.download_button(
            "Download gps_records_ALL.xlsx",
            data=st.session_state["export_all_bytes"],
            file_name="gps_records_ALL.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="exp_all_dl",
        )

    st.divider()
    st.markdown("### 2) Export geselecteerde speler(s) (elk tabblad = speler)")

    export_players = st.multiselect(
        "Selecteer speler(s)",
        options=player_options if player_options else [],
        key="exp_sel_players",
    )

    date_col1, date_col2 = st.columns([1, 1])
    with date_col1:
        exp_from = st.date_input("Van datum", value=date.today().replace(day=1), key="exp_sel_from")
    with date_col2:
        exp_to = st.date_input("Tot datum", value=date.today(), key="exp_sel_to")

    if st.button("Generate export (selected players)", key="exp_sel_btn"):
        try:
            if not export_players:
                toast_err("Selecteer minimaal 1 speler.")
                st.stop()

            bio = io.BytesIO()
            used = set()
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                for p in export_players:
                    pname = requests.utils.quote(str(p), safe="")
                    q = (
                        f"select={','.join(GPS_COLS)}"
                        f"&player_name=eq.{pname}"
                        f"&datum=gte.{exp_from.isoformat()}"
                        f"&datum=lte.{exp_to.isoformat()}"
                        f"&order=datum.asc"
                        f"&limit=200000"
                    )
                    dfp = rest_get(access_token, "gps_records", q)
                    ordered = [c for c in GPS_COLS if c in dfp.columns]
                    dfp = dfp[ordered] if not dfp.empty else pd.DataFrame(columns=GPS_COLS)
                    sheet = safe_sheet_name(p, used)
                    dfp.to_excel(writer, index=False, sheet_name=sheet)

            st.session_state["export_sel_bytes"] = bio.getvalue()
            toast_ok("Export bevestigd: bestand klaar voor download.")
        except Exception as e:
            toast_err(str(e))

    if st.session_state.get("export_sel_bytes"):
        st.download_button(
            "Download gps_records_SELECTED_PLAYERS.xlsx",
            data=st.session_state["export_sel_bytes"],
            file_name="gps_records_SELECTED_PLAYERS.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="exp_sel_dl",
        )

    st.divider()
    st.markdown("### 3) Scouting export")
    st.caption(
        "Per speler: gemiddelde van top-3 wedstrijden op basis van 90-minutenwaarden + weektotalen op basis van Summary events."
    )

    scout_players = st.multiselect(
        "Selecteer speler(s) voor scouting export",
        options=player_options if player_options else [],
        key="exp_scout_players",
    )

    scout_col1, scout_col2 = st.columns([1, 1])
    with scout_col1:
        scout_from = st.date_input(
            "Scouting export van datum",
            value=date.today().replace(month=1, day=1),
            key="exp_scout_from",
        )
    with scout_col2:
        scout_to = st.date_input(
            "Scouting export tot datum",
            value=date.today(),
            key="exp_scout_to",
        )

    if st.button("Generate scouting export", key="exp_scout_btn"):
        try:
            if not scout_players:
                toast_err("Selecteer minimaal 1 speler voor de scouting export.")
                st.stop()

            raw_df = _fetch_selected_players_df(access_token, scout_players, scout_from, scout_to)
            if raw_df.empty:
                toast_err("Geen gps_records gevonden voor de gekozen spelers/periode.")
                st.stop()

            prepared_df, mapping = _prepare_export_df(raw_df)
            top3_df, top3_detail_df = _build_top3_match_export(prepared_df, mapping)
            weekly_df = _build_weekly_export(prepared_df, mapping)
            info_df = _build_info_sheet(prepared_df, mapping)

            if top3_df.empty and weekly_df.empty:
                toast_err("Geen bruikbare data gevonden voor scouting export.")
                st.stop()

            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                if not top3_df.empty:
                    top3_df.to_excel(writer, index=False, sheet_name="Top3_90_Avg")
                if not top3_detail_df.empty:
                    top3_detail_df.to_excel(writer, index=False, sheet_name="Sources")
                if not weekly_df.empty:
                    weekly_df.to_excel(writer, index=False, sheet_name="WeekTotals")
                info_df.to_excel(writer, index=False, sheet_name="Info")

            st.session_state["export_scout_bytes"] = bio.getvalue()
            toast_ok("Scouting export bevestigd: bestand klaar voor download.")
        except Exception as e:
            toast_err(str(e))

    if st.session_state.get("export_scout_bytes"):
        st.download_button(
            "Download gps_records_SCOUTING.xlsx",
            data=st.session_state["export_scout_bytes"],
            file_name="gps_records_SCOUTING.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="exp_scout_dl",
        )
