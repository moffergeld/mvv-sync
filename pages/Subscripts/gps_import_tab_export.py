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
    # If duration looks like seconds instead of minutes, convert to minutes.
    if median_value > 200:
        return s / 60.0
    return s


def _prepare_export_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    if df is None or df.empty:
        return pd.DataFrame(), {}

    mapping: dict[str, str] = {}
    for key, aliases in METRIC_ALIASES.items():
        col = _find_col(df, aliases)
        if col:
            mapping[key] = col

    work = df.copy()

    # Basic required identifiers
    required = ["player_name", "date", "event"]
    missing_required = [k for k in required if k not in mapping]
    if missing_required:
        raise ValueError(
            "Scouting export mist verplichte kolommen: " + ", ".join(missing_required)
        )

    # Standardize identifiers
    work["_player_name"] = work[mapping["player_name"]].astype(str).str.strip()
    work["_date"] = pd.to_datetime(work[mapping["date"]], errors="coerce").dt.date
    work["_event"] = work[mapping["event"]].astype(str).str.strip()

    # Metrics
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

    # Week
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

    # Match key for combining first and second half into one match
    match_parts = [work["_date"].astype(str)]
    used_extra = False
    for candidate in MATCH_KEY_CANDIDATES:
        if candidate in work.columns:
            match_parts.append(work[candidate].astype(str).str.strip())
            used_extra = True
            break
    if not used_extra:
        # Fallback: only date, which is acceptable if there is max one match per player/day.
        match_parts.append(work["_player_name"])
    work["_match_key"] = pd.Series(match_parts[0], index=work.index)
    for part in match_parts[1:]:
        work["_match_key"] = work["_match_key"] + " | " + part

    # Session type inference used for weekly counts
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


def _build_top3_match_export(prepared_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    for metric in ["total_distance", "sprint_distance", "high_sprint_distance", "playerload2d"]:
        match_df[f"{metric}_90"] = (match_df[metric] / match_df["duration_minutes"]) * 90.0

    # Top 3 per metric per player
    player_summaries = []
    metric_cols_90 = [
        "total_distance_90",
        "sprint_distance_90",
        "high_sprint_distance_90",
        "playerload2d_90",
    ]

    for player, g in match_df.groupby("player_name", dropna=False):
        row: dict[str, object] = {"player_name": player, "qualified_matches_30_plus": len(g)}

        valid_speed = g.loc[g["max_speed"].notna() & (g["max_speed"] <= 37), "max_speed"]
        row["max_speed_threshold_37"] = valid_speed.max() if not valid_speed.empty else pd.NA

        for metric in metric_cols_90:
            top3 = g.nlargest(3, metric)[metric].dropna()
            row[f"avg_top3_{metric}"] = top3.mean() if not top3.empty else pd.NA
            row[f"n_used_{metric}"] = int(top3.shape[0])

        player_summaries.append(row)

    summary_df = pd.DataFrame(player_summaries).sort_values("player_name").reset_index(drop=True)

    # Helpful detailed sheet with source matches
    detail_frames = []
    for metric in metric_cols_90:
        tmp = (
            match_df.sort_values(["player_name", metric], ascending=[True, False])
            .groupby("player_name", as_index=False, group_keys=False)
            .head(3)
            .copy()
        )
        tmp.insert(1, "ranking_metric", metric)
        detail_frames.append(tmp)
    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()

    return summary_df, detail_df


def _build_weekly_export(prepared_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = prepared_df[_mask_summary_events(prepared_df["_event"])].copy()
    summary_df = summary_df.dropna(subset=["_player_name", "_date"])
    if summary_df.empty:
        return pd.DataFrame()

    metrics = ["_total_distance", "_sprint_distance", "_high_sprint_distance", "_playerload2d"]
    group_cols = ["_player_name", "_week"]

    agg_dict: dict[str, list[str]] = {
        metric: ["sum", "std", "min", "max"] for metric in metrics
    }
    weekly = summary_df.groupby(group_cols, dropna=False).agg(agg_dict)
    weekly.columns = [f"{col[0].replace('_', '', 1)}_{col[1]}" for col in weekly.columns]
    weekly = weekly.reset_index().rename(columns={
        "_player_name": "player_name",
        "_week": "week",
    })

    # Counts per week from summary events
    counts = (
        summary_df.groupby(group_cols, dropna=False)
        .agg(
            sessions_in_week=("_date", "size"),
            trainings_in_week=("_session_kind", lambda s: int((s == "training").sum())),
            matches_in_week=("_session_kind", lambda s: int((s == "match").sum())),
        )
        .reset_index()
        .rename(columns={"_player_name": "player_name", "_week": "week"})
    )

    weekly = weekly.merge(counts, on=["player_name", "week"], how="left")

    # Order columns more cleanly
    ordered_cols = ["player_name", "week", "sessions_in_week", "trainings_in_week", "matches_in_week"]
    metric_bases = ["totaldistance", "sprintdistance", "highsprintdistance", "playerload2d"]
    for base in metric_bases:
        ordered_cols.extend([
            f"{base}_sum",
            f"{base}_std",
            f"{base}_min",
            f"{base}_max",
        ])
    ordered_cols = [c for c in ordered_cols if c in weekly.columns]
    weekly = weekly[ordered_cols].sort_values(["player_name", "week"]).reset_index(drop=True)
    return weekly


def _build_info_sheet(prepared_df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    lines = [
        ["Scouting export", "Top-3 matchgemiddelden + weektotalen"],
        ["Top-3 bron", "Alleen events First Half / Second Half, niet Summary"],
        ["Top-3 drempel", "Alleen wedstrijden met totale duur >= 30 minuten"],
        ["Normalisatie", "Total Distance, Sprint Distance, High Sprint Distance en PlayerLoad2D genormaliseerd naar 90 minuten"],
        ["Max speed", "Hoogste geldige waarde <= 37; waarden boven 37 worden genegeerd"],
        ["Weekexport bron", "Alle Summary events"],
        ["Weekstatistieken", "som, standaarddeviatie, minimum en maximum per speler per week"],
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
            top3_df, top3_detail_df = _build_top3_match_export(prepared_df)
            weekly_df = _build_weekly_export(prepared_df)
            info_df = _build_info_sheet(prepared_df, mapping)

            if top3_df.empty and weekly_df.empty:
                toast_err("Geen bruikbare data gevonden voor scouting export.")
                st.stop()

            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                if not top3_df.empty:
                    top3_df.to_excel(writer, index=False, sheet_name="Top3_90_Avg")
                if not top3_detail_df.empty:
                    top3_detail_df.to_excel(writer, index=False, sheet_name="Top3_Source_Matches")
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
