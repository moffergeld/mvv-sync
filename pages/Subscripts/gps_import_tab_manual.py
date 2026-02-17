# gps_import_tab_manual.py
# ============================================================
# Subtab: Manual add
# ============================================================

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from Subscripts.gps_import_common import (
    INT_DB_COLS,
    MATCH_TYPES,
    TYPE_OPTIONS,
    fetch_gps_match_ids_on_date,
    json_safe,
    normalize_name,
    resolve_match_id_for_date,
    rest_upsert,
    toast_err,
    toast_ok,
    ui_pick_match_if_needed,
)


def tab_manual_add_main(access_token: str, name_to_id: dict, player_options: list[str]) -> None:
    st.subheader("Manual add (tabel)")
    st.caption("match_id wordt automatisch gezet bij Match / Practice Match (op basis van datum + bestaande gps_records).")

    template_cols = [
        "player_name",
        "datum",
        "type",
        "event",
        "match_id",
        "duration",
        "total_distance",
        "walking",
        "jogging",
        "running",
        "sprint",
        "high_sprint",
        "number_of_sprints",
        "number_of_high_sprints",
        "number_of_repeated_sprints",
        "max_speed",
        "avg_speed",
        "playerload3d",
        "playerload2d",
        "total_accelerations",
        "high_accelerations",
        "total_decelerations",
        "high_decelerations",
        "hrzone1",
        "hrzone2",
        "hrzone3",
        "hrzone4",
        "hrzone5",
        "hrtrimp",
        "hrzoneanaerobic",
        "avg_hr",
        "max_hr",
        "source_file",
    ]

    BASIC_COLS = ["player_name", "datum", "type", "event", "match_id", "source_file"]
    METRIC_COLS = [c for c in template_cols if c not in BASIC_COLS]

    DEFAULT_METRICS = [
        "duration",
        "total_distance",
        "sprint",
        "high_sprint",
        "number_of_sprints",
        "max_speed",
        "avg_speed",
        "playerload2d",
        "total_accelerations",
        "total_decelerations",
    ]

    def _blank_row() -> dict:
        return {
            "player_name": player_options[0] if player_options else "",
            "datum": date.today(),
            "type": "Practice",
            "event": "Summary",
            "match_id": None,
            "duration": None,
            "total_distance": None,
            "walking": None,
            "jogging": None,
            "running": None,
            "sprint": None,
            "high_sprint": None,
            "number_of_sprints": None,
            "number_of_high_sprints": None,
            "number_of_repeated_sprints": None,
            "max_speed": None,
            "avg_speed": None,
            "playerload3d": None,
            "playerload2d": None,
            "total_accelerations": None,
            "high_accelerations": None,
            "total_decelerations": None,
            "high_decelerations": None,
            "hrzone1": None,
            "hrzone2": None,
            "hrzone3": None,
            "hrzone4": None,
            "hrzone5": None,
            "hrtrimp": None,
            "hrzoneanaerobic": None,
            "avg_hr": None,
            "max_hr": None,
            "source_file": "manual",
        }

    if "manual_df" not in st.session_state:
        st.session_state["manual_df"] = pd.DataFrame([_blank_row()], columns=template_cols)

    if "manual_selected_metrics" not in st.session_state:
        st.session_state["manual_selected_metrics"] = [m for m in DEFAULT_METRICS if m in METRIC_COLS]

    top_left, top_mid, top_right = st.columns([1.0, 1.2, 3.8], vertical_alignment="bottom")

    with top_left:
        add_clicked = st.button("＋", type="primary", key="manual_add_row_btn", help="Voeg een rij toe")
        if add_clicked:
            df0 = st.session_state["manual_df"].copy()
            df0 = pd.concat([pd.DataFrame([_blank_row()], columns=template_cols), df0], ignore_index=True)
            st.session_state["manual_df"] = df0
            st.rerun()

    with top_mid:
        show_metrics = st.toggle("Toon metrics", value=True, key="manual_show_metrics")

    with top_right:
        selected = st.multiselect(
            "Selecteer metrics (kolommen)",
            options=METRIC_COLS,
            default=st.session_state["manual_selected_metrics"],
            key="manual_metrics_picker",
            help="Kies welke metric-kolommen je wilt zien/bewerken in de tabel.",
        )
        st.session_state["manual_selected_metrics"] = selected

    visible_cols = BASIC_COLS + list(st.session_state["manual_selected_metrics"]) if show_metrics else BASIC_COLS
    df_for_editor = st.session_state["manual_df"][visible_cols].copy()

    colcfg = {
        "player_name": st.column_config.SelectboxColumn("player_name", options=player_options, required=True),
        "datum": st.column_config.DateColumn("datum", required=True),
        "type": st.column_config.SelectboxColumn("type", options=TYPE_OPTIONS, required=True),
        "event": st.column_config.TextColumn("event", required=True),
        "match_id": st.column_config.NumberColumn(
            "match_id",
            help="Wordt automatisch gevuld bij Match/Practice Match (op basis van datum).",
            step=1,
            disabled=True,
        ),
        "source_file": st.column_config.TextColumn("source_file"),
    }

    for m in METRIC_COLS:
        if m in df_for_editor.columns:
            colcfg[m] = st.column_config.NumberColumn(m, step=1) if m in INT_DB_COLS else st.column_config.NumberColumn(m)

    edited = st.data_editor(
        df_for_editor,
        use_container_width=True,
        height=520,
        num_rows="fixed",
        column_config=colcfg,
        key="manual_editor",
    )

    # merge edited columns back into full df
    full_df = st.session_state["manual_df"].copy()
    for c in edited.columns:
        full_df[c] = edited[c].values
    st.session_state["manual_df"] = full_df

    df_preview = full_df.copy()

    # auto match_id preview fill
    if not df_preview.empty:
        df_preview["type"] = df_preview["type"].astype(str).str.strip()
        dt_series = pd.to_datetime(df_preview["datum"], errors="coerce")
        df_preview["_datum_iso"] = dt_series.dt.date.astype(str)

        needed = sorted(
            {
                (d_iso, t)
                for d_iso, t in zip(df_preview["_datum_iso"], df_preview["type"])
                if t in MATCH_TYPES and d_iso != "NaT"
            }
        )

        chosen_map: dict[tuple[str, str], int | None] = {}
        if needed:
            with st.expander("Automatische match-koppeling (alleen als nodig)", expanded=False):
                for d_iso, t in needed:
                    d_obj = pd.to_datetime(d_iso).date()
                    mid = ui_pick_match_if_needed(access_token, d_obj, t, key_prefix="manual_match_pick")
                    chosen_map[(d_iso, t)] = mid

            cur = pd.to_numeric(df_preview.get("match_id"), errors="coerce")
            is_missing = cur.isna()

            for (d_iso, t), mid in chosen_map.items():
                mask = (df_preview["_datum_iso"] == d_iso) & (df_preview["type"] == t) & is_missing
                if mid is not None:
                    df_preview.loc[mask, "match_id"] = int(mid)

            st.session_state["manual_df"] = df_preview.drop(columns=["_datum_iso"], errors="ignore").copy()

    b1, b2 = st.columns([1, 1], vertical_alignment="bottom")
    with b1:
        if st.button("Reset table", key="manual_reset_btn"):
            st.session_state["manual_df"] = pd.DataFrame([_blank_row()], columns=template_cols)
            toast_ok("Reset bevestigd.")
            st.rerun()
    with b2:
        save_clicked = st.button("Save rows (upsert)", type="primary", key="manual_save_btn")

    if not save_clicked:
        return

    try:
        dfm = st.session_state["manual_df"].copy()

        dfm["player_name"] = dfm["player_name"].astype(str).str.strip()
        dfm["event"] = dfm["event"].astype(str).str.strip()
        dfm["type"] = dfm["type"].astype(str).str.strip()

        dfm = dfm.dropna(subset=["player_name", "datum", "type", "event"])
        dfm = dfm[(dfm["player_name"] != "") & (dfm["event"] != "")]
        if dfm.empty:
            toast_err("Geen geldige rijen om op te slaan.")
            st.stop()

        dt_series = pd.to_datetime(dfm["datum"], errors="coerce")
        if dt_series.isna().any():
            toast_err("Ongeldige datum in één of meer rijen.")
            st.stop()

        dfm["week"] = dt_series.dt.isocalendar().week.astype(int)
        dfm["year"] = dt_series.dt.year.astype(int)
        dfm["datum"] = dt_series.dt.date.astype(str)

        dfm["player_id"] = dfm["player_name"].map(lambda x: name_to_id.get(normalize_name(x)))

        dfm["match_id"] = pd.to_numeric(dfm.get("match_id"), errors="coerce")
        dfm["match_id"] = dfm["match_id"].map(lambda v: int(v) if pd.notna(v) else None)

        # enforce match_id consistency per (datum,type)
        for (d_iso, t), g in dfm.groupby(["datum", "type"], dropna=False):
            if t not in MATCH_TYPES:
                dfm.loc[g.index, "match_id"] = None
                continue

            d_obj = pd.to_datetime(d_iso).date()
            existing_ids = fetch_gps_match_ids_on_date(access_token, d_obj, t)
            if not existing_ids.empty:
                forced_id = int(existing_ids.value_counts().idxmax())
            else:
                forced_id, _ = resolve_match_id_for_date(access_token, d_obj, t)
                if forced_id is None:
                    picked = pd.to_numeric(dfm.loc[g.index, "match_id"], errors="coerce").dropna()
                    if not picked.empty:
                        forced_id = int(picked.iloc[0])
                    else:
                        toast_err(f"Geen match_id beschikbaar voor {d_iso} ({t}). Voeg match toe of kies match.")
                        st.stop()
            dfm.loc[g.index, "match_id"] = forced_id

        metric_keys = [c for c in template_cols if c not in ["player_name", "datum", "type", "event", "match_id", "source_file"]]
        for c in metric_keys:
            if c in dfm.columns:
                dfm[c] = pd.to_numeric(dfm[c], errors="coerce")

        rows = []
        for _, r in dfm.iterrows():
            row = {
                "player_id": json_safe(r.get("player_id")),
                "player_name": json_safe(r.get("player_name")),
                "datum": json_safe(r.get("datum")),
                "week": json_safe(int(r.get("week")) if pd.notna(r.get("week")) else None),
                "year": json_safe(int(r.get("year")) if pd.notna(r.get("year")) else None),
                "type": json_safe(r.get("type")),
                "event": json_safe(r.get("event")),
                "match_id": json_safe(int(r.get("match_id")) if pd.notna(r.get("match_id")) else None),
                "source_file": json_safe(r.get("source_file") or "manual"),
            }

            for k in metric_keys:
                v = r.get(k)
                if k in INT_DB_COLS:
                    vv = pd.to_numeric(v, errors="coerce")
                    row[k] = json_safe(int(vv) if pd.notna(vv) else None)
                else:
                    vv = pd.to_numeric(v, errors="coerce")
                    row[k] = json_safe(float(vv) if pd.notna(vv) else None)

            rows.append(row)

        rest_upsert(access_token, "gps_records", rows, on_conflict="player_name,datum,type,event")
        st.session_state["manual_df"] = dfm[template_cols].copy()
        toast_ok(f"Save bevestigd: rows = {len(rows)}")
    except Exception as e:
        toast_err(f"Save fout: {e}")
