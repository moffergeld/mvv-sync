# ============================================================
# Subtab: Import (player metrics CSV)
# ============================================================

from __future__ import annotations

import streamlit as st

from pages.Subscripts.gps_import_common import (
    CSV_DIRECT_COLS,
    CSV_SOURCE_COLUMN_MAP,
    ID_COLS_IN_PARSER,
    METRIC_MAP,
    TYPE_OPTIONS,
    apply_auto_match_ids_to_rows,
    df_to_db_rows,
    normalize_key,
    parse_player_metrics_csv,
    rest_upsert,
    toast_err,
    toast_ok,
)


def _extra_source_columns(columns: list[str]) -> list[str]:
    excluded = set(ID_COLS_IN_PARSER)
    extra = []
    for column in columns:
        key = normalize_key(column)
        if column in excluded or key in METRIC_MAP or key in CSV_SOURCE_COLUMN_MAP:
            continue
        extra.append(str(column))
    return extra


def tab_import_csv_main(access_token: str, name_to_id: dict) -> None:
    st.subheader("Import CSV → gps_records")
    st.caption(
        "Voor brede speler-/sessie-exports. Kernmetrics blijven gekoppeld aan de bestaande "
        "GPS-kolommen; alle overige bronvelden krijgen een csv_-kolom en onbekende toekomstige "
        "velden blijven veilig beschikbaar in extra_metrics."
    )

    selected_type = st.selectbox(
        "Importeer als",
        TYPE_OPTIONS,
        index=0,
        key="gps_csv_import_type",
        help="De leverancierstype-naam, zoals Match Day +3, wordt niet automatisch als dashboardtype gebruikt. Kies hier Practice, Match of Practice Match.",
    )
    uploaded = st.file_uploader(
        "Upload speler-metrics CSV",
        type=["csv"],
        key="gps_csv_import_file",
    )

    if not uploaded:
        st.info("Upload een CSV om eerst een controleerbare preview te maken.")
        return

    file_bytes = uploaded.getvalue()
    signature = f"{uploaded.name}:{len(file_bytes)}:{selected_type}"
    if st.session_state.get("gps_csv_preview_signature") != signature:
        st.session_state.pop("gps_csv_preview", None)
        st.session_state["gps_csv_preview_signature"] = signature

    if st.button("Preview CSV", type="secondary", key="gps_csv_preview_button"):
        try:
            parsed = parse_player_metrics_csv(file_bytes, selected_type=selected_type)
            st.session_state["gps_csv_preview"] = {
                "filename": uploaded.name,
                "df": parsed,
            }
            toast_ok(f"CSV gelezen: {len(parsed)} speler-sessies.")
        except Exception as exc:
            st.session_state.pop("gps_csv_preview", None)
            toast_err(f"CSV kon niet worden gelezen: {exc}")

    preview = st.session_state.get("gps_csv_preview")
    if not preview:
        return

    parsed = preview["df"]
    extra_columns = _extra_source_columns(list(parsed.columns))
    player_count = parsed["Speler"].nunique(dropna=True)
    event_count = parsed["Event"].nunique(dropna=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rijen", len(parsed))
    c2.metric("Spelers", player_count)
    c3.metric("Events", event_count)
    c4.metric("Directe CSV-velden", len(CSV_DIRECT_COLS))

    st.markdown(f"**{preview['filename']}** — type: **{selected_type}**")
    visible_columns = ["Speler", "Datum", "Type", "Event"]
    for column in parsed.columns:
        if column not in visible_columns and normalize_key(column) in METRIC_MAP:
            visible_columns.append(column)
    st.dataframe(parsed[visible_columns].head(80), width="stretch", hide_index=True)

    if extra_columns:
        with st.expander(f"Onbekende toekomstige velden naar extra_metrics ({len(extra_columns)})"):
            st.write(", ".join(extra_columns))

    st.warning(
        "Niet-gematchte spelers krijgen player_id = NULL. De rij wordt wel opgeslagen, "
        "zodat je de naam/mapping later kunt corrigeren."
    )
    if st.button("Importeer naar Supabase", type="primary", key="gps_csv_import_button"):
        try:
            rows, unmapped = df_to_db_rows(
                parsed,
                source_file=preview["filename"],
                name_to_id=name_to_id,
            )
            rows = apply_auto_match_ids_to_rows(
                access_token,
                rows,
                ui_key_prefix="gps_csv_apply",
            )
            rest_upsert(
                access_token,
                "gps_records",
                rows,
                on_conflict="player_name,datum,type,event",
            )

            if unmapped:
                st.warning("Niet-gematchte namen: " + ", ".join(unmapped[:30]))
            st.session_state.pop("gps_csv_preview", None)
            toast_ok(f"CSV-import voltooid: {len(rows)} rijen opgeslagen.")
        except Exception as exc:
            toast_err(f"CSV-import mislukt: {exc}")
