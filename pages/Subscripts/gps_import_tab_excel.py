# gps_import_tab_excel.py
# ============================================================
# Subtab: Import (Excel)
# ============================================================

from __future__ import annotations

from datetime import date

import streamlit as st

from Subscripts.gps_import_common import (
    TYPE_OPTIONS,
    apply_auto_match_ids_to_rows,
    df_to_db_rows,
    is_flat_gps_excel,
    parse_exercises_excel,
    parse_flat_gps_excel,
    parse_summary_excel,
    rest_upsert,
    toast_err,
    toast_ok,
)



def tab_import_excel_main(access_token: str, name_to_id: dict) -> None:
    st.subheader("Import Excel → gps_records")
    st.caption(
        "Ondersteunt: JOHAN SUMMARY/EXERCISES én 'Flat GPS' Excel (zoals Map1.xlsx). "
        "match_id wordt automatisch gekoppeld per (datum,type) bij Match/Practice Match."
    )

    col1, col2, col3 = st.columns([1.2, 1.2, 2.6])
    with col1:
        selected_date = st.date_input("Datum (alleen nodig voor JOHAN files)", value=date.today(), key="gps_imp_date")
    with col2:
        selected_type = st.selectbox("Type (alleen nodig voor JOHAN files)", TYPE_OPTIONS, key="gps_imp_type")
    with col3:
        uploaded_files = st.file_uploader(
            "Upload Excel (je mag meerdere tegelijk selecteren)",
            type=["xlsx", "xls"],
            accept_multiple_files=True,
            key="gps_imp_files",
        )

    if not uploaded_files:
        return

    st.divider()
    st.subheader("Preview / Parse")

    if st.button("Preview (parse)", type="secondary", key="gps_preview_btn"):
        all_parsed = []
        errors = []

        for up in uploaded_files:
            filename = up.name
            file_bytes = up.getvalue()

            is_summary = "summary" in filename.lower()
            is_exercises = "exercises" in filename.lower()

            try:
                if is_flat_gps_excel(file_bytes):
                    df_parsed = parse_flat_gps_excel(file_bytes)
                    kind = "FLAT GPS (auto)"
                elif is_summary:
                    df_parsed = parse_summary_excel(file_bytes, selected_date, selected_type)
                    kind = "SUMMARY"
                elif is_exercises:
                    df_parsed = parse_exercises_excel(file_bytes, selected_date, selected_type)
                    kind = "EXERCISES"
                else:
                    try:
                        df_parsed = parse_summary_excel(file_bytes, selected_date, selected_type)
                        kind = "SUMMARY (auto)"
                    except Exception:
                        df_parsed = parse_exercises_excel(file_bytes, selected_date, selected_type)
                        kind = "EXERCISES (auto)"

                all_parsed.append((filename, kind, df_parsed))
            except Exception as e:
                errors.append((filename, str(e)))

        st.session_state["gps_parsed_multi"] = all_parsed

        if errors:
            st.error("Sommige bestanden konden niet geparsed worden:")
            for fn, msg in errors:
                st.write(f"- {fn}: {msg}")

        if all_parsed:
            toast_ok(f"Preview bevestigd: parsed bestanden = {len(all_parsed)}")
            for fn, kind, dfp in all_parsed:
                st.markdown(f"**{fn}** — {kind} — rijen: {len(dfp)}")
                st.dataframe(dfp.head(120), use_container_width=True)

    all_parsed = st.session_state.get("gps_parsed_multi", [])
    if not all_parsed:
        return

    st.divider()
    st.subheader("Import → Supabase (upsert)")

    if st.button("Import (upsert naar gps_records)", type="primary", key="gps_upsert_btn"):
        try:
            all_rows: list[dict] = []
            all_unmapped: set[str] = set()

            for (filename, kind, dfp) in all_parsed:
                rows, unmapped = df_to_db_rows(dfp, source_file=filename, name_to_id=name_to_id)
                all_rows.extend(rows)
                all_unmapped.update(unmapped)

            if all_rows:
                all_rows = apply_auto_match_ids_to_rows(access_token, all_rows, ui_key_prefix="gps_excel_apply")

            if all_unmapped:
                st.warning(
                    "Niet gematchte namen (player_id blijft NULL, import gaat wel door):\n- "
                    + "\n- ".join(sorted(list(all_unmapped))[:30])
                    + (f"\n... (+{len(all_unmapped)-30})" if len(all_unmapped) > 30 else "")
                )

            rest_upsert(access_token, "gps_records", all_rows, on_conflict="player_name,datum,type,event")

            st.session_state["gps_parsed_multi"] = []
            toast_ok(f"Import bevestigd: upserted rows = {len(all_rows)}")
        except Exception as e:
            toast_err(f"Import fout: {e}")
