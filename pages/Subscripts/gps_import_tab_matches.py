# Main dashboard flow for importing and maintaining matches.

from __future__ import annotations

import hashlib
import math
from datetime import date

import pandas as pd
import streamlit as st

from pages.Subscripts.gps_import_common import (
    HOME_AWAY_OPTIONS,
    MATCH_TYPE_OPTIONS,
    TEAM_NAME_MATCHES,
    build_fixture,
    build_result,
    default_season_today,
    fetch_matches_range,
    normalize_matches_dataframe,
    parse_matches_csv,
    rest_delete,
    season_options,
    sync_matches,
    toast_err,
    toast_ok,
)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return int(number)
    except (TypeError, ValueError):
        return default


def _clear_match_caches() -> None:
    st.cache_data.clear()


def _match_label(row: pd.Series) -> str:
    match_id = row.get("match_id")
    id_label = f"#{int(match_id)} | " if pd.notna(match_id) else ""
    fixture = str(row.get("fixture") or "").strip()
    result = build_result(row.get("goals_for"), row.get("goals_against"))
    return f"{id_label}{row.get('match_date')} | {fixture} | {result or '-'}"


def _preview_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "_import_status",
        "_import_message",
        "_source_row",
        "match_date",
        "fixture",
        "home_away",
        "opponent",
        "match_type",
        "season",
        "result",
        "goals_for",
        "goals_against",
        "match_id",
    ]
    return [column for column in preferred if column in df.columns]


def _render_csv_import(access_token: str) -> None:
    st.markdown("### CSV importeren en controleren")
    st.caption(
        "Upload een Matches.csv. De import herkent Nederlandse en Engelse kolomnamen, "
        "meerdere scheidingstekens en veelgebruikte datumnotaties."
    )

    template = pd.DataFrame(
        [
            {
                "Datum": "2026-08-10",
                "Tegenstander": "FC Eindhoven",
                "Thuis/Uit": "Away",
                "Wedstrijd": "FC Eindhoven - MVV Maastricht",
                "Wedstrijdtype": "Competitie",
                "Seizoen": "2026/2027",
                "Uitslag": "2-1",
                "Doelpunten voor": 1,
                "Doelpunten tegen": 2,
            }
        ]
    )
    st.download_button(
        "Download CSV-template",
        data=template.to_csv(index=False, sep=";", encoding="utf-8-sig"),
        file_name="Matches_template.csv",
        mime="text/csv",
        key="matches_template_download",
    )

    uploaded = st.file_uploader(
        "Upload Matches.csv",
        type=["csv"],
        key="matches_csv_uploader",
    )
    if uploaded is None:
        st.info("Upload een CSV om eerst een controle-preview te maken.")
        return

    file_bytes = uploaded.getvalue()
    signature = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state.get("matches_preview_signature") != signature:
        st.session_state["matches_preview_signature"] = signature
        st.session_state["matches_preview"] = None

    if st.button("Preview en controleren", key="matches_preview_button"):
        try:
            preview = parse_matches_csv(file_bytes)
            st.session_state["matches_preview"] = preview
            toast_ok(f"Preview gemaakt: {len(preview)} regels gecontroleerd.")
        except Exception as exc:
            st.session_state["matches_preview"] = None
            toast_err(f"CSV kan niet worden gelezen: {exc}")

    preview = st.session_state.get("matches_preview")
    if not isinstance(preview, pd.DataFrame) or preview.empty:
        return

    status = preview["_import_status"].astype(str)
    counts = {
        "Totaal": len(preview),
        "OK": int(status.eq("OK").sum()),
        "Waarschuwing": int(status.eq("WAARSCHUWING").sum()),
        "Niet importeren": int(status.isin(["FOUT", "DUPLICAAT"]).sum()),
    }
    metric_cols = st.columns(4)
    for column, (label, value) in zip(metric_cols, counts.items()):
        column.metric(label, value)

    st.dataframe(
        preview[_preview_columns(preview)],
        width="stretch",
        hide_index=True,
    )

    include_warnings = st.checkbox(
        "Importeer ook regels met waarschuwingen",
        value=True,
        key="matches_include_warnings",
        help="Automatisch bepaald seizoen of Home als standaardwaarde wordt als waarschuwing getoond.",
    )
    allowed = ["OK", "WAARSCHUWING"] if include_warnings else ["OK"]
    import_df = preview[status.isin(allowed)]
    st.caption(
        f"{len(import_df)} regels klaar voor synchronisatie. Fouten en dubbele regels worden nooit geïmporteerd."
    )
    if st.button(
        "Synchroniseer naar Supabase",
        type="primary",
        disabled=import_df.empty,
        key="matches_sync_button",
    ):
        try:
            result = sync_matches(access_token, import_df, source_file=uploaded.name)
            st.session_state["matches_preview"] = None
            _clear_match_caches()
            toast_ok(
                f"Synchronisatie voltooid: {result['inserted']} toegevoegd, "
                f"{result['updated']} bijgewerkt."
            )
            st.rerun()
        except Exception as exc:
            toast_err(f"Synchronisatie mislukt: {exc}")


def _render_existing_matches(access_token: str) -> None:
    st.markdown("### Bestaande wedstrijden beheren")
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        date_from = st.date_input(
            "Van",
            value=date.today().replace(month=1, day=1),
            key="matches_edit_from",
        )
    with c2:
        date_to = st.date_input("Tot", value=date.today(), key="matches_edit_to")
    with c3:
        season_filter = st.selectbox(
            "Seizoen",
            options=["(alles)"] + season_options(start_year=2020, years_ahead=6),
            key="matches_edit_season",
        )

    if date_from > date_to:
        st.warning("De startdatum moet vóór de einddatum liggen.")
        return

    try:
        matches = fetch_matches_range(
            access_token,
            date_from,
            date_to,
            "" if season_filter == "(alles)" else season_filter,
        )
    except Exception as exc:
        toast_err(f"Wedstrijden ophalen mislukt: {exc}")
        return

    if matches.empty:
        st.info("Geen wedstrijden gevonden in deze periode.")
        return

    matches = matches.copy()
    matches["label"] = matches.apply(_match_label, axis=1)
    selected_label = st.selectbox(
        "Kies wedstrijd",
        options=matches["label"].tolist(),
        key="matches_edit_selection",
    )
    selected = matches.loc[matches["label"] == selected_label].iloc[0]
    match_id = int(selected["match_id"])

    with st.form(f"edit_match_{match_id}"):
        e1, e2, e3 = st.columns([1, 1.4, 1])
        with e1:
            edit_date = st.date_input(
                "Datum",
                value=pd.to_datetime(selected["match_date"]).date(),
            )
        with e2:
            edit_opponent = st.text_input(
                "Tegenstander",
                value=str(selected.get("opponent") or ""),
            )
        with e3:
            home_away = st.selectbox(
                "Home/Away",
                options=HOME_AWAY_OPTIONS,
                index=(
                    HOME_AWAY_OPTIONS.index(selected.get("home_away"))
                    if selected.get("home_away") in HOME_AWAY_OPTIONS
                    else 0
                ),
            )

        e4, e5, e6 = st.columns([1, 1, 1.2])
        with e4:
            match_type = st.selectbox(
                "Wedstrijdtype",
                options=MATCH_TYPE_OPTIONS,
                index=(
                    MATCH_TYPE_OPTIONS.index(selected.get("match_type"))
                    if selected.get("match_type") in MATCH_TYPE_OPTIONS
                    else 0
                ),
            )
        with e5:
            seasons = season_options(start_year=2020, years_ahead=6)
            current_season = str(selected.get("season") or "").strip()
            if current_season and current_season not in seasons:
                seasons = [current_season] + seasons
            season = st.selectbox(
                "Seizoen",
                options=seasons,
                index=seasons.index(current_season) if current_season in seasons else 0,
            )
        with e6:
            edit_gf = st.number_input(
                "Goals voor",
                min_value=0,
                step=1,
                value=_safe_int(selected.get("goals_for")),
            )
            edit_ga = st.number_input(
                "Goals tegen",
                min_value=0,
                step=1,
                value=_safe_int(selected.get("goals_against")),
            )

        edit_fixture = build_fixture(TEAM_NAME_MATCHES, home_away, edit_opponent)
        st.caption(f"Fixture wordt opgeslagen als: `{edit_fixture}`")
        save = st.form_submit_button("Wedstrijd opslaan", type="primary")

    if save:
        raw = pd.DataFrame(
            [
                {
                    "match_id": match_id,
                    "match_date": edit_date,
                    "fixture": edit_fixture,
                    "home_away": home_away,
                    "opponent": edit_opponent,
                    "match_type": match_type,
                    "season": season,
                    "goals_for": edit_gf,
                    "goals_against": edit_ga,
                }
            ]
        )
        normalized = normalize_matches_dataframe(raw)
        if normalized.iloc[0]["_import_status"] == "FOUT":
            toast_err(str(normalized.iloc[0]["_import_message"]))
        else:
            try:
                result = sync_matches(access_token, normalized, source_file="dashboard-edit")
                _clear_match_caches()
                toast_ok(
                    f"Wedstrijd bijgewerkt: {result['updated']} record(s) gesynchroniseerd."
                )
                st.rerun()
            except Exception as exc:
                toast_err(f"Opslaan mislukt: {exc}")

    st.divider()
    if st.button("Wedstrijd verwijderen", key=f"delete_match_{match_id}"):
        st.session_state["confirm_match_delete"] = match_id
    if st.session_state.get("confirm_match_delete") == match_id:
        st.warning("Verwijderen is definitief en kan GPS-koppelingen beïnvloeden.")
        confirm, cancel = st.columns(2)
        with confirm:
            if st.button(
                "Ja, definitief verwijderen",
                type="primary",
                key=f"confirm_delete_{match_id}",
            ):
                try:
                    rest_delete(access_token, "matches", f"match_id=eq.{match_id}")
                    st.session_state["confirm_match_delete"] = None
                    _clear_match_caches()
                    toast_ok("Wedstrijd verwijderd.")
                    st.rerun()
                except Exception as exc:
                    toast_err(f"Verwijderen mislukt: {exc}")
        with cancel:
            if st.button("Annuleren", key=f"cancel_delete_{match_id}"):
                st.session_state["confirm_match_delete"] = None
                st.rerun()


def _render_manual_match(access_token: str) -> None:
    st.markdown("### Handmatig een wedstrijd toevoegen")
    with st.form("new_match_form"):
        c1, c2, c3 = st.columns([1, 1.4, 1])
        with c1:
            match_date = st.date_input("Datum", value=date.today())
        with c2:
            opponent = st.text_input("Tegenstander")
        with c3:
            home_away = st.selectbox("Home/Away", options=HOME_AWAY_OPTIONS)

        c4, c5, c6 = st.columns([1, 1, 1.2])
        with c4:
            match_type = st.selectbox("Wedstrijdtype", options=MATCH_TYPE_OPTIONS)
        with c5:
            seasons = season_options(start_year=2020, years_ahead=6)
            current_season = default_season_today()
            season = st.selectbox(
                "Seizoen",
                options=seasons,
                index=seasons.index(current_season) if current_season in seasons else 0,
            )
        with c6:
            goals_for = st.number_input("Goals voor", min_value=0, step=1, value=0)
            goals_against = st.number_input("Goals tegen", min_value=0, step=1, value=0)
        submit = st.form_submit_button("Wedstrijd toevoegen", type="primary")

    if not submit:
        return
    if not opponent.strip():
        toast_err("Tegenstander is verplicht.")
        return

    raw = pd.DataFrame(
        [
            {
                "match_date": match_date,
                "fixture": build_fixture(TEAM_NAME_MATCHES, home_away, opponent),
                "home_away": home_away,
                "opponent": opponent,
                "match_type": match_type,
                "season": season,
                "goals_for": goals_for,
                "goals_against": goals_against,
            }
        ]
    )
    normalized = normalize_matches_dataframe(raw)
    if normalized.iloc[0]["_import_status"] == "FOUT":
        toast_err(str(normalized.iloc[0]["_import_message"]))
        return
    try:
        result = sync_matches(access_token, normalized, source_file="dashboard-manual")
        _clear_match_caches()
        toast_ok(
            f"Wedstrijd gesynchroniseerd: {result['inserted']} toegevoegd, "
            f"{result['updated']} bijgewerkt."
        )
        st.rerun()
    except Exception as exc:
        toast_err(f"Toevoegen mislukt: {exc}")


def tab_matches_main(access_token: str) -> None:
    st.subheader("Matches")
    st.caption(
        "Beheer wedstrijden vanuit het dashboard. Eerst controleren, daarna synchroniseren naar Supabase. "
        "De unieke sleutel is datum + fixture + seizoen."
    )
    _render_csv_import(access_token)
    st.divider()
    _render_existing_matches(access_token)
    st.divider()
    _render_manual_match(access_token)
