from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from acwr_settings import (
    ACWR_MODE_LOG_6W,
    ACWR_MODE_STANDARD,
    get_acwr_mode,
    get_acwr_mode_meta,
    set_acwr_mode,
)
from pages.Subscripts.gps_import_common import (
    ALLOWED_IMPORT,
    SUPABASE_URL,
    get_access_token,
    get_players_map,
    get_profile_role,
    rest_headers,
)
from pages.Subscripts.gps_import_tab_excel import tab_import_excel_main
from pages.Subscripts.gps_import_tab_export import tab_export_main
from pages.Subscripts.gps_import_tab_manual import tab_manual_add_main
from pages.Subscripts.gps_import_tab_matches import tab_matches_main
from pages.Subscripts.mvv_branding import TEAM_HERO_BG, TEAM_LOGO, build_data_uri
from pages.Subscripts.wr_common import fetch_active_players_cached
from roles import get_profile, get_sb, is_staff_user, render_sidebar_footer, render_sidebar_navigation
from utils.streamlit_ui import apply_streamlit_chrome


st.set_page_config(page_title="Management | MVV Dashboard", layout="wide")
apply_streamlit_chrome()

PAGE_BG_URI = build_data_uri(TEAM_HERO_BG)
TEAM_LOGO_URI = build_data_uri(TEAM_LOGO)
MATCH_FILTER_REGULAR = "Normale wedstrijd"
MATCH_FILTER_FRIENDLY = "Oefenwedstrijd"


def rest_get_paged(
    access_token: str,
    table: str,
    base_query: str,
    page_size: int = 5000,
    timeout: int = 120,
) -> pd.DataFrame:
    url = f"{SUPABASE_URL}/rest/v1/{table}?{base_query}"
    headers = rest_headers(access_token) | {"Range-Unit": "items"}

    all_rows: list[dict] = []
    start = 0

    while True:
        end = start + page_size - 1
        h = headers | {"Range": f"{start}-{end}"}
        response = requests.get(url, headers=h, timeout=timeout)
        if not response.ok:
            raise RuntimeError(f"GET {table} failed ({response.status_code}): {response.text}")

        batch = response.json()
        if not batch:
            break

        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size

    return pd.DataFrame(all_rows)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_calendar_quality_cached(access_token: str) -> pd.DataFrame:
    raw = rest_get_paged(
        access_token=access_token,
        table="gps_records",
        base_query="select=datum,type,event&event=eq.Summary&order=datum.desc",
        page_size=5000,
        timeout=120,
    )
    if raw.empty:
        return pd.DataFrame(columns=["Datum", "Type", "Event"])

    df = raw.rename(columns={"datum": "Datum", "type": "Type", "event": "Event"}).copy()
    df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce").dt.date
    for col in ["Type", "Event"]:
        df[col] = df[col].astype(str).str.strip()
    return df.dropna(subset=["Datum"]).copy()


def _match_type_bucket(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if any(token in normalized for token in ["oefen", "friendly", "vriend", "test"]):
        return MATCH_FILTER_FRIENDLY
    return MATCH_FILTER_REGULAR


@st.cache_data(ttl=300, show_spinner=False)
def fetch_match_reports_quality_cached(access_token: str) -> pd.DataFrame:
    raw = rest_get_paged(
        access_token=access_token,
        table="matches",
        base_query="select=match_id,match_date,opponent,match_type&order=match_date.desc",
        page_size=5000,
        timeout=120,
    )
    if raw.empty:
        return pd.DataFrame(columns=["match_id", "match_date", "opponent", "match_type", "match_bucket"])

    df = raw.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.date
    df["opponent"] = df["opponent"].fillna("").astype(str).str.strip()
    df["match_type"] = df["match_type"].fillna("").astype(str).str.strip()
    df["match_bucket"] = df["match_type"].apply(_match_type_bucket)
    return df.dropna(subset=["match_id", "match_date"]).copy()


def render_summary_cards(cards: list[tuple[str, str, str]]) -> None:
    summary_markup = "".join(
        f"""<div class="mgmt-card">
<div class="mgmt-label">{label}</div>
<div class="mgmt-value">{value}</div>
<div class="mgmt-foot">{foot}</div>
</div>"""
        for label, value, foot in cards
    )

    st.markdown(
        f"""
        <div class="mgmt-summary-grid">
          {summary_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_section(section_label: str, section_title: str, cards: list[tuple[str, str, str]]) -> None:
    st.markdown(
        f"""
        <div class="mgmt-tab-shell" style="margin-bottom: 0.85rem;">
          <div class="mgmt-section-label">{section_label}</div>
          <div class="mgmt-section-title">{section_title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_summary_cards(cards)


def render_management_css() -> None:
    background = (
        f"linear-gradient(180deg, rgba(6, 10, 20, 0.82) 0%, rgba(6, 10, 20, 0.80) 100%), "
        f"radial-gradient(circle at top left, rgba(200, 16, 46, 0.16), rgba(200, 16, 46, 0.02) 24%, transparent 46%), "
        f"radial-gradient(circle at top right, rgba(234, 51, 81, 0.10), rgba(234, 51, 81, 0.02) 18%, transparent 42%), "
        f"url('{PAGE_BG_URI}')"
        if PAGE_BG_URI
        else "radial-gradient(circle at top left, rgba(200, 16, 46, 0.28), rgba(200, 16, 46, 0.03) 26%, transparent 48%), radial-gradient(circle at top right, rgba(234, 51, 81, 0.18), rgba(234, 51, 81, 0.03) 18%, transparent 44%), linear-gradient(180deg, #070c18 0%, #0a1020 100%)"
    )
    st.markdown(
        """
        <style>
        .stApp {
          background: __MANAGEMENT_BACKGROUND__ !important;
          background-size: cover !important;
          background-position: center top !important;
          background-attachment: fixed !important;
        }

        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, rgba(16, 23, 38, 0.98), rgba(9, 13, 23, 0.98)) !important;
          border-right: 1px solid rgba(255,255,255,0.06) !important;
        }

        .block-container {
          padding-top: 1.2rem !important;
          padding-bottom: 2.4rem !important;
          max-width: 1380px !important;
        }

        .mgmt-hero-shell {
          display: flex;
          flex-direction: column;
          gap: 1.1rem;
          margin-bottom: 1.4rem;
        }

        .mgmt-hero,
        .mgmt-card,
        .mgmt-tab-shell {
          border-radius: 8px;
          border: 1px solid rgba(234, 51, 81, 0.14);
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
          box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
        }

        .mgmt-hero {
          min-height: 280px;
          padding: 2rem 1.75rem 1.85rem 1.75rem;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(135deg, rgba(18, 25, 42, 0.88), rgba(10, 15, 27, 0.84));
        }

        .mgmt-logo {
          width: 82px;
          height: 82px;
          object-fit: contain;
          margin-bottom: 0.9rem;
          filter: drop-shadow(0 8px 22px rgba(0,0,0,0.28));
        }

        .mgmt-kicker {
          color: rgba(255,255,255,0.76);
          font-size: 0.74rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.18em;
          margin-bottom: 0.35rem;
        }

        .mgmt-title {
          margin: 0;
          font-size: 2.55rem;
          line-height: 1;
          font-weight: 800;
          color: #ffffff;
        }

        .mgmt-copy {
          margin-top: 0.8rem;
          max-width: 74ch;
          color: rgba(255,255,255,0.84);
          line-height: 1.62;
        }

        .mgmt-pill-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
          margin-top: 1rem;
        }

        .mgmt-pill {
          display: inline-flex;
          align-items: center;
          padding: 0.42rem 0.76rem;
          border-radius: 999px;
          font-size: 0.78rem;
          font-weight: 800;
          border: 1px solid rgba(234, 51, 81, 0.22);
          background: rgba(255,255,255,0.06);
          color: rgba(255,255,255,0.92);
        }

        .mgmt-summary-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 1rem;
        }

        .mgmt-card {
          min-height: 120px;
          padding: 1rem 1.05rem 0.95rem 1.05rem;
        }

        .mgmt-label {
          color: rgba(255,255,255,0.68);
          font-size: 0.8rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }

        .mgmt-value {
          margin-top: 0.55rem;
          font-size: 1.95rem;
          line-height: 1.1;
          font-weight: 800;
          color: #ffffff;
        }

        .mgmt-foot {
          margin-top: 0.65rem;
          color: rgba(255,255,255,0.8);
          font-size: 0.86rem;
          line-height: 1.4;
        }

        .mgmt-tab-shell {
          padding: 1rem 1rem 1.15rem 1rem;
          margin-bottom: 1rem;
        }

        .mgmt-section-label {
          color: rgba(255,255,255,0.62);
          font-size: 0.75rem;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }

        .mgmt-section-title {
          margin-top: 0.25rem;
          color: #ffffff;
          font-size: 1.05rem;
          font-weight: 700;
        }

        .mgmt-empty {
          padding: 1.4rem 1.2rem;
          border-radius: 8px;
          border: 1px dashed rgba(234, 51, 81, 0.22);
          background: rgba(255,255,255,0.03);
          color: rgba(255,255,255,0.82);
        }

        @media (max-width: 1100px) {
          .mgmt-summary-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
        }

        @media (max-width: 768px) {
          .mgmt-hero {
            min-height: auto;
            padding: 1.55rem 1rem;
          }

          .mgmt-title {
            font-size: 2rem;
          }

          .mgmt-summary-grid {
            grid-template-columns: 1fr;
          }
        }
        </style>
        """.replace("__MANAGEMENT_BACKGROUND__", background),
        unsafe_allow_html=True,
    )


def render_management_hero(role_ui: str) -> None:
    logo_markup = (
        f'<img src="{TEAM_LOGO_URI}" alt="MVV Maastricht" class="mgmt-logo" />'
        if TEAM_LOGO_URI
        else ""
    )
    st.markdown(
        f"""
        <div class="mgmt-hero-shell">
          <div class="mgmt-hero">
            {logo_markup}
            <div class="mgmt-kicker">MVV Maastricht | Management | Staff</div>
            <h1 class="mgmt-title">Management</h1>
            <div class="mgmt-copy">
              Centrale beheeromgeving voor datakwaliteit, GPS import/export en toekomstige instellingen.
              Zo blijven de analysepagina's compact en zitten operationele workflows op een plek.
            </div>
            <div class="mgmt-pill-row">
              <span class="mgmt-pill">Rol: {role_ui.title()}</span>
              <span class="mgmt-pill">Datakwaliteit en import gebundeld</span>
              <span class="mgmt-pill">Settings alvast voorbereid</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quality_tab(access_token: str) -> None:
    try:
        calendar_df = fetch_calendar_quality_cached(access_token)
    except Exception as exc:
        st.warning(f"Kon GPS Summary-datakwaliteit niet laden: {exc}")
        calendar_df = pd.DataFrame(columns=["Datum", "Type", "Event"])

    if calendar_df.empty:
        st.info("Nog geen Summary-data gevonden voor GPS-datakwaliteit.")
    else:
        sessions_df = calendar_df[["Datum", "Type", "Event"]].drop_duplicates().copy()
        sessions_df = sessions_df.sort_values(["Datum", "Type", "Event"], ascending=[False, True, True]).reset_index(drop=True)

        session_days = int(sessions_df["Datum"].nunique())
        unique_sessions = int(len(sessions_df))
        session_types = int(
            sessions_df["Type"].astype(str).str.strip().replace("", pd.NA).dropna().nunique()
        )
        multi_session_days = int(
            sessions_df.groupby("Datum")["Type"].nunique().gt(1).sum()
        )
        latest_session_day = sessions_df["Datum"].max()

        summary_cards = [
            ("Sessiedagen", str(session_days), "Unieke dagen met Summary-sessies"),
            ("Sessies", str(unique_sessions), "Unieke datum/type/event-combinaties"),
            ("Datatypen", str(session_types), "Trainings- en wedstrijdbuckets in de dataset"),
            (
                "Dubbelsessies",
                str(multi_session_days),
                f"Laatste sessiedag: {latest_session_day.strftime('%d-%m-%Y') if latest_session_day else '--'}",
            ),
        ]
        render_summary_cards(summary_cards)

        st.markdown(
            """
            <div class="mgmt-tab-shell">
              <div class="mgmt-section-label">Datakwaliteit</div>
              <div class="mgmt-section-title">Sessie-overzicht per type en recente kalenderitems</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        type_summary = (
            sessions_df.groupby("Type", dropna=False)
            .size()
            .reset_index(name="Sessies")
            .sort_values(["Sessies", "Type"], ascending=[False, True])
            .reset_index(drop=True)
        )
        type_summary["Type"] = type_summary["Type"].replace("", "Onbekend")

        multi_day_df = (
            sessions_df.groupby("Datum", as_index=False)
            .agg(
                aantal_sessies=("Type", "nunique"),
                types=("Type", lambda values: " | ".join(sorted({str(v).strip() for v in values if str(v).strip()}))),
            )
            .sort_values("Datum", ascending=False)
            .reset_index(drop=True)
        )
        multi_day_df = multi_day_df[multi_day_df["aantal_sessies"] > 1].copy()

        recent_sessions = sessions_df.rename(
            columns={"Datum": "Datum", "Type": "Type", "Event": "Event"}
        ).copy()
        recent_sessions["Datum"] = recent_sessions["Datum"].apply(
            lambda value: value.strftime("%d-%m-%Y") if pd.notna(value) else "--"
        )

        col_left, col_right = st.columns(2, gap="large")
        with col_left:
            st.dataframe(type_summary, use_container_width=True, hide_index=True)
        with col_right:
            if multi_day_df.empty:
                st.info("Geen dagen met meerdere Summary-sessies gevonden.")
            else:
                multi_day_df["Datum"] = multi_day_df["Datum"].apply(
                    lambda value: value.strftime("%d-%m-%Y") if pd.notna(value) else "--"
                )
                st.dataframe(
                    multi_day_df.rename(columns={"aantal_sessies": "Sessies", "types": "Typen"}),
                    use_container_width=True,
                    hide_index=True,
                )

        st.dataframe(
            recent_sessions.head(30),
            use_container_width=True,
            hide_index=True,
            height=460,
        )

    st.markdown(
        """
        <div class="mgmt-tab-shell" style="margin-top: 1rem;">
          <div class="mgmt-section-label">Module-overzichten</div>
          <div class="mgmt-section-title">Samenvattingskaarten verplaatst uit de losse pagina's</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        matches_df = fetch_match_reports_quality_cached(access_token)
    except Exception as exc:
        st.warning(f"Match Reports samenvatting kon niet geladen worden: {exc}")
    else:
        total_matches = len(matches_df)
        regular_count = int((matches_df["match_bucket"] == MATCH_FILTER_REGULAR).sum()) if not matches_df.empty else 0
        friendly_count = int((matches_df["match_bucket"] == MATCH_FILTER_FRIENDLY).sum()) if not matches_df.empty else 0
        opponent_count = (
            matches_df["opponent"].replace("", pd.NA).dropna().nunique()
            if "opponent" in matches_df
            else 0
        )
        render_summary_section(
            "Match Reports",
            "Verplaatste samenvatting van de rapportagepagina",
            [
                ("Wedstrijden", str(total_matches), "Totaal beschikbare match reports"),
                ("Normaal", str(regular_count), "Competitie, beker en overige officiele duels"),
                ("Oefen", str(friendly_count), "Oefenwedstrijden en vriendschappelijke duels"),
                ("Tegenstanders", str(opponent_count), "Unieke opponenten in deze rapportage"),
            ]
        )

    sb = get_sb()
    if sb is None:
        st.warning("Wellness samenvatting kon niet geladen worden: Supabase client niet beschikbaar.")
    else:
        sb_url_key = str(st.secrets.get("SUPABASE_URL", "sb"))
        try:
            wellness_players = fetch_active_players_cached(sb_url_key, sb, ttl_salt="management_quality")
        except Exception as exc:
            st.warning(f"Wellness samenvatting kon niet geladen worden: {exc}")
        else:
            render_summary_section(
                "Wellness & RPE",
                "Verplaatste samenvatting van de wellness-overview",
                [
                    ("Selectiespelers", str(len(wellness_players)), "Actieve spelers in deze wellness-overview"),
                    ("Views", "4", "Dag, Week, Injury en Checklist"),
                    ("Rol", "Staff", "Deze pagina is alleen beschikbaar voor de staf"),
                    ("Focus", "RPE + Wellness", "Dagelijkse monitoring en teamoverzicht"),
                ]
            )

    try:
        _, player_options = get_players_map(access_token)
    except Exception as exc:
        st.warning(f"GPS Import samenvatting kon niet geladen worden: {exc}")
    else:
        render_summary_section(
            "GPS Import",
            "Verplaatste samenvatting van de importomgeving",
            [
                ("Rol", "Staff", "Autorisatie voor import, export en matchbeheer"),
                ("Spelers", str(len(player_options)), "Beschikbare spelers voor handmatige invoer"),
                ("Modules", "4", "Import, Manual add, Export en Matches"),
                ("Workflow", "GPS Import", "Dagelijkse data-ingang voor de performance-omgeving"),
            ]
        )


def render_import_export_tab(access_token: str, role_ui: str) -> None:
    st.markdown(
        """
        <div class="mgmt-tab-shell">
          <div class="mgmt-section-label">Import & Export</div>
          <div class="mgmt-section-title">Beheer GPS-workflows vanuit een centrale managementomgeving</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if role_ui not in ALLOWED_IMPORT:
        st.warning("Je rol heeft geen rechten voor import/export in deze omgeving.")
        return

    name_to_id, player_options = get_players_map(access_token)
    workflow = st.radio(
        "Workflow",
        options=["Import (Excel)", "Manual add", "Export", "Matches"],
        horizontal=True,
        key="management_workflow",
    )

    if workflow == "Import (Excel)":
        tab_import_excel_main(access_token=access_token, name_to_id=name_to_id)

    elif workflow == "Manual add":
        tab_manual_add_main(
            access_token=access_token,
            name_to_id=name_to_id,
            player_options=player_options,
        )

    elif workflow == "Export":
        tab_export_main(access_token=access_token, player_options=player_options)

    elif workflow == "Matches":
        tab_matches_main(access_token=access_token)


def render_settings_tab() -> None:
    acwr_mode = get_acwr_mode()
    acwr_meta = get_acwr_mode_meta(acwr_mode)
    mode_labels = {
        ACWR_MODE_STANDARD: "Standaard 4 weken gemiddelde",
        ACWR_MODE_LOG_6W: "Logaritmisch gewogen 6 weken",
    }
    mode_by_label = {label: mode for mode, label in mode_labels.items()}

    st.markdown(
        """
        <div class="mgmt-tab-shell">
          <div class="mgmt-section-label">Settings</div>
          <div class="mgmt-section-title">Globale instellingen voor ACWR en toekomstige beheeropties</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("management_acwr_settings"):
        selected_label = st.radio(
            "ACWR-model",
            options=list(mode_by_label.keys()),
            index=list(mode_by_label.values()).index(acwr_mode),
        )
        selected_mode = mode_by_label[selected_label]
        selected_meta = get_acwr_mode_meta(selected_mode)

        st.info(
            f"Huidig model: {selected_meta['label']}. {selected_meta['description']}"
        )
        st.caption("Deze instelling wordt in je browser opgeslagen en gebruikt op Home, Team Page Beta en GPS ACWR.")

        save_clicked = st.form_submit_button("ACWR-instelling opslaan", use_container_width=True)

    if save_clicked:
        set_acwr_mode(selected_mode)
        st.success(f"ACWR-model opgeslagen: {selected_meta['label']}.")
        st.rerun()

    st.markdown(
        """
        <div class="mgmt-empty" style="margin-top: 1rem;">
          Settings is nu actief voor ACWR. Hier kunnen later ook andere applicatie-instellingen, mappings en beheeropties komen.
        </div>
        """,
        unsafe_allow_html=True,
    )


render_management_css()

sb = get_sb()
profile = get_profile(sb) if sb is not None else None
if not is_staff_user(profile):
    st.error("Geen toegang: deze pagina is alleen voor staff.")
    st.stop()
render_sidebar_navigation(profile)

access_token = get_access_token()
if not access_token:
    st.error("Niet ingelogd (access_token ontbreekt).")
    st.stop()

try:
    _, _, role, _ = get_profile_role(access_token)
except Exception:
    role = st.session_state.get("role", "staff")

role_ui = str(role or st.session_state.get("role") or "staff").strip().lower()
render_management_hero(role_ui)

tab_quality, tab_import, tab_settings = st.tabs(
    ["Datakwaliteit", "Import & Export", "Settings"]
)

with tab_quality:
    render_quality_tab(access_token)

with tab_import:
    render_import_export_tab(access_token, role_ui)

with tab_settings:
    render_settings_tab()

render_sidebar_footer(profile)
