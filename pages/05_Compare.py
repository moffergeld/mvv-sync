from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from pages.Subscripts.mvv_branding import TEAM_LOGO, build_data_uri
from roles import get_access_token, get_profile, get_sb, is_staff_user, render_sidebar_footer, render_sidebar_navigation, require_auth
from pages.Subscripts.compare_page_common import (
    apply_metric_view,
    build_compare_chart,
    build_compare_table,
    build_player_plot_df,
    build_team_average_df,
    fetch_asrm_compare_range_cached,
    fetch_gps_compare_range_cached,
    fetch_players_lookup_cached,
    fetch_rpe_headers_compare_range_cached,
    format_metric_value,
    get_metric_options,
    metric_spec_for_key,
    prepare_asrm_compare_df,
    prepare_gps_compare_df,
    prepare_rpe_compare_df,
)
from utils.streamlit_ui import apply_streamlit_chrome

st.set_page_config(page_title="Compare", layout="wide")
apply_streamlit_chrome()

TEAM_LOGO_URI = build_data_uri(TEAM_LOGO)

COMPARE_CSS = """
<style>
:root {
  --mvv-red: #C8102E;
  --mvv-red-light: #E8213F;
  --glass-border: rgba(255,255,255,0.08);
  --glass-bg: linear-gradient(180deg, rgba(255,255,255,0.045) 0%, rgba(255,255,255,0.022) 100%);
  --text-main: #F5F7FB;
  --text-soft: rgba(245,247,251,0.72);
  --text-dim: rgba(245,247,251,0.48);
}

.stApp {
  background:
    radial-gradient(circle at top left, rgba(200,16,46,0.18) 0%, rgba(200,16,46,0.05) 24%, rgba(0,0,0,0) 48%),
    radial-gradient(circle at bottom right, rgba(85,168,255,0.13) 0%, rgba(85,168,255,0.03) 18%, rgba(0,0,0,0) 40%),
    linear-gradient(180deg, #040915 0%, #050A16 100%);
  color: var(--text-main);
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.022) 100%);
  border-right: 1px solid rgba(255,255,255,0.06);
}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
  color: var(--text-main);
}

.block-container {
  padding-top: 1.7rem;
  padding-bottom: 2rem;
}

.compare-hero {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 28px;
  padding: 22px 24px 18px 24px;
  margin-bottom: 18px;
  background:
    radial-gradient(circle at top left, rgba(200,16,46,0.20) 0%, rgba(200,16,46,0.08) 24%, rgba(0,0,0,0) 58%),
    linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.025) 42%, rgba(255,255,255,0.015) 100%);
  box-shadow: 0 18px 42px rgba(0,0,0,0.22);
}

.compare-logo {
  width: 82px;
  height: 82px;
  object-fit: contain;
  margin-bottom: 0;
  flex-shrink: 0;
  filter: drop-shadow(0 8px 22px rgba(0,0,0,0.28));
}

.compare-head {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  margin-bottom: 1rem;
}

.compare-head-copy {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 0.12rem;
  text-align: left;
}

.compare-kicker {
  font-size: 11px;
  letter-spacing: 0.24em;
  font-weight: 800;
  text-transform: uppercase;
  color: rgba(255,255,255,0.74);
  margin-bottom: 0;
}

.compare-title {
  font-size: 30px;
  font-weight: 800;
  line-height: 1.05;
  color: #FFFFFF;
  margin: 0;
}

.compare-subtitle {
  font-size: 14px;
  line-height: 1.65;
  max-width: 1040px;
  color: rgba(255,255,255,0.82);
  margin-bottom: 14px;
}

.compare-pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.compare-pill {
  border-radius: 999px;
  padding: 8px 12px;
  font-size: 12px;
  font-weight: 700;
  color: white;
  border: 1px solid rgba(255,255,255,0.08);
  background: linear-gradient(180deg, rgba(255,255,255,0.045) 0%, rgba(255,255,255,0.018) 100%);
}

.compare-card {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 22px;
  padding: 14px 16px 12px 16px;
  margin-bottom: 16px;
  background: var(--glass-bg);
  box-shadow: 0 10px 28px rgba(0,0,0,0.16);
}

.compare-card-title {
  font-size: 11px;
  letter-spacing: 0.22em;
  font-weight: 800;
  text-transform: uppercase;
  color: rgba(255,255,255,0.72);
  margin-bottom: 10px;
}

.compare-card-copy {
  font-size: 13px;
  line-height: 1.55;
  color: rgba(255,255,255,0.76);
  margin-bottom: 8px;
}

.compare-stat {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  min-height: 92px;
  padding: 0.85rem 1rem;
  background: rgba(255,255,255,0.035);
}

.compare-stat-label {
  font-size: 0.74rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: rgba(245,247,251,0.62);
  font-weight: 700;
}

.compare-stat-value {
  font-size: 1.42rem;
  color: white;
  font-weight: 800;
  margin-top: 0.35rem;
}

.compare-stat-sub {
  font-size: 0.84rem;
  color: rgba(245,247,251,0.56);
  margin-top: 0.28rem;
}

div[data-baseweb="select"] > div,
.stSelectbox div[data-baseweb="select"] > div,
.stDateInput > div > div,
.stMultiSelect div[data-baseweb="select"] > div {
  border-radius: 14px !important;
}

@media (max-width: 768px) {
  .compare-head {
    flex-direction: column;
    gap: 0.8rem;
  }

  .compare-head-copy {
    text-align: center;
  }

  .compare-title {
    font-size: 34px;
  }
}
</style>
"""


def _default_period() -> tuple[date, date]:
    today = date.today()
    return today - timedelta(days=27), today


def _render_stat_card(label: str, value: str, subtext: str = "") -> None:
    st.markdown(
        f"""
        <div class="compare-stat">
          <div class="compare-stat-label">{label}</div>
          <div class="compare-stat-value">{value}</div>
          <div class="compare-stat-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _resolve_default_selection(options: list[str], current: list[str] | None, fallback_count: int) -> list[str]:
    valid_current = [item for item in (current or []) if item in options]
    if valid_current:
        return valid_current
    return options[: min(fallback_count, len(options))]


def _load_compare_domain_df(
    *,
    domain: str,
    sb,
    sb_url_key: str,
    supabase_url: str,
    supabase_anon_key: str,
    access_token: str,
    d0: date,
    d1: date,
    player_lookup: pd.DataFrame,
) -> pd.DataFrame:
    d0_iso = d0.isoformat()
    d1_iso = d1.isoformat()

    if domain == "GPS":
        raw = fetch_gps_compare_range_cached(
            supabase_url=supabase_url,
            supabase_anon_key=supabase_anon_key,
            access_token=access_token,
            d0_iso=d0_iso,
            d1_iso=d1_iso,
        )
        return prepare_gps_compare_df(raw)

    if domain == "Wellness":
        raw = fetch_asrm_compare_range_cached(sb_url_key, sb, d0_iso, d1_iso)
        return prepare_asrm_compare_df(raw, player_lookup)

    raw_headers = fetch_rpe_headers_compare_range_cached(sb_url_key, sb, d0_iso, d1_iso)
    return prepare_rpe_compare_df(sb_url_key, sb, raw_headers, player_lookup)


def main() -> None:
    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    profile = get_profile(sb)
    if not is_staff_user(profile):
        st.error("Geen toegang: deze pagina is alleen voor staff.")
        st.stop()
    render_sidebar_navigation(profile)

    supabase_url = st.secrets.get("SUPABASE_URL", "").strip()
    supabase_anon_key = st.secrets.get("SUPABASE_ANON_KEY", "").strip()
    if not supabase_url or not supabase_anon_key:
        st.error("Missing secrets: SUPABASE_URL / SUPABASE_ANON_KEY")
        st.stop()

    access_token = get_access_token()
    sb_url_key = str(supabase_url)
    logo_markup = f'<img src="{TEAM_LOGO_URI}" alt="MVV Maastricht" class="compare-logo" />' if TEAM_LOGO_URI else ""

    st.markdown(COMPARE_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="compare-hero">
          <div class="compare-head">
            {logo_markup}
            <div class="compare-head-copy">
              <div class="compare-title">Compare</div>
              <div class="compare-kicker">MVV Performance Dashboard</div>
            </div>
          </div>
          <div class="compare-subtitle">
            Vergelijk GPS, Wellness en RPE over meerdere sessies in een flow.
            Schakel tussen absolute waardes, relatieve aandelen en intensity waar de metric dat ondersteunt.
          </div>
          <div class="compare-pill-row">
            <div class="compare-pill">GPS, Wellness en RPE in 1 pagina</div>
            <div class="compare-pill">Absolute en relatieve views</div>
            <div class="compare-pill">Volume en intensity voor load metrics</div>
            <div class="compare-pill">Team average als referentie</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    player_lookup = fetch_players_lookup_cached(sb_url_key, sb)

    default_start, default_end = _default_period()
    with st.sidebar:
        st.markdown("### Compare instellingen")
        domain = st.selectbox("Dataset", ["GPS", "Wellness", "RPE"], index=0, key="compare_domain")

        metric_specs = get_metric_options(domain)
        if not metric_specs:
            st.error("Geen metrics beschikbaar.")
            st.stop()

        metric_key = st.selectbox(
            "Metric",
            options=[spec.key for spec in metric_specs],
            format_func=lambda key: metric_spec_for_key(key).label,
            key=f"compare_metric_{domain.lower()}",
        )
        metric_spec = metric_spec_for_key(metric_key)

        d0 = st.date_input("Startdatum", value=default_start, key="compare_start")
        d1 = st.date_input("Einddatum", value=default_end, key="compare_end")
        if d0 > d1:
            st.error("Startdatum moet voor de einddatum liggen.")
            st.stop()

        value_options = ["Absoluut", "Relatief"] if metric_spec.supports_relative else ["Absoluut"]
        value_mode = st.selectbox(
            "Waardeweergave",
            options=value_options,
            key=f"compare_value_mode_{metric_key}",
        )

        intensity_disabled = value_mode == "Relatief" or not metric_spec.supports_intensity
        load_options = ["Volume", "Intensity"] if metric_spec.supports_intensity else ["Volume"]
        load_mode = st.selectbox(
            "Loadweergave",
            options=load_options if not intensity_disabled else ["Volume"],
            key=f"compare_load_mode_{metric_key}",
            disabled=intensity_disabled,
        )
        if value_mode == "Relatief" and metric_spec.supports_relative:
            st.caption("Relatieve view gebruikt altijd volume als basis.")

        chart_mode = st.radio(
            "Grafiekmodus",
            options=["Per speler", "Gemiddelde selectie"],
            horizontal=False,
            key="compare_chart_mode",
        )
        show_team_average = st.toggle("Toon team average", value=True, key="compare_team_average")

    with st.spinner(f"{domain} data laden..."):
        domain_df = _load_compare_domain_df(
            domain=domain,
            sb=sb,
            sb_url_key=sb_url_key,
            supabase_url=supabase_url,
            supabase_anon_key=supabase_anon_key,
            access_token=access_token,
            d0=d0,
            d1=d1,
            player_lookup=player_lookup,
        )

    if domain_df.empty:
        st.info("Geen data gevonden voor deze selectie.")
        st.stop()

    metric_df, metric_meta = apply_metric_view(
        domain_df,
        metric_spec,
        value_mode=value_mode,
        load_mode=load_mode,
    )
    if metric_df.empty:
        st.info("Geen bruikbare waardes gevonden voor deze metric in de gekozen periode.")
        st.stop()

    session_options_desc = (
        metric_df[["session_date", "session_label"]]
        .drop_duplicates()
        .sort_values("session_date", ascending=False)
        ["session_label"]
        .tolist()
    )
    player_options = sorted(metric_df["player_name"].dropna().astype(str).unique().tolist())

    if not session_options_desc or not player_options:
        st.info("Onvoldoende data om te vergelijken.")
        st.stop()

    session_state_key = f"compare_session_pick_{domain.lower()}_{metric_key}"
    player_state_key = f"compare_player_pick_{domain.lower()}_{metric_key}"

    if session_state_key in st.session_state:
        st.session_state[session_state_key] = [
            item for item in st.session_state[session_state_key]
            if item in session_options_desc
        ]
    if player_state_key in st.session_state:
        st.session_state[player_state_key] = [
            item for item in st.session_state[player_state_key]
            if item in player_options
        ]

    left_col, right_col = st.columns([0.98, 2.15], gap="large")

    with left_col:
        st.markdown('<div class="compare-card">', unsafe_allow_html=True)
        st.markdown('<div class="compare-card-title">Sessies</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="compare-card-copy">Kies welke dagen je in de vergelijking wilt meenemen.</div>',
            unsafe_allow_html=True,
        )
        selected_sessions = st.multiselect(
            "Selecteer sessies",
            options=session_options_desc,
            default=_resolve_default_selection(
                session_options_desc,
                st.session_state.get(session_state_key),
                fallback_count=6,
            ),
            key=session_state_key,
            label_visibility="collapsed",
            placeholder="Kies sessies",
        )
        if selected_sessions:
            st.caption(f"{len(selected_sessions)} sessie(s) geselecteerd")
        else:
            st.caption("Geen sessies geselecteerd")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="compare-card">', unsafe_allow_html=True)
        st.markdown('<div class="compare-card-title">Spelers</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="compare-card-copy">Vergelijk losse spelers of bekijk het gemiddelde van je selectie.</div>',
            unsafe_allow_html=True,
        )
        selected_players = st.multiselect(
            "Selecteer spelers",
            options=player_options,
            default=_resolve_default_selection(
                player_options,
                st.session_state.get(player_state_key),
                fallback_count=1,
            ),
            key=player_state_key,
            label_visibility="collapsed",
            placeholder="Kies spelers",
        )
        if selected_players:
            st.caption(f"{len(selected_players)} speler(s) geselecteerd")
        else:
            st.caption("Geen spelers geselecteerd")
        st.markdown("</div>", unsafe_allow_html=True)

    render_sidebar_footer(profile)

    plot_df = build_player_plot_df(
        metric_df,
        selected_players=selected_players,
        selected_sessions=selected_sessions,
        chart_mode=chart_mode,
    )
    if show_team_average:
        team_avg_df = build_team_average_df(metric_df, selected_sessions)
    else:
        team_avg_df = pd.DataFrame()

    with right_col:
        effective_view = "Relatief" if metric_meta.get("applied_relative") else value_mode
        effective_load = metric_meta.get("effective_load_mode", "Volume")
        selection_avg = plot_df["value"].mean() if not plot_df.empty else None
        team_avg = team_avg_df["value"].mean() if not team_avg_df.empty else None
        peak_row = (
            plot_df.sort_values("value", ascending=False).iloc[0]
            if not plot_df.empty
            else None
        )

        st.markdown('<div class="compare-card">', unsafe_allow_html=True)
        st.markdown('<div class="compare-card-title">Actieve selectie</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="compare-pill-row">
              <div class="compare-pill">Dataset: {domain}</div>
              <div class="compare-pill">Metric: {metric_meta.get("label", metric_spec.label)}</div>
              <div class="compare-pill">Waarde: {effective_view}</div>
              <div class="compare-pill">Load: {effective_load}</div>
              <div class="compare-pill">Periode: {d0.isoformat()} t/m {d1.isoformat()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            _render_stat_card("Sessies", str(len(selected_sessions)), "Aantal gekozen dagen")
        with s2:
            _render_stat_card("Spelers", str(len(selected_players)), "Aantal gekozen spelers")
        with s3:
            _render_stat_card(
                "Selection avg",
                format_metric_value(selection_avg, metric_meta.get("decimals", 1), metric_meta.get("unit", "")),
                "Gemiddelde van de zichtbare bars",
            )
        with s4:
            peak_value = format_metric_value(
                None if peak_row is None else peak_row["value"],
                metric_meta.get("decimals", 1),
                metric_meta.get("unit", ""),
            )
            peak_sub = "-"
            if peak_row is not None:
                peak_sub = str(peak_row.get("session_label", "-"))
                if "series" in peak_row.index:
                    peak_sub = f"{peak_sub} | {peak_row['series']}"
            _render_stat_card("Peak", peak_value, peak_sub)

        if show_team_average and team_avg is not None:
            st.caption(
                f"Team average over de gekozen sessies: "
                f"{format_metric_value(team_avg, metric_meta.get('decimals', 1), metric_meta.get('unit', ''))}"
            )

        st.markdown('<div class="compare-card">', unsafe_allow_html=True)
        st.markdown('<div class="compare-card-title">Grafiek</div>', unsafe_allow_html=True)
        if not selected_sessions or not selected_players:
            st.info("Selecteer minimaal 1 sessie en 1 speler om de vergelijking te tonen.")
        elif plot_df.empty:
            st.info("Geen overlap tussen deze spelers, sessies en metric.")
        else:
            if chart_mode == "Per speler" and len(selected_players) > 5:
                st.caption("Tip: bij veel spelers wordt 'Gemiddelde selectie' meestal leesbaarder.")

            fig = build_compare_chart(
                plot_df,
                team_avg_df=team_avg_df,
                chart_title=f"{metric_meta.get('label', metric_spec.label)} per sessie",
                y_title=f"{metric_meta.get('label', metric_spec.label)} ({metric_meta.get('unit', metric_spec.unit)})",
                decimals=metric_meta.get("decimals", 1),
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="compare-card">', unsafe_allow_html=True)
        st.markdown('<div class="compare-card-title">Tabel</div>', unsafe_allow_html=True)
        if plot_df.empty:
            st.info("Geen tabeldata beschikbaar voor deze selectie.")
        else:
            table_df = build_compare_table(
                plot_df,
                team_avg_df=team_avg_df,
                decimals=metric_meta.get("decimals", 1),
                chart_mode=chart_mode,
            )
            st.dataframe(table_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
