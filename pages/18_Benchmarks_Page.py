from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from pages.Subscripts.mvv_branding import TEAM_HERO_BG, TEAM_LOGO, build_data_uri
from roles import (
    get_profile,
    get_sb,
    is_staff_user,
    render_sidebar_footer,
    render_sidebar_navigation,
    require_auth,
)
from utils.streamlit_ui import apply_streamlit_chrome


st.set_page_config(page_title="Benchmarks", layout="wide", initial_sidebar_state="expanded")
apply_streamlit_chrome()

ROOT_DIR = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT_DIR / "Assets" / "Benchmarks"
BENCHMARKS_PDF = BENCHMARKS_DIR / "Positional_Benchmarks_MVV.pdf"
BENCHMARKS_KKD_IMAGE = BENCHMARKS_DIR / "benchmarks_kkd.png"
BENCHMARKS_EREDIVISIE_IMAGE = BENCHMARKS_DIR / "benchmarks_eredivisie.png"
BENCHMARKS_COMPARE_IMAGE = BENCHMARKS_DIR / "benchmarks_vergelijking.png"

PAGE_BG_URI = build_data_uri(TEAM_HERO_BG)
TEAM_LOGO_URI = build_data_uri(TEAM_LOGO)
KKD_IMAGE_URI = build_data_uri(BENCHMARKS_KKD_IMAGE)
EREDIVISIE_IMAGE_URI = build_data_uri(BENCHMARKS_EREDIVISIE_IMAGE)
COMPARE_IMAGE_URI = build_data_uri(BENCHMARKS_COMPARE_IMAGE)

TABLE_COLUMNS = [
    "Positie",
    "Totale afstand (m)",
    "HI afstand >20.0 (m)",
    "Sprintafstand >25.0 (m)",
    "Runs count >15.0 (#)",
    "Sprint count >25.0 (#)",
    "Tot. dist. (m/min)",
    "Intensiteit (%)",
]

KKD_BENCHMARKS = pd.DataFrame(
    [
        ["AM", "12.561", "1.058", "218", "46,4", "10,6", "132", "8,4%"],
        ["CB", "10.913", "651", "149", "29,6", "7,3", "115", "6,0%"],
        ["CF", "11.720", "1.087", "278", "44,6", "13,2", "123", "9,3%"],
        ["CM", "12.478", "948", "179", "43,2", "9,0", "131", "7,6%"],
        ["DM", "12.282", "845", "158", "39,1", "7,9", "129", "6,9%"],
        ["GK", "5.973", "39", "6", "2,9", "0,3", "63", "0,7%"],
        ["LB", "11.309", "973", "273", "37,7", "12,2", "119", "8,6%"],
        ["LW", "11.602", "1.126", "318", "43,1", "14,1", "122", "9,7%"],
        ["RB", "11.488", "1.000", "283", "38,6", "12,4", "121", "8,7%"],
        ["RW", "11.659", "1.171", "345", "44,3", "15,1", "123", "10,0%"],
    ],
    columns=TABLE_COLUMNS,
)

EREDIVISIE_BENCHMARKS = pd.DataFrame(
    [
        ["AM", "12.139", "1.019", "202", "38,1", "9,0", "128", "8,4%"],
        ["CB", "10.579", "615", "138", "22,6", "5,9", "111", "5,8%"],
        ["CF", "11.193", "992", "235", "35,1", "10,0", "118", "8,9%"],
        ["CM", "11.700", "970", "215", "34,2", "9,1", "123", "8,3%"],
        ["DEF", "9.931", "635", "183", "19,3", "6,3", "105", "6,4%"],
        ["DM", "11.912", "882", "171", "33,7", "7,6", "125", "7,4%"],
        ["FOR", "10.931", "1.225", "352", "37,0", "14,8", "115", "11,2%"],
        ["GK", "5.436", "25", "3", "1,2", "0,1", "57", "0,5%"],
        ["LB", "11.029", "990", "277", "30,9", "10,9", "116", "9,0%"],
        ["LW", "11.415", "1.120", "308", "36,0", "11,9", "120", "9,8%"],
        ["MID", "11.201", "1.032", "263", "33,6", "9,2", "118", "9,2%"],
        ["RB", "11.203", "981", "274", "30,5", "10,7", "118", "8,8%"],
        ["RW", "11.463", "1.143", "309", "36,7", "12,2", "121", "10,0%"],
    ],
    columns=TABLE_COLUMNS,
)

COMPARISON_BENCHMARKS = pd.DataFrame(
    [
        ["AM", "-422", "-39", "-16", "-8,3", "-1,6", "-4", "0,0%"],
        ["CB", "-334", "-36", "-11", "-7,0", "-1,4", "-4", "-0,2%"],
        ["CF", "-527", "-95", "-43", "-9,5", "-3,2", "-5", "-0,4%"],
        ["CM", "-778", "+22", "+36", "-9,0", "+0,1", "-8", "+0,7%"],
        ["DM", "-370", "+37", "+13", "-5,4", "-0,3", "-4", "+0,5%"],
        ["GK", "-537", "-14", "-3", "-1,7", "-0,2", "-6", "-0,2%"],
        ["LB", "-280", "+17", "+4", "-6,8", "-1,3", "-3", "+0,4%"],
        ["LW", "-187", "-6", "-10", "-7,1", "-2,2", "-2", "+0,1%"],
        ["RB", "-285", "-19", "-9", "-8,1", "-1,7", "-3", "+0,1%"],
        ["RW", "-196", "-28", "-36", "-7,6", "-2,9", "-2", "0,0%"],
    ],
    columns=TABLE_COLUMNS,
)

BENCHMARK_TABLES = [
    ("KKD", "Keuken Kampioen Divisie 2024/2025", "10 posities, per 90 minuten", KKD_BENCHMARKS),
    ("Eredivisie", "Dutch Eredivisie 2025/2026", "13 posities, per 90 minuten", EREDIVISIE_BENCHMARKS),
    ("Vergelijking", "Eredivisie minus KKD", "Alleen overlappende posities", COMPARISON_BENCHMARKS),
]

MARKS_TABLES = BENCHMARK_TABLES[:2]
COMPARE_TABLE = BENCHMARK_TABLES[2]


def parse_metric_value(value: object) -> float:
    text = str(value).strip().replace("%", "")
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    else:
        text = text.replace(".", "")
    try:
        return float(text)
    except ValueError:
        return 0.0


def build_stat_card(label: str, value: str, note: str) -> str:
    return f"""
    <div class="bench-stat-card">
      <div class="bench-stat-label">{label}</div>
      <div class="bench-stat-value">{value}</div>
      <div class="bench-stat-note">{note}</div>
    </div>
    """


def build_visual_card(title: str, note: str, image_uri: str | None) -> str:
    image_markup = (
        f'<img src="{image_uri}" alt="{title}" class="bench-visual-image" />'
        if image_uri
        else '<div class="bench-visual-empty">Visual niet beschikbaar</div>'
    )
    return f"""
    <div class="bench-visual-card">
      <div class="bench-visual-label">Visual</div>
      <div class="bench-visual-title">{title}</div>
      <div class="bench-visual-note">{note}</div>
      {image_markup}
    </div>
    """


@st.cache_data(show_spinner=False)
def load_pdf_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def render_css() -> None:
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
          background: __BENCH_BG__;
          background-size: cover;
          background-position: center top;
          background-attachment: fixed;
        }

        .block-container {
          max-width: 1380px;
          padding-top: 1.4rem;
          padding-bottom: 2.4rem;
        }

        .bench-hero {
          border-radius: 10px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(135deg, rgba(18, 25, 42, 0.88), rgba(10, 15, 27, 0.84));
          padding: 1.85rem 1.6rem;
          box-shadow: 0 18px 34px rgba(0, 0, 0, 0.22);
          margin-bottom: 1.2rem;
        }

        .bench-head {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 1rem;
          margin-bottom: 0.9rem;
        }

        .bench-logo {
          width: 78px;
          height: 78px;
          object-fit: contain;
          flex-shrink: 0;
          filter: drop-shadow(0 8px 22px rgba(0,0,0,0.28));
        }

        .bench-copyhead {
          display: flex;
          flex-direction: column;
          justify-content: center;
          gap: 0.12rem;
          text-align: left;
        }

        .bench-title {
          margin: 0;
          font-size: 2.45rem;
          line-height: 1;
          font-weight: 800;
          color: #ffffff;
        }

        .bench-kicker {
          color: rgba(255,255,255,0.76);
          font-size: 0.74rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.18em;
        }

        .bench-copy {
          max-width: 72ch;
          color: rgba(255,255,255,0.84);
          line-height: 1.55;
        }

        .bench-pill-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
          margin-top: 0.95rem;
        }

        .bench-pill {
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

        .bench-sheet-card {
          border-radius: 12px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(180deg, rgba(17, 23, 38, 0.96), rgba(11, 16, 29, 0.96));
          padding: 0.95rem;
          box-shadow: 0 14px 28px rgba(0, 0, 0, 0.18);
        }

        .bench-sheet-kicker {
          color: rgba(255,255,255,0.62);
          font-size: 0.74rem;
          font-weight: 800;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          margin-bottom: 0.35rem;
        }

        .bench-sheet-title {
          color: #ffffff;
          font-size: 1.05rem;
          font-weight: 800;
          margin-bottom: 0.18rem;
        }

        .bench-sheet-note {
          color: rgba(255,255,255,0.72);
          font-size: 0.88rem;
          margin-bottom: 0.8rem;
        }

        .bench-table-note {
          color: rgba(255,255,255,0.66);
          font-size: 0.8rem;
          margin: 0.75rem 0 0 0;
        }

        .bench-download-card {
          border-radius: 12px;
          border: 1px solid rgba(234, 51, 81, 0.16);
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.96), rgba(11, 16, 29, 0.96));
          padding: 1rem 1rem 0.9rem 1rem;
          box-shadow: 0 14px 28px rgba(0, 0, 0, 0.18);
          margin-bottom: 1rem;
        }

        .bench-download-label {
          color: rgba(255,255,255,0.62);
          font-size: 0.74rem;
          font-weight: 800;
          letter-spacing: 0.14em;
          text-transform: uppercase;
        }

        .bench-download-title {
          color: #ffffff;
          font-size: 1.15rem;
          font-weight: 800;
          margin-top: 0.35rem;
        }

        .bench-download-copy {
          color: rgba(255,255,255,0.76);
          font-size: 0.88rem;
          line-height: 1.5;
          margin-top: 0.35rem;
        }

        .bench-stat-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 0.75rem;
          margin-bottom: 1rem;
        }

        .bench-stat-card {
          border-radius: 12px;
          border: 1px solid rgba(234, 51, 81, 0.18);
          background: linear-gradient(180deg, rgba(18, 25, 42, 0.94), rgba(11, 16, 29, 0.96));
          padding: 0.9rem 1rem;
          box-shadow: 0 12px 24px rgba(0, 0, 0, 0.16);
        }

        .bench-stat-label {
          color: rgba(255,255,255,0.62);
          font-size: 0.72rem;
          font-weight: 800;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          margin-bottom: 0.4rem;
        }

        .bench-stat-value {
          color: #ffffff;
          font-size: 1.45rem;
          font-weight: 800;
          line-height: 1.05;
          margin-bottom: 0.28rem;
        }

        .bench-stat-note {
          color: rgba(255,255,255,0.74);
          font-size: 0.84rem;
          line-height: 1.4;
        }

        .bench-visual-card {
          border-radius: 12px;
          border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(180deg, rgba(17, 23, 38, 0.96), rgba(11, 16, 29, 0.96));
          padding: 0.95rem;
          box-shadow: 0 14px 28px rgba(0, 0, 0, 0.18);
          margin-bottom: 0.95rem;
        }

        .bench-visual-label {
          color: rgba(255,255,255,0.62);
          font-size: 0.72rem;
          font-weight: 800;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          margin-bottom: 0.28rem;
        }

        .bench-visual-title {
          color: #ffffff;
          font-size: 1.02rem;
          font-weight: 800;
          margin-bottom: 0.18rem;
        }

        .bench-visual-note {
          color: rgba(255,255,255,0.74);
          font-size: 0.84rem;
          line-height: 1.45;
          margin-bottom: 0.75rem;
        }

        .bench-visual-image {
          width: 100%;
          display: block;
          border-radius: 12px;
          border: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.02);
        }

        .bench-visual-empty {
          border-radius: 12px;
          border: 1px dashed rgba(255,255,255,0.16);
          color: rgba(255,255,255,0.56);
          padding: 1rem;
          text-align: center;
        }

        .stTabs [data-baseweb="tab-list"] {
          gap: 0.55rem;
          margin-bottom: 0.85rem;
        }

        .stTabs [data-baseweb="tab"] {
          border-radius: 999px;
          background: rgba(12, 18, 31, 0.82);
          border: 1px solid rgba(255,255,255,0.08);
          color: rgba(255,255,255,0.82);
          font-weight: 800;
          padding: 0.5rem 0.95rem;
        }

        .stTabs [aria-selected="true"] {
          border-color: rgba(234, 51, 81, 0.28);
          color: #ffffff;
        }

        @media (max-width: 768px) {
          .bench-head {
            flex-direction: column;
            gap: 0.8rem;
          }

          .bench-copyhead {
            text-align: center;
          }

          .bench-title {
            font-size: 2rem;
          }
        }
        </style>
        """.replace("__BENCH_BG__", background),
        unsafe_allow_html=True,
    )


def main() -> None:
    render_css()
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

    if not BENCHMARKS_PDF.exists():
        st.error("De benchmark-PDF is niet gevonden in de assets.")
        render_sidebar_footer(profile)
        st.stop()

    logo_markup = (
        f'<img src="{TEAM_LOGO_URI}" alt="MVV Maastricht" class="bench-logo" />'
        if TEAM_LOGO_URI
        else ""
    )
    st.markdown(
        f"""
        <div class="bench-hero">
          <div class="bench-head">
            {logo_markup}
            <div class="bench-copyhead">
              <h1 class="bench-title">Benchmarks</h1>
              <div class="bench-kicker">MVV Maastricht | Data | Benchmarks</div>
            </div>
          </div>
          <div class="bench-copy">
            Positionele referentiekaarten per 90 minuten voor KKD, Eredivisie en het directe verschil tussen beide competities.
          </div>
          <div class="bench-pill-row">
            <span class="bench-pill">KKD 2024/2025</span>
            <span class="bench-pill">Eredivisie 2025/2026</span>
            <span class="bench-pill">Vergelijking per positie</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info_col, download_col = st.columns([0.72, 0.28], gap="large")
    with info_col:
        marks_tab, compare_tab = st.tabs(["Marks", "Compare"])

        with marks_tab:
            kkd_top_td_idx = KKD_BENCHMARKS["Totale afstand (m)"].apply(parse_metric_value).idxmax()
            eredivisie_top_hsr_idx = EREDIVISIE_BENCHMARKS["HI afstand >20.0 (m)"].apply(parse_metric_value).idxmax()
            marks_cards = "".join(
                [
                    build_stat_card("Sets", "2 benchmarkbladen", "KKD en Eredivisie direct naast elkaar"),
                    build_stat_card("KKD posities", str(len(KKD_BENCHMARKS)), "Referenties uit seizoen 2024/2025"),
                    build_stat_card("Eredivisie posities", str(len(EREDIVISIE_BENCHMARKS)), "Referenties uit seizoen 2025/2026"),
                    build_stat_card(
                        "Top volume KKD",
                        f"{KKD_BENCHMARKS.loc[kkd_top_td_idx, 'Positie']} | {KKD_BENCHMARKS.loc[kkd_top_td_idx, 'Totale afstand (m)']} m",
                        "Hoogste totale afstand per 90 minuten",
                    ),
                    build_stat_card(
                        "Top HSR Eredivisie",
                        f"{EREDIVISIE_BENCHMARKS.loc[eredivisie_top_hsr_idx, 'Positie']} | {EREDIVISIE_BENCHMARKS.loc[eredivisie_top_hsr_idx, 'HI afstand >20.0 (m)']} m",
                        "Hoogste high-intensity afstand per 90 minuten",
                    ),
                ]
            )
            st.markdown(f'<div class="bench-stat-grid">{marks_cards}</div>', unsafe_allow_html=True)

            left_col, right_col = st.columns(2, gap="large")
            visual_uris = [KKD_IMAGE_URI, EREDIVISIE_IMAGE_URI]
            for column, (_, title, note, table_df), image_uri in zip((left_col, right_col), MARKS_TABLES, visual_uris):
                with column:
                    st.markdown(
                        build_visual_card(title, "Snelle visuele samenvatting van de belangrijkste positionele referenties.", image_uri),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                        <div class="bench-sheet-card">
                          <div class="bench-sheet-kicker">Benchmarkblad</div>
                          <div class="bench-sheet-title">{title}</div>
                          <div class="bench-sheet-note">{note}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.dataframe(table_df, width="stretch", hide_index=True)
                    st.markdown(
                        '<div class="bench-table-note">Waardes per 90 minuten. Afstanden in meters, totale afstand in m/min en intensiteit in %.</div>',
                        unsafe_allow_html=True,
                    )

        with compare_tab:
            _, title, note, table_df = COMPARE_TABLE
            comparison_cards = "".join(
                [
                    build_stat_card("Overlap", str(len(table_df)), "Posities die 1-op-1 te vergelijken zijn"),
                    build_stat_card(
                        "HSR hoger in ED",
                        str(int((table_df["HI afstand >20.0 (m)"].apply(parse_metric_value) > 0).sum())),
                        "Aantal posities waar Eredivisie hoger ligt",
                    ),
                    build_stat_card(
                        "Sprintafstand hoger in ED",
                        str(int((table_df["Sprintafstand >25.0 (m)"].apply(parse_metric_value) > 0).sum())),
                        "Aantal posities met hogere sprintafstand",
                    ),
                    build_stat_card(
                        "TD hoger in KKD",
                        str(int((table_df["Totale afstand (m)"].apply(parse_metric_value) < 0).sum())),
                        "Aantal posities waar KKD hogere totale afstand heeft",
                    ),
                ]
            )
            st.markdown(f'<div class="bench-stat-grid">{comparison_cards}</div>', unsafe_allow_html=True)
            st.markdown(
                build_visual_card(title, "Visuele vergelijking van het verschil tussen Eredivisie en KKD op overlappende posities.", COMPARE_IMAGE_URI),
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="bench-sheet-card">
                  <div class="bench-sheet-kicker">Vergelijking</div>
                  <div class="bench-sheet-title">{title}</div>
                  <div class="bench-sheet-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(table_df, width="stretch", hide_index=True)
            st.markdown(
                '<div class="bench-table-note">Positieve waardes betekenen hoger in de Eredivisie; negatieve waardes betekenen hoger in de KKD.</div>',
                unsafe_allow_html=True,
            )
    with download_col:
        st.markdown(
            """
            <div class="bench-download-card">
              <div class="bench-download-label">Bronbestand</div>
              <div class="bench-download-title">Positional Benchmarks [MVV]</div>
              <div class="bench-download-copy">
                Download hier de originele PDF zoals die in de Benchmarks-pagina is opgenomen.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download PDF",
            data=load_pdf_bytes(str(BENCHMARKS_PDF)),
            file_name="Positional_Benchmarks_MVV.pdf",
            mime="application/pdf",
            width="stretch",
            key="benchmarks_download_pdf",
        )
        st.page_link("pages/10_Data_Page_Beta.py", label="Terug naar Data")

    render_sidebar_footer(profile)


if __name__ == "__main__":
    main()
