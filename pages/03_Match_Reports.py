# pages/02_Match_Reports.py
# ============================================================
# Streamlit — Match Reports
# Gebaseerd op match_analysis.py :contentReference[oaicite:0]{index=0}
#
# Features:
# - Selecteer wedstrijd (Matches sheet)
# - Preview “match card” + dashboard (matplotlib -> Streamlit)
# - Genereer PDF Match Report en download
# - Team-logo’s uit repo-map: Assets/Afbeeldingen/Team_Logos
#
# Verwacht in Database.xlsx:
# - Sheet "Matches" met kolommen: Datum, Tegenstander, Home/Away, Goals for, Goals against (optioneel Wedstrijd)
# - Sheet "GPS" met kolommen: Datum, Speler, Event (Summary/First Half/Second Half), metrics
# - Sheet "Spelerlijst" met kolommen: Naam (of Voornaam/Achternaam), Activity, Subpositie (optioneel)
# ============================================================

from __future__ import annotations

import io
import os
import re
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageChops, ImageOps


# =========================
# App config
# =========================
st.set_page_config(page_title="Match Reports", layout="wide")

# MVV kleuren
MVV_RED = "#FF0033"


# =========================
# Repo paths (logo's)
# =========================
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent  # pages/ -> repo root
TEAM_LOGOS_DIR = REPO_ROOT / "Assets" / "Afbeeldingen" / "Team_Logos"

LOGO_EXTS = [".png", ".jpg", ".jpeg", ".webp"]
LOGO_MAX_HEIGHT_PX = 175
LOGO_ZOOM = 1.0
TRIM_ALPHA_EDGES = True
TRIM_WHITE_EDGES = True


# =========================
# Excel sheet/kolommen (zelfde als voorbeeld)
# =========================
MATCHES_SHEET = "Matches"
GPS_SHEET = "GPS"
PLAYERS_SHEET = "Spelerlijst"

COL_MATCH_DATE = "Datum"
COL_MATCH_Match = "Wedstrijd"
COL_MATCH_OPP = "Tegenstander"
COL_MATCH_HOMEAWAY = "Home/Away"  # H/A of Home/Away
COL_MATCH_GF = "Goals for"
COL_MATCH_GA = "Goals against"

COL_DATE = "Datum"
COL_PLAYER = "Speler"
COL_EVENT = "Event"  # Summary / First Half / Second Half

# Kernmetrics
COL_TD = "Total Distance"
COL_RUN = "Running"
COL_SPRINT = "Sprint"
COL_HS = "High Sprint"

# Optioneel
COL_ACC_TOT = "Total Accelerations"
COL_ACC_HI = "High Accelerations"
COL_DEC_TOT = "Total Decelerations"
COL_DEC_HI = "High Decelerations"
HR_COLS = ["HRzone1", "HRzone2", "HRzone3", "HRzone4", "HRzone5"]
COL_TRIMP = "HR Trimp"
COL_MAX_SPEED = "Max Speed"

DURATION_CANDIDATES = ["Duration", "Minutes", "Duration (min)", "Session Time (min)", "Duur (min)"]

SUBPOS_CANDIDATES = ["Subpositie", "Subposities", "Subpos", "SubPositie", "SubPosities"]
COL_SUBPOS_OUT = "Subpositie"

COL_PL_LASTNAME = "Achternaam"
COL_PL_FIRSTNAME = "Voornaam"
COL_PL_FULLNAME = "Naam"
COL_PL_ACTIVITY = "Activity"


DISPLAY_TITLES = {
    COL_TD: "TD",
    COL_RUN: "14,4 - 19,8",
    COL_SPRINT: "19.8 - 25.2",
    COL_HS: "25.2+",
    COL_MAX_SPEED: "Max Speed",
}

SIDE_TABLE_METRICS = [COL_TD, COL_RUN, COL_SPRINT, COL_HS]
SIDE_TABLES_PER_ROW = 5
SIDE_TABLE_BASE_FONTSIZE = 6
SIDE_TABLE_MIN_FONTSIZE = 5
SIDE_TABLE_PLAYER_COLW = 0.52
SIDE_TABLE_SCALE_Y = 1.00

GRADIENT_CMAP = LinearSegmentedColormap.from_list(
    "red_orange_green", [(0.00, "#d73027"), (0.50, "#fdae61"), (1.00, "#1a9850")]
)
GRADIENT_ALPHA = 0.75


# =========================
# Helpers (logo)
# =========================
def _sanitize_filename(s: str) -> str:
    return str(s).replace("/", "-").replace("\\", "-").replace(":", "").strip()


def _norm_name(s: str) -> str:
    t = str(s).lower().strip()
    for pre in ("sc ", "fc ", "vv ", "ssv ", "rksv ", "rkvv ", "sv ", "sbv "):
        if t.startswith(pre):
            t = t[len(pre) :]
            break
    return "".join(ch for ch in t if ch.isalnum())


def find_logo_path(club_name: str) -> str | None:
    if not club_name:
        return None
    if club_name.strip().lower() in {"mvv", "mvv maastricht"}:
        club_name = "MVV Maastricht"

    base = _sanitize_filename(club_name)
    base_dir = TEAM_LOGOS_DIR

    if not base_dir.exists():
        return None

    # 1) direct filename
    for ext in LOGO_EXTS:
        for cand in (base, base.lower(), base.upper()):
            p = base_dir / f"{cand}{ext}"
            if p.is_file():
                return str(p)

    # 2) normalised match
    want = _norm_name(base)
    try:
        for fn in os.listdir(base_dir):
            name, ext = os.path.splitext(fn)
            if ext.lower() in LOGO_EXTS and _norm_name(name) == want:
                return str(base_dir / fn)
    except Exception:
        pass

    return None


def _autocrop_logo(img: Image.Image, trim_alpha: bool = True, trim_white: bool = True, white_tol: int = 245) -> Image.Image:
    img = ImageOps.exif_transpose(img)

    if trim_alpha and "A" in img.getbands():
        alpha = img.split()[-1]
        bb = ImageChops.difference(alpha, Image.new("L", alpha.size, 0)).getbbox()
        if bb:
            img = img.crop(bb)

    if trim_white:
        rgb = img.convert("RGB")
        bg = Image.new("RGB", rgb.size, (255, 255, 255))
        diff = ImageChops.difference(rgb, bg)
        diff = ImageOps.autocontrast(diff)
        bb = diff.getbbox()
        if bb:
            img = img.crop(bb)

    return img


def _prepare_logo_image(path: str, target_height_px: int = LOGO_MAX_HEIGHT_PX) -> Image.Image | None:
    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        return None

    img = _autocrop_logo(img, trim_alpha=TRIM_ALPHA_EDGES, trim_white=TRIM_WHITE_EDGES)

    w, h = img.size
    if h <= 0:
        return img
    scale = target_height_px / float(h)
    new_size = (max(1, int(round(w * scale))), target_height_px)
    return img.resize(new_size, Image.LANCZOS)


def draw_logo(ax, club_name: str, x: float, y: float, zoom: float | None = None):
    path = find_logo_path(club_name)
    if not path:
        return
    img = _prepare_logo_image(path, target_height_px=LOGO_MAX_HEIGHT_PX)
    if img is None:
        return
    oi = OffsetImage(img, zoom=(LOGO_ZOOM if zoom is None else zoom))
    ab = AnnotationBbox(oi, (x, y), frameon=False, xycoords="axes fraction")
    ax.add_artist(ab)


# =========================
# Helpers (data)
# =========================
def _normalize_event(e: str) -> str:
    s = str(e).strip().lower()
    if s in {"first half", "1st half", "firsthalf", "h1", "1e helft", "1ste helft"}:
        return "first"
    if s in {"second half", "2nd half", "secondhalf", "h2", "2e helft", "2de helft"}:
        return "second"
    if s == "summary":
        return "summary"
    return s


def _pick_duration_col(df: pd.DataFrame) -> str | None:
    for c in DURATION_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _match_gps_for_date(gps: pd.DataFrame, match_date: date) -> pd.DataFrame:
    g = gps.copy()
    g["_DATE_"] = pd.to_datetime(g[COL_DATE], errors="coerce", dayfirst=True).dt.date
    return g[g["_DATE_"] == match_date].copy()


def _fmt_int_thousands(v):
    try:
        if pd.isna(v):
            return "—"
        return f"{float(v):,.0f}".replace(",", " ")
    except Exception:
        return str(v)


# =========================
# Excel load
# =========================
@st.cache_data(show_spinner=False)
def read_excel_from_bytes(xlsx_bytes: bytes) -> dict[str, pd.DataFrame]:
    with io.BytesIO(xlsx_bytes) as bio:
        xl = pd.ExcelFile(bio)
        out = {}
        for s in (MATCHES_SHEET, GPS_SHEET, PLAYERS_SHEET):
            if s in xl.sheet_names:
                out[s] = pd.read_excel(xl, s)
            else:
                out[s] = pd.DataFrame()
        return out


def load_active_players(players_df: pd.DataFrame) -> set[str]:
    pl = players_df.copy()
    if pl.empty:
        return set()

    if COL_PL_FULLNAME not in pl.columns:
        pl[COL_PL_FULLNAME] = (
            pl.get(COL_PL_FIRSTNAME, "").fillna("").astype(str).str.strip()
            + " "
            + pl.get(COL_PL_LASTNAME, "").fillna("").astype(str).str.strip()
        )

    pl[COL_PL_FULLNAME] = pl[COL_PL_FULLNAME].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    if COL_PL_ACTIVITY in pl.columns:
        pl = pl[pl[COL_PL_ACTIVITY].fillna("").astype(str).str.strip().str.lower().eq("actief")]

    return set(pl[COL_PL_FULLNAME].dropna().astype(str).str.replace(r"\s+", " ", regex=True).str.strip())


def load_player_subpos(players_df: pd.DataFrame) -> dict[str, str]:
    pl = players_df.copy()
    if pl.empty:
        return {}

    if COL_PL_FULLNAME not in pl.columns:
        pl[COL_PL_FULLNAME] = (
            pl.get(COL_PL_FIRSTNAME, "").fillna("").astype(str).str.strip()
            + " "
            + pl.get(COL_PL_LASTNAME, "").fillna("").astype(str).str.strip()
        )
    pl[COL_PL_FULLNAME] = pl[COL_PL_FULLNAME].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    subpos_col = next((c for c in SUBPOS_CANDIDATES if c in pl.columns), None)
    if subpos_col is None:
        return {}

    if COL_PL_ACTIVITY in pl.columns:
        pl = pl[pl[COL_PL_ACTIVITY].fillna("").astype(str).str.strip().str.lower().eq("actief")]

    out = {}
    for _, r in pl.iterrows():
        name = str(r[COL_PL_FULLNAME]).strip()
        sp = str(r[subpos_col]).strip().upper()
        if name:
            out[name] = sp
    return out


def load_matches(matches_df: pd.DataFrame) -> pd.DataFrame:
    df = matches_df.copy()
    if df.empty:
        return df

    if COL_MATCH_DATE not in df.columns:
        raise ValueError(f"Sheet '{MATCHES_SHEET}' mist kolom '{COL_MATCH_DATE}'.")

    df[COL_MATCH_DATE] = pd.to_datetime(df[COL_MATCH_DATE], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[COL_MATCH_DATE]).copy()

    def _ha(x):
        s = str(x).strip().lower()
        if s.startswith("h") or s.startswith("home"):
            return "Home"
        if s.startswith("a") or s.startswith("away"):
            return "Away"
        return ""

    if COL_MATCH_HOMEAWAY in df.columns:
        df[COL_MATCH_HOMEAWAY] = df[COL_MATCH_HOMEAWAY].apply(_ha)
    else:
        df[COL_MATCH_HOMEAWAY] = ""

    for c in [COL_MATCH_GF, COL_MATCH_GA]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = [c for c in [COL_MATCH_DATE, COL_MATCH_HOMEAWAY, COL_MATCH_OPP, COL_MATCH_GF, COL_MATCH_GA, COL_MATCH_Match] if c in df.columns]
    df = df.sort_values(COL_MATCH_DATE).reset_index(drop=True)
    return df[keep]


def load_gps(gps_df: pd.DataFrame, active_players: set[str]) -> pd.DataFrame:
    df = gps_df.copy()
    if df.empty:
        return df

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[COL_DATE, COL_PLAYER]).copy()
    df[COL_PLAYER] = df[COL_PLAYER].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    if active_players:
        df = df[df[COL_PLAYER].isin(active_players)].copy()

    if COL_EVENT not in df.columns:
        df[COL_EVENT] = ""

    # numeric cols
    core_cols = [COL_TD, COL_RUN, COL_SPRINT, COL_HS]
    opt_cols = [COL_ACC_TOT, COL_ACC_HI, COL_DEC_TOT, COL_DEC_HI]
    num_cols = core_cols + opt_cols + HR_COLS + ([COL_TRIMP] if COL_TRIMP in df.columns else []) + ([COL_MAX_SPEED] if COL_MAX_SPEED in df.columns else [])
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    dur_col = _pick_duration_col(df)
    if dur_col:
        df[dur_col] = pd.to_numeric(df[dur_col], errors="coerce")

    df["_DURATION_COL_"] = dur_col
    return df


# =========================
# Report building (matplotlib)
# =========================
def match_card_page(fig, ax, row: pd.Series):
    ax.axis("off")

    dt = pd.to_datetime(row[COL_MATCH_DATE]).date()
    opp = str(row.get(COL_MATCH_OPP, "")).strip()
    ha = str(row.get(COL_MATCH_HOMEAWAY, "")).strip().lower()
    gf = pd.to_numeric(row.get(COL_MATCH_GF, np.nan), errors="coerce")
    ga = pd.to_numeric(row.get(COL_MATCH_GA, np.nan), errors="coerce")

    if ha.startswith("home"):
        left_team, right_team = "MVV Maastricht", opp
        thuisuit = "Thuis"
    elif ha.startswith("away"):
        left_team, right_team = opp, "MVV Maastricht"
        thuisuit = "Uit"
    else:
        left_team, right_team = "MVV Maastricht", opp
        thuisuit = ""

    title = f"Match • {dt.strftime('%d-%m-%Y')}" + (f" • {thuisuit}" if thuisuit else "")
    ax.text(0.5, 0.90, title, ha="center", va="center", fontsize=22, weight="bold", transform=ax.transAxes)

    ax.text(0.5, 0.38, f"{left_team} – {right_team}", ha="center", va="center", fontsize=42, weight="bold", transform=ax.transAxes)

    draw_logo(ax, left_team, 0.33, 0.64)
    draw_logo(ax, right_team, 0.67, 0.64)

    subtitle = ""
    if pd.notna(gf) and pd.notna(ga):
        r = "W" if gf > ga else ("L" if gf < ga else "D")
        score_str = f"{int(ga)}-{int(gf)}" if ha.startswith("away") else f"{int(gf)}-{int(ga)}"
        subtitle = f"Resultaat: {score_str} ({r})"

    if subtitle:
        ax.text(0.5, 0.16, subtitle, ha="center", va="center", fontsize=18, transform=ax.transAxes)


def _sum_first_second(gps_match: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = gps_match[gps_match["_EVENT_NORM_"].isin(["first", "second"])].copy()
    have = [c for c in cols if c in df.columns]
    if not have:
        return pd.DataFrame(columns=[COL_PLAYER] + cols)
    out = df.groupby(COL_PLAYER, as_index=False)[have].sum()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
    return out


def match_dashboard_page(fig, gps_match: pd.DataFrame, match_label: str):
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(hspace=0.38)
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # TD
    df_td = _sum_first_second(gps_match, [COL_TD]).sort_values(COL_TD, ascending=False)
    if df_td.empty:
        ax1.text(0.5, 0.5, "Geen data", ha="center")
        ax1.axis("off")
    else:
        players = df_td[COL_PLAYER].to_numpy()
        vals = df_td[COL_TD].to_numpy()
        x = np.arange(len(players))
        ax1.set_title("Total Distance", fontsize=12)
        ax1.bar(x, vals, color=MVV_RED)
        ax1.set_xticks(x)
        ax1.set_xticklabels(players, rotation=90, ha="right", fontsize=7)
        ax1.set_ylabel("m")

    # Sprint & HS
    df_sp = _sum_first_second(gps_match, [COL_SPRINT, COL_HS])
    if df_sp.empty:
        ax2.text(0.5, 0.5, "Geen sprint-kolommen", ha="center")
        ax2.axis("off")
    else:
        df_sp = df_sp.sort_values(COL_SPRINT, ascending=False)
        players = df_sp[COL_PLAYER].to_numpy()
        x = np.arange(len(players))
        width = 0.4
        s_vals = df_sp[COL_SPRINT].to_numpy()
        hs_vals = df_sp[COL_HS].to_numpy()
        ax2.set_title("Sprint & High Sprint", fontsize=12)
        ax2.bar(x - width / 2, s_vals, width=width, label="Sprint")
        ax2.bar(x + width / 2, hs_vals, width=width, label="High Sprint")
        ax2.set_xticks(x)
        ax2.set_xticklabels(players, rotation=90, ha="right", fontsize=7)
        ax2.set_ylabel("m")
        ax2.legend(loc="upper right", fontsize=8, frameon=False)

    # Acc/Dec (als aanwezig)
    have_ad = all(c in gps_match.columns for c in [COL_ACC_TOT, COL_ACC_HI, COL_DEC_TOT, COL_DEC_HI])
    if not have_ad:
        ax3.text(0.5, 0.5, "Geen Acc/Dec kolommen", ha="center")
        ax3.axis("off")
    else:
        df_ad = _sum_first_second(gps_match, [COL_ACC_TOT, COL_ACC_HI, COL_DEC_TOT, COL_DEC_HI]).sort_values(COL_ACC_TOT, ascending=False)
        players = df_ad[COL_PLAYER].to_numpy()
        x = np.arange(len(players))
        w = 0.2
        ax3.set_title("Acc / Dec", fontsize=12)
        ax3.bar(x - 1.5 * w, df_ad[COL_ACC_TOT].to_numpy(), width=w, label="Tot Acc")
        ax3.bar(x - 0.5 * w, df_ad[COL_ACC_HI].to_numpy(), width=w, label="High Acc")
        ax3.bar(x + 0.5 * w, df_ad[COL_DEC_TOT].to_numpy(), width=w, label="Tot Dec")
        ax3.bar(x + 1.5 * w, df_ad[COL_DEC_HI].to_numpy(), width=w, label="High Dec")
        ax3.set_xticks(x)
        ax3.set_xticklabels(players, rotation=90, ha="right", fontsize=7)
        ax3.legend(loc="upper right", fontsize=7, frameon=False)

    # HR zones (als aanwezig)
    have_hr = any(c in gps_match.columns for c in HR_COLS)
    if not have_hr:
        ax4.text(0.5, 0.5, "Geen HR-zone kolommen", ha="center")
        ax4.axis("off")
    else:
        zone_cols = [c for c in HR_COLS if c in gps_match.columns]
        df_hr = _sum_first_second(gps_match, zone_cols)
        if df_hr.empty:
            ax4.text(0.5, 0.5, "Geen HR-data", ha="center")
            ax4.axis("off")
        else:
            df_hr["_sum"] = df_hr[zone_cols].sum(axis=1)
            df_hr = df_hr.sort_values("_sum", ascending=False).drop(columns="_sum")
            players = df_hr[COL_PLAYER].to_numpy()
            x = np.arange(len(players))
            w = 0.14
            ax4.set_title("Time in HR zone", fontsize=12)
            for i, z in enumerate(zone_cols):
                ax4.bar(x + (i - 2) * w, df_hr[z].to_numpy(), width=w, label=z)
            ax4.set_xticks(x)
            ax4.set_xticklabels(players, rotation=90, ha="right", fontsize=7)
            ax4.legend(loc="upper right", fontsize=7, frameon=False)

    fig.suptitle(f"Match Dashboard • {match_label}", fontsize=16, fontweight="bold", y=0.98)


def build_match_pdf_bytes(gps: pd.DataFrame, match_row: pd.Series, subpos_map: dict[str, str]) -> bytes:
    mdate = pd.to_datetime(match_row[COL_MATCH_DATE]).date()
    gps_match = _match_gps_for_date(gps, mdate).copy()
    gps_match["_EVENT_NORM_"] = gps_match[COL_EVENT].map(_normalize_event)
    gps_match[COL_SUBPOS_OUT] = gps_match[COL_PLAYER].map(subpos_map).fillna("")

    match_label = mdate.strftime("%d-%m-%Y")

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Card
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        match_card_page(fig, ax, match_row)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Dashboard
        fig = plt.figure(figsize=(11.69, 8.27))
        match_dashboard_page(fig, gps_match, match_label)
        fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

    buf.seek(0)
    return buf.read()


# =========================
# UI
# =========================
st.title("Match Reports")

with st.expander("Database (Excel)"):
    st.caption("Gebruik bij voorkeur st.secrets['DATABASE_XLSX'] of upload handmatig.")
    up = st.file_uploader("Upload Database.xlsx", type=["xlsx"])
    use_demo = st.checkbox("Gebruik local repo file (als aanwezig)", value=False)

xlsx_bytes: bytes | None = None

# 1) secrets path (optioneel)
# - je kan op Streamlit Cloud een file niet makkelijk als bytes in secrets stoppen,
#   dus dit blok is bewust beperkt. Upload is de standaard.
#
# 2) upload
if up is not None:
    xlsx_bytes = up.getvalue()

# 3) local (handig lokaal / in repo)
if xlsx_bytes is None and use_demo:
    cand = REPO_ROOT / "Database" / "Database.xlsx"
    if cand.exists():
        xlsx_bytes = cand.read_bytes()
    else:
        st.warning("Local Database.xlsx niet gevonden in /Database/Database.xlsx")

if xlsx_bytes is None:
    st.stop()

data = read_excel_from_bytes(xlsx_bytes)
matches_df = load_matches(data.get(MATCHES_SHEET, pd.DataFrame()))
players_df = data.get(PLAYERS_SHEET, pd.DataFrame())
gps_df_raw = data.get(GPS_SHEET, pd.DataFrame())

active_players = load_active_players(players_df)
subpos_map = load_player_subpos(players_df)
gps_df = load_gps(gps_df_raw, active_players)

if matches_df.empty:
    st.error("Geen wedstrijden gevonden in sheet 'Matches'.")
    st.stop()

# Select match
matches_df = matches_df.sort_values(COL_MATCH_DATE, ascending=False).reset_index(drop=True)

def _label_row(r: pd.Series) -> str:
    dt = pd.to_datetime(r[COL_MATCH_DATE]).date()
    ha = str(r.get(COL_MATCH_HOMEAWAY, "")).strip()
    ha = "H" if ha.lower().startswith("home") else ("A" if ha.lower().startswith("away") else "")
    opp = str(r.get(COL_MATCH_OPP, "")).strip()
    return f"{dt.strftime('%d-%m-%Y')}  ({ha})  vs {opp}"

labels = [_label_row(r) for _, r in matches_df.iterrows()]
pick = st.selectbox("Kies wedstrijd", options=list(range(len(labels))), format_func=lambda i: labels[i])
match_row = matches_df.iloc[int(pick)]

mdate = pd.to_datetime(match_row[COL_MATCH_DATE]).date()
match_label = mdate.strftime("%d-%m-%Y")

c1, c2 = st.columns([1, 1], gap="large")

with c1:
    st.subheader("Preview — Match card")
    fig, ax = plt.subplots(figsize=(11.69, 4.2))
    match_card_page(fig, ax, match_row)
    st.pyplot(fig, clear_figure=True)

with c2:
    st.subheader("Preview — Dashboard")
    gps_match = _match_gps_for_date(gps_df, mdate).copy()
    gps_match["_EVENT_NORM_"] = gps_match[COL_EVENT].map(_normalize_event)
    fig = plt.figure(figsize=(11.69, 4.2))
    match_dashboard_page(fig, gps_match, match_label)
    st.pyplot(fig, clear_figure=True)

st.divider()

col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("PDF")
    st.caption("Genereert (voor nu) Card + Dashboard. Tabellen/posities kunnen daarna toegevoegd worden.")
    if st.button("Genereer Match Report PDF", use_container_width=True):
        pdf_bytes = build_match_pdf_bytes(gps_df, match_row, subpos_map)
        fname = f"Match Report_{_sanitize_filename(_label_row(match_row).replace('  ', ' '))}.pdf"
        st.session_state["match_pdf_bytes"] = pdf_bytes
        st.session_state["match_pdf_name"] = fname
        st.success("PDF gegenereerd.")

with col_b:
    st.subheader("Download")
    pdf_bytes = st.session_state.get("match_pdf_bytes")
    pdf_name = st.session_state.get("match_pdf_name", f"Match Report_{match_label}.pdf")
    st.download_button(
        "Download PDF",
        data=pdf_bytes if pdf_bytes else b"",
        file_name=pdf_name,
        mime="application/pdf",
        use_container_width=True,
        disabled=pdf_bytes is None,
    )
