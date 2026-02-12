# pages/02_Match_Reports.py
# ============================================================
# Streamlit — Match Reports (Supabase)
# - Match selector gebruikt public.matches (fixture/home_away/opponent/result/score)
# - GPS bron: public.v_gps_match_events (First Half / Second Half)
# - Plotly grafieken + st.dataframe (zelfde stijl als Player Pages)
# - Team logos: Assets/Afbeeldingen/Team_Logos (repo)
# ============================================================

from __future__ import annotations

import io
import os
from pathlib import Path
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image, ImageChops, ImageOps

from roles import get_sb, require_auth  # jouw bestaande helpers

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="Match Reports", layout="wide")

MVV_RED = "#FF0033"

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
TEAM_LOGOS_DIR = REPO_ROOT / "Assets" / "Afbeeldingen" / "Team_Logos"
LOGO_EXTS = [".png", ".jpg", ".jpeg", ".webp"]

MATCHES_TABLE = "matches"
GPS_MATCH_VIEW = "v_gps_match_events"

MATCH_EVENTS = ["First Half", "Second Half"]

GPS_COLS = [
    "match_id",
    "datum",
    "player_id",
    "player_name",
    "type",
    "event",  # First Half / Second Half
    "duration",
    "total_distance",
    "running",
    "sprint",
    "high_sprint",
    "max_speed",
]

# -------------------------
# Logo helpers
# -------------------------
def _sanitize(s: str) -> str:
    return "".join(ch for ch in str(s).strip() if ch.isalnum()).lower()


def _norm_name(s: str) -> str:
    t = str(s).strip().lower()
    for pre in ("sc ", "fc ", "vv ", "sv ", "rksv ", "rkvv ", "sbv ", "ssv "):
        if t.startswith(pre):
            t = t[len(pre) :]
            break
    return _sanitize(t)


def find_logo_path(club_name: str) -> Optional[str]:
    if not club_name:
        return None

    if club_name.strip().lower() in {"mvv", "mvv maastricht"}:
        club_name = "MVV Maastricht"

    if not TEAM_LOGOS_DIR.exists():
        return None

    base = club_name.strip()

    # 1) exact match
    for ext in LOGO_EXTS:
        p = TEAM_LOGOS_DIR / f"{base}{ext}"
        if p.is_file():
            return str(p)

    # 2) normalized match
    want = _norm_name(base)
    try:
        for fn in os.listdir(TEAM_LOGOS_DIR):
            stem, ext = os.path.splitext(fn)
            if ext.lower() in LOGO_EXTS and _norm_name(stem) == want:
                return str(TEAM_LOGOS_DIR / fn)
    except Exception:
        pass

    return None


def _autocrop_logo(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGBA")

    # crop alpha border
    if "A" in img.getbands():
        alpha = img.split()[-1]
        bb = ImageChops.difference(alpha, Image.new("L", alpha.size, 0)).getbbox()
        if bb:
            img = img.crop(bb)

    # crop near-white border
    rgb = img.convert("RGB")
    bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, bg)
    diff = ImageOps.autocontrast(diff)
    bb = diff.getbbox()
    if bb:
        img = img.crop(bb)

    return img


@st.cache_data(show_spinner=False)
def load_logo_bytes(club_name: str, target_h: int = 140) -> Optional[bytes]:
    p = find_logo_path(club_name)
    if not p:
        return None
    try:
        img = Image.open(p)
        img = _autocrop_logo(img)
        w, h = img.size
        if h > 0:
            scale = target_h / float(h)
            img = img.resize((max(1, int(w * scale)), target_h), Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return None


# -------------------------
# Supabase helpers
# -------------------------
def _df_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


@st.cache_data(show_spinner=False)
def fetch_matches(sb, limit: int = 400) -> pd.DataFrame:
    """
    public.matches:
      match_id, match_date, fixture, home_away, opponent, season, result, goals_for, goals_against
    """
    try:
        rows = (
            sb.table(MATCHES_TABLE)
            .select("match_id,match_date,fixture,home_away,opponent,season,result,goals_for,goals_against,match_type")
            .order("match_date", desc=True)
            .limit(limit)
            .execute()
            .data
            or []
        )
        df = _df_from_rows(rows)
        if df.empty:
            return df
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.date
        return df.dropna(subset=["match_id", "match_date"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def fetch_gps_match_events(sb, match_id: int, limit: int = 10000) -> pd.DataFrame:
    """
    v_gps_match_events:
      alle spelers + events (First Half/Second Half)
    """
    try:
        rows = (
            sb.table(GPS_MATCH_VIEW)
            .select(",".join(GPS_COLS))
            .eq("match_id", match_id)
            .in_("event", MATCH_EVENTS)
            .order("player_name")
            .order("event")
            .limit(limit)
            .execute()
            .data
            or []
        )
        df = _df_from_rows(rows)
        if df.empty:
            return df
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce").dt.date
        for c in ["duration", "total_distance", "running", "sprint", "high_sprint", "max_speed"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df["event"] = df["event"].astype(str)
        df["player_name"] = df["player_name"].astype(str)
        return df
    except Exception:
        return pd.DataFrame()


# -------------------------
# Plotly blocks
# -------------------------
def plot_team_bar(df: pd.DataFrame, metric: str, title: str):
    if df.empty or metric not in df.columns:
        st.info("Geen data.")
        return

    d = df.groupby("player_name", as_index=False)[metric].sum().sort_values(metric, ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["player_name"], y=d[metric], marker=dict(color=MVV_RED)))
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        title=title,
        xaxis=dict(tickangle=-55),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_stack_sprint(df: pd.DataFrame):
    need = ["sprint", "high_sprint"]
    if df.empty or any(c not in df.columns for c in need):
        st.info("Geen sprint data.")
        return

    d = df.groupby("player_name", as_index=False)[need].sum()
    d["sum"] = d["sprint"] + d["high_sprint"]
    d = d.sort_values("sum", ascending=False).drop(columns="sum")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["player_name"], y=d["sprint"], name="19.8–25.1 km/h", marker=dict(color="#8b0000")))
    fig.add_trace(go.Bar(x=d["player_name"], y=d["high_sprint"], name=">25.1 km/h", marker=dict(color=MVV_RED)))
    fig.update_layout(
        barmode="stack",
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Sprint zones (First+Second)",
        xaxis=dict(tickangle=-55),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def match_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    cols = ["datum", "player_name", "event", "total_distance", "running", "sprint", "high_sprint", "max_speed"]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()

    ev_order = {"First Half": 0, "Second Half": 1}
    out["_ev"] = out["event"].map(lambda x: ev_order.get(str(x), 99))
    out = out.sort_values(["player_name", "_ev"]).drop(columns=["_ev"])

    return out.rename(
        columns={
            "datum": "Date",
            "player_name": "Player",
            "event": "Event",
            "total_distance": "Total Distance (m)",
            "running": "14.4–19.7 km/h",
            "sprint": "19.8–25.1 km/h",
            "high_sprint": ">25.1 km/h",
            "max_speed": "Max Speed (km/u)",
        }
    )


# -------------------------
# UI
# -------------------------
def _match_label(row: Dict[str, Any]) -> str:
    md = row.get("match_date")
    dt = md.strftime("%d-%m-%Y") if isinstance(md, date) else str(md)

    fixture = (row.get("fixture") or "").strip()
    opp = (row.get("opponent") or "").strip()
    ha = (row.get("home_away") or "").strip()
    season = (row.get("season") or "").strip()

    gf = row.get("goals_for")
    ga = row.get("goals_against")
    score = ""
    if gf is not None and ga is not None and not (pd.isna(gf) or pd.isna(ga)):
        score = f"{int(gf)}-{int(ga)}"

    parts = [dt]
    if season:
        parts.append(season)
    if fixture:
        parts.append(fixture)
    elif opp:
        parts.append(f"{'vs' if ha.lower().startswith('h') else '@'} {opp}")
    if score:
        parts.append(score)

    return " — ".join(parts)


def main():
    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar.")
        st.stop()

    st.title("Match Reports")

    matches_df = fetch_matches(sb, limit=500)
    if matches_df.empty:
        st.error("public.matches is leeg of niet leesbaar.")
        st.stop()

    match_rows = matches_df.to_dict("records")
    sel = st.selectbox("Kies match", options=match_rows, format_func=_match_label, key="mr_match_pick")

    match_id = int(sel["match_id"])
    match_date = sel["match_date"]
    opponent = (sel.get("opponent") or "").strip()
    fixture = (sel.get("fixture") or "").strip()
    ha = (sel.get("home_away") or "").strip()

    gf = sel.get("goals_for")
    ga = sel.get("goals_against")
    score_txt = "—"
    try:
        if gf is not None and ga is not None and not (pd.isna(gf) or pd.isna(ga)):
            score_txt = f"{int(gf)} - {int(ga)}"
    except Exception:
        pass

    # Header: logos + score
    left_team = "MVV Maastricht"
    right_team = opponent if opponent else "Opponent"

    c1, c2, c3 = st.columns([1, 2, 1], vertical_alignment="center")
    with c1:
        lb = load_logo_bytes(left_team)
        if lb:
            st.image(lb)
        st.markdown(f"**{left_team}**")
    with c2:
        top = fixture if fixture else f"{'Home' if ha.lower().startswith('h') else 'Away'} vs {right_team}"
        st.markdown(f"### {top}")
        st.markdown(f"### {match_date.strftime('%d-%m-%Y')}")
        st.markdown(f"## {score_txt}")
    with c3:
        rb = load_logo_bytes(right_team)
        if rb:
            st.image(rb)
        st.markdown(f"**{right_team}**")

    st.divider()

    gps = fetch_gps_match_events(sb, match_id)
    if gps.empty:
        st.info("Geen data gevonden in v_gps_match_events voor deze match_id.")
        st.stop()

    tab_overview, tab_tables = st.tabs(["Overview", "Tables"])

    with tab_overview:
        team_df = gps.copy()

        a, b = st.columns(2, gap="large")
        with a:
            st.subheader("Total Distance (First+Second)")
            plot_team_bar(team_df, "total_distance", "Total Distance")
        with b:
            st.subheader("Sprint zones (First+Second)")
            plot_stack_sprint(team_df)

        c, d = st.columns(2, gap="large")
        with c:
            st.subheader("Running 14.4–19.7 (First+Second)")
            plot_team_bar(team_df, "running", "14.4–19.7 km/h")
        with d:
            st.subheader("Max Speed (First+Second)")
            plot_team_bar(team_df, "max_speed", "Max Speed (km/u)")

    with tab_tables:
        st.subheader("Per speler x event (First Half / Second Half)")
        st.dataframe(match_table(gps), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
