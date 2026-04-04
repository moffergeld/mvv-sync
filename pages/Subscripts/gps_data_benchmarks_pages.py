# pages/Subscripts/gps_data_benchmarks_pages.py
# ==========================================================
# Benchmarks tab (Gref)
# - Bron: public.v_gps_match_events
# - Match = First Half + Second Half (gesommeerd per speler per datum)
# - Filter: alleen matches met duration >= min_minutes (default 75)
# - Top-K (default 5) uit die selectie
# - Tabel 1: Gref (optioneel genormaliseerd naar 90 min)
# - Tabel 2: dezelfde selectie maar waarden per minuut
#
# Metrics (zoals gevraagd):
#   duration
#   total_distance
#   sprint
#   high_sprint
#   playerload2d
#   total_accelerations
#   high_accelerations
#   total_decelerations
#   high_decelerations
# ==========================================================

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st



TEXT = "#F5F7FB"
TEXT_MUTED = "rgba(245,247,251,0.68)"

def _section_label(title: str, subtitle: str | None = None) -> None:
    html = f'<div style="margin:0 0 0.8rem 0;"><div style="font-size:0.82rem;letter-spacing:0.16em;text-transform:uppercase;color:{TEXT_MUTED};font-weight:700;">{title}</div>'
    if subtitle:
        html += f'<div style="font-size:0.95rem;color:{TEXT_MUTED};margin-top:0.32rem;">{subtitle}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def _metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div style="
            border:1px solid rgba(255,255,255,0.08);
            border-radius:18px;
            padding:0.8rem 1rem;
            background:rgba(255,255,255,0.035);
            min-height:92px;">
            <div style="font-size:0.78rem;letter-spacing:0.14em;text-transform:uppercase;color:{TEXT_MUTED};font-weight:700;">{label}</div>
            <div style="font-size:1.45rem;color:{TEXT};font-weight:800;margin-top:0.35rem;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------
# REST helpers (standalone)
# -------------------------
def _rest_headers(supabase_anon_key: str, access_token: str) -> dict:
    return {
        "apikey": supabase_anon_key,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Prefer": "count=exact",
    }


def _rest_get_paged(
    *,
    supabase_url: str,
    supabase_anon_key: str,
    access_token: str,
    table: str,
    base_query: str,
    page_size: int = 5000,
    timeout: int = 120,
) -> pd.DataFrame:
    url = f"{supabase_url}/rest/v1/{table}?{base_query}"
    headers = _rest_headers(supabase_anon_key, access_token) | {"Range-Unit": "items"}

    all_rows: list[dict] = []
    start = 0
    while True:
        end = start + page_size - 1
        h = headers | {"Range": f"{start}-{end}"}

        r = requests.get(url, headers=h, timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"GET {table} failed ({r.status_code}): {r.text}")

        batch = r.json()
        if not batch:
            break

        all_rows.extend(batch)

        if len(batch) < page_size:
            break

        start += page_size

    return pd.DataFrame(all_rows)


# -------------------------
# Fetch from v_gps_match_events
# -------------------------
GREF_SELECT_COLS = [
    "gps_id",
    "datum",
    "player_id",
    "player_name",
    "type",
    "event",
    "duration",
    "total_distance",
    "sprint",
    "high_sprint",
    "playerload2d",
    "total_accelerations",
    "high_accelerations",
    "total_decelerations",
    "high_decelerations",
]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_match_events_all_cached(
    *,
    supabase_url: str,
    supabase_anon_key: str,
    access_token: str,
    user_id: str,
) -> pd.DataFrame:
    # user_id in signature => cache per user
    select = ",".join(GREF_SELECT_COLS)
    base_query = f"select={select}&order=datum.asc,gps_id.asc"
    return _rest_get_paged(
        supabase_url=supabase_url,
        supabase_anon_key=supabase_anon_key,
        access_token=access_token,
        table="v_gps_match_events",
        base_query=base_query,
        page_size=5000,
        timeout=120,
    )


# -------------------------
# Core: build match totals (First+Second Half)
# -------------------------
def _prepare_match_totals(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per player per match-date with summed First+Second Half.
    Expected columns exist from GREF_SELECT_COLS.
    """
    if df_events is None or df_events.empty:
        return pd.DataFrame()

    df = df_events.copy()

    df["datum"] = pd.to_datetime(df["datum"], errors="coerce").dt.date
    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    df["event"] = df["event"].astype(str).str.strip()

    # match + halves only
    df = df[df["type"].eq("match")].copy()
    df = df[df["event"].isin(["First Half", "Second Half"])].copy()
    if df.empty:
        return pd.DataFrame()

    num_cols = [
        "duration",
        "total_distance",
        "sprint",
        "high_sprint",
        "playerload2d",
        "total_accelerations",
        "high_accelerations",
        "total_decelerations",
        "high_decelerations",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # sum halves -> match totals
    match_totals = df.groupby(["player_name", "datum"], as_index=False)[num_cols].sum()
    return match_totals


# -------------------------
# Table 1: Gref (optional normalized to 90)
# -------------------------
def compute_gref(
    df_events: pd.DataFrame,
    *,
    min_minutes: float = 75.0,
    top_k: int = 5,
    normalize_to_90: bool = True,
) -> pd.DataFrame:
    """
    - Match totals: First+Second Half summed per player per date
    - Filter first: duration >= min_minutes
    - Optional: normalize each metric to 90 minutes based on that match duration
    - Gref per player per metric: mean(top_k values)
    """
    match_totals = _prepare_match_totals(df_events)
    if match_totals.empty:
        return pd.DataFrame()

    # 1) filter first (critical)
    match_totals = match_totals[match_totals["duration"] >= float(min_minutes)].copy()
    if match_totals.empty:
        return pd.DataFrame()

    metrics = [
        "total_distance",
        "sprint",
        "high_sprint",
        "playerload2d",
        "total_accelerations",
        "high_accelerations",
        "total_decelerations",
        "high_decelerations",
    ]

    # 2) optional normalize to 90
    if normalize_to_90:
        dur = match_totals["duration"].replace(0, pd.NA).astype(float)
        scale = 90.0 / dur
        for m in metrics:
            match_totals[m] = (match_totals[m].astype(float) * scale).fillna(0.0)

    # 3) top-k mean per player
    out_rows: list[dict] = []
    for player, g in match_totals.groupby("player_name"):
        row = {"Speler": player}

        for m in metrics:
            vals = g[m].nlargest(top_k)
            row[m] = float(vals.mean()) if len(vals) else 0.0

        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out

    out = out.rename(
        columns={
            "total_distance": "Total Distance",
            "sprint": "Sprint",
            "high_sprint": "High Sprint",
            "playerload2d": "playerload2D",
            "total_accelerations": "Total Accelerations",
            "high_accelerations": "High Accelerations",
            "total_decelerations": "Total Decelerations",
            "high_decelerations": "High Decelerations",
        }
    )

    col_order = [
        "Speler",
        "Total Distance",
        "Sprint",
        "High Sprint",
        "playerload2D",
        "Total Accelerations",
        "High Accelerations",
        "Total Decelerations",
        "High Decelerations",
    ]
    out = out[[c for c in col_order if c in out.columns]].sort_values("Speler").reset_index(drop=True)

    # rounding for display
    for c in out.columns:
        if c != "Speler":
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).round(0)

    return out


# -------------------------
# Table 2: same selection but per minute
# -------------------------
def compute_gref_per_min(
    df_events: pd.DataFrame,
    *,
    min_minutes: float = 75.0,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    - Match totals: First+Second Half summed
    - Filter first: duration >= min_minutes
    - Convert each metric to per minute: metric / duration
    - Gref/min per player per metric: mean(top_k values)
    """
    match_totals = _prepare_match_totals(df_events)
    if match_totals.empty:
        return pd.DataFrame()

    match_totals = match_totals[match_totals["duration"] >= float(min_minutes)].copy()
    if match_totals.empty:
        return pd.DataFrame()

    metrics = [
        "total_distance",
        "sprint",
        "high_sprint",
        "playerload2d",
        "total_accelerations",
        "high_accelerations",
        "total_decelerations",
        "high_decelerations",
    ]

    dur = match_totals["duration"].replace(0, pd.NA).astype(float)
    for m in metrics:
        match_totals[m] = (match_totals[m].astype(float) / dur).fillna(0.0)

    out_rows: list[dict] = []
    for player, g in match_totals.groupby("player_name"):
        row = {"Speler": player}
        for m in metrics:
            vals = g[m].nlargest(top_k)
            row[m] = float(vals.mean()) if len(vals) else 0.0
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out

    out = out.rename(
        columns={
            "total_distance": "Total Distance /min",
            "sprint": "Sprint /min",
            "high_sprint": "High Sprint /min",
            "playerload2d": "playerload2D /min",
            "total_accelerations": "Total Acc /min",
            "high_accelerations": "High Acc /min",
            "total_decelerations": "Total Dec /min",
            "high_decelerations": "High Dec /min",
        }
    )

    col_order = [
        "Speler",
        "Total Distance /min",
        "Sprint /min",
        "High Sprint /min",
        "playerload2D /min",
        "Total Acc /min",
        "High Acc /min",
        "Total Dec /min",
        "High Dec /min",
    ]
    out = out[[c for c in col_order if c in out.columns]].sort_values("Speler").reset_index(drop=True)

    # rounding for display (per-min needs decimals)
    for c in out.columns:
        if c == "Speler":
            continue
        if "Acc" in c or "Dec" in c:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).round(3)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).round(2)

    return out


# -------------------------
# Public render API
# -------------------------
def benchmarks_pages_main(
    *,
    supabase_url: str,
    supabase_anon_key: str,
    access_token: str,
    user_id: str,
):
    _section_label("Benchmarks", "Gref-tabellen op basis van First Half + Second Half match events.")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        min_minutes = st.number_input(
            "Minuten (min. voor Gref)",
            min_value=0.0,
            max_value=120.0,
            value=75.0,
            step=1.0,
            key="gref_min_minutes",
        )
    with c2:
        top_k = st.number_input(
            "Top-K matches",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key="gref_top_k",
        )
    with c3:
        normalize_to_90 = st.toggle(
            "Normaliseer naar 90 min",
            value=True,
            key="gref_norm_90",
        )

    try:
        df_events = fetch_match_events_all_cached(
            supabase_url=supabase_url,
            supabase_anon_key=supabase_anon_key,
            access_token=access_token,
            user_id=user_id,
        )
    except Exception as e:
        st.error(f"Kon v_gps_match_events niet laden: {e}")
        return

    # Table 1: Gref
    df_gref = compute_gref(
        df_events,
        min_minutes=float(min_minutes),
        top_k=int(top_k),
        normalize_to_90=bool(normalize_to_90),
    )
    if df_gref.empty:
        st.info("Geen bruikbare matchdata (First+Second Half) gevonden voor deze filters.")
        return

    kc1, kc2, kc3 = st.columns(3)
    with kc1:
        _metric_card("Spelers", str(len(df_gref)))
    with kc2:
        _metric_card("Minutenfilter", f"{float(min_minutes):.0f}+")
    with kc3:
        _metric_card("Top-K matches", str(int(top_k)))

    st.markdown("#### Gref")
    st.dataframe(df_gref, use_container_width=True, height=420, hide_index=True)

    # Table 2: per minute
    df_gref_min = compute_gref_per_min(
        df_events,
        min_minutes=float(min_minutes),
        top_k=int(top_k),
    )
    if df_gref_min.empty:
        st.info("Geen data voor per-min tabel met deze filters.")
        return

    st.markdown("#### Gref per minuut")
    st.dataframe(df_gref_min, use_container_width=True, height=420, hide_index=True)
