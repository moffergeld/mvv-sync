# pages/Subscripts/gps_data_benchmarks_pages.py
# ==========================================================
# Benchmarks tab (Gref)
# - Bron: public.v_gps_match_events
# - Match = First Half + Second Half (gesommeerd per speler per datum)
# - Minimaal hele helft gespeeld: duration >= 40 min (na sommatie)
# - Gref = mean(top-5 matchwaarden) per speler per metric
# ==========================================================

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st


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
# Fetch + Gref compute
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


def compute_gref(
    df_events: pd.DataFrame,
    *,
    min_minutes: float = 40.0,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    - Filter: type == 'Match' (case-insensitive), event in {'First Half','Second Half'}
    - Aggregate per speler per datum: sum(First+Second) over duration + alle metrics
    - Filter: duration >= min_minutes (na aggregatie)
    - Gref per speler: mean(top_k matchwaarden) per metric
    """
    if df_events is None or df_events.empty:
        return pd.DataFrame()

    df = df_events.copy()

    # normalize strings/dates
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce").dt.date
    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    df["event"] = df["event"].astype(str).str.strip()

    # match + halves
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

    # per match totalen (First+Second Half)
    match_totals = df.groupby(["player_name", "datum"], as_index=False)[num_cols].sum()

    # minimaal hele helft (na sommatie)
    match_totals = match_totals[match_totals["duration"] >= float(min_minutes)].copy()
    if match_totals.empty:
        return pd.DataFrame()

    # Gref per speler
    metrics = [c for c in num_cols if c != "duration"]
    out_rows: list[dict] = []

    for player, g in match_totals.groupby("player_name"):
        row = {"Speler": player}

        # duration benchmark (ook gevraagd)
        row["Duration"] = float(g["duration"].nlargest(top_k).mean()) if len(g) else 0.0

        for m in metrics:
            vals = g[m].nlargest(top_k)
            row[m] = float(vals.mean()) if len(vals) else 0.0

        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out

    # rename naar dashboard labels
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
        "Duration",
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

    # afronden voor tabel
    for c in out.columns:
        if c != "Speler":
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).round(0)

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
    st.subheader("Gref")

    c1, c2 = st.columns([1, 1])
    with c1:
        min_minutes = st.number_input(
            "Minuten (min. hele helft)",
            min_value=0.0,
            max_value=90.0,
            value=40.0,
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

    df_gref = compute_gref(df_events, min_minutes=float(min_minutes), top_k=int(top_k))
    if df_gref.empty:
        st.info("Geen bruikbare matchdata (First+Second Half) gevonden voor deze filters.")
        return

    st.dataframe(df_gref, use_container_width=True, height=520)
