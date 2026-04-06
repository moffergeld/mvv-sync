# pages/Subscripts/gps_data_benchmarks_pages.py
# ==========================================================
# Benchmarks tab (Gref)
#
# Aangepast:
# - filters naar sidebar
# - extra berekening: per minuut * gekozen minuten
# - extra tabel "Omgerekend naar X minuten"
# ==========================================================

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st


# -------------------------
# REST helpers
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
# Core
# -------------------------
def _prepare_match_totals(df_events: pd.DataFrame) -> pd.DataFrame:
    if df_events is None or df_events.empty:
        return pd.DataFrame()

    df = df_events.copy()

    df["datum"] = pd.to_datetime(df["datum"], errors="coerce").dt.date
    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    df["event"] = df["event"].astype(str).str.strip()

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

    match_totals = df.groupby(["player_name", "datum"], as_index=False)[num_cols].sum()
    return match_totals


def compute_gref(
    df_events: pd.DataFrame,
    *,
    min_minutes: float = 75.0,
    top_k: int = 5,
    normalize_to_90: bool = True,
) -> pd.DataFrame:
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

    if normalize_to_90:
        dur = match_totals["duration"].replace(0, pd.NA).astype(float)
        scale = 90.0 / dur
        for m in metrics:
            match_totals[m] = (match_totals[m].astype(float) * scale).fillna(0.0)

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

    for c in out.columns:
        if c != "Speler":
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).round(0)

    return out


def compute_gref_per_min(
    df_events: pd.DataFrame,
    *,
    min_minutes: float = 75.0,
    top_k: int = 5,
) -> pd.DataFrame:
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

    for c in out.columns:
        if c == "Speler":
            continue
        if "Acc" in c or "Dec" in c:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).round(3)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).round(2)

    return out


def compute_gref_for_minutes(df_gref_per_min: pd.DataFrame, minutes: float) -> pd.DataFrame:
    if df_gref_per_min is None or df_gref_per_min.empty:
        return pd.DataFrame()

    out = df_gref_per_min.copy()
    rename_map = {
        "Total Distance /min": "Total Distance",
        "Sprint /min": "Sprint",
        "High Sprint /min": "High Sprint",
        "playerload2D /min": "playerload2D",
        "Total Acc /min": "Total Accelerations",
        "High Acc /min": "High Accelerations",
        "Total Dec /min": "Total Decelerations",
        "High Dec /min": "High Decelerations",
    }

    for src_col, dst_col in rename_map.items():
        if src_col in out.columns:
            out[dst_col] = pd.to_numeric(out[src_col], errors="coerce").fillna(0.0) * float(minutes)

    keep_cols = ["Speler"] + [c for c in rename_map.values() if c in out.columns]
    out = out[keep_cols].copy()

    for c in out.columns:
        if c != "Speler":
            if "Acceleration" in c or "Deceleration" in c:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).round(2)
            else:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).round(0)

    return out.sort_values("Speler").reset_index(drop=True)


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
    st.subheader("Gref (per 90min)")

    with st.sidebar:
        with st.expander("Benchmarks filters", expanded=False):
            min_minutes = st.number_input(
                "Minuten (min. voor Gref)",
                min_value=0.0,
                max_value=120.0,
                value=75.0,
                step=1.0,
                key="gref_min_minutes_sidebar",
            )
            top_k = st.number_input(
                "Top-K matches",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                key="gref_top_k_sidebar",
            )
            normalize_to_90 = st.toggle(
                "Normaliseer naar 90 min",
                value=True,
                key="gref_norm_90_sidebar",
            )
            calc_minutes = st.number_input(
                "Bereken voor minuten",
                min_value=1.0,
                max_value=120.0,
                value=33.0,
                step=1.0,
                key="gref_calc_minutes_sidebar",
                help="Vermenigvuldigt de per-minuut waardes met dit aantal minuten.",
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

    df_gref = compute_gref(
        df_events,
        min_minutes=float(min_minutes),
        top_k=int(top_k),
        normalize_to_90=bool(normalize_to_90),
    )
    if df_gref.empty:
        st.info("Geen bruikbare matchdata (First+Second Half) gevonden voor deze filters.")
        return

    df_gref_min = compute_gref_per_min(
        df_events,
        min_minutes=float(min_minutes),
        top_k=int(top_k),
    )
    if df_gref_min.empty:
        st.info("Geen data voor per-min tabel met deze filters.")
        return

    df_gref_calc = compute_gref_for_minutes(df_gref_min, float(calc_minutes))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Spelers", len(df_gref))
    with c2:
        st.metric("Minutenfilter", f"{int(min_minutes)}+")
    with c3:
        st.metric("Top-K matches", int(top_k))

    st.markdown("#### Gref")
    st.dataframe(df_gref, use_container_width=True, height=420)

    st.markdown("#### Gref per minuut")
    st.dataframe(df_gref_min, use_container_width=True, height=420)

    st.markdown(f"#### Omgerekend naar {int(calc_minutes) if float(calc_minutes).is_integer() else calc_minutes} minuten")
    st.dataframe(df_gref_calc, use_container_width=True, height=420)
