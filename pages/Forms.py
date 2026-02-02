# pages/Forms.py
# MVV – Forms (Wellness + RPE) with Staff Team/Individual views + Timeline controls
# Uses st.date_input for date ranges (more stable on Streamlit Cloud)

from __future__ import annotations

from datetime import date
import pandas as pd
import requests
import streamlit as st

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="MVV – Forms", layout="wide")

# -------------------------
# Config / Secrets
# -------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

if "access_token" not in st.session_state:
    st.error("Niet ingelogd. Ga terug naar Home (app.py) en log in.")
    st.stop()

ACCESS_TOKEN = st.session_state["access_token"]


# -------------------------
# REST helper (RLS via user token)
# -------------------------
def rest_get(table: str, query: str, show_error: bool = True) -> pd.DataFrame:
    """
    Calls Supabase PostgREST with Bearer token (RLS enforced).
    query example: "select=*&order=form_date.asc&limit=5000"
    """
    url = f"{SUPABASE_URL}/rest/v1/{table}?{query}"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {ACCESS_TOKEN}",
    }
    r = requests.get(url, headers=headers, timeout=60)

    if not r.ok:
        if show_error:
            st.error(f"REST error for '{table}' ({r.status_code})")
            st.code(r.text)
        return pd.DataFrame()

    return pd.DataFrame(r.json())


# -------------------------
# Data loaders (cached)
# -------------------------
@st.cache_data(ttl=60)
def get_profile() -> dict:
    # profiles is RLS self-only; returns exactly one row for logged-in user
    df = rest_get("profiles", "select=role,team,player_id", show_error=False)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


@st.cache_data(ttl=60)
def load_wellness() -> pd.DataFrame:
    df = rest_get("wellness", "select=*&order=form_date.asc&limit=5000", show_error=False)
    if not df.empty and "form_date" in df.columns:
        df["form_date"] = pd.to_datetime(df["form_date"], errors="coerce").dt.date
    return df


@st.cache_data(ttl=60)
def load_rpe() -> pd.DataFrame:
    df = rest_get("rpe", "select=*&order=form_date.asc&limit=5000", show_error=False)
    if not df.empty and "form_date" in df.columns:
        df["form_date"] = pd.to_datetime(df["form_date"], errors="coerce").dt.date

    # Convert known numeric fields if present
    for c in ["ex1_duration_min", "ex1_exertion", "ex2_duration_min", "ex2_exertion"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute sRPE if possible
    if not df.empty:
        df["srpe_ex1"] = df.get("ex1_duration_min") * df.get("ex1_exertion")
        df["srpe_ex2"] = df.get("ex2_duration_min") * df.get("ex2_exertion")
        df["srpe_total"] = df["srpe_ex1"].fillna(0) + df["srpe_ex2"].fillna(0)

    return df


@st.cache_data(ttl=60)
def load_players_map() -> pd.DataFrame:
    """
    We don't show a Players tab, but staff needs a dropdown.
    Robustly detect a display-name column.
    """
    df = rest_get("players", "select=*&limit=5000", show_error=False)
    if df.empty:
        return df

    # Find ID column
    id_col_candidates = ["player_id", "id"]
    id_col = next((c for c in id_col_candidates if c in df.columns), None)
    if id_col is None:
        return pd.DataFrame()

    # Find name column
    name_col_candidates = ["full_name", "name", "player_name", "Naam", "Speler"]
    name_col = next((c for c in name_col_candidates if c in df.columns), None)

    if name_col is None:
        df["__display_name"] = df[id_col].astype(str)
    else:
        df["__display_name"] = df[name_col].astype(str)

    out = df[[id_col, "__display_name"]].rename(columns={id_col: "player_id"})
    out = (
        out.dropna(subset=["player_id"])
        .drop_duplicates("player_id")
        .sort_values("__display_name")
        .reset_index(drop=True)
    )
    return out


# -------------------------
# Utilities
# -------------------------
def date_bounds(*dfs: pd.DataFrame) -> tuple[date, date] | None:
    all_dates: list[date] = []
    for df in dfs:
        if df is not None and not df.empty and "form_date" in df.columns:
            all_dates.extend([d for d in df["form_date"].dropna().tolist() if isinstance(d, date)])
    if not all_dates:
        return None
    return min(all_dates), max(all_dates)


def filter_range(df: pd.DataFrame, start_d: date, end_d: date) -> pd.DataFrame:
    if df is None or df.empty or "form_date" not in df.columns:
        return df
    return df[(df["form_date"] >= start_d) & (df["form_date"] <= end_d)].copy()


# -------------------------
# Page UI
# -------------------------
profile = get_profile()
role = (profile.get("role") or "unknown").lower()
team = profile.get("team") or "-"

st.title("Forms")
st.caption(f"Ingelogd als: {st.session_state.get('user_email','')}")

m1, m2 = st.columns([1.2, 2.5])
with m1:
    st.metric("Role", role)
with m2:
    st.metric("Team", team)

dfw = load_wellness()
dfr = load_rpe()

players_map = pd.DataFrame()
if role in ("staff", "admin"):
    players_map = load_players_map()

bounds = date_bounds(dfw, dfr)
if bounds is None:
    st.warning("Geen data gevonden (nog).")
    st.stop()

min_d, max_d = bounds

# -------------------------
# Controls (staff: team/individual + timeline)
# -------------------------
if role in ("staff", "admin"):
    top = st.columns([1.4, 2.2, 2.2])

    with top[0]:
        view_mode = st.radio("Weergave", ["Team", "Individueel"], horizontal=True)

    with top[1]:
        date_range = st.date_input(
            "Timeline (range)",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
        )

        # date_input can return a single date; normalize to range
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = date_range
        else:
            start_d, end_d = min_d, max_d

        if start_d > end_d:
            start_d, end_d = end_d, start_d

    with top[2]:
        day_focus = st.date_input(
            "1 dag bekijken (optioneel)",
            value=max_d,
            min_value=min_d,
            max_value=max_d,
        )

else:
    view_mode = "Individueel"
    start_d, end_d = min_d, max_d
    day_focus = max_d

dfw_f = filter_range(dfw, start_d, end_d)
dfr_f = filter_range(dfr, start_d, end_d)

tab_w, tab_r = st.tabs(["Wellness", "RPE"])


# -------------------------
# WELLNESS TAB
# -------------------------
with tab_w:
    st.subheader("Wellness")

    if dfw_f is None or dfw_f.empty:
        st.info("Geen wellness data in deze periode.")
    else:
        wellness_metrics = [
            c
            for c in ["muscle_soreness", "fatigue", "sleep_quality", "stress", "mood"]
            if c in dfw_f.columns
        ]

        if view_mode == "Team":
            # Team average per day (multi-line chart)
            if wellness_metrics:
                daily = dfw_f.groupby("form_date")[wellness_metrics].mean().sort_index()
                st.caption("Team gemiddelde per dag")
                st.line_chart(daily)

            # Day focus: distribution across players
            st.caption(f"1 dag focus: {day_focus}")
            day_df = dfw_f[dfw_f["form_date"] == day_focus].copy()
            if day_df.empty:
                st.info("Geen wellness entries op deze dag.")
            else:
                metric = st.selectbox(
                    "Metric (dagoverzicht)",
                    wellness_metrics if wellness_metrics else [c for c in day_df.columns if c not in ["created_by"]],
                    index=0,
                )

                if "player_id" in day_df.columns and metric in day_df.columns:
                    show = day_df[["player_id", metric]].copy()
                else:
                    show = day_df.copy()

                # Add names if possible
                if not players_map.empty and "player_id" in show.columns:
                    show = show.merge(players_map, on="player_id", how="left")
                    if "__display_name" in show.columns:
                        show = show.rename(columns={"__display_name": "player"})

                st.dataframe(show, use_container_width=True)

                if "player" in show.columns and metric in show.columns:
                    chart = show[["player", metric]].dropna().set_index("player")
                    st.bar_chart(chart)

        else:
            # Individual
            if role in ("staff", "admin"):
                if players_map.empty:
                    st.warning("Kan spelerslijst niet ophalen (players). Check kolomnamen/RLS voor players.")
                    sub = dfw_f.copy()
                else:
                    choice = st.selectbox("Selecteer speler", players_map["__display_name"].tolist())
                    pid = players_map.loc[players_map["__display_name"] == choice, "player_id"].iloc[0]
                    sub = dfw_f[dfw_f["player_id"] == pid].copy() if "player_id" in dfw_f.columns else dfw_f.copy()
            else:
                sub = dfw_f.copy()

            if sub is None or sub.empty:
                st.info("Geen wellness data voor deze speler/periode.")
            else:
                # One chart per category over time
                if wellness_metrics:
                    for m in wellness_metrics:
                        st.caption(m)
                        s = sub[["form_date", m]].dropna().groupby("form_date")[m].mean().sort_index()
                        st.line_chart(s)

                st.dataframe(sub, use_container_width=True)


# -------------------------
# RPE TAB
# -------------------------
with tab_r:
    st.subheader("RPE")

    if dfr_f is None or dfr_f.empty:
        st.info("Geen RPE data in deze periode.")
    else:
        rpe_value_col = "srpe_total" if "srpe_total" in dfr_f.columns else None

        if view_mode == "Team":
            # Team total sRPE per day
            if rpe_value_col:
                daily = dfr_f.groupby("form_date")[rpe_value_col].sum().sort_index()
                st.caption("Team totaal sRPE per dag")
                st.line_chart(daily)
            else:
                st.warning("srpe_total ontbreekt (missen duur/inspanning kolommen).")

            # Day focus
            st.caption(f"1 dag focus: {day_focus}")
            day_df = dfr_f[dfr_f["form_date"] == day_focus].copy()
            if day_df.empty:
                st.info("Geen RPE entries op deze dag.")
            else:
                if rpe_value_col and "player_id" in day_df.columns:
                    show = day_df[["player_id", rpe_value_col]].copy()

                    if not players_map.empty:
                        show = show.merge(players_map, on="player_id", how="left")
                        if "__display_name" in show.columns:
                            show = show.rename(columns={"__display_name": "player"})

                    st.dataframe(show, use_container_width=True)

                    if "player" in show.columns:
                        chart = show[["player", rpe_value_col]].dropna().set_index("player")
                        st.bar_chart(chart)
                else:
                    st.dataframe(day_df, use_container_width=True)

        else:
            # Individual
            if role in ("staff", "admin"):
                if players_map.empty:
                    st.warning("Kan spelerslijst niet ophalen (players). Check kolomnamen/RLS voor players.")
                    sub = dfr_f.copy()
                else:
                    choice = st.selectbox(
                        "Selecteer speler",
                        players_map["__display_name"].tolist(),
                        key="rpe_player",
                    )
                    pid = players_map.loc[players_map["__display_name"] == choice, "player_id").iloc[0]
                    sub = dfr_f[dfr_f["player_id"] == pid].copy() if "player_id" in dfr_f.columns else dfr_f.copy()
            else:
                sub = dfr_f.copy()

            if sub is None or sub.empty:
                st.info("Geen RPE data voor deze speler/periode.")
            else:
                if rpe_value_col:
                    s = sub.groupby("form_date")[rpe_value_col].sum().sort_index()
                    st.caption("sRPE verloop in tijd")
                    st.line_chart(s)

                st.dataframe(sub, use_container_width=True)
