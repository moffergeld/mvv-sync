from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="MVV – GPS Import", layout="wide")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

# login state (moet al in app.py gezet worden)
if "access_token" not in st.session_state:
    st.error("Niet ingelogd.")
    st.stop()

ACCESS_TOKEN = st.session_state["access_token"]

def rest_get(table: str, query: str) -> pd.DataFrame:
    url = f"{SUPABASE_URL}/rest/v1/{table}?{query}"
    headers = {"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {ACCESS_TOKEN}"}
    r = requests.get(url, headers=headers, timeout=60)
    if not r.ok:
        st.error(f"GET {table} failed ({r.status_code})")
        st.code(r.text)
        return pd.DataFrame()
    return pd.DataFrame(r.json())

def rest_upsert(table: str, rows: list[dict], on_conflict: str) -> tuple[bool, str]:
    url = f"{SUPABASE_URL}/rest/v1/{table}?on_conflict={on_conflict}"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }
    r = requests.post(url, headers=headers, json=rows, timeout=120)
    if not r.ok:
        return False, f"{r.status_code}\n{r.text}"
    return True, "OK"

@st.cache_data(ttl=60)
def get_role() -> str:
    df = rest_get("profiles", "select=role&limit=1")
    if df.empty:
        return "unknown"
    return str(df.iloc[0]["role"]).lower()

@st.cache_data(ttl=60)
def get_players_map() -> pd.DataFrame:
    df = rest_get("players", "select=player_id,full_name,is_active&limit=5000")
    if df.empty:
        return df
    df["full_name"] = df["full_name"].astype(str).str.strip().str.lower()
    df = df[df["is_active"] == True]
    return df[["player_id", "full_name"]].drop_duplicates("full_name")

role = get_role()
st.title("GPS Import")

if role not in ("staff", "admin"):
    st.error("Alleen staff/admin mag importeren.")
    st.stop()

players_map = get_players_map()

uploaded = st.file_uploader("Upload GPS CSV(s)", type=["csv"], accept_multiple_files=True)

REQUIRED = ["Speler", "Datum", "Type", "Event"]

COLMAP = {
    "Speler": "player_name",
    "Datum": "datum",
    "Week": "week",
    "Year": "year",
    "Type": "type",
    "Event": "event",
    "Duration": "duration",
    "Total Distance": "total_distance",
    "Walking": "walking",
    "Jogging": "jogging",
    "Running": "running",
    "Sprint": "sprint",
    "High Sprint": "high_sprint",
    "Number of sprints": "number_of_sprints",
    "Number of high sprints": "number_of_high_sprints",
    "Number of repeated sprints": "number_of_repeated_sprints",
    "Max Speed": "max_speed",
    "Avg Speed": "avg_speed",
    "playerload3D": "playerload3d",
    "playerload2D": "playerload2d",
    "Total Accelerations": "total_accelerations",
    "High Accelerations": "high_accelerations",
    "Total Decelerations": "total_decelerations",
    "High Decelerations": "high_decelerations",
    "HRzone1": "hrzone1",
    "HRzone2": "hrzone2",
    "HRzone3": "hrzone3",
    "HRzone4": "hrzone4",
    "HRzone5": "hrzone5",
    "HRtrimp": "hrtrimp",
    "HRzoneanaerobic": "hrzoneanaerobic",
    "Avg HR": "avg_hr",
    "Max HR": "max_hr",
}

def normalize_df(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV mist verplichte kolommen: {missing}")

    keep = [c for c in df.columns if c in COLMAP]
    df = df[keep].rename(columns={k: v for k, v in COLMAP.items() if k in keep}).copy()

    # basics
    df["datum"] = pd.to_datetime(df["datum"], dayfirst=True, errors="coerce").dt.date
    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip()
    df["event"] = df["event"].astype(str).str.strip()

    # numeric
    for c in df.columns:
        if c in ("player_name", "datum", "type", "event"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # map player_name -> player_id (via players.full_name)
    df["_name_lc"] = df["player_name"].str.lower()
    df = df.merge(players_map, left_on="_name_lc", right_on="full_name", how="left")
    df = df.drop(columns=["_name_lc", "full_name"])

    df["source_file"] = source_file

    # drop unusable rows
    df = df.dropna(subset=["player_name", "datum", "type", "event"])

    return df

def chunked(lst, n=500):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

if uploaded:
    all_batches = []
    preview_df = None

    for f in uploaded:
        raw = pd.read_csv(f, sep=";", encoding="utf-8-sig", engine="python")
        df = normalize_df(raw, f.name)
        if preview_df is None:
            preview_df = df.copy()
        all_batches.append((f.name, df))

    st.subheader("Preview (eerste bestand)")
    st.dataframe(preview_df.head(50), use_container_width=True)

    unmapped = int(preview_df["player_id"].isna().sum()) if "player_id" in preview_df.columns else 0
    st.caption(f"Player match: {len(preview_df)-unmapped} gematcht, {unmapped} niet gematcht (die blijven zichtbaar voor staff).")

    if st.button("Importeer (upsert)", type="primary"):
        total = 0
        for name, df in all_batches:
            rows = df.to_dict(orient="records")
            for batch in chunked(rows, 500):
                ok, msg = rest_upsert("gps_records", batch, on_conflict="player_name,datum,type,event")
                if not ok:
                    st.error(f"❌ Import faalde: {name}")
                    st.code(msg)
                    st.stop()
                total += len(batch)
            st.success(f"✅ {name}: OK")
        st.success(f"✅ Klaar. Verwerkt: {total} rijen.")
