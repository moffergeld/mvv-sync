# pages/06_GPS_Import.py
# - Upload JOHAN Excel (SUMMARY / EXERCISES) -> parse -> upsert into public.gps_records
# - Date is manual (default today)
# - Export ALL gps_records to Excel (staff roles)
# - Export selected player(s) to Excel (each player -> own sheet)
# - Manual add rows via editable table (+ rows) with player select
#
# Requires:
#   st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_ANON_KEY"]
#   st.session_state["access_token"] set by your login (JWT access token)

import io
import re
from datetime import date

import pandas as pd
import requests
import streamlit as st

# Excel engine check (Streamlit Cloud must have openpyxl in requirements.txt)
try:
    import openpyxl  # noqa: F401
except Exception:
    st.error("Excel support ontbreekt: installeer openpyxl via requirements.txt")
    st.stop()

# =========================
# Page / Config
# =========================
st.set_page_config(page_title="GPS Import", layout="wide")

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing secrets: SUPABASE_URL / SUPABASE_ANON_KEY")
    st.stop()

ALLOWED_IMPORT = {"admin", "data_scientist", "staff", "physio", "performance_coach"}
TYPE_OPTIONS = ["Practice", "Practice (1)", "Practice (2)", "Match", "Practice Match"]

# =========================
# Auth / REST helpers
# =========================
def get_access_token() -> str | None:
    tok = st.session_state.get("access_token")
    if tok:
        return tok
    sess = st.session_state.get("sb_session")
    if sess is not None:
        token = getattr(sess, "access_token", None)
        if token:
            return token
    return None


def rest_headers(access_token: str) -> dict:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def rest_get(access_token: str, table: str, query: str) -> pd.DataFrame:
    url = f"{SUPABASE_URL}/rest/v1/{table}?{query}"
    r = requests.get(url, headers=rest_headers(access_token), timeout=60)
    if not r.ok:
        raise RuntimeError(f"GET {table} failed ({r.status_code}): {r.text}")
    return pd.DataFrame(r.json())


def rest_upsert(access_token: str, table: str, rows: list[dict], on_conflict: str) -> None:
    if not rows:
        return
    url = f"{SUPABASE_URL}/rest/v1/{table}?on_conflict={on_conflict}"
    headers = rest_headers(access_token)
    headers["Prefer"] = "resolution=merge-duplicates"
    CHUNK = 500
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i : i + CHUNK]
        r = requests.post(url, headers=headers, json=chunk, timeout=120)
        if not r.ok:
            raise RuntimeError(f"UPSERT {table} failed ({r.status_code}): {r.text}")


def auth_get_user(access_token: str) -> dict:
    url = f"{SUPABASE_URL}/auth/v1/user"
    r = requests.get(url, headers=rest_headers(access_token), timeout=30)
    if not r.ok:
        raise RuntimeError(f"AUTH user fetch failed ({r.status_code}): {r.text}")
    return r.json()


def normalize_role(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if "." in s:
        s = s.split(".")[-1]
    if "::" in s:
        s = s.split("::")[0]
    return s.strip() or None


@st.cache_data(ttl=60)
def get_profile_role(access_token: str) -> tuple[str | None, str | None, str | None, str | None]:
    u = auth_get_user(access_token)
    user_id = u.get("id")
    email = u.get("email")

    role = None
    team = None

    if user_id:
        dfp = rest_get(
            access_token,
            "profiles",
            f"select=user_id,role,team&user_id=eq.{user_id}&limit=1",
        )
        if not dfp.empty:
            role = normalize_role(dfp.iloc[0].get("role"))
            team = dfp.iloc[0].get("team")
    return user_id, email, role, team


# =========================
# Mapping helpers
# =========================
def normalize_key(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


@st.cache_data(ttl=120)
def get_players_map(access_token: str) -> tuple[dict, list[str]]:
    df = rest_get(access_token, "players", "select=player_id,full_name,is_active&is_active=eq.true&limit=5000")
    if df.empty:
        return {}, []
    df["full_name"] = df["full_name"].astype(str).str.strip()
    df = df.dropna(subset=["player_id", "full_name"])
    name_to_id = {normalize_name(n): pid for n, pid in zip(df["full_name"], df["player_id"])}
    display_names = sorted(df["full_name"].tolist())
    return name_to_id, display_names


# =========================
# gps_records columns
# =========================
GPS_COLS = [
    "player_id",
    "player_name",
    "datum",
    "week",
    "year",
    "type",
    "event",
    "duration",
    "total_distance",
    "walking",
    "jogging",
    "running",
    "sprint",
    "high_sprint",
    "number_of_sprints",
    "number_of_high_sprints",
    "number_of_repeated_sprints",
    "max_speed",
    "avg_speed",
    "playerload3d",
    "playerload2d",
    "total_accelerations",
    "high_accelerations",
    "total_decelerations",
    "high_decelerations",
    "hrzone1",
    "hrzone2",
    "hrzone3",
    "hrzone4",
    "hrzone5",
    "hrtrimp",
    "hrzoneanaerobic",
    "avg_hr",
    "max_hr",
    "source_file",
]

METRIC_MAP = {
    "duration": "duration",
    "totaldistance": "total_distance",
    "walking": "walking",
    "jogging": "jogging",
    "running": "running",
    "sprint": "sprint",
    "highsprint": "high_sprint",
    "numberofsprints": "number_of_sprints",
    "numberofhighsprints": "number_of_high_sprints",
    "numberofrepeatedsprints": "number_of_repeated_sprints",
    "maxspeed": "max_speed",
    "avgspeed": "avg_speed",
    "playerload3d": "playerload3d",
    "playerload2d": "playerload2d",
    "totalaccelerations": "total_accelerations",
    "highaccelerations": "high_accelerations",
    "totaldecelerations": "total_decelerations",
    "highdecelerations": "high_decelerations",
    "hrzone1": "hrzone1",
    "hrzone2": "hrzone2",
    "hrzone3": "hrzone3",
    "hrzone4": "hrzone4",
    "hrzone5": "hrzone5",
    "hrtrimp": "hrtrimp",
    "hrzoneanaerobic": "hrzoneanaerobic",
    "avghr": "avg_hr",
    "maxhr": "max_hr",
}

ID_COLS_IN_PARSER = ["Speler", "Datum", "Week", "Year", "Type", "Event"]

# Integer columns in Supabase schema
INT_COLS = {
    "number_of_sprints",
    "number_of_high_sprints",
    "number_of_repeated_sprints",
    "total_accelerations",
    "high_accelerations",
    "total_decelerations",
    "high_decelerations",
}


def drop_min_columns(df: pd.DataFrame) -> pd.DataFrame:
    min_cols = [c for c in df.columns if str(c).strip().endswith("/min")]
    return df.drop(columns=min_cols) if min_cols else df


def coerce_value(col: str, v):
    if pd.isna(v):
        return None

    if isinstance(v, (int, float)):
        if col in INT_COLS:
            try:
                return int(round(float(v)))
            except Exception:
                return None
        return float(v)

    s = str(v).strip()
    if s == "":
        return None

    s2 = s.replace(",", ".")

    if col in INT_COLS:
        try:
            return int(round(float(s2)))
        except Exception:
            return None

    try:
        return float(s2)
    except Exception:
        return None


def df_to_db_rows(df: pd.DataFrame, source_file: str, name_to_id: dict) -> tuple[list[dict], list[str]]:
    rows = []
    unmapped = set()

    parsed_dates = pd.to_datetime(df["Datum"], dayfirst=True, errors="coerce")
    if parsed_dates.isna().any():
        bad = df.loc[parsed_dates.isna(), "Datum"].head(5).tolist()
        raise ValueError(f"Kon sommige Datum waarden niet parsen, voorbeelden: {bad}")

    dates_iso = parsed_dates.dt.date.astype(str)

    for idx, r in df.iterrows():
        speler = str(r.get("Speler", "")).strip()
        if not speler:
            continue

        pid = name_to_id.get(normalize_name(speler))
        if not pid:
            unmapped.add(speler)

        dt = parsed_dates.iloc[idx].date()

        base = {
            "player_id": pid,
            "player_name": speler,
            "datum": dates_iso.iloc[idx],
            "week": int(dt.isocalendar().week),
            "year": int(dt.year),
            "type": str(r.get("Type", "")).strip(),
            "event": str(r.get("Event", "")).strip(),
            "source_file": source_file,
        }

        for c in df.columns:
            if c in ID_COLS_IN_PARSER or str(c).strip().endswith("/min"):
                continue
            key = normalize_key(c)
            if key in METRIC_MAP:
                db_col = METRIC_MAP[key]
                base[db_col] = coerce_value(db_col, r[c])

        base = {k: v for k, v in base.items() if k in GPS_COLS}
        rows.append(base)

    return rows, sorted(unmapped)


# =========================
# JOHAN Parsers
# =========================
def parse_summary_excel(file_bytes: bytes, selected_date: date, selected_type: str) -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(file_bytes), header=None)

    total_work_start = raw[raw[0] == "Total Work"].index[0]
    intensity_start = raw[raw[0] == "Intensity"].index[0]

    total_work_df = raw.iloc[total_work_start + 1 : intensity_start].dropna(how="all")
    total_work_df.columns = ["Variabele", "Eenheid"] + raw.iloc[0, 2:].tolist()
    total_work_df.set_index("Variabele", inplace=True)
    total_work_df = total_work_df.drop(columns=["Eenheid"])

    intensity_df = raw.iloc[intensity_start + 1 :].dropna(how="all")
    intensity_df.columns = ["Variabele", "Eenheid"] + raw.iloc[0, 2:].tolist()
    intensity_df.set_index("Variabele", inplace=True)
    intensity_df = intensity_df.drop(columns=["Eenheid"])

    intensity_df_renamed = intensity_df.copy()
    intensity_df_renamed.index = intensity_df_renamed.index + "/min"

    combined_df = pd.concat([total_work_df, intensity_df_renamed])
    combined_df.columns.name = None

    result_df = combined_df.transpose().reset_index().rename(columns={"index": "Speler"})

    result_df["Datum"] = pd.to_datetime(selected_date).strftime("%d-%m-%Y")
    result_df["Type"] = selected_type
    result_df["Event"] = "Summary"

    result_df = drop_min_columns(result_df)

    metric_cols = [c for c in result_df.columns if c not in ["Speler", "Datum", "Type", "Event"]]
    result_df[metric_cols] = result_df[metric_cols].fillna(0)

    dt = pd.to_datetime(selected_date)
    result_df["Week"] = int(dt.isocalendar().week)
    result_df["Year"] = int(dt.year)

    fixed = ["Speler", "Datum", "Week", "Year", "Type", "Event"]
    rest = [c for c in result_df.columns if c not in fixed]
    return result_df[fixed + rest]


def maak_lijst_uniek(lijst):
    seen = {}
    out = []
    for item in lijst:
        if item in seen:
            seen[item] += 1
            out.append(f"{item}_{seen[item]}")
        else:
            seen[item] = 1
            out.append(item)
    return out


def parse_exercises_excel(file_bytes: bytes, selected_date: date, selected_type: str) -> pd.DataFrame:
    xlsx = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = [s for s in xlsx.sheet_names if s.lower() != "spelerlijst"]

    alle = []
    for sheet in sheets:
        df = pd.read_excel(xlsx, sheet_name=sheet, header=None)

        speler = df.iloc[1, 0]
        oefenvormen = df.iloc[0, 2:].dropna().tolist()

        total_work_start = df[df[0] == "Total Work"].index[0]
        intensity_start = df[df[0] == "Intensity"].index[0]

        total_work_df = df.iloc[total_work_start + 1 : intensity_start].dropna(how="all")
        huidige_oefenvormen = oefenvormen[: total_work_df.shape[1] - 2]
        total_work_df.columns = maak_lijst_uniek(["Variabele", "Eenheid"] + huidige_oefenvormen)
        total_work_df.set_index("Variabele", inplace=True)

        intensity_df = df.iloc[intensity_start + 1 :].dropna(how="all")
        huidige_oefenvormen_i = oefenvormen[: intensity_df.shape[1] - 2]
        intensity_df.columns = maak_lijst_uniek(["Variabele", "Eenheid"] + huidige_oefenvormen_i)
        intensity_df.set_index("Variabele", inplace=True)

        for oef in [c for c in total_work_df.columns if c != "Eenheid"]:
            rec = {
                "Speler": speler,
                "Datum": pd.to_datetime(selected_date).strftime("%d-%m-%Y"),
                "Type": selected_type,
                "Event": str(oef).split("_")[0],
            }

            for var in total_work_df.index:
                rec[var] = total_work_df.at[var, oef]

            for var in intensity_df.index:
                if oef in intensity_df.columns:
                    rec[f"{var}/min"] = intensity_df.at[var, oef]

            alle.append(rec)

    out = pd.DataFrame(alle)
    out = drop_min_columns(out)

    metric_cols = [c for c in out.columns if c not in ["Speler", "Datum", "Type", "Event"]]
    out[metric_cols] = out[metric_cols].fillna(0)

    dt = pd.to_datetime(selected_date)
    out["Week"] = int(dt.isocalendar().week)
    out["Year"] = int(dt.year)

    fixed = ["Speler", "Datum", "Week", "Year", "Type", "Event"]
    rest = [c for c in out.columns if c not in fixed]
    # Zorg dat Event uniek is wanneer dezelfde oefening meerdere keren voorkomt
    out = ensure_unique_events(out)
    return out[fixed + rest]

def ensure_unique_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maakt Event uniek per (Speler, Datum, Type, Event).
    - Als een oefening maar 1x voorkomt: Event blijft onveranderd
    - Als een oefening meerdere keren voorkomt: suffix (1),(2),(3),... op ALLE occurrences
    """
    df = df.copy()

    keys = ["Speler", "Datum", "Type", "Event"]
    df["Event"] = df["Event"].astype(str).str.strip()

    # 0,1,2,... per groep
    idx = df.groupby(keys).cumcount()
    # groepsgrootte per rij
    grp_size = df.groupby(keys)["Event"].transform("size")

    # Alleen suffix toevoegen als groepsgrootte > 1
    mask = grp_size > 1
    df.loc[mask, "Event"] = df.loc[mask, "Event"] + " (" + (idx[mask] + 1).astype(str) + ")"

    return df

# =========================
# Export helpers
# =========================
def fetch_all_gps_records(access_token: str, limit: int = 200000) -> pd.DataFrame:
    query = f"select={','.join(GPS_COLS)}&order=datum.desc&limit={limit}"
    return rest_get(access_token, "gps_records", query)


def fetch_gps_for_players(access_token: str, player_names: list[str], limit_per_player: int = 200000) -> dict[str, pd.DataFrame]:
    """
    Fetch gps_records per player_name. Returns dict {player_name: df}
    Uses exact match on player_name.
    """
    out = {}
    for name in player_names:
        safe = str(name).replace('"', "")
        # PostgREST: player_name=eq.<value> (URL encode spaces via requests automatically? we build manually: replace spaces with %20)
        # We'll do minimal encoding:
        encoded = requests.utils.quote(safe, safe="")
        query = (
            f"select={','.join(GPS_COLS)}"
            f"&player_name=eq.{encoded}"
            f"&order=datum.desc"
            f"&limit={int(limit_per_player)}"
        )
        df = rest_get(access_token, "gps_records", query)
        out[name] = df
    return out


def df_to_excel_bytes_single(df: pd.DataFrame, sheet_name: str = "gps_records") -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return bio.getvalue()


def safe_sheet_name(name: str, used: set[str]) -> str:
    """
    Excel sheet max 31 chars, cannot contain: : \ / ? * [ ]
    Also must be unique within file.
    """
    s = str(name).strip()
    s = re.sub(r"[:\\/?*\[\]]", "_", s)
    s = s[:31] if len(s) > 31 else s
    if not s:
        s = "Sheet"
    base = s
    i = 1
    while s in used:
        suffix = f"_{i}"
        s = (base[: 31 - len(suffix)] + suffix) if len(base) + len(suffix) > 31 else (base + suffix)
        i += 1
    used.add(s)
    return s


def dfs_to_excel_bytes_multi(dfs: dict[str, pd.DataFrame]) -> bytes:
    """
    Each dict key -> one worksheet.
    """
    bio = io.BytesIO()
    used = set()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for player_name, df in dfs.items():
            sheet = safe_sheet_name(player_name, used)
            if df is None or df.empty:
                pd.DataFrame(columns=GPS_COLS).to_excel(writer, index=False, sheet_name=sheet)
            else:
                ordered = [c for c in GPS_COLS if c in df.columns]
                df[ordered].to_excel(writer, index=False, sheet_name=sheet)
    return bio.getvalue()


# =========================
# UI
# =========================
st.title("GPS Import")

access_token = get_access_token()
if not access_token:
    st.error("Niet ingelogd (access_token ontbreekt).")
    st.stop()

# Role from Supabase profiles
try:
    user_id, email, role, team = get_profile_role(access_token)
except Exception as e:
    st.error(f"Kon profiel/role niet ophalen: {e}")
    st.stop()

with st.expander("Debug (auth/role)", expanded=False):
    st.write({"email": email, "user_id": user_id, "role": role, "team": team})
    st.write({"session_role_raw": st.session_state.get("role")})

if role not in ALLOWED_IMPORT:
    st.error("Geen rechten voor GPS import/export.")
    st.stop()

name_to_id, player_options = get_players_map(access_token)

tab_import, tab_manual, tab_export = st.tabs(["Import (Excel)", "Manual add", "Export"])

# -------------------------
# TAB 1: Import (Excel)
# -------------------------
with tab_import:
    st.subheader("Import JOHAN Excel")

    col1, col2, col3 = st.columns([1.2, 1.2, 2.6])
    with col1:
        selected_date = st.date_input("Datum", value=date.today())
    with col2:
        selected_type = st.selectbox("Type", TYPE_OPTIONS)
    with col3:
        uploaded_files = st.file_uploader(
            "Upload Excel (meerdere bestanden: SUMMARY + EXERCISES)",
            type=["xlsx", "xls"],
            accept_multiple_files=True,
        )

    if uploaded_files:
        colP1, colP2 = st.columns([1, 1])
        with colP1:
            do_preview = st.button("Preview (parse) – alle bestanden")
        with colP2:
            do_import = st.button("Import (upsert) – alle bestanden", type="primary")

        if do_preview or do_import:
            all_parsed = []
            all_rows = []
            all_unmapped = set()

            for uf in uploaded_files:
                filename = uf.name
                file_bytes = uf.getvalue()

                is_summary = "summary" in filename.lower()
                is_exercises = "exercises" in filename.lower()

                # Parse per file
                if is_summary:
                    df_parsed = parse_summary_excel(file_bytes, selected_date, selected_type)
                elif is_exercises:
                    df_parsed = parse_exercises_excel(file_bytes, selected_date, selected_type)
                else:
                    # fallback: probeer summary, anders exercises
                    try:
                        df_parsed = parse_summary_excel(file_bytes, selected_date, selected_type)
                    except Exception:
                        df_parsed = parse_exercises_excel(file_bytes, selected_date, selected_type)

                all_parsed.append((filename, df_parsed))

                # Rows voor import
                rows, unmapped = df_to_db_rows(df_parsed, source_file=filename, name_to_id=name_to_id)
                all_rows.extend(rows)
                all_unmapped.update(unmapped)

            # Preview output
            if do_preview:
                st.success(
                    f"Parsed bestanden: {len(all_parsed)} | totaal rijen: {sum(len(d) for _, d in all_parsed)}"
                )
                for fn, d in all_parsed:
                    st.markdown(f"**{fn}** ({len(d)} rijen)")
                    st.dataframe(d.head(50), use_container_width=True)

                # optioneel debug (zet aan als je wilt)
                # if all_parsed:
                #     st.write("Voorbeeld kolommen:", list(all_parsed[0][1].columns))

            # Import (upsert) output
            if do_import:
                if all_unmapped:
                    st.warning(
                        "Niet gematchte namen (player_id blijft NULL):\n- "
                        + "\n- ".join(sorted(list(all_unmapped))[:50])
                        + (f"\n... (+{len(all_unmapped)-50})" if len(all_unmapped) > 50 else "")
                    )

                # Deduplicate binnen dezelfde batch op conflict-key
                # (voorkomt: ON CONFLICT DO UPDATE command cannot affect row a second time)
                dedup = {}
                for r in all_rows:
                    k = (r.get("player_name"), r.get("datum"), r.get("type"), r.get("event"))
                    dedup[k] = r
                all_rows = list(dedup.values())

                try:
                    rest_upsert(
                        access_token,
                        "gps_records",
                        all_rows,
                        on_conflict="player_name,datum,type,event",
                    )
                    st.success(f"✅ Import klaar. Upserted rows: {len(all_rows)}")
                except Exception as e:
                    st.error(f"Import fout: {e}")


# -------------------------
# TAB 2: Manual add
# -------------------------
with tab_manual:
    st.subheader("Manual add (tabel)")
    st.caption(
        "Voeg rijen toe met het + icoon. Kies speler in de eerste kolom. "
        "Klik daarna op 'Save rows (upsert)'."
    )

    template_cols = [
        "player_name",
        "datum",
        "type",
        "event",
        "duration",
        "total_distance",
        "walking",
        "jogging",
        "running",
        "sprint",
        "high_sprint",
        "number_of_sprints",
        "number_of_high_sprints",
        "number_of_repeated_sprints",
        "max_speed",
        "avg_speed",
        "playerload3d",
        "playerload2d",
        "total_accelerations",
        "high_accelerations",
        "total_decelerations",
        "high_decelerations",
        "hrzone1",
        "hrzone2",
        "hrzone3",
        "hrzone4",
        "hrzone5",
        "hrtrimp",
        "hrzoneanaerobic",
        "avg_hr",
        "max_hr",
        "source_file",
    ]

    if "manual_df" not in st.session_state:
        st.session_state["manual_df"] = pd.DataFrame(
            [
                {
                    "player_name": player_options[0] if player_options else "",
                    "datum": date.today(),
                    "type": "Practice",
                    "event": "Summary",
                    "source_file": "manual",
                }
            ],
            columns=template_cols,
        )

    colcfg = {
        "player_name": st.column_config.SelectboxColumn(
            "player_name",
            options=player_options,
            required=True,
        ),
        "datum": st.column_config.DateColumn("datum", required=True),
        "type": st.column_config.SelectboxColumn("type", options=TYPE_OPTIONS, required=True),
        "event": st.column_config.TextColumn("event", required=True),
        "source_file": st.column_config.TextColumn("source_file"),
    }

    edited = st.data_editor(
        st.session_state["manual_df"],
        use_container_width=True,
        num_rows="dynamic",
        column_config=colcfg,
        key="manual_editor",
    )

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Reset table"):
            st.session_state["manual_df"] = pd.DataFrame(
                [
                    {
                        "player_name": player_options[0] if player_options else "",
                        "datum": date.today(),
                        "type": "Practice",
                        "event": "Summary",
                        "source_file": "manual",
                    }
                ],
                columns=template_cols,
            )
            st.rerun()

    with colB:
        if st.button("Save rows (upsert)", type="primary"):
            try:
                dfm = edited.copy()

                dfm["player_name"] = dfm["player_name"].astype(str).str.strip()
                dfm["event"] = dfm["event"].astype(str).str.strip()
                dfm["type"] = dfm["type"].astype(str).str.strip()
                dfm = dfm.dropna(subset=["player_name", "datum", "type", "event"])
                dfm = dfm[(dfm["player_name"] != "") & (dfm["event"] != "")]

                if dfm.empty:
                    st.error("Geen geldige rijen om op te slaan.")
                    st.stop()

                dt_series = pd.to_datetime(dfm["datum"], errors="coerce")
                if dt_series.isna().any():
                    st.error("Ongeldige datum in één of meer rijen.")
                    st.stop()

                dfm["week"] = dt_series.dt.isocalendar().week.astype(int)
                dfm["year"] = dt_series.dt.year.astype(int)
                dfm["datum"] = dt_series.dt.date.astype(str)

                dfm["player_id"] = dfm["player_name"].map(lambda x: name_to_id.get(normalize_name(x)))

                # numeric coercion
                for c in dfm.columns:
                    if c in {"player_name", "datum", "type", "event", "source_file"}:
                        continue
                    if c in {"player_id", "week", "year"}:
                        continue
                    dfm[c] = pd.to_numeric(dfm[c], errors="coerce")

                rows = []
                metric_keys = [
                    "duration","total_distance","walking","jogging","running","sprint","high_sprint",
                    "number_of_sprints","number_of_high_sprints","number_of_repeated_sprints",
                    "max_speed","avg_speed","playerload3d","playerload2d",
                    "total_accelerations","high_accelerations","total_decelerations","high_decelerations",
                    "hrzone1","hrzone2","hrzone3","hrzone4","hrzone5","hrtrimp","hrzoneanaerobic","avg_hr","max_hr"
                ]

                for _, r in dfm.iterrows():
                    row = {
                        "player_id": r.get("player_id"),
                        "player_name": r.get("player_name"),
                        "datum": r.get("datum"),
                        "week": int(r.get("week")) if pd.notna(r.get("week")) else None,
                        "year": int(r.get("year")) if pd.notna(r.get("year")) else None,
                        "type": r.get("type"),
                        "event": r.get("event"),
                        "source_file": r.get("source_file") or "manual",
                    }
                    for k in metric_keys:
                        row[k] = coerce_num(r.get(k))
                    rows.append(row)

                # Deduplicate binnen batch op conflict-key
                dedup = {}
                for rr in rows:
                    k = (rr.get("player_name"), rr.get("datum"), rr.get("type"), rr.get("event"))
                    dedup[k] = rr
                rows = list(dedup.values())

                rest_upsert(
                    access_token,
                    "gps_records",
                    rows,
                    on_conflict="player_name,datum,type,event",
                )
                st.success(f"✅ Saved rows: {len(rows)}")
                st.session_state["manual_df"] = edited
            except Exception as e:
                st.error(f"Save fout: {e}")


# -------------------------
# TAB 3: Export
# -------------------------
with tab_export:
    st.subheader("Export gps_records → Excel")

    c1, c2 = st.columns([1.4, 2.6])
    with c1:
        limit = st.number_input(
            "Max rows (veiligheidslimiet)",
            min_value=1,
            max_value=500000,
            value=200000,
            step=10000,
        )
    with c2:
        st.caption("Export haalt kolommen op en maakt een .xlsx download.")

    if st.button("Generate export"):
        try:
            df = fetch_all_gps_records(access_token, limit=int(limit))
            if df.empty:
                st.warning("Geen data gevonden.")
            else:
                ordered = [c for c in GPS_COLS if c in df.columns]
                df = df[ordered]
                xbytes = df_to_excel_bytes(df, sheet_name="gps_records")
                st.download_button(
                    "Download gps_records.xlsx",
                    data=xbytes,
                    file_name="gps_records.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                st.success(f"✅ Klaar: {len(df)} rijen")
        except Exception as e:
            st.error(str(e))

