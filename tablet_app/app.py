from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import extra_streamlit_components as stx
import streamlit as st
from supabase import create_client


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
PAGES_DIR = ROOT_DIR / "pages"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PAGES_DIR))

from Subscripts.player_tab_forms import (  # noqa: E402
    _legend_asrm,
    _legend_rpe,
    load_asrm,
    load_rpe,
    save_asrm,
    save_rpe,
)


APP_TITLE = "MVV Tablet Check-in"
ACCESS_COOKIE_NAME = "mvv_tablet_access"
ACCESS_COOKIE_SECONDS = 60 * 60 * 24 * 30


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
      #MainMenu,
      header,
      footer,
      [data-testid="stSidebar"],
      [data-testid="collapsedControl"] {
        display: none !important;
      }

      .block-container {
        max-width: 1180px;
        padding-top: 1rem;
        padding-bottom: 2rem;
      }

      div.stButton > button {
        min-height: 5.25rem;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 700;
        white-space: pre-line;
      }

      div[data-testid="stMetricValue"] {
        font-size: 2rem;
      }

      .tablet-hero {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(49, 51, 63, 0.15);
        border-radius: 10px;
        background: rgba(240, 242, 246, 0.6);
        margin-bottom: 1rem;
      }

      .tablet-hero h1 {
        margin: 0;
        font-size: 2rem;
        line-height: 1.1;
      }

      .tablet-hero p {
        margin: 0.45rem 0 0 0;
        font-size: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def amsterdam_today():
    return datetime.now(ZoneInfo("Europe/Amsterdam")).date()


def cookie_mgr():
    if "_tablet_cookie_mgr" not in st.session_state:
        st.session_state["_tablet_cookie_mgr"] = stx.CookieManager(key="tablet_cookie_mgr")
    return st.session_state["_tablet_cookie_mgr"]


def get_tablet_code() -> str:
    for key in ("TABLET_SHARED_CODE", "TABLET_CODE", "KIOSK_CODE"):
        value = str(st.secrets.get(key, "") or "").strip()
        if value:
            return value
    return ""


def get_service_key() -> str:
    for key in ("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_SERVICE_KEY", "SUPABASE_SERVICE_ROLE"):
        value = str(st.secrets.get(key, "") or "").strip()
        if value:
            return value
    return ""


def get_tablet_sb():
    url = str(st.secrets.get("SUPABASE_URL", "") or "").strip()
    service_key = get_service_key()
    if not url or not service_key:
        return None

    if "_tablet_sb_client" not in st.session_state or st.session_state.get("_tablet_sb_client") is None:
        st.session_state["_tablet_sb_client"] = create_client(url, service_key)

    return st.session_state["_tablet_sb_client"]


def grant_tablet_access() -> None:
    cm = cookie_mgr()
    cm.set(ACCESS_COOKIE_NAME, "1", max_age=ACCESS_COOKIE_SECONDS, key="tablet_access_set")
    st.session_state["tablet_unlocked"] = True
    time.sleep(0.10)
    st.rerun()


def lock_tablet() -> None:
    cm = cookie_mgr()
    cm.set(ACCESS_COOKIE_NAME, "", max_age=1, key="tablet_access_clear")
    for key in (
        "tablet_unlocked",
        "tablet_player_id",
        "tablet_player_name",
        "tablet_flash",
    ):
        st.session_state.pop(key, None)
    time.sleep(0.10)
    st.rerun()


def ensure_tablet_access() -> None:
    shared_code = get_tablet_code()
    if not shared_code:
        st.error("Tabletcode ontbreekt in Streamlit secrets. Voeg TABLET_SHARED_CODE toe.")
        st.stop()

    cm = cookie_mgr()
    _ = cm.get(ACCESS_COOKIE_NAME)

    if st.session_state.get("tablet_unlocked"):
        return

    query_code = str(st.query_params.get("code", "") or "").strip()
    if query_code and query_code == shared_code:
        grant_tablet_access()

    cookie_value = str(cm.get(ACCESS_COOKIE_NAME) or "").strip()
    if cookie_value == "1":
        st.session_state["tablet_unlocked"] = True
        return

    st.markdown(
        """
        <div class="tablet-hero">
          <h1>Tablet toegang</h1>
          <p>Voer de teamcode in om de check-in pagina te openen.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("tablet_unlock_form", clear_on_submit=False):
        code_value = st.text_input("Tabletcode", type="password")
        submitted = st.form_submit_button("Open tablet", use_container_width=True)

    if submitted:
        if code_value.strip() == shared_code:
            grant_tablet_access()
        else:
            st.error("Code klopt niet.")

    st.stop()


@st.cache_data(show_spinner=False, ttl=120)
def fetch_active_players(_sb) -> List[Dict[str, str]]:
    try:
        rows = (
            _sb.table("players")
            .select("player_id,full_name")
            .eq("is_active", True)
            .order("full_name")
            .execute()
            .data
            or []
        )
    except Exception:
        rows = []

    out: List[Dict[str, str]] = []
    for row in rows:
        player_id = row.get("player_id")
        full_name = str(row.get("full_name") or "").strip()
        if player_id and full_name:
            out.append({"player_id": str(player_id), "full_name": full_name})
    return out


@st.cache_data(show_spinner=False, ttl=30)
def fetch_daily_completion(_sb, entry_date_iso: str) -> Dict[str, List[str]]:
    try:
        asrm_rows = (
            _sb.table("asrm_entries")
            .select("player_id")
            .eq("entry_date", entry_date_iso)
            .execute()
            .data
            or []
        )
    except Exception:
        asrm_rows = []

    try:
        rpe_rows = (
            _sb.table("rpe_entries")
            .select("player_id")
            .eq("entry_date", entry_date_iso)
            .execute()
            .data
            or []
        )
    except Exception:
        rpe_rows = []

    asrm_ids = sorted({str(row.get("player_id")) for row in asrm_rows if row.get("player_id")})
    rpe_ids = sorted({str(row.get("player_id")) for row in rpe_rows if row.get("player_id")})
    return {"asrm_ids": asrm_ids, "rpe_ids": rpe_ids}


def clear_daily_cache() -> None:
    fetch_daily_completion.clear()


def show_flash() -> None:
    flash = st.session_state.pop("tablet_flash", None)
    if flash:
        st.success(str(flash))


def render_top_actions(show_back: bool = False) -> None:
    cols = st.columns([1, 1, 4])

    with cols[0]:
        if show_back and st.button("Spelers", use_container_width=True, key="tablet_back_to_list"):
            st.session_state.pop("tablet_player_id", None)
            st.session_state.pop("tablet_player_name", None)
            st.rerun()

    with cols[1]:
        if st.button("Vergrendel", use_container_width=True, key=f"tablet_lock_{show_back}"):
            lock_tablet()


def render_player_picker(sb) -> None:
    entry_date = amsterdam_today()
    players = fetch_active_players(sb)
    if not players:
        st.warning("Geen actieve spelers gevonden.")
        return

    completion = fetch_daily_completion(sb, entry_date.isoformat())
    asrm_ids = set(completion.get("asrm_ids", []))
    rpe_ids = set(completion.get("rpe_ids", []))

    render_top_actions(show_back=False)
    show_flash()

    st.markdown(
        f"""
        <div class="tablet-hero">
          <h1>{APP_TITLE}</h1>
          <p>Selecteer een speler voor de invoer van vandaag ({entry_date.strftime("%d-%m-%Y")}).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat_1, stat_2, stat_3 = st.columns(3)
    with stat_1:
        st.metric("Actieve spelers", len(players))
    with stat_2:
        st.metric("Wellness ingevuld", sum(1 for player in players if player["player_id"] in asrm_ids))
    with stat_3:
        st.metric("RPE ingevuld", sum(1 for player in players if player["player_id"] in rpe_ids))

    search = st.text_input("Zoek speler", value="", placeholder="Typ een naam")
    search_value = search.strip().lower()
    filtered_players = [
        player for player in players
        if not search_value or search_value in player["full_name"].lower()
    ]

    if not filtered_players:
        st.info("Geen speler gevonden.")
        return

    cols = st.columns(3)
    for idx, player in enumerate(filtered_players):
        player_id = player["player_id"]
        player_name = player["full_name"]
        wellness_state = "OK" if player_id in asrm_ids else "--"
        rpe_state = "OK" if player_id in rpe_ids else "--"
        label = f"{player_name}\nWellness: {wellness_state} | RPE: {rpe_state}"

        with cols[idx % 3]:
            if st.button(label, use_container_width=True, key=f"tablet_pick_{player_id}"):
                st.session_state["tablet_player_id"] = player_id
                st.session_state["tablet_player_name"] = player_name
                st.rerun()


def _session_value(rpe_sessions: List[Dict[str, Any]], idx: int, key: str, default: int) -> int:
    hit = next((row for row in rpe_sessions if int(row.get("session_index", 0) or 0) == idx), None)
    if not hit:
        return default
    value = hit.get(key)
    return int(value) if value is not None else default


def render_player_forms(sb, player_id: str, player_name: str) -> None:
    entry_date = amsterdam_today()
    existing_asrm = load_asrm(sb, player_id, entry_date) or {}
    rpe_header, rpe_sessions = load_rpe(sb, player_id, entry_date)
    rpe_header = rpe_header or {}
    rpe_sessions = rpe_sessions or []

    has_wellness = bool(existing_asrm)
    has_rpe = bool(rpe_header)
    has_s2 = any(int(row.get("session_index", 0) or 0) == 2 for row in rpe_sessions)
    injury_default = bool(rpe_header.get("injury", False))

    render_top_actions(show_back=True)
    show_flash()

    st.markdown(
        f"""
        <div class="tablet-hero">
          <h1>{player_name}</h1>
          <p>Invoer voor vandaag ({entry_date.strftime("%d-%m-%Y")}).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status_1, status_2 = st.columns(2)
    with status_1:
        st.metric("Wellness", "OK" if has_wellness else "Open")
    with status_2:
        st.metric("RPE", "OK" if has_rpe else "Open")

    tab_asrm, tab_rpe = st.tabs(["Wellness", "RPE"])

    with tab_asrm:
        if has_wellness:
            st.success("Wellness staat al ingevuld voor vandaag.")
        else:
            st.info("Wellness staat nog open voor vandaag.")

        with st.form(f"tablet_asrm_form_{player_id}", clear_on_submit=False):
            _legend_asrm()

            ms = st.slider(
                "Muscle soreness (1-10)",
                1,
                10,
                value=int(existing_asrm.get("muscle_soreness", 5)),
                key=f"tablet_asrm_ms_{player_id}",
            )
            fat = st.slider(
                "Fatigue (1-10)",
                1,
                10,
                value=int(existing_asrm.get("fatigue", 5)),
                key=f"tablet_asrm_fat_{player_id}",
            )
            sleep = st.slider(
                "Sleep quality (1-10)",
                1,
                10,
                value=int(existing_asrm.get("sleep_quality", 5)),
                key=f"tablet_asrm_sleep_{player_id}",
            )
            stress = st.slider(
                "Stress (1-10)",
                1,
                10,
                value=int(existing_asrm.get("stress", 5)),
                key=f"tablet_asrm_stress_{player_id}",
            )
            mood = st.slider(
                "Mood (1-10)",
                1,
                10,
                value=int(existing_asrm.get("mood", 5)),
                key=f"tablet_asrm_mood_{player_id}",
            )

            asrm_submit = st.form_submit_button("Wellness opslaan", use_container_width=True)

        if asrm_submit:
            try:
                save_asrm(sb, player_id, entry_date, ms, fat, sleep, stress, mood)
                clear_daily_cache()
                st.session_state["tablet_flash"] = f"Wellness opgeslagen voor {player_name}."
                st.rerun()
            except Exception as exc:
                st.error(f"Opslaan faalde: {exc}")

    with tab_rpe:
        if has_rpe:
            st.success("RPE staat al ingevuld voor vandaag.")
        else:
            st.info("RPE staat nog open voor vandaag.")

        injury_locations = [
            "None",
            "Foot",
            "Ankle",
            "Lower leg",
            "Knee",
            "Upper leg",
            "Hip",
            "Groin",
            "Glute",
            "Lower back",
            "Abdomen",
            "Chest",
            "Shoulder",
            "Upper arm",
            "Elbow",
            "Forearm",
            "Wrist",
            "Hand",
            "Neck",
            "Head",
            "Other",
        ]

        existing_loc = str(rpe_header.get("injury_type") or "None").strip() or "None"
        if existing_loc not in injury_locations:
            existing_loc = "Other"

        with st.form(f"tablet_rpe_form_{player_id}", clear_on_submit=False):
            _legend_rpe()

            st.markdown("### Session 1")
            s1_dur = st.number_input(
                "[1] Duration (min)",
                min_value=0,
                max_value=600,
                value=_session_value(rpe_sessions, 1, "duration_min", 0),
                key=f"tablet_rpe_s1_dur_{player_id}",
            )
            s1_rpe = st.slider(
                "[1] RPE (1-10)",
                1,
                10,
                value=_session_value(rpe_sessions, 1, "rpe", 5),
                key=f"tablet_rpe_s1_rpe_{player_id}",
            )

            st.divider()

            enable_s2 = st.toggle(
                "Add 2nd session?",
                value=has_s2,
                key=f"tablet_rpe_enable_s2_{player_id}",
            )

            st.markdown("### Session 2")
            s2_dur = st.number_input(
                "[2] Duration (min)",
                min_value=0,
                max_value=600,
                value=_session_value(rpe_sessions, 2, "duration_min", 0),
                key=f"tablet_rpe_s2_dur_{player_id}",
            )
            s2_rpe = st.slider(
                "[2] RPE (1-10)",
                1,
                10,
                value=_session_value(rpe_sessions, 2, "rpe", 5),
                key=f"tablet_rpe_s2_rpe_{player_id}",
            )

            st.divider()
            st.markdown("### Injury")

            injury = st.toggle("Injury?", value=injury_default, key=f"tablet_rpe_injury_{player_id}")

            loc_col, pain_col = st.columns([1.2, 2.0])
            with loc_col:
                injury_loc = st.selectbox(
                    "Location",
                    options=injury_locations,
                    index=injury_locations.index(existing_loc),
                    key=f"tablet_rpe_loc_{player_id}",
                )
            with pain_col:
                injury_pain = st.slider(
                    "Pain (0-10)",
                    0,
                    10,
                    value=int(rpe_header.get("injury_pain", 0) or 0),
                    key=f"tablet_rpe_pain_{player_id}",
                )

            notes = st.text_area(
                "Notes (optional)",
                value=str(rpe_header.get("notes") or ""),
                key=f"tablet_rpe_notes_{player_id}",
            )
            rpe_submit = st.form_submit_button("RPE opslaan", use_container_width=True)

        if rpe_submit:
            try:
                sessions_payload: List[Dict[str, int]] = []

                if int(s1_dur) > 0:
                    sessions_payload.append(
                        {"session_index": 1, "duration_min": int(s1_dur), "rpe": int(s1_rpe)}
                    )

                if bool(enable_s2) and int(s2_dur) > 0:
                    sessions_payload.append(
                        {"session_index": 2, "duration_min": int(s2_dur), "rpe": int(s2_rpe)}
                    )

                injury_type_to_save = None
                injury_pain_to_save = None
                if bool(injury):
                    injury_type_to_save = None if injury_loc == "None" else injury_loc
                    injury_pain_to_save = int(injury_pain)

                save_rpe(
                    sb,
                    player_id=player_id,
                    entry_date=entry_date,
                    injury=bool(injury),
                    injury_type=injury_type_to_save,
                    injury_pain=injury_pain_to_save,
                    notes=notes,
                    sessions=sessions_payload,
                )
                clear_daily_cache()
                st.session_state["tablet_flash"] = f"RPE opgeslagen voor {player_name}."
                st.rerun()
            except Exception as exc:
                st.error(f"Opslaan faalde: {exc}")

    st.divider()

    if has_wellness and has_rpe:
        st.success("Alles staat erin voor vandaag.")
    else:
        st.info("Sla beide onderdelen op en ga daarna terug naar de spelerslijst.")

    if st.button("Klaar / volgende speler", use_container_width=True, key=f"tablet_done_{player_id}"):
        st.session_state.pop("tablet_player_id", None)
        st.session_state.pop("tablet_player_name", None)
        st.rerun()


def main() -> None:
    ensure_tablet_access()

    sb = get_tablet_sb()
    if sb is None:
        st.error(
            "Supabase service key ontbreekt. Voeg SUPABASE_SERVICE_ROLE_KEY toe in Streamlit secrets "
            "voor deze losse tablet-app."
        )
        st.stop()

    selected_player_id = str(st.session_state.get("tablet_player_id") or "").strip()
    selected_player_name = str(st.session_state.get("tablet_player_name") or "").strip()

    if selected_player_id and selected_player_name:
        render_player_forms(sb, selected_player_id, selected_player_name)
    else:
        render_player_picker(sb)


if __name__ == "__main__":
    main()
