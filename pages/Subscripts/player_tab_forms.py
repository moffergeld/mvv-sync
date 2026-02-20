# Subscripts/player_tab_forms.py
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


def load_asrm(sb, player_id: str, entry_date: date) -> Optional[Dict[str, Any]]:
    try:
        resp = (
            sb.table("asrm_entries")
            .select("*")
            .eq("player_id", player_id)
            .eq("entry_date", entry_date.isoformat())
            .maybe_single()
            .execute()
        )
        return resp.data
    except Exception:
        return None


def save_asrm(sb, player_id: str, entry_date: date, ms: int, fat: int, sleep: int, stress: int, mood: int):
    payload = {
        "player_id": player_id,
        "entry_date": entry_date.isoformat(),
        "muscle_soreness": int(ms),
        "fatigue": int(fat),
        "sleep_quality": int(sleep),
        "stress": int(stress),
        "mood": int(mood),
    }
    sb.table("asrm_entries").upsert(payload, on_conflict="player_id,entry_date").execute()


def load_rpe(sb, player_id: str, entry_date: date) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    header = None
    sessions: List[Dict[str, Any]] = []
    try:
        r = (
            sb.table("rpe_entries")
            .select("*")
            .eq("player_id", player_id)
            .eq("entry_date", entry_date.isoformat())
            .maybe_single()
            .execute()
        )
        header = r.data
        if header and header.get("id"):
            s = (
                sb.table("rpe_sessions")
                .select("*")
                .eq("rpe_entry_id", header["id"])
                .order("session_index")
                .execute()
            )
            sessions = s.data or []
    except Exception:
        pass
    return header, sessions


def _get_rpe_entry_id(sb, player_id: str, entry_date: date) -> Optional[str]:
    try:
        r = (
            sb.table("rpe_entries")
            .select("id")
            .eq("player_id", player_id)
            .eq("entry_date", entry_date.isoformat())
            .maybe_single()
            .execute()
        )
        if r.data and r.data.get("id"):
            return str(r.data["id"])
    except Exception:
        pass
    return None


def save_rpe(
    sb,
    player_id: str,
    entry_date: date,
    injury: bool,
    injury_type: Optional[str],
    injury_pain: Optional[int],
    notes: str,
    sessions: List[Dict[str, int]],
):
    header_payload = {
        "player_id": player_id,
        "entry_date": entry_date.isoformat(),
        "injury": bool(injury),
        "injury_type": injury_type if injury else None,
        "injury_pain": int(injury_pain) if (injury and injury_pain is not None) else None,
        "notes": notes.strip() if notes else None,
    }

    sb.table("rpe_entries").upsert(header_payload, on_conflict="player_id,entry_date").execute()

    rpe_entry_id = _get_rpe_entry_id(sb, player_id, entry_date)
    if not rpe_entry_id:
        raise RuntimeError("Kon rpe_entry_id niet ophalen na opslaan.")

    payload: List[Dict[str, Any]] = []
    for s in sessions:
        payload.append(
            {
                "rpe_entry_id": rpe_entry_id,
                "session_index": int(s["session_index"]),
                "duration_min": int(s["duration_min"]),
                "rpe": int(s["rpe"]),
            }
        )
    if payload:
        sb.table("rpe_sessions").upsert(payload, on_conflict="rpe_entry_id,session_index").execute()


def render_forms_tab(sb, target_player_id: str):
    st.header("Forms")
    entry_date = st.date_input("Datum", value=date.today(), key="form_date")

    # Alleen bij datum-wissel nieuwe fetch (Streamlit rerunt sowieso, maar dit is minimaal)
    existing_asrm = load_asrm(sb, target_player_id, entry_date) or {}
    rpe_header, rpe_sessions = load_rpe(sb, target_player_id, entry_date)
    rpe_header = rpe_header or {}
    rpe_sessions = rpe_sessions or []

    has_wellness = bool(existing_asrm)
    has_rpe = bool(rpe_header)

    col_asrm, col_rpe = st.columns(2)

    # -----------------
    # ASRM FORM
    # -----------------
    with col_asrm:
        st.subheader("ASRM (1 = best, 10 = worst)")
        st.success("✅ Wellness is al ingevuld voor deze dag.") if has_wellness else st.info(
            "ℹ️ Wellness is nog niet ingevuld voor deze dag."
        )

        with st.form("asrm_form", clear_on_submit=False):
            ms = st.slider("Muscle soreness (1–10)", 1, 10, value=int(existing_asrm.get("muscle_soreness", 5)), key="asrm_ms")
            fat = st.slider("Fatigue (1–10)", 1, 10, value=int(existing_asrm.get("fatigue", 5)), key="asrm_fat")
            sleep = st.slider("Sleep quality (1–10)", 1, 10, value=int(existing_asrm.get("sleep_quality", 5)), key="asrm_sleep")
            stress = st.slider("Stress (1–10)", 1, 10, value=int(existing_asrm.get("stress", 5)), key="asrm_stress")
            mood = st.slider("Mood (1–10)", 1, 10, value=int(existing_asrm.get("mood", 5)), key="asrm_mood")
            asrm_submit = st.form_submit_button("ASRM opslaan", use_container_width=True)

        if asrm_submit:
            try:
                save_asrm(sb, target_player_id, entry_date, ms, fat, sleep, stress, mood)
                st.success("ASRM opgeslagen.")
                st.rerun()
            except Exception as e:
                st.error(f"Opslaan faalde: {e}")

    # -----------------
    # RPE FORM
    # -----------------
    with col_rpe:
        st.subheader("RPE (Session)")
        st.success("✅ RPE is al ingevuld voor deze dag.") if has_rpe else st.info("ℹ️ RPE is nog niet ingevuld voor deze dag.")

        INJURY_LOCATIONS_EN = [
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

        def _sess(idx: int, key: str, default: int) -> int:
            hit = next((s for s in rpe_sessions if int(s.get("session_index", 0) or 0) == idx), None)
            if not hit:
                return default
            v = hit.get(key)
            return int(v) if v is not None else default

        has_s2 = any(int(s.get("session_index", 0) or 0) == 2 for s in rpe_sessions)
        injury_default = bool(rpe_header.get("injury", False))

        existing_loc = str(rpe_header.get("injury_type") or "None").strip() or "None"
        if existing_loc not in INJURY_LOCATIONS_EN:
            existing_loc = "Other"

        with st.form("rpe_form", clear_on_submit=False):
            st.markdown("### Session 1")
            s1_dur = st.number_input("[1] Duration (min)", 0, 600, value=_sess(1, "duration_min", 0), key="rpe_s1_dur")
            s1_rpe = st.slider("[1] RPE (1–10)", 1, 10, value=_sess(1, "rpe", 5), key="rpe_s1_rpe")

            st.divider()

            # onder de lijn
            enable_s2 = st.toggle("Add 2nd session?", value=has_s2, key="rpe_enable_s2")

            st.markdown("### Session 2")
            # inputs staan altijd in de form (zoals je wilde)
            s2_dur = st.number_input("[2] Duration (min)", 0, 600, value=_sess(2, "duration_min", 0), key="rpe_s2_dur")
            s2_rpe = st.slider("[2] RPE (1–10)", 1, 10, value=_sess(2, "rpe", 5), key="rpe_s2_rpe")

            st.divider()
            st.markdown("### Injury")

            injury = st.toggle("Injury?", value=injury_default, key="rpe_injury")

            # dropdown naast pain slider
            loc_col, pain_col = st.columns([1.2, 2.0])
            with loc_col:
                injury_loc = st.selectbox(
                    "Location",
                    options=INJURY_LOCATIONS_EN,
                    index=INJURY_LOCATIONS_EN.index(existing_loc),
                    key="rpe_injury_loc",
                )
            with pain_col:
                injury_pain = st.slider("Pain (0–10)", 0, 10, value=int(rpe_header.get("injury_pain", 0) or 0), key="rpe_pain")

            notes = st.text_area("Notes (optional)", value=str(rpe_header.get("notes") or ""), key="rpe_notes")
            rpe_submit = st.form_submit_button("RPE opslaan", use_container_width=True)

        if rpe_submit:
            try:
                sessions_payload: List[Dict[str, int]] = []
                if int(s1_dur) > 0:
                    sessions_payload.append({"session_index": 1, "duration_min": int(s1_dur), "rpe": int(s1_rpe)})

                # alleen opslaan als toggle aan + duration > 0
                if bool(enable_s2) and int(s2_dur) > 0:
                    sessions_payload.append({"session_index": 2, "duration_min": int(s2_dur), "rpe": int(s2_rpe)})

                injury_type_to_save = None
                injury_pain_to_save = None
                if bool(injury):
                    injury_type_to_save = None if injury_loc == "None" else injury_loc
                    injury_pain_to_save = int(injury_pain)

                save_rpe(
                    sb,
                    player_id=target_player_id,
                    entry_date=entry_date,
                    injury=bool(injury),
                    injury_type=injury_type_to_save,
                    injury_pain=injury_pain_to_save,
                    notes=notes,
                    sessions=sessions_payload,
                )
                st.success("RPE opgeslagen.")
                st.rerun()
            except Exception as e:
                st.error(f"Opslaan faalde: {e}")

