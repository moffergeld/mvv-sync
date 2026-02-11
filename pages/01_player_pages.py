# player_pages.py
# ============================================================
# Player pagina: tabs Data + Forms
# ============================================================

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from roles import get_sb, require_auth, get_profile, pick_target_player


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


def load_rpe(sb, player_id: str, entry_date: date) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    header: Optional[Dict[str, Any]] = None
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


def save_asrm(
    sb,
    player_id: str,
    entry_date: date,
    muscle_soreness: int,
    fatigue: int,
    sleep_quality: int,
    stress: int,
    mood: int,
):
    payload = {
        "player_id": player_id,
        "entry_date": entry_date.isoformat(),
        "muscle_soreness": int(muscle_soreness),
        "fatigue": int(fatigue),
        "sleep_quality": int(sleep_quality),
        "stress": int(stress),
        "mood": int(mood),
    }
    sb.table("asrm_entries").upsert(payload, on_conflict="player_id,entry_date").execute()


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

    resp = (
        sb.table("rpe_entries")
        .upsert(header_payload, on_conflict="player_id,entry_date")
        .select("id")
        .execute()
    )
    if not resp.data:
        raise RuntimeError("Kon rpe_entries niet opslaan.")
    rpe_entry_id = resp.data[0]["id"]

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


def player_pages_main():
    require_auth()
    sb = get_sb()
    if sb is None:
        st.error("Supabase client niet beschikbaar. Controleer secrets + supabase package.")
        st.stop()

    profile = get_profile(sb)  # werkt nu, want uid komt uit get_user(token)
    target_player_id, target_player_name, _ = pick_target_player(
        sb, profile, label="Speler", key="pp_player_select"
    )

    if not target_player_id:
        st.error("Geen speler beschikbaar (players leeg of geen toegang).")
        st.stop()

    st.title(f"Player: {target_player_name}")

    tab_data, tab_forms = st.tabs(["Data", "Forms"])

    with tab_data:
        st.info("Data-tab komt later.")

    with tab_forms:
        st.subheader("Forms")
        entry_date = st.date_input("Datum", value=date.today(), key="pp_date")

        col_asrm, col_rpe = st.columns(2)

        with col_asrm:
            st.markdown("### ASRM (Wellbeing)")
            existing = load_asrm(sb, target_player_id, entry_date) or {}

            ms = st.slider("Muscle soreness (1–10)", 1, 10, value=int(existing.get("muscle_soreness", 5)), key="asrm_ms")
            fat = st.slider("Fatigue (1–10)", 1, 10, value=int(existing.get("fatigue", 5)), key="asrm_fat")
            sleep = st.slider("Sleep quality (1–10)", 1, 10, value=int(existing.get("sleep_quality", 5)), key="asrm_sleep")
            stress = st.slider("Stress (1–10)", 1, 10, value=int(existing.get("stress", 5)), key="asrm_stress")
            mood = st.slider("Mood (1–10)", 1, 10, value=int(existing.get("mood", 5)), key="asrm_mood")

            if st.button("ASRM opslaan", use_container_width=True, key="asrm_save"):
                try:
                    save_asrm(sb, target_player_id, entry_date, ms, fat, sleep, stress, mood)
                    st.success("ASRM opgeslagen.")
                except Exception as e:
                    st.error(f"Opslaan faalde: {e}")

        with col_rpe:
            st.markdown("### RPE (Session)")
            header, sessions = load_rpe(sb, target_player_id, entry_date)
            header = header or {}
            sessions = sessions or []

            injury_default = bool(header.get("injury", False))
            injury = st.toggle("Injury?", value=injury_default, key="rpe_injury")

            injury_type = st.text_input(
                "Injury type",
                value=str(header.get("injury_type") or ""),
                disabled=not injury,
                key="rpe_injury_type",
            )
            injury_pain = st.slider(
                "Pain (0–10)",
                0,
                10,
                value=int(header.get("injury_pain", 0) or 0),
                disabled=not injury,
                key="rpe_pain",
            )

            notes = st.text_area("Notes (optioneel)", value=str(header.get("notes") or ""), key="rpe_notes")

            st.markdown("#### Sessions")

            def _sess(idx: int, key: str, default: int) -> int:
                hit = next((s for s in sessions if int(s.get("session_index", 0)) == idx), None)
                if not hit:
                    return default
                v = hit.get(key)
                return int(v) if v is not None else default

            s1_dur = st.number_input("[1] Duration (min)", 0, 600, value=_sess(1, "duration_min", 0), key="rpe_s1_dur")
            s1_rpe = st.slider("[1] RPE (1–10)", 1, 10, value=_sess(1, "rpe", 5), key="rpe_s1_rpe")

            s2_dur = st.number_input("[2] Duration (min)", 0, 600, value=_sess(2, "duration_min", 0), key="rpe_s2_dur")
            s2_rpe = st.slider("[2] RPE (1–10)", 1, 10, value=_sess(2, "rpe", 5), key="rpe_s2_rpe")

            sessions_payload: List[Dict[str, int]] = []
            if s1_dur > 0:
                sessions_payload.append({"session_index": 1, "duration_min": int(s1_dur), "rpe": int(s1_rpe)})
            if s2_dur > 0:
                sessions_payload.append({"session_index": 2, "duration_min": int(s2_dur), "rpe": int(s2_rpe)})

            if st.button("RPE opslaan", use_container_width=True, key="rpe_save"):
                try:
                    save_rpe(
                        sb,
                        player_id=target_player_id,
                        entry_date=entry_date,
                        injury=injury,
                        injury_type=injury_type.strip() or None,
                        injury_pain=int(injury_pain) if injury else None,
                        notes=notes,
                        sessions=sessions_payload,
                    )
                    st.success("RPE opgeslagen.")
                except Exception as e:
                    st.error(f"Opslaan faalde: {e}")


if __name__ == "__main__":
    player_pages_main()
