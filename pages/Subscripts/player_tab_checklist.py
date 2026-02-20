# Subscripts/player_tab_checklist.py
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


def _df_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows or [])


def _fmt_dt_hhmm(v: Any) -> str:
    if v is None or v == "":
        return ""
    try:
        dt = pd.to_datetime(v, utc=True).tz_convert("Europe/Amsterdam")
        return dt.strftime("%d-%m-%Y %H:%M")
    except Exception:
        try:
            dt = pd.to_datetime(v)
            return dt.strftime("%d-%m-%Y %H:%M")
        except Exception:
            return str(v)


def fetch_active_players(sb) -> pd.DataFrame:
    try:
        rows = (
            sb.table("players")
            .select("player_id,full_name,is_active")
            .eq("is_active", True)
            .order("full_name")
            .execute()
            .data
            or []
        )
        return _df_from_rows(rows)
    except Exception:
        return pd.DataFrame(columns=["player_id", "full_name", "is_active"])


def _fetch_asrm_filled_players(sb, d: date) -> Dict[str, str]:
    """
    Returns: {player_id: "dd-mm-YYYY HH:MM"} gebaseerd op asrm_entries.created_at
    """
    try:
        rows = (
            sb.table("asrm_entries")
            .select("player_id,created_at")
            .eq("entry_date", d.isoformat())
            .execute()
            .data
            or []
        )
    except Exception:
        rows = []
        try:
            rows = (
                sb.table("asrm_entries")
                .select("player_id")
                .eq("entry_date", d.isoformat())
                .execute()
                .data
                or []
            )
        except Exception:
            rows = []

    out: Dict[str, str] = {}
    for r in rows:
        pid = r.get("player_id")
        if not pid:
            continue
        out[str(pid)] = _fmt_dt_hhmm(r.get("created_at"))
    return out


def _fetch_rpe_filled_players(sb, d: date) -> Dict[str, str]:
    """
    Returns: {player_id: "dd-mm-YYYY HH:MM"} gebaseerd op LAATSTE rpe_sessions.created_at van die dag.
    """
    try:
        headers = (
            sb.table("rpe_entries")
            .select("id,player_id")
            .eq("entry_date", d.isoformat())
            .execute()
            .data
            or []
        )
    except Exception:
        return {}

    if not headers:
        return {}

    id_to_player = {str(h["id"]): str(h["player_id"]) for h in headers if h.get("id") and h.get("player_id")}
    entry_ids = list(id_to_player.keys())
    if not entry_ids:
        return {}

    try:
        sess_rows = (
            sb.table("rpe_sessions")
            .select("rpe_entry_id,created_at")
            .in_("rpe_entry_id", entry_ids)
            .execute()
            .data
            or []
        )
    except Exception:
        try:
            sess_rows = (
                sb.table("rpe_sessions")
                .select("rpe_entry_id")
                .in_("rpe_entry_id", entry_ids)
                .execute()
                .data
                or []
            )
        except Exception:
            return {}

    latest: Dict[str, Any] = {}
    for r in sess_rows:
        eid = r.get("rpe_entry_id")
        if not eid:
            continue
        pid = id_to_player.get(str(eid))
        if not pid:
            continue
        ts = r.get("created_at")
        if ts is None:
            latest.setdefault(pid, None)
            continue
        try:
            dt = pd.to_datetime(ts)
        except Exception:
            dt = None

        if pid not in latest or (dt is not None and latest[pid] is not None and dt > latest[pid]) or (
            dt is not None and latest[pid] is None
        ):
            latest[pid] = dt

    out: Dict[str, str] = {}
    for pid, dt in latest.items():
        out[pid] = _fmt_dt_hhmm(dt) if dt is not None else ""
    return out


def build_checklist_table(sb, d: date) -> pd.DataFrame:
    players = fetch_active_players(sb)
    if players.empty:
        return pd.DataFrame(columns=["Player", "Wellness", "Wellness time", "RPE", "RPE time"])

    asrm_map = _fetch_asrm_filled_players(sb, d)
    rpe_map = _fetch_rpe_filled_players(sb, d)

    rows: List[Dict[str, Any]] = []
    for _, p in players.iterrows():
        pid = str(p["player_id"])
        pname = str(p["full_name"])

        a_time = asrm_map.get(pid, "")
        r_time = rpe_map.get(pid, "")

        rows.append(
            {
                "Player": pname,
                "Wellness": "✅" if pid in asrm_map else "❌",
                "Wellness time": a_time,
                "RPE": "✅" if pid in rpe_map else "❌",
                "RPE time": r_time,
            }
        )

    return pd.DataFrame(rows)


def render_checklist_tab(sb):
    st.header("Checklist")
    d = st.date_input("Datum (Checklist)", value=date.today(), key="checklist_date")

    df = build_checklist_table(sb, d)
    if df.empty:
        st.info("Geen actieve spelers gevonden, of geen data.")
        return

    only_missing = st.toggle("Toon alleen ontbrekend", value=False, key="checklist_only_missing")
    if only_missing:
        df = df[(df["Wellness"] == "❌") | (df["RPE"] == "❌")].copy()

    st.dataframe(df, use_container_width=True, hide_index=True)

