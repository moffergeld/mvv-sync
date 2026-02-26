# gps_import_tab_manual.py
# ============================================================
# Manual add (cards v3)
# - Default metrics = 4 (duration, total_distance, sprint, high_sprint)
# - Load existing records by (player_id + datum + type)  ✅ NOT filtered on event
# - Per player: pick which event-record you want to edit (dropdown from existing events)
# - Day mean/median: compute from current UI values; apply to chosen players; optional source-event filter
# - Remove match_id from UI (still auto-enforced on save for Match/Practice Match)
# - Fix "Vul dag mediaan" per player
# - Keep bulk (works) + copy
# ============================================================

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import requests
import streamlit as st

from pages.Subscripts.gps_import_common import (
    INT_DB_COLS,
    MATCH_TYPES,
    TYPE_OPTIONS,
    fetch_gps_match_ids_on_date,
    json_safe,
    normalize_name,
    resolve_match_id_for_date,
    rest_upsert,
    toast_err,
    toast_ok,
    ui_pick_match_if_needed,
)

# ------------------------------------------------------------
# REST helpers (GET / DELETE)
# ------------------------------------------------------------


def _rest_base() -> str:
    return st.secrets["SUPABASE_URL"].rstrip("/") + "/rest/v1"


def _rest_headers(access_token: str) -> dict[str, str]:
    return {
        "apikey": st.secrets["SUPABASE_ANON_KEY"],
        "Authorization": f"Bearer {access_token}",
    }


def rest_get(access_token: str, table: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    url = f"{_rest_base()}/{table}"
    r = requests.get(url, headers=_rest_headers(access_token), params=params, timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(f"GET {table} failed ({r.status_code}): {r.text}")
    return r.json() or []


def rest_delete(access_token: str, table: str, params: dict[str, Any]) -> int:
    url = f"{_rest_base()}/{table}"
    r = requests.delete(url, headers=_rest_headers(access_token), params=params, timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(f"DELETE {table} failed ({r.status_code}): {r.text}")
    return 1


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

TEMPLATE_COLS = [
    "player_name",
    "datum",
    "type",
    "event",
    "match_id",
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

BASIC_KEYS = ["player_name", "datum", "type", "event", "match_id", "source_file"]
METRIC_KEYS = [c for c in TEMPLATE_COLS if c not in BASIC_KEYS]

# ✅ jouw wens: standaard = 4 metrics
DEFAULT_METRICS = ["duration", "total_distance", "sprint", "high_sprint"]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def _blank_record(player_name: str, d: date, t: str, e: str) -> dict[str, Any]:
    row = {k: None for k in TEMPLATE_COLS}
    row["player_name"] = player_name
    row["datum"] = d
    row["type"] = t
    row["event"] = e
    row["match_id"] = None
    row["source_file"] = "manual"
    return row


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def _parse_metric_input(text: str, metric_name: str) -> tuple[float | int | None, str | None]:
    txt = (text or "").strip()
    if txt == "":
        return None, None
    txt = txt.replace(",", ".")
    try:
        num = float(txt)
    except Exception:
        return None, f"Ongeldige waarde voor {metric_name}: '{text}'"
    if metric_name in INT_DB_COLS:
        if abs(num - round(num)) > 1e-9:
            return None, f"{metric_name} moet een geheel getal zijn."
        return int(round(num)), None
    return float(num), None


def _format_metric_value(v: Any, metric_name: str) -> str:
    if _is_missing(v):
        return ""
    try:
        if metric_name in INT_DB_COLS:
            return str(int(float(v)))
        fv = float(v)
        s = f"{fv:.2f}".rstrip("0").rstrip(".")
        return s
    except Exception:
        return str(v)


def _coerce_number(v: Any) -> float | int | None:
    if _is_missing(v):
        return None
    try:
        fv = float(v)
        if pd.isna(fv):
            return None
        return fv
    except Exception:
        return None


def _fetch_existing_for_day_type(
    access_token: str,
    player_ids: list[str],
    d_iso: str,
    t: str,
) -> list[dict[str, Any]]:
    """✅ Load NOT filtered on event: returns all events for players on (date,type)."""
    if not player_ids:
        return []
    pid_in = "in.(" + ",".join([str(x) for x in player_ids]) + ")"
    params = {
        "select": ",".join(
            ["player_id", "player_name", "datum", "type", "event", "match_id", "source_file"] + METRIC_KEYS
        ),
        "player_id": pid_in,
        "datum": f"eq.{d_iso}",
        "type": f"eq.{t}",
        "limit": "5000",
    }
    return rest_get(access_token, "gps_records", params)


def _existing_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """pid -> event -> record"""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        pid = str(r.get("player_id") or "")
        ev = str(r.get("event") or "").strip()
        if not pid or not ev:
            continue
        out.setdefault(pid, {})[ev] = r
    return out


def _get_current_row_from_ui(pid: str, nm: str, d_iso: str, t: str, ev: str, metrics: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Parse current widget values (source of truth) into a row dict for stats/save."""
    row: dict[str, Any] = {
        "player_id": pid,
        "player_name": nm,
        "datum": d_iso,
        "type": t,
        "event": ev,
        "source_file": "manual",
    }
    errs: list[str] = []
    for m in metrics:
        txt = st.session_state.get(f"manual_txt_{pid}_{m}", "")
        parsed, err = _parse_metric_input(txt, m)
        if err:
            errs.append(f"{nm} → {err}")
        row[m] = parsed
    return row, errs


def _compute_day_stat_from_ui(pids: list[str], pid_to_name: dict[str, str], d_iso: str, t: str, metrics: list[str], source_event: str | None) -> tuple[dict[str, float | None], list[str]]:
    """Compute mean/median using CURRENT UI values; optionally filter by event."""
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for pid in pids:
        rec = st.session_state["manual_cards"].get(pid) or {}
        ev = str(rec.get("event") or "").strip()
        if source_event and source_event != "(alle events)" and ev != source_event:
            continue
        r, errs = _get_current_row_from_ui(pid, pid_to_name.get(pid, pid), d_iso, t, ev, metrics)
        errors.extend(errs)
        rows.append(r)

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    out: dict[str, float | None] = {}
    if df.empty:
        for m in metrics:
            out[m] = None
        return out, errors

    for m in metrics:
        s = pd.to_numeric(df.get(m), errors="coerce").dropna() if m in df.columns else pd.Series([], dtype=float)
        if s.empty:
            out[m] = None
        else:
            how = st.session_state.get("manual_day_stat_how_sidebar", "median")
            out[m] = float(s.median()) if how == "median" else float(s.mean())
    return out, errors


def _apply_values_to_players(
    target_pids: list[str],
    values: dict[str, Any],
    metrics: list[str],
    only_fill_missing: bool,
) -> None:
    """Apply to cards + widget texts."""
    cards = st.session_state["manual_cards"]
    for pid in target_pids:
        rec = cards.get(pid) or {}
        for m in metrics:
            if m not in values:
                continue
            if only_fill_missing:
                cur_txt = st.session_state.get(f"manual_txt_{pid}_{m}", "")
                if (cur_txt or "").strip() != "":
                    continue
            rec[m] = values[m]
            st.session_state[f"manual_txt_{pid}_{m}"] = _format_metric_value(values[m], m)
        cards[pid] = rec
    st.session_state["manual_cards"] = cards


# ------------------------------------------------------------
# Sidebar tools
# ------------------------------------------------------------


def _render_sidebar_tools(selected_pids: list[str], pid_to_name: dict[str, str], d_iso: str, t: str) -> None:
    with st.sidebar:
        st.markdown("## Manual tools")

        # Metrics
        st.markdown("### Metrics tonen")
        default = st.session_state.get("manual_selected_metrics", DEFAULT_METRICS)
        st.session_state["manual_selected_metrics"] = st.multiselect(
            "Kies metrics voor spelerkaarten",
            options=METRIC_KEYS,
            default=default,
            key="manual_metrics_picker_sidebar",
        )
        if st.button("Reset metrics naar standaard", key="manual_reset_metrics_sidebar"):
            st.session_state["manual_selected_metrics"] = DEFAULT_METRICS.copy()
            st.rerun()

        st.divider()

        tabs = st.tabs(["Dag-stat", "Bulk", "Kopie"])

        metrics = st.session_state.get("manual_selected_metrics", []) or DEFAULT_METRICS

        # ------------------ Dag-stat ------------------
        with tabs[0]:
            st.caption("Bereken mediaan/gemiddelde op basis van selectie en wijs toe aan spelers.")

            if not selected_pids:
                st.info("Selecteer eerst spelers.")
            else:
                # source event filter (optional)
                # collect known events from loaded existing
                known_events = set()
                existing_by_pid = st.session_state.get("manual_existing_by_pid", {})
                for pid in selected_pids:
                    for ev in (existing_by_pid.get(pid, {}) or {}).keys():
                        known_events.add(ev)
                known_events = sorted([e for e in known_events if e])

                source_event = st.selectbox(
                    "Bron-event voor berekening",
                    options=["(alle events)"] + known_events,
                    index=0,
                    key="manual_stat_source_event",
                )

                st.selectbox("Dag statistiek", ["median", "mean"], key="manual_day_stat_how_sidebar")
                fill_mode = st.selectbox(
                    "Invullen",
                    ["alleen lege velden", "alles overschrijven"],
                    key="manual_day_fill_mode_sidebar",
                )

                # target selection
                target_default = [pid for pid in selected_pids if st.session_state.get(f"manual_inc_{pid}", True)]
                target_pids = st.multiselect(
                    "Toewijzen aan (spelers)",
                    options=selected_pids,
                    default=target_default if target_default else selected_pids,
                    format_func=lambda pid: pid_to_name.get(pid, pid),
                    key="manual_stat_target_players",
                )

                if st.button("Bereken + toepassen", type="primary", key="manual_apply_day_stat_btn_sidebar"):
                    stat_vals, errs = _compute_day_stat_from_ui(
                        pids=selected_pids,
                        pid_to_name=pid_to_name,
                        d_iso=d_iso,
                        t=t,
                        metrics=metrics,
                        source_event=source_event,
                    )
                    if errs:
                        toast_err(errs[0])
                        if len(errs) > 1:
                            toast_err(f"Nog {len(errs)-1} fout(en).")
                        st.stop()

                    vals = {k: v for k, v in stat_vals.items() if v is not None}
                    if not vals:
                        toast_err("Geen waarden beschikbaar om toe te wijzen (alles leeg).")
                        st.stop()

                    only_missing = fill_mode == "alleen lege velden"
                    _apply_values_to_players(target_pids, vals, metrics, only_fill_missing=only_missing)
                    toast_ok("Dag-stat toegepast.")
                    st.rerun()

        # ------------------ Bulk ------------------
        with tabs[1]:
            st.caption("Leeg laten = niet toepassen.")
            if not selected_pids:
                st.info("Selecteer eerst spelers.")
            else:
                st.toggle("Alleen lege velden vullen", value=True, key="manual_bulk_only_missing_sidebar")
                for m in metrics:
                    st.text_input(
                        m,
                        value=st.session_state.get(f"manual_bulk_txt_{m}", ""),
                        key=f"manual_bulk_txt_{m}",
                        placeholder="Leeg = overslaan",
                    )

                if st.button("Toepassen op geselecteerde spelers", type="primary", key="manual_bulk_apply_btn_sidebar"):
                    include = [pid for pid in selected_pids if st.session_state.get(f"manual_inc_{pid}", True)]
                    if not include:
                        toast_err("Geen spelers geselecteerd in cards (Meenemen).")
                        st.stop()

                    vals: dict[str, Any] = {}
                    errs: list[str] = []
                    for m in metrics:
                        txt = st.session_state.get(f"manual_bulk_txt_{m}", "")
                        parsed, err = _parse_metric_input(txt, m)
                        if err:
                            errs.append(err)
                        elif parsed is not None:
                            vals[m] = parsed

                    if errs:
                        toast_err(errs[0])
                        st.stop()
                    if not vals:
                        toast_err("Geen bulkwaarden ingevuld.")
                        st.stop()

                    _apply_values_to_players(
                        include,
                        vals,
                        metrics,
                        only_fill_missing=st.session_state.get("manual_bulk_only_missing_sidebar", True),
                    )
                    toast_ok(f"Bulk toegepast op {len(include)} spelers.")
                    st.rerun()

        # ------------------ Kopie ------------------
        with tabs[2]:
            if not selected_pids:
                st.info("Selecteer eerst spelers.")
            else:
                cards = st.session_state["manual_cards"]
                src_pid = st.selectbox(
                    "Bronspeler",
                    options=selected_pids,
                    format_func=lambda pid: pid_to_name.get(pid, pid),
                    key="manual_copy_src_sidebar",
                )
                dst_pids = st.multiselect(
                    "Doelspelers",
                    options=[p for p in selected_pids if p != src_pid],
                    default=[p for p in selected_pids if p != src_pid],
                    format_func=lambda pid: pid_to_name.get(pid, pid),
                    key="manual_copy_dst_sidebar",
                )
                st.toggle("Alleen lege velden vullen", value=False, key="manual_copy_only_missing_sidebar")

                if st.button("Kopieer waarden", type="primary", key="manual_copy_btn_sidebar"):
                    src = cards.get(src_pid) or {}
                    vals = {}
                    for m in metrics:
                        # use current UI if available
                        txt = st.session_state.get(f"manual_txt_{src_pid}_{m}", "")
                        parsed, err = _parse_metric_input(txt, m)
                        if err:
                            toast_err(f"{pid_to_name.get(src_pid, src_pid)} → {err}")
                            st.stop()
                        if parsed is not None:
                            vals[m] = parsed
                    if not vals:
                        toast_err("Bronspeler heeft geen ingevulde waarden in gekozen metrics.")
                        st.stop()

                    _apply_values_to_players(
                        dst_pids,
                        vals,
                        metrics,
                        only_fill_missing=st.session_state.get("manual_copy_only_missing_sidebar", False),
                    )
                    toast_ok("Kopieeractie uitgevoerd.")
                    st.rerun()

        st.divider()
        st.caption("Tip: vink 'Meenemen' uit bij spelers die je tijdelijk niet wilt opslaan.")


# ------------------------------------------------------------
# Player card
# ------------------------------------------------------------


def _render_player_card(
    access_token: str,
    pid: str,
    nm: str,
    d: date,
    d_iso: str,
    t: str,
    default_event: str,
    metrics: list[str],
) -> None:
    cards = st.session_state["manual_cards"]
    existing_by_pid = st.session_state.get("manual_existing_by_pid", {})
    existing_events = sorted([e for e in (existing_by_pid.get(pid, {}) or {}).keys() if e])

    rec = cards.get(pid) or _blank_record(nm, d, t, default_event)
    # ensure event exists
    cur_event = str(rec.get("event") or default_event).strip() or "Summary"
    rec["event"] = cur_event

    exists_flag = (cur_event in (existing_by_pid.get(pid, {}) or {}))

    label = f"{nm}  •  {'Bestaand' if exists_flag else 'Nieuw'}"
    expanded_default = len(st.session_state.get("manual_selected_players", [])) <= 2

    with st.expander(label, expanded=expanded_default):
        with st.container(border=True):
            h1, h2, h3, h4 = st.columns([2.3, 1.1, 1.0, 1.1], vertical_alignment="center")
            with h1:
                st.markdown(f"**{nm}**")
                st.caption(f"player_id: {pid}")
            with h2:
                st.toggle("Meenemen", value=st.session_state.get(f"manual_inc_{pid}", True), key=f"manual_inc_{pid}")
            with h3:
                st.write("Bestaand" if exists_flag else "Nieuw")
            with h4:
                if st.button("Verwijder", key=f"manual_delete_{pid}"):
                    try:
                        # delete only this specific event record (player+day+type+event)
                        params = {
                            "player_id": f"eq.{pid}",
                            "datum": f"eq.{d_iso}",
                            "type": f"eq.{t}",
                            "event": f"eq.{cur_event}",
                        }
                        rest_delete(access_token, "gps_records", params)

                        # remove from existing map if present
                        if pid in existing_by_pid and cur_event in existing_by_pid[pid]:
                            del existing_by_pid[pid][cur_event]
                            st.session_state["manual_existing_by_pid"] = existing_by_pid

                        # reset card to blank but keep event
                        cards[pid] = _blank_record(nm, d, t, cur_event)
                        st.session_state["manual_cards"] = cards
                        for m in metrics:
                            st.session_state[f"manual_txt_{pid}_{m}"] = ""
                        toast_ok("Record verwijderd.")
                        st.rerun()
                    except Exception as ex:
                        toast_err(str(ex))

            # Event picker (✅ per speler, omdat load niet op event filtert)
            ev1, ev2, ev3 = st.columns([1.6, 1.2, 2.2], vertical_alignment="bottom")
            with ev1:
                ev_options = ["(nieuw)"] + existing_events if existing_events else ["(nieuw)"]
                # if current event is one of existing, preselect it; else "(nieuw)"
                ev_idx = 0
                if cur_event in existing_events:
                    ev_idx = ev_options.index(cur_event)
                ev_pick = st.selectbox(
                    "Event (record kiezen)",
                    options=ev_options,
                    index=ev_idx,
                    key=f"manual_event_pick_{pid}",
                )

            with ev2:
                # default for new
                new_ev = st.text_input(
                    "Nieuwe event naam",
                    value=cur_event if (ev_pick == "(nieuw)") else "",
                    key=f"manual_event_new_{pid}",
                    placeholder="bv. Summary / First Half / ...",
                )

            with ev3:
                st.caption("match_id wordt automatisch gezet bij Match/Practice Match (niet zichtbaar).")

            # If event changed: load its record into card + widgets
            chosen_event = cur_event
            if ev_pick != "(nieuw)":
                chosen_event = ev_pick
            else:
                chosen_event = (new_ev or "").strip() or default_event or "Summary"

            if chosen_event != cur_event:
                # switch card to existing record or blank
                rec2 = _blank_record(nm, d, t, chosen_event)
                if chosen_event in (existing_by_pid.get(pid, {}) or {}):
                    rec2.update(existing_by_pid[pid][chosen_event])
                rec2["player_name"] = nm
                rec2["datum"] = d
                rec2["type"] = t
                rec2["event"] = chosen_event
                rec2["source_file"] = "manual"
                cards[pid] = rec2
                st.session_state["manual_cards"] = cards
                for m in metrics:
                    st.session_state[f"manual_txt_{pid}_{m}"] = _format_metric_value(rec2.get(m), m)
                st.rerun()

            # Quick actions
            q1, q2, q3 = st.columns([1.2, 1.2, 2.6], vertical_alignment="center")
            with q1:
                if st.button("Maak metrics leeg", key=f"manual_clear_metrics_{pid}"):
                    for m in metrics:
                        rec[m] = None
                        st.session_state[f"manual_txt_{pid}_{m}"] = ""
                    cards[pid] = rec
                    st.session_state["manual_cards"] = cards
                    st.rerun()

            with q2:
                if st.button("Vul dag mediaan", key=f"manual_fill_median_{pid}"):
                    # ✅ compute median from UI (optionally same event as this card)
                    source_event = chosen_event  # per jouw wens: median ook op event kunnen invullen
                    st.session_state["manual_day_stat_how_sidebar"] = "median"
                    vals, errs = _compute_day_stat_from_ui(
                        pids=st.session_state.get("manual_selected_pid_list", []),
                        pid_to_name=st.session_state.get("manual_pid_to_name", {}),
                        d_iso=d_iso,
                        t=t,
                        metrics=metrics,
                        source_event=source_event,
                    )
                    if errs:
                        toast_err(errs[0])
                        if len(errs) > 1:
                            toast_err(f"Nog {len(errs)-1} fout(en).")
                        st.stop()
                    fill_vals = {k: v for k, v in vals.items() if v is not None}
                    if not fill_vals:
                        toast_err("Geen medianen beschikbaar (alles leeg voor deze event).")
                        st.stop()
                    _apply_values_to_players([pid], fill_vals, metrics, only_fill_missing=False)
                    toast_ok("Dag mediaan ingevuld.")
                    st.rerun()

            with q3:
                st.caption("Leeg laten = leeg opslaan. Komma of punt mag voor decimalen.")

            # Metrics grid
            num_cols = 2 if len(metrics) > 8 else 4
            grid = st.columns(num_cols, vertical_alignment="bottom")

            for i, m in enumerate(metrics):
                with grid[i % num_cols]:
                    key_txt = f"manual_txt_{pid}_{m}"
                    if key_txt not in st.session_state:
                        st.session_state[key_txt] = _format_metric_value(rec.get(m), m)
                    txt = st.text_input(m, value=st.session_state[key_txt], key=key_txt, placeholder="leeg")
                    parsed, err = _parse_metric_input(txt, m)
                    if err:
                        st.caption(f"⚠️ {err}")
                    else:
                        rec[m] = parsed

            rec["player_name"] = nm
            rec["datum"] = d
            rec["type"] = t
            rec["event"] = chosen_event
            rec["source_file"] = "manual"
            cards[pid] = rec
            st.session_state["manual_cards"] = cards


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------


def tab_manual_add_main(access_token: str, name_to_id: dict, player_options: list[str]) -> None:
    st.subheader("Manual add")
    st.caption(
        "Laadt alle records per speler op die datum + type (ongeacht event). "
        "Je kiest per speler welk event-record je bewerkt."
    )

    # State init
    if "manual_cards" not in st.session_state:
        st.session_state["manual_cards"] = {}
    if "manual_existing_by_pid" not in st.session_state:
        st.session_state["manual_existing_by_pid"] = {}
    if "manual_selected_metrics" not in st.session_state:
        st.session_state["manual_selected_metrics"] = DEFAULT_METRICS.copy()
    if "manual_selected_players" not in st.session_state:
        st.session_state["manual_selected_players"] = []
    if "manual_selected_pid_list" not in st.session_state:
        st.session_state["manual_selected_pid_list"] = []
    if "manual_pid_to_name" not in st.session_state:
        st.session_state["manual_pid_to_name"] = {}

    # 1) Basisselectie (✅ event is alleen default; load filtert niet op event)
    st.markdown("### 1) Basisselectie")
    c1, c2, c3, c4 = st.columns([1.2, 1.0, 1.3, 1.1], vertical_alignment="bottom")
    with c1:
        d = st.date_input("Datum", value=st.session_state.get("manual_day", date.today()), key="manual_day")
    with c2:
        default_type_idx = TYPE_OPTIONS.index("Practice") if "Practice" in TYPE_OPTIONS else 0
        t = st.selectbox("Type", options=TYPE_OPTIONS, index=default_type_idx, key="manual_type")
    with c3:
        default_event = st.text_input("Default event (voor nieuwe records)", value="Summary", key="manual_default_event")
    with c4:
        load_clicked = st.button("Laad / ververs", type="primary", key="manual_load_existing")

    c5, c6 = st.columns([5, 1], vertical_alignment="bottom")
    with c5:
        players = st.multiselect(
            "Spelers",
            options=player_options,
            default=st.session_state["manual_selected_players"],
            key="manual_players",
        )
        st.session_state["manual_selected_players"] = players
    with c6:
        st.write("")  # spacer

    d_iso = pd.to_datetime(d).date().isoformat()
    t = str(t or "").strip()

    # Map selected players -> ids
    selected_pids: list[str] = []
    pid_to_name: dict[str, str] = {}
    for nm in players:
        pid = name_to_id.get(normalize_name(nm))
        if pid is not None:
            pid_s = str(pid)
            selected_pids.append(pid_s)
            pid_to_name[pid_s] = nm

    st.session_state["manual_selected_pid_list"] = selected_pids
    st.session_state["manual_pid_to_name"] = pid_to_name

    # Sidebar tools needs d_iso/t
    _render_sidebar_tools(selected_pids, pid_to_name, d_iso, t)

    # Ensure cards exist
    for pid in selected_pids:
        if pid not in st.session_state["manual_cards"]:
            st.session_state["manual_cards"][pid] = _blank_record(pid_to_name.get(pid, ""), d, t, default_event)

    # Load existing by (date,type) regardless of event
    if load_clicked:
        if not selected_pids:
            toast_err("Selecteer eerst minimaal één speler.")
            return
        try:
            rows = _fetch_existing_for_day_type(access_token, selected_pids, d_iso, t)
            mp = _existing_map(rows)
            st.session_state["manual_existing_by_pid"] = mp

            # For each player: choose sensible default event
            cards = st.session_state["manual_cards"]
            for pid in selected_pids:
                nm = pid_to_name.get(pid, pid)
                evs = sorted(list((mp.get(pid, {}) or {}).keys()))
                if evs:
                    # pick Summary if exists else first
                    ev = "Summary" if "Summary" in evs else evs[0]
                    rec = _blank_record(nm, d, t, ev)
                    rec.update(mp[pid][ev])
                    rec["player_name"] = nm
                    rec["datum"] = d
                    rec["type"] = t
                    rec["event"] = ev
                    rec["source_file"] = "manual"
                    cards[pid] = rec
                else:
                    cards[pid] = _blank_record(nm, d, t, default_event or "Summary")

                # sync widgets
                for m in st.session_state.get("manual_selected_metrics", DEFAULT_METRICS):
                    st.session_state[f"manual_txt_{pid}_{m}"] = _format_metric_value(cards[pid].get(m), m)

            st.session_state["manual_cards"] = cards
            toast_ok("Data geladen (ongeacht event).")
            st.rerun()

        except Exception as ex:
            toast_err(str(ex))
            return

    # 2) Invoer per speler
    st.markdown("### 2) Invoer per speler")
    if not selected_pids:
        st.info("Selecteer spelers en klik op **Laad / ververs**.")
        return

    metrics = st.session_state.get("manual_selected_metrics", DEFAULT_METRICS) or DEFAULT_METRICS

    s1, s2, s3 = st.columns(3)
    s1.metric("Geselecteerde spelers", len(selected_pids))
    s2.metric("Meenemen bij save", sum(1 for pid in selected_pids if st.session_state.get(f"manual_inc_{pid}", True)))
    s3.metric("Getoonde metrics", len(metrics))

    st.markdown("#### Snelle acties")
    qa1, qa2, qa3 = st.columns([1.2, 1.2, 2.6], vertical_alignment="bottom")
    with qa1:
        if st.button("Select all meenemen", key="manual_select_all_include"):
            for pid in selected_pids:
                st.session_state[f"manual_inc_{pid}"] = True
            st.rerun()
    with qa2:
        if st.button("Deselect all meenemen", key="manual_deselect_all_include"):
            for pid in selected_pids:
                st.session_state[f"manual_inc_{pid}"] = False
            st.rerun()
    with qa3:
        search_txt = st.text_input("Zoek speler", value=st.session_state.get("manual_search_player", ""), key="manual_search_player")

    q = (search_txt or "").strip().lower()
    visible_pids = [pid for pid in selected_pids if (not q) or (q in pid_to_name.get(pid, pid).lower())]
    if not visible_pids:
        st.info("Geen spelers gevonden met deze zoekterm.")
        return

    left, right = st.columns(2, vertical_alignment="top")
    for i, pid in enumerate(visible_pids):
        target = left if i % 2 == 0 else right
        with target:
            _render_player_card(
                access_token=access_token,
                pid=pid,
                nm=pid_to_name.get(pid, pid),
                d=d,
                d_iso=d_iso,
                t=t,
                default_event=default_event or "Summary",
                metrics=metrics,
            )

    # 3) Acties / Save
    st.divider()
    st.markdown("### 3) Acties")

    a1, a2, a3 = st.columns([1.2, 1.2, 2.6], vertical_alignment="bottom")
    with a1:
        if st.button("Reset selectie (leegmaken)", key="manual_reset_cards_btn"):
            cards = st.session_state["manual_cards"]
            for pid in selected_pids:
                nm = pid_to_name.get(pid, pid)
                ev = str(cards.get(pid, {}).get("event") or default_event or "Summary")
                cards[pid] = _blank_record(nm, d, t, ev)
                for m in metrics:
                    st.session_state[f"manual_txt_{pid}_{m}"] = ""
            st.session_state["manual_cards"] = cards
            toast_ok("Reset bevestigd.")
            st.rerun()

    with a2:
        save_clicked = st.button("Save (upsert)", type="primary", key="manual_save_cards_btn")

    with a3:
        st.caption("match_id wordt automatisch gezet bij Match/Practice Match.")

    if not save_clicked:
        return

    # Save
    try:
        include_pids = [pid for pid in selected_pids if st.session_state.get(f"manual_inc_{pid}", True)]
        if not include_pids:
            toast_err("Geen spelers geselecteerd om op te slaan (Meenemen).")
            return

        df_rows: list[dict[str, Any]] = []
        all_errors: list[str] = []

        cards = st.session_state["manual_cards"]

        for pid in include_pids:
            rec = (cards.get(pid) or {}).copy()
            nm = pid_to_name.get(pid, pid)
            ev = str(rec.get("event") or default_event or "Summary").strip() or "Summary"

            # parse visible metric widgets
            for m in metrics:
                txt = st.session_state.get(f"manual_txt_{pid}_{m}", "")
                parsed, err = _parse_metric_input(txt, m)
                if err:
                    all_errors.append(f"{nm} → {err}")
                rec[m] = parsed

            rec["player_id"] = pid
            rec["player_name"] = str(rec.get("player_name") or nm).strip()
            rec["datum"] = d_iso
            rec["type"] = t
            rec["event"] = ev
            rec["source_file"] = "manual"

            dt = pd.to_datetime(rec["datum"], errors="coerce")
            if pd.isna(dt):
                all_errors.append(f"Ongeldige datum voor {rec['player_name']}")
                continue

            rec["week"] = int(dt.isocalendar().week)
            rec["year"] = int(dt.year)

            # coerce all metrics (incl. not-visible)
            for m in METRIC_KEYS:
                rec[m] = _coerce_number(rec.get(m))
                if m in INT_DB_COLS and rec[m] is not None:
                    rec[m] = int(float(rec[m]))

            df_rows.append(rec)

        if all_errors:
            toast_err(all_errors[0])
            if len(all_errors) > 1:
                toast_err(f"Nog {len(all_errors)-1} fout(en).")
            return

        if not df_rows:
            toast_err("Geen geldige rijen om op te slaan.")
            return

        dfm = pd.DataFrame(df_rows)

        # enforce match_id per (datum,type)
        for (d0, t0), g in dfm.groupby(["datum", "type"], dropna=False):
            if t0 not in MATCH_TYPES:
                dfm.loc[g.index, "match_id"] = None
                continue

            d_obj = pd.to_datetime(d0).date()
            existing_ids = fetch_gps_match_ids_on_date(access_token, d_obj, t0)

            if not existing_ids.empty:
                forced_id = int(existing_ids.value_counts().idxmax())
            else:
                forced_id, _ = resolve_match_id_for_date(access_token, d_obj, t0)
                if forced_id is None:
                    with st.expander(f"Match koppeling nodig voor {d0} ({t0})", expanded=True):
                        picked = ui_pick_match_if_needed(access_token, d_obj, t0, key_prefix=f"manual_cards_match_{d0}_{t0}")
                    if picked is None:
                        toast_err(f"Geen match_id beschikbaar voor {d0} ({t0}). Voeg match toe of kies match.")
                        return
                    forced_id = int(picked)

            dfm.loc[g.index, "match_id"] = forced_id

        # payload
        rows_for_save: list[dict[str, Any]] = []
        for _, r in dfm.iterrows():
            row = {
                "player_id": json_safe(r.get("player_id")),
                "player_name": json_safe(r.get("player_name")),
                "datum": json_safe(r.get("datum")),
                "week": json_safe(int(r.get("week")) if pd.notna(r.get("week")) else None),
                "year": json_safe(int(r.get("year")) if pd.notna(r.get("year")) else None),
                "type": json_safe(r.get("type")),
                "event": json_safe(r.get("event")),
                "match_id": json_safe(int(r.get("match_id")) if pd.notna(r.get("match_id")) else None),
                "source_file": json_safe(r.get("source_file") or "manual"),
            }

            for k in METRIC_KEYS:
                v = r.get(k)
                if k in INT_DB_COLS:
                    vv = pd.to_numeric(v, errors="coerce")
                    row[k] = json_safe(int(vv) if pd.notna(vv) else None)
                else:
                    vv = pd.to_numeric(v, errors="coerce")
                    row[k] = json_safe(float(vv) if pd.notna(vv) else None)

            rows_for_save.append(row)

        # If your DB constraint is on player_id, prefer this:
        # rest_upsert(access_token, "gps_records", rows_for_save, on_conflict="player_id,datum,type,event")
        rest_upsert(access_token, "gps_records", rows_for_save, on_conflict="player_name,datum,type,event")

        toast_ok(f"Save bevestigd: rows = {len(rows_for_save)}")
        st.rerun()

    except Exception as ex:
        toast_err(f"Save fout: {ex}")
