# gps_import_tab_manual.py
# ============================================================
# Subtab: Manual add (cards v2 - cleaner UX)
# - Compact basisselectie
# - Tools in sidebar (metrics / dag-stat / bulk / copy)
# - Player cards in 2 columns
# - Existing records prefill by speler+datum+type+event
# - Delete per speler record
# - Bulk apply / copy values / dag mean-median
# - "Echte lege velden" via text_input (i.p.v. forced 0)
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

DEFAULT_METRICS = [
    "duration",
    "total_distance",
    "sprint",
    "high_sprint",
    "number_of_sprints",
    "max_speed",
    "avg_speed",
    "playerload2d",
    "total_accelerations",
    "total_decelerations",
]


# ------------------------------------------------------------
# Small helpers
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
        # Laat alleen hele waarden toe voor int-kolommen
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
        # nette weergave: geen onnodige nullen
        s = f"{fv:.2f}"
        s = s.rstrip("0").rstrip(".")
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


def _has_existing_record_signal(rec: dict[str, Any], metrics_to_check: list[str]) -> bool:
    if rec is None:
        return False
    if not _is_missing(rec.get("match_id")):
        return True
    for m in metrics_to_check:
        if not _is_missing(rec.get(m)):
            return True
    return False


def _fetch_existing_for_day(
    access_token: str,
    player_ids: list[str],
    d_iso: str,
    t: str,
    e: str,
) -> list[dict[str, Any]]:
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
        "event": f"eq.{e}",
        "limit": "5000",
    }
    return rest_get(access_token, "gps_records", params)


def _compute_day_stat(rows: list[dict[str, Any]], metrics: list[str], how: str) -> dict[str, float | None]:
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    out: dict[str, float | None] = {}
    for m in metrics:
        if df.empty or m not in df.columns:
            out[m] = None
            continue
        s = pd.to_numeric(df[m], errors="coerce").dropna()
        if s.empty:
            out[m] = None
        else:
            out[m] = float(s.median()) if how == "median" else float(s.mean())
    return out


def _apply_bulk_to_players(
    state_cards: dict[str, dict[str, Any]],
    selected_pids: list[str],
    bulk_values: dict[str, Any],
    only_fill_missing: bool,
) -> None:
    for pid in selected_pids:
        card = state_cards.get(pid) or {}
        for k, v in bulk_values.items():
            if k not in METRIC_KEYS:
                continue
            if only_fill_missing and (not _is_missing(card.get(k))):
                continue
            card[k] = v
        state_cards[pid] = card


def _sync_selected_players_cards(
    cards: dict[str, dict[str, Any]],
    selected_pids: list[str],
    pid_to_name: dict[str, str],
    d: date,
    t: str,
    e: str,
) -> dict[str, dict[str, Any]]:
    changed = False
    for pid in selected_pids:
        if pid not in cards:
            cards[pid] = _blank_record(pid_to_name.get(pid, ""), d, t, e)
            changed = True
        else:
            cards[pid]["player_name"] = pid_to_name.get(pid, cards[pid].get("player_name") or "")
            cards[pid]["datum"] = d
            cards[pid]["type"] = t
            cards[pid]["event"] = e
            cards[pid]["source_file"] = "manual"
    return cards if changed else cards


# ------------------------------------------------------------
# Sidebar tools
# ------------------------------------------------------------

def _render_sidebar_tools(selected_pids: list[str], pid_to_name: dict[str, str]) -> None:
    with st.sidebar:
        st.markdown("## Manual tools")

        # Metrics selector
        st.markdown("### Metrics tonen")
        st.session_state["manual_selected_metrics"] = st.multiselect(
            "Kies metrics voor spelerkaarten",
            options=METRIC_KEYS,
            default=st.session_state.get("manual_selected_metrics", DEFAULT_METRICS),
            key="manual_metrics_picker_sidebar",
        )

        st.divider()

        tool_tabs = st.tabs(["Dag-stat", "Bulk", "Kopie"])

        # ------------------------
        # Tab 1: Day stats
        # ------------------------
        with tool_tabs[0]:
            st.caption("Bereken mediaan/gemiddelde op basis van geselecteerde spelers.")
            if not selected_pids or not st.session_state["manual_selected_metrics"]:
                st.info("Selecteer spelers + metrics.")
            else:
                cards = st.session_state["manual_cards"]
                rows = [cards.get(pid, {}) for pid in selected_pids]
                stat_how = st.selectbox("Statistiek", ["median", "mean"], key="manual_day_stat_how")
                fill_mode = st.selectbox(
                    "Invullen",
                    ["niets", "alleen lege velden", "alles overschrijven"],
                    key="manual_day_fill_mode",
                )

                day_stat = _compute_day_stat(rows, st.session_state["manual_selected_metrics"], stat_how)

                # Compacte preview (max 6)
                preview_keys = st.session_state["manual_selected_metrics"][:6]
                for m in preview_keys:
                    v = day_stat.get(m)
                    st.metric(m, "-" if v is None else _format_metric_value(v, m))

                if st.button("Pas dag-stat toe", key="manual_apply_day_stat_btn", type="primary"):
                    if fill_mode == "niets":
                        toast_err("Kies eerst een invulmodus.")
                    else:
                        vals = {k: v for k, v in day_stat.items() if v is not None}
                        _apply_bulk_to_players(
                            st.session_state["manual_cards"],
                            selected_pids,
                            vals,
                            only_fill_missing=(fill_mode == "alleen lege velden"),
                        )
                        toast_ok("Dag-stat toegepast.")
                        st.rerun()

        # ------------------------
        # Tab 2: Bulk apply
        # ------------------------
        with tool_tabs[1]:
            st.caption("Vul alleen de metrics in die je wilt toepassen. Leeg = overslaan.")
            if not selected_pids:
                st.info("Selecteer eerst spelers.")
            else:
                bulk_only_missing = st.toggle(
                    "Alleen lege velden vullen",
                    value=True,
                    key="manual_bulk_only_missing",
                )

                st.markdown("**Bulkwaarden**")
                for m in st.session_state["manual_selected_metrics"]:
                    # text_input -> echte lege velden mogelijk
                    _ = st.text_input(
                        m,
                        value=st.session_state.get(f"manual_bulk_txt_{m}", ""),
                        key=f"manual_bulk_txt_{m}",
                        placeholder="Leeg laten = niet toepassen",
                    )

                if st.button("Toepassen op geselecteerde spelers", type="primary", key="manual_bulk_apply_btn"):
                    try:
                        include = [pid for pid in selected_pids if st.session_state.get(f"manual_inc_{pid}", True)]
                        if not include:
                            toast_err("Geen spelers geselecteerd in de cards (Meenemen).")
                        else:
                            bulk_vals: dict[str, Any] = {}
                            for m in st.session_state["manual_selected_metrics"]:
                                txt = st.session_state.get(f"manual_bulk_txt_{m}", "")
                                val, err = _parse_metric_input(txt, m)
                                if err:
                                    toast_err(err)
                                    st.stop()
                                if val is not None:
                                    bulk_vals[m] = val

                            if not bulk_vals:
                                toast_err("Geen bulkwaarden ingevuld.")
                                st.stop()

                            _apply_bulk_to_players(
                                st.session_state["manual_cards"],
                                include,
                                bulk_vals,
                                only_fill_missing=bulk_only_missing,
                            )
                            toast_ok(f"Bulk toegepast op {len(include)} spelers.")
                            st.rerun()
                    except Exception as ex:
                        toast_err(str(ex))

        # ------------------------
        # Tab 3: Copy
        # ------------------------
        with tool_tabs[2]:
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
                copy_only_missing = st.toggle(
                    "Alleen lege velden vullen",
                    value=False,
                    key="manual_copy_only_missing_sidebar",
                )
                if st.button("Kopieer waarden", type="primary", key="manual_copy_btn_sidebar"):
                    try:
                        src = cards.get(src_pid) or {}
                        vals = {}
                        for m in st.session_state["manual_selected_metrics"]:
                            v = src.get(m)
                            if not _is_missing(v):
                                vals[m] = v
                        if not vals:
                            toast_err("Bronspeler heeft geen ingevulde waarden in de gekozen metrics.")
                            st.stop()
                        _apply_bulk_to_players(cards, dst_pids, vals, only_fill_missing=copy_only_missing)
                        st.session_state["manual_cards"] = cards
                        toast_ok("Kopieeractie uitgevoerd.")
                        st.rerun()
                    except Exception as ex:
                        toast_err(str(ex))

        st.divider()
        st.caption("Tip: vink 'Meenemen' uit bij spelers die je tijdelijk niet wilt opslaan.")


# ------------------------------------------------------------
# Player card renderer
# ------------------------------------------------------------

def _render_player_card(
    access_token: str,
    pid: str,
    nm: str,
    d: date,
    d_iso: str,
    t: str,
    e: str,
    metrics: list[str],
) -> None:
    cards = st.session_state["manual_cards"]
    rec = cards.get(pid) or _blank_record(nm, d, t, e)

    exists_flag = _has_existing_record_signal(rec, metrics)

    # Expander per speler
    exp_label = f"{nm}  •  {'Bestaand' if exists_flag else 'Nieuw'}"
    with st.expander(exp_label, expanded=(len(st.session_state.get("manual_selected_players", [])) <= 2)):
        with st.container(border=True):
            h1, h2, h3, h4 = st.columns([2.3, 1.0, 1.0, 1.2], vertical_alignment="center")

            with h1:
                st.markdown(f"**{nm}**")
                st.caption(f"player_id: {pid}")

            with h2:
                # per speler meenemen toggle
                st.toggle("Meenemen", value=st.session_state.get(f"manual_inc_{pid}", True), key=f"manual_inc_{pid}")

            with h3:
                st.write("Bestaand" if exists_flag else "Nieuw")

            with h4:
                if st.button("Verwijder", key=f"manual_delete_{pid}"):
                    try:
                        params = {
                            "player_id": f"eq.{pid}",
                            "datum": f"eq.{d_iso}",
                            "type": f"eq.{t}",
                            "event": f"eq.{e}",
                        }
                        rest_delete(access_token, "gps_records", params)
                        cards[pid] = _blank_record(nm, d, t, e)
                        st.session_state["manual_cards"] = cards

                        # clear widget values for this player's metric fields
                        for m in metrics:
                            st.session_state[f"manual_txt_{pid}_{m}"] = ""

                        toast_ok("Record verwijderd.")
                        st.rerun()
                    except Exception as ex:
                        toast_err(str(ex))

            # Meta info row
            m1, m2, m3 = st.columns([1.2, 1.2, 2.6], vertical_alignment="bottom")
            with m1:
                st.text_input("Event", value=e, disabled=True, key=f"manual_meta_event_{pid}")
            with m2:
                st.text_input("Type", value=t, disabled=True, key=f"manual_meta_type_{pid}")
            with m3:
                st.text_input(
                    "match_id (auto bij Match / Practice Match)",
                    value="" if _is_missing(rec.get("match_id")) else str(rec.get("match_id")),
                    disabled=True,
                    key=f"manual_meta_matchid_{pid}",
                )

            # Quick preset buttons
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
                    rows = [st.session_state["manual_cards"].get(x, {}) for x in st.session_state.get("manual_selected_pid_list", [])]
                    med_vals = _compute_day_stat(rows, metrics, "median")
                    for m, v in med_vals.items():
                        if v is not None:
                            rec[m] = int(v) if m in INT_DB_COLS else float(v)
                            st.session_state[f"manual_txt_{pid}_{m}"] = _format_metric_value(rec[m], m)
                    cards[pid] = rec
                    st.session_state["manual_cards"] = cards
                    st.rerun()
            with q3:
                st.caption("Laat veld leeg om leeg op te slaan. Gebruik komma of punt voor decimalen.")

            # Metrics grid (text_input => echte lege velden)
            grid_cols = st.columns(2) if len(metrics) > 8 else st.columns(4)
            for i, m in enumerate(metrics):
                with grid_cols[i % len(grid_cols)]:
                    key_txt = f"manual_txt_{pid}_{m}"
                    if key_txt not in st.session_state:
                        st.session_state[key_txt] = _format_metric_value(rec.get(m), m)

                    val_txt = st.text_input(
                        m,
                        value=st.session_state[key_txt],
                        key=key_txt,
                        placeholder="leeg",
                    )
                    parsed, err = _parse_metric_input(val_txt, m)
                    if err:
                        st.warning(err)
                    else:
                        rec[m] = parsed

            rec["player_name"] = nm
            rec["datum"] = d
            rec["type"] = t
            rec["event"] = e
            rec["source_file"] = "manual"
            cards[pid] = rec
            st.session_state["manual_cards"] = cards


# ------------------------------------------------------------
# Main tab
# ------------------------------------------------------------

def tab_manual_add_main(access_token: str, name_to_id: dict, player_options: list[str]) -> None:
    st.subheader("Manual add")
    st.caption(
        "Kies datum/type/event en spelers. Bestaande records worden geladen zodat je direct kunt aanpassen. "
        "Tools (metrics, bulk, kopiëren, dag-stat) staan in de sidebar voor een rustigere pagina."
    )

    # ----------------------------
    # State init
    # ----------------------------
    if "manual_cards" not in st.session_state:
        st.session_state["manual_cards"] = {}

    if "manual_selected_metrics" not in st.session_state:
        st.session_state["manual_selected_metrics"] = [m for m in DEFAULT_METRICS if m in METRIC_KEYS]

    if "manual_selected_players" not in st.session_state:
        st.session_state["manual_selected_players"] = []

    if "manual_selected_pid_list" not in st.session_state:
        st.session_state["manual_selected_pid_list"] = []

    # ----------------------------
    # 1) Basisselectie (compact)
    # ----------------------------
    st.markdown("### 1) Basisselectie")

    r1c1, r1c2, r1c3 = st.columns([1.2, 1.0, 1.2], vertical_alignment="bottom")
    with r1c1:
        d = st.date_input("Datum", value=st.session_state.get("manual_day", date.today()), key="manual_day")
    with r1c2:
        default_type_idx = TYPE_OPTIONS.index("Practice") if "Practice" in TYPE_OPTIONS else 0
        t = st.selectbox("Type", options=TYPE_OPTIONS, index=default_type_idx, key="manual_type")
    with r1c3:
        e = st.text_input("Event", value=st.session_state.get("manual_event", "Summary"), key="manual_event")

    r2c1, r2c2 = st.columns([5, 1], vertical_alignment="bottom")
    with r2c1:
        players = st.multiselect(
            "Spelers",
            options=player_options,
            default=st.session_state["manual_selected_players"],
            key="manual_players",
        )
        st.session_state["manual_selected_players"] = players
    with r2c2:
        load_clicked = st.button("Laad / ververs", type="primary", key="manual_load_existing")

    d_iso = pd.to_datetime(d).date().isoformat()
    t = str(t or "").strip()
    e = str(e or "").strip()

    # Player IDs mapping
    selected_pids: list[str] = []
    pid_to_name: dict[str, str] = {}
    for nm in players:
        pid = name_to_id.get(normalize_name(nm))
        if pid is not None:
            pid_s = str(pid)
            selected_pids.append(pid_s)
            pid_to_name[pid_s] = nm

    st.session_state["manual_selected_pid_list"] = selected_pids

    # Sidebar tools (cleaner page)
    _render_sidebar_tools(selected_pids, pid_to_name)

    # Ensure cards exist / keep metadata synced
    st.session_state["manual_cards"] = _sync_selected_players_cards(
        st.session_state["manual_cards"], selected_pids, pid_to_name, d, t, e
    )

    # ----------------------------
    # Load existing records
    # ----------------------------
    if load_clicked:
        if not selected_pids:
            toast_err("Selecteer eerst minimaal één speler.")
            return
        if not e:
            toast_err("Event mag niet leeg zijn.")
            return

        try:
            existing = _fetch_existing_for_day(access_token, selected_pids, d_iso, t, e)
            by_pid: dict[str, dict[str, Any]] = {}
            for r in existing:
                pid = str(r.get("player_id"))
                if pid:
                    by_pid[pid] = r

            cards = st.session_state["manual_cards"]

            for pid in selected_pids:
                nm = pid_to_name.get(pid, "")
                if pid in by_pid:
                    rec = _blank_record(nm, d, t, e)
                    rec.update(by_pid[pid])
                    rec["player_name"] = nm or rec.get("player_name")
                    rec["datum"] = d
                    rec["type"] = t
                    rec["event"] = e
                    rec["source_file"] = "manual"
                    cards[pid] = rec
                else:
                    cards[pid] = _blank_record(nm, d, t, e)

                # sync widget text values from card after load
                for m in st.session_state["manual_selected_metrics"]:
                    st.session_state[f"manual_txt_{pid}_{m}"] = _format_metric_value(cards[pid].get(m), m)

            st.session_state["manual_cards"] = cards
            toast_ok(f"Geladen: {len(existing)} bestaand, {len(selected_pids) - len(existing)} nieuw.")
            st.rerun()
        except Exception as ex:
            toast_err(str(ex))
            return

    # ----------------------------
    # 2) Invoer per speler (hoofdsectie)
    # ----------------------------
    st.markdown("### 2) Invoer per speler")

    if not selected_pids:
        st.info("Selecteer spelers en klik op **Laad / ververs**.")
        return

    # Top summary chips
    csum1, csum2, csum3 = st.columns(3)
    csum1.metric("Geselecteerde spelers", len(selected_pids))
    include_count = sum(1 for pid in selected_pids if st.session_state.get(f"manual_inc_{pid}", True))
    csum2.metric("Meenemen bij save", include_count)
    csum3.metric("Getoonde metrics", len(st.session_state["manual_selected_metrics"]))

    # Two-column card layout
    metrics = st.session_state["manual_selected_metrics"] or DEFAULT_METRICS
    if not metrics:
        st.warning("Kies minimaal 1 metric in de sidebar.")
        return

    col_left, col_right = st.columns(2, vertical_alignment="top")
    for idx, pid in enumerate(selected_pids):
        target_col = col_left if idx % 2 == 0 else col_right
        with target_col:
            _render_player_card(
                access_token=access_token,
                pid=pid,
                nm=pid_to_name.get(pid, pid),
                d=d,
                d_iso=d_iso,
                t=t,
                e=e,
                metrics=metrics,
            )

    # ----------------------------
    # 3) Acties (onderaan)
    # ----------------------------
    st.divider()
    st.markdown("### 3) Acties")

    a1, a2, a3 = st.columns([1.2, 1.2, 2.6], vertical_alignment="bottom")

    with a1:
        if st.button("Reset selectie (leegmaken)", key="manual_reset_cards_btn"):
            cards = st.session_state["manual_cards"]
            for pid in selected_pids:
                nm = pid_to_name.get(pid, pid)
                cards[pid] = _blank_record(nm, d, t, e)
                for m in metrics:
                    st.session_state[f"manual_txt_{pid}_{m}"] = ""
            st.session_state["manual_cards"] = cards
            toast_ok("Reset bevestigd.")
            st.rerun()

    with a2:
        save_clicked = st.button("Save (upsert)", type="primary", key="manual_save_cards_btn")

    with a3:
        st.caption(
            "Opslaan gebeurt op conflict: player_name + datum + type + event. "
            "Lege velden blijven leeg (None)."
        )

    if not save_clicked:
        return

    # ----------------------------
    # Validate + Save
    # ----------------------------
    try:
        include_pids = [pid for pid in selected_pids if st.session_state.get(f"manual_inc_{pid}", True)]
        if not include_pids:
            toast_err("Geen spelers geselecteerd om op te slaan (Meenemen).")
            return

        if not e:
            toast_err("Event mag niet leeg zijn.")
            return

        cards = st.session_state["manual_cards"]

        df_rows: list[dict[str, Any]] = []
        errors: list[str] = []

        for pid in include_pids:
            rec = (cards.get(pid) or {}).copy()

            # Re-parse visible metric fields from widget state (bron van waarheid)
            for m in metrics:
                txt = st.session_state.get(f"manual_txt_{pid}_{m}", "")
                parsed, err = _parse_metric_input(txt, m)
                if err:
                    errors.append(f"{pid_to_name.get(pid, pid)} → {err}")
                rec[m] = parsed

            rec["player_id"] = pid
            rec["player_name"] = str(rec.get("player_name") or pid_to_name.get(pid, "")).strip()
            rec["datum"] = d_iso
            rec["type"] = t
            rec["event"] = e
            rec["source_file"] = "manual"

            if not rec["player_name"]:
                errors.append(f"Spelernaam ontbreekt voor player_id={pid}")
                continue

            dt = pd.to_datetime(rec["datum"], errors="coerce")
            if pd.isna(dt):
                errors.append(f"Ongeldige datum voor {rec['player_name']}")
                continue

            rec["week"] = int(dt.isocalendar().week)
            rec["year"] = int(dt.year)

            # Coerce all metrics (ook niet-zichtbare)
            for m in METRIC_KEYS:
                rec[m] = _coerce_number(rec.get(m))
                if m in INT_DB_COLS and rec[m] is not None:
                    rec[m] = int(float(rec[m]))

            df_rows.append(rec)

        if errors:
            for err in errors[:5]:
                toast_err(err)
            if len(errors) > 5:
                toast_err(f"... en nog {len(errors)-5} fout(en).")
            return

        if not df_rows:
            toast_err("Geen geldige rijen om op te slaan.")
            return

        dfm = pd.DataFrame(df_rows)

        # Enforce match_id consistency per (datum,type)
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
                    # manual pick (once)
                    with st.expander(f"Match koppeling nodig voor {d0} ({t0})", expanded=True):
                        picked = ui_pick_match_if_needed(
                            access_token,
                            d_obj,
                            t0,
                            key_prefix=f"manual_cards_match_{d0}_{t0}",
                        )
                    if picked is None:
                        toast_err(f"Geen match_id beschikbaar voor {d0} ({t0}). Voeg match toe of kies match.")
                        return
                    forced_id = int(picked)

            dfm.loc[g.index, "match_id"] = forced_id

        # Build payload
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

        rest_upsert(access_token, "gps_records", rows_for_save, on_conflict="player_name,datum,type,event")
        toast_ok(f"Save bevestigd: rows = {len(rows_for_save)}")

        # Optional: sync cards back from df for clean state
        for _, r in dfm.iterrows():
            pid = str(r["player_id"])
            if pid in st.session_state["manual_cards"]:
                for k in ["match_id"] + METRIC_KEYS:
                    st.session_state["manual_cards"][pid][k] = r.get(k)

        st.rerun()

    except Exception as ex:
        toast_err(f"Save fout: {ex}")
