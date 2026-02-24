# gps_import_tab_manual.py
# ============================================================
# Subtab: Manual add (cards)
# - Select date/type/event + players
# - Prefill from existing gps_records (same player+date+type+event)
# - Edit via professional input cards (no data_editor table)
# - Bulk apply values
# - Copy values from one player to others
# - Compute day mean/median and optionally fill missing
# - Delete per player record
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


def _rest_headers(access_token: str, prefer_return: bool = False) -> dict[str, str]:
    h = {
        "apikey": st.secrets["SUPABASE_ANON_KEY"],
        "Authorization": f"Bearer {access_token}",
    }
    if prefer_return:
        h["Prefer"] = "return=representation"
    return h


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
    # PostgREST returns empty body by default; count not guaranteed
    return 1


# ------------------------------------------------------------
# UI + logic
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


def _blank_record(player_name: str, d: date, t: str, e: str) -> dict[str, Any]:
    row = {k: None for k in TEMPLATE_COLS}
    row["player_name"] = player_name
    row["datum"] = d
    row["type"] = t
    row["event"] = e
    row["match_id"] = None
    row["source_file"] = "manual"
    return row


def _coerce_number(v: Any) -> float | int | None:
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return None
    try:
        fv = float(v)
        if pd.isna(fv):
            return None
        return fv
    except Exception:
        return None


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    if isinstance(v, str) and v.strip() == "":
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
    # PostgREST: player_id=in.(a,b,c)
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
            continue
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


def tab_manual_add_main(access_token: str, name_to_id: dict, player_options: list[str]) -> None:
    st.subheader("Manual add")
    st.caption(
        "Selecteer datum/type/event en spelers. Bestaande records worden automatisch geladen zodat je ze kunt aanpassen. "
        "Je kunt ook verwijderen en bulk-wijzigingen toepassen."
    )

    # ----------------------------
    # Session state init
    # ----------------------------
    if "manual_cards" not in st.session_state:
        # dict[player_id] -> record dict (TEMPLATE_COLS-ish)
        st.session_state["manual_cards"] = {}

    if "manual_selected_metrics" not in st.session_state:
        st.session_state["manual_selected_metrics"] = [m for m in DEFAULT_METRICS if m in METRIC_KEYS]

    if "manual_selected_players" not in st.session_state:
        st.session_state["manual_selected_players"] = []

    # ----------------------------
    # Controls (top)
    # ----------------------------
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 2.4], vertical_alignment="bottom")

    with c1:
        d = st.date_input("Datum", value=date.today(), key="manual_day")
    with c2:
        t = st.selectbox("Type", options=TYPE_OPTIONS, index=TYPE_OPTIONS.index("Practice") if "Practice" in TYPE_OPTIONS else 0, key="manual_type")
    with c3:
        e = st.text_input("Event", value="Summary", key="manual_event")
    with c4:
        selected_metrics = st.multiselect(
            "Metrics om te tonen",
            options=METRIC_KEYS,
            default=st.session_state["manual_selected_metrics"],
            key="manual_metrics_picker_cards",
        )
        st.session_state["manual_selected_metrics"] = selected_metrics

    # Players
    pcol1, pcol2 = st.columns([3.0, 1.2], vertical_alignment="bottom")
    with pcol1:
        players = st.multiselect(
            "Spelers",
            options=player_options,
            default=st.session_state["manual_selected_players"],
            key="manual_players",
        )
        st.session_state["manual_selected_players"] = players

    with pcol2:
        load_clicked = st.button("Laad / ververs", type="primary", key="manual_load_existing")

    d_iso = pd.to_datetime(d).date().isoformat()
    t = str(t or "").strip()
    e = str(e or "").strip()

    # Resolve selected player_ids
    selected_pids: list[str] = []
    pid_to_name: dict[str, str] = {}
    for nm in players:
        pid = name_to_id.get(normalize_name(nm))
        if pid is not None:
            pid_s = str(pid)
            selected_pids.append(pid_s)
            pid_to_name[pid_s] = nm

    # ----------------------------
    # Load existing (prefill)
    # ----------------------------
    if load_clicked and selected_pids and d_iso and t and e:
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
                    # merge existing into card
                    rec = _blank_record(nm, d, t, e)
                    rec.update(by_pid[pid])
                    rec["player_name"] = nm or rec.get("player_name")  # keep UI name
                    rec["datum"] = d
                    rec["type"] = t
                    rec["event"] = e
                    cards[pid] = rec
                else:
                    # create fresh
                    cards[pid] = _blank_record(nm, d, t, e)

            st.session_state["manual_cards"] = cards
            toast_ok(f"Geladen: {len(existing)} bestaand, {len(selected_pids) - len(existing)} nieuw.")
        except Exception as ex:
            toast_err(str(ex))

    # If players selected but not yet in cards, ensure they exist (without fetching)
    if selected_pids:
        cards = st.session_state["manual_cards"]
        changed = False
        for pid in selected_pids:
            if pid not in cards:
                cards[pid] = _blank_record(pid_to_name.get(pid, ""), d, t, e)
                changed = True
            else:
                # keep day/type/event in sync
                cards[pid]["datum"] = d
                cards[pid]["type"] = t
                cards[pid]["event"] = e
        if changed:
            st.session_state["manual_cards"] = cards

    # ----------------------------
    # Day summary (mean/median)
    # ----------------------------
    if selected_pids and selected_metrics:
        cards = st.session_state["manual_cards"]
        day_rows = [cards[pid] for pid in selected_pids if pid in cards]

        s1, s2, s3 = st.columns([1.2, 1.2, 3.6], vertical_alignment="bottom")
        with s1:
            stat_how = st.selectbox("Dag statistiek", options=["median", "mean"], index=0, key="manual_day_stat_how")
        with s2:
            fill_mode = st.selectbox("Invullen", options=["niets", "alleen lege velden", "alles overschrijven"], index=0, key="manual_day_fill_mode")
        with s3:
            apply_day_stat = st.button("Pas dag-stat toe op geselecteerde spelers", key="manual_apply_day_stat")

        day_stat = _compute_day_stat(day_rows, selected_metrics, stat_how)
        # compacte preview
        preview_df = pd.DataFrame([day_stat])
        st.dataframe(preview_df, use_container_width=True, height=80)

        if apply_day_stat:
            try:
                only_missing = fill_mode == "alleen lege velden"
                overwrite = fill_mode == "alles overschrijven"
                if fill_mode == "niets":
                    toast_err("Kies eerst een invulmodus (alleen leeg / overschrijven).")
                else:
                    bulk_vals = {}
                    for k, v in day_stat.items():
                        if v is None:
                            continue
                        bulk_vals[k] = v
                    if overwrite:
                        _apply_bulk_to_players(st.session_state["manual_cards"], selected_pids, bulk_vals, only_fill_missing=False)
                    else:
                        _apply_bulk_to_players(st.session_state["manual_cards"], selected_pids, bulk_vals, only_fill_missing=True)
                    toast_ok("Dag-stat toegepast.")
                    st.rerun()
            except Exception as ex:
                toast_err(str(ex))

    # ----------------------------
    # Bulk apply / Copy tools
    # ----------------------------
    if selected_pids:
        with st.expander("Bulk aanpassen (1x invullen → toepassen op meerdere spelers)", expanded=False):
            b1, b2, b3 = st.columns([1.2, 1.2, 2.6], vertical_alignment="bottom")
            with b1:
                bulk_only_missing = st.toggle("Alleen lege velden vullen", value=True, key="manual_bulk_only_missing")
            with b2:
                bulk_apply = st.button("Toepassen op geselecteerde spelers", type="primary", key="manual_bulk_apply")
            with b3:
                st.caption("Tip: selecteer spelers in de cards via ‘Meenemen’ en pas daarna bulk toe.")

            bulk_values: dict[str, Any] = {}
            # toon dezelfde metric set als gekozen
            cols = st.columns(4)
            for i, m in enumerate(st.session_state["manual_selected_metrics"]):
                with cols[i % 4]:
                    step = 1 if m in INT_DB_COLS else None
                    v = st.number_input(
                        m,
                        value=0.0,
                        step=step if step is not None else 0.01,
                        key=f"manual_bulk_{m}",
                    )
                    bulk_values[m] = float(v) if step is None else int(v)

            if bulk_apply:
                try:
                    include = []
                    for pid in selected_pids:
                        if st.session_state.get(f"manual_inc_{pid}", True):
                            include.append(pid)
                    if not include:
                        toast_err("Geen spelers geselecteerd in de cards (Meenemen).")
                    else:
                        _apply_bulk_to_players(st.session_state["manual_cards"], include, bulk_values, only_fill_missing=bulk_only_missing)
                        toast_ok(f"Bulk toegepast op {len(include)} spelers.")
                        st.rerun()
                except Exception as ex:
                    toast_err(str(ex))

        with st.expander("Kopieer waarden van 1 speler naar anderen", expanded=False):
            cards = st.session_state["manual_cards"]
            src_pid = st.selectbox(
                "Bronspeler",
                options=selected_pids,
                format_func=lambda pid: pid_to_name.get(pid, pid),
                key="manual_copy_src",
            )
            dst_pids = st.multiselect(
                "Doelspelers",
                options=[p for p in selected_pids if p != src_pid],
                default=[p for p in selected_pids if p != src_pid],
                format_func=lambda pid: pid_to_name.get(pid, pid),
                key="manual_copy_dst",
            )
            copy_only_missing = st.toggle("Alleen lege velden vullen", value=False, key="manual_copy_only_missing")
            copy_btn = st.button("Kopieer", type="primary", key="manual_copy_btn")

            if copy_btn:
                try:
                    src = cards.get(src_pid) or {}
                    bulk_vals = {}
                    for m in st.session_state["manual_selected_metrics"]:
                        v = src.get(m)
                        if _is_missing(v):
                            continue
                        bulk_vals[m] = v
                    _apply_bulk_to_players(cards, dst_pids, bulk_vals, only_fill_missing=copy_only_missing)
                    st.session_state["manual_cards"] = cards
                    toast_ok("Kopieeractie uitgevoerd.")
                    st.rerun()
                except Exception as ex:
                    toast_err(str(ex))

    # ----------------------------
    # Player cards (professional inputs)
    # ----------------------------
    if not selected_pids:
        st.info("Selecteer spelers om te beginnen.")
        return

    st.markdown("### Invoer per speler")

    cards = st.session_state["manual_cards"]

    for pid in selected_pids:
        nm = pid_to_name.get(pid, pid)
        rec = cards.get(pid) or _blank_record(nm, d, t, e)

        with st.container(border=True):
            h1, h2, h3, h4 = st.columns([2.2, 1.2, 1.2, 1.4], vertical_alignment="center")

            with h1:
                st.markdown(f"**{nm}**")
                st.caption(f"player_id: {pid}")

            with h2:
                st.toggle("Meenemen", value=True, key=f"manual_inc_{pid}")

            with h3:
                # show if existing (heuristic: match_id or any metric not None)
                exists_flag = any((not _is_missing(rec.get(k))) for k in ["match_id"] + st.session_state["manual_selected_metrics"])
                st.write("Bestaand" if exists_flag else "Nieuw")

            with h4:
                del_btn = st.button("Verwijder record", key=f"manual_delete_{pid}")
                if del_btn:
                    try:
                        params = {
                            "player_id": f"eq.{pid}",
                            "datum": f"eq.{d_iso}",
                            "type": f"eq.{t}",
                            "event": f"eq.{e}",
                        }
                        rest_delete(access_token, "gps_records", params)
                        # reset local card
                        cards[pid] = _blank_record(nm, d, t, e)
                        st.session_state["manual_cards"] = cards
                        toast_ok("Record verwijderd.")
                        st.rerun()
                    except Exception as ex:
                        toast_err(str(ex))

            # match_id logic (preview / enforced at save)
            mi_col1, mi_col2 = st.columns([1.2, 3.8], vertical_alignment="bottom")
            with mi_col1:
                st.text_input("Event", value=e, disabled=True, key=f"manual_ev_{pid}")
            with mi_col2:
                mi = rec.get("match_id")
                st.text_input(
                    "match_id (auto bij Match/Practice Match)",
                    value="" if mi is None else str(mi),
                    disabled=True,
                    key=f"manual_mi_{pid}",
                )

            # metrics inputs
            metrics = st.session_state["manual_selected_metrics"]
            grid = st.columns(4, vertical_alignment="bottom")

            for i, m in enumerate(metrics):
                with grid[i % 4]:
                    step = 1 if m in INT_DB_COLS else 0.01
                    v0 = rec.get(m)
                    # st.number_input requires a number; map None -> 0 but keep None in state using empty toggle
                    val_num = 0.0 if _is_missing(v0) else float(v0)
                    new_v = st.number_input(
                        m,
                        value=val_num,
                        step=step,
                        key=f"manual_{pid}_{m}",
                    )
                    # store back (always numeric); allow "0" as valid
                    rec[m] = int(new_v) if m in INT_DB_COLS else float(new_v)

            # persist record back
            rec["player_name"] = nm
            rec["datum"] = d
            rec["type"] = t
            rec["event"] = e
            rec["source_file"] = "manual"
            cards[pid] = rec

    st.session_state["manual_cards"] = cards

    # ----------------------------
    # Save all
    # ----------------------------
    st.divider()
    sleft, sright = st.columns([1.2, 1.2], vertical_alignment="bottom")

    with sleft:
        if st.button("Reset selectie (leegmaken)", key="manual_reset_cards"):
            for pid in selected_pids:
                nm = pid_to_name.get(pid, pid)
                cards[pid] = _blank_record(nm, d, t, e)
            st.session_state["manual_cards"] = cards
            toast_ok("Reset bevestigd.")
            st.rerun()

    with sright:
        save_clicked = st.button("Save (upsert)", type="primary", key="manual_save_cards")

    if not save_clicked:
        return

    try:
        # Build rows from included players
        include_pids = [pid for pid in selected_pids if st.session_state.get(f"manual_inc_{pid}", True)]
        if not include_pids:
            toast_err("Geen spelers geselecteerd om op te slaan (Meenemen).")
            return

        rows_for_save: list[dict[str, Any]] = []
        dfm_rows: list[dict[str, Any]] = []

        for pid in include_pids:
            r = (st.session_state["manual_cards"].get(pid) or {}).copy()
            r["player_id"] = pid
            r["player_name"] = str(r.get("player_name") or "").strip()
            r["datum"] = d_iso
            r["type"] = str(t).strip()
            r["event"] = str(e).strip()
            r["source_file"] = "manual"

            if not r["player_name"] or not r["datum"] or not r["type"] or not r["event"]:
                continue

            # week/year
            dt = pd.to_datetime(r["datum"])
            r["week"] = int(dt.isocalendar().week)
            r["year"] = int(dt.year)

            # numeric coercion
            for k in METRIC_KEYS:
                if k in r:
                    r[k] = _coerce_number(r.get(k))
                    if k in INT_DB_COLS and r[k] is not None:
                        r[k] = int(float(r[k]))

            dfm_rows.append(r)

        if not dfm_rows:
            toast_err("Geen geldige rijen om op te slaan.")
            return

        dfm = pd.DataFrame(dfm_rows)

        # enforce match_id per (datum,type) group
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
                    # allow manual pick (once per group)
                    with st.expander(f"Match koppeling nodig voor {d0} ({t0})", expanded=True):
                        picked = ui_pick_match_if_needed(access_token, d_obj, t0, key_prefix=f"manual_cards_match_{d0}_{t0}")
                    if picked is None:
                        toast_err(f"Geen match_id beschikbaar voor {d0} ({t0}). Voeg match toe of kies match.")
                        return
                    forced_id = int(picked)
            dfm.loc[g.index, "match_id"] = forced_id

        # Build payload for upsert
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

    except Exception as ex:
        toast_err(f"Save fout: {ex}")
