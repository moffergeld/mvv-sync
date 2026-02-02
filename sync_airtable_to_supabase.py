import os
import requests
from dateutil.parser import parse as dtparse

AIRTABLE_TOKEN = os.environ["AIRTABLE_TOKEN"]
AIRTABLE_BASE_ID = os.environ["AIRTABLE_BASE_ID"]

# Airtable table names (exact)
T_PLAYERS = os.environ.get("AIRTABLE_TABLE_PLAYERS", "Playerlist")
T_WELLNESS = os.environ.get("AIRTABLE_TABLE_WELLNESS", "Welness")
T_RPE = os.environ.get("AIRTABLE_TABLE_RPE", "RPE")

AIRTABLE_VIEW = os.environ.get("AIRTABLE_VIEW")  # optional

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

def airtable_fetch_all(table_name: str):
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{table_name}"
    headers = {"Authorization": f"Bearer {AIRTABLE_TOKEN}"}
    params = {}
    if AIRTABLE_VIEW:
        params["view"] = AIRTABLE_VIEW

    out = []
    offset = None
    while True:
        if offset:
            params["offset"] = offset
        r = requests.get(url, headers=headers, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("records", []))
        offset = data.get("offset")
        if not offset:
            break
    return out

def sb_headers():
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }

def sb_upsert(table: str, rows: list, on_conflict: str):
    if not rows:
        return []
    url = f"{SUPABASE_URL}/rest/v1/{table}?on_conflict={on_conflict}"
    headers = sb_headers()

    results = []
    chunk_size = 500
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i+chunk_size]
        r = requests.post(url, headers=headers, json=chunk, timeout=60)
        r.raise_for_status()
        results.extend(r.json() if r.text else [])
    return results

def sb_select_players_map():
    # map airtable_record_id -> player_id
    url = f"{SUPABASE_URL}/rest/v1/players?select=player_id,airtable_record_id"
    headers = sb_headers()
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    return {row["airtable_record_id"]: row["player_id"] for row in data}

def _to_int(x):
    if x is None or x == "":
        return None
    try:
        return int(float(x))
    except Exception:
        return None

def _to_date_iso(x):
    if x is None or x == "":
        return None
    try:
        return dtparse(str(x)).date().isoformat()
    except Exception:
        return None

def _to_ts_iso(x):
    if x is None or x == "":
        return None
    try:
        return dtparse(str(x)).isoformat()
    except Exception:
        return None

def transform_players(records):
    rows = []
    for rec in records:
        rid = rec.get("id")
        f = rec.get("fields", {})
        name = f.get("Name")
        if not rid or not name:
            continue

        rows.append({
            "airtable_record_id": rid,
            "full_name": str(name).strip(),
            "birth_date": _to_date_iso(f.get("Birth of date")),
        })
    return rows

def _extract_linked_record_id(value):
    # Airtable linked record is usually a list of record ids
    if isinstance(value, list) and value:
        return value[0]
    return None

def transform_wellness(records, players_map):
    rows = []
    for rec in records:
        rid = rec.get("id")
        f = rec.get("fields", {})
        if not rid:
            continue

        player_air_id = _extract_linked_record_id(f.get("Player name"))
        player_id = players_map.get(player_air_id) if player_air_id else None
        form_date = _to_date_iso(f.get("Form date"))

        if not player_id or not form_date:
            continue

        rows.append({
            "player_id": player_id,
            "form_date": form_date,
            "created_at_airtable": _to_ts_iso(f.get("Created")),
            "muscle_soreness": _to_int(f.get("Muscle Soreness")),
            "fatigue": _to_int(f.get("Fatigue")),
            "sleep_quality": _to_int(f.get("Sleep Quality")),
            "stress": _to_int(f.get("Stress")),
            "mood": _to_int(f.get("Mood")),
            "created_by": f.get("Created by"),
            "source_airtable_record_id": rid,
        })
    return rows

def _extract_attachment_url(value):
    # Airtable attachment field can be list of dicts with 'url'
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0].get("url")
    if isinstance(value, str):
        return value
    return None

def transform_rpe(records, players_map):
    rows = []
    for rec in records:
        rid = rec.get("id")
        f = rec.get("fields", {})
        if not rid:
            continue

        player_air_id = _extract_linked_record_id(f.get("Player name"))
        player_id = players_map.get(player_air_id) if player_air_id else None
        form_date = _to_date_iso(f.get("Form date"))

        if not player_id or not form_date:
            continue

        rows.append({
            "player_id": player_id,
            "form_date": form_date,
            "created_at_airtable": _to_ts_iso(f.get("Created")),

            "ex1_duration_min": _to_int(f.get("[1] Duration of Exercise (Minutes)")),
            "ex1_exertion": _to_int(f.get("[1] Physical Exertion")),
            "ex2_duration_min": _to_int(f.get("[2] Duration of Exercise (Minutes)")),
            "ex2_exertion": _to_int(f.get("[2] Physical Exertion")),

            "injury": f.get("Injury"),
            "injury_type": f.get("Injury type") if isinstance(f.get("Injury type"), list) else None,
            "pain_scale_injury": _to_int(f.get("Pain scale Injury")),

            "created_by": f.get("Created by"),
            "attachment_url": _extract_attachment_url(f.get("Attachment")),

            "source_airtable_record_id": rid,
        })
    return rows

def main():
    # 1) players
    at_players = airtable_fetch_all(T_PLAYERS)
    p_rows = transform_players(at_players)
    sb_upsert("players", p_rows, "airtable_record_id")

    # reload mapping after upsert
    players_map = sb_select_players_map()

    # 2) wellness
    at_well = airtable_fetch_all(T_WELLNESS)
    w_rows = transform_wellness(at_well, players_map)
    sb_upsert("wellness", w_rows, "source_airtable_record_id")

    # 3) rpe
    at_rpe = airtable_fetch_all(T_RPE)
    r_rows = transform_rpe(at_rpe, players_map)
    sb_upsert("rpe", r_rows, "source_airtable_record_id")

    print(f"Players: Airtable {len(at_players)} -> upsert {len(p_rows)}")
    print(f"Wellness: Airtable {len(at_well)} -> upsert {len(w_rows)}")
    print(f"RPE: Airtable {len(at_rpe)} -> upsert {len(r_rows)}")

if __name__ == "__main__":
    main()
