# roles.py
# ============================================================
# Centrale helpers voor auth/role/profile + player picker
# - Robuuste token retrieval (access_token of sb_session.access_token)
# - Robuuste PostgREST auth header injectie (supabase-py versieverschillen)
# - get_profile(): haalt role + player_id uit public.profiles
# - pick_target_player(): staff dropdown, player forced eigen player_id
# ============================================================

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from supabase import create_client


# -----------------------------
# Supabase init (singleton)
# -----------------------------
@st.cache_resource(show_spinner=False)
def _get_supabase_client():
    url = st.secrets.get("SUPABASE_URL", "").strip()
    key = st.secrets.get("SUPABASE_ANON_KEY", "").strip()
    if not url or not key:
        raise RuntimeError("Missing secrets: SUPABASE_URL / SUPABASE_ANON_KEY")
    return create_client(url, key)


def get_sb():
    try:
        return _get_supabase_client()
    except Exception:
        return None


# -----------------------------
# Auth helpers
# -----------------------------
def require_auth() -> None:
    """
    UI-gate: zorg dat er een sessie is.
    """
    if st.session_state.get("access_token"):
        return
    sess = st.session_state.get("sb_session")
    tok = getattr(sess, "access_token", None) if sess is not None else None
    if tok:
        st.session_state["access_token"] = tok
        return

    st.error("Niet ingelogd.")
    st.stop()


def get_access_token() -> str:
    require_auth()
    tok = st.session_state.get("access_token")
    if tok:
        return str(tok)

    sess = st.session_state.get("sb_session")
    tok2 = getattr(sess, "access_token", None) if sess is not None else None
    if tok2:
        st.session_state["access_token"] = str(tok2)
        return str(tok2)

    st.error("Niet ingelogd (access_token ontbreekt).")
    st.stop()
    return ""


def _ensure_postgrest_auth(sb) -> None:
    """
    Zorgt dat sb.table(...).select(...) calls de JWT gebruiken.
    Supabase-py heeft meerdere varianten; daarom meerdere fallbacks.
    """
    token = get_access_token()

    # 1) supabase-py: sb.postgrest.auth(token)
    try:
        sb.postgrest.auth(token)
        return
    except Exception:
        pass

    # 2) fallback: session headers (httpx session)
    try:
        # veel supabase-py builds hebben sb.postgrest.session.headers
        hdrs = sb.postgrest.session.headers
        hdrs["Authorization"] = f"Bearer {token}"
        # apikey is vaak al gezet, maar voor zekerheid:
        ak = st.secrets.get("SUPABASE_ANON_KEY", "").strip()
        if ak:
            hdrs["apikey"] = ak
        return
    except Exception:
        pass

    # 3) laatste poging: supabase auth set_session als beschikbaar
    try:
        sb.auth.set_session(token, token)  # refresh token onbekend; toch soms genoeg
    except Exception:
        pass


def get_auth_uid(sb) -> Optional[str]:
    """
    Haal user_id uit GoTrue op basis van token.
    """
    token = get_access_token()
    try:
        u = sb.auth.get_user(token)
        return u.user.id
    except Exception:
        return None


# -----------------------------
# Profile / role
# -----------------------------
def normalize_role(v) -> str:
    """
    Normaliseert role strings zoals:
    - "player"
    - "user_role_v2.player"
    - "player::something"
    """
    if v is None:
        return ""
    s = str(v).strip().lower()
    if "." in s:
        s = s.split(".")[-1]
    if "::" in s:
        s = s.split("::")[0]
    return s.strip()


def get_profile(sb) -> dict:
    """
    Leest public.profiles (user_id, role, player_id).
    Verwacht dat profiles.user_id = auth.uid()
    """
    _ensure_postgrest_auth(sb)

    uid = get_auth_uid(sb)
    if not uid:
        return {}

    try:
        resp = (
            sb.table("profiles")
            .select("user_id,role,player_id,created_at")
            .eq("user_id", uid)
            .maybe_single()
            .execute()
        )
        p = resp.data or {}
        p["role"] = normalize_role(p.get("role"))
        return p
    except Exception:
        return {}


# -----------------------------
# Players list + picker
# -----------------------------
def list_players(sb) -> pd.DataFrame:
    """
    Actieve spelers voor staff dropdown.
    """
    _ensure_postgrest_auth(sb)
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
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["player_id", "full_name", "is_active"])
        df["full_name"] = df["full_name"].astype(str).str.strip()
        return df
    except Exception:
        return pd.DataFrame(columns=["player_id", "full_name", "is_active"])


def pick_target_player(
    sb,
    profile: dict,
    label: str = "Speler",
    key: str = "pick_player",
) -> Tuple[Optional[str], Optional[str], pd.DataFrame]:
    """
    Staff: dropdown met actieve spelers.
    Player: geen dropdown (wordt in Player Page zelf geforceerd).
    """
    df = list_players(sb)
    if df.empty:
        return None, None, df

    labels = df["full_name"].tolist()
    default_i = 0

    # als staff zelf ook gekoppeld is aan player_id: preselect die speler
    my_pid = profile.get("player_id")
    if my_pid:
        hit = df.index[df["player_id"].astype(str) == str(my_pid)]
        if len(hit) > 0:
            default_i = int(hit[0])

    choice = st.selectbox(label, options=labels, index=default_i, key=key)
    row = df[df["full_name"] == choice].iloc[0]
    return str(row["player_id"]), str(row["full_name"]), df
