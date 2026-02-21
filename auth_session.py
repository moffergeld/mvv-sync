# auth_session.py
from __future__ import annotations

import time
from typing import Optional, Tuple

import streamlit as st
import extra_streamlit_components as stx
from supabase import create_client


@st.cache_resource
def get_sb_client():
    url = st.secrets.get("SUPABASE_URL", "").strip()
    key = st.secrets.get("SUPABASE_ANON_KEY", "").strip()
    if not url or not key:
        return None
    return create_client(url, key)


def cookie_mgr():
    """
    Maak CookieManager maar 1x per run aan, anders krijg je
    StreamlitDuplicateElementKey (default key='init').
    """
    if "_cookie_mgr_instance" not in st.session_state:
        st.session_state["_cookie_mgr_instance"] = stx.CookieManager(key="mvv_cookie_mgr")
    return st.session_state["_cookie_mgr_instance"]


def set_tokens_in_cookie(access_token: str, refresh_token: str, email: str | None = None):
    cm = cookie_mgr()
    cm.set("sb_access", str(access_token or ""), max_age=60 * 60, key="set_sb_access")               # 1 uur
    cm.set("sb_refresh", str(refresh_token or ""), max_age=60 * 60 * 24 * 30, key="set_sb_refresh")  # 30 dagen
    if email:
        cm.set("sb_email", str(email), max_age=60 * 60 * 24 * 30, key="set_sb_email")


def clear_tokens_in_cookie():
    cm = cookie_mgr()
    cm.set("sb_access", "", max_age=1, key="clear_sb_access")
    cm.set("sb_refresh", "", max_age=1, key="clear_sb_refresh")
    cm.set("sb_email", "", max_age=1, key="clear_sb_email")


def set_postgrest_auth_safely(sb, token: Optional[str]):
    if not sb or not token:
        return
    try:
        sb.postgrest.auth(token)
    except Exception:
        pass


def get_access_token_from_state() -> Optional[str]:
    tok = st.session_state.get("access_token")
    if tok:
        return str(tok)
    sess = st.session_state.get("sb_session")
    if sess is not None:
        token = getattr(sess, "access_token", None)
        if token:
            return str(token)
    return None


def try_restore_or_refresh_session(sb=None) -> bool:
    """
    Herstel sessie uit cookies wanneer Streamlit session_state leeg is geraakt.
    """
    if sb is None:
        sb = get_sb_client()
    if sb is None:
        st.session_state["auth_err"] = "Supabase client niet beschikbaar"
        return False

    cm = cookie_mgr()
    access = cm.get("sb_access")
    refresh = cm.get("sb_refresh")
    email = cm.get("sb_email")

    if email and not st.session_state.get("user_email"):
        st.session_state["user_email"] = email

    if not refresh:
        return False

    last_err = None
    for _ in range(2):
        try:
            if access:
                try:
                    sb.auth.set_session(access, refresh)
                except Exception:
                    pass

            refreshed = sb.auth.refresh_session(refresh)
            sess = getattr(refreshed, "session", None)

            if sess and getattr(sess, "access_token", None) and getattr(sess, "refresh_token", None):
                st.session_state["access_token"] = sess.access_token
                st.session_state["sb_session"] = sess
                set_postgrest_auth_safely(sb, sess.access_token)
                set_tokens_in_cookie(sess.access_token, sess.refresh_token, st.session_state.get("user_email"))
                time.sleep(0.35)
                return True

        except Exception as e:
            last_err = e
            time.sleep(0.35)

    st.session_state["auth_err"] = str(last_err) if last_err else "Unknown restore error"
    return False


def ensure_auth_restored(sb=None) -> Tuple[bool, Optional[str]]:
    token = get_access_token_from_state()
    if token:
        if sb is None:
            sb = get_sb_client()
        set_postgrest_auth_safely(sb, token)
        return True, token

    ok = try_restore_or_refresh_session(sb=sb)
    token = get_access_token_from_state()
    return bool(ok and token), token


def hard_logout(sb=None):
    try:
        if sb is None:
            sb = get_sb_client()
        if sb is not None:
            sb.auth.sign_out()
    except Exception:
        pass

    clear_tokens_in_cookie()
    time.sleep(0.35)
    st.session_state.clear()
