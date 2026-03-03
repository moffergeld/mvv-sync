# auth_session.py
# ============================================================
# Thin wrapper (Optie A)
# - roles.py is the single source of truth for auth/session/cookies
# - auth_session.py blijft bestaan zodat oude imports niet breken
# ============================================================

from __future__ import annotations

from typing import Optional, Tuple

import streamlit as st

from roles import (
    get_sb as get_sb_client,
    cookie_mgr,
    set_tokens_in_cookie,
    clear_tokens_in_cookie,
    try_restore_or_refresh_session,
)


def ensure_auth_restored(sb=None) -> Tuple[bool, Optional[str]]:
    """
    Backwards compatible wrapper.
    """
    if sb is None:
        sb = get_sb_client()
    ok = try_restore_or_refresh_session(sb)
    token = st.session_state.get("access_token")
    return bool(ok and token), (str(token) if token else None)


def hard_logout(sb=None) -> None:
    """
    Backwards compatible logout. Uses roles helpers.
    """
    try:
        if sb is None:
            sb = get_sb_client()
        if sb is not None:
            sb.auth.sign_out()
    except Exception:
        pass

    clear_tokens_in_cookie()
    st.session_state.clear()
