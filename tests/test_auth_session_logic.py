from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_roles_module():
    streamlit = types.ModuleType("streamlit")
    streamlit.session_state = {}
    streamlit.secrets = {}
    streamlit.sidebar = types.SimpleNamespace()
    streamlit.cache_data = lambda *args, **kwargs: lambda func: func

    components = types.ModuleType("extra_streamlit_components")
    components.CookieManager = object

    supabase = types.ModuleType("supabase")
    supabase.create_client = lambda *args, **kwargs: None

    previous = {
        name: sys.modules.get(name)
        for name in ("streamlit", "extra_streamlit_components", "supabase")
    }
    sys.modules["streamlit"] = streamlit
    sys.modules["extra_streamlit_components"] = components
    sys.modules["supabase"] = supabase

    try:
        module_path = Path(__file__).resolve().parents[1] / "roles.py"
        spec = importlib.util.spec_from_file_location("roles_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class FakeCookieManager:
    def __init__(self, values):
        self.values = values

    def get(self, key):
        return self.values.get(key)


class FakeAuth:
    def __init__(self, session):
        self.session = session
        self.set_calls = []
        self.refresh_calls = []

    def set_session(self, access, refresh):
        self.set_calls.append((access, refresh))
        return types.SimpleNamespace(session=self.session)

    def refresh_session(self, refresh):
        self.refresh_calls.append(refresh)
        return types.SimpleNamespace(session=self.session)


class AuthSessionTests(unittest.TestCase):
    def setUp(self):
        self.roles = load_roles_module()
        self.roles.st.session_state.clear()
        self.roles.COOKIE_SETTLE_SECONDS = 0
        self.roles.set_tokens_in_cookie = lambda *args, **kwargs: None
        self.session = types.SimpleNamespace(
            access_token="new-access",
            refresh_token="new-refresh",
        )

    def test_restore_with_access_token_does_not_refresh_twice(self):
        auth = FakeAuth(self.session)
        sb = types.SimpleNamespace(auth=auth, postgrest=types.SimpleNamespace())
        self.roles.cookie_mgr = lambda: FakeCookieManager(
            {"sb_access": "access", "sb_refresh": "refresh", "sb_email": "user@example.com"}
        )
        self.roles._set_postgrest_auth_safely = lambda *args: None

        self.assertTrue(self.roles.try_restore_or_refresh_session(sb))
        self.assertEqual(auth.set_calls, [("access", "refresh")])
        self.assertEqual(auth.refresh_calls, [])

    def test_restore_without_access_token_refreshes_once(self):
        auth = FakeAuth(self.session)
        sb = types.SimpleNamespace(auth=auth, postgrest=types.SimpleNamespace())
        self.roles.cookie_mgr = lambda: FakeCookieManager(
            {"sb_access": None, "sb_refresh": "refresh", "sb_email": "user@example.com"}
        )
        self.roles._set_postgrest_auth_safely = lambda *args: None

        self.assertTrue(self.roles.try_restore_or_refresh_session(sb))
        self.assertEqual(auth.set_calls, [])
        self.assertEqual(auth.refresh_calls, ["refresh"])

    def test_clearing_auth_state_closes_client(self):
        closed = []
        self.roles.st.session_state["_sb_client"] = types.SimpleNamespace(
            close=lambda: closed.append(True)
        )

        self.roles.clear_auth_state()

        self.assertEqual(closed, [True])
        self.assertNotIn("_sb_client", self.roles.st.session_state)


if __name__ == "__main__":
    unittest.main()
