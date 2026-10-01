import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


def _cache_data(*_args, **_kwargs):
    def decorator(function):
        return function
    return decorator


sys.modules.setdefault("streamlit", SimpleNamespace(cache_data=_cache_data, secrets={}))

from pages.Subscripts import gps_hybrid_source as source


REQUESTED_COLUMNS = [
    "gps_id",
    "player_id",
    "player_name",
    "datum",
    "type",
    "event",
    "match_id",
    "total_distance",
]


def _row(gps_id: int, day: str, distance: float) -> dict:
    return {
        "gps_id": gps_id,
        "player_id": "player-1",
        "player_name": "Test Player",
        "datum": day,
        "type": "Practice",
        "event": "Summary",
        "match_id": None,
        "total_distance": distance,
    }


class HybridGpsSourceTests(unittest.TestCase):
    def test_unique_surname_and_initial_repairs_encoding_only_name_mismatch(self):
        mapped = source._extend_player_map_for_api_names(
            {"na�m matoug": "player-1", "other player": "player-2"},
            pd.Series(["Naim Matoug"]),
        )
        self.assertEqual(mapped["naim matoug"], "player-1")

    def test_every_gps_dashboard_consumer_uses_shared_hybrid_loader(self):
        root = Path(__file__).resolve().parents[1]
        consumers = [
            "app.py",
            "pages/02_Match_Reports.py",
            "pages/12_Session_Load.py",
            "pages/01_Week_Overview.py",
            "pages/15_Year_Report.py",
            "pages/16_Month_Report.py",
            "pages/17_Player_Report.py",
            "pages/18_Benchmarks_Page.py",
            "pages/Subscripts/gps_data_benchmarks_pages.py",
            "pages/Subscripts/gps_import_tab_export.py",
            "pages/Subscripts/player_tab_data.py",
        ]

        missing = [
            path
            for path in consumers
            if "load_hybrid_gps" not in (root / path).read_text(encoding="utf-8")
        ]
        self.assertEqual(missing, [])

    def test_navigation_has_no_beta_or_management_routes(self):
        root = Path(__file__).resolve().parents[1]
        roles_source = (root / "roles.py").read_text(encoding="utf-8")
        self.assertNotIn("_Beta.py", roles_source)
        self.assertNotIn("09_Management.py", roles_source)

        visible_page_names = [path.name for path in (root / "pages").glob("*.py")]
        self.assertFalse(any("Beta" in name for name in visible_page_names))
        self.assertNotIn("09_Management.py", visible_page_names)

    def test_boundary_combines_supabase_before_and_api_from_cutover(self):
        old = pd.DataFrame([_row(1, "2026-08-23", 1000)])
        api = pd.DataFrame([_row(-2, "2026-08-24", 2000), _row(-3, "2026-08-25", 3000)])

        with (
            patch.object(source, "_rest_get_paged", return_value=old) as supabase_fetch,
            patch.object(source, "_statsports_db_frame", return_value=api) as api_fetch,
        ):
            result = source.load_hybrid_gps(
                "token",
                REQUESTED_COLUMNS,
                start="2026-08-23",
                end="2026-08-25",
                event="Summary",
            )

        self.assertEqual(result["datum"].dt.strftime("%Y-%m-%d").tolist(), ["2026-08-23", "2026-08-24", "2026-08-25"])
        self.assertEqual(result["total_distance"].tolist(), [1000, 2000, 3000])
        self.assertIn("datum=lt.2026-08-24", supabase_fetch.call_args.args[2])
        self.assertIn("event=like.Summary*", supabase_fetch.call_args.args[2])
        api_fetch.assert_called_once()

    def test_period_before_cutover_never_calls_statsports(self):
        old = pd.DataFrame([_row(1, "2026-08-20", 1000)])
        with (
            patch.object(source, "_rest_get_paged", return_value=old),
            patch.object(source, "_statsports_db_frame") as api_fetch,
        ):
            result = source.load_hybrid_gps(
                "token",
                REQUESTED_COLUMNS,
                start="2026-08-01",
                end="2026-08-23",
                event="Summary",
            )

        self.assertEqual(len(result), 1)
        api_fetch.assert_not_called()

    def test_period_from_cutover_never_reads_gps_records(self):
        api = pd.DataFrame([_row(-2, "2026-08-24", 2000)])
        with (
            patch.object(source, "_rest_get_paged") as supabase_fetch,
            patch.object(source, "_statsports_db_frame", return_value=api),
        ):
            result = source.load_hybrid_gps(
                "token",
                REQUESTED_COLUMNS,
                start="2026-08-24",
                end="2026-08-24",
                event="Summary",
            )

        self.assertEqual(len(result), 1)
        supabase_fetch.assert_not_called()

    def test_api_filters_are_applied_before_page_receives_rows(self):
        api = pd.DataFrame(
            [
                _row(-1, "2026-08-24", 1000),
                {**_row(-2, "2026-08-24", 2000), "player_id": "player-2"},
                {**_row(-3, "2026-08-24", 3000), "event": "First Half"},
            ]
        )
        with patch.object(source, "_statsports_db_frame", return_value=api):
            result = source.load_hybrid_gps(
                "token",
                REQUESTED_COLUMNS,
                start="2026-08-24",
                end="2026-08-24",
                event="Summary",
                player_id="player-1",
            )

        self.assertEqual(result["total_distance"].tolist(), [1000])

    def test_numbered_summary_is_kept_for_double_sessions(self):
        api = pd.DataFrame(
            [
                _row(-1, "2026-08-24", 1000),
                {**_row(-2, "2026-08-24", 2000), "event": "Summary (2)"},
                {**_row(-3, "2026-08-24", 3000), "event": "Drill"},
            ]
        )
        with patch.object(source, "_statsports_db_frame", return_value=api):
            result = source.load_hybrid_gps(
                "token",
                REQUESTED_COLUMNS,
                start="2026-08-24",
                end="2026-08-24",
                event="Summary",
            )

        self.assertEqual(sorted(result["total_distance"].tolist()), [1000, 2000])


if __name__ == "__main__":
    unittest.main()
