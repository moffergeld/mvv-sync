import unittest
import sys
from types import SimpleNamespace

import pandas as pd


def _cache_data(*_args, **_kwargs):
    def decorator(function):
        return function
    return decorator


sys.modules.setdefault(
    "streamlit",
    SimpleNamespace(cache_data=_cache_data, session_state={}, secrets={}),
)
sys.modules.setdefault("roles", SimpleNamespace(cookie_mgr=lambda: SimpleNamespace(get=lambda *_args, **_kwargs: None)))

from acwr_settings import ACWR_MODE_STANDARD
from pages.Subscripts.acwr_shared import build_player_monitor, build_weekly_loads


class AcwrSharedTests(unittest.TestCase):
    def test_weekly_loads_aggregate_four_required_metrics(self):
        source = pd.DataFrame(
            [
                {"player_id": "p1", "player_name": "Speler", "datum": "2026-09-28", "total_distance": 1000, "total_distance_zone_5": 50, "total_distance_zone_6": 10, "high_metabolic_load_distance": 200},
                {"player_id": "p1", "player_name": "Speler", "datum": "2026-09-30", "total_distance": 2000, "total_distance_zone_5": 70, "total_distance_zone_6": 20, "high_metabolic_load_distance": 300},
            ]
        )
        weekly = build_weekly_loads(source)
        self.assertEqual(float(weekly.iloc[0]["total_distance"]), 3000)
        self.assertEqual(float(weekly.iloc[0]["total_distance_zone_5"]), 120)
        self.assertEqual(float(weekly.iloc[0]["total_distance_zone_6"]), 30)
        self.assertEqual(float(weekly.iloc[0]["high_metabolic_load_distance"]), 500)

    def test_monitor_reports_remaining_load_to_point_eight_and_one_point_five(self):
        starts = pd.date_range("2026-09-07", periods=5, freq="7D")
        weekly = pd.DataFrame(
            [
                {
                    "week_start": week,
                    "week_label": "",
                    "player_id": "p1",
                    "player_name": "Speler",
                    "total_distance": value,
                    "total_distance_zone_5": value / 20,
                    "total_distance_zone_6": value / 100,
                    "high_metabolic_load_distance": value / 5,
                }
                for week, value in zip(starts, [1000, 1000, 1000, 1000, 500])
            ]
        )
        monitor = build_player_monitor(weekly, week_start=starts[-1], mode=ACWR_MODE_STANDARD).iloc[0]
        self.assertAlmostEqual(float(monitor["total_distance_acwr"]), 0.5)
        self.assertAlmostEqual(float(monitor["total_distance_to_low"]), 300.0)
        self.assertAlmostEqual(float(monitor["total_distance_to_high"]), 1000.0)

    def test_missing_calendar_week_stays_unknown(self):
        source = pd.DataFrame(
            [
                {"player_id": "p1", "player_name": "Speler", "datum": "2026-08-03", "total_distance": 1000, "total_distance_zone_5": 10, "total_distance_zone_6": 5, "high_metabolic_load_distance": 50},
                {"player_id": "p1", "player_name": "Speler", "datum": "2026-08-17", "total_distance": 1200, "total_distance_zone_5": 12, "total_distance_zone_6": 6, "high_metabolic_load_distance": 60},
            ]
        )

        weekly = build_weekly_loads(source, through_week=pd.Timestamp("2026-08-17"))

        self.assertEqual(len(weekly), 3)
        self.assertTrue(pd.isna(weekly.iloc[1]["total_distance"]))


if __name__ == "__main__":
    unittest.main()
