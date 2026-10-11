from __future__ import annotations

import unittest

import pandas as pd

from pages.Subscripts.median_reference import build_selected_medians, display_median_table, median_map
from report_monitoring import build_monitoring_grouped_summary


class MedianReferenceTests(unittest.TestCase):
    def test_selected_players_control_each_metric_median(self) -> None:
        frame = pd.DataFrame(
            {
                "player": ["A", "B", "C"],
                "distance": [1000, 2000, 9000],
                "hr_seconds": [60, 180, 1200],
            }
        )
        summary = build_selected_medians(
            frame,
            ["A", "B"],
            [
                ("distance", "TD", "m", 1.0, 0),
                ("hr_seconds", "HR Z1", "min", 1 / 60, 1),
            ],
            player_column="player",
        )

        values = median_map(summary)
        self.assertEqual(values["distance"], 1500.0)
        self.assertEqual(values["hr_seconds"], 2.0)
        self.assertEqual(summary["players"].tolist(), [2, 2])

    def test_empty_selection_has_no_reference_values(self) -> None:
        frame = pd.DataFrame({"player": ["A"], "distance": [1000]})
        summary = build_selected_medians(
            frame,
            [],
            [("distance", "TD", "m", 1.0, 0)],
            player_column="player",
        )
        self.assertTrue(summary.empty)
        self.assertTrue(display_median_table(summary).empty)

    def test_wellness_and_rpe_medians_are_grouped_per_day(self) -> None:
        frame = pd.DataFrame(
            {
                "entry_date": ["2026-10-01", "2026-10-01", "2026-10-02"],
                "player_id": ["1", "2", "1"],
                "player_name": ["A", "B", "A"],
                "muscle_soreness": [2.0, 8.0, 4.0],
                "fatigue": [3.0, 7.0, 5.0],
                "sleep_quality": [4.0, 6.0, 8.0],
                "stress": [1.0, 9.0, 3.0],
                "mood": [5.0, 9.0, 7.0],
                "wellness_avg": [3.0, 7.8, 5.4],
                "readiness_score": [70.0, 40.0, 60.0],
                "avg_rpe": [4.0, 8.0, 6.0],
            }
        )

        grouped = build_monitoring_grouped_summary(frame, "day")

        self.assertEqual(grouped.loc[0, "muscle_soreness_median"], 5.0)
        self.assertEqual(grouped.loc[0, "avg_rpe_median"], 6.0)
        self.assertEqual(grouped.loc[0, "wellness_players"], 2)
        self.assertEqual(grouped.loc[0, "rpe_players"], 2)
        self.assertEqual(grouped.loc[1, "muscle_soreness_median"], 4.0)


if __name__ == "__main__":
    unittest.main()
