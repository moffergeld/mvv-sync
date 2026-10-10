from __future__ import annotations

import math
import unittest

import pandas as pd

from pages.Subscripts.time_metrics import duration_seconds, normalize_hr_zone_seconds


class DurationSecondsTests(unittest.TestCase):
    def test_numeric_seconds_are_preserved(self) -> None:
        self.assertEqual(duration_seconds(125), 125.0)
        self.assertEqual(duration_seconds("125.5"), 125.5)

    def test_clock_and_interval_values_are_converted(self) -> None:
        self.assertEqual(duration_seconds("12:34"), 754.0)
        self.assertEqual(duration_seconds("01:02:03"), 3723.0)
        self.assertEqual(duration_seconds("1 day 01:02:03"), 90123.0)

    def test_missing_and_invalid_values_return_none(self) -> None:
        for value in (None, "", "nan", "<NA>", "not-a-duration", math.nan, True):
            self.assertIsNone(duration_seconds(value))

    def test_hr_zones_prefer_absolute_values_and_fall_back_to_legacy_minutes(self) -> None:
        frame = pd.DataFrame(
            {
                "absolute_1": ["00:12:30", None],
                "legacy_1": [99, "7.5"],
                "legacy_2": [3, None],
            }
        )

        normalized = normalize_hr_zone_seconds(
            frame,
            ["absolute_1", "absolute_2", "absolute_3"],
            ["legacy_1", "legacy_2"],
        )

        self.assertEqual(normalized["absolute_1"].tolist(), [750.0, 450.0])
        self.assertEqual(normalized["absolute_2"].tolist(), [180.0, 0.0])
        self.assertEqual(normalized["absolute_3"].tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
