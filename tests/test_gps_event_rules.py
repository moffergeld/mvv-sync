from __future__ import annotations

import unittest

import pandas as pd

from pages.Subscripts.gps_event_rules import (
    MATCH_FIRST,
    MATCH_FULL,
    MATCH_SECOND,
    select_match_phase_rows,
)


class GpsEventRulesTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame(
            {
                "event": [
                    "Summary",
                    "Entire Session - Live",
                    "Match-Entire Match",
                    "Match-Halfes-First Half",
                    "Match-Halfes-Second Half",
                ],
                "total_distance": [9999, 9999, 9999, 4500, 4300],
            }
        )

    def test_full_match_uses_only_both_halves(self):
        selected = select_match_phase_rows(self.rows, MATCH_FULL)
        self.assertEqual(selected["total_distance"].tolist(), [4500, 4300])

    def test_halves_can_be_selected_independently(self):
        first = select_match_phase_rows(self.rows, MATCH_FIRST)
        second = select_match_phase_rows(self.rows, MATCH_SECOND)
        self.assertEqual(first["total_distance"].tolist(), [4500])
        self.assertEqual(second["total_distance"].tolist(), [4300])
