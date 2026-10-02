from __future__ import annotations

import unittest

from datetime import date

from pages.Subscripts.week_session_labels import is_goalkeeper_position, md_label, moment_axis, session_time, source_value


class WeekSessionLabelTests(unittest.TestCase):
    def test_reads_api_session_fields(self):
        extra = {"Session Type": "Match Day +3", "Session Start Time": "2026-09-29T10:43:58Z"}
        self.assertEqual(md_label(extra, "Practice"), "MD+3")
        self.assertEqual(session_time(extra), "10:43")

    def test_reads_nested_supabase_source_columns(self):
        extra = {"source_columns": {"sessionType": "Match Day -1", "sessionStartTime": "15:07:00"}}
        self.assertEqual(md_label(extra, "Practice"), "MD-1")
        self.assertEqual(source_value(extra, "Session Type"), "Match Day -1")

    def test_reads_direct_statsports_columns(self):
        self.assertEqual(md_label({}, "Practice", "Match Day -2"), "MD-2")
        self.assertEqual(session_time({}, "2026-10-01T14:35:00Z"), "14:35")

    def test_goalkeeper_positions_are_recognised_without_matching_field_players(self):
        for value in ("GK", "Keeper", "Goalkeeper", "Doelman", "GK / Goalkeeper"):
            self.assertTrue(is_goalkeeper_position(value))
        for value in ("CB", "Forward", "Middenvelder", ""):
            self.assertFalse(is_goalkeeper_position(value))

    def test_match_fallback_and_unknown_training(self):
        self.assertEqual(md_label({}, "Match"), "MD")
        self.assertEqual(md_label({}, "Practice"), "MD onbekend")

    def test_two_sessions_share_one_day_group_with_md_and_time_per_column(self):
        rows = [
            {"datum": date(2026, 9, 29), "md_label": "MD+3", "session_time": "10:43", "event_group": "Training"},
            {"datum": date(2026, 9, 29), "md_label": "MD", "session_time": "14:41", "event_group": "Match"},
            {"datum": date(2026, 9, 30), "md_label": "MD-3", "session_time": "11:00", "event_group": "Training"},
        ]
        labels = moment_axis(rows)
        self.assertEqual([item["group_label"] for item in labels[:2]], ["Di · 29/09", "Di · 29/09"])
        self.assertEqual([item["moment_label"] for item in labels[:2]], ["MD+3 · 10:43", "MD · 14:41"])
        self.assertEqual(labels[2]["group_label"], "Wo · 30/09")
        self.assertEqual(labels[2]["moment_label"], "MD-3 · 11:00")


if __name__ == "__main__":
    unittest.main()
