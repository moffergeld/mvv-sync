from __future__ import annotations

import unittest

from datetime import date

from pages.Subscripts.week_session_labels import md_label, moment_axis, session_time, source_value


class WeekSessionLabelTests(unittest.TestCase):
    def test_reads_api_session_fields(self):
        extra = {"Session Type": "Match Day +3", "Session Start Time": "2026-09-29T10:43:58Z"}
        self.assertEqual(md_label(extra, "Practice"), "MD+3")
        self.assertEqual(session_time(extra), "10:43")

    def test_reads_nested_supabase_source_columns(self):
        extra = {"source_columns": {"sessionType": "Match Day -1", "sessionStartTime": "15:07:00"}}
        self.assertEqual(md_label(extra, "Practice"), "MD-1")
        self.assertEqual(source_value(extra, "Session Type"), "Match Day -1")

    def test_match_fallback_and_unknown_training(self):
        self.assertEqual(md_label({}, "Match"), "MD")
        self.assertEqual(md_label({}, "Practice"), "MD onbekend")

    def test_two_sessions_share_one_md_group_with_two_moments(self):
        rows = [
            {"datum": date(2026, 9, 29), "md_label": "MD+3", "session_time": "10:43", "event_group": "Training"},
            {"datum": date(2026, 9, 29), "md_label": "MD+3", "session_time": "14:41", "event_group": "Training"},
            {"datum": date(2026, 9, 30), "md_label": "MD-3", "session_time": "11:00", "event_group": "Training"},
        ]
        labels = moment_axis(rows)
        self.assertEqual([item["group_label"] for item in labels[:2]], ["MD+3 · 29/09", "MD+3 · 29/09"])
        self.assertEqual([item["moment_label"] for item in labels[:2]], ["Moment 1 · 10:43", "Moment 2 · 14:41"])
        self.assertEqual(labels[2]["moment_label"], "11:00")


if __name__ == "__main__":
    unittest.main()
