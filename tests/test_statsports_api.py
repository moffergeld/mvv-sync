from __future__ import annotations

import unittest
from datetime import date

from pages.Subscripts.statsports_api import (
    fetch_statsports_range,
    sessions_to_dataframe,
    validate_api_key,
)


API_KEY = "00000000-0000-0000-0000-000000000001"


class _Response:
    status_code = 200
    ok = True
    text = "payload"

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _Session:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response([{"id": len(self.calls)}])


class StatsportsApiTests(unittest.TestCase):
    def test_api_key_must_be_uuid(self):
        self.assertEqual(validate_api_key(f"  {API_KEY}  "), API_KEY)
        with self.assertRaises(ValueError):
            validate_api_key("not-a-key")

    def test_range_is_requested_in_seven_day_chunks(self):
        http = _Session()
        result = fetch_statsports_range(
            API_KEY,
            date(2026, 8, 24),
            date(2026, 9, 2),
            session=http,
        )

        self.assertEqual(len(http.calls), 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(http.calls[0][1]["json"]["sessionStartDate"], "2026-08-24T00:00:00Z")
        self.assertEqual(http.calls[0][1]["json"]["sessionEndDate"], "2026-08-30T23:59:59Z")
        self.assertEqual(http.calls[1][1]["json"]["sessionStartDate"], "2026-08-31T00:00:00Z")

    def test_payload_is_normalised_and_raw_context_is_preserved(self):
        payload = [{
            "id": "session-1",
            "sessionName": "MD-1",
            "sessionDetails": {
                "sessionDate": "2026-09-30T00:00:00Z",
                "sessionType": "Training",
                "startTime": "2026-09-30T09:00:00Z",
            },
            "sessionPlayers": [{
                "playerDetails": {"id": "player-1", "displayName": "Test Speler"},
                "drills": [
                    {
                        "id": "drill-1",
                        "drillName": "Passing",
                        "startTime": "2026-09-30T09:00:00Z",
                        "drillKpi": {"totalTime": 600, "maxSpeed": 10, "vendorMetric": 42},
                    },
                    {
                        "id": "drill-2",
                        "drillName": "Passing",
                        "startTime": "2026-09-30T09:20:00Z",
                        "drillKpi": {"totalTime": 300},
                    },
                ],
            }],
        }]

        frame = sessions_to_dataframe(payload)

        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[0]["Type"], "Practice")
        self.assertEqual(frame.iloc[0]["totalTime"], 10)
        self.assertEqual(frame.iloc[0]["maxSpeed"], 36)
        self.assertEqual(frame["Event"].tolist(), ["Passing (1)", "Passing (2)"])
        self.assertEqual(frame.iloc[0]["STATSports Raw"]["drill"]["drillKpi"]["vendorMetric"], 42)
        self.assertTrue(frame.attrs["preserve_extra_metrics"])

    def test_live_entire_session_becomes_the_single_summary_row(self):
        payload = [{
            "id": "session-1",
            "sessionName": "MD Opponent",
            "sessionDetails": {"sessionDate": "2026-09-30T00:00:00Z", "sessionType": "Match Day"},
            "sessionPlayers": [{
                "playerDetails": {"id": "player-1", "displayName": "Test Speler"},
                "drills": [
                    {"drillName": "Entire Session", "startTime": "2026-09-30T17:00:00Z", "drillKpi": {"totalTime": 9000}},
                    {"drillName": "Entire Session - Live", "startTime": "2026-09-30T17:10:00Z", "drillKpi": {"totalTime": 7200}},
                    {"drillName": "Match-Entire Match", "startTime": "2026-09-30T18:00:00Z", "drillKpi": {"totalTime": 6000}},
                ],
            }],
        }]

        frame = sessions_to_dataframe(payload)

        summary = frame.loc[frame["Event"].eq("Summary")]
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]["totalTime"], 120)
        self.assertEqual(summary.iloc[0]["STATSports Raw"]["drill"]["drillName"], "Entire Session - Live")
        self.assertEqual(set(frame["Event"]), {"Entire Session", "Summary", "Match-Entire Match"})

    def test_load_summary_is_absent_when_live_entire_session_is_absent(self):
        payload = [{
            "sessionName": "MD Opponent",
            "sessionDetails": {"sessionDate": "2026-09-30T00:00:00Z", "sessionType": "Match Day"},
            "sessionPlayers": [{
                "playerDetails": {"displayName": "Test Speler"},
                "drills": [
                    {"drillName": "Entire Session", "drillKpi": {"totalTime": 9000}},
                    {"drillName": "Match-Entire Match", "drillKpi": {"totalTime": 6000}},
                ],
            }],
        }]

        frame = sessions_to_dataframe(payload)

        self.assertNotIn("Summary", frame["Event"].tolist())

    def test_two_live_sessions_on_one_day_are_both_kept(self):
        def session(session_id, start, distance):
            return {
                "id": session_id,
                "sessionName": "MD-3",
                "sessionDetails": {
                    "sessionDate": "2026-09-30T00:00:00Z",
                    "sessionType": "Training",
                    "startTime": start,
                },
                "sessionPlayers": [{
                    "playerDetails": {"displayName": "Test Speler"},
                    "drills": [{
                        "drillName": "Entire Session - Live",
                        "startTime": start,
                        "drillKpi": {"distanceTotal": distance},
                    }],
                }],
            }

        frame = sessions_to_dataframe(
            [
                session("morning", "2026-09-30T08:00:00Z", 3000),
                session("afternoon", "2026-09-30T14:00:00Z", 4000),
            ]
        )

        self.assertEqual(frame["Event"].tolist(), ["Summary", "Summary"])
        self.assertEqual(frame["Session ID"].tolist(), ["morning", "afternoon"])


if __name__ == "__main__":
    unittest.main()
