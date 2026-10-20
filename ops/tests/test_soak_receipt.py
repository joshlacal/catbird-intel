import importlib.util
from pathlib import Path
import unittest
from datetime import datetime, timezone

spec = importlib.util.spec_from_file_location("soak", Path(__file__).parents[1] / "evaluate_soak_receipt.py")
soak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soak)


def evaluate(data):
    return soak.evaluate(data, now=datetime(2026, 10, 22, 14, tzinfo=timezone.utc))

class SoakTests(unittest.TestCase):
    def test_absent_probe_never_passes(self):
        result = evaluate({})
        self.assertEqual(result["qualification"], "UNQUALIFIED")
        self.assertEqual(result["gates"]["controlled_create_delete"]["status"], "UNPERFORMED")
        self.assertFalse(result["deployment_authorized"])

    def test_freshness_is_not_visibility_latency(self):
        row = dict(status="measured", source_receipt="receipt.json", observed_start="a", observed_end="b", clock="ledger_head_age", sample_count=100, p99_ms=10, max_ms=10)
        self.assertEqual(evaluate({"gates": {"visibility_latency": row}})["gates"]["visibility_latency"]["status"], "UNKNOWN")

    def test_negative_and_nan_metrics_cannot_pass(self):
        for value in (-1, float("nan"), True):
            row = dict(status="measured", source_receipt="receipt.json", observed_start="a", observed_end="b", visible_deleted_records=value)
            self.assertEqual(evaluate({"gates": {"delete_invariant": row}})["gates"]["delete_invariant"]["status"], "UNKNOWN")

    def test_actual_violation_fails(self):
        row = dict(status="measured", source_receipt="receipt.json", observed_start="a", observed_end="b", visible_deleted_records=1)
        self.assertEqual(evaluate({"gates": {"delete_invariant": row}})["gates"]["delete_invariant"]["status"], "FAIL")

    def test_invalid_time_and_short_windows(self):
        cases = [("a", "b"), ("2026-10-19T13:00:00", "2026-10-22T13:00:00Z"),
                 ("2026-10-22T13:00:00Z", "2026-10-19T13:00:00Z"),
                 ("2026-10-22T12:59:59Z", "2026-10-22T13:00:00Z")]
        for start, end in cases:
            row = dict(status="measured", source_receipt="receipt.json", observed_start=start, observed_end=end, coverage=1)
            self.assertEqual(evaluate({"gates": {"continuity": row}})["gates"]["continuity"]["status"], "UNKNOWN")

    def test_full_continuity_separate_from_final_day(self):
        row = dict(status="measured", source_receipt="receipt.json", observed_start="2026-10-19T13:00:00Z", observed_end="2026-10-22T13:00:00Z", coverage=1)
        result = evaluate({"gates": {"continuity": row}})
        self.assertEqual(result["gates"]["continuity"]["status"], "PASS")
        self.assertEqual(result["gates"]["final_24h_continuity"]["status"], "UNPERFORMED")
        self.assertEqual(result["qualification"], "UNQUALIFIED")

    def test_fractional_count_and_out_of_range_fraction(self):
        for gate, metric, value in [("delete_invariant", "visible_deleted_records", .5), ("continuity", "coverage", 1.1)]:
            row = dict(status="measured", source_receipt="receipt.json", observed_start="2026-10-19T13:00:00Z", observed_end="2026-10-22T13:00:00Z")
            row[metric] = value
            self.assertEqual(evaluate({"gates": {gate: row}})["gates"][gate]["status"], "UNKNOWN")

    def test_violation_survives_missing_provenance_and_metric(self):
        result = evaluate({"gates": {"cursor_completeness": {"status": "measured", "unresolved_notices": 1}}})
        gate = result["gates"]["cursor_completeness"]
        self.assertEqual(gate["status"], "FAIL")
        self.assertIn("unresolved_notices", gate["failed_metrics"])
        self.assertTrue(gate["reasons"])

    def test_merge_trend_requires_final_six_hours(self):
        row = dict(status="measured", source_receipt="receipt.json", observed_start="2026-10-22T08:00:00Z", observed_end="2026-10-22T13:00:00Z", monotone_growth_windows_6h=0)
        self.assertEqual(evaluate({"gates": {"merge_trend": row}})["gates"]["merge_trend"]["status"], "UNKNOWN")
        row["observed_start"] = "2026-10-22T07:00:00Z"
        self.assertEqual(evaluate({"gates": {"merge_trend": row}})["gates"]["merge_trend"]["status"], "PASS")

    def test_probe_and_audit_must_be_within_target_soak(self):
        base = dict(status="measured", source_receipt="receipt.json", write_authorization_receipt="approval.json", create_observed=1, delete_observed=1,
                    sampling_rule="1000_or_all_if_fewer", eligible_delete_population=1, sample_count=1, unexplained_conflicts=0, resurrections=0)
        for gate in ("controlled_create_delete", "cross_archive"):
            for start, end in [("2020-01-01T00:00:00Z", "2020-01-01T01:00:00Z"),
                               ("2026-10-22T13:00:00Z", "2026-10-22T13:30:00Z")]:
                row = dict(base, observed_start=start, observed_end=end)
                self.assertEqual(evaluate({"gates": {gate: row}})["gates"][gate]["status"], "UNKNOWN")
            row = dict(base, observed_start="2026-10-22T12:00:00Z", observed_end="2026-10-22T13:00:00Z")
            self.assertEqual(evaluate({"gates": {gate: row}})["gates"][gate]["status"], "PASS")

    def test_future_collection_cannot_pass(self):
        row = dict(status="measured", source_receipt="receipt.json", observed_start="2026-10-19T13:00:00Z", observed_end="2026-10-22T13:00:00Z", coverage=1)
        result = soak.evaluate({"gates": {"continuity": row}}, now=datetime(2026, 10, 20, tzinfo=timezone.utc))
        self.assertEqual(result["gates"]["continuity"]["status"], "UNKNOWN")

if __name__ == "__main__":
    unittest.main()
