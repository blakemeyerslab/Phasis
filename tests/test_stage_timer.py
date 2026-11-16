from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from phasis import stage_timer


class StageTimerTests(unittest.TestCase):
    def test_stage_lines_are_appended_after_the_index_time(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(stage_timer, "_LOG_PATH", None):
            path = stage_timer.open_run_log(tmpdir)
            self.assertTrue(os.path.basename(path).startswith("runtime_"))
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("Indexing Time: 0s\n")
            captured = io.StringIO()
            with redirect_stdout(captured):
                with stage_timer.stage("Mapping"):
                    pass
                with self.assertRaises(ValueError):
                    with stage_timer.stage("Parsing alignments"):
                        raise ValueError("stage failed")

            with open(path, encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            self.assertEqual(lines[0], "Indexing Time: 0s")
            rows = stage_timer.read_stage_times(path)
            self.assertEqual([r["stage"] for r in rows], ["Mapping", "Parsing alignments"])
            self.assertTrue(all(r["seconds"] >= 0 for r in rows))
            self.assertIn("[stage-time] Mapping: 0:00:00", captured.getvalue())

    def test_without_a_run_log_nothing_is_written_and_nothing_fails(self):
        with mock.patch.object(stage_timer, "_LOG_PATH", None), redirect_stdout(io.StringIO()):
            with stage_timer.stage("Classification"):
                pass
            self.assertIsNone(stage_timer.current_run_log())

    def test_ensure_run_log_keeps_the_file_of_the_run(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(stage_timer, "_LOG_PATH", None):
            first = stage_timer.ensure_run_log(tmpdir)
            self.assertEqual(stage_timer.ensure_run_log(tmpdir), first)

    def test_format_duration(self):
        self.assertEqual(stage_timer.format_duration(4 * 60 + 36.4), "0:04:36")
        self.assertEqual(stage_timer.format_duration(3 * 3600 + 59), "3:00:59")


if __name__ == "__main__":
    unittest.main()
