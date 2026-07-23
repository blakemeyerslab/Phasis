from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from phasis import runtime as rt
from phasis.stages import cluster_aggregation


def _serial_parallel_runner(func, iterable, **_kwargs):
    return [func(item) for item in iterable]


class ClusterAggregationProgressTests(unittest.TestCase):
    def test_reports_single_core_consolidation_after_parallel_file_parsing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cluster_path = os.path.join(tmpdir, "libA.24-PHAS.candidate.clusters")
            with open(cluster_path, "w", encoding="utf-8") as handle:
                handle.write(">cluster = libA-1\n")
                handle.write(
                    "1\tw\t100\t24\t1\t5\t0.1\t1\t1\t0.1\t0.1\t0.1\t1\t1\t0.1\t0.1\ttag-1\tAAAA\n"
                )

            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with mock.patch.multiple(
                    rt,
                    phase=24,
                    concat_libs=False,
                    memFile=os.path.join(tmpdir, "phasis.mem"),
                    compress_intermediates=False,
                    create=True,
                ):
                    with mock.patch.object(
                        cluster_aggregation,
                        "run_parallel_with_progress",
                        side_effect=_serial_parallel_runner,
                    ):
                        captured = io.StringIO()
                        with redirect_stdout(captured):
                            result = cluster_aggregation.aggregate_and_write_processed_clusters(
                                [cluster_path],
                            )
            finally:
                os.chdir(old_cwd)

        self.assertEqual(len(result), 1)
        log = captured.getvalue()
        self.assertIn("Consolidating 1 candidate-cluster records", log)
        self.assertIn("uses one CPU core and may take several minutes", log)
        self.assertIn("Writing consolidated processed clusters", log)


if __name__ == "__main__":
    unittest.main()
