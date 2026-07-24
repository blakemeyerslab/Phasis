from __future__ import annotations

import gzip
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import pandas as pd

from phasis import runtime as rt
from phasis.stages import cluster_aggregation


def _serial_parallel_runner(func, iterable, **_kwargs):
    return [func(item) for item in iterable]


def _candidate_line(
    chromosome: int,
    strand: str,
    position: int,
    tag: str,
    *,
    pval_h_f: float = 0.1,
    tag_seq: str = "AAAA",
) -> str:
    return (
        f"{chromosome}\t{strand}\t{position}\t21\t1\t5\t{pval_h_f}\t1\t1\t0.1\t0.1\t"
        f"0.1\t1\t1\t0.1\t0.1\t{tag}\t{tag_seq}\n"
    )


def _write_candidate(path: str, clusters, *, gzipped: bool = False) -> None:
    opener = gzip.open if gzipped else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for cluster_id, rows in clusters:
            handle.write(f">cluster = {cluster_id}\n")
            for row in rows:
                handle.write(_candidate_line(*row))


class ClusterAggregationExternalMergeTests(unittest.TestCase):
    def _run_aggregation(self, paths, tmpdir: str, *, compress_intermediates: bool = False):
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            with mock.patch.multiple(
                rt,
                phase=21,
                concat_libs=False,
                memFile=os.path.join(tmpdir, "phasis.mem"),
                compress_intermediates=compress_intermediates,
                create=True,
            ):
                with mock.patch.dict(
                    os.environ,
                    {
                        "PHASIS_CLUSTER_AGGREGATION_CHUNK_ROWS": "2",
                        "PHASIS_CLUSTER_AGGREGATION_MERGE_FAN_IN": "2",
                    },
                    clear=False,
                ):
                    with mock.patch.object(
                        cluster_aggregation,
                        "run_parallel_with_progress",
                        side_effect=_serial_parallel_runner,
                    ):
                        captured = io.StringIO()
                        with redirect_stdout(captured):
                            result = cluster_aggregation.aggregate_and_write_processed_clusters(paths)
        finally:
            os.chdir(old_cwd)
        return result, captured.getvalue()

    def test_disk_backed_merge_matches_legacy_order_and_cleans_spills(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            a_path = os.path.join(tmpdir, "alpha.21-PHAS.candidate.clusters")
            z_path = os.path.join(tmpdir, "zeta.21-PHAS.candidate.clusters")
            _write_candidate(
                a_path,
                [
                    ("cluster-B", [(1, "w", 200, "a-b-200")]),
                    ("cluster-A", [(1, "w", 120, "a-a-120"), (1, "c", 100, "a-a-100")]),
                ],
            )
            _write_candidate(
                z_path,
                [
                    ("cluster-A", [(1, "w", 100, "z-a-100"), (1, "w", 80, "z-a-080")]),
                    ("cluster-C", [(10, "c", 50, "z-c-050")]),
                ],
            )

            # This is the exact former in-memory aggregation behavior.
            legacy_rows = []
            for path in sorted([z_path, a_path], key=os.path.basename):
                legacy_rows.extend(cluster_aggregation.process_single_lib_cluster(path))
            legacy_frame = pd.DataFrame(
                legacy_rows,
                columns=cluster_aggregation.PROCESSED_CLUSTER_COLUMNS,
            ).sort_values(by=["clusterID", "pos"])

            result, log = self._run_aggregation([z_path, a_path], tmpdir)

            expected_path = os.path.join(tmpdir, "expected.tsv")
            legacy_frame.to_csv(expected_path, sep="\t", index=False, header=True)
            output_path = os.path.join(tmpdir, "21_processed_clusters.tab")
            with open(expected_path, "rb") as expected, open(output_path, "rb") as actual:
                self.assertEqual(actual.read(), expected.read())
            pd.testing.assert_frame_equal(
                result,
                legacy_frame.reset_index(drop=True),
                check_dtype=True,
            )
            self.assertEqual(list(result.columns), cluster_aggregation.PROCESSED_CLUSTER_COLUMNS)
            self.assertEqual(len(result), len(legacy_frame))
            self.assertIn("disk-backed sorted runs", log)
            self.assertIn("one bounded chunk per active worker", log)
            self.assertEqual(
                [],
                [name for name in os.listdir(tmpdir) if name.startswith(".phasis_cluster_aggregation_")],
            )

    def test_gzipped_candidate_input_and_cache_hit_do_not_reparse_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logical_path = os.path.join(tmpdir, "libA.21-PHAS.candidate.clusters")
            _write_candidate(
                f"{logical_path}.gz",
                [("cluster-A", [(1, "w", 100, "tag-1")])],
                gzipped=True,
            )

            first_result, _ = self._run_aggregation(
                [logical_path],
                tmpdir,
                compress_intermediates=True,
            )
            self.assertEqual(len(first_result), 1)
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "21_processed_clusters.tab")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "21_processed_clusters.tab.gz")))

            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with mock.patch.multiple(
                    rt,
                    phase=21,
                    concat_libs=False,
                    memFile=os.path.join(tmpdir, "phasis.mem"),
                    compress_intermediates=True,
                    create=True,
                ):
                    with mock.patch.object(
                        cluster_aggregation,
                        "run_parallel_with_progress",
                        side_effect=AssertionError("cache hit should not parse candidate rows"),
                    ):
                        cached = cluster_aggregation.aggregate_and_write_processed_clusters([logical_path])
            finally:
                os.chdir(old_cwd)

            self.assertEqual(len(cached), 1)

    def test_preserves_text_identifiers_and_legacy_nan_serialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "001.21-PHAS.candidate.clusters")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(">cluster = 0007\n")
                handle.write(
                    _candidate_line(
                        10,
                        "NA",
                        100,
                        "0003",
                        pval_h_f=float("nan"),
                        tag_seq="NA",
                    )
                )

            legacy_frame = pd.DataFrame(
                cluster_aggregation.process_single_lib_cluster(path),
                columns=cluster_aggregation.PROCESSED_CLUSTER_COLUMNS,
            ).sort_values(by=["clusterID", "pos"])
            result, _ = self._run_aggregation([path], tmpdir)

            expected_path = os.path.join(tmpdir, "expected.tsv")
            legacy_frame.to_csv(expected_path, sep="\t", index=False, header=True)
            output_path = os.path.join(tmpdir, "21_processed_clusters.tab")
            with open(expected_path, "rb") as expected, open(output_path, "rb") as actual:
                self.assertEqual(actual.read(), expected.read())

            row = result.iloc[0]
            self.assertEqual(row["alib"], "001")
            self.assertEqual(row["clusterID"], "0007")
            self.assertEqual(row["chromosome"], "10")
            self.assertEqual(row["strand"], "NA")
            self.assertEqual(row["tag_id"], "0003")
            self.assertEqual(row["tag_seq"], "NA")
            self.assertTrue(pd.isna(row["pval_h_f"]))
            pd.testing.assert_frame_equal(
                result,
                legacy_frame.reset_index(drop=True),
                check_dtype=True,
            )

    def test_header_only_candidate_writes_an_empty_processed_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.21-PHAS.candidate.clusters")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(">cluster = no_rows\n")

            result, _ = self._run_aggregation([path], tmpdir)

            self.assertTrue(result.empty)
            self.assertEqual(list(result.columns), cluster_aggregation.PROCESSED_CLUSTER_COLUMNS)


if __name__ == "__main__":
    unittest.main()
