from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import pandas as pd

from phasis import runtime as rt
from phasis.stages import phas_clusters, window_selection


def _phas_rows() -> pd.DataFrame:
    """Return group-major, cluster-major PHAS rows with a reader-boundary cluster."""
    rows = [
        # cB spans the 3-row reader/batch boundary and is intentionally larger
        # than that bound: a whole-cluster task must retain all four rows.
        ["libB", "cB", "chr2", 100, 0.4, 0.3],
        ["libB", "cB", "chr2", 124, 0.2, 0.5],
        ["libB", "cB", "chr2", 148, 0.27586206896551724, 0.2],
        ["libB", "cB", "chr2", 172, 0.3, 0.1],
        ["libB", "cC", "chr2", 300, 0.6, 0.7],
        ["libB", "cC", "chr2", 324, 0.2, 0.4],
        ["libA", "cA", "chr1", 500, 0.3, 0.2],
        ["libA", "cA", "chr1", 524, 0.1, 0.5],
        ["libA", "cA", "chr1", 548, 0.2, 0.1],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "alib",
            "clusterID",
            "chromosome",
            "pos",
            "pval_corr_f",
            "pval_corr_r",
        ],
    )


def _serial_parallel(captured):
    """Run both path-streaming and final chunk merge deterministically in-process."""

    def runner(func, data, **kwargs):
        if func is window_selection.select_windows_stream_task_worker:
            captured["sequence_type"] = type(data)
            captured["task_count"] = len(data)
            captured["parallel_kwargs"] = kwargs
            on_result = kwargs.get("on_result")
            if on_result is None or kwargs.get("return_results") is not False:
                raise AssertionError("streaming window tasks must use a result consumer")
            captured["batches"] = []
            for index in range(len(data)):
                task = data[index]
                captured["batches"].append(
                    (task["key"], tuple(task["df"]["clusterID"].tolist()))
                )
                on_result(func(task))
            return None

        if func is window_selection.load_window_chunk_file:
            return [func(path) for path in data]

        raise AssertionError(f"unexpected parallel worker: {func}")

    return runner


class StreamingWindowSelectionTests(unittest.TestCase):
    def _run_dataframe_path(self, frame: pd.DataFrame, tmpdir: str) -> pd.DataFrame:
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            with mock.patch.multiple(
                rt,
                phase=21,
                concat_libs=False,
                ncores=12,
                memFile=os.path.join(tmpdir, "phasis.mem"),
                compress_intermediates=False,
                create=True,
            ):
                with redirect_stdout(io.StringIO()):
                    return window_selection.select_scoring_windows(
                        frame,
                        window_len=48,
                        sliding=24,
                        minClusterLength=48,
                        memFile=os.path.join(tmpdir, "phasis.mem"),
                    )
        finally:
            os.chdir(old_cwd)

    def _run_streaming_path(self, source_path: str, tmpdir: str, captured) -> pd.DataFrame:
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            with mock.patch.multiple(
                rt,
                phase=21,
                concat_libs=False,
                ncores=12,
                memFile=os.path.join(tmpdir, "phasis.mem"),
                compress_intermediates=False,
                window_selection_worker_cap=None,
                window_selection_batch_rows=None,
                create=True,
            ):
                with mock.patch.object(
                    window_selection,
                    "run_parallel_with_progress",
                    side_effect=_serial_parallel(captured),
                ):
                    with redirect_stdout(io.StringIO()):
                        return window_selection.select_scoring_windows_from_path(
                            source_path,
                            window_len=48,
                            sliding=24,
                            minClusterLength=48,
                            memFile=os.path.join(tmpdir, "phasis.mem"),
                            batch_rows=3,
                        )
        finally:
            os.chdir(old_cwd)

    def test_path_streaming_matches_dataframe_output_and_carries_cluster_boundaries(self):
        frame = _phas_rows()
        with tempfile.TemporaryDirectory() as tmpdir:
            dataframe_dir = os.path.join(tmpdir, "dataframe")
            streaming_dir = os.path.join(tmpdir, "streaming")
            os.makedirs(dataframe_dir)
            os.makedirs(streaming_dir)
            # The normal logical artifact can be physically gzip-compressed;
            # the disk-backed reader must resolve that sibling without loading
            # it through the former all-rows PHAS DataFrame loader.
            source_path = os.path.join(streaming_dir, "21_PHAS_to_detect.tab")
            frame.to_csv(f"{source_path}.gz", sep="\t", index=False, compression="gzip")

            # Compare against the old Python-parser PHAS loader, not merely
            # the original in-memory fixture. This protects p-value precision
            # while the streaming route uses pandas' C parser.
            expected = self._run_dataframe_path(
                phas_clusters.load_phas_to_detect_output(source_path),
                dataframe_dir,
            )
            captured = {}
            actual = self._run_streaming_path(source_path, streaming_dir, captured)

            expected_path = os.path.join(dataframe_dir, "21_clusters_windows_to_score.tsv")
            actual_path = os.path.join(streaming_dir, "21_clusters_windows_to_score.tsv")
            with open(expected_path, "rb") as expected_handle, open(actual_path, "rb") as actual_handle:
                self.assertEqual(actual_handle.read(), expected_handle.read())

        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        self.assertEqual(captured["task_count"], 3)
        self.assertEqual(
            captured["batches"],
            [
                ("libB__chrchr2", ("cB", "cB", "cB", "cB")),
                ("libB__chrchr2", ("cC", "cC")),
                ("libA__chrchr1", ("cA", "cA", "cA")),
            ],
        )
        self.assertEqual(captured["parallel_kwargs"]["initial_worker_cap"], 2)
        self.assertEqual(captured["parallel_kwargs"]["max_worker_cap"], 8)
        self.assertEqual(captured["parallel_kwargs"]["max_chunk_size"], 3)
        self.assertTrue(captured["parallel_kwargs"]["adaptive_recovery"])
        self.assertEqual(captured["sequence_type"], window_selection._StreamingWindowTaskSequence)

    def test_path_streaming_rejects_noncontiguous_clusters_before_workers_start(self):
        frame = _phas_rows().iloc[[0, 4, 1]].copy()
        frame.loc[:, "clusterID"] = ["c1", "c2", "c1"]
        frame.loc[:, "chromosome"] = "chr1"
        frame.loc[:, "alib"] = "libA"

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "21_PHAS_to_detect.tab")
            frame.to_csv(source_path, sep="\t", index=False)
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with mock.patch.multiple(
                    rt,
                    phase=21,
                    concat_libs=False,
                    ncores=1,
                    memFile=os.path.join(tmpdir, "phasis.mem"),
                    compress_intermediates=False,
                    create=True,
                ):
                    with self.assertRaisesRegex(ValueError, "not contiguous by cluster"):
                        window_selection.select_scoring_windows_from_path(
                            source_path,
                            window_len=48,
                            sliding=24,
                            minClusterLength=48,
                            memFile=os.path.join(tmpdir, "phasis.mem"),
                            batch_rows=2,
                        )
            finally:
                os.chdir(old_cwd)

    def test_streaming_worker_cap_defaults_to_seventy_percent(self):
        with mock.patch.multiple(
            rt,
            ncores=12,
            window_selection_worker_cap=None,
            window_selection_batch_rows=None,
            create=True,
        ):
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(window_selection._window_selection_worker_cap(), 8)
                self.assertEqual(window_selection._window_selection_batch_rows(), 100_000)
                self.assertEqual(
                    window_selection._window_selection_parallel_kwargs(12),
                    {
                        "initial_worker_cap": 2,
                        "max_worker_cap": 8,
                        "initial_chunk_size": 2,
                        "max_chunk_size": 8,
                        "adaptive_recovery": True,
                    },
                )


if __name__ == "__main__":
    unittest.main()
