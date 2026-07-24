from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import pandas as pd

from phasis import runtime as rt
from phasis.stages import phas_clusters


def _cluster_row(alib: str, cluster_id: str, chromosome: str, pos: int, tag_id: str):
    """Return one complete processed-cluster row with stable CSV formatting."""
    return {
        "alib": alib,
        "clusterID": cluster_id,
        "chromosome": chromosome,
        "strand": "w",
        "pos": pos,
        "len": 21,
        "hits": 1,
        "abun": 10,
        "pval_h_f": 0.1,
        "N_f": 1,
        "X_f": 1,
        "pval_r_f": 0.1,
        "pval_corr_f": 0.1,
        "pval_h_r": 0.1,
        "N_r": 1,
        "X_r": 1,
        "pval_r_r": 0.1,
        "pval_corr_r": 0.1,
        "tag_id": tag_id,
        "tag_seq": "ACGTACGTACGTACGTACGTA",
    }


def _serial_streaming_runner(captured):
    """Exercise the result-consumer path without retaining worker results."""

    def runner(func, batches, **kwargs):
        captured["kwargs"] = kwargs
        captured["batch_count"] = len(batches)
        on_result = kwargs.get("on_result")
        if on_result is None:
            raise AssertionError("PHAS-cluster batches must stream results through on_result")
        if kwargs.get("return_results") is not False:
            raise AssertionError("PHAS-cluster batches must not retain all worker results")

        for task in batches:
            key, batch = task
            captured.setdefault("batches", []).append(
                (key, len(batch), tuple(batch["tag_id"].tolist()))
            )
            on_result(func(task))
        return None

    return runner


class PhasClusterBatchedStreamingTests(unittest.TestCase):
    def setUp(self):
        self.universal_ids = {
            "raw-A": "chr2:400..442",
            "raw-B": "chr1:100..121",
            "raw-C": "chr10:900..921",
        }

    def _lookup_universal_id(self, cluster_id):
        """Use a real callable: pandas.Series.map treats a Mock as a mapping."""
        return self.universal_ids.get(str(cluster_id))

    def _input_frame(self) -> pd.DataFrame:
        # The first-seen group order deliberately differs from lexical order and
        # group A is larger than the two-row cap. This catches use of
        # groupby.indices.items(), which need not preserve legacy multi-key
        # group iteration order.
        rows = [
            _cluster_row("libB", "raw-A", "chr2", 400, "b-400"),
            _cluster_row("libA", "raw-B", "chr1", 100, "a-100"),
            _cluster_row("libA", "raw-C", "chr10", 900, "c-900"),
            _cluster_row("libB", "raw-A", "chr2", 421, "b-421"),
            _cluster_row("libA", "raw-B", "chr1", 121, "a-121"),
            _cluster_row("libB", "raw-A", "chr2", 442, "b-442"),
            _cluster_row("libA", "raw-C", "chr10", 921, "c-921"),
        ]
        return pd.DataFrame(rows, columns=phas_clusters.REQUIRED_20_COLS)

    def _legacy_frame(self, all_clusters: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for key, group in all_clusters.groupby(["chromosome", "alib"], sort=False):
            frames.append(
                phas_clusters.process_phas_cluster_group(
                    (key, group.loc[:, phas_clusters.REQUIRED_20_COLS].values.tolist())
                )
            )
        return pd.concat(frames, ignore_index=True)

    def _run_builder(
        self,
        all_clusters: pd.DataFrame,
        tmpdir: str,
        runner,
        *,
        return_dataframe: bool,
    ):
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            with mock.patch.multiple(
                rt,
                phase=21,
                concat_libs=False,
                memFile=os.path.join(tmpdir, "phasis.mem"),
                ncores=12,
                compress_intermediates=False,
                create=True,
            ):
                with mock.patch.dict(
                    os.environ,
                    {"PHASIS_PHAS_CLUSTER_BATCH_ROWS": "2"},
                    clear=False,
                ):
                    with mock.patch.object(
                        phas_clusters.ids,
                        "ensure_mergedClusterDict",
                        return_value=self.universal_ids,
                    ):
                        with mock.patch.object(
                            phas_clusters.ids,
                            "getUniversalID",
                            new=self._lookup_universal_id,
                        ):
                            with mock.patch.object(
                                phas_clusters,
                                "run_parallel_with_progress",
                                side_effect=runner,
                            ):
                                with redirect_stdout(io.StringIO()):
                                    return phas_clusters.build_and_save_phas_clusters(
                                        all_clusters,
                                        phase=21,
                                        memFile=os.path.join(tmpdir, "phasis.mem"),
                                        concat_libs=False,
                                        return_dataframe=return_dataframe,
                                    )
        finally:
            os.chdir(old_cwd)

    def test_batch_worker_matches_legacy_group_processor(self):
        all_clusters = self._input_frame()
        key = ("chr2", "libB")
        batch = all_clusters.loc[[0, 3], phas_clusters.REQUIRED_20_COLS]

        with mock.patch.object(
            phas_clusters.ids,
            "ensure_mergedClusterDict",
            return_value=self.universal_ids,
        ):
            with mock.patch.object(
                phas_clusters.ids,
                "getUniversalID",
                new=self._lookup_universal_id,
            ):
                expected = phas_clusters.process_phas_cluster_group((key, batch.values.tolist()))
                actual = phas_clusters.process_phas_cluster_batch((key, batch))

        pd.testing.assert_frame_equal(actual, expected, check_dtype=True)

    def test_fixed_batches_stream_in_legacy_group_order_without_concat(self):
        all_clusters = self._input_frame()
        with mock.patch.object(
            phas_clusters.ids,
            "ensure_mergedClusterDict",
            return_value=self.universal_ids,
        ):
            with mock.patch.object(
                phas_clusters.ids,
                "getUniversalID",
                new=self._lookup_universal_id,
            ):
                expected = self._legacy_frame(all_clusters)

        captured = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            # A concat after worker dispatch would retain every batch result in
            # memory. The disk-backed writer must not call it.
            with mock.patch.object(
                phas_clusters.pd,
                "concat",
                side_effect=AssertionError("streaming PHAS output must not concatenate worker frames"),
            ):
                returned = self._run_builder(
                    all_clusters,
                    tmpdir,
                    _serial_streaming_runner(captured),
                    return_dataframe=False,
                )

            output_path = os.path.join(tmpdir, "21_PHAS_to_detect.tab")
            self.assertTrue(os.path.isfile(output_path))
            self.assertNotIsInstance(returned, pd.DataFrame)
            # Advanced positional selection must give workers an isolated batch
            # without a second explicit DataFrame copy in the parent.
            self.assertNotIn("identifier", all_clusters.columns)

            expected_path = os.path.join(tmpdir, "expected.tsv")
            expected.to_csv(expected_path, sep="\t", index=False)
            with open(expected_path, "rb") as expected_handle, open(output_path, "rb") as actual_handle:
                self.assertEqual(actual_handle.read(), expected_handle.read())

            actual = phas_clusters.load_phas_to_detect_output(output_path)

        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        self.assertEqual(
            captured["batches"],
            [
                (("chr2", "libB"), 2, ("b-400", "b-421")),
                (("chr2", "libB"), 1, ("b-442",)),
                (("chr1", "libA"), 2, ("a-100", "a-121")),
                (("chr10", "libA"), 2, ("c-900", "c-921")),
            ],
        )
        self.assertEqual(captured["batch_count"], 4)
        self.assertTrue(all(size <= 2 for _key, size, _tags in captured["batches"]))
        self.assertEqual(captured["kwargs"]["initial_worker_cap"], 2)
        self.assertEqual(captured["kwargs"]["max_worker_cap"], 8)
        self.assertEqual(captured["kwargs"]["initial_chunk_size"], 2)
        # The configured worker cap is 8 (70% of 12 cores), but this fixture
        # plans only four batches, so the runner never needs a larger task
        # window.
        self.assertEqual(captured["kwargs"]["max_chunk_size"], 4)
        self.assertTrue(captured["kwargs"]["adaptive_recovery"])

    def test_write_only_cache_hit_does_not_reload_the_finished_table(self):
        all_clusters = self._input_frame()
        captured = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            first = self._run_builder(
                all_clusters,
                tmpdir,
                _serial_streaming_runner(captured),
                return_dataframe=False,
            )
            self.assertNotIsInstance(first, pd.DataFrame)

            # In write-only mode a cache hit must not allocate the final table
            # before Phase II releases its raw processed-cluster dataframe.
            with mock.patch.object(
                phas_clusters.pd,
                "read_csv",
                side_effect=AssertionError("write-only cache hit must not read PHAS_to_detect"),
            ):
                cached = self._run_builder(
                    all_clusters,
                    tmpdir,
                    AssertionError("cache hit should not start PHAS-cluster workers"),
                    return_dataframe=False,
                )
            self.assertNotIsInstance(cached, pd.DataFrame)

            output_path = os.path.join(tmpdir, "21_PHAS_to_detect.tab")
            loaded = phas_clusters.load_phas_to_detect_output(output_path)

        self.assertEqual(len(loaded), len(all_clusters))
        self.assertEqual(
            loaded["identifier"].tolist(),
            [
                "chr2:400..442",
                "chr2:400..442",
                "chr2:400..442",
                "chr1:100..121",
                "chr1:100..121",
                "chr10:900..921",
                "chr10:900..921",
            ],
        )

    def test_default_api_still_returns_the_finished_dataframe(self):
        all_clusters = self._input_frame()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run_builder(
                all_clusters,
                tmpdir,
                _serial_streaming_runner({}),
                return_dataframe=True,
            )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(all_clusters))
        self.assertEqual(
            result["identifier"].tolist(),
            [
                "chr2:400..442",
                "chr2:400..442",
                "chr2:400..442",
                "chr1:100..121",
                "chr1:100..121",
                "chr10:900..921",
                "chr10:900..921",
            ],
        )

    def test_compact_loader_preserves_text_values_and_uses_requested_columns(self):
        frame = self._input_frame()
        frame["identifier"] = ["chr2:400..442"] * 3 + ["chr1:100..121"] * 2 + ["chr10:900..921"] * 2
        frame.loc[0, "clusterID"] = "001"
        frame.loc[0, "tag_seq"] = "NA"
        requested = (
            "alib",
            "clusterID",
            "chromosome",
            "strand",
            "pos",
            "len",
            "hits",
            "abun",
            "pval_corr_f",
            "pval_corr_r",
            "tag_seq",
            "identifier",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "21_PHAS_to_detect.tab")
            frame.to_csv(output_path, sep="\t", index=False)
            compact = phas_clusters.load_phas_to_detect_output(
                output_path,
                columns=requested,
            )

        self.assertEqual(compact.columns.tolist(), list(requested))
        self.assertEqual(compact.loc[0, "clusterID"], "001")
        self.assertEqual(compact.loc[0, "tag_seq"], "NA")
        self.assertEqual(compact.loc[0, "hits"], 1)


if __name__ == "__main__":
    unittest.main()
