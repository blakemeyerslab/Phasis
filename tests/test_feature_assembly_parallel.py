from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from phasis import runtime as rt
from phasis.stages import feature_assembly


def _feature_row():
    return ["chr1:100..124", "cluster-1", "libA"] + [0.0] * (
        len(feature_assembly.FEATURE_COLS) - 3
    )


def _feature_row_with(**overrides):
    row = _feature_row()
    for column, value in overrides.items():
        row[feature_assembly.FEATURE_COLS.index(column)] = value
    return row


class FeatureAssemblyParallelTests(unittest.TestCase):
    def test_default_worker_cap_uses_seventy_percent_and_explicit_cap_respects_cores(self):
        with mock.patch.multiple(
            rt,
            ncores=12,
            feature_assembly_worker_cap=None,
            feature_assembly_batch_rows=None,
            create=True,
        ):
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(feature_assembly._feature_assembly_worker_cap(), 8)
                self.assertEqual(feature_assembly._feature_assembly_batch_rows(), 100_000)
                self.assertEqual(
                    feature_assembly._feature_assembly_parallel_kwargs(12),
                    {
                        "initial_worker_cap": 2,
                        "max_worker_cap": 8,
                        "initial_chunk_size": 2,
                        "max_chunk_size": 8,
                        "adaptive_recovery": True,
                    },
                )

            with mock.patch.dict(
                os.environ,
                {"PHASIS_FEATURE_ASSEMBLY_WORKER_CAP": "20"},
                clear=True,
            ):
                self.assertEqual(feature_assembly._feature_assembly_worker_cap(), 12)

            with mock.patch.dict(
                os.environ,
                {"PHASIS_FEATURE_ASSEMBLY_BATCH_ROWS": "50000"},
                clear=True,
            ):
                self.assertEqual(feature_assembly._feature_assembly_batch_rows(), 50_000)

    def test_batches_preserve_cluster_boundaries_and_isolate_oversized_clusters(self):
        clusters = pd.DataFrame(
            [
                ["cluster-1", "chr1", "w", 1],
                ["cluster-1", "chr1", "w", 2],
                ["cluster-2", "chr1", "w", 3],
                ["cluster-2", "chr1", "w", 4],
                ["cluster-2", "chr1", "w", 5],
                ["cluster-3", "chr2", "w", 6],
                ["cluster-3", "chr2", "w", 7],
                ["cluster-3", "chr2", "w", 8],
                ["cluster-3", "chr2", "w", 9],
            ],
            columns=["clusterID", "chromosome", "strand", "pos"],
        )

        batches, oversized = feature_assembly._build_feature_assembly_batches(
            clusters,
            batch_rows=3,
        )

        self.assertEqual([len(batch) for batch in batches], [2, 3, 4])
        self.assertEqual(oversized, 1)
        cluster_batch_count = {}
        for batch in batches:
            for cluster_id in batch["clusterID"].unique():
                cluster_batch_count[cluster_id] = cluster_batch_count.get(cluster_id, 0) + 1
        self.assertEqual(cluster_batch_count, {"cluster-1": 1, "cluster-2": 1, "cluster-3": 1})

    def test_lazy_batches_match_eager_batches_without_holding_dataframe_copies(self):
        clusters = pd.DataFrame(
            [
                ["cluster-1", "chr1", "w", 1, 24, 2, "tag-1", "AAAA", "libA"],
                ["cluster-1", "chr1", "w", 2, 24, 2, "tag-2", "AAAT", "libA"],
                ["cluster-2", "chr1", "c", 3, 24, 2, "tag-3", "CCCC", "libA"],
                ["cluster-2", "chr1", "c", 4, 24, 2, "tag-4", "CCCT", "libA"],
                ["cluster-3", "chr2", "w", 5, 24, 2, "tag-5", "GGGG", "libA"],
            ],
            columns=[
                "clusterID",
                "chromosome",
                "strand",
                "pos",
                "len",
                "abun",
                "identifier",
                "tag_seq",
                "alib",
            ],
        )
        eager_batches, eager_oversized = feature_assembly._build_feature_assembly_batches(
            clusters,
            batch_rows=3,
        )
        positions, lazy_oversized = feature_assembly._build_feature_assembly_batch_positions(
            clusters,
            batch_rows=3,
        )
        lazy_batches = feature_assembly._LazyFeatureAssemblyBatches(
            clusters,
            required_cols=list(clusters.columns),
            batch_positions=positions,
        )

        self.assertEqual(lazy_oversized, eager_oversized)
        self.assertEqual(len(lazy_batches), len(eager_batches))
        for index, eager_batch in enumerate(eager_batches):
            pd.testing.assert_frame_equal(lazy_batches[index], eager_batch)

        # Advanced positional indexing gives the worker an isolated batch even
        # though production no longer makes a second redundant ``.copy()``.
        materialized = lazy_batches[0]
        materialized.iloc[0, materialized.columns.get_loc("tag_seq")] = "MUTATED"
        self.assertEqual(clusters.iloc[0]["tag_seq"], "AAAA")

    def test_batched_processing_matches_the_former_chromosome_units(self):
        clusters = pd.DataFrame(
            [
                ["cluster-1", "chr1", "w", 100, 24, 5, "tag-1", "AAAA", "libA"],
                ["cluster-1", "chr1", "w", 124, 24, 4, "tag-2", "AAAT", "libA"],
                ["cluster-2", "chr1", "c", 200, 24, 3, "tag-3", "CCCC", "libA"],
                ["cluster-2", "chr1", "c", 224, 24, 2, "tag-4", "CCCT", "libA"],
                ["cluster-3", "chr2", "w", 300, 24, 6, "tag-5", "GGGG", "libA"],
                ["cluster-3", "chr2", "w", 324, 24, 1, "tag-6", "GGGT", "libA"],
            ],
            columns=[
                "clusterID",
                "chromosome",
                "strand",
                "pos",
                "len",
                "abun",
                "identifier",
                "tag_seq",
                "alib",
            ],
        )
        old_units = [df for _, df in clusters.groupby("chromosome", sort=False)]
        batches, _ = feature_assembly._build_feature_assembly_batches(
            clusters,
            batch_rows=2,
        )

        with mock.patch.multiple(rt, phase=24, clusters_scored_tsv=None, create=True):
            old_rows = []
            for unit in old_units:
                old_rows.extend(feature_assembly.process_chromosome_features(unit)["rows"])
            batch_rows = []
            for batch in batches:
                batch_rows.extend(feature_assembly.process_chromosome_features(batch)["rows"])

        pd.testing.assert_frame_equal(
            pd.DataFrame(old_rows, columns=feature_assembly.FEATURE_COLS),
            pd.DataFrame(batch_rows, columns=feature_assembly.FEATURE_COLS),
            check_dtype=False,
        )

    def test_feature_assembly_caps_workers_and_submitted_groups_together(self):
        clusters = pd.DataFrame(
            [
                ["cluster-1", "chr1", "w", 100, 24, 5, "tag-1", "AAAA", "libA"],
                ["cluster-2", "chr2", "w", 200, 24, 5, "tag-2", "CCCC", "libA"],
                ["cluster-3", "chr3", "w", 300, 24, 5, "tag-3", "GGGG", "libA"],
            ],
            columns=[
                "clusterID",
                "chromosome",
                "strand",
                "pos",
                "len",
                "abun",
                "identifier",
                "tag_seq",
                "alib",
            ],
        )
        captured = {}
        expected_rows = [
            _feature_row_with(
                identifier="chr1:100..124",
                cID="cluster-1",
                # This decimal differs after a default CSV parse/rewrite by
                # one ULP, so it protects the round-trip parser used by the
                # disk-backed streaming normalization path.
                complexity=0.27586206896551724,
                # A later blank forces the final full-table dtype inference
                # used by the legacy whole-table writer.
                w_window_start=100,
            ),
            _feature_row_with(
                identifier="chr2:100..124",
                cID="cluster-2",
                w_window_start=None,
            ),
        ]

        def fake_parallel(_func, groups, **kwargs):
            captured["group_count"] = len(groups)
            captured["kwargs"] = kwargs
            captured["groups_type"] = type(groups)
            captured["batch_sizes"] = []
            for index in range(len(groups)):
                # Access one batch at a time, matching the bounded scheduler
                # contract instead of building an eager list in this test.
                captured["batch_sizes"].append(len(groups[index]))
                kwargs["on_result"](
                    {
                        "rows": [expected_rows[index]],
                        "debug_rows": [],
                    }
                )
            return None

        produced_text = None
        leftover_temp_names = None
        cached_result = None
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                output_path = os.path.join(tmpdir, "24_cluster_set_features.tsv")
                with mock.patch.multiple(
                    rt,
                    phase=24,
                    ncores=12,
                    feature_assembly_worker_cap=None,
                    feature_assembly_batch_rows=2,
                    run_dir=tmpdir,
                    memFile=os.path.join(tmpdir, "phasis.mem"),
                    compress_intermediates=False,
                    create=True,
                ):
                    with mock.patch.object(
                        feature_assembly,
                        "run_parallel_with_progress",
                        side_effect=fake_parallel,
                    ):
                        result = feature_assembly.features_to_detection(
                            clusters,
                            phase=24,
                            outdir=tmpdir,
                            memFile=os.path.join(tmpdir, "phasis.mem"),
                            outfname=output_path,
                        )
                    with mock.patch.object(
                        feature_assembly,
                        "run_parallel_with_progress",
                        side_effect=AssertionError("cache hit should not schedule feature work"),
                    ):
                        cached_result = feature_assembly.features_to_detection(
                            clusters,
                            phase=24,
                            outdir=tmpdir,
                            memFile=os.path.join(tmpdir, "phasis.mem"),
                            outfname=output_path,
                        )
                with open(output_path, encoding="utf-8") as handle:
                    produced_text = handle.read()
                leftover_temp_names = [
                    name
                    for name in os.listdir(tmpdir)
                    if name.startswith(".phasis_feature_assembly_")
                ]
            finally:
                os.chdir(old_cwd)

        self.assertEqual(len(result), 2)
        self.assertEqual(captured["group_count"], 2)
        self.assertIs(captured["groups_type"], feature_assembly._LazyFeatureAssemblyBatches)
        self.assertEqual(captured["batch_sizes"], [2, 1])
        self.assertEqual(captured["kwargs"]["initial_worker_cap"], 2)
        self.assertEqual(captured["kwargs"]["max_worker_cap"], 8)
        self.assertEqual(captured["kwargs"]["initial_chunk_size"], 2)
        self.assertEqual(captured["kwargs"]["max_chunk_size"], 2)
        self.assertTrue(captured["kwargs"]["adaptive_recovery"])
        self.assertFalse(captured["kwargs"]["return_results"])
        self.assertTrue(callable(captured["kwargs"]["on_result"]))
        self.assertEqual(result["w_window_start"].iloc[0], 100.0)
        self.assertTrue(pd.isna(result["w_window_start"].iloc[1]))
        self.assertEqual(leftover_temp_names, [])
        pd.testing.assert_frame_equal(result, cached_result)

        expected_frame = feature_assembly._coerce_feature_frame(
            pd.DataFrame(expected_rows, columns=feature_assembly.FEATURE_COLS)
        )
        self.assertEqual(produced_text, expected_frame.to_csv(sep="\t", index=False))

    def test_disk_backed_input_matches_dataframe_path_across_chunk_boundaries(self):
        columns = [
            "clusterID",
            "chromosome",
            "strand",
            "pos",
            "len",
            "abun",
            "identifier",
            "tag_seq",
            "alib",
        ]
        rows = []
        # The PHAS writer emits one chromosome/library group at a time.  This
        # library-major order deliberately revisits a chromosome for libB: it
        # proves the streamed result is restored to the legacy groupby.indices
        # order (chromosome first, then cluster) before writing its TSV.
        for library_index, library in enumerate(("libA", "libB")):
            for chromosome_index in range(2):
                chromosome = f"chr{chromosome_index + 1}"
                for local_cluster_index in range(6):
                    cluster_index = (
                        (library_index * 12)
                        + (chromosome_index * 6)
                        + local_cluster_index
                    )
                    cluster_id = f"cluster-{cluster_index:02d}"
                    for offset, strand in enumerate(("w", "c")):
                        position = 1_000 + (cluster_index * 100) + (offset * 24)
                        rows.append(
                            [
                                cluster_id,
                                chromosome,
                                strand,
                                position,
                                24,
                                offset + 1,
                                f"{chromosome}:{position}..{position + 24}",
                                "ACGTACGTACGTACGTACGTACGT",
                                library,
                            ]
                        )
        clusters = pd.DataFrame(rows, columns=columns)

        def serial_runner(captured):
            def _run(func, groups, **kwargs):
                captured.append(
                    {
                        "kwargs": dict(kwargs),
                        "batches": [groups[index].copy() for index in range(len(groups))],
                    }
                )
                for index in range(len(groups)):
                    kwargs["on_result"](func(groups[index]))
                if kwargs.get("return_state"):
                    return (
                        None,
                        {
                            "worker_cap": kwargs["max_worker_cap"],
                            "chunk_size": kwargs["max_chunk_size"],
                            "had_failures": False,
                        },
                    )
                return None

            return _run

        with tempfile.TemporaryDirectory() as tmpdir:
            source_logical = os.path.join(tmpdir, "24_PHAS_to_detect.tab")
            clusters.to_csv(
                f"{source_logical}.gz",
                sep="\t",
                index=False,
                compression="gzip",
            )
            dataframe_output = os.path.join(tmpdir, "dataframe_features.tsv")
            streamed_output = os.path.join(tmpdir, "streamed_features.tsv")
            dataframe_calls = []
            streamed_calls = []
            with mock.patch.multiple(
                rt,
                phase=24,
                ncores=12,
                feature_assembly_worker_cap=None,
                feature_assembly_batch_rows=2,
                phase2_streaming_rows=3,
                clusters_scored_tsv=None,
                run_dir=tmpdir,
                memFile=os.path.join(tmpdir, "phasis.mem"),
                compress_intermediates=False,
                create=True,
            ), mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                feature_assembly.st,
                "WIN_SCORE_LOOKUP",
                {},
            ):
                with mock.patch.object(
                    feature_assembly,
                    "run_parallel_with_progress",
                    side_effect=serial_runner(dataframe_calls),
                ):
                    dataframe_result = feature_assembly.features_to_detection(
                        clusters.copy(),
                        phase=24,
                        outdir=tmpdir,
                        memFile=os.path.join(tmpdir, "phasis.mem"),
                        outfname=dataframe_output,
                    )
                with mock.patch.object(
                    feature_assembly,
                    "run_parallel_with_progress",
                    side_effect=serial_runner(streamed_calls),
                ):
                    streamed_result = feature_assembly.features_to_detection(
                        clusters_path=source_logical,
                        phase=24,
                        outdir=tmpdir,
                        memFile=os.path.join(tmpdir, "phasis.mem"),
                        outfname=streamed_output,
                    )

            with open(dataframe_output, encoding="utf-8") as handle:
                dataframe_text = handle.read()
            with open(streamed_output, encoding="utf-8") as handle:
                streamed_text = handle.read()

        pd.testing.assert_frame_equal(
            dataframe_result,
            streamed_result,
            check_dtype=False,
        )
        self.assertEqual(dataframe_text, streamed_text)
        self.assertEqual(
            [len(call["batches"]) for call in streamed_calls],
            [2, 2, 4, 4, 8, 4],
        )
        self.assertEqual(
            [call["kwargs"]["initial_worker_cap"] for call in streamed_calls],
            [2, 2, 4, 4, 8, 8],
        )
        self.assertEqual(
            [call["kwargs"]["max_worker_cap"] for call in streamed_calls],
            [2, 2, 4, 4, 8, 8],
        )
        self.assertTrue(
            all(call["kwargs"]["return_state"] for call in streamed_calls)
        )
        self.assertTrue(
            all(not call["kwargs"]["show_progress"] for call in streamed_calls)
        )

        streamed_batches = [
            batch
            for call in streamed_calls
            for batch in call["batches"]
        ]
        self.assertEqual([len(batch) for batch in streamed_batches], [2] * 24)
        self.assertEqual(
            {
                (str(row.chromosome), str(row.clusterID))
                for batch in streamed_batches
                for row in batch[["chromosome", "clusterID"]].drop_duplicates().itertuples(index=False)
            },
            {
                (str(row.chromosome), str(row.clusterID))
                for row in clusters[["chromosome", "clusterID"]].drop_duplicates().itertuples(index=False)
            },
        )

    def test_disk_backed_input_rejects_recurrent_chromosome_library_groups(self):
        required = [
            "clusterID",
            "chromosome",
            "strand",
            "pos",
            "len",
            "abun",
            "identifier",
            "tag_seq",
            "alib",
        ]
        clusters = pd.DataFrame(
            [
                ["cluster-1", "chr1", "w", 100, 24, 1, "chr1:100..124", "AAAA", "libA"],
                ["cluster-2", "chr2", "w", 200, 24, 1, "chr2:200..224", "CCCC", "libA"],
                ["cluster-3", "chr1", "c", 124, 24, 1, "chr1:124..148", "AAAT", "libA"],
            ],
            columns=required,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "24_PHAS_to_detect.tab")
            clusters.to_csv(source_path, sep="\t", index=False)
            stats = feature_assembly._FeatureAssemblyStreamStats()
            with self.assertRaisesRegex(ValueError, "chromosome, alib"):
                list(
                    feature_assembly._iter_streamed_feature_assembly_batches(
                        source_path,
                        required_cols=required,
                        batch_rows=10,
                        reader_chunk_rows=2,
                        stats=stats,
                    )
                )


if __name__ == "__main__":
    unittest.main()
