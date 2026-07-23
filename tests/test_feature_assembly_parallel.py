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

        def fake_parallel(_func, groups, **kwargs):
            captured["group_count"] = len(groups)
            captured["kwargs"] = kwargs
            return [{"rows": [_feature_row()], "debug_rows": []}]

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
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
                            outfname=os.path.join(tmpdir, "24_cluster_set_features.tsv"),
                        )
            finally:
                os.chdir(old_cwd)

        self.assertEqual(len(result), 1)
        self.assertEqual(captured["group_count"], 2)
        self.assertEqual(captured["kwargs"]["initial_worker_cap"], 2)
        self.assertEqual(captured["kwargs"]["max_worker_cap"], 8)
        self.assertEqual(captured["kwargs"]["initial_chunk_size"], 2)
        self.assertEqual(captured["kwargs"]["max_chunk_size"], 2)
        self.assertTrue(captured["kwargs"]["adaptive_recovery"])


if __name__ == "__main__":
    unittest.main()
