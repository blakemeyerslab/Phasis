from __future__ import annotations

import gzip
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from phasis import cli
from phasis.config import Phase2Config
from phasis.stages import feature_assembly, phase2_pipeline


class RunOptionParsingTests(unittest.TestCase):
    def test_locus_plots_default_on_with_visible_opt_out(self):
        parser = cli.build_parser()

        self.assertTrue(parser.parse_args([]).locus_plots)
        self.assertFalse(parser.parse_args(["--no_locus_plots"]).locus_plots)
        self.assertIn("--no_locus_plots", parser.format_help())

    def test_phase2_config_reads_locus_plots_from_runtime(self):
        with mock.patch.object(phase2_pipeline.rt, "locus_plots", False, create=True):
            self.assertFalse(Phase2Config.from_runtime().locus_plots)
        with mock.patch.object(phase2_pipeline.rt, "locus_plots", True, create=True):
            self.assertTrue(Phase2Config.from_runtime().locus_plots)


class CleanupCompressedIntermediatesTests(unittest.TestCase):
    def test_cleanup_removes_compressed_phase2_tables_and_keeps_results_and_index(self):
        from phasis import cache

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "21_results"))
            os.makedirs(os.path.join(tmpdir, "index"))
            os.makedirs(os.path.join(tmpdir, "21_clusters"))
            keep = [
                os.path.join(tmpdir, "21_results", "21_calls.tsv"),
                os.path.join(tmpdir, "index", "genome.1.ht2"),
                os.path.join(tmpdir, "library.tag.gz"),
            ]
            drop = [
                os.path.join(tmpdir, "libA.21-PHAS.candidate.clusters"),
                os.path.join(tmpdir, "21_clusters", "libA-1.lclust"),
                os.path.join(tmpdir, "21_PHAS_to_detect.tab"),
            ]
            drop += [
                os.path.join(tmpdir, f"21_{name}.gz")
                for name in (
                    "processed_clusters.tab",
                    "candidate.loci_table.tab",
                    "merged_clusters.tab",
                    "mergedClusterDict.tab",
                    "PHAS_to_detect.tab",
                    "clusters_windows_to_score.tsv",
                    "clusters_scored.tsv",
                    "cluster_set_features.tsv",
                )
            ]
            for path in keep + drop:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("x\n")

            with mock.patch.object(cache.rt, "outdir", os.path.join(tmpdir, "21_results"), create=True):
                cache.cleanup(tmpdir)

            for path in keep:
                self.assertTrue(os.path.exists(path), path)
            for path in drop:
                self.assertFalse(os.path.exists(path), path)


class FeatureAssemblyWorkReportTests(unittest.TestCase):
    REQUIRED = ["clusterID", "chromosome", "strand", "pos", "len", "abun", "identifier", "tag_seq", "alib"]

    def _clusters(self) -> pd.DataFrame:
        rows = []
        for cluster, n_rows in (("c1", 3), ("c2", 2), ("c3", 4), ("c4", 1)):
            for i in range(n_rows):
                rows.append([cluster, "chr1", "w", 100 + i, 24, 1, f"chr1:{cluster}", "AAAA", "libA"])
        return pd.DataFrame(rows, columns=self.REQUIRED)

    def test_row_counter_counts_data_rows_of_plain_and_gzip_tables(self):
        clusters = self._clusters()
        with tempfile.TemporaryDirectory() as tmpdir:
            plain = os.path.join(tmpdir, "24_PHAS_to_detect.tab")
            clusters.to_csv(plain, sep="\t", index=False)
            gz = plain + ".gz"
            with open(plain, "rb") as src, gzip.open(gz, "wb") as dst:
                dst.write(src.read())

            for path in (plain, gz):
                counter = feature_assembly._StreamRowCounter(path)
                counter.join(30)
                self.assertEqual(counter.total, len(clusters), path)

            missing = feature_assembly._StreamRowCounter(os.path.join(tmpdir, "absent.tab"))
            missing.join(30)
            self.assertIsNone(missing.total)

    def test_stats_track_rows_and_clusters_of_handed_out_batches(self):
        clusters = self._clusters()
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "24_PHAS_to_detect.tab")
            clusters.to_csv(source_path, sep="\t", index=False)
            stats = feature_assembly._FeatureAssemblyStreamStats()
            seen_rows = 0
            batches = feature_assembly._iter_streamed_feature_assembly_batches(
                source_path,
                required_cols=self.REQUIRED,
                batch_rows=5,
                reader_chunk_rows=2,
                stats=stats,
            )
            for batch in batches:
                seen_rows += len(batch)
                # Counters never run ahead of the batches already handed out.
                self.assertEqual(stats.batched_rows, seen_rows)

        self.assertEqual(stats.batched_rows, len(clusters))
        self.assertEqual(stats.batched_clusters, 4)

    def test_progress_line_reports_share_rate_and_time_left(self):
        line = feature_assembly._feature_assembly_progress_line(
            batches=60,
            rows_done=6_000_000,
            rows_total=10_000_000,
            clusters_done=123_456,
            feature_rows=120_000,
            workers=8,
            elapsed=3 * 3600,
        )
        self.assertTrue(line.startswith("[feature-assembly] 60 batches done"))
        self.assertIn("6,000,000 of 10,000,000 input rows (60.0%)", line)
        self.assertIn("123,456 clusters", line)
        self.assertIn("120,000 feature rows written", line)
        self.assertIn("8 concurrent batch(es)", line)
        self.assertIn("elapsed 3:00:00", line)
        self.assertIn("556 input rows/s", line)
        self.assertIn("about 2:00:00 left", line)

    def test_progress_line_without_total_omits_share_and_time_left(self):
        line = feature_assembly._feature_assembly_progress_line(
            batches=2,
            rows_done=200_000,
            rows_total=None,
            clusters_done=10,
            feature_rows=5,
            workers=2,
            elapsed=100,
        )
        self.assertIn("200,000 input rows |", line)
        self.assertNotIn("%", line)
        self.assertNotIn("left", line)


if __name__ == "__main__":
    unittest.main()
