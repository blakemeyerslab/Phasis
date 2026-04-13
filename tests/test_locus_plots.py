from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from phasis.stages import locus_plots


def _serial_parallel_runner(func, data, **_kwargs):
    return [func(item) for item in data]


class LocusPlotHelperTests(unittest.TestCase):
    def test_remote_mount_detection_from_mountinfo(self):
        mountinfo = (
            "29 23 0:25 / / rw,relatime - apfs /dev/disk3s1 rw\n"
            "44 29 0:99 / /quobyte rw,relatime - fuse.quobyte quobyte rw\n"
        )
        is_remote, mount_point, fs_type = locus_plots._detect_remote_filesystem(
            "/quobyte/project/run/24_KNN_PHAS_locus_plots",
            mountinfo_text=mountinfo,
        )
        self.assertTrue(is_remote)
        self.assertEqual(mount_point, "/quobyte")
        self.assertEqual(fs_type, "fuse.quobyte")

    def test_remote_path_prefix_fallback_detects_quobyte(self):
        is_remote, mount_point, fs_type = locus_plots._detect_remote_filesystem(
            "/quobyte/project/run/24_KNN_PHAS_locus_plots",
            mountinfo_text="",
        )
        self.assertTrue(is_remote)
        self.assertIsNone(mount_point)
        self.assertEqual(fs_type, "path-prefix")

    def test_scheduler_driven_auto_mode_prefers_local_staging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(locus_plots.rt, "plot_staging", None):
                strategy = locus_plots._resolve_plot_staging_strategy(
                    "/tmp/24_KNN_PHAS_locus_plots",
                    env={"SLURM_JOB_ID": "123", "TMPDIR": tmpdir},
                    mountinfo_text="",
                )
        self.assertEqual(strategy["mode"], "local")
        self.assertEqual(strategy["staging_root"], os.path.abspath(tmpdir))

    def test_requested_local_mode_falls_back_to_direct_without_writable_scratch(self):
        with mock.patch.object(locus_plots.rt, "plot_staging", None):
            with mock.patch.object(locus_plots.os.path, "isdir", return_value=False):
                with mock.patch.object(locus_plots.os, "access", return_value=False):
                    strategy = locus_plots._resolve_plot_staging_strategy(
                        "/tmp/24_KNN_PHAS_locus_plots",
                        env={"PHASIS_PLOT_STAGING": "local", "TMPDIR": "/path/that/does/not/exist"},
                        mountinfo_text="",
                    )
        self.assertEqual(strategy["mode"], "direct")
        self.assertIsNone(strategy["staging_root"])

    def test_finalize_staged_plot_dir_replaces_destination(self):
        with tempfile.TemporaryDirectory() as staged_parent, tempfile.TemporaryDirectory() as final_parent:
            staged_plot_dir = os.path.join(staged_parent, "plots")
            final_plot_dir = os.path.join(final_parent, "plots")
            os.makedirs(staged_plot_dir, exist_ok=True)
            os.makedirs(final_plot_dir, exist_ok=True)

            with open(os.path.join(staged_plot_dir, "new_plot.png"), "w", encoding="utf-8") as handle:
                handle.write("new")
            with open(os.path.join(final_plot_dir, "old_plot.png"), "w", encoding="utf-8") as handle:
                handle.write("old")

            locus_plots._finalize_staged_plot_dir(staged_plot_dir, final_plot_dir)

            self.assertTrue(os.path.isfile(os.path.join(final_plot_dir, "new_plot.png")))
            self.assertFalse(os.path.exists(os.path.join(final_plot_dir, "old_plot.png")))

    def test_collect_export_rows_prefers_exact_then_best_offset_and_exact_extended_only(self):
        prepared = locus_plots._prepare_cluster_df(
            [
                {"pos": 100, "abun": 5, "len": 24, "strand": "w", "tag_seq": "EXACT_A", "hits": 1},
                {"pos": 100, "abun": 2, "len": 24, "strand": "w", "tag_seq": "EXACT_B", "hits": 1},
                {"pos": 99, "abun": 7, "len": 24, "strand": "w", "tag_seq": "OFFSET_LEFT_IGNORED", "hits": 1},
                {"pos": 101, "abun": 6, "len": 24, "strand": "w", "tag_seq": "OFFSET_RIGHT_IGNORED", "hits": 1},
                {"pos": 123, "abun": 4, "len": 24, "strand": "w", "tag_seq": "OFFSET_LEFT", "hits": 1},
                {"pos": 125, "abun": 9, "len": 24, "strand": "w", "tag_seq": "OFFSET_RIGHT", "hits": 1},
                {"pos": 147, "abun": 6, "len": 24, "strand": "w", "tag_seq": "OFFSET_TIE_LEFT", "hits": 1},
                {"pos": 149, "abun": 6, "len": 24, "strand": "w", "tag_seq": "OFFSET_TIE_RIGHT", "hits": 1},
                {"pos": 172, "abun": 8, "len": 24, "strand": "w", "tag_seq": "EXTENDED_EXACT", "hits": 1},
                {"pos": 171, "abun": 10, "len": 24, "strand": "w", "tag_seq": "EXTENDED_OFFSET_IGNORED", "hits": 1},
                {"pos": 200, "abun": 9, "len": 24, "strand": "c", "tag_seq": "OTHER_STRAND", "hits": 1},
            ]
        )

        export_rows = locus_plots._collect_export_rows_for_strand(
            prepared,
            phase_value=24,
            strand_code="w",
            base_positions=[100, 124, 148],
            extended_positions=[172, 196],
            identifier_text="chr1:100..196",
            cid_value="cluster_1",
            alib_value="libA",
        )

        observed = [(row["observed_pos"], row["expected_register_pos"], row["register_class"], row["tag_seq"]) for row in export_rows]
        self.assertIn((100, 100, "core_exact", "EXACT_A"), observed)
        self.assertIn((100, 100, "core_exact", "EXACT_B"), observed)
        self.assertIn((125, 124, "core_offset", "OFFSET_RIGHT"), observed)
        self.assertIn((147, 148, "core_offset", "OFFSET_TIE_LEFT"), observed)
        self.assertIn((172, 172, "extended_exact", "EXTENDED_EXACT"), observed)
        self.assertNotIn((99, 100, "core_offset", "OFFSET_LEFT_IGNORED"), observed)
        self.assertNotIn((171, 172, "extended_exact", "EXTENDED_OFFSET_IGNORED"), observed)


class LocusPlotExportIntegrationTests(unittest.TestCase):
    def test_write_individual_plots_exports_only_phas_loci(self):
        labeled_features = pd.DataFrame(
            [
                {"identifier": "chr1:100..196", "alib": "libA", "cID": "cluster_1", "label": "PHAS"},
                {"identifier": "chr2:500..596", "alib": "libB", "cID": "cluster_2", "label": "non-PHAS"},
            ]
        )
        clusters_data = pd.DataFrame(
            [
                {"clusterID": "cluster_1", "identifier": "chr1:100..196", "alib": "libA", "pos": 100, "abun": 5, "len": 24, "strand": "w", "tag_seq": "A1", "hits": 1},
                {"clusterID": "cluster_1", "identifier": "chr1:100..196", "alib": "libA", "pos": 124, "abun": 6, "len": 24, "strand": "w", "tag_seq": "A2", "hits": 1},
                {"clusterID": "cluster_1", "identifier": "chr1:100..196", "alib": "libA", "pos": 148, "abun": 7, "len": 24, "strand": "w", "tag_seq": "A3", "hits": 1},
                {"clusterID": "cluster_1", "identifier": "chr1:100..196", "alib": "libA", "pos": 172, "abun": 8, "len": 24, "strand": "w", "tag_seq": "A4", "hits": 1},
                {"clusterID": "cluster_2", "identifier": "chr2:500..596", "alib": "libB", "pos": 500, "abun": 9, "len": 24, "strand": "w", "tag_seq": "B1", "hits": 1},
                {"clusterID": "cluster_2", "identifier": "chr2:500..596", "alib": "libB", "pos": 524, "abun": 10, "len": 24, "strand": "w", "tag_seq": "B2", "hits": 1},
                {"clusterID": "cluster_2", "identifier": "chr2:500..596", "alib": "libB", "pos": 548, "abun": 11, "len": 24, "strand": "w", "tag_seq": "B3", "hits": 1},
                {"clusterID": "cluster_2", "identifier": "chr2:500..596", "alib": "libB", "pos": 572, "abun": 12, "len": 24, "strand": "w", "tag_seq": "B4", "hits": 1},
            ]
        )

        with tempfile.TemporaryDirectory() as outdir:
            with mock.patch.object(locus_plots, "run_parallel_with_progress", side_effect=_serial_parallel_runner):
                with mock.patch.object(locus_plots.rt, "plot_staging", "direct"):
                    with mock.patch.object(locus_plots.rt, "save_snapshot", return_value=None):
                        locus_plots.write_individual_phas_locus_plots(
                            "KNN",
                            labeled_features,
                            clusters_data,
                            job_outdir=outdir,
                            job_phase=24,
                        )

            export_path = os.path.join(outdir, "24_KNN_phasiRNAs.tsv")
            self.assertTrue(os.path.isfile(export_path))
            export_df = pd.read_csv(export_path, sep="\t")
            self.assertEqual(sorted(export_df["identifier"].unique().tolist()), ["chr1:100..196"])


if __name__ == "__main__":
    unittest.main()
