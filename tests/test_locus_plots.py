from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from phasis.stages import locus_plots


def _serial_parallel_runner(func, data, **_kwargs):
    return [func(item) for item in data]


class LocusPlotHelperTests(unittest.TestCase):
    def test_build_howell_rows_clean_mode_includes_browser_style_non_in_phase_points(self):
        trace_rows = [
            {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 10.0, "best_register": 0},
            {"anchor_position": 121, "window_start": 121, "window_end": 330, "score": 9.0, "best_register": 0},
            {"anchor_position": 122, "window_start": 122, "window_end": 331, "score": 8.0, "best_register": 0},
            {"anchor_position": 123, "window_start": 123, "window_end": 332, "score": 7.0, "best_register": 0},
        ]

        debug_rows, _, _ = locus_plots._build_howell_rows(trace_rows, 21, "w", plot_mode="debug")
        clean_rows, _, _ = locus_plots._build_howell_rows(trace_rows, 21, "w", plot_mode="clean")

        self.assertIn("other", {row["phase_relation"] for row in debug_rows})
        self.assertIn("other", {row["phase_relation"] for row in clean_rows})

    def test_plot_legend_uses_browser_style_non_in_phase_label(self):
        groups = locus_plots._build_plot_legend_groups(24, plot_mode="clean")
        labels = [item.get_label() for item in groups["howell"]]
        self.assertIn("Non-in-phase phased window", labels)

    def test_format_locus_title_italicizes_phas_classes(self):
        self.assertIn(r"$\it{PHAS}$", locus_plots._format_locus_title("libA", "chr1:100..196", 24, "PHAS"))
        self.assertIn(r"$\it{PHAS}$-like", locus_plots._format_locus_title("libA", "chr1:100..196", 24, "PHAS-like"))

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
    def test_strip_sections_hide_raw_context_in_clean_mode_and_collapse_na_sections(self):
        task = {
            "Howell_exact_support_score": 0.0,
            "Howell_origin_class": "insufficient_exact_support",
            "Howell_ambiguity_count": np.nan,
            "Howell_alt_register_count": np.nan,
            "Howell_overlap_margin": np.nan,
            "Howell_extension_window_count": np.nan,
            "Howell_extension_span_nt": np.nan,
            "Howell_origin_window_count": np.nan,
            "Howell_origin_frame_count": np.nan,
            "Howell_origin_margin": np.nan,
            "Howell_additional_peak_count": 0,
            "Howell_additional_peak_best_score": np.nan,
            "Howell_crowding_window_count": 0,
            "Howell_crowding_best_score": np.nan,
            "Howell_crowding_score_gap": np.nan,
            "final_class": "non-PHAS",
        }
        payload = locus_plots._build_ambiguity_sidebar_payload(task, plot_mode="clean")
        sections = locus_plots._build_strip_sections(task, payload, plot_mode="clean")
        titles = [section["title"] for section in sections]

        self.assertNotIn("Raw context", titles)
        self.assertNotIn("Coherent extension", titles)
        self.assertNotIn("Origin ambiguity", titles)
        self.assertIn("Interpretation", titles)
        self.assertIn("Notes", titles)

    def test_clean_mode_analyze_single_locus_uses_browser_style_other_points(self):
        task = {
            "plot_path": "/tmp/test_plot.png",
            "title_text": r"libA | chr1:100..196 | 24-$\it{PHAS}$",
            "identifier_text": "chr1:100..196",
            "cid_value": "cluster_1",
            "alib_value": "libA",
            "phase": 24,
            "final_class": "PHAS",
            "cluster_rows": [
                {"pos": 100, "abun": 5, "len": 24, "strand": "w", "tag_seq": "A1", "hits": 1},
                {"pos": 124, "abun": 6, "len": 24, "strand": "w", "tag_seq": "A2", "hits": 1},
                {"pos": 148, "abun": 7, "len": 24, "strand": "w", "tag_seq": "A3", "hits": 1},
                {"pos": 172, "abun": 8, "len": 24, "strand": "w", "tag_seq": "A4", "hits": 1},
            ],
        }
        browser_trace_context = {
            "w": [
                {"anchor_position": 100, "window_start": 100, "window_end": 339, "score": 14.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": True},
                {"anchor_position": 124, "window_start": 124, "window_end": 363, "score": 12.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": False},
                {"anchor_position": 125, "window_start": 125, "window_end": 364, "score": 11.5, "best_register": 0, "phase_relation": "offset", "is_hpsp": False},
                {"anchor_position": 130, "window_start": 130, "window_end": 369, "score": 11.0, "best_register": 0, "phase_relation": "other", "is_hpsp": False},
            ],
            "c": [],
            "strand_hpsp_rows": {"w": {"anchor_position": 100, "window_start": 100, "window_end": 339, "score": 14.0, "best_register": 0}, "c": None},
            "strand_register_origins": {"w": 100, "c": None},
            "winner_strand": "w",
            "winner_row": {"anchor_position": 100, "window_start": 100, "window_end": 339, "score": 14.0, "best_register": 0},
            "Howell_crowding_window_count": 5,
            "Howell_crowding_best_score": 11.0,
            "Howell_crowding_score_gap": 3.0,
            "crowding_rows": [
                {"anchor_position": 130, "window_start": 130, "window_end": 369, "score": 11.0, "phase_relation": "other", "is_hpsp": False}
            ],
        }
        exact_context = {
            "summary": {
                "Howell_exact_support_score": 9.0,
                "Howell_ambiguity_count": 1,
                "Howell_alt_register_count": 0,
                "Howell_overlap_margin": 0.5,
                "Howell_extension_window_count": 0,
                "Howell_extension_span_nt": 210,
                "Howell_origin_window_count": 1,
                "Howell_origin_frame_count": 1,
                "Howell_origin_margin": 0.5,
                "Howell_origin_class": "ambiguous_origin",
            },
            "competing_windows": [],
        }

        with mock.patch.object(locus_plots.rt, "locus_plot_mode", "clean"):
            with mock.patch.object(
                locus_plots.st_feat,
                "enumerate_relaxed_howell_trace",
                return_value={"w": browser_trace_context["w"], "c": []},
            ):
                with mock.patch.object(
                    locus_plots.st_feat,
                    "classify_browser_style_relaxed_trace",
                    return_value=browser_trace_context,
                ):
                    with mock.patch.object(locus_plots.st_feat, "collect_exact_only_peak_competitors", return_value=exact_context):
                        result = locus_plots._analyze_single_locus(task)

        phase_relations = {row["phase_relation"] for row in result["plot_payload"]["howell_rows"]}
        self.assertIn("other", phase_relations)
        self.assertNotIn("competitor", phase_relations)
        self.assertEqual(int(result["plot_payload"]["Howell_crowding_window_count"]), 5)

    def test_write_individual_plots_routes_phas_like_to_separate_outputs(self):
        labeled_features = pd.DataFrame(
            [
                {"identifier": "chr1:100..196", "alib": "libA", "cID": "cluster_1", "label": "PHAS", "final_class": "PHAS"},
                {"identifier": "chr2:500..596", "alib": "libB", "cID": "cluster_2", "label": "non-PHAS", "final_class": "PHAS-like"},
                {"identifier": "chr3:900..996", "alib": "libC", "cID": "cluster_3", "label": "non-PHAS", "final_class": "non-PHAS"},
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
                    with mock.patch.object(locus_plots.rt, "locus_plot_mode", "clean"):
                        with mock.patch.object(locus_plots.rt, "save_snapshot", return_value=None):
                            locus_plots.write_individual_phas_locus_plots(
                                "KNN",
                                labeled_features,
                                clusters_data,
                                job_outdir=outdir,
                                job_phase=24,
                            )

            phas_export = pd.read_csv(os.path.join(outdir, "24_KNN_phasiRNAs.tsv"), sep="\t")
            phas_like_export = pd.read_csv(os.path.join(outdir, "24_KNN_PHAS_like_phasiRNAs.tsv"), sep="\t")
            self.assertEqual(sorted(phas_export["identifier"].unique().tolist()), ["chr1:100..196"])
            self.assertEqual(sorted(phas_like_export["identifier"].unique().tolist()), ["chr2:500..596"])
            self.assertFalse(os.path.exists(os.path.join(outdir, "24_KNN_non_PHAS_locus_plots")))

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
