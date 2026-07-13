from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
        self.assertIn("phasiRNA", groups["abundance_labels"])
        self.assertEqual(groups["howell"][0].get_markerfacecolor(), locus_plots._main_unit_color(24))

        phasiRNA_halo = groups["abundance"][2]
        self.assertIsInstance(phasiRNA_halo, tuple)
        self.assertEqual(
            {handle.get_markeredgecolor() for handle in phasiRNA_halo},
            {locus_plots._main_unit_color(24)},
        )
        self.assertEqual(
            {handle.get_markerfacecolor() for handle in phasiRNA_halo},
            {"none"},
        )

    def test_build_howell_rows_demotes_non_main_exact_points_when_membership_known(self):
        trace_rows = [
            {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": True},
            {"anchor_position": 121, "window_start": 121, "window_end": 330, "score": 18.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": False},
            {"anchor_position": 142, "window_start": 142, "window_end": 351, "score": 16.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": False},
            {"anchor_position": 143, "window_start": 143, "window_end": 352, "score": 15.0, "best_register": 0, "phase_relation": "offset", "is_hpsp": False},
        ]
        trace_context = {
            "w": trace_rows,
            "strand_hpsp_rows": {"w": trace_rows[0]},
            "strand_register_origins": {"w": 100},
        }

        rows, _, _ = locus_plots._build_howell_rows(
            trace_rows,
            21,
            "w",
            plot_mode="clean",
            trace_context=trace_context,
            exact_color=locus_plots._main_unit_color(21),
            main_unit_positions={("w", 100), ("w", 121)},
        )
        by_x = {round(row["x"]): row for row in rows}

        self.assertEqual(by_x[121]["facecolor"], locus_plots._main_unit_color(21))
        self.assertEqual(by_x[142]["edgecolor"], locus_plots.CONTEXT_PHASE_BLUE)
        self.assertEqual(by_x[142]["facecolor"], locus_plots.CONTEXT_PHASE_BLUE)
        self.assertLess(by_x[142]["alpha"], by_x[121]["alpha"])
        self.assertEqual(by_x[143]["edgecolor"], locus_plots.CONTEXT_PHASE_BLUE)
        self.assertEqual(by_x[143]["facecolor"], "none")

    def test_grouped_legends_add_panel_context_headers(self):
        fig = plt.figure(figsize=(4, 3))
        try:
            locus_plots._add_grouped_legends(
                fig,
                21,
                main_left=0.08,
                main_right=0.80,
                legend_y=0.88,
                plot_mode="clean",
            )
            labels = [text.get_text() for text in fig.texts]
            legend_titles = [legend.get_title().get_text() for legend in fig.legends]
        finally:
            plt.close(fig)

        self.assertIn("Abundance context", labels)
        self.assertIn("Score context", labels)
        self.assertNotIn("Abundance panel", legend_titles)
        self.assertNotIn("Score panel", legend_titles)

    def test_build_plot_legend_groups_can_include_grouped_alternative_entries(self):
        alt_groups = [
            {"label": "Secondary phased windows", "colors": ["#AA5500"]},
            {"label": "Overlapping alternative windows", "colors": ["#CC6600"]},
        ]
        groups = locus_plots._build_plot_legend_groups(24, plot_mode="clean", alternative_legend_groups=alt_groups)
        self.assertEqual([item["label"] for item in groups["alternatives"]], ["Secondary phased windows", "Overlapping alternative windows"])

    def test_promoted_unit_pair_counts_require_both_exported_strands(self):
        c_only = {
            "paired_unit": True,
            "members": [
                {"strand": "c", "rows": [{"anchor_position": 100}], "peak_row": {"anchor_position": 100}},
            ],
        }
        w_only = {
            "paired_unit": True,
            "members": [
                {"strand": "w", "rows": [{"anchor_position": 121}], "peak_row": {"anchor_position": 121}},
            ],
        }
        paired = {
            "members": [
                {"strand": "w", "rows": [{"anchor_position": 121}], "peak_row": {"anchor_position": 121}},
                {"strand": "c", "rows": [{"anchor_position": 100}], "peak_row": {"anchor_position": 100}},
            ],
        }

        self.assertEqual(locus_plots._promoted_unit_pair_counts([c_only]), (0, 1))
        self.assertEqual(locus_plots._promoted_unit_pair_counts([w_only]), (0, 1))
        self.assertEqual(locus_plots._promoted_unit_pair_counts([paired]), (1, 0))

    def test_promoted_unit_pair_counts_ignore_empty_partner_members(self):
        unit = {
            "paired_unit": True,
            "members": [
                {"strand": "w", "rows": [{"anchor_position": 121}], "peak_row": {"anchor_position": 121}},
                {"strand": "c", "rows": [], "peak_row": None},
            ],
        }

        self.assertEqual(locus_plots._promoted_unit_pair_counts([unit]), (0, 1))

    def test_build_alternative_plot_layers_keeps_merged_strand_pair_on_one_color(self):
        summary = {
            "additional_peak_groups": [],
            "overlapping_alt_groups": [
                {
                    "category": "overlapping_alternative",
                    "peak_score": 15.0,
                    "shift_nt": 10,
                    "strand": "w",
                    "members": [
                        {
                            "strand": "w",
                            "rows": [
                                {"anchor_position": 110, "window_start": 110, "window_end": 319, "score": 15.0, "best_register": 0},
                                {"anchor_position": 131, "window_start": 131, "window_end": 340, "score": 14.0, "best_register": 0},
                            ],
                            "peak_row": {"anchor_position": 110, "window_start": 110, "window_end": 319, "score": 15.0, "best_register": 0},
                            "peak_score": 15.0,
                            "register_origin": 110,
                            "shift_nt": 10,
                            "min_start": 110,
                            "max_end": 340,
                        },
                        {
                            "strand": "c",
                            "rows": [
                                {"anchor_position": 301, "window_start": 92, "window_end": 301, "score": 14.5, "best_register": 0},
                                {"anchor_position": 280, "window_start": 71, "window_end": 280, "score": 13.5, "best_register": 0},
                            ],
                            "peak_row": {"anchor_position": 301, "window_start": 92, "window_end": 301, "score": 14.5, "best_register": 0},
                            "peak_score": 14.5,
                            "register_origin": 301,
                            "shift_nt": 10,
                            "min_start": 92,
                            "max_end": 301,
                        },
                    ],
                }
            ],
        }

        layers = locus_plots._build_alternative_plot_layers(summary, 21)

        self.assertEqual(len(layers["legend_groups"]["overlapping_alternative"]["colors"]), 1)
        self.assertTrue(layers["guide_specs_w"])
        self.assertTrue(layers["guide_specs_c"])
        overlay_colors = {row["edgecolor"] for row in layers["overlay_rows"]}
        self.assertEqual(len(overlay_colors), 1)

    def test_build_alternative_plot_layers_uses_distinct_category_colors(self):
        summary = {
            "additional_peak_groups": [
                {
                    "category": "other_local_peak",
                    "peak_score": 15.0,
                    "shift_nt": None,
                    "strand": "w",
                    "members": [
                        {
                            "strand": "w",
                            "rows": [
                                {"anchor_position": 500, "window_start": 500, "window_end": 709, "score": 15.0, "best_register": 0},
                                {"anchor_position": 521, "window_start": 521, "window_end": 730, "score": 14.0, "best_register": 0},
                            ],
                            "peak_row": {"anchor_position": 500, "window_start": 500, "window_end": 709, "score": 15.0, "best_register": 0},
                            "peak_score": 15.0,
                            "register_origin": 500,
                            "shift_nt": None,
                            "min_start": 500,
                            "max_end": 730,
                        }
                    ],
                }
            ],
            "overlapping_alt_groups": [
                {
                    "category": "overlapping_alternative",
                    "peak_score": 14.5,
                    "shift_nt": 12,
                    "strand": "w",
                    "members": [
                        {
                            "strand": "w",
                            "rows": [
                                {"anchor_position": 112, "window_start": 112, "window_end": 321, "score": 14.5, "best_register": 0},
                                {"anchor_position": 133, "window_start": 133, "window_end": 342, "score": 13.8, "best_register": 0},
                            ],
                            "peak_row": {"anchor_position": 112, "window_start": 112, "window_end": 321, "score": 14.5, "best_register": 0},
                            "peak_score": 14.5,
                            "register_origin": 112,
                            "shift_nt": 12,
                            "min_start": 112,
                            "max_end": 342,
                        }
                    ],
                }
            ],
        }

        layers = locus_plots._build_alternative_plot_layers(summary, 21)

        overlap_colors = set(layers["legend_groups"]["overlapping_alternative"]["colors"])
        other_colors = set(layers["legend_groups"]["other_local_peak"]["colors"])
        self.assertEqual(len(overlap_colors), 1)
        self.assertEqual(len(other_colors), 1)
        self.assertNotEqual(overlap_colors, other_colors)

    def test_alternative_color_series_spreads_successive_secondary_unit_tones(self):
        colors = locus_plots._alternative_color_series(21, "other_local_peak", count=3)
        self.assertEqual(len(colors), 3)
        self.assertEqual(len(set(colors)), 3)
        self.assertNotIn(locus_plots.READ_LEN_COLORS[23].upper(), {color.upper() for color in colors})

    def test_format_locus_title_italicizes_phas_classes(self):
        self.assertIn(r"$\it{PHAS}$", locus_plots._format_locus_title("libA", "chr1:100..196", 24, "PHAS"))
        self.assertIn(r"$\it{PHAS}$-like", locus_plots._format_locus_title("libA", "chr1:100..196", 24, "PHAS-like"))

    def test_remote_mount_detection_from_mountinfo(self):
        mountinfo = (
            "29 23 0:25 / / rw,relatime - apfs /dev/disk3s1 rw\n"
            "44 29 0:99 / /quobyte rw,relatime - fuse.quobyte quobyte rw\n"
        )
        is_remote, mount_point, fs_type = locus_plots._detect_remote_filesystem(
            "/quobyte/project/run/24_PHAS_locus_plots",
            mountinfo_text=mountinfo,
        )
        self.assertTrue(is_remote)
        self.assertEqual(mount_point, "/quobyte")
        self.assertEqual(fs_type, "fuse.quobyte")

    def test_remote_path_prefix_fallback_detects_quobyte(self):
        is_remote, mount_point, fs_type = locus_plots._detect_remote_filesystem(
            "/quobyte/project/run/24_PHAS_locus_plots",
            mountinfo_text="",
        )
        self.assertTrue(is_remote)
        self.assertIsNone(mount_point)
        self.assertEqual(fs_type, "path-prefix")

    def test_scheduler_driven_auto_mode_prefers_local_staging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(locus_plots.rt, "plot_staging", None):
                strategy = locus_plots._resolve_plot_staging_strategy(
                    "/tmp/24_PHAS_locus_plots",
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
                        "/tmp/24_PHAS_locus_plots",
                        env={"Phasis_PLOT_STAGING": "local", "TMPDIR": "/path/that/does/not/exist"},
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
        self.assertTrue(all("window_unit_id" in row for row in export_rows))
        self.assertTrue(all("window_unit_role" in row for row in export_rows))


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
            "Howell_overlapping_alt_count": 0,
            "Howell_overlapping_alt_best_score": np.nan,
            "Howell_overlapping_alt_best_shift_nt": np.nan,
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
        self.assertEqual(payload["relaxed_peak_score"], "NA")
        note_lines = [line for section in sections if section["title"] == "Notes" for line in section["lines"]]
        self.assertTrue(any("10-cycle window" in line for line in note_lines))
        self.assertFalse(any("Grey" in line for line in note_lines))

    def test_context_section_uses_non_in_phase_title_and_plural_sentence_without_promoted_units(self):
        task = {
            "Howell_exact_support_score": 14.7,
            "Peak_Howell_score": 15.14,
            "Howell_origin_class": "coherent_extension",
            "Howell_extension_window_count": 86,
            "Howell_extension_span_nt": 317,
            "Howell_origin_window_count": 0,
            "Howell_origin_frame_count": 0,
            "Howell_origin_margin": np.nan,
            "Howell_exact_context_window_count": 2,
            "Howell_exact_context_best_score": 13.73,
            "Howell_exact_context_score_gap": 0.97,
            "Howell_additional_peak_count": 2,
            "Howell_additional_peak_best_score": 12.9,
            "Howell_promoted_additional_peak_count": 0,
            "Howell_promoted_additional_peak_best_score": np.nan,
            "Howell_overlapping_alt_count": 0,
            "Howell_overlapping_alt_best_score": np.nan,
            "Howell_overlapping_alt_best_shift_nt": np.nan,
            "final_class": "PHAS",
        }

        payload = locus_plots._build_ambiguity_sidebar_payload(task, plot_mode="clean")
        sections = locus_plots._build_strip_sections(task, payload, plot_mode="clean")
        context_section = next(section for section in sections if section["title"] == "Exact-only non-in-phase context")

        self.assertEqual(
            payload["non_in_phase_context_sentence"],
            "Two exact-only non-in-phase context windows overlap the main scored peak on the same strand.",
        )
        self.assertIn(payload["non_in_phase_context_sentence"], context_section["lines"])
        self.assertNotIn("Overlapping alternative windows", [section["title"] for section in sections])
        self.assertNotIn("Secondary phased windows", [section["title"] for section in sections])

    def test_context_section_removes_duplicate_context_line_from_interpretation(self):
        task = {
            "Howell_exact_support_score": 14.7,
            "Peak_Howell_score": 15.14,
            "Howell_origin_class": "coherent_extension",
            "Howell_extension_window_count": 86,
            "Howell_extension_span_nt": 317,
            "Howell_origin_window_count": 0,
            "Howell_origin_frame_count": 0,
            "Howell_origin_margin": np.nan,
            "Howell_overlapping_alt_count": 0,
            "Howell_promoted_additional_peak_count": 0,
            "Howell_exact_context_window_count": 2,
            "Howell_exact_context_best_score": 13.73,
            "Howell_exact_context_score_gap": 0.97,
            "final_class": "PHAS",
        }

        payload = locus_plots._build_ambiguity_sidebar_payload(task, plot_mode="clean")
        sections = locus_plots._build_strip_sections(task, payload, plot_mode="clean")
        interpretation = next(section for section in sections if section["title"] == "Interpretation")

        self.assertFalse(any("non-in-phase context" in str(line).lower() for line in interpretation["lines"]))

    def test_origin_ambiguity_section_hidden_for_zero_zero_na(self):
        task = {
            "Howell_exact_support_score": 21.94,
            "Peak_Howell_score": 24.81,
            "Howell_origin_class": "coherent_extension",
            "Howell_origin_window_count": 0,
            "Howell_origin_frame_count": 0,
            "Howell_origin_margin": np.nan,
            "final_class": "PHAS",
        }

        payload = locus_plots._build_ambiguity_sidebar_payload(task, plot_mode="clean")
        sections = locus_plots._build_strip_sections(task, payload, plot_mode="clean")
        self.assertNotIn("Origin ambiguity", [section["title"] for section in sections])

    def test_score_labels_clarify_float_margin_and_gap_metrics(self):
        task = {
            "Howell_exact_support_score": 21.94,
            "Peak_Howell_score": 24.81,
            "Howell_origin_class": "ambiguous_origin",
            "Howell_origin_window_count": 2,
            "Howell_origin_frame_count": 1,
            "Howell_origin_margin": 0.5,
            "Howell_exact_context_window_count": 3,
            "Howell_exact_context_best_score": 24.2,
            "Howell_exact_context_score_gap": 0.61,
            "final_class": "PHAS",
        }

        payload = locus_plots._build_ambiguity_sidebar_payload(task, plot_mode="clean")
        sections = locus_plots._build_strip_sections(task, payload, plot_mode="clean")
        origin_section = next(section for section in sections if section["title"] == "Origin ambiguity")
        context_section = next(section for section in sections if section["title"] == "Exact-only non-in-phase context")

        self.assertIn("Score margin: 0.50", origin_section["lines"])
        self.assertIn("Score gap: 0.61", context_section["lines"])

    def test_context_section_uses_singular_sentence_for_one_window(self):
        task = {
            "Howell_exact_support_score": 14.7,
            "Peak_Howell_score": 15.14,
            "Howell_origin_class": "coherent_extension",
            "Howell_overlapping_alt_count": 0,
            "Howell_promoted_additional_peak_count": 0,
            "Howell_exact_context_window_count": 1,
            "Howell_exact_context_best_score": 13.73,
            "Howell_exact_context_score_gap": 0.97,
            "final_class": "PHAS",
        }

        payload = locus_plots._build_ambiguity_sidebar_payload(task, plot_mode="clean")
        self.assertEqual(
            payload["non_in_phase_context_sentence"],
            "One exact-only non-in-phase context window overlaps the main scored peak on the same strand.",
        )

    def test_context_section_sentence_stays_exact_only_when_promoted_units_exist(self):
        task = {
            "Howell_exact_support_score": 14.7,
            "Peak_Howell_score": 15.14,
            "Howell_origin_class": "coherent_extension",
            "Howell_additional_peak_count": 1,
            "Howell_additional_peak_best_score": 12.9,
            "Howell_promoted_additional_peak_count": 0,
            "Howell_promoted_additional_peak_best_score": np.nan,
            "Howell_overlapping_alt_count": 1,
            "Howell_overlapping_alt_best_score": 14.5,
            "Howell_overlapping_alt_best_shift_nt": 2.0,
            "Howell_exact_context_window_count": 2,
            "Howell_exact_context_best_score": 13.73,
            "Howell_exact_context_score_gap": 0.97,
            "final_class": "PHAS",
        }

        payload = locus_plots._build_ambiguity_sidebar_payload(task, plot_mode="clean")
        sections = locus_plots._build_strip_sections(task, payload, plot_mode="clean")
        context_section = next(section for section in sections if section["title"] == "Exact-only non-in-phase context")

        self.assertEqual(
            payload["non_in_phase_context_sentence"],
            "Two exact-only non-in-phase context windows overlap the main scored peak on the same strand.",
        )
        self.assertIn(payload["non_in_phase_context_sentence"], context_section["lines"])
        self.assertIn("Overlapping alternative windows", [section["title"] for section in sections])

    def test_strip_top_summary_uses_bold_inline_labels(self):
        task = {
            "Howell_exact_support_score": 21.94,
            "Peak_Howell_score": 24.81,
            "Howell_origin_class": "coherent_extension",
            "Main_opposite_partner_present": True,
            "Main_opposite_partner_shift_nt": 2.0,
            "final_class": "PHAS",
        }

        payload = locus_plots._build_ambiguity_sidebar_payload(task, plot_mode="clean")
        sections = locus_plots._build_strip_sections(task, payload, plot_mode="clean")
        top_section = sections[0]

        self.assertIsNone(top_section["title"])
        self.assertEqual(
            top_section["lines"][:3],
            [
                "Exact-only support: 21.94",
                "Relaxed peak: 24.81",
                "Class: Coherent extension",
            ],
        )
        self.assertEqual(top_section["line_weights"][:3], ["bold", "bold", "bold"])
        self.assertIn("Main opposite-strand partner: detected (+2 nt)", top_section["lines"])

    def test_strip_uses_partner_status_and_paired_secondary_counts(self):
        task = {
            "Howell_exact_support_score": 21.94,
            "Peak_Howell_score": 24.81,
            "Howell_origin_class": "coherent_extension",
            "Howell_promoted_additional_peak_count": 1,
            "Howell_promoted_additional_peak_best_score": 15.10,
            "Howell_additional_peak_count": 1,
            "Howell_additional_peak_best_score": 15.10,
            "Howell_overlapping_alt_count": 1,
            "Howell_overlapping_alt_best_score": 14.55,
            "Howell_overlapping_alt_best_shift_nt": 2.0,
            "Howell_promoted_additional_peak_paired_count": 1,
            "Howell_promoted_additional_peak_unpaired_count": 0,
            "Howell_promoted_overlapping_alt_paired_count": 0,
            "Howell_promoted_overlapping_alt_unpaired_count": 1,
            "Main_opposite_partner_present": False,
            "final_class": "PHAS",
        }

        payload = locus_plots._build_ambiguity_sidebar_payload(task, plot_mode="clean")
        sections = locus_plots._build_strip_sections(task, payload, plot_mode="clean")
        top_section = sections[0]
        other_section = next(section for section in sections if section["title"] == "Secondary phased windows")
        overlap_section = next(section for section in sections if section["title"] == "Overlapping alternative windows")

        self.assertIn("Main opposite-strand partner: not detected", top_section["lines"])
        self.assertIn("Paired: 1", other_section["lines"])
        self.assertIn("Unpaired: 0", other_section["lines"])
        self.assertIn("Paired: 0", overlap_section["lines"])
        self.assertIn("Unpaired: 1", overlap_section["lines"])

    def test_locus_layout_keeps_figure_size_but_uses_more_strip_area(self):
        self.assertEqual(locus_plots.LOCUS_LAYOUT["figsize"], (13.2, 6.8))
        self.assertEqual(locus_plots.LOCUS_LAYOUT["abun_height"], locus_plots.LOCUS_LAYOUT["howell_height"])
        self.assertLess(locus_plots.LOCUS_LAYOUT["strip_left"], 0.835)
        self.assertGreater(locus_plots.LOCUS_LAYOUT["strip_top"], 0.80)
        self.assertLess(locus_plots.LOCUS_LAYOUT["strip_bottom"], 0.13)
        self.assertLess(locus_plots.LOCUS_LAYOUT["separator_x"], 0.818)

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

    def test_analyze_single_locus_builds_grouped_alternative_overlays(self):
        task = {
            "plot_path": "/tmp/test_plot_alt.png",
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
                {"anchor_position": 100, "window_start": 100, "window_end": 339, "score": 18.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": True},
                {"anchor_position": 124, "window_start": 124, "window_end": 363, "score": 14.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": False},
                {"anchor_position": 130, "window_start": 130, "window_end": 369, "score": 13.5, "best_register": 0, "phase_relation": "other", "is_hpsp": False},
            ],
            "c": [],
            "strand_hpsp_rows": {"w": {"anchor_position": 100, "window_start": 100, "window_end": 339, "score": 18.0, "best_register": 0}, "c": None},
            "strand_register_origins": {"w": 100, "c": None},
            "winner_strand": "w",
            "winner_row": {"anchor_position": 100, "window_start": 100, "window_end": 339, "score": 18.0, "best_register": 0},
            "Howell_crowding_window_count": 1,
            "Howell_crowding_best_score": 13.5,
            "Howell_crowding_score_gap": 4.5,
            "crowding_rows": [],
        }
        exact_context = {"summary": {}, "competing_windows": []}
        alt_summary = {
            "Howell_additional_peak_count": 1,
            "Howell_additional_peak_best_score": 15.0,
            "Howell_overlapping_alt_count": 1,
            "Howell_overlapping_alt_best_score": 14.5,
            "Howell_overlapping_alt_best_shift_nt": 12.0,
            "additional_peak_groups": [
                {
                    "category": "other_local_peak",
                    "strand": "w",
                    "rows": [
                        {"anchor_position": 500, "window_start": 500, "window_end": 739, "score": 15.0, "best_register": 0},
                        {"anchor_position": 524, "window_start": 524, "window_end": 763, "score": 14.2, "best_register": 0},
                    ],
                    "peak_row": {"anchor_position": 500, "window_start": 500, "window_end": 739, "score": 15.0, "best_register": 0},
                    "peak_score": 15.0,
                    "register_origin": 500,
                    "shift_nt": None,
                    "min_start": 500,
                    "max_end": 763,
                }
            ],
            "overlapping_alt_groups": [
                {
                    "category": "overlapping_alternative",
                    "strand": "w",
                    "rows": [
                        {"anchor_position": 112, "window_start": 112, "window_end": 351, "score": 14.5, "best_register": 0},
                        {"anchor_position": 136, "window_start": 136, "window_end": 375, "score": 13.8, "best_register": 0},
                    ],
                    "peak_row": {"anchor_position": 112, "window_start": 112, "window_end": 351, "score": 14.5, "best_register": 0},
                    "peak_score": 14.5,
                    "register_origin": 112,
                    "shift_nt": 12,
                    "min_start": 112,
                    "max_end": 375,
                }
            ],
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
                        with mock.patch.object(locus_plots.st_feat, "summarize_relaxed_trace_subregions", return_value=alt_summary):
                            result = locus_plots._analyze_single_locus(task)

        payload = result["plot_payload"]
        self.assertEqual(int(payload["Howell_overlapping_alt_count"]), 1)
        self.assertEqual(float(payload["Howell_overlapping_alt_best_shift_nt"]), 12.0)
        self.assertIn("other_local_peak", payload["alternative_legend_groups"])
        self.assertIn("overlapping_alternative", payload["alternative_legend_groups"])
        self.assertTrue(payload["alternative_howell_overlay_rows"])
        self.assertEqual(int(payload["Howell_promoted_additional_peak_count"]), 1)

    def test_export_rows_include_main_unit_metadata(self):
        labeled_features = pd.DataFrame(
            [
                {"identifier": "chr1:100..196", "alib": "libA", "cID": "cluster_1", "label": "PHAS", "final_class": "PHAS"},
            ]
        )
        clusters_data = pd.DataFrame(
            [
                {"clusterID": "cluster_1", "identifier": "chr1:100..196", "alib": "libA", "pos": 100, "abun": 5, "len": 24, "strand": "w", "tag_seq": "A1", "hits": 1},
                {"clusterID": "cluster_1", "identifier": "chr1:100..196", "alib": "libA", "pos": 124, "abun": 6, "len": 24, "strand": "w", "tag_seq": "A2", "hits": 1},
                {"clusterID": "cluster_1", "identifier": "chr1:100..196", "alib": "libA", "pos": 148, "abun": 7, "len": 24, "strand": "w", "tag_seq": "A3", "hits": 1},
                {"clusterID": "cluster_1", "identifier": "chr1:100..196", "alib": "libA", "pos": 172, "abun": 8, "len": 24, "strand": "w", "tag_seq": "A4", "hits": 1},
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

            export_df = pd.read_csv(os.path.join(outdir, "24_phasiRNAs.tsv"), sep="\t")
            self.assertIn("window_unit_id", export_df.columns)
            self.assertIn("window_unit_role", export_df.columns)
            self.assertIn("window_unit_rank", export_df.columns)
            self.assertIn("window_unit_shift_nt", export_df.columns)
            self.assertEqual(sorted(export_df["window_unit_role"].unique().tolist()), ["main_hpsp"])
            self.assertEqual(sorted(export_df["window_unit_id"].unique().tolist()), ["unit_main"])

    def test_export_rows_include_main_partner_and_extension_roles(self):
        task = {
            "plot_path": "/tmp/test_plot_main_unit.png",
            "title_text": r"libA | chr1:100..196 | 21-$\it{PHAS}$",
            "identifier_text": "chr1:100..196",
            "cid_value": "cluster_1",
            "alib_value": "libA",
            "phase": 21,
            "final_class": "PHAS",
            "cluster_rows": [
                {"pos": 100, "abun": 5, "len": 21, "strand": "w", "tag_seq": "A1", "hits": 1},
                {"pos": 121, "abun": 6, "len": 21, "strand": "w", "tag_seq": "A2", "hits": 1},
                {"pos": 142, "abun": 7, "len": 21, "strand": "w", "tag_seq": "A3", "hits": 1},
                {"pos": 291, "abun": 8, "len": 21, "strand": "c", "tag_seq": "C1", "hits": 1},
                {"pos": 270, "abun": 9, "len": 21, "strand": "c", "tag_seq": "C2", "hits": 1},
                {"pos": 373, "abun": 10, "len": 21, "strand": "w", "tag_seq": "A4", "hits": 1},
                {"pos": 394, "abun": 11, "len": 21, "strand": "w", "tag_seq": "A5", "hits": 1},
            ],
        }
        trace_rows = {
            "w": [
                {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": True},
                {"anchor_position": 121, "window_start": 121, "window_end": 330, "score": 19.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": False},
                {"anchor_position": 142, "window_start": 142, "window_end": 351, "score": 18.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": False},
                {"anchor_position": 373, "window_start": 373, "window_end": 582, "score": 11.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": False},
                {"anchor_position": 394, "window_start": 394, "window_end": 603, "score": 10.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": False},
            ],
            "c": [
                {"anchor_position": 291, "window_start": 82, "window_end": 291, "score": 17.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": False},
                {"anchor_position": 270, "window_start": 61, "window_end": 270, "score": 16.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": False},
            ],
        }
        browser_trace_context = {
            "w": trace_rows["w"],
            "c": trace_rows["c"],
            "strand_hpsp_rows": {"w": trace_rows["w"][0], "c": None},
            "strand_register_origins": {"w": 100, "c": 291},
            "winner_strand": "w",
            "winner_row": trace_rows["w"][0],
            "Howell_crowding_window_count": 0,
            "Howell_crowding_best_score": np.nan,
            "Howell_crowding_score_gap": np.nan,
            "crowding_rows": [],
        }
        exact_context = {"summary": {}, "competing_windows": []}
        alt_summary = {
            "Howell_additional_peak_count": 0,
            "Howell_additional_peak_best_score": np.nan,
            "Howell_overlapping_alt_count": 0,
            "Howell_overlapping_alt_best_score": np.nan,
            "Howell_overlapping_alt_best_shift_nt": np.nan,
            "promoted_secondary_units": [],
            "promoted_additional_peak_groups": [],
            "main_biogenesis_unit": {
                "members": [
                    {
                        "unit_role": "main_hpsp",
                        "strand": "w",
                        "rows": [
                            {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0},
                            {"anchor_position": 121, "window_start": 121, "window_end": 330, "score": 19.0, "best_register": 0},
                            {"anchor_position": 142, "window_start": 142, "window_end": 351, "score": 18.0, "best_register": 0},
                        ],
                        "peak_row": {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0},
                    },
                    {
                        "unit_role": "main_partner",
                        "strand": "c",
                        "shift_nt": 4,
                        "rows": [
                            {"anchor_position": 291, "window_start": 82, "window_end": 291, "score": 17.0, "best_register": 0},
                            {"anchor_position": 270, "window_start": 61, "window_end": 270, "score": 16.0, "best_register": 0},
                        ],
                        "peak_row": {"anchor_position": 291, "window_start": 82, "window_end": 291, "score": 17.0, "best_register": 0},
                    },
                    {
                        "unit_role": "main_extension",
                        "strand": "w",
                        "shift_nt": 0,
                        "rows": [
                            {"anchor_position": 373, "window_start": 373, "window_end": 582, "score": 11.0, "best_register": 0},
                            {"anchor_position": 394, "window_start": 394, "window_end": 603, "score": 10.0, "best_register": 0},
                        ],
                        "peak_row": {"anchor_position": 373, "window_start": 373, "window_end": 582, "score": 11.0, "best_register": 0},
                    },
                ],
                "main_partner_shift_nt": 2,
            },
        }

        with mock.patch.object(locus_plots.rt, "locus_plot_mode", "clean"):
            with mock.patch.object(
                locus_plots.st_feat,
                "enumerate_relaxed_howell_trace",
                return_value=trace_rows,
            ):
                with mock.patch.object(
                    locus_plots.st_feat,
                    "classify_browser_style_relaxed_trace",
                    return_value=browser_trace_context,
                ):
                    with mock.patch.object(locus_plots.st_feat, "collect_exact_only_peak_competitors", return_value=exact_context):
                        with mock.patch.object(locus_plots.st_feat, "summarize_relaxed_trace_subregions", return_value=alt_summary):
                            result = locus_plots._analyze_single_locus(task)

        export_rows = result["phasiRNA_rows"]
        roles = sorted({row["window_unit_role"] for row in export_rows})
        unit_ids = sorted({row["window_unit_id"] for row in export_rows})
        main_partner_shifts = {
            row["window_unit_shift_nt"]
            for row in export_rows
            if row["window_unit_role"] == "main_partner"
        }
        self.assertEqual(unit_ids, ["unit_main"])
        self.assertEqual(roles, ["main_extension", "main_hpsp", "main_partner"])
        self.assertEqual(main_partner_shifts, {2})

    def test_analyze_single_locus_builds_main_partner_semantic_overlays(self):
        task = {
            "plot_path": "/tmp/test_plot_partner_overlay.png",
            "title_text": r"libA | chr1:100..196 | 21-$\it{PHAS}$",
            "identifier_text": "chr1:100..196",
            "cid_value": "cluster_1",
            "alib_value": "libA",
            "phase": 21,
            "final_class": "PHAS",
            "cluster_rows": [
                {"pos": 100, "abun": 5, "len": 21, "strand": "w", "tag_seq": "A1", "hits": 1},
                {"pos": 121, "abun": 6, "len": 21, "strand": "w", "tag_seq": "A2", "hits": 1},
                {"pos": 291, "abun": 8, "len": 21, "strand": "c", "tag_seq": "C1", "hits": 1},
                {"pos": 270, "abun": 9, "len": 21, "strand": "c", "tag_seq": "C2", "hits": 2},
                {"pos": 500, "abun": 7, "len": 21, "strand": "w", "tag_seq": "S1", "hits": 1},
                {"pos": 521, "abun": 8, "len": 21, "strand": "w", "tag_seq": "S2", "hits": 2},
            ],
        }
        trace_rows = {
            "w": [
                {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": True},
                {"anchor_position": 121, "window_start": 121, "window_end": 330, "score": 19.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": False},
            ],
            "c": [
                {"anchor_position": 291, "window_start": 82, "window_end": 291, "score": 17.0, "best_register": 0, "phase_relation": "other", "is_hpsp": False},
                {"anchor_position": 270, "window_start": 61, "window_end": 270, "score": 16.0, "best_register": 0, "phase_relation": "other", "is_hpsp": False},
            ],
        }
        browser_trace_context = {
            "w": trace_rows["w"],
            "c": trace_rows["c"],
            "strand_hpsp_rows": {"w": trace_rows["w"][0], "c": None},
            "strand_register_origins": {"w": 100, "c": 291},
            "winner_strand": "w",
            "winner_row": trace_rows["w"][0],
            "Howell_crowding_window_count": 0,
            "Howell_crowding_best_score": np.nan,
            "Howell_crowding_score_gap": np.nan,
            "crowding_rows": [],
        }
        exact_context = {"summary": {}, "competing_windows": []}
        alt_summary = {
            "Howell_additional_peak_count": 1,
            "Howell_additional_peak_best_score": 15.0,
            "Howell_overlapping_alt_count": 0,
            "Howell_overlapping_alt_best_score": np.nan,
            "Howell_overlapping_alt_best_shift_nt": np.nan,
            "promoted_secondary_units": [
                {
                    "category": "other_local_peak",
                    "shift_nt": None,
                    "members": [
                        {
                            "strand": "w",
                            "rows": [
                                {"anchor_position": 500, "window_start": 500, "window_end": 709, "score": 15.0, "best_register": 0},
                                {"anchor_position": 521, "window_start": 521, "window_end": 730, "score": 14.0, "best_register": 0},
                            ],
                            "peak_row": {"anchor_position": 500, "window_start": 500, "window_end": 709, "score": 15.0, "best_register": 0},
                            "peak_score": 15.0,
                            "register_origin": 500,
                            "shift_nt": None,
                            "min_start": 500,
                            "max_end": 730,
                        }
                    ],
                }
            ],
            "promoted_additional_peak_groups": [
                {
                    "category": "other_local_peak",
                    "shift_nt": None,
                    "strand": "w",
                    "rows": [
                        {"anchor_position": 500, "window_start": 500, "window_end": 709, "score": 15.0, "best_register": 0},
                        {"anchor_position": 521, "window_start": 521, "window_end": 730, "score": 14.0, "best_register": 0},
                    ],
                    "peak_row": {"anchor_position": 500, "window_start": 500, "window_end": 709, "score": 15.0, "best_register": 0},
                    "peak_score": 15.0,
                    "register_origin": 500,
                    "min_start": 500,
                    "max_end": 730,
                }
            ],
            "main_biogenesis_unit": {
                "members": [
                    {
                        "unit_role": "main_hpsp",
                        "strand": "w",
                        "rows": [
                            {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0},
                            {"anchor_position": 121, "window_start": 121, "window_end": 330, "score": 19.0, "best_register": 0},
                        ],
                        "peak_row": {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0},
                    },
                    {
                        "unit_role": "main_partner",
                        "strand": "c",
                        "shift_nt": -2,
                        "rows": [
                            {"anchor_position": 291, "window_start": 82, "window_end": 291, "score": 17.0, "best_register": 0},
                            {"anchor_position": 270, "window_start": 61, "window_end": 270, "score": 16.0, "best_register": 0},
                        ],
                        "peak_row": {"anchor_position": 291, "window_start": 82, "window_end": 291, "score": 17.0, "best_register": 0},
                    },
                ],
                "main_partner_shift_nt": -2,
                "main_partner_present": True,
            },
        }

        with mock.patch.object(locus_plots.rt, "locus_plot_mode", "clean"):
            with mock.patch.object(
                locus_plots.st_feat,
                "enumerate_relaxed_howell_trace",
                return_value=trace_rows,
            ):
                with mock.patch.object(
                    locus_plots.st_feat,
                    "classify_browser_style_relaxed_trace",
                    return_value=browser_trace_context,
                ):
                    with mock.patch.object(locus_plots.st_feat, "collect_exact_only_peak_competitors", return_value=exact_context):
                        with mock.patch.object(locus_plots.st_feat, "summarize_relaxed_trace_subregions", return_value=alt_summary):
                            result = locus_plots._analyze_single_locus(task)

        payload = result["plot_payload"]
        main_partner_overlay = payload["main_howell_overlay_rows"]
        self.assertTrue(main_partner_overlay)
        self.assertTrue(all(row["edgecolor"] == locus_plots._main_unit_color(21) for row in main_partner_overlay))
        self.assertTrue(all(row["zorder"] > 5.0 for row in main_partner_overlay))
        self.assertEqual(sorted({round(row["x"]) for row in main_partner_overlay}), [270, 291])

        main_abundance_overlay = payload["main_abundance_overlay_rows"]
        partner_abundance = [row for row in main_abundance_overlay if row["unit_role"] == "main_partner"]
        self.assertEqual(len(partner_abundance), 2)
        self.assertIn(locus_plots._read_length_color_hex(21), {row["facecolor"] for row in partner_abundance})
        self.assertIn("none", {row["facecolor"] for row in partner_abundance})
        self.assertTrue(all(row["edgecolor"] == locus_plots._main_unit_color(21) for row in partner_abundance))
        self.assertTrue(all(row["zorder"] > 3.0 for row in partner_abundance))
        self.assertTrue(payload["alternative_howell_overlay_rows"])
        self.assertTrue(all(row["zorder"] < main_partner_overlay[0]["zorder"] for row in payload["alternative_howell_overlay_rows"]))

        secondary_abundance = payload["secondary_abundance_overlay_rows"]
        self.assertEqual(len(secondary_abundance), 2)
        self.assertEqual({row["category"] for row in secondary_abundance}, {"other_local_peak"})
        self.assertEqual({row["edgecolor"] for row in secondary_abundance}, {payload["alternative_legend_groups"]["other_local_peak"]["colors"][0]})
        self.assertIn(locus_plots._read_length_color_hex(21), {row["facecolor"] for row in secondary_abundance})
        self.assertIn("none", {row["facecolor"] for row in secondary_abundance})
        self.assertTrue(all(row["zorder"] > 3.0 for row in secondary_abundance))

    def test_export_rows_include_promoted_secondary_unit_metadata(self):
        task = {
            "plot_path": "/tmp/test_plot_secondary_unit.png",
            "title_text": r"libA | chr1:100..196 | 21-$\it{PHAS}$",
            "identifier_text": "chr1:100..196",
            "cid_value": "cluster_1",
            "alib_value": "libA",
            "phase": 21,
            "final_class": "PHAS",
            "cluster_rows": [
                {"pos": 100, "abun": 5, "len": 21, "strand": "w", "tag_seq": "A1", "hits": 1},
                {"pos": 121, "abun": 6, "len": 21, "strand": "w", "tag_seq": "A2", "hits": 1},
                {"pos": 142, "abun": 7, "len": 21, "strand": "w", "tag_seq": "A3", "hits": 1},
                {"pos": 310, "abun": 8, "len": 21, "strand": "c", "tag_seq": "C1", "hits": 1},
                {"pos": 289, "abun": 9, "len": 21, "strand": "c", "tag_seq": "C2", "hits": 1},
            ],
        }
        trace_rows = {
            "w": [
                {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": True},
                {"anchor_position": 121, "window_start": 121, "window_end": 330, "score": 19.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": False},
                {"anchor_position": 142, "window_start": 142, "window_end": 351, "score": 18.0, "best_register": 0, "phase_relation": "exact", "is_hpsp": False},
            ],
            "c": [
                {"anchor_position": 310, "window_start": 101, "window_end": 310, "score": 15.5, "best_register": 0, "phase_relation": "exact", "is_hpsp": False},
                {"anchor_position": 289, "window_start": 80, "window_end": 289, "score": 14.8, "best_register": 0, "phase_relation": "exact", "is_hpsp": False},
            ],
        }
        browser_trace_context = {
            "w": trace_rows["w"],
            "c": trace_rows["c"],
            "strand_hpsp_rows": {"w": trace_rows["w"][0], "c": None},
            "strand_register_origins": {"w": 100, "c": 310},
            "winner_strand": "w",
            "winner_row": trace_rows["w"][0],
            "Howell_crowding_window_count": 0,
            "Howell_crowding_best_score": np.nan,
            "Howell_crowding_score_gap": np.nan,
            "crowding_rows": [],
        }
        exact_context = {"summary": {}, "competing_windows": []}
        alt_summary = {
            "Howell_additional_peak_count": 0,
            "Howell_additional_peak_best_score": np.nan,
            "Howell_overlapping_alt_count": 1,
            "Howell_overlapping_alt_best_score": 15.5,
            "Howell_overlapping_alt_best_shift_nt": 1.0,
            "promoted_secondary_units": [
                {
                    "category": "overlapping_alternative",
                    "shift_nt": 1,
                    "members": [
                        {
                            "strand": "c",
                            "rows": [
                                {"anchor_position": 310, "window_start": 101, "window_end": 310, "score": 15.5, "best_register": 0},
                                {"anchor_position": 289, "window_start": 80, "window_end": 289, "score": 14.8, "best_register": 0},
                            ],
                            "peak_row": {"anchor_position": 310, "window_start": 101, "window_end": 310, "score": 15.5, "best_register": 0},
                        }
                    ],
                }
            ],
            "promoted_additional_peak_groups": [],
            "main_biogenesis_unit": {
                "members": [
                    {
                        "unit_role": "main_hpsp",
                        "strand": "w",
                        "rows": [
                            {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0},
                            {"anchor_position": 121, "window_start": 121, "window_end": 330, "score": 19.0, "best_register": 0},
                            {"anchor_position": 142, "window_start": 142, "window_end": 351, "score": 18.0, "best_register": 0},
                        ],
                        "peak_row": {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0},
                    }
                ],
            },
        }

        with mock.patch.object(locus_plots.rt, "locus_plot_mode", "clean"):
            with mock.patch.object(
                locus_plots.st_feat,
                "enumerate_relaxed_howell_trace",
                return_value=trace_rows,
            ):
                with mock.patch.object(
                    locus_plots.st_feat,
                    "classify_browser_style_relaxed_trace",
                    return_value=browser_trace_context,
                ):
                    with mock.patch.object(locus_plots.st_feat, "collect_exact_only_peak_competitors", return_value=exact_context):
                        with mock.patch.object(locus_plots.st_feat, "summarize_relaxed_trace_subregions", return_value=alt_summary):
                            result = locus_plots._analyze_single_locus(task)

        export_rows = result["phasiRNA_rows"]
        secondary_rows = [row for row in export_rows if row["window_unit_role"] == "overlapping_alternative"]
        self.assertTrue(secondary_rows)
        self.assertEqual({row["window_unit_id"] for row in secondary_rows}, {"unit_secondary_1"})
        self.assertEqual({row["window_unit_rank"] for row in secondary_rows}, {1})
        self.assertEqual({row["window_unit_shift_nt"] for row in secondary_rows}, {1})

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

            phas_export = pd.read_csv(os.path.join(outdir, "24_phasiRNAs.tsv"), sep="\t")
            phas_like_export = pd.read_csv(
                os.path.join(outdir, "24_PHAS_like", "24_PHAS_like_phasiRNAs.tsv"),
                sep="\t",
            )
            self.assertEqual(sorted(phas_export["identifier"].unique().tolist()), ["chr1:100..196"])
            self.assertEqual(sorted(phas_like_export["identifier"].unique().tolist()), ["chr2:500..596"])
            self.assertTrue(os.path.isdir(os.path.join(outdir, "24_PHAS_like", "locus_plots")))
            self.assertFalse(os.path.exists(os.path.join(outdir, "24_non_PHAS_locus_plots")))

    def test_duplicate_library_locus_calls_use_one_strongest_plot_source(self):
        labeled_features = pd.DataFrame(
            [
                {
                    "identifier": "chr1:100..196",
                    "alib": "libA",
                    "cID": "cluster_1",
                    "label": "PHAS",
                    "final_class": "PHAS",
                    "Howell_exact_support_score": 10.0,
                    "Peak_Howell_score": 30.0,
                    "phasis_score": 300.0,
                },
                {
                    "identifier": "chr1:100..196",
                    "alib": "libA",
                    "cID": "cluster_2",
                    "label": "PHAS",
                    "final_class": "PHAS",
                    "Howell_exact_support_score": 20.0,
                    "Peak_Howell_score": 25.0,
                    "phasis_score": 250.0,
                },
            ]
        )
        clusters_data = pd.DataFrame(
            [
                {"clusterID": "cluster_1", "identifier": "chr1:100..196", "alib": "libA", "pos": 100, "abun": 5, "len": 24, "strand": "w", "tag_seq": "A1", "hits": 1},
                {"clusterID": "cluster_1", "identifier": "chr1:100..196", "alib": "libA", "pos": 124, "abun": 6, "len": 24, "strand": "w", "tag_seq": "A2", "hits": 1},
                {"clusterID": "cluster_1", "identifier": "chr1:100..196", "alib": "libA", "pos": 148, "abun": 7, "len": 24, "strand": "w", "tag_seq": "A3", "hits": 1},
                {"clusterID": "cluster_1", "identifier": "chr1:100..196", "alib": "libA", "pos": 172, "abun": 8, "len": 24, "strand": "w", "tag_seq": "A4", "hits": 1},
                {"clusterID": "cluster_2", "identifier": "chr1:100..196", "alib": "libA", "pos": 100, "abun": 9, "len": 24, "strand": "w", "tag_seq": "B1", "hits": 1},
                {"clusterID": "cluster_2", "identifier": "chr1:100..196", "alib": "libA", "pos": 124, "abun": 10, "len": 24, "strand": "w", "tag_seq": "B2", "hits": 1},
                {"clusterID": "cluster_2", "identifier": "chr1:100..196", "alib": "libA", "pos": 148, "abun": 11, "len": 24, "strand": "w", "tag_seq": "B3", "hits": 1},
                {"clusterID": "cluster_2", "identifier": "chr1:100..196", "alib": "libA", "pos": 172, "abun": 12, "len": 24, "strand": "w", "tag_seq": "B4", "hits": 1},
            ]
        )

        with tempfile.TemporaryDirectory() as outdir:
            with mock.patch.object(locus_plots, "run_parallel_with_progress", side_effect=_serial_parallel_runner):
                with mock.patch.object(locus_plots.rt, "plot_staging", "direct"):
                    with mock.patch.object(locus_plots.rt, "save_snapshot", return_value=None):
                        locus_plots.write_individual_phas_locus_plots(
                            "GMM",
                            labeled_features,
                            clusters_data,
                            job_outdir=outdir,
                            job_phase=24,
                        )

            plot_dir = os.path.join(outdir, "24_PHAS_locus_plots")
            self.assertEqual(len([name for name in os.listdir(plot_dir) if name.endswith(".png")]), 1)
            export_df = pd.read_csv(os.path.join(outdir, "24_phasiRNAs.tsv"), sep="\t")
            self.assertEqual(set(export_df["cID"].astype(str)), {"cluster_2"})

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

            export_path = os.path.join(outdir, "24_phasiRNAs.tsv")
            self.assertTrue(os.path.isfile(export_path))
            export_df = pd.read_csv(export_path, sep="\t")
            self.assertEqual(sorted(export_df["identifier"].unique().tolist()), ["chr1:100..196"])
            phas_like_export = pd.read_csv(
                os.path.join(outdir, "24_PHAS_like", "24_PHAS_like_phasiRNAs.tsv"),
                sep="\t",
            )
            self.assertTrue(phas_like_export.empty)


if __name__ == "__main__":
    unittest.main()
