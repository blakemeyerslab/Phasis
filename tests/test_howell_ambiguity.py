from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from phasis.stages import feature_assembly
from phasis.stages import locus_plots
from phasis.stages import output


class _DummyPool:
    def map(self, func, jobs):
        return [func(job) for job in jobs]

    def close(self):
        return None

    def join(self):
        return None

    def terminate(self):
        return None


class HowellAmbiguityScoringTests(unittest.TestCase):
    def test_summarize_relaxed_trace_subregions_counts_additional_regions_above_cutoff(self):
        trace = {
            "w": [
                {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 8.0},
                {"anchor_position": 120, "window_start": 120, "window_end": 329, "score": 15.0},
                {"anchor_position": 140, "window_start": 140, "window_end": 349, "score": 13.0},
                {"anchor_position": 600, "window_start": 600, "window_end": 809, "score": 14.0},
                {"anchor_position": 620, "window_start": 620, "window_end": 829, "score": 17.0},
            ],
            "c": [
                {"anchor_position": 1200, "window_start": 991, "window_end": 1200, "score": 12.6},
                {"anchor_position": 1220, "window_start": 1011, "window_end": 1220, "score": 11.0},
            ],
        }

        summary = feature_assembly.summarize_relaxed_trace_subregions(trace, score_cutoff=12.5, phase=21)

        self.assertEqual(summary["Howell_additional_peak_count"], 2)
        self.assertAlmostEqual(summary["Howell_additional_peak_best_score"], 15.0, places=6)
        self.assertEqual(summary["Howell_overlapping_alt_count"], 0)
        self.assertEqual(len(summary["additional_peak_groups"]), 2)

    def test_summarize_relaxed_trace_subregions_detects_overlapping_shifted_alternative(self):
        trace = {
            "w": [
                {"anchor_position": 100, "window_start": 100, "window_end": 339, "score": 20.0, "best_register": 0},
                {"anchor_position": 124, "window_start": 124, "window_end": 363, "score": 18.5, "best_register": 0},
                {"anchor_position": 112, "window_start": 112, "window_end": 351, "score": 18.0, "best_register": 0},
                {"anchor_position": 136, "window_start": 136, "window_end": 375, "score": 17.5, "best_register": 0},
            ],
            "c": [],
        }

        summary = feature_assembly.summarize_relaxed_trace_subregions(trace, score_cutoff=12.5, phase=24)

        self.assertEqual(summary["Howell_additional_peak_count"], 0)
        self.assertEqual(summary["Howell_overlapping_alt_count"], 1)
        self.assertAlmostEqual(summary["Howell_overlapping_alt_best_score"], 18.0, places=6)
        self.assertEqual(int(summary["Howell_overlapping_alt_best_shift_nt"]), 12)
        self.assertEqual(len(summary["overlapping_alt_groups"]), 1)

    def test_summarize_relaxed_trace_subregions_merges_complementary_distal_peak_pair(self):
        trace = {
            "w": [
                {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0},
                {"anchor_position": 600, "window_start": 600, "window_end": 809, "score": 16.0, "best_register": 0},
            ],
            "c": [
                {"anchor_position": 791, "window_start": 582, "window_end": 791, "score": 15.0, "best_register": 0},
            ],
        }

        summary = feature_assembly.summarize_relaxed_trace_subregions(trace, score_cutoff=12.5, phase=21)

        self.assertEqual(summary["Howell_additional_peak_count"], 1)
        self.assertEqual(summary["Howell_overlapping_alt_count"], 0)
        self.assertEqual(len(summary["additional_peak_groups"]), 1)
        unit = summary["additional_peak_groups"][0]
        self.assertEqual(int(unit["member_count"]), 2)
        self.assertEqual(sorted(unit["member_strands"]), ["c", "w"])

    def test_summarize_relaxed_trace_subregions_merges_main_block_reflex_pair(self):
        trace = {
            "w": [
                {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0},
            ],
            "c": [
                {"anchor_position": 291, "window_start": 82, "window_end": 291, "score": 17.5, "best_register": 0},
                {"anchor_position": 270, "window_start": 61, "window_end": 270, "score": 16.5, "best_register": 0},
                {"anchor_position": 249, "window_start": 40, "window_end": 249, "score": 15.5, "best_register": 0},
                {"anchor_position": 228, "window_start": 19, "window_end": 228, "score": 15.0, "best_register": 0},
                {"anchor_position": 207, "window_start": -2, "window_end": 207, "score": 14.5, "best_register": 0},
                {"anchor_position": 186, "window_start": -23, "window_end": 186, "score": 14.0, "best_register": 0},
                {"anchor_position": 165, "window_start": -44, "window_end": 165, "score": 13.8, "best_register": 0},
                {"anchor_position": 144, "window_start": -65, "window_end": 144, "score": 13.6, "best_register": 0},
                {"anchor_position": 123, "window_start": -86, "window_end": 123, "score": 13.4, "best_register": 0},
                {"anchor_position": 102, "window_start": -107, "window_end": 102, "score": 13.2, "best_register": 0},
            ],
        }

        summary = feature_assembly.summarize_relaxed_trace_subregions(trace, score_cutoff=12.5, phase=21)

        self.assertEqual(summary["Howell_additional_peak_count"], 0)
        self.assertEqual(summary["Howell_overlapping_alt_count"], 0)
        self.assertEqual(len(summary["overlapping_alt_groups"]), 0)
        self.assertIsNotNone(summary["main_biogenesis_unit"])
        self.assertTrue(summary["main_biogenesis_unit"]["main_partner_present"])
        self.assertEqual(sorted(summary["main_biogenesis_unit"]["member_strands"]), ["c", "w"])

    def test_summarize_relaxed_trace_subregions_finds_below_cutoff_cross_strand_partner_from_trace(self):
        trace = {
            "w": [
                {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0},
                {"anchor_position": 121, "window_start": 121, "window_end": 330, "score": 19.0, "best_register": 0},
                {"anchor_position": 142, "window_start": 142, "window_end": 351, "score": 18.0, "best_register": 0},
            ],
            "c": [
                {"anchor_position": 291, "window_start": 82, "window_end": 291, "score": 10.5, "best_register": 0},
                {"anchor_position": 270, "window_start": 61, "window_end": 270, "score": 10.1, "best_register": 0},
                {"anchor_position": 249, "window_start": 40, "window_end": 249, "score": 9.8, "best_register": 0},
                {"anchor_position": 228, "window_start": 19, "window_end": 228, "score": 9.5, "best_register": 0},
                {"anchor_position": 207, "window_start": -2, "window_end": 207, "score": 9.3, "best_register": 0},
                {"anchor_position": 186, "window_start": -23, "window_end": 186, "score": 9.0, "best_register": 0},
                {"anchor_position": 165, "window_start": -44, "window_end": 165, "score": 8.8, "best_register": 0},
                {"anchor_position": 144, "window_start": -65, "window_end": 144, "score": 8.5, "best_register": 0},
                {"anchor_position": 123, "window_start": -86, "window_end": 123, "score": 8.2, "best_register": 0},
                {"anchor_position": 102, "window_start": -107, "window_end": 102, "score": 8.0, "best_register": 0},
            ],
        }

        summary = feature_assembly.summarize_relaxed_trace_subregions(trace, score_cutoff=12.5, phase=21)

        self.assertIsNotNone(summary["main_biogenesis_unit"])
        self.assertTrue(summary["main_biogenesis_unit"]["main_partner_present"])
        self.assertEqual(sorted(summary["main_biogenesis_unit"]["member_strands"]), ["c", "w"])
        self.assertEqual(summary["Howell_overlapping_alt_count"], 0)

    def test_summarize_relaxed_trace_subregions_rejects_sparse_cross_strand_bridge(self):
        trace = {
            "w": [
                {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0},
                {"anchor_position": 121, "window_start": 121, "window_end": 330, "score": 19.0, "best_register": 0},
                {"anchor_position": 142, "window_start": 142, "window_end": 351, "score": 18.0, "best_register": 0},
            ],
            "c": [
                {"anchor_position": 291, "window_start": 82, "window_end": 291, "score": 10.5, "best_register": 0},
                {"anchor_position": 270, "window_start": 61, "window_end": 270, "score": 10.1, "best_register": 0},
            ],
        }

        summary = feature_assembly.summarize_relaxed_trace_subregions(trace, score_cutoff=12.5, phase=21)

        self.assertIsNotNone(summary["main_biogenesis_unit"])
        self.assertFalse(summary["main_biogenesis_unit"]["main_partner_present"])

    def test_compute_phase_shift_nt_uses_signed_minimal_shift(self):
        self.assertEqual(feature_assembly._compute_phase_shift_nt(100, 120, 21), -1)
        self.assertEqual(feature_assembly._compute_phase_shift_nt(100, 112, 24), 12)

    def test_summarize_relaxed_trace_subregions_requires_support_before_promoting_overlap(self):
        trace = {
            "w": [
                {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0},
                {"anchor_position": 112, "window_start": 112, "window_end": 321, "score": 18.9, "best_register": 0},
            ],
            "c": [],
        }

        summary = feature_assembly.summarize_relaxed_trace_subregions(trace, score_cutoff=12.5, phase=21)

        self.assertEqual(summary["Howell_overlapping_alt_count"], 0)
        self.assertEqual(len(summary["promoted_secondary_units"]), 0)
        self.assertEqual(len(summary["unpromoted_secondary_units"]), 1)

    def test_summarize_peak_howell_ambiguity_separates_extension_and_origin_frames(self):
        winner = {
            "strand": "w",
            "window_start": 100,
            "window_end": 309,
            "score": 10.0,
            "best_register": 0,
            "register_scores": [10.0, 9.2, 4.5],
            "exact_score": 8.0,
            "exact_best_register": 0,
            "exact_register_scores": [8.0, 7.3, 3.0],
            "exact_frame": 16,
        }
        candidates = [
            dict(winner),
            {"strand": "w", "window_start": 120, "window_end": 329, "score": 9.8, "best_register": 2, "exact_score": 7.4, "exact_best_register": 2, "exact_frame": 16},
            {"strand": "w", "window_start": 125, "window_end": 334, "score": 9.7, "best_register": 4, "exact_score": 7.3, "exact_best_register": 4, "exact_frame": 4},
            {"strand": "w", "window_start": 130, "window_end": 339, "score": 9.6, "best_register": 5, "exact_score": 7.2, "exact_best_register": 5, "exact_frame": 7},
            {"strand": "w", "window_start": 400, "window_end": 609, "score": 9.8, "best_register": 7, "exact_score": 7.8, "exact_best_register": 7, "exact_frame": 16},
            {"strand": "c", "window_start": 120, "window_end": 329, "score": 9.7, "best_register": 1, "exact_score": 7.6, "exact_best_register": 1, "exact_frame": 6},
        ]

        summary = feature_assembly._summarize_peak_howell_ambiguity(winner, candidates)

        self.assertEqual(summary["Howell_ambiguity_count"], 3)
        self.assertEqual(summary["Howell_alt_register_count"], 1)
        self.assertAlmostEqual(summary["Howell_exact_support_score"], 8.0, places=6)
        self.assertAlmostEqual(summary["Howell_overlap_margin"], 0.6, places=6)
        self.assertAlmostEqual(summary["best_overlapping_competitor_score"], 7.4, places=6)
        self.assertEqual(summary["Howell_extension_window_count"], 1)
        self.assertEqual(summary["Howell_extension_span_nt"], 230)
        self.assertEqual(summary["Howell_origin_window_count"], 2)
        self.assertEqual(summary["Howell_origin_frame_count"], 2)
        self.assertAlmostEqual(summary["Howell_origin_margin"], 0.7, places=6)
        self.assertEqual(summary["Howell_origin_class"], "mixed_extension_and_ambiguity")

    def test_summarize_peak_howell_ambiguity_marks_same_frame_plateau_as_coherent_extension(self):
        winner = {
            "strand": "w",
            "window_start": 500,
            "window_end": 709,
            "score": 12.0,
            "best_register": 0,
            "register_scores": [12.0, 4.0],
            "exact_score": 10.0,
            "exact_best_register": 0,
            "exact_register_scores": [10.0, 3.0],
            "exact_frame": 17,
        }
        candidates = [
            dict(winner),
            {"strand": "w", "window_start": 520, "window_end": 729, "score": 11.5, "best_register": 1, "exact_score": 9.8, "exact_best_register": 1, "exact_frame": 17},
            {"strand": "w", "window_start": 540, "window_end": 749, "score": 11.3, "best_register": 2, "exact_score": 9.4, "exact_best_register": 2, "exact_frame": 17},
        ]

        summary = feature_assembly._summarize_peak_howell_ambiguity(winner, candidates)

        self.assertEqual(summary["Howell_extension_window_count"], 2)
        self.assertEqual(summary["Howell_extension_span_nt"], 250)
        self.assertEqual(summary["Howell_origin_window_count"], 0)
        self.assertEqual(summary["Howell_origin_frame_count"], 0)
        self.assertTrue(np.isnan(summary["Howell_origin_margin"]))
        self.assertEqual(summary["Howell_origin_class"], "coherent_extension")

    def test_compute_phasing_score_howell_uses_exact_only_ambiguity_for_adjacent_relaxed_ties(self):
        aclust = pd.DataFrame(
            [
                {"pos": 100, "len": 21, "abun": 12, "strand": "w"},
                {"pos": 121, "len": 21, "abun": 11, "strand": "w"},
                {"pos": 142, "len": 21, "abun": 13, "strand": "w"},
                {"pos": 163, "len": 21, "abun": 10, "strand": "w"},
            ]
        )

        (
            w_score,
            (_w_start, _w_end),
            c_score,
            (_c_start, _c_end),
            detail,
        ) = feature_assembly.compute_phasing_score_Howell(aclust, return_detail=True)

        self.assertGreater(float(w_score), 0.0)
        self.assertIsNone(c_score)
        self.assertIsNotNone(detail)
        self.assertGreater(float(detail["Howell_exact_support_score"]), 0.0)
        self.assertEqual(detail["Howell_ambiguity_count"], 0)
        self.assertEqual(detail["Howell_alt_register_count"], 0)
        self.assertTrue(np.isnan(detail["Howell_overlap_margin"]))
        self.assertEqual(detail["Howell_extension_window_count"], 0)
        self.assertEqual(detail["Howell_extension_span_nt"], 210)
        self.assertEqual(detail["Howell_origin_window_count"], 0)
        self.assertEqual(detail["Howell_origin_frame_count"], 0)
        self.assertTrue(np.isnan(detail["Howell_origin_margin"]))
        self.assertEqual(detail["Howell_origin_class"], "unique_origin")

    def test_compute_phasing_score_howell_marks_exact_only_ambiguity_unassessable_for_rescue_driven_case(self):
        aclust = pd.DataFrame(
            [
                {"pos": 99, "len": 21, "abun": 12, "strand": "w"},
                {"pos": 122, "len": 21, "abun": 11, "strand": "w"},
                {"pos": 141, "len": 21, "abun": 13, "strand": "w"},
                {"pos": 164, "len": 21, "abun": 10, "strand": "w"},
            ]
        )

        (
            w_score,
            (_w_start, _w_end),
            _c_score,
            (_c_start, _c_end),
            detail,
        ) = feature_assembly.compute_phasing_score_Howell(aclust, return_detail=True)

        self.assertGreater(float(w_score), 0.0)
        self.assertIsNotNone(detail)
        self.assertEqual(float(detail["Howell_exact_support_score"]), 0.0)
        self.assertTrue(np.isnan(detail["Howell_ambiguity_count"]))
        self.assertTrue(np.isnan(detail["Howell_alt_register_count"]))
        self.assertTrue(np.isnan(detail["Howell_overlap_margin"]))
        self.assertTrue(np.isnan(detail["Howell_extension_window_count"]))
        self.assertTrue(np.isnan(detail["Howell_extension_span_nt"]))
        self.assertTrue(np.isnan(detail["Howell_origin_window_count"]))
        self.assertTrue(np.isnan(detail["Howell_origin_frame_count"]))
        self.assertTrue(np.isnan(detail["Howell_origin_margin"]))
        self.assertEqual(detail["Howell_origin_class"], "insufficient_exact_support")


class HowellAmbiguityPlotTests(unittest.TestCase):
    def test_build_ambiguity_sidebar_payload_formats_values(self):
        payload = locus_plots._build_ambiguity_sidebar_payload(
            {
                "Howell_exact_support_score": 5.2,
                "Howell_ambiguity_count": 3,
                "Howell_alt_register_count": 1,
                "Howell_overlap_margin": 1.234,
                "Howell_extension_window_count": 2,
                "Howell_extension_span_nt": 288,
                "Howell_origin_window_count": 1,
                "Howell_origin_frame_count": 1,
                "Howell_origin_margin": -0.456,
                "Howell_origin_class": "mixed_extension_and_ambiguity",
                "Howell_additional_peak_count": 2,
                "Howell_additional_peak_best_score": 14.2,
            }
        )

        self.assertEqual(payload["exact_support"], "5.20")
        self.assertEqual(payload["origin_class"], "Mixed extension + ambiguity")
        self.assertEqual(payload["extension_window_count"], "2")
        self.assertEqual(payload["extension_span_nt"], "288")
        self.assertEqual(payload["origin_window_count"], "1")
        self.assertEqual(payload["origin_frame_count"], "1")
        self.assertEqual(payload["origin_margin"], "-0.46")
        self.assertEqual(payload["additional_peak_count"], "2")
        self.assertEqual(payload["additional_peak_best_score"], "14.20")
        self.assertEqual(payload["raw_overlap_count"], "3")
        self.assertEqual(payload["raw_alt_register_count"], "1")
        self.assertEqual(payload["raw_overlap_margin"], "1.23")

    def test_analyze_single_locus_preserves_precomputed_ambiguity_metrics(self):
        result = locus_plots._analyze_single_locus(
            {
                "plot_path": "/tmp/test_plot.png",
                "title_text": "libA | chr1:100..163 | 21-PHAS",
                "identifier_text": "chr1:100..163",
                "cid_value": "cluster_1",
                "alib_value": "libA",
                "phase": 21,
                "Howell_exact_support_score": 6.5,
                "Howell_ambiguity_count": 4,
                "Howell_alt_register_count": 2,
                "Howell_overlap_margin": 0.75,
                "Howell_extension_window_count": 3,
                "Howell_extension_span_nt": 252,
                "Howell_origin_window_count": 1,
                "Howell_origin_frame_count": 1,
                "Howell_origin_margin": -0.5,
                "Howell_origin_class": "mixed_extension_and_ambiguity",
                "Howell_additional_peak_count": 2,
                "Howell_additional_peak_best_score": 13.2,
                "cluster_rows": [
                    {"pos": 100, "abun": 12, "len": 21, "strand": "w", "tag_seq": "A1", "hits": 1},
                    {"pos": 121, "abun": 11, "len": 21, "strand": "w", "tag_seq": "A2", "hits": 1},
                    {"pos": 142, "abun": 13, "len": 21, "strand": "w", "tag_seq": "A3", "hits": 1},
                    {"pos": 163, "abun": 10, "len": 21, "strand": "w", "tag_seq": "A4", "hits": 1},
                ],
            }
        )

        payload = result["plot_payload"]
        self.assertAlmostEqual(payload["Howell_exact_support_score"], 6.5, places=6)
        self.assertEqual(payload["Howell_ambiguity_count"], 4)
        self.assertEqual(payload["Howell_alt_register_count"], 2)
        self.assertAlmostEqual(payload["Howell_overlap_margin"], 0.75, places=6)
        self.assertEqual(payload["Howell_extension_window_count"], 3)
        self.assertEqual(payload["Howell_extension_span_nt"], 252)
        self.assertEqual(payload["Howell_origin_window_count"], 1)
        self.assertEqual(payload["Howell_origin_frame_count"], 1)
        self.assertAlmostEqual(payload["Howell_origin_margin"], -0.5, places=6)
        self.assertEqual(payload["Howell_origin_class"], "mixed_extension_and_ambiguity")
        self.assertEqual(payload["Howell_additional_peak_count"], 2)
        self.assertAlmostEqual(payload["Howell_additional_peak_best_score"], 13.2, places=6)

    def test_write_single_locus_plot_renders_detachable_metrics_strip(self):
        result = locus_plots._analyze_single_locus(
            {
                "plot_path": "/tmp/test_plot.png",
                "title_text": "libA | chr1:100..163 | 21-PHAS",
                "identifier_text": "chr1:100..163",
                "cid_value": "cluster_1",
                "alib_value": "libA",
                "phase": 21,
                "Howell_exact_support_score": 6.5,
                "Howell_ambiguity_count": 4,
                "Howell_alt_register_count": 2,
                "Howell_overlap_margin": 0.75,
                "Howell_extension_window_count": 3,
                "Howell_extension_span_nt": 252,
                "Howell_origin_window_count": 1,
                "Howell_origin_frame_count": 1,
                "Howell_origin_margin": -0.5,
                "Howell_origin_class": "mixed_extension_and_ambiguity",
                "Howell_additional_peak_count": 2,
                "Howell_additional_peak_best_score": 13.2,
                "cluster_rows": [
                    {"pos": 100, "abun": 12, "len": 21, "strand": "w", "tag_seq": "A1", "hits": 1},
                    {"pos": 121, "abun": 11, "len": 21, "strand": "w", "tag_seq": "A2", "hits": 1},
                    {"pos": 142, "abun": 13, "len": 21, "strand": "w", "tag_seq": "A3", "hits": 1},
                    {"pos": 163, "abun": 10, "len": 21, "strand": "w", "tag_seq": "A4", "hits": 1},
                ],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = os.path.join(tmpdir, "locus.png")
            payload = dict(result["plot_payload"])
            payload["plot_path"] = plot_path
            locus_plots._write_single_locus_plot(payload)
            self.assertTrue(os.path.isfile(plot_path))
            self.assertGreater(os.path.getsize(plot_path), 0)


class HowellAmbiguityOutputTests(unittest.TestCase):
    def test_finalize_and_write_results_persists_ambiguity_columns(self):
        features = pd.DataFrame(
            [
                {
                    "identifier": "chr1:100..400",
                    "cID": "cluster_1",
                    "alib": "libA",
                    "complexity": 0.05,
                    "strand_bias": 0.8,
                    "log_clust_len_norm_counts": 1.2,
                    "ratio_abund_len_phase": 5.5,
                    "phasis_score": 300.0,
                    "combined_fishers": 1e-8,
                    "total_abund": 500.0,
                    "w_Howell_score": 12.3,
                    "w_window_start": 100,
                    "w_window_end": 309,
                    "c_Howell_score": 0.0,
                    "c_window_start": np.nan,
                    "c_window_end": np.nan,
                    "Peak_Howell_score": 12.3,
                    "Howell_exact_support_score": 8.1,
                    "Howell_ambiguity_count": 2,
                    "Howell_alt_register_count": 1,
                    "Howell_overlap_margin": 0.9,
                    "Howell_extension_window_count": 1,
                    "Howell_extension_span_nt": 230,
                    "Howell_origin_window_count": 1,
                    "Howell_origin_frame_count": 1,
                    "Howell_origin_margin": 0.4,
                    "Howell_origin_class": "mixed_extension_and_ambiguity",
                    "Howell_additional_peak_count": 2,
                    "Howell_additional_peak_best_score": 15.5,
                    "Howell_overlapping_alt_count": 1,
                    "Howell_overlapping_alt_best_score": 14.2,
                    "Howell_overlapping_alt_best_shift_nt": 12.0,
                    "Howell_exact_relaxed_ratio": 0.6585,
                    "Howell_strict_relaxed_ratio": 0.8211,
                    "w_Howell_score_strict": 10.1,
                    "w_window_start_strict": 100,
                    "w_window_end_strict": 309,
                    "c_Howell_score_strict": 0.0,
                    "c_window_start_strict": np.nan,
                    "c_window_end_strict": np.nan,
                    "Peak_Howell_score_strict": 10.1,
                    "label": "PHAS",
                },
                {
                    "identifier": "chr2:500..700",
                    "cID": "cluster_2",
                    "alib": "libB",
                    "complexity": 0.6,
                    "strand_bias": 0.5,
                    "log_clust_len_norm_counts": 0.2,
                    "ratio_abund_len_phase": 1.2,
                    "phasis_score": 10.0,
                    "combined_fishers": 0.2,
                    "total_abund": 30.0,
                    "w_Howell_score": 0.0,
                    "w_window_start": np.nan,
                    "w_window_end": np.nan,
                    "c_Howell_score": 0.0,
                    "c_window_start": np.nan,
                    "c_window_end": np.nan,
                    "Peak_Howell_score": 0.0,
                    "Howell_exact_support_score": 0.0,
                    "Howell_ambiguity_count": np.nan,
                    "Howell_alt_register_count": np.nan,
                    "Howell_overlap_margin": np.nan,
                    "Howell_extension_window_count": np.nan,
                    "Howell_extension_span_nt": np.nan,
                    "Howell_origin_window_count": np.nan,
                    "Howell_origin_frame_count": np.nan,
                    "Howell_origin_margin": np.nan,
                    "Howell_origin_class": "insufficient_exact_support",
                    "Howell_additional_peak_count": 0,
                    "Howell_additional_peak_best_score": np.nan,
                    "Howell_overlapping_alt_count": 0,
                    "Howell_overlapping_alt_best_score": np.nan,
                    "Howell_overlapping_alt_best_shift_nt": np.nan,
                    "Howell_exact_relaxed_ratio": np.nan,
                    "Howell_strict_relaxed_ratio": np.nan,
                    "w_Howell_score_strict": 0.0,
                    "w_window_start_strict": np.nan,
                    "w_window_end_strict": np.nan,
                    "c_Howell_score_strict": 0.0,
                    "c_window_start_strict": np.nan,
                    "c_window_end_strict": np.nan,
                    "Peak_Howell_score_strict": 0.0,
                    "label": "non-PHAS",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as outdir:
            with mock.patch.object(output, "make_pool", return_value=_DummyPool()):
                with mock.patch.object(output, "plot_report_heat_map", return_value=None):
                    with mock.patch.object(output, "plot_phasAbundance_heat_map", return_value=None):
                        with mock.patch.object(output, "plot_totalAbundance_heat_map", return_value=None):
                            with mock.patch.object(output, "plot_howell_score_heat_maps", return_value=None):
                                output.finalize_and_write_results(
                                    "KNN",
                                    features,
                                    job_outdir=outdir,
                                    job_phase=21,
                                )

            all_path = os.path.join(outdir, "21_KNN_all_clusters.tsv")
            calls_path = os.path.join(outdir, "21_KNN_calls.tsv")
            self.assertTrue(os.path.isfile(all_path))
            self.assertTrue(os.path.isfile(calls_path))

            all_df = pd.read_csv(all_path, sep="\t")
            calls_df = pd.read_csv(calls_path, sep="\t")

            for column in (
                "Howell_exact_support_score",
                "Howell_ambiguity_count",
                "Howell_alt_register_count",
                "Howell_overlap_margin",
                "Howell_extension_window_count",
                "Howell_extension_span_nt",
                "Howell_origin_window_count",
                "Howell_origin_frame_count",
                "Howell_origin_margin",
                "Howell_origin_class",
                "Howell_additional_peak_count",
                "Howell_additional_peak_best_score",
                "Howell_overlapping_alt_count",
                "Howell_overlapping_alt_best_score",
                "Howell_overlapping_alt_best_shift_nt",
            ):
                self.assertIn(column, all_df.columns)
                self.assertIn(column, calls_df.columns)

            self.assertEqual(len(calls_df), 1)
            self.assertAlmostEqual(float(calls_df.loc[0, "Howell_exact_support_score"]), 8.1, places=6)
            self.assertEqual(int(calls_df.loc[0, "Howell_ambiguity_count"]), 2)
            self.assertEqual(int(calls_df.loc[0, "Howell_alt_register_count"]), 1)
            self.assertAlmostEqual(float(calls_df.loc[0, "Howell_overlap_margin"]), 0.9, places=6)
            self.assertEqual(int(calls_df.loc[0, "Howell_extension_window_count"]), 1)
            self.assertEqual(int(calls_df.loc[0, "Howell_extension_span_nt"]), 230)
            self.assertEqual(int(calls_df.loc[0, "Howell_origin_window_count"]), 1)
            self.assertEqual(int(calls_df.loc[0, "Howell_origin_frame_count"]), 1)
            self.assertAlmostEqual(float(calls_df.loc[0, "Howell_origin_margin"]), 0.4, places=6)
            self.assertEqual(str(calls_df.loc[0, "Howell_origin_class"]), "mixed_extension_and_ambiguity")
            self.assertEqual(int(calls_df.loc[0, "Howell_additional_peak_count"]), 2)
            self.assertAlmostEqual(float(calls_df.loc[0, "Howell_additional_peak_best_score"]), 15.5, places=6)
            self.assertEqual(int(calls_df.loc[0, "Howell_overlapping_alt_count"]), 1)
            self.assertAlmostEqual(float(calls_df.loc[0, "Howell_overlapping_alt_best_score"]), 14.2, places=6)
            self.assertEqual(int(calls_df.loc[0, "Howell_overlapping_alt_best_shift_nt"]), 12)


if __name__ == "__main__":
    unittest.main()
