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
    def test_summarize_peak_howell_ambiguity_counts_only_same_strand_overlapping_near_ties(self):
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
        }
        candidates = [
            dict(winner),
            {"strand": "w", "window_start": 120, "window_end": 329, "score": 9.8, "best_register": 2, "exact_score": 7.4, "exact_best_register": 2},
            {"strand": "w", "window_start": 125, "window_end": 334, "score": 9.7, "best_register": 4, "exact_score": 7.19, "exact_best_register": 4},
            {"strand": "w", "window_start": 400, "window_end": 609, "score": 9.8, "best_register": 7, "exact_score": 7.8, "exact_best_register": 7},
            {"strand": "c", "window_start": 120, "window_end": 329, "score": 9.7, "best_register": 1, "exact_score": 7.6, "exact_best_register": 1},
        ]

        summary = feature_assembly._summarize_peak_howell_ambiguity(winner, candidates)

        self.assertEqual(summary["Howell_ambiguity_count"], 1)
        self.assertEqual(summary["Howell_alt_register_count"], 1)
        self.assertAlmostEqual(summary["Howell_exact_support_score"], 8.0, places=6)
        self.assertAlmostEqual(summary["Howell_overlap_margin"], 0.6, places=6)
        self.assertAlmostEqual(summary["best_overlapping_competitor_score"], 7.4, places=6)

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


class HowellAmbiguityPlotTests(unittest.TestCase):
    def test_build_ambiguity_sidebar_entries_formats_values(self):
        entries = locus_plots._build_ambiguity_sidebar_entries(
            {
                "Howell_exact_support_score": 5.2,
                "Howell_ambiguity_count": 3,
                "Howell_alt_register_count": 1,
                "Howell_overlap_margin": 1.234,
            }
        )

        self.assertEqual(
            entries,
            [
                ("Exact HPSP support", "5.20"),
                ("Near-tied windows", "3"),
                ("Alt. registers", "1"),
                ("Overlap margin", "1.23"),
            ],
        )

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
            ):
                self.assertIn(column, all_df.columns)
                self.assertIn(column, calls_df.columns)

            self.assertEqual(len(calls_df), 1)
            self.assertAlmostEqual(float(calls_df.loc[0, "Howell_exact_support_score"]), 8.1, places=6)
            self.assertEqual(int(calls_df.loc[0, "Howell_ambiguity_count"]), 2)
            self.assertEqual(int(calls_df.loc[0, "Howell_alt_register_count"]), 1)
            self.assertAlmostEqual(float(calls_df.loc[0, "Howell_overlap_margin"]), 0.9, places=6)


if __name__ == "__main__":
    unittest.main()
