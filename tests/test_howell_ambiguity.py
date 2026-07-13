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
        self.assertEqual(
            sorted(member.get("unit_role") for member in summary["main_biogenesis_unit"]["members"]),
            ["main_hpsp", "main_partner"],
        )

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
        self.assertEqual(
            sorted(member.get("unit_role") for member in summary["main_biogenesis_unit"]["members"]),
            ["main_hpsp", "main_partner"],
        )

    def test_summarize_relaxed_trace_subregions_finds_c_to_w_partner_with_zero_based_registers(self):
        trace = {
            "w": [
                {"anchor_position": 104, "window_start": 104, "window_end": 313, "score": 14.0, "best_register": 0},
                {"anchor_position": 125, "window_start": 125, "window_end": 334, "score": 14.2, "best_register": 0},
                {"anchor_position": 146, "window_start": 146, "window_end": 355, "score": 14.4, "best_register": 0},
                {"anchor_position": 167, "window_start": 167, "window_end": 376, "score": 14.6, "best_register": 0},
                {"anchor_position": 188, "window_start": 188, "window_end": 397, "score": 14.8, "best_register": 0},
                {"anchor_position": 209, "window_start": 209, "window_end": 418, "score": 15.0, "best_register": 0},
                {"anchor_position": 230, "window_start": 230, "window_end": 439, "score": 15.2, "best_register": 0},
                {"anchor_position": 251, "window_start": 251, "window_end": 460, "score": 15.4, "best_register": 0},
                {"anchor_position": 272, "window_start": 272, "window_end": 481, "score": 15.6, "best_register": 0},
                {"anchor_position": 293, "window_start": 293, "window_end": 502, "score": 15.8, "best_register": 0},
            ],
            "c": [
                {"anchor_position": 291, "window_start": 82, "window_end": 291, "score": 20.0, "best_register": 0},
                {"anchor_position": 270, "window_start": 61, "window_end": 270, "score": 19.0, "best_register": 0},
                {"anchor_position": 249, "window_start": 40, "window_end": 249, "score": 18.0, "best_register": 0},
                {"anchor_position": 228, "window_start": 19, "window_end": 228, "score": 17.0, "best_register": 0},
                {"anchor_position": 207, "window_start": -2, "window_end": 207, "score": 16.0, "best_register": 0},
                {"anchor_position": 186, "window_start": -23, "window_end": 186, "score": 15.0, "best_register": 0},
                {"anchor_position": 165, "window_start": -44, "window_end": 165, "score": 14.8, "best_register": 0},
                {"anchor_position": 144, "window_start": -65, "window_end": 144, "score": 14.6, "best_register": 0},
                {"anchor_position": 123, "window_start": -86, "window_end": 123, "score": 14.4, "best_register": 0},
                {"anchor_position": 102, "window_start": -107, "window_end": 102, "score": 14.2, "best_register": 0},
            ],
        }

        summary = feature_assembly.summarize_relaxed_trace_subregions(trace, score_cutoff=12.5, phase=21)

        self.assertIsNotNone(summary["main_biogenesis_unit"])
        self.assertTrue(summary["main_biogenesis_unit"]["main_partner_present"])
        self.assertEqual(sorted(summary["main_biogenesis_unit"]["member_strands"]), ["c", "w"])
        self.assertEqual(int(summary["main_biogenesis_unit"]["main_partner_shift_nt"]), 2)
        self.assertEqual(summary["Howell_overlapping_alt_count"], 0)
        self.assertEqual(
            sorted(member.get("unit_role") for member in summary["main_biogenesis_unit"]["members"]),
            ["main_hpsp", "main_partner"],
        )

    def test_summarize_relaxed_trace_subregions_uses_projected_partner_origin_when_peak_register_is_shifted(self):
        trace = {
            "w": [
                {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0},
                {"anchor_position": 121, "window_start": 121, "window_end": 330, "score": 19.0, "best_register": 0},
                {"anchor_position": 142, "window_start": 142, "window_end": 351, "score": 18.0, "best_register": 0},
                {"anchor_position": 163, "window_start": 163, "window_end": 372, "score": 17.0, "best_register": 0},
                {"anchor_position": 184, "window_start": 184, "window_end": 393, "score": 16.0, "best_register": 0},
            ],
            "c": [
                {"anchor_position": 291, "window_start": 82, "window_end": 291, "score": 18.0, "best_register": 2},
                {"anchor_position": 270, "window_start": 61, "window_end": 270, "score": 17.0, "best_register": 2},
                {"anchor_position": 249, "window_start": 40, "window_end": 249, "score": 16.0, "best_register": 2},
                {"anchor_position": 228, "window_start": 19, "window_end": 228, "score": 15.0, "best_register": 2},
                {"anchor_position": 207, "window_start": -2, "window_end": 207, "score": 14.0, "best_register": 2},
            ],
        }

        summary = feature_assembly.summarize_relaxed_trace_subregions(trace, score_cutoff=12.5, phase=21)

        self.assertIsNotNone(summary["main_biogenesis_unit"])
        self.assertTrue(summary["main_biogenesis_unit"]["main_partner_present"])
        self.assertEqual(int(summary["main_biogenesis_unit"]["main_partner_shift_nt"]), 2)
        self.assertEqual(
            sorted(member.get("unit_role") for member in summary["main_biogenesis_unit"]["members"]),
            ["main_hpsp", "main_partner"],
        )

    def test_summarize_relaxed_trace_subregions_absorbs_shifted_cross_strand_secondary_into_main_unit(self):
        trace = {
            "w": [
                {"anchor_position": 104, "window_start": 104, "window_end": 313, "score": 18.0, "best_register": 0},
                {"anchor_position": 125, "window_start": 125, "window_end": 334, "score": 17.0, "best_register": 0},
                {"anchor_position": 146, "window_start": 146, "window_end": 355, "score": 16.0, "best_register": 0},
                {"anchor_position": 167, "window_start": 167, "window_end": 376, "score": 15.0, "best_register": 0},
                {"anchor_position": 188, "window_start": 188, "window_end": 397, "score": 14.0, "best_register": 0},
                {"anchor_position": 209, "window_start": 209, "window_end": 418, "score": 13.5, "best_register": 0},
                {"anchor_position": 230, "window_start": 230, "window_end": 439, "score": 13.2, "best_register": 0},
                {"anchor_position": 251, "window_start": 251, "window_end": 460, "score": 13.0, "best_register": 0},
                {"anchor_position": 272, "window_start": 272, "window_end": 481, "score": 12.8, "best_register": 0},
                {"anchor_position": 293, "window_start": 293, "window_end": 502, "score": 12.6, "best_register": 0},
            ],
            "c": [
                {"anchor_position": 291, "window_start": 82, "window_end": 291, "score": 20.0, "best_register": 0},
                {"anchor_position": 270, "window_start": 61, "window_end": 270, "score": 19.0, "best_register": 0},
                {"anchor_position": 249, "window_start": 40, "window_end": 249, "score": 18.0, "best_register": 0},
                {"anchor_position": 228, "window_start": 19, "window_end": 228, "score": 17.0, "best_register": 0},
                {"anchor_position": 207, "window_start": -2, "window_end": 207, "score": 16.0, "best_register": 0},
                {"anchor_position": 186, "window_start": -23, "window_end": 186, "score": 15.0, "best_register": 0},
                {"anchor_position": 165, "window_start": -44, "window_end": 165, "score": 14.8, "best_register": 0},
                {"anchor_position": 144, "window_start": -65, "window_end": 144, "score": 14.6, "best_register": 0},
                {"anchor_position": 123, "window_start": -86, "window_end": 123, "score": 14.4, "best_register": 0},
                {"anchor_position": 102, "window_start": -107, "window_end": 102, "score": 14.2, "best_register": 0},
            ],
        }

        summary = feature_assembly.summarize_relaxed_trace_subregions(trace, score_cutoff=12.5, phase=21)

        self.assertIsNotNone(summary["main_biogenesis_unit"])
        self.assertTrue(summary["main_biogenesis_unit"]["main_partner_present"])
        self.assertEqual(summary["Howell_overlapping_alt_count"], 0)
        self.assertEqual(summary["Howell_additional_peak_count"], 0)
        self.assertEqual(
            sorted(member.get("unit_role") for member in summary["main_biogenesis_unit"]["members"]),
            ["main_hpsp", "main_partner"],
        )

    def test_best_main_unit_match_reanchors_cross_strand_candidate_to_row_origin(self):
        phase = 21
        winner_group = {
            "rows": [
                {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0},
                {"anchor_position": 121, "window_start": 121, "window_end": 330, "score": 19.0, "best_register": 0},
                {"anchor_position": 142, "window_start": 142, "window_end": 351, "score": 18.0, "best_register": 0},
                {"anchor_position": 163, "window_start": 163, "window_end": 372, "score": 17.0, "best_register": 0},
                {"anchor_position": 184, "window_start": 184, "window_end": 393, "score": 16.0, "best_register": 0},
            ],
            "min_start": 100,
            "max_end": 393,
        }
        winner_member = feature_assembly._build_relaxed_group_summary(
            winner_group,
            strand_code="w",
            category="main_hpsp",
            phase=phase,
            winner_register_origin=100,
        )

        candidate_group = {
            "rows": [
                {"anchor_position": 312, "window_start": 103, "window_end": 312, "score": 18.0, "best_register": 2},
                {"anchor_position": 291, "window_start": 82, "window_end": 291, "score": 17.5, "best_register": 0},
                {"anchor_position": 270, "window_start": 61, "window_end": 270, "score": 17.0, "best_register": 0},
                {"anchor_position": 249, "window_start": 40, "window_end": 249, "score": 16.5, "best_register": 0},
                {"anchor_position": 228, "window_start": 19, "window_end": 228, "score": 16.0, "best_register": 0},
            ],
            "min_start": 19,
            "max_end": 312,
        }
        candidate_member = feature_assembly._build_relaxed_group_summary(
            candidate_group,
            strand_code="c",
            category="trace_segment",
            phase=phase,
            winner_register_origin=winner_member.get("register_origin"),
        )

        trace_rows_by_strand = {
            "w": list(winner_group["rows"]),
            "c": list(candidate_group["rows"]),
        }
        best_member, match_detail = feature_assembly._best_main_unit_match_for_candidate(
            candidate_member,
            winner_member,
            [winner_member],
            trace_rows_by_strand,
            phase=phase,
        )

        self.assertIsNotNone(best_member)
        self.assertIsNotNone(match_detail)
        self.assertTrue(bool(match_detail.get("cross_strand")))
        self.assertEqual(int(match_detail.get("shift_nt", 0)), 2)
        self.assertEqual(int(best_member.get("register_origin")), 291)

    def test_evaluate_main_unit_candidate_match_reports_duplex_geometry_mismatch(self):
        phase = 21
        winner_member = {
            "strand": "w",
            "register_origin": 100,
            "relation_rows": [
                {"phase_relation": "exact", "expected_position": 100},
                {"phase_relation": "exact", "expected_position": 121},
                {"phase_relation": "exact", "expected_position": 142},
            ],
            "non_grey_row_count": 3,
            "exact_row_count": 3,
        }
        candidate_member = {
            "strand": "c",
            "register_origin": 286,
            "relation_rows": [
                {"phase_relation": "exact", "expected_position": 202},
                {"phase_relation": "exact", "expected_position": 181},
            ],
            "non_grey_row_count": 2,
            "exact_row_count": 2,
        }
        detail = feature_assembly._evaluate_main_unit_candidate_match(
            winner_member,
            candidate_member,
            [winner_member],
            {
                "w": [
                    {"anchor_position": 100, "window_start": 100, "window_end": 309, "best_register": 0},
                    {"anchor_position": 121, "window_start": 121, "window_end": 330, "best_register": 0},
                    {"anchor_position": 142, "window_start": 142, "window_end": 351, "best_register": 0},
                ],
                "c": [
                    {"anchor_position": 202, "window_start": -7, "window_end": 202, "best_register": 0},
                    {"anchor_position": 181, "window_start": -28, "window_end": 181, "best_register": 0},
                ],
            },
            phase=phase,
        )
        self.assertFalse(bool(detail.get("accepted")))
        self.assertEqual(str(detail.get("reject_reason")), "duplex_geometry_mismatch")

    def test_evaluate_main_unit_candidate_match_accepts_projected_cross_strand_geometry_rescue(self):
        phase = 21
        winner_member = {
            "strand": "w",
            "register_origin": 100,
            "relation_rows": [
                {"phase_relation": "exact", "expected_position": 100},
                {"phase_relation": "exact", "expected_position": 121},
                {"phase_relation": "exact", "expected_position": 142},
            ],
            "non_grey_row_count": 3,
            "exact_row_count": 3,
        }
        candidate_member = {
            "strand": "c",
            "register_origin": 98,
            "relation_rows": [
                {"phase_relation": "exact", "expected_position": 98},
                {"phase_relation": "exact", "expected_position": 77},
                {"phase_relation": "exact", "expected_position": 56},
            ],
            "non_grey_row_count": 3,
            "exact_row_count": 3,
        }
        detail = feature_assembly._evaluate_main_unit_candidate_match(
            winner_member,
            candidate_member,
            [winner_member],
            {
                "w": [
                    {"anchor_position": 100, "window_start": 100, "window_end": 309, "best_register": 0},
                    {"anchor_position": 121, "window_start": 121, "window_end": 330, "best_register": 0},
                    {"anchor_position": 142, "window_start": 142, "window_end": 351, "best_register": 0},
                ],
                "c": [
                    {"anchor_position": 98, "window_start": -111, "window_end": 98, "best_register": 0},
                    {"anchor_position": 77, "window_start": -132, "window_end": 77, "best_register": 0},
                    {"anchor_position": 56, "window_start": -153, "window_end": 56, "best_register": 0},
                ],
            },
            phase=phase,
        )
        self.assertTrue(bool(detail.get("accepted")))
        self.assertEqual(int(detail.get("shift_nt", 0)), -2)
        self.assertTrue(bool(detail.get("duplex_orientation_ok")))
        self.assertTrue(bool(detail.get("canonical_compatible")))
        self.assertEqual(str(detail.get("candidate_tier")), "canonical")

    def test_best_main_unit_match_prefers_canonical_partner_over_stronger_noncanonical(self):
        phase = 21
        winner_member = {
            "strand": "w",
            "register_origin": 100,
            "relation_rows": [
                {"phase_relation": "exact", "expected_position": 100},
                {"phase_relation": "exact", "expected_position": 121},
                {"phase_relation": "exact", "expected_position": 142},
            ],
            "non_grey_row_count": 3,
            "exact_row_count": 3,
            "peak_score": 12.0,
        }
        canonical_rows = [
            {"anchor_position": 98, "window_start": -111, "window_end": 98, "best_register": 0, "score": 12.0},
            {"anchor_position": 77, "window_start": -132, "window_end": 77, "best_register": 0, "score": 11.0},
            {"anchor_position": 56, "window_start": -153, "window_end": 56, "best_register": 0, "score": 10.0},
        ]
        fallback_rows = [
            {"anchor_position": 100, "window_start": -109, "window_end": 100, "best_register": 0, "score": 20.0},
            {"anchor_position": 79, "window_start": -130, "window_end": 79, "best_register": 0, "score": 19.0},
            {"anchor_position": 58, "window_start": -151, "window_end": 58, "best_register": 0, "score": 18.0},
            {"anchor_position": 37, "window_start": -172, "window_end": 37, "best_register": 0, "score": 17.0},
            {"anchor_position": 16, "window_start": -193, "window_end": 16, "best_register": 0, "score": 16.0},
        ]
        candidate = {
            "category": "trace_segment",
            "members": [
                {
                    "strand": "c",
                    "register_origin": 100,
                    "relation_rows": [
                        {"phase_relation": "exact", "expected_position": 100},
                        {"phase_relation": "exact", "expected_position": 79},
                        {"phase_relation": "exact", "expected_position": 58},
                        {"phase_relation": "exact", "expected_position": 37},
                        {"phase_relation": "exact", "expected_position": 16},
                    ],
                    "non_grey_row_count": 5,
                    "exact_row_count": 5,
                    "peak_score": 20.0,
                    "rows": fallback_rows,
                },
                {
                    "strand": "c",
                    "register_origin": 98,
                    "relation_rows": [
                        {"phase_relation": "exact", "expected_position": 98},
                        {"phase_relation": "exact", "expected_position": 77},
                        {"phase_relation": "exact", "expected_position": 56},
                    ],
                    "non_grey_row_count": 3,
                    "exact_row_count": 3,
                    "peak_score": 12.0,
                    "rows": canonical_rows,
                },
            ],
        }
        trace_rows_by_strand = {
            "w": [
                {"anchor_position": 100, "window_start": 100, "window_end": 309, "best_register": 0},
                {"anchor_position": 121, "window_start": 121, "window_end": 330, "best_register": 0},
                {"anchor_position": 142, "window_start": 142, "window_end": 351, "best_register": 0},
            ],
            "c": canonical_rows + fallback_rows,
        }

        best_member, detail = feature_assembly._best_main_unit_match_for_candidate(
            candidate,
            winner_member,
            [winner_member],
            trace_rows_by_strand,
            phase=phase,
        )

        self.assertIsNotNone(best_member)
        self.assertIsNotNone(detail)
        self.assertEqual(int(best_member.get("register_origin")), 98)
        self.assertTrue(bool(detail.get("canonical_compatible")))
        self.assertEqual(str(detail.get("candidate_tier")), "canonical")
        self.assertEqual(int(detail.get("shift_nt", 0)), -2)

    def test_best_main_unit_match_uses_noncanonical_fallback_only_when_canonical_absent(self):
        phase = 21
        winner_member = {
            "strand": "w",
            "register_origin": 100,
            "relation_rows": [
                {"phase_relation": "exact", "expected_position": 100},
                {"phase_relation": "exact", "expected_position": 121},
                {"phase_relation": "exact", "expected_position": 142},
            ],
            "non_grey_row_count": 3,
            "exact_row_count": 3,
            "peak_score": 12.0,
        }
        fallback_rows = [
            {"anchor_position": 100, "window_start": -109, "window_end": 100, "best_register": 0, "score": 20.0},
            {"anchor_position": 79, "window_start": -130, "window_end": 79, "best_register": 0, "score": 19.0},
            {"anchor_position": 58, "window_start": -151, "window_end": 58, "best_register": 0, "score": 18.0},
        ]
        candidate = {
            "category": "trace_segment",
            "members": [
                {
                    "strand": "c",
                    "register_origin": 100,
                    "relation_rows": [
                        {"phase_relation": "exact", "expected_position": 100},
                        {"phase_relation": "exact", "expected_position": 79},
                        {"phase_relation": "exact", "expected_position": 58},
                    ],
                    "non_grey_row_count": 3,
                    "exact_row_count": 3,
                    "peak_score": 20.0,
                    "rows": fallback_rows,
                },
            ],
        }
        trace_rows_by_strand = {
            "w": [
                {"anchor_position": 100, "window_start": 100, "window_end": 309, "best_register": 0},
                {"anchor_position": 121, "window_start": 121, "window_end": 330, "best_register": 0},
                {"anchor_position": 142, "window_start": 142, "window_end": 351, "best_register": 0},
            ],
            "c": fallback_rows,
        }

        best_member, detail = feature_assembly._best_main_unit_match_for_candidate(
            candidate,
            winner_member,
            [winner_member],
            trace_rows_by_strand,
            phase=phase,
        )

        self.assertIsNotNone(best_member)
        self.assertIsNotNone(detail)
        self.assertEqual(int(best_member.get("register_origin")), 100)
        self.assertFalse(bool(detail.get("canonical_compatible")))
        self.assertEqual(str(detail.get("candidate_tier")), "fallback_noncanonical")
        self.assertEqual(int(detail.get("shift_nt", 0)), 0)

    def test_select_main_biogenesis_partner_rejects_noncanonical_fallback_partner(self):
        phase = 21
        winner_member = {
            "strand": "w",
            "register_origin": 100,
            "relation_rows": [
                {"phase_relation": "exact", "expected_position": 100},
                {"phase_relation": "exact", "expected_position": 121},
                {"phase_relation": "exact", "expected_position": 142},
            ],
            "non_grey_row_count": 3,
            "exact_row_count": 3,
            "peak_score": 12.0,
            "rows": [
                {"anchor_position": 100, "window_start": 100, "window_end": 309, "best_register": 0, "score": 12.0},
                {"anchor_position": 121, "window_start": 121, "window_end": 330, "best_register": 0, "score": 11.0},
                {"anchor_position": 142, "window_start": 142, "window_end": 351, "best_register": 0, "score": 10.0},
            ],
        }
        fallback_rows = [
            {"anchor_position": 100, "window_start": -109, "window_end": 100, "best_register": 0, "score": 20.0},
            {"anchor_position": 79, "window_start": -130, "window_end": 79, "best_register": 0, "score": 19.0},
            {"anchor_position": 58, "window_start": -151, "window_end": 58, "best_register": 0, "score": 18.0},
        ]
        candidate = {
            "category": "trace_segment",
            "members": [
                {
                    "strand": "c",
                    "register_origin": 100,
                    "relation_rows": [
                        {"phase_relation": "exact", "expected_position": 100},
                        {"phase_relation": "exact", "expected_position": 79},
                        {"phase_relation": "exact", "expected_position": 58},
                    ],
                    "non_grey_row_count": 3,
                    "exact_row_count": 3,
                    "peak_score": 20.0,
                    "rows": fallback_rows,
                },
            ],
        }
        trace_rows_by_strand = {
            "w": list(winner_member["rows"]),
            "c": fallback_rows,
        }

        partner_index, partner_member, partner_detail = feature_assembly._select_main_biogenesis_partner(
            winner_member,
            [candidate],
            trace_rows_by_strand,
            phase=phase,
        )

        self.assertIsNone(partner_index)
        self.assertIsNone(partner_member)
        self.assertIsNone(partner_detail)

    def test_evaluate_main_unit_candidate_match_reports_bridge_threshold_reason(self):
        phase = 21
        winner_group = {
            "rows": [
                {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0},
                {"anchor_position": 121, "window_start": 121, "window_end": 330, "score": 19.0, "best_register": 0},
                {"anchor_position": 142, "window_start": 142, "window_end": 351, "score": 18.0, "best_register": 0},
            ],
            "min_start": 100,
            "max_end": 351,
        }
        winner_member = feature_assembly._build_relaxed_group_summary(
            winner_group,
            strand_code="w",
            category="main_hpsp",
            phase=phase,
            winner_register_origin=100,
        )
        candidate_group = {
            "rows": [
                {"anchor_position": 291, "window_start": 82, "window_end": 291, "score": 10.5, "best_register": 0},
                {"anchor_position": 270, "window_start": 61, "window_end": 270, "score": 10.1, "best_register": 0},
            ],
            "min_start": 61,
            "max_end": 291,
        }
        candidate_member = feature_assembly._build_relaxed_group_summary(
            candidate_group,
            strand_code="c",
            category="trace_segment",
            phase=phase,
            winner_register_origin=winner_member.get("register_origin"),
        )
        detail = feature_assembly._evaluate_main_unit_candidate_match(
            winner_member,
            candidate_member,
            [winner_member],
            {"w": list(winner_group["rows"]), "c": list(candidate_group["rows"])},
            phase=phase,
        )
        self.assertFalse(bool(detail.get("accepted")))
        self.assertIn(
            str(detail.get("reject_reason")),
            {"support_ratio_below_threshold", "unsupported_run_too_long"},
        )

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

    def test_evaluate_main_unit_bridge_trims_terminal_cross_strand_flanks(self):
        phase = 21
        winner_member = {
            "strand": "w",
            "register_origin": 100,
            "relation_rows": [
                {"expected_position": position, "phase_relation": "exact"}
                for position in (100, 121, 142, 163, 184, 205, 226, 247, 268)
            ],
        }
        candidate_member = {
            "strand": "c",
            "register_origin": 287,
            "relation_rows": [
                {"expected_position": position, "phase_relation": "exact"}
                for position in (245, 224, 203)
            ],
        }
        trace_rows_by_strand = {
            "w": [],
            "c": [
                {"anchor_position": position}
                for position in (245, 224, 203)
            ],
        }

        legacy_like = feature_assembly._evaluate_main_unit_bridge(
            candidate_member,
            [winner_member],
            trace_rows_by_strand,
            phase=phase,
            cross_strand=False,
        )
        self.assertFalse(bool(legacy_like.get("passes")))
        self.assertEqual(str(legacy_like.get("reject_reason")), "unsupported_run_too_long")

        refined = feature_assembly._evaluate_main_unit_bridge(
            candidate_member,
            [winner_member],
            trace_rows_by_strand,
            phase=phase,
            cross_strand=True,
        )
        self.assertTrue(bool(refined.get("passes")))
        self.assertEqual(int(refined.get("max_zero_run", -1)), 0)

    def test_summarize_relaxed_trace_subregions_accepts_projected_cross_strand_geometry_rescue(self):
        trace = {
            "w": [
                {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0},
                {"anchor_position": 121, "window_start": 121, "window_end": 330, "score": 19.0, "best_register": 0},
                {"anchor_position": 142, "window_start": 142, "window_end": 351, "score": 18.0, "best_register": 0},
            ],
            "c": [
                {"anchor_position": 287, "window_start": 78, "window_end": 287, "score": 16.0, "best_register": 0},
                {"anchor_position": 266, "window_start": 57, "window_end": 266, "score": 15.5, "best_register": 0},
                {"anchor_position": 245, "window_start": 36, "window_end": 245, "score": 15.0, "best_register": 0},
                {"anchor_position": 224, "window_start": 15, "window_end": 224, "score": 14.5, "best_register": 0},
                {"anchor_position": 203, "window_start": -6, "window_end": 203, "score": 14.0, "best_register": 0},
                {"anchor_position": 182, "window_start": -27, "window_end": 182, "score": 13.8, "best_register": 0},
                {"anchor_position": 161, "window_start": -48, "window_end": 161, "score": 13.6, "best_register": 0},
                {"anchor_position": 140, "window_start": -69, "window_end": 140, "score": 13.4, "best_register": 0},
                {"anchor_position": 119, "window_start": -90, "window_end": 119, "score": 13.2, "best_register": 0},
                {"anchor_position": 98, "window_start": -111, "window_end": 98, "score": 13.0, "best_register": 0},
            ],
        }

        summary = feature_assembly.summarize_relaxed_trace_subregions(trace, score_cutoff=12.5, phase=21)

        self.assertIsNotNone(summary["main_biogenesis_unit"])
        self.assertTrue(summary["main_biogenesis_unit"]["main_partner_present"])
        self.assertEqual(
            sorted(member.get("unit_role") for member in summary["main_biogenesis_unit"]["members"]),
            ["main_hpsp", "main_partner"],
        )

    def test_summarize_relaxed_trace_subregions_counts_paired_overlapping_secondary_once(self):
        trace = {
            "w": [
                {"anchor_position": 100, "window_start": 100, "window_end": 339, "score": 20.0, "best_register": 0},
                {"anchor_position": 112, "window_start": 112, "window_end": 351, "score": 18.0, "best_register": 0},
                {"anchor_position": 136, "window_start": 136, "window_end": 375, "score": 17.4, "best_register": 0},
            ],
            "c": [
                {"anchor_position": 330, "window_start": 91, "window_end": 330, "score": 17.5, "best_register": 0},
                {"anchor_position": 306, "window_start": 67, "window_end": 306, "score": 16.8, "best_register": 0},
            ],
        }

        summary = feature_assembly.summarize_relaxed_trace_subregions(trace, score_cutoff=12.5, phase=24)

        self.assertEqual(summary["Howell_overlapping_alt_count"], 1)
        self.assertEqual(len(summary["overlapping_alt_groups"]), 1)
        self.assertTrue(bool(summary["overlapping_alt_groups"][0].get("paired_unit")))

    def test_summarize_relaxed_trace_subregions_does_not_emit_diverted_secondary_trace_reason(self):
        trace = {
            "w": [
                {"anchor_position": 100, "window_start": 100, "window_end": 339, "score": 20.0, "best_register": 0},
                {"anchor_position": 112, "window_start": 112, "window_end": 351, "score": 18.0, "best_register": 0},
                {"anchor_position": 136, "window_start": 136, "window_end": 375, "score": 17.4, "best_register": 0},
            ],
            "c": [
                {"anchor_position": 330, "window_start": 91, "window_end": 330, "score": 17.5, "best_register": 0},
                {"anchor_position": 306, "window_start": 67, "window_end": 306, "score": 16.8, "best_register": 0},
            ],
        }

        summary = feature_assembly.summarize_relaxed_trace_subregions(
            trace,
            score_cutoff=12.5,
            phase=24,
            debug_context={"identifier": "x:1..2", "alib": "lib"},
        )

        trace_rows = pd.DataFrame(summary["main_partner_debug_rows"])
        diverted = trace_rows[trace_rows["first_reject_reason"].astype(str) == "accepted_then_diverted_to_secondary"]
        self.assertTrue(diverted.empty)

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

    def test_summarize_relaxed_trace_subregions_allows_promoted_overlap_without_shift(self):
        trace = {
            "w": [
                {"anchor_position": 100, "window_start": 100, "window_end": 309, "score": 20.0, "best_register": 0},
            ],
            "c": [],
        }
        overlap_unit = {
            "category": "overlapping_alternative",
            "peak_score": 18.0,
            "shift_nt": None,
        }

        with (
            mock.patch.object(
                feature_assembly,
                "_main_biogenesis_unit",
                return_value=({"category": "main_hpsp"}, [dict(overlap_unit)]),
            ),
            mock.patch.object(
                feature_assembly,
                "_annotate_secondary_units_against_main_unit",
                return_value=[dict(overlap_unit)],
            ),
            mock.patch.object(
                feature_assembly,
                "_annotate_promoted_secondary_units",
                return_value=([dict(overlap_unit)], []),
            ),
        ):
            summary = feature_assembly.summarize_relaxed_trace_subregions(
                trace,
                score_cutoff=12.5,
                phase=21,
            )

        self.assertEqual(summary["Howell_overlapping_alt_count"], 1)
        self.assertAlmostEqual(summary["Howell_overlapping_alt_best_score"], 18.0)
        self.assertTrue(np.isnan(summary["Howell_overlapping_alt_best_shift_nt"]))

    def test_secondary_unit_is_promotable_for_strong_noncanonical_cross_strand_secondary(self):
        unit = {
            "category": "overlapping_alternative",
            "peak_score": 25.4,
            "promote_as_noncanonical_secondary": True,
            "main_cross_strand_candidate": True,
            "main_cross_strand_canonical": False,
            "main_cross_strand_shift_nt": 1,
            "main_cross_strand_shared_cycles": 5,
            "main_cross_strand_support_ratio": 0.75,
        }

        promotable = feature_assembly._secondary_unit_is_promotable(
            unit,
            main_peak_score=39.17,
            score_cutoff=12.5,
        )

        self.assertTrue(promotable)

    def test_secondary_unit_noncanonical_override_still_requires_peak_cutoff(self):
        unit = {
            "category": "overlapping_alternative",
            "peak_score": 10.0,
            "promote_as_noncanonical_secondary": True,
        }

        promotable = feature_assembly._secondary_unit_is_promotable(
            unit,
            main_peak_score=39.17,
            score_cutoff=12.5,
        )

        self.assertFalse(promotable)

    def test_unpaired_secondary_unit_promotes_by_relative_score_support(self):
        unit = {
            "category": "other_local_peak",
            "peak_score": 16.03,
            "non_grey_row_count": 4,
            "exact_row_count": 2,
        }

        promotable = feature_assembly._secondary_unit_is_promotable(
            unit,
            main_peak_score=22.36,
            score_cutoff=12.5,
        )

        self.assertTrue(promotable)

    def test_unpaired_secondary_unit_relative_rule_still_requires_fractional_support(self):
        unit = {
            "category": "other_local_peak",
            "peak_score": 18.0,
            "non_grey_row_count": 5,
            "exact_row_count": 3,
        }

        promotable = feature_assembly._secondary_unit_is_promotable(
            unit,
            main_peak_score=30.0,
            score_cutoff=12.5,
        )

        self.assertFalse(promotable)

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

            all_path = os.path.join(outdir, "21_all_clusters.tsv")
            calls_path = os.path.join(outdir, "21_calls.tsv")
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
