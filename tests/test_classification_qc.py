from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from phasis.stages import classify
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


def _base_feature_row(**overrides):
    row = {
        "identifier": "chr1:100..400",
        "cID": "cluster_1",
        "alib": "libA",
        "label": "PHAS",
        "Peak_Howell_score": 20.0,
        "Howell_exact_support_score": 18.0,
        "Howell_origin_class": "coherent_extension",
        "Howell_origin_window_count": 0,
        "Howell_origin_frame_count": 0,
        "Howell_alt_register_count": 0,
        "Howell_additional_peak_count": 0,
        "Howell_additional_peak_best_score": np.nan,
        "Howell_crowding_window_count": 0,
        "Howell_crowding_best_score": np.nan,
        "Howell_crowding_score_gap": np.nan,
        "phasis_score": 300.0,
        "complexity": 0.05,
        "strand_bias": 0.8,
        "log_clust_len_norm_counts": 1.2,
        "ratio_abund_len_phase": 5.5,
        "combined_fishers": 1e-8,
        "total_abund": 500.0,
        "w_Howell_score": 12.3,
        "w_window_start": 100,
        "w_window_end": 309,
        "c_Howell_score": 0.0,
        "c_window_start": np.nan,
        "c_window_end": np.nan,
        "Howell_ambiguity_count": 0,
        "Howell_overlap_margin": np.nan,
        "Howell_extension_window_count": 2,
        "Howell_extension_span_nt": 260,
        "Howell_origin_margin": np.nan,
        "w_Howell_score_strict": 10.1,
        "w_window_start_strict": 100,
        "w_window_end_strict": 309,
        "c_Howell_score_strict": 0.0,
        "c_window_start_strict": np.nan,
        "c_window_end_strict": np.nan,
        "Peak_Howell_score_strict": 10.1,
    }
    row.update(overrides)
    return row


class QCReclassificationTests(unittest.TestCase):
    def test_classifier_non_phas_stays_non_phas(self):
        features = pd.DataFrame([_base_feature_row(label="non-PHAS")])
        out = classify.apply_qc_reclassification(features, phase=24)

        self.assertEqual(str(out.loc[0, "pre_qc_label"]), "non-PHAS")
        self.assertEqual(str(out.loc[0, "final_class"]), "non-PHAS")
        self.assertEqual(str(out.loc[0, "report_label"]), "non-PHAS")
        self.assertEqual(str(out.loc[0, "qc_reason"]), "classifier_non_phas")
        self.assertEqual(str(out.loc[0, "label"]), "non-PHAS")

    def test_zero_exact_support_demotes_to_non_phas(self):
        features = pd.DataFrame(
            [
                _base_feature_row(
                    Howell_exact_support_score=0.0,
                    Howell_origin_class="insufficient_exact_support",
                )
            ]
        )
        out = classify.apply_qc_reclassification(features, phase=24)

        self.assertEqual(str(out.loc[0, "final_class"]), "non-PHAS")
        self.assertEqual(str(out.loc[0, "report_label"]), "non-PHAS")
        self.assertEqual(str(out.loc[0, "qc_reason"]), "insufficient_exact_support")

    def test_low_score_crowded_window_context_becomes_phas_like(self):
        features = pd.DataFrame(
            [
                _base_feature_row(
                    Peak_Howell_score=16.0,
                    Howell_crowding_window_count=6,
                    Howell_crowding_best_score=14.5,
                    Howell_crowding_score_gap=1.5,
                )
            ]
        )
        out = classify.apply_qc_reclassification(features, phase=24)

        self.assertEqual(str(out.loc[0, "final_class"]), "PHAS-like")
        self.assertEqual(str(out.loc[0, "report_label"]), "non-PHAS")
        self.assertEqual(str(out.loc[0, "qc_reason"]), "low_score_crowded_window_context")

    def test_strong_multi_peak_coherent_extension_stays_phas(self):
        features = pd.DataFrame(
            [
                _base_feature_row(
                    Peak_Howell_score=39.0,
                    Howell_additional_peak_count=3,
                    Howell_additional_peak_best_score=33.0,
                    Howell_crowding_window_count=1,
                )
            ]
        )
        out = classify.apply_qc_reclassification(features, phase=24)

        self.assertEqual(str(out.loc[0, "final_class"]), "PHAS")
        self.assertEqual(str(out.loc[0, "report_label"]), "PHAS")
        self.assertEqual(str(out.loc[0, "qc_reason"]), "pass")

    def test_ambiguous_origin_remains_phas_in_v1(self):
        features = pd.DataFrame(
            [
                _base_feature_row(
                    Howell_origin_class="ambiguous_origin",
                    Howell_origin_window_count=2,
                    Howell_origin_frame_count=2,
                    Howell_alt_register_count=1,
                )
            ]
        )
        out = classify.apply_qc_reclassification(features, phase=24)

        self.assertEqual(str(out.loc[0, "final_class"]), "PHAS")
        self.assertEqual(str(out.loc[0, "report_label"]), "PHAS")
        self.assertEqual(str(out.loc[0, "qc_reason"]), "pass")

    def test_manual_override_takes_precedence(self):
        features = pd.DataFrame([_base_feature_row()])
        with tempfile.TemporaryDirectory() as tmpdir:
            override_path = os.path.join(tmpdir, "overrides.tsv")
            pd.DataFrame(
                [
                    {
                        "identifier": "chr1:100..400",
                        "alib": "libA",
                        "final_class": "PHAS-like",
                        "qc_reason": "manual_curated_complex_locus",
                        "note": "split later",
                    }
                ]
            ).to_csv(override_path, sep="\t", index=False)
            out = classify.apply_qc_reclassification(
                features,
                phase=24,
                overrides_path=override_path,
            )

        self.assertEqual(str(out.loc[0, "final_class"]), "PHAS-like")
        self.assertEqual(str(out.loc[0, "report_label"]), "non-PHAS")
        self.assertEqual(str(out.loc[0, "qc_reason"]), "manual_curated_complex_locus")
        self.assertEqual(str(out.loc[0, "override_note"]), "split later")

    def test_duplicate_override_keys_fail_fast(self):
        features = pd.DataFrame([_base_feature_row()])
        with tempfile.TemporaryDirectory() as tmpdir:
            override_path = os.path.join(tmpdir, "overrides.tsv")
            pd.DataFrame(
                [
                    {"identifier": "chr1:100..400", "alib": "libA", "final_class": "PHAS"},
                    {"identifier": "chr1:100..400", "alib": "libA", "final_class": "non-PHAS"},
                ]
            ).to_csv(override_path, sep="\t", index=False)
            with self.assertRaises(ValueError):
                classify.apply_qc_reclassification(
                    features,
                    phase=24,
                    overrides_path=override_path,
                )


class QCOutputTests(unittest.TestCase):
    def test_finalize_writes_binary_main_tables_and_multiclass_qc_table(self):
        features = pd.DataFrame(
            [
                classify.apply_qc_reclassification(pd.DataFrame([_base_feature_row()]), phase=24).iloc[0].to_dict(),
                classify.apply_qc_reclassification(
                    pd.DataFrame(
                        [
                            _base_feature_row(
                                identifier="chr2:500..900",
                                cID="cluster_2",
                                alib="libB",
                                Peak_Howell_score=17.0,
                                Howell_crowding_window_count=7,
                                Howell_crowding_best_score=15.5,
                                Howell_crowding_score_gap=1.5,
                            )
                        ]
                    ),
                    phase=24,
                ).iloc[0].to_dict(),
                classify.apply_qc_reclassification(
                    pd.DataFrame(
                        [
                            _base_feature_row(
                                identifier="chr3:1000..1400",
                                cID="cluster_3",
                                alib="libC",
                                Howell_exact_support_score=0.0,
                                Howell_origin_class="insufficient_exact_support",
                            )
                        ]
                    ),
                    phase=24,
                ).iloc[0].to_dict(),
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
                                    job_phase=24,
                                )

            all_df = pd.read_csv(os.path.join(outdir, "24_KNN_all_clusters.tsv"), sep="\t")
            calls_df = pd.read_csv(os.path.join(outdir, "24_KNN_calls.tsv"), sep="\t")
            qc_df = pd.read_csv(os.path.join(outdir, "24_KNN_classification_qc.tsv"), sep="\t")

            self.assertEqual(sorted(all_df["label"].unique().tolist()), ["PHAS", "non-PHAS"])
            self.assertEqual(calls_df["identifier"].tolist(), ["chr1:100..400"])
            self.assertEqual(sorted(qc_df["final_class"].tolist()), ["PHAS", "PHAS-like", "non-PHAS"])
            row = qc_df.loc[qc_df["identifier"] == "chr2:500..900"].iloc[0]
            self.assertEqual(str(row["report_label"]), "non-PHAS")
            self.assertEqual(str(row["final_class"]), "PHAS-like")
            self.assertEqual(str(row["qc_reason"]), "low_score_crowded_window_context")
            self.assertEqual(int(row["Howell_crowding_window_count"]), 7)


if __name__ == "__main__":
    unittest.main()
