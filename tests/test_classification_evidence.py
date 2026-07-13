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
        "Howell_overlapping_alt_count": 0,
        "Howell_overlapping_alt_best_score": np.nan,
        "Howell_overlapping_alt_best_shift_nt": np.nan,
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


class EvidenceClassificationTests(unittest.TestCase):
    def test_gmm_classifier_retries_threadpoolctl_initialization_failure(self):
        calls = []

        class FakeGaussianMixture:
            def __init__(self, *args, **kwargs):
                calls.append(kwargs)

            def fit_predict(self, X):
                if len(calls) == 1:
                    raise AttributeError("'NoneType' object has no attribute 'split'")
                return np.array([0, 1])

        features = pd.DataFrame(
            [
                _base_feature_row(phasis_score=10.0, Peak_Howell_score=20.0),
                _base_feature_row(phasis_score=100.0, Peak_Howell_score=20.0),
            ]
        )
        with mock.patch.object(classify, "GaussianMixture", FakeGaussianMixture):
            with self.assertWarnsRegex(RuntimeWarning, "retrying with deterministic"):
                out = classify.gmm_classify(
                    features,
                    phasisScoreCutoff=0.0,
                    min_Howell_score=0.0,
                    max_complexity=1.0,
                )

        self.assertEqual(calls[1]["init_params"], "random_from_data")
        self.assertEqual(list(out["label"]), ["non-PHAS", "PHAS"])

    def test_classifier_non_phas_stays_non_phas(self):
        features = pd.DataFrame([_base_feature_row(label="non-PHAS")])
        out = classify.apply_evidence_classification(features, phase=24)

        self.assertEqual(str(out.loc[0, "initial_classifier_label"]), "non-PHAS")
        self.assertEqual(str(out.loc[0, "final_class"]), "non-PHAS")
        self.assertEqual(str(out.loc[0, "report_label"]), "non-PHAS")
        self.assertEqual(str(out.loc[0, "evidence_reason"]), "classifier_non_phas")
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
        out = classify.apply_evidence_classification(features, phase=24)

        self.assertEqual(str(out.loc[0, "final_class"]), "non-PHAS")
        self.assertEqual(str(out.loc[0, "report_label"]), "non-PHAS")
        self.assertEqual(str(out.loc[0, "evidence_reason"]), "insufficient_exact_support")

    def test_missing_exact_support_demotes_to_non_phas(self):
        features = pd.DataFrame([_base_feature_row(Howell_exact_support_score=np.nan)])
        out = classify.apply_evidence_classification(features, phase=24)

        self.assertEqual(str(out.loc[0, "final_class"]), "non-PHAS")
        self.assertEqual(str(out.loc[0, "report_label"]), "non-PHAS")
        self.assertEqual(str(out.loc[0, "evidence_reason"]), "insufficient_exact_support")

    def test_low_positive_exact_support_becomes_phas_like(self):
        features = pd.DataFrame(
            [
                _base_feature_row(Howell_exact_support_score=0.16),
                _base_feature_row(Howell_exact_support_score=4.99),
            ]
        )
        out = classify.apply_evidence_classification(features, phase=24)

        self.assertEqual(out["final_class"].tolist(), ["PHAS-like", "PHAS-like"])
        self.assertEqual(out["report_label"].tolist(), ["non-PHAS", "non-PHAS"])
        self.assertEqual(out["evidence_reason"].tolist(), ["weak_exact_support", "weak_exact_support"])

    def test_exact_support_threshold_boundary_stays_phas_eligible(self):
        features = pd.DataFrame(
            [
                _base_feature_row(Howell_exact_support_score=5.0),
                _base_feature_row(Howell_exact_support_score=18.0),
            ]
        )
        out = classify.apply_evidence_classification(features, phase=24)

        self.assertEqual(out["final_class"].tolist(), ["PHAS", "PHAS"])
        self.assertEqual(out["report_label"].tolist(), ["PHAS", "PHAS"])
        self.assertEqual(out["evidence_reason"].tolist(), ["pass", "pass"])

    def test_coffee_like_low_exact_support_becomes_phas_like(self):
        features = pd.DataFrame(
            [
                _base_feature_row(
                    Peak_Howell_score=14.8000847587,
                    Howell_exact_support_score=0.159482006,
                    Howell_origin_class="coherent_extension",
                    Howell_extension_window_count=104,
                    Howell_extension_span_nt=314,
                )
            ]
        )
        out = classify.apply_evidence_classification(features, phase=21)

        self.assertEqual(str(out.loc[0, "final_class"]), "PHAS-like")
        self.assertEqual(str(out.loc[0, "report_label"]), "non-PHAS")
        self.assertEqual(str(out.loc[0, "evidence_reason"]), "weak_exact_support")

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
        out = classify.apply_evidence_classification(features, phase=24)

        self.assertEqual(str(out.loc[0, "final_class"]), "PHAS-like")
        self.assertEqual(str(out.loc[0, "report_label"]), "non-PHAS")
        self.assertEqual(str(out.loc[0, "evidence_reason"]), "low_score_crowded_window_context")

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
        out = classify.apply_evidence_classification(features, phase=24)

        self.assertEqual(str(out.loc[0, "final_class"]), "PHAS")
        self.assertEqual(str(out.loc[0, "report_label"]), "PHAS")
        self.assertEqual(str(out.loc[0, "evidence_reason"]), "pass")

    def test_weak_exact_support_precedes_weak_scaffold_context(self):
        features = pd.DataFrame(
            [
                _base_feature_row(
                    Peak_Howell_score=24.0,
                    Howell_exact_support_score=3.0,
                    Peak_Howell_score_strict=7.0,
                    Howell_overlapping_alt_count=2,
                    Howell_overlapping_alt_best_score=21.0,
                    Howell_overlapping_alt_best_shift_nt=12.0,
                    complexity=0.24,
                )
            ]
        )
        out = classify.apply_evidence_classification(features, phase=24)

        self.assertEqual(str(out.loc[0, "final_class"]), "PHAS-like")
        self.assertEqual(str(out.loc[0, "report_label"]), "non-PHAS")
        self.assertEqual(str(out.loc[0, "evidence_reason"]), "weak_exact_support")

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
        out = classify.apply_evidence_classification(features, phase=24)

        self.assertEqual(str(out.loc[0, "final_class"]), "PHAS")
        self.assertEqual(str(out.loc[0, "report_label"]), "PHAS")
        self.assertEqual(str(out.loc[0, "evidence_reason"]), "pass")

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
                        "evidence_reason": "manual_curated_complex_locus",
                        "note": "split later",
                    }
                ]
            ).to_csv(override_path, sep="\t", index=False)
            out = classify.apply_evidence_classification(
                features,
                phase=24,
                overrides_path=override_path,
            )

        self.assertEqual(str(out.loc[0, "final_class"]), "PHAS-like")
        self.assertEqual(str(out.loc[0, "report_label"]), "non-PHAS")
        self.assertEqual(str(out.loc[0, "evidence_reason"]), "manual_curated_complex_locus")
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
                classify.apply_evidence_classification(
                    features,
                    phase=24,
                    overrides_path=override_path,
                )


class EvidenceOutputTests(unittest.TestCase):
    def test_finalize_writes_binary_main_tables_and_multiclass_evidence_table(self):
        features = pd.DataFrame(
            [
                classify.apply_evidence_classification(pd.DataFrame([_base_feature_row()]), phase=24).iloc[0].to_dict(),
                classify.apply_evidence_classification(
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
                classify.apply_evidence_classification(
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

            all_df = pd.read_csv(os.path.join(outdir, "24_all_clusters.tsv"), sep="\t")
            calls_df = pd.read_csv(os.path.join(outdir, "24_calls.tsv"), sep="\t")
            evidence_df = pd.read_csv(os.path.join(outdir, "24_classification_evidence.tsv"), sep="\t")
            phas_like_dir = os.path.join(outdir, "24_PHAS_like")
            phas_like_calls = pd.read_csv(
                os.path.join(phas_like_dir, "24_PHAS_like_calls.tsv"),
                sep="\t",
            )
            phas_like_evidence = pd.read_csv(
                os.path.join(phas_like_dir, "24_PHAS_like_classification_evidence.tsv"),
                sep="\t",
            )
            with open(os.path.join(phas_like_dir, "24_PHAS_like.gff"), encoding="utf-8") as handle:
                phas_like_gff = handle.read()

            self.assertEqual(sorted(all_df["label"].unique().tolist()), ["PHAS", "non-PHAS"])
            self.assertEqual(calls_df["identifier"].tolist(), ["chr1:100..400"])
            self.assertEqual(sorted(evidence_df["final_class"].tolist()), ["PHAS", "PHAS-like", "non-PHAS"])
            row = evidence_df.loc[evidence_df["identifier"] == "chr2:500..900"].iloc[0]
            self.assertEqual(str(row["report_label"]), "non-PHAS")
            self.assertEqual(str(row["final_class"]), "PHAS-like")
            self.assertEqual(str(row["evidence_reason"]), "low_score_crowded_window_context")
            self.assertEqual(int(row["Howell_crowding_window_count"]), 7)
            self.assertIn("Howell_exact_relaxed_ratio", evidence_df.columns)
            self.assertIn("Howell_strict_relaxed_ratio", evidence_df.columns)
            self.assertEqual(phas_like_calls["identifier"].tolist(), ["chr2:500..900"])
            self.assertEqual(phas_like_evidence["final_class"].tolist(), ["PHAS-like"])
            self.assertIn("24-PHAS-like", phas_like_gff)
            self.assertFalse(
                any(
                    "GMM" in name or "KNN" in name
                    for _root, _dirs, files in os.walk(outdir)
                    for name in files
                )
            )

    def test_finalize_creates_empty_phas_like_bundle(self):
        features = classify.apply_evidence_classification(
            pd.DataFrame([_base_feature_row()]),
            phase=24,
        )

        with tempfile.TemporaryDirectory() as outdir:
            with mock.patch.object(output, "make_pool", return_value=_DummyPool()):
                with mock.patch.object(output, "plot_report_heat_map", return_value=None):
                    with mock.patch.object(output, "plot_phasAbundance_heat_map", return_value=None):
                        with mock.patch.object(output, "plot_totalAbundance_heat_map", return_value=None):
                            with mock.patch.object(output, "plot_howell_score_heat_maps", return_value=None):
                                output.finalize_and_write_results(
                                    "GMM",
                                    features,
                                    job_outdir=outdir,
                                    job_phase=24,
                                )

            bundle = os.path.join(outdir, "24_PHAS_like")
            self.assertTrue(os.path.isdir(bundle))
            self.assertTrue(os.path.isdir(os.path.join(bundle, "locus_plots")))
            self.assertEqual(
                len(pd.read_csv(os.path.join(bundle, "24_PHAS_like_calls.tsv"), sep="\t")),
                0,
            )
            self.assertEqual(
                len(
                    pd.read_csv(
                        os.path.join(bundle, "24_PHAS_like_classification_evidence.tsv"),
                        sep="\t",
                    )
                ),
                0,
            )
            self.assertEqual(os.path.getsize(os.path.join(bundle, "24_PHAS_like.gff")), 0)


if __name__ == "__main__":
    unittest.main()
