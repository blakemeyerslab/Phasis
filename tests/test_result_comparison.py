import json
import os
import tempfile
import unittest
from collections import Counter

import pandas as pd

from phasis.result_comparison import compare_result_directories


CALL_COLUMNS = [
    "identifier",
    "phasis_score",
    "achr",
    "start",
    "end",
    "alib",
    "Peak_Howell_score",
    "Peak_Howell_score_strict",
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
    "Howell_exact_relaxed_ratio",
    "Howell_strict_relaxed_ratio",
]

ALL_CLUSTER_COLUMNS = [
    "identifier",
    "phasis_score",
    "achr",
    "start",
    "end",
    "complexity",
    "strand_bias",
    "log_clust_len_norm_counts",
    "ratio_abund_len_phase",
    "label",
    "alib",
]

EVIDENCE_COLUMNS = [
    "identifier",
    "alib",
    "cID",
    "initial_classifier_label",
    "report_label",
    "final_class",
    "evidence_reason",
    "Peak_Howell_score",
    "Howell_exact_support_score",
    "Howell_origin_class",
]

PHASIRNA_COLUMNS = [
    "identifier",
    "cID",
    "alib",
    "phase",
    "window_unit_id",
    "window_unit_role",
    "window_unit_rank",
    "window_unit_shift_nt",
    "strand",
    "observed_pos",
    "expected_register_pos",
    "register_class",
    "abun",
    "tag_seq",
    "hits",
]


def _locus(identifier, chromosome, start, end, *detections):
    return {
        "identifier": identifier,
        "achr": chromosome,
        "start": start,
        "end": end,
        "detections": detections,
    }


def _write_result_directory(directory, *, phase, loci):
    os.makedirs(directory, exist_ok=True)
    calls = []
    all_clusters = []
    evidence = []
    phasirnas = []

    for locus in loci:
        for detection_index, (alib, cid) in enumerate(locus["detections"]):
            call = {
                "identifier": locus["identifier"],
                "phasis_score": 300.0 - detection_index,
                "achr": locus["achr"],
                "start": locus["start"],
                "end": locus["end"],
                "alib": alib,
                "Peak_Howell_score": 20.0 - detection_index,
                "Peak_Howell_score_strict": 12.0 - detection_index,
                "Howell_exact_support_score": 8.0 - detection_index,
                "Howell_ambiguity_count": 0,
                "Howell_alt_register_count": 0,
                "Howell_overlap_margin": 1.0,
                "Howell_extension_window_count": 0,
                "Howell_extension_span_nt": phase * 10,
                "Howell_origin_window_count": 1,
                "Howell_origin_frame_count": 1,
                "Howell_origin_margin": 1.0,
                "Howell_origin_class": "unique_origin",
                "Howell_additional_peak_count": 0,
                "Howell_additional_peak_best_score": 0.0,
                "Howell_overlapping_alt_count": 0,
                "Howell_overlapping_alt_best_score": 0.0,
                "Howell_overlapping_alt_best_shift_nt": 0,
                "Howell_exact_relaxed_ratio": 0.5,
                "Howell_strict_relaxed_ratio": 0.5,
            }
            calls.append(call)
            all_clusters.append(
                {
                    "identifier": locus["identifier"],
                    "phasis_score": call["phasis_score"],
                    "achr": locus["achr"],
                    "start": locus["start"],
                    "end": locus["end"],
                    "complexity": 0.2,
                    "strand_bias": 0.5,
                    "log_clust_len_norm_counts": 1.2,
                    "ratio_abund_len_phase": 2.0,
                    "label": "PHAS",
                    "alib": alib,
                }
            )
            evidence.append(
                {
                    "identifier": locus["identifier"],
                    "alib": alib,
                    "cID": cid,
                    "initial_classifier_label": "PHAS",
                    "report_label": "PHAS",
                    "final_class": "PHAS",
                    "evidence_reason": "classifier_phas",
                    "Peak_Howell_score": call["Peak_Howell_score"],
                    "Howell_exact_support_score": call["Howell_exact_support_score"],
                    "Howell_origin_class": "unique_origin",
                }
            )
            phasirnas.append(
                {
                    "identifier": locus["identifier"],
                    "cID": cid,
                    "alib": alib,
                    "phase": phase,
                    "window_unit_id": "unit_main",
                    "window_unit_role": "main_hpsp",
                    "window_unit_rank": 0,
                    "window_unit_shift_nt": 0,
                    "strand": "w",
                    "observed_pos": locus["start"] + phase,
                    "expected_register_pos": locus["start"] + phase,
                    "register_class": "core_exact",
                    "abun": 10.0,
                    "tag_seq": "A" * phase,
                    "hits": 1,
                }
            )

    # Real all-cluster and evidence tables also contain candidates which were
    # not called as PHAS.  They must not leak into a comparison bundle.
    all_clusters.append(
        {
            "identifier": "chr9:9000..9200",
            "phasis_score": 0.0,
            "achr": "chr9",
            "start": 9000,
            "end": 9200,
            "complexity": 0.9,
            "strand_bias": 0.0,
            "log_clust_len_norm_counts": 0.0,
            "ratio_abund_len_phase": 0.0,
            "label": "non-PHAS",
            "alib": loci[0]["detections"][0][0],
        }
    )
    evidence.append(
        {
            "identifier": "chr9:9000..9200",
            "alib": loci[0]["detections"][0][0],
            "cID": "distractor-cid",
            "initial_classifier_label": "non-PHAS",
            "report_label": "non-PHAS",
            "final_class": "non-PHAS",
            "evidence_reason": "classifier_non_phas",
            "Peak_Howell_score": 0.0,
            "Howell_exact_support_score": 0.0,
            "Howell_origin_class": "insufficient_exact_support",
        }
    )

    pd.DataFrame(calls, columns=CALL_COLUMNS).to_csv(
        os.path.join(directory, f"{phase}_calls.tsv"), sep="\t", index=False
    )
    pd.DataFrame(all_clusters, columns=ALL_CLUSTER_COLUMNS).to_csv(
        os.path.join(directory, f"{phase}_all_clusters.tsv"), sep="\t", index=False
    )
    pd.DataFrame(evidence, columns=EVIDENCE_COLUMNS).to_csv(
        os.path.join(directory, f"{phase}_classification_evidence.tsv"),
        sep="\t",
        index=False,
    )
    pd.DataFrame(phasirnas, columns=PHASIRNA_COLUMNS).to_csv(
        os.path.join(directory, f"{phase}_phasiRNAs.tsv"), sep="\t", index=False
    )

    with open(os.path.join(directory, f"{phase}_PHAS.gff"), "w", encoding="utf-8") as handle:
        for locus in loci:
            handle.write(
                f"{locus['achr']}\tPhasis\t{phase}-PHAS\t{locus['start']}\t"
                f"{locus['end']}\t300.0\t.\t.\tid={locus['identifier']};complexity=0.2\n"
            )


def _summary_dict(result):
    summary = getattr(result, "summary", result)
    return dict(summary)


class ResultComparisonTests(unittest.TestCase):
    def _build_pair(self, root):
        run_a = os.path.join(root, "run_a")
        run_b = os.path.join(root, "run_b")

        # Deliberately unsorted. The broad pooled call supports both finer,
        # non-overlapping calls without merging those finer calls together.
        _write_result_directory(
            run_a,
            phase=21,
            loci=[
                _locus("chr10:100..250", "chr10", 100, 250, ("ALL_LIBS", "a-only")),
                _locus("chr2:5000..5500", "chr2", 5000, 5500, ("ALL_LIBS", "a-exact")),
                _locus("chr1:100..1000", "chr1", 100, 1000, ("ALL_LIBS", "a-broad")),
            ],
        )
        _write_result_directory(
            run_b,
            phase=21,
            loci=[
                _locus("chr10:1000..1200", "chr10", 1000, 1200, ("leaf_1", "b-only")),
                _locus("chr1:700..900", "chr1", 700, 900, ("leaf_1", "b-fine-2")),
                _locus(
                    "chr2:5000..5500",
                    "chr2",
                    5000,
                    5500,
                    ("leaf_1", "b-exact-1"),
                    ("leaf_2", "b-exact-2"),
                ),
                _locus("chr1:120..300", "chr1", 120, 300, ("leaf_1", "b-fine-1")),
            ],
        )
        return run_a, run_b

    def test_auto_prefers_non_pooled_and_preserves_finer_shared_loci(self):
        with tempfile.TemporaryDirectory() as root:
            run_a, run_b = self._build_pair(root)
            outdir = os.path.join(root, "comparison")
            result = compare_result_directories(
                run_a,
                run_b,
                outdir,
                label_a="pooled",
                label_b="individual",
                prefer="auto",
                overlap_buffer=0,
            )
            summary = _summary_dict(result)

            self.assertEqual(summary["phase"], 21)
            self.assertEqual(summary["mode_a"], "pooled")
            self.assertEqual(summary["mode_b"], "non-pooled")
            self.assertEqual(summary["preferred_run"], "run_b")
            self.assertEqual(summary["input_loci_a"], 3)
            self.assertEqual(summary["input_loci_b"], 4)
            self.assertEqual(summary["overlap_pairs"], 3)
            self.assertEqual(summary["shared_loci"], 3)
            self.assertEqual(summary["run_a_only_loci"], 1)
            self.assertEqual(summary["run_b_only_loci"], 1)
            self.assertEqual(summary["combined_loci"], 5)

            expected_outputs = {
                "21_calls.tsv",
                "21_all_clusters.tsv",
                "21_classification_evidence.tsv",
                "21_phasiRNAs.tsv",
                "21_PHAS.gff",
                "21_combined_loci.tsv",
                "21_shared_loci.tsv",
                "21_run_a_only_loci.tsv",
                "21_run_b_only_loci.tsv",
                "21_source_locus_map.tsv",
                "comparison_summary.txt",
                "comparison_manifest.json",
            }
            self.assertTrue(expected_outputs.issubset(set(os.listdir(outdir))))

            combined = pd.read_csv(os.path.join(outdir, "21_combined_loci.tsv"), sep="\t")
            expected_ids = [
                "chr1:120..300",
                "chr1:700..900",
                "chr2:5000..5500",
                "chr10:100..250",
                "chr10:1000..1200",
            ]
            self.assertEqual(combined["identifier"].tolist(), expected_ids)
            self.assertEqual(
                combined["category"].tolist(),
                ["shared", "shared", "shared", "run_a_only", "run_b_only"],
            )
            self.assertEqual(
                combined["representative_run"].tolist(),
                ["run_b", "run_b", "run_b", "run_a", "run_b"],
            )

            shared = pd.read_csv(os.path.join(outdir, "21_shared_loci.tsv"), sep="\t")
            a_only = pd.read_csv(os.path.join(outdir, "21_run_a_only_loci.tsv"), sep="\t")
            b_only = pd.read_csv(os.path.join(outdir, "21_run_b_only_loci.tsv"), sep="\t")
            self.assertEqual(shared["identifier"].tolist(), expected_ids[:3])
            self.assertEqual(a_only["identifier"].tolist(), ["chr10:100..250"])
            self.assertEqual(b_only["identifier"].tolist(), ["chr10:1000..1200"])

            # Calls retain separate library detections for the selected
            # non-pooled representative, but not the pooled duplicate.
            calls = pd.read_csv(os.path.join(outdir, "21_calls.tsv"), sep="\t")
            self.assertEqual(
                Counter(calls["identifier"]),
                Counter(
                    {
                        "chr1:120..300": 1,
                        "chr1:700..900": 1,
                        "chr2:5000..5500": 2,
                        "chr10:100..250": 1,
                        "chr10:1000..1200": 1,
                    }
                ),
            )
            exact_calls = calls.loc[calls["identifier"] == "chr2:5000..5500"]
            self.assertEqual(sorted(exact_calls["alib"].tolist()), ["leaf_1", "leaf_2"])

            # Every companion output is filtered using both source run and
            # source identifier. This matters when both inputs use the exact
            # same identifier but one source is preferred.
            selected_ids = set(expected_ids)
            for filename in (
                "21_all_clusters.tsv",
                "21_classification_evidence.tsv",
                "21_phasiRNAs.tsv",
            ):
                frame = pd.read_csv(os.path.join(outdir, filename), sep="\t")
                self.assertEqual(set(frame["identifier"]), selected_ids)
                self.assertNotIn("chr9:9000..9200", set(frame["identifier"]))
                exact_rows = frame.loc[frame["identifier"] == "chr2:5000..5500"]
                self.assertEqual(sorted(exact_rows["alib"].tolist()), ["leaf_1", "leaf_2"])

            source_map = pd.read_csv(
                os.path.join(outdir, "21_source_locus_map.tsv"), sep="\t"
            )
            broad_support = source_map.loc[
                (source_map["source_run"] == "run_a")
                & (source_map["source_identifier"] == "chr1:100..1000")
            ]
            self.assertEqual(
                set(broad_support["combined_identifier"]),
                {"chr1:120..300", "chr1:700..900"},
            )
            for identifier in ("chr1:120..300", "chr1:700..900", "chr2:5000..5500"):
                representative = source_map.loc[
                    (source_map["combined_identifier"] == identifier)
                    & (source_map["source_run"] == "run_b")
                ]
                self.assertTrue(
                    all(str(value).lower() == "true" for value in representative["is_representative"])
                )

            overlap_pairs = pd.read_csv(
                os.path.join(outdir, "21_overlap_pairs.tsv"), sep="\t"
            )
            broad_pairs = overlap_pairs.loc[
                overlap_pairs["run_a_identifier"] == "chr1:100..1000"
            ]
            self.assertEqual(len(broad_pairs), 2)
            self.assertEqual(set(broad_pairs["run_a_match_count"]), {2})
            self.assertEqual(
                set(broad_pairs["relationship_shape"]),
                {"run_a_one_to_many_run_b"},
            )

            with open(os.path.join(outdir, "21_PHAS.gff"), encoding="utf-8") as handle:
                gff_lines = [line.rstrip("\n") for line in handle if line.strip()]
            self.assertEqual(len(gff_lines), 5)
            self.assertEqual([line.split("\t")[0] for line in gff_lines], ["chr1", "chr1", "chr2", "chr10", "chr10"])
            self.assertFalse(any("chr1:100..1000" in line for line in gff_lines))

            with open(
                os.path.join(outdir, "comparison_manifest.json"), encoding="utf-8"
            ) as handle:
                manifest = json.load(handle)
            for key, value in summary.items():
                self.assertEqual(manifest[key], value)

            with open(os.path.join(outdir, "comparison_summary.txt"), encoding="utf-8") as handle:
                summary_text = handle.read().lower()
            self.assertIn("shared", summary_text)
            self.assertIn("combined", summary_text)

            generated = [
                path
                for current_root, _dirs, files in os.walk(outdir)
                for filename in files
                for path in [os.path.join(current_root, filename)]
            ]
            self.assertFalse(
                any(path.lower().endswith((".pdf", ".png", ".svg")) for path in generated)
            )
            self.assertFalse(any("plot" in os.path.basename(path).lower() for path in generated))

    def test_phase_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            run_a = os.path.join(root, "run_a")
            run_b = os.path.join(root, "run_b")
            locus = [_locus("chr1:100..300", "chr1", 100, 300, ("lib", "cid"))]
            _write_result_directory(run_a, phase=21, loci=locus)
            _write_result_directory(run_b, phase=24, loci=locus)

            with self.assertRaisesRegex(ValueError, "phase|21|24"):
                compare_result_directories(run_a, run_b, os.path.join(root, "out"))

    def test_auto_preference_is_independent_of_input_order(self):
        with tempfile.TemporaryDirectory() as root:
            pooled, non_pooled = self._build_pair(root)
            forward = os.path.join(root, "forward")
            reverse = os.path.join(root, "reverse")

            forward_result = compare_result_directories(
                pooled,
                non_pooled,
                forward,
                prefer="auto",
            )
            reverse_result = compare_result_directories(
                non_pooled,
                pooled,
                reverse,
                prefer="auto",
            )

            self.assertEqual(forward_result.summary["preferred_run"], "run_b")
            self.assertEqual(reverse_result.summary["preferred_run"], "run_a")
            self.assertEqual(forward_result.summary["combined_loci"], 5)
            self.assertEqual(reverse_result.summary["combined_loci"], 5)
            for filename in (
                "21_calls.tsv",
                "21_all_clusters.tsv",
                "21_classification_evidence.tsv",
                "21_phasiRNAs.tsv",
                "21_PHAS.gff",
            ):
                with open(os.path.join(forward, filename), "rb") as forward_handle:
                    forward_bytes = forward_handle.read()
                with open(os.path.join(reverse, filename), "rb") as reverse_handle:
                    reverse_bytes = reverse_handle.read()
                self.assertEqual(forward_bytes, reverse_bytes, filename)

    def test_missing_required_calls_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            run_a, run_b = self._build_pair(root)
            calls_path = os.path.join(run_b, "21_calls.tsv")
            calls = pd.read_csv(calls_path, sep="\t").drop(columns=["end"])
            calls.to_csv(calls_path, sep="\t", index=False)

            with self.assertRaisesRegex(ValueError, "end|column|schema"):
                compare_result_directories(run_a, run_b, os.path.join(root, "out"))

    def test_output_directory_cannot_be_an_input_directory(self):
        with tempfile.TemporaryDirectory() as root:
            run_a, run_b = self._build_pair(root)

            with self.assertRaisesRegex(ValueError, "output|input|same"):
                compare_result_directories(run_a, run_b, run_a)
            with self.assertRaisesRegex(ValueError, "output|input|inside"):
                compare_result_directories(
                    run_a,
                    run_b,
                    os.path.join(run_a, "comparison"),
                )

    def test_nonempty_output_requires_force_and_force_preserves_unknown_files(self):
        with tempfile.TemporaryDirectory() as root:
            run_a, run_b = self._build_pair(root)
            outdir = os.path.join(root, "comparison")
            os.makedirs(outdir)
            sentinel = os.path.join(outdir, "keep_me.txt")
            with open(sentinel, "w", encoding="utf-8") as handle:
                handle.write("unrelated user file\n")

            with self.assertRaises((ValueError, FileExistsError)):
                compare_result_directories(run_a, run_b, outdir)
            self.assertTrue(os.path.isfile(sentinel))
            self.assertEqual(os.listdir(outdir), ["keep_me.txt"])

            compare_result_directories(run_a, run_b, outdir, force=True)
            self.assertTrue(os.path.isfile(sentinel))
            self.assertTrue(os.path.isfile(os.path.join(outdir, "21_calls.tsv")))


if __name__ == "__main__":
    unittest.main()
