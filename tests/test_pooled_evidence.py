from __future__ import annotations

import csv
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from phasis import cli
from phasis import pooled_evidence as pe
from phasis import runtime as rt
from phasis.stages import feature_assembly as fa

from test_result_comparison import _locus, _write_result_directory


def _phased_cluster(start, phase, cycles, *, strand="w", abun=20.0, noise=None):
    rows = []
    for k in range(cycles):
        rows.append((strand, start + k * phase, phase, abun))
    for pos, length, a in (noise or []):
        rows.append((strand, pos, length, a))
    return rows


def _frame(rows):
    return pd.DataFrame(rows, columns=["strand", "pos", "len", "abun"])


class StrictScoreTests(unittest.TestCase):
    def test_strict_score_equals_feature_assembly(self):
        rng = np.random.default_rng(3)
        with mock.patch.object(rt, "phase", 24, create=True):
            for _ in range(30):
                n = int(rng.integers(5, 80))
                rows = [
                    (str(rng.choice(["w", "c"])), int(1000 + rng.integers(0, 900)),
                     int(rng.choice([24, 24, 23, 22])), float(rng.choice([1, 2, 5, 40])))
                    for _ in range(n)
                ]
                df = _frame(rows)
                w, _, c, _ = fa.compute_phasing_score_Howell_strict(df)
                expected = max(x for x in (w, c) if x is not None)
                got = pe.strict_score_arrays(df["strand"], df["pos"], df["len"], df["abun"], 24)
                self.assertEqual(repr(float(expected)), repr(got))

    def test_decoys_are_reproducible_and_remove_phasing(self):
        df = _frame(_phased_cluster(5000, 21, 14))
        observed = pe.strict_score_arrays(df["strand"], df["pos"], df["len"], df["abun"], 21)
        first = pe.decoy_scores_arrays(df["strand"], df["pos"], df["len"], df["abun"], 21, 10, 7)
        again = pe.decoy_scores_arrays(df["strand"], df["pos"], df["len"], df["abun"], 21, 10, 7)
        self.assertEqual(first, again)
        self.assertGreater(observed, max(first))


class CutoffTests(unittest.TestCase):
    def test_qvalues_and_cutoff(self):
        observed = [30.0, 25.0, 22.0, 15.0, 12.0, 10.0]
        # two replicates: decoys reach 15 and 12 often, never 22
        decoys = [16.0, 15.0, 12.0, 11.0, 10.0, 9.0, 13.0, 12.0, 8.0, 7.0, 6.0, 5.0]
        q = pe.qvalues(observed, decoys, 2)
        self.assertEqual(q[0], 0.0)
        self.assertEqual(q[2], 0.0)
        self.assertGreater(q[3], 0.05)
        self.assertEqual(pe.auto_cutoff(observed, q, 0.05), 22.0)
        self.assertTrue(np.all(np.diff(q[np.argsort(observed)]) <= 1e-12))

    def test_no_cutoff_when_nothing_beats_the_decoys(self):
        q = pe.qvalues([5.0, 4.0], [9.0, 8.0, 7.0, 6.0], 2)
        self.assertEqual(pe.auto_cutoff([5.0, 4.0], q, 0.05), float("inf"))

    def test_parse_cutoff(self):
        self.assertEqual(pe.parse_cutoff("auto"), "auto")
        self.assertEqual(pe.parse_cutoff("OFF"), "off")
        self.assertEqual(pe.parse_cutoff("17.5"), 17.5)
        for bad in ("-1", "nan", "high"):
            with self.assertRaises(pe.EvidenceError):
                pe.parse_cutoff(bad)


class CliTests(unittest.TestCase):
    def _args(self, argv):
        args = cli.build_parser().parse_args(argv)
        cli._validate_args(args)
        return args

    def test_modes(self):
        libs = ["-libs", "a.tag", "b.tag", "-reference", "g.fa"]
        self.assertEqual(self._args(libs).evidence_mode, "combined")
        self.assertFalse(self._args(libs).concat_libs)
        self.assertEqual(self._args(libs + ["--pool_libraries"]).evidence_mode, "pooled")
        self.assertTrue(self._args(libs + ["--pool_libraries"]).concat_libs)
        self.assertEqual(self._args(libs + ["--evidence_mode", "individual"]).evidence_mode, "individual")
        self.assertEqual(self._args(["-libs", "a.tag", "-reference", "g.fa"]).evidence_mode, "individual")
        self.assertEqual(self._args(libs + ["-steps", "cfind"]).evidence_mode, "individual")
        self.assertEqual(self._args(libs + ["--pooled_strict_howell_cutoff", "17.5"]).pooled_strict_howell_cutoff, 17.5)
        with self.assertRaises(SystemExit):
            self._args(libs + ["--pool_libraries", "--evidence_mode", "individual"])
        with self.assertRaises(SystemExit):
            self._args(libs + ["--pooled_strict_howell_cutoff", "x"])

    def test_pass_arguments_drop_the_combined_options(self):
        argv = ["-libs", "a", "b", "--outdir=x", "--evidence_mode", "combined", "-phase", "24",
                "--pooled_strict_howell_cutoff", "auto", "--pooled_fdr", "0.1", "--pool_libraries", "-cores", "4"]
        self.assertEqual(cli._strip_combined_options(argv), ["-libs", "a", "b", "-phase", "24", "-cores", "4"])


def _set_strict(directory, phase, values):
    path = os.path.join(directory, f"{phase}_calls.tsv")
    df = pd.read_csv(path, sep="\t")
    df["Peak_Howell_score_strict"] = [values[i] for i in df["identifier"]]
    df.to_csv(path, sep="\t", index=False)


class CombineEvidenceTests(unittest.TestCase):
    def test_combined_bundle_keeps_individual_and_filters_pooled_only(self):
        phase = 21
        with tempfile.TemporaryDirectory() as root:
            ind = os.path.join(root, "21_results_individual")
            pool = os.path.join(root, "21_results_pooled")
            out = os.path.join(root, "21_results")
            _write_result_directory(ind, phase=phase, loci=[
                _locus("1:1000..1500", "1", 1000, 1500, ("libA", "a1")),
                _locus("1:9000..9400", "1", 9000, 9400, ("libB", "b1")),
            ])
            _write_result_directory(pool, phase=phase, loci=[
                _locus("1:1100..1600", "1", 1100, 1600, ("ALL_LIBS", "p1")),      # anchor
                _locus("1:20000..20400", "1", 20000, 20400, ("ALL_LIBS", "p2")),  # pooled-only, phased
                _locus("1:30000..30400", "1", 30000, 30400, ("ALL_LIBS", "p3")),  # pooled-only, weak
            ])
            _set_strict(pool, phase, {"1:1100..1600": 25.0, "1:20000..20400": 18.0, "1:30000..30400": 9.0})

            for cutoff, kept, explore in (("off", 2, 0), (15.0, 1, 1), (30.0, 0, 2)):
                result = pe.combine_evidence(ind, pool, out, run_dir=root, cutoff=cutoff)
                calls = pd.read_csv(os.path.join(out, "21_calls.tsv"), sep="\t")
                self.assertEqual(set(calls.loc[calls.alib != "ALL_LIBS", "identifier"]), {"1:1000..1500", "1:9000..9400"})
                self.assertEqual(result.summary["pooled_only_kept"], kept)
                self.assertEqual(result.summary["pooled_only_exploratory"], explore)
                self.assertEqual(len(calls.loc[calls.alib == "ALL_LIBS"]), kept)
                expl = pd.read_csv(os.path.join(out, "21_exploratory", "21_calls.tsv"), sep="\t")
                self.assertEqual(len(expl), explore)
                table = pd.read_csv(os.path.join(out, "21_pooled_evidence.tsv"), sep="\t")
                self.assertEqual(dict(zip(table.identifier, table.category))["1:1100..1600"], "anchor")
            self.assertFalse(os.path.exists(os.path.join(root, ".phasis_evidence_21")))

    def test_auto_cutoff_uses_decoys_from_the_pooled_read_table(self):
        phase = 21
        with tempfile.TemporaryDirectory() as root:
            ind = os.path.join(root, "21_results_individual")
            pool = os.path.join(root, "21_results_pooled")
            out = os.path.join(root, "21_results")
            _write_result_directory(ind, phase=phase, loci=[_locus("1:1000..1500", "1", 1000, 1500, ("libA", "a1"))])
            phased = _phased_cluster(20000, 21, 14)
            rng = np.random.default_rng(1)
            noise = [("w", int(30000 + rng.integers(0, 400)), 21, 2.0) for _ in range(40)]
            _write_result_directory(pool, phase=phase, loci=[
                _locus("1:20000..20400", "1", 20000, 20400, ("ALL_LIBS", "p2")),
                _locus("1:30000..30400", "1", 30000, 30400, ("ALL_LIBS", "p3")),
            ])
            observed = {}
            rows = []
            for identifier, cid, cluster in (("1:20000..20400", "c2", phased), ("1:30000..30400", "c3", noise)):
                df = _frame(cluster)
                observed[identifier] = pe.strict_score_arrays(df.strand, df.pos, df.len, df.abun, phase)
                for strand, pos, length, abun in cluster:
                    rows.append({"alib": "ALL_LIBS", "clusterID": cid, "chromosome": "1", "strand": strand,
                                 "pos": pos, "len": length, "abun": abun, "identifier": identifier})
            pd.DataFrame(rows).to_csv(os.path.join(root, "concat_21_PHAS_to_detect.tab.gz"), sep="\t", index=False)
            _set_strict(pool, phase, observed)

            result = pe.combine_evidence(ind, pool, out, run_dir=root, cutoff="auto", fdr=0.05, replicates=20)
            self.assertEqual(result.summary["recomputed_score_mismatches"], 0)
            self.assertEqual(result.summary["decoy_scores"], 40)
            table = pd.read_csv(os.path.join(out, "21_pooled_evidence.tsv"), sep="\t")
            tiers = dict(zip(table.identifier, table.tier))
            self.assertEqual(tiers["1:20000..20400"], "kept")
            self.assertEqual(tiers["1:30000..30400"], "exploratory")

    def test_fallback_without_read_table(self):
        phase = 21
        with tempfile.TemporaryDirectory() as root:
            ind = os.path.join(root, "ind")
            pool = os.path.join(root, "pool")
            _write_result_directory(ind, phase=phase, loci=[_locus("1:1000..1500", "1", 1000, 1500, ("libA", "a1"))])
            _write_result_directory(pool, phase=phase, loci=[_locus("1:5000..5400", "1", 5000, 5400, ("ALL_LIBS", "p"))])
            _set_strict(pool, phase, {"1:5000..5400": 21.0})
            result = pe.combine_evidence(ind, pool, os.path.join(root, "out"), run_dir=root)
            self.assertEqual(result.cutoff, pe.FALLBACK_STRICT_CUTOFF)
            self.assertEqual(result.summary["pooled_only_kept"], 1)


if __name__ == "__main__":
    unittest.main()
