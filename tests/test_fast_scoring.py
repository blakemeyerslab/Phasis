from __future__ import annotations

import copy
import os
import random
import unittest
from unittest import mock

import numpy as np
from scipy.stats import combine_pvalues, hypergeom, mannwhitneyu

from phasis.stages import cluster_scoring, fast_scoring


def _same_float(a, b) -> bool:
    a = float(a)
    b = float(b)
    if a != a and b != b:  # both NaN
        return True
    return a == b


class MannWhitneyTests(unittest.TestCase):
    def _check(self, x, y):
        expected = mannwhitneyu(x, y, alternative="greater")[1]
        got = fast_scoring.mannwhitneyu_greater_pvalue(x, y)
        self.assertTrue(_same_float(expected, got), msg=f"x={x} y={y} scipy={expected!r} fast={got!r}")

    def test_asymptotic_with_ties_matches_scipy(self):
        rng = random.Random(1)
        for _ in range(3000):
            n1 = rng.randint(1, 40)
            n2 = rng.randint(1, 120)
            top = rng.choice([2, 5, 50, 5000])
            x = [rng.randint(1, top) for _ in range(n1)]
            y = [rng.randint(1, top) for _ in range(n2)]
            if len(set(x + y)) == 1:
                continue
            self._check(x, y)

    def test_exact_branch_matches_scipy(self):
        rng = random.Random(2)
        for _ in range(3000):
            n1 = rng.randint(1, 8)
            n2 = rng.randint(1, 60)
            values = rng.sample(range(1, 100000), n1 + n2)  # no ties
            x, y = values[:n1], values[n1:]
            if rng.random() < 0.5:
                x, y = y, x
            self._check(x, y)

    def test_large_tie_free_samples_use_asymptotic(self):
        rng = random.Random(3)
        for _ in range(300):
            n1 = rng.randint(9, 60)
            n2 = rng.randint(9, 60)
            values = rng.sample(range(1, 100000), n1 + n2)
            self._check(values[:n1], values[n1:])

    def test_ranksum_matches_legacy(self):
        rng = random.Random(4)
        cases = [([], []), ([3], []), ([], [3]), ([2, 2], [2]), ([0], [0])]
        for _ in range(1000):
            x = [rng.randint(1, 4) for _ in range(rng.randint(0, 12))]
            y = [rng.randint(1, 4) for _ in range(rng.randint(0, 12))]
            cases.append((x, y))
        for x, y in cases:
            expected = cluster_scoring.ranksumtest([0, 0, 0, list(x), list(y), []])
            got = fast_scoring.ranksum_rounded(list(x), list(y))
            self.assertTrue(_same_float(expected, got), msg=f"x={x} y={y} {expected!r} {got!r}")


class StoufferAndHypergeomTests(unittest.TestCase):
    def test_stouffer_matches_scipy(self):
        rng = random.Random(5)
        pvals = [0.0, 1.0, 0.5, 1e-10, 1 - 1e-10]
        pvals += [round(rng.random(), 10) for _ in range(300)]
        pvals += [round(10 ** -rng.uniform(0, 10), 10) for _ in range(300)]
        with np.errstate(all="ignore"):
            for _ in range(5000):
                p1 = np.float64(rng.choice(pvals))
                p2 = rng.choice(pvals)
                expected = round(combine_pvalues([p1, p2], method="stouffer", weights=None)[1], 15)
                got = fast_scoring.stouffer2_rounded(p1, p2)
                self.assertTrue(_same_float(expected, got), msg=f"{p1!r} {p2!r} {expected!r} {got!r}")

    def test_hypergeom_matches_scipy(self):
        fast_scoring._HYPER_CACHE.clear()
        for phase in (21, 24):
            M = 15 * phase * 2
            K = 90
            for N in list(range(0, 200, 7)) + [M - 1, M]:
                for X in range(0, min(N, K) + 1, 3):
                    expected = round(hypergeom.sf(X - 1, M, K, N), 10)
                    got = fast_scoring.hyper_sf_rounded(X, M, K, N)
                    self.assertTrue(_same_float(expected, got), msg=f"X={X} N={N} {expected!r} {got!r}")


class RoundLikeNumpyTests(unittest.TestCase):
    def test_matches_numpy_scalar_round_bit_for_bit(self):
        rng = random.Random(7)
        values = [0.0, 1.0, 0.5, 1e-10, 5e-11, 1.5e-10, 2.5e-10, 1e-15, 5e-16, 1 - 1e-10,
                  1 - 5e-11, 0.12345678905, 0.12345678915, 0.123456789012345, 1e-300,
                  float("nan"), float("inf")]
        values += [rng.random() for _ in range(200000)]
        values += [10 ** -rng.uniform(0, 20) for _ in range(200000)]
        values += [(k + 0.5) / 1e10 for k in rng.sample(range(10 ** 9), 20000)]
        values += [1 - 10 ** -rng.uniform(0, 17) for _ in range(50000)]
        with np.errstate(all="ignore"):
            for digits, scale in ((10, 1e10), (15, 1e15)):
                for v in values:
                    expected = round(np.float64(v), digits)
                    got = fast_scoring._round_like_numpy(v, scale)
                    self.assertTrue(_same_float(expected, got), msg=f"{v!r} d={digits} {expected!r} {got!r}")
                    self.assertEqual(str(expected), str(got))
                    self.assertIs(type(got), np.float64)


def _random_cluster(rng, phase, n_sites, span, phased_fraction):
    """Random tags over both strands; some placed in a phased register."""
    start = 1000
    tags = []
    for idx in range(n_sites):
        if rng.random() < phased_fraction:
            k = rng.randint(0, span // phase)
            strand = rng.choice("wc")
            pos = start + k * phase + rng.choice([-1, 0, 0, 0, 1]) - (3 if strand == "c" else 0)
        else:
            strand = rng.choice("wc")
            pos = start + rng.randint(-10, span)
        length = rng.choice([phase, phase, phase, 20, 22, 23])
        abun = rng.choice([1, 1, 1, 2, 3, 5, 10, 50, 400])
        tags.append(["chr1", strand, rng.choice([1, 1, 2, 5]), f"SEQ{idx}", f"tag{idx}", pos, length, abun])
    tags.sort(key=lambda t: t[5])
    agg = {}
    for _chr, strand, _hits, _seq, _name, pos, length, abun in tags:
        ent = agg.setdefault((pos, 0 if strand == "w" else 1), [pos, strand, 0, 0, 0])
        ent[2] += 1
        ent[3] += abun
        ent[4] += abun if length == phase else 0
    poslist = [tuple(agg[key]) for key in sorted(agg)]
    return tags, poslist


class ClustScoreEquivalenceTests(unittest.TestCase):
    def _score(self, tags, poslist, sens, asens, legacy):
        env = {"Phasis_CLUSTER_SCORING_LEGACY_STATS": "1" if legacy else "0"}
        with mock.patch.dict(os.environ, env):
            return cluster_scoring.clustscore(copy.deepcopy(tags), poslist, sens, asens)

    def test_fast_clustscore_equals_legacy(self):
        rng = random.Random(6)
        for phase in (21, 24):
            with mock.patch.object(cluster_scoring, "phase", phase):
                sens, asens = cluster_scoring.getPhasedIndexes(cluster_scoring.WINDOW_SIZE)
                for _ in range(12):
                    n_sites = rng.choice([10, 25, 60, 150])
                    span = rng.choice([100, 400, 1200])
                    tags, poslist = _random_cluster(rng, phase, n_sites, span, rng.choice([0.0, 0.5, 0.9]))
                    with np.errstate(all="ignore"):
                        legacy = self._score(tags, poslist, sens, asens, legacy=True)
                        fast = self._score(tags, poslist, sens, asens, legacy=False)
                    self.assertEqual(len(legacy), len(fast))
                    for row_l, row_f in zip(legacy, fast):
                        self.assertEqual(row_l[:8], row_f[:8])
                        self.assertEqual(len(row_l), len(row_f))
                        for a, b in zip(row_l[8:], row_f[8:]):
                            self.assertTrue(_same_float(a, b), msg=f"{row_l} != {row_f}")
                            self.assertEqual(str(a), str(b))


if __name__ == "__main__":
    unittest.main()
