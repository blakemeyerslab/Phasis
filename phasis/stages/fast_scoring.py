"""
Fast, value-preserving replacements for the per-position statistics used by
``cluster_scoring.clustscore``.

The legacy code calls ``scipy.stats.hypergeom.sf``, ``scipy.stats.mannwhitneyu``
and ``scipy.stats.combine_pvalues`` twice for every occupied position.  On the
tiny inputs seen here almost all of that time is SciPy call overhead (input
validation, array-API dispatch, decorators), not arithmetic.  This module
computes the same numbers with:

* a lookup table for the hypergeometric tail (it depends on two small integers);
* a direct Mann-Whitney U implementation that follows SciPy's own algorithm
  (average ranks, tie-corrected normal approximation with continuity
  correction, or the exact null distribution for small tie-free samples);
* the closed form of Stouffer's method for two p-values;
* a window-bounded labelling pass (the legacy pass scanned the whole cluster).

Floating-point operations are kept in the same order as SciPy's so results are
bit-identical to the legacy path with the SciPy version this was validated on
(1.17.1).  ``tests/test_fast_scoring.py`` checks this.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from itertools import accumulate
from math import sqrt

import numpy as np
from scipy.special import binom as _binom
from scipy.special import ndtr as _ndtr
from scipy.special import ndtri as _ndtri
from scipy.stats import hypergeom as _hypergeom
from scipy.stats import mannwhitneyu as _scipy_mannwhitneyu

_SQRT2 = float(np.linalg.norm(np.ones(2)))
_UINT64_MAX = int(np.iinfo(np.uint64).max)
_f64 = np.float64

# (X, M, n, N) -> rounded hypergeometric tail
_HYPER_CACHE: dict = {}
# (n_small, n_large) -> (pmf list, cdf list) for u = 0 .. n_small*n_large//2
_MWU_EXACT_CACHE: dict = {}
_MWU_EXACT_CACHE_MAX = 4096


# --------------------------------------------------------------------------- #
# Hypergeometric tail
# --------------------------------------------------------------------------- #
def hyper_sf_rounded(X: int, M: int, n: int, N: int):
    """``round(hypergeom.sf(X - 1, M, n, N), 10)``, memoised.

    On a miss the whole row X = 0 .. n is filled with one vectorised call.
    """
    key = (X, M, n, N)
    val = _HYPER_CACHE.get(key)
    if val is None:
        if 0 <= X <= n:
            row = np.round(_hypergeom.sf(np.arange(-1, n), M, n, N), 10)
            for x, v in enumerate(row):
                _HYPER_CACHE[(x, M, n, N)] = v
            val = _HYPER_CACHE[key]
        else:
            val = round(_hypergeom.sf(X - 1, M, n, N), 10)
            _HYPER_CACHE[key] = val
    return val


# --------------------------------------------------------------------------- #
# Mann-Whitney U, alternative='greater'
# --------------------------------------------------------------------------- #
def _mwu_exact_tables(n1: int, n2: int):
    """
    Null distribution of U for tie-free samples of sizes ``n1`` and ``n2``.

    Returns ``(pmf, cdf)`` as lists of float64 for u = 0 .. n1*n2//2, built with
    the same arithmetic SciPy uses (integer counts / binom, sequential cumsum),
    or ``None`` when the counts would overflow uint64 (SciPy then switches to
    floats; the caller falls back to SciPy).
    """
    small, large = (n1, n2) if n1 <= n2 else (n2, n1)
    key = (small, large)
    hit = _MWU_EXACT_CACHE.get(key)
    if hit is not None:
        return hit

    total_int = 1
    for i in range(1, small + 1):
        total_int = total_int * (large + i) // i
    if total_int > _UINT64_MAX:
        return None

    maxu = (small * large) // 2
    # Coefficients of the Gaussian binomial [small+large choose small]_q,
    # truncated at degree maxu: prod_i (1 - q^(large+i)) / (1 - q^i).
    coef = [0] * (maxu + 1)
    coef[0] = 1
    for i in range(1, small + 1):
        d = large + i
        if d <= maxu:
            for u in range(maxu, d - 1, -1):
                coef[u] -= coef[u - d]
        for u in range(i, maxu + 1):
            coef[u] += coef[u - i]

    total = _binom(small + large, small)
    pmf = [_f64(c) / total for c in coef]
    cdf = list(accumulate(pmf))
    if len(_MWU_EXACT_CACHE) >= _MWU_EXACT_CACHE_MAX:
        _MWU_EXACT_CACHE.clear()
    _MWU_EXACT_CACHE[key] = (pmf, cdf)
    return pmf, cdf


def mannwhitneyu_greater_pvalue(x, y):
    """
    p-value of ``scipy.stats.mannwhitneyu(x, y, alternative='greater')`` for
    non-empty lists of integers, with SciPy's default ``method='auto'``.
    """
    n1 = len(x)
    n2 = len(y)

    counts = Counter(y)
    xcounts = Counter(x)
    counts.update(xcounts)

    # Sum of average ranks of x, and the tie term sum(t^3 - t).
    r1 = 0.0
    tie_term = 0
    seen = 0
    get_x = xcounts.get
    for v in sorted(counts):
        t = counts[v]
        cx = get_x(v)
        if cx:
            r1 += cx * (seen + (t + 1) / 2)
        if t > 1:
            tie_term += t * t * t - t
        seen += t
    ties = tie_term > 0

    u1 = r1 - n1 * (n1 + 1) / 2

    if ties or (n1 > 8 and n2 > 8):
        # Normal approximation with tie and continuity correction.
        n = n1 + n2
        s = sqrt(n1 * n2 / 12 * ((n + 1) - float(tie_term) / (n * (n - 1))))
        if s == 0.0:
            return _scipy_mannwhitneyu(x, y, alternative="greater")[1]
        z = ((u1 - n1 * n2 / 2) - 0.5) / s
        p = _ndtr(-z)
    else:
        tables = _mwu_exact_tables(n1, n2)
        if tables is None:
            return _scipy_mannwhitneyu(x, y, alternative="greater")[1]
        pmf, cdf = tables
        k = int(u1)
        kc = n1 * n2 - k
        if k < kc:
            p = 1.0 - cdf[k] + pmf[k]
        else:
            p = cdf[kc]

    if p < 0.0:
        p = _f64(0.0)
    elif p > 1.0:
        p = _f64(1.0)
    return p


def ranksum_rounded(abun_p_all, abun_np_all):
    """Legacy ``ranksumtest`` semantics on the two abundance lists."""
    if not abun_p_all:
        abun_p_all = [0]
    if not abun_np_all:
        abun_np_all = [0]
    first = abun_p_all[0]
    same = True
    for v in abun_p_all:
        if v != first:
            same = False
            break
    if same:
        for v in abun_np_all:
            if v != first:
                same = False
                break
    if same:
        return 1.0
    return round(_f64(mannwhitneyu_greater_pvalue(abun_p_all, abun_np_all)), 10)


# --------------------------------------------------------------------------- #
# Stouffer for two p-values
# --------------------------------------------------------------------------- #
def stouffer2_rounded(p1, p2):
    """``round(combine_pvalues([p1, p2], method='stouffer').pvalue, 15)``."""
    stat = ((-_ndtri(p1)) + (-_ndtri(p2))) / _SQRT2
    return round(_f64(_ndtr(-stat)), 15)


# --------------------------------------------------------------------------- #
# Cluster scoring
# --------------------------------------------------------------------------- #
def position_stats(poslist, sens, window_size: int, phase: int):
    """
    Per-position statistics for one cluster.

    ``poslist`` is the legacy sorted list of
    ``(apos, strand, poscount, abun_all, abun_phase)``.  Returns a dict
    ``apos -> (pval_h_f, N_f, X_f, pval_r_f, pval_corr_f,
               pval_h_r, N_r, X_r, pval_r_r, pval_corr_r)``
    equal to what the legacy ``clustscore`` loop builds.
    """
    sens_set = frozenset(sens)
    max_off = max(sens)
    M = window_size * int(phase) * 2
    K = window_size * 2 * 3

    positions = [ent[0] for ent in poslist]
    strands = [ent[1] for ent in poslist]
    abuns = [int(ent[3]) for ent in poslist]
    npos = len(poslist)
    pos_dict = {}

    for ind in range(npos):
        refpos = positions[ind]
        ref_w = strands[ind] == "w"

        # ---------------- forward: poslist[ind:], window-bounded ------------
        # Phased offsets (bpos - refpos) are in `sens` on the reference strand
        # and shifted by 3 nt on the opposite strand.
        if ref_w:
            hi = bisect_right(positions, refpos + max_off, ind, npos)
            w_shift, c_shift = 0, 3
        else:
            hi = bisect_right(positions, refpos + max_off + 3, ind, npos)
            w_shift, c_shift = -3, 0
        abun_p = []
        abun_np = []
        for j in range(ind, hi):
            d = positions[j] - refpos
            st = strands[j]
            if (st == "w" and (d + w_shift) in sens_set) or (st == "c" and (d + c_shift) in sens_set):
                abun_p.append(abuns[j])
            else:
                abun_np.append(abuns[j])
        X_f = len(abun_p)
        N_f = X_f + len(abun_np)
        pval_h_f = hyper_sf_rounded(X_f, M, K, N_f)
        pval_r_f = ranksum_rounded(abun_p, abun_np)
        pval_corr_f = stouffer2_rounded(pval_h_f, pval_r_f)

        # ---------------- reverse: poslist[:ind+1], window-bounded ----------
        if ref_w:
            lo = bisect_left(positions, refpos - max_off - 3, 0, ind + 1)
            w_shift, c_shift = 0, -3
        else:
            lo = bisect_left(positions, refpos - max_off, 0, ind + 1)
            w_shift, c_shift = 3, 0
        abun_p = []
        abun_np = []
        for j in range(lo, ind + 1):
            d = refpos - positions[j]
            st = strands[j]
            if (st == "w" and (d + w_shift) in sens_set) or (st == "c" and (d + c_shift) in sens_set):
                abun_p.append(abuns[j])
            else:
                abun_np.append(abuns[j])
        X_r = len(abun_p)
        N_r = X_r + len(abun_np)
        pval_h_r = hyper_sf_rounded(X_r, M, K, N_r)
        pval_r_r = ranksum_rounded(abun_p, abun_np)
        pval_corr_r = stouffer2_rounded(pval_h_r, pval_r_r)

        pos_dict[int(refpos)] = (
            pval_h_f, N_f, X_f, pval_r_f, pval_corr_f,
            pval_h_r, N_r, X_r, pval_r_r, pval_corr_r,
        )

    return pos_dict
