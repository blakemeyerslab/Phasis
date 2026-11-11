#!/usr/bin/env python3
"""
Validate and time the fast cluster-scoring statistics against the legacy
SciPy-per-position path, using an existing scored cluster file
(``<lib>-<chunk>.sRNA_<phase>.cluster``) as input.

The scored file holds every tag of every scored cluster, so the scoring inputs
can be rebuilt from it.  For each cluster the script runs the legacy and the
fast scorer and checks that the ten statistics per tag are identical to each
other and to the values stored in the file.

Usage:
    validate_fast_cluster_scoring.py FILE.cluster --phase 24 [--max-clusters N]
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

from phasis.stages import cluster_scoring as cs


def read_clusters(path, max_clusters=None, max_positions=None):
    clusters = []
    cur = None
    npos = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur:
                    clusters.append(cur)
                    npos += len({(t[0][5], t[0][1]) for t in cur})
                    if max_clusters and len(clusters) >= max_clusters:
                        cur = None
                        break
                    if max_positions and npos >= max_positions:
                        cur = None
                        break
                cur = []
                continue
            f = line.rstrip("\n").split("\t")
            achr, astrand, apos, alen, ahits, abun = f[0], f[1], int(f[2]), int(f[3]), int(f[4]), int(f[5])
            stored = f[6:16]
            aname, atag = f[16], f[17]
            cur.append(([achr, astrand, ahits, atag, aname, apos, alen, abun], stored))
    if cur:
        clusters.append(cur)
    return clusters


def build_poslist(tags, phase):
    agg = {}
    for (taginfo, _stored) in tags:
        _chr, strand, _hits, _tag, _name, apos, alen, abun = taginfo
        ent = agg.setdefault((apos, 0 if strand == "w" else 1), [apos, strand, 0, 0, 0])
        ent[2] += 1
        ent[3] += abun
        ent[4] += abun if alen == phase else 0
    return [tuple(agg[k]) for k in sorted(agg)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cluster_file")
    ap.add_argument("--phase", type=int, required=True)
    ap.add_argument("--max-clusters", type=int, default=None)
    ap.add_argument("--max-positions", type=int, default=None)
    ap.add_argument("--skip-legacy", action="store_true", help="time the fast path only; compare to stored values")
    args = ap.parse_args()

    cs.phase = args.phase
    sens, asens = cs.getPhasedIndexes(cs.WINDOW_SIZE)
    clusters = read_clusters(args.cluster_file, args.max_clusters, args.max_positions)
    inputs = []
    npos = 0
    for tags in clusters:
        poslist = build_poslist(tags, args.phase)
        npos += len(poslist)
        inputs.append(([t[0] for t in tags], poslist, [t[1] for t in tags]))
    ntags = sum(len(i[0]) for i in inputs)
    print(f"file: {args.cluster_file}")
    print(f"clusters: {len(inputs)}  occupied positions: {npos}  tags: {ntags}")

    def run(legacy):
        os.environ["Phasis_CLUSTER_SCORING_LEGACY_STATS"] = "1" if legacy else "0"
        out = []
        t0 = time.process_time()
        for aclust, poslist, _stored in inputs:
            out.append(cs.clustscore(copy.deepcopy(aclust), poslist, sens, asens))
        return out, time.process_time() - t0

    fast, t_fast = run(False)
    print(f"fast:   {t_fast:9.2f} CPU-s  {1e6 * t_fast / npos:9.1f} us/position")
    legacy = None
    if not args.skip_legacy:
        legacy, t_leg = run(True)
        print(f"legacy: {t_leg:9.2f} CPU-s  {1e6 * t_leg / npos:9.1f} us/position")
        print(f"speed-up: {t_leg / t_fast:.1f}x")

    bad_legacy = bad_stored = bad_legacy_stored = 0
    for ci, (aclust, poslist, stored) in enumerate(inputs):
        for ti in range(len(aclust)):
            f_vals = fast[ci][ti][8:]
            f_str = [str(v) for v in f_vals]
            if f_str != stored[ti]:
                bad_stored += 1
                if bad_stored <= 5:
                    print("  fast != stored :", fast[ci][ti][:6], f_str, stored[ti])
            if legacy is not None:
                l_vals = legacy[ci][ti][8:]
                l_str = [str(v) for v in l_vals]
                if l_str != f_str or any(repr(float(a)) != repr(float(b)) for a, b in zip(l_vals, f_vals)):
                    bad_legacy += 1
                    if bad_legacy <= 5:
                        print("  fast != legacy :", fast[ci][ti][:6], f_str, l_str)
                if l_str != stored[ti]:
                    bad_legacy_stored += 1
    print(f"tags where fast differs from stored file values:   {bad_stored} / {ntags}")
    if legacy is not None:
        print(f"tags where fast differs from legacy (this run):    {bad_legacy} / {ntags}")
        print(f"tags where legacy differs from stored file values: {bad_legacy_stored} / {ntags}")
    return 1 if (bad_stored or bad_legacy) else 0


if __name__ == "__main__":
    sys.exit(main())
