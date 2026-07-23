from __future__ import annotations

"""
phasis.stages.cluster_aggregation
--------------------------------

Phase II helper stage: read one or more *.PHAS.candidate.clusters files and
write {phase}_processed_clusters.tab.

Design constraints:
- spawn-safe (top-level functions only)
- no imports inside functions
- minimal behavior drift vs legacy implementation
"""

import os
import re
from typing import List, Sequence, Tuple

import pandas as pd

import phasis.runtime as rt
from phasis.cache import (
    MemCache,
    artifact_exists,
    default_memfile_path,
    finalize_text_artifact,
    open_text_artifact,
    phase2_basename,
    resolve_artifact_path,
    stage_signature,
)
from phasis.parallel import run_parallel_with_progress


# Canonical column order for processed cluster rows
PROCESSED_CLUSTER_COLUMNS: List[str] = [
    "alib",
    "clusterID",
    "chromosome",
    "strand",
    "pos",
    "len",
    "hits",
    "abun",
    "pval_h_f",
    "N_f",
    "X_f",
    "pval_r_f",
    "pval_corr_f",
    "pval_h_r",
    "N_r",
    "X_r",
    "pval_r_r",
    "pval_corr_r",
    "tag_id",
    "tag_seq",
]


def process_single_lib_cluster(filename: str) -> List[Tuple]:
    """
    Parse a single *.PHAS.candidate.clusters file into a list of tuples matching
    PROCESSED_CLUSTER_COLUMNS.
    """
    if not artifact_exists(filename):
        raise FileNotFoundError(f"Cluster file not found: {filename}")

    clustlist: List[Tuple] = []

    # library name from file basename:
    # AR_1_nocontam.21-PHAS.candidate.clusters -> AR_1_nocontam
    base = os.path.basename(filename)
    alib = re.sub(r"\.\d+-PHAS\.candidate\.clusters$", "", base)

    with open_text_artifact(filename, "rt") as fh:
        lines = fh.readlines()

    aid = None
    for line in lines:
        if line.startswith(">"):
            # header like: ">cluster = lobe_3_nocontam-1_3894_1"
            m = re.search(r"cluster\s*=\s*([^\s]+)", line)
            if not m:
                aid = None
                continue
            aid = m.group(1).strip()
            continue

        if not aid:
            continue

        ent = line.rstrip("\n").split("\t")
        if len(ent) < 18:
            continue

        achr = str(ent[0])
        astrand = str(ent[1])
        apos = int(ent[2])
        alen = int(ent[3])
        ahits = int(ent[4])
        abun = int(ent[5])
        pval_h_f = float(ent[6])
        N_f = int(ent[7])
        X_f = int(ent[8])
        pval_r_f = float(ent[9])
        pval_corr_f = float(ent[10])
        pval_h_r = float(ent[11])
        N_r = int(ent[12])
        X_r = int(ent[13])
        pval_r_r = float(ent[14])
        pval_corr_r = float(ent[15])
        tag_id = str(ent[16])
        tag_seq = str(ent[17])

        clustlist.append(
            (
                alib,
                aid,
                achr,
                astrand,
                apos,
                alen,
                ahits,
                abun,
                pval_h_f,
                N_f,
                X_f,
                pval_r_f,
                pval_corr_f,
                pval_h_r,
                N_r,
                X_r,
                pval_r_r,
                pval_corr_r,
                tag_id,
                tag_seq,
            )
        )

    return clustlist


def _coerce_paths(clusterFiles: Sequence[str] | str) -> List[str]:
    if isinstance(clusterFiles, str):
        return [clusterFiles]
    return [str(x) for x in clusterFiles if str(x).strip()]


def _raise_on_parallel_errors(results: List[object]) -> None:
    errs = [x for x in results if isinstance(x, RuntimeError)]
    if errs:
        raise errs[0]


def aggregate_and_write_processed_clusters(
    clusterFiles: Sequence[str] | str,
    *,
    memFile: str | None = None,
) -> pd.DataFrame:
    """
    Aggregate candidate cluster files and write {phase}_processed_clusters.tab.

    Returns: allClusters dataframe (sorted by clusterID, pos)
    """
    print("### Aggregating and processing candidate cluster files per library ###")

    paths = _coerce_paths(clusterFiles)
    if not paths:
        raise ValueError("No cluster files provided to aggregate_and_write_processed_clusters().")

    missing = [p for p in paths if not artifact_exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing cluster file(s): {missing}")

    # Stable ordering for signatures and processing
    paths = sorted(paths, key=os.path.basename)

    outfname = phase2_basename("processed_clusters.tab")
    mem_path = memFile or getattr(rt, "memFile", None) or default_memfile_path()

    cache = MemCache.load(str(mem_path))
    input_sig = stage_signature(
        files=paths,
        params={
            "phase": getattr(rt, "phase", None),
            "concat_libs": bool(getattr(rt, "concat_libs", False)),
        },
    )

    if cache.hit("PROCESSED", outfname, input_sig):
        print(f"  - Output up-to-date (hash+sig match). Skipping aggregation: {outfname}")
        df_cached = pd.read_csv(resolve_artifact_path(outfname) or outfname, sep="\t")
        print(f"Processed clusters written to {outfname}")
        return df_cached

    all_clustlists = run_parallel_with_progress(
        process_single_lib_cluster,
        paths,
        desc="Aggregating cluster files",
        min_chunk=1,
        unit="lib",
    )
    _raise_on_parallel_errors(all_clustlists)

    parsed_row_count = sum(len(sublist) for sublist in all_clustlists)
    print(
        "[INFO] Consolidating "
        f"{parsed_row_count:,} candidate-cluster records into one table and sorting "
        "by cluster and genomic position. This uses one CPU core and may take several "
        "minutes for large runs.",
        flush=True,
    )
    flat_clustlist = [item for sublist in all_clustlists for item in sublist]

    allClusters = pd.DataFrame(flat_clustlist, columns=PROCESSED_CLUSTER_COLUMNS)
    allClusters = allClusters.sort_values(by=["clusterID", "pos"])

    print(f"[INFO] Writing consolidated processed clusters: {outfname}", flush=True)
    allClusters.to_csv(outfname, sep="\t", index=False, header=True)

    fp = finalize_text_artifact(cache, "PROCESSED", outfname, input_sig)
    if fp:
        print(f"Hash for {outfname}: {fp}")

    print(f"Processed clusters written to {outfname}")
    return allClusters
