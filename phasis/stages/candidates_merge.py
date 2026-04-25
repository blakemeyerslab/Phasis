from __future__ import annotations

import os
from functools import partial

import pandas as pd

import phasis.runtime as rt
from phasis.cache import MemCache, default_memfile_path, phase2_basename, stage_signature
from phasis.parallel import run_parallel_with_progress


def chromosome_clusters_to_candidate_loci(
    chromosome_df: pd.DataFrame,
    *,
    minClusterLength: int | None = None,
):
    """Convert per-chromosome cluster rows into candidate loci rows.

    Returns list-of-lists rows in the legacy-compatible format:
        [clusterID, 0, chromosome, min_pos, max_pos]
    """
    mcl = int(getattr(rt, "minClusterLength", 0) or 0) if minClusterLength is None else int(minClusterLength)

    # Group by clusterID and calculate min/max positions
    cluster_positions = chromosome_df.groupby("clusterID")["pos"].agg(["min", "max"]).reset_index()

    # Merge to get chromosome for each clusterID (legacy behavior)
    cluster_info = chromosome_df.merge(cluster_positions, on="clusterID")

    # Clean up clusterID
    cluster_info["clusterID"] = cluster_info["clusterID"].astype(str).str.strip()

    # Mask for clusters longer than minClusterLength
    mask = (cluster_info["max"] - cluster_info["min"]) >= mcl
    sub = cluster_info.loc[mask, ["clusterID", "chromosome", "min", "max"]]

    lociTablelist = []
    for cid, achr, s, e in sub.itertuples(index=False, name=None):
        lociTablelist.append([
            str(cid).replace("\t", "").strip(),
            0,
            int(achr),
            int(s),
            int(e),
        ])

    return lociTablelist


def loci_table_from_clusters(
    allClusters: pd.DataFrame,
    *,
    memFile: str | None = None,
    minClusterLength: int | None = None,
    outfname: str | None = None,
) -> pd.DataFrame:
    """Build {phase}_candidate.loci_table.tab from {phase}_processed_clusters.tab.

    Cache correctness requirement:
      Reuse ONLY when both output fingerprint AND upstream-input signature match.
    """
    print("### Building loci table from clusters per chromosome ###")

    memFile_local = memFile or getattr(rt, "memFile", None) or default_memfile_path()
    outfname = outfname or phase2_basename("candidate.loci_table.tab")

    # Resolve minClusterLength early for both signature and computation
    mcl = int(getattr(rt, "minClusterLength", 0) or 0) if minClusterLength is None else int(minClusterLength)

    cache = MemCache.load(memFile_local)

    # Inputs that should invalidate loci-table cache when they change.
    processed_clusters_path = phase2_basename("processed_clusters.tab")
    input_sig = stage_signature(
        files=[processed_clusters_path],
        params={
            "phase": getattr(rt, "phase", None),
            "minClusterLength": mcl,
        },
    )

    if cache.hit("LOCI_TABLE", outfname, input_sig):
        print(f"File {outfname} is up-to-date (hash+sig match). Skipping recomputation.")
        print(f"Loci table written to {outfname}")
        with open(outfname, "r") as fh:
            file_lines = fh.readlines()[1:]  # skip header
            lociTablelist_unique = [ln.strip().split("	") for ln in file_lines]
        return pd.DataFrame(lociTablelist_unique, columns=["name", "pval", "chr", "start", "end"])

    worker_fn = partial(chromosome_clusters_to_candidate_loci, minClusterLength=mcl)

    chromosome_groups = [df for _, df in allClusters.groupby("chromosome")]

    lociTablelist = run_parallel_with_progress(
        worker_fn,
        chromosome_groups,
        desc="LociTable chromosomes",
        min_chunk=1,
        unit="lib-chr",
    )

    lociTablelist = [item for sublist in lociTablelist for item in sublist]

    seen = set()
    lociTablelist_unique = []
    for item in lociTablelist:
        row_tuple = tuple(item)
        if row_tuple not in seen:
            seen.add(row_tuple)
            lociTablelist_unique.append(item)

    with open(outfname, "w") as fh:
        fh.write("Cluster\tvalue1\tchromosome\tStart\tEnd\n")
        for row in lociTablelist_unique:
            fh.write("\t".join(map(str, row)) + "\n")

    if os.path.isfile(outfname):
        fp = cache.record("LOCI_TABLE", outfname, input_sig)
        if fp:
            print(f"Hash for {outfname}: {fp}")

    print(f"Loci table written to {outfname}")
    return pd.DataFrame(lociTablelist_unique, columns=["name", "pval", "chr", "start", "end"])


def merge_candidate_clusters_across_libs(
    loci_table_path: str,
    out_path: str,
    *,
    memFile: str | None = None,
    concat_libs: bool | None = None,
) -> pd.DataFrame:
    """Merge candidate loci across libs.

    On cache hit, load+return cached file.
    """
    print("### Merging candidate clusters across libraries (per chromosome) ###")

    memFile_local = memFile or getattr(rt, "memFile", None) or default_memfile_path()
    concat_local = bool(getattr(rt, "concat_libs", False)) if concat_libs is None else bool(concat_libs)

    cache = MemCache.load(memFile_local)
    input_sig = stage_signature(
        files=[loci_table_path],
        params={
            "phase": getattr(rt, "phase", None),
            "concat_libs": bool(concat_local),
        },
    )

    if cache.hit("MERGED_CANDIDATES", out_path, input_sig):
        print("Outputs up-to-date (hash+sig match). Skipping merge computation.")
        df_cached = pd.read_csv(out_path, sep="	")

        if "chromosome" not in df_cached.columns and "chr" in df_cached.columns:
            df_cached = df_cached.rename(columns={"chr": "chromosome"})
        if "alib" not in df_cached.columns and concat_local:
            df_cached["alib"] = "ALL_LIBS"
        if "alib" not in df_cached.columns and not concat_local:
            print("[WARN] Cached merged table lacks 'alib' in non-concat mode.")
        return df_cached

    if not os.path.isfile(loci_table_path):
        print(f"[WARN] Loci table not found: {loci_table_path}. Returning empty DataFrame.")
        return pd.DataFrame()

    merged_df = pd.read_csv(loci_table_path, sep="\t")

    if "chromosome" not in merged_df.columns and "chr" in merged_df.columns:
        merged_df = merged_df.rename(columns={"chr": "chromosome"})

    if "alib" not in merged_df.columns:
        if concat_local:
            merged_df["alib"] = "ALL_LIBS"
        else:
            print("[WARN] 'alib' missing in loci table on non-concat run; setting 'alib'='UNKNOWN'.")
            merged_df["alib"] = "UNKNOWN"

    merged_df.to_csv(out_path, sep="	", index=False)
    fp = cache.record("MERGED_CANDIDATES", out_path, input_sig)
    if fp:
        print(f"Hash for {os.path.basename(out_path)}: {fp}")

    return merged_df
