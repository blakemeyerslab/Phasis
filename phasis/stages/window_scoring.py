from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues

import phasis.runtime as rt
from phasis.cache import MemCache, default_memfile_path, phase2_basename, stage_signature
from phasis.parallel import run_parallel_with_progress

from .. import state as st

WINDOW_MULTIPLIER = 10  # 10 cycles
WIN_SCORE_LOOKUP = st.WIN_SCORE_LOOKUP


def fishers(pvals):
    """
    Combine p-values using Fisher's method with defensive clipping.
    """
    if pvals is None:
        return 1.0
    arr = np.asarray(list(pvals), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    tiny = 1e-300
    arr = np.clip(arr, tiny, 1.0)
    _, p = combine_pvalues(arr, method="fisher", weights=None)
    return float(p)


def _record_clusters_scored_tsv_path(path: str) -> None:
    """
    Persist scored TSV path into runtime + snapshot (spawn-safe).
    """
    try:
        if not path:
            return
        p = os.path.abspath(os.path.expanduser(path))
        rt.clusters_scored_tsv = p
        snap = getattr(rt, "runtime_snapshot", None)
        if hasattr(rt, "save_snapshot"):
            rt.save_snapshot(snap)
    except Exception:
        return


def infer_library_from_cluster_id(cid: str, phase_value: int) -> str:
    """
    Infer library prefix from cluster_id by splitting on '{phase}-PHAS' (or swap-phase tag).
    This avoids the common trap where '-' appears only inside '21-PHAS'.
    """
    s = str(cid)

    tag_main = f"{phase_value}-PHAS"
    swap_phase = 21 if phase_value == 24 else 24 if phase_value == 21 else phase_value
    tag_alt = f"{swap_phase}-PHAS"

    if tag_main in s:
        pref = s.split(tag_main)[0]
    elif tag_alt in s:
        pref = s.split(tag_alt)[0]
    else:
        return "UNKNOWN"

    return pref.rstrip(".-_")


def infer_chromosome_from_cluster_id(cid: str):
    """
    Infer chromosome from cluster_id (last '_' chunk). Falls back to string.
    """
    s = str(cid)
    if "_" not in s:
        return "NA"
    last = s.rsplit("_", 1)[-1]
    try:
        return int(last)
    except Exception:
        return last


def compute_scores_for_group(group_payload):
    """
    Worker: ((chromosome, library), rows_list) -> list rows [cID, phasis_score, combined_fishers]
    """
    (chromosome, lib), data_list = group_payload
    if not data_list:
        return []

    cols = [
        "cluster_id",
        "window_n",
        "fw_pval_corr",
        "rv_pval_corr",
        "combined_window_p_value",
    ]
    df = pd.DataFrame(data_list, columns=cols)

    for c in ("window_n", "fw_pval_corr", "rv_pval_corr", "combined_window_p_value"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out = []
    for cID, aclust in df.groupby("cluster_id", sort=False):
        vals = aclust["combined_window_p_value"].dropna()
        if vals.empty:
            combined_fishers = 1.0
            phasis_score = 0.0
        else:
            combined_fishers = fishers(vals.tolist())
            if not np.isfinite(combined_fishers) or combined_fishers <= 0.0:
                combined_fishers = max(
                    float(combined_fishers) if np.isfinite(combined_fishers) else 0.0, 1e-300
                )
            phasis_score = -np.log10(combined_fishers)
            if not np.isfinite(phasis_score):
                phasis_score = 300.0

        if phasis_score > 300.0:
            phasis_score = 300.0

        out.append([str(cID), float(phasis_score), float(combined_fishers)])

    return out


def compute_and_save_phasis_scores(clusters: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Phasis scores per (chromosome, library) group in parallel.
    Reads/writes {phase}_clusters_scored.tsv and uses centralized MemCache.

    Input expectation (from window_selection):
      columns: cluster_id, window_n, fw_pval_corr, rv_pval_corr, combined_window_p_value
      (chromosome/alib are NOT required; we infer from cluster_id)
    """
    print("### Step: Compute Phasis scores per (chromosome, library) ###")

    phase = getattr(rt, "phase", None)
    memFile = getattr(rt, "memFile", None) or default_memfile_path()
    outfname = phase2_basename("clusters_scored.tsv")
    windows_path = phase2_basename("clusters_windows_to_score.tsv")

    cache = MemCache.load(memFile)

    # Stable signature: depends only on upstream windows TSV + phase
    input_sig = stage_signature(files=[windows_path], params={"phase": phase})

    if cache.hit("CLUSTERS_SCORED", outfname, input_sig):
        print(f"  - Output up-to-date (hash+sig match). Skipping computation: {outfname}")
        df_cached = pd.read_csv(outfname, sep="\t")
        for c in ("phasis_score", "combined_fishers"):
            if c in df_cached.columns:
                df_cached[c] = pd.to_numeric(df_cached[c], errors="coerce").fillna(0.0)
        _record_clusters_scored_tsv_path(outfname)
        return df_cached

    # Guard empty input
    if clusters is None or getattr(clusters, "empty", True):
        print("[INFO] compute_and_save_phasis_scores: empty input; writing empty file.")
        empty = pd.DataFrame(columns=["cID", "phasis_score", "combined_fishers"])
        empty.to_csv(outfname, sep="\t", index=False)
        cache.record("CLUSTERS_SCORED", outfname, input_sig)
        _record_clusters_scored_tsv_path(outfname)
        return empty

    clusters = clusters.copy()

    # Normalize cluster_id column name
    if "cluster_id" not in clusters.columns:
        if "clusterID" in clusters.columns:
            clusters = clusters.rename(columns={"clusterID": "cluster_id"})
        else:
            raise KeyError("compute_and_save_phasis_scores: input must contain 'cluster_id' column")

    # Ensure required numeric columns exist
    required = ["window_n", "fw_pval_corr", "rv_pval_corr", "combined_window_p_value"]
    missing = [c for c in required if c not in clusters.columns]
    if missing:
        raise ValueError(f"compute_and_save_phasis_scores: missing required columns: {missing}")

    # Infer chromosome + library from cluster_id (window_selection output does not include them)
    phase_value = int(phase) if phase is not None else 21
    clusters["chromosome"] = [infer_chromosome_from_cluster_id(x) for x in clusters["cluster_id"].tolist()]
    clusters["library"] = [infer_library_from_cluster_id(x, phase_value) for x in clusters["cluster_id"].tolist()]

    # Form groups
    groups = []
    for (chrom, lib), df in clusters.groupby(["chromosome", "library"], sort=False, dropna=False):
        rows = df[
            ["cluster_id", "window_n", "fw_pval_corr", "rv_pval_corr", "combined_window_p_value"]
        ].values.tolist()
        groups.append(((chrom, lib), rows))

    print(f"  - Found {len(groups)} (chromosome, library) groups")
    if not groups:
        print("[INFO] No groups formed; writing empty scored TSV.")
        empty = pd.DataFrame(columns=["cID", "phasis_score", "combined_fishers"])
        empty.to_csv(outfname, sep="\t", index=False)
        cache.record("CLUSTERS_SCORED", outfname, input_sig)
        _record_clusters_scored_tsv_path(outfname)
        return empty

    preferred_start = getattr(rt, "mp_start_method", None) or os.environ.get("PHASIS_MP_START_METHOD")
    if preferred_start is None and sys.platform != "darwin":
        preferred_start = "forkserver"

    results = run_parallel_with_progress(
        compute_scores_for_group,
        groups,
        desc="Scoring windows via Fisher's method",
        min_chunk=1,
        unit="lib-chr",
        start_method=preferred_start,
        kind="compute",
    )

    flat = [item for sub in (results or []) for item in (sub or [])]
    win_phasis_score = pd.DataFrame(flat, columns=["cID", "phasis_score", "combined_fishers"])

    win_phasis_score.to_csv(outfname, sep="\t", index=False)
    fp = cache.record("CLUSTERS_SCORED", outfname, input_sig)
    if fp:
        print(f"  - Wrote {outfname} (md5: {fp})")
    else:
        print(f"  - Wrote {outfname}")

    _record_clusters_scored_tsv_path(outfname)
    return win_phasis_score
