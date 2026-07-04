from __future__ import annotations

"""
phasis.stages.phas_clusters
---------------------------

Phase II stage: build per-(chromosome, library) PHAS cluster rows and write
{phase}_PHAS_to_detect.tab (via phase2_basename("PHAS_to_detect.tab")).

Key requirements:
- spawn-safe (top-level functions only)
- no nested functions; no imports inside functions
- runtime-first (uses phasis.runtime for defaults), but allows explicit args
- minimal behavior drift vs legacy implementation
"""

import os
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from scipy.stats import combine_pvalues

import phasis.runtime as rt
from phasis.cache import (
    MemCache,
    artifact_exists,
    default_memfile_path,
    finalize_text_artifact,
    phase2_basename,
    resolve_artifact_path,
    stage_signature,
)
from phasis.parallel import run_parallel_with_progress
import phasis.ids as ids


# ---- required 20-col schema (ORDER MATTERS) ----
REQUIRED_20_COLS: List[str] = [
    "alib", "clusterID", "chromosome", "strand", "pos", "len", "hits", "abun",
    "pval_h_f", "N_f", "X_f", "pval_r_f", "pval_corr_f",
    "pval_h_r", "N_r", "X_r", "pval_r_r", "pval_corr_r",
    "tag_id", "tag_seq",
]
REQUIRED_20_SET = set(REQUIRED_20_COLS)


def load_processed_clusters_fallback() -> pd.DataFrame:
    """
    Load {phase}_processed_clusters.tab or return empty DF.

    NOTE: the actual filename comes from phasis.cache.phase2_basename, which uses rt.phase/rt.concat_libs.
    """
    proc_path = phase2_basename("processed_clusters.tab")
    if artifact_exists(proc_path):
        print(f"  - Detected non-20-col input; loading processed-clusters fallback: {proc_path}")
        physical_proc_path = resolve_artifact_path(proc_path) or proc_path
        try:
            return pd.read_csv(physical_proc_path, sep="\t", engine="python")
        except Exception:
            return pd.read_csv(physical_proc_path, sep="\t")
    print(f"[WARN] Processed-clusters fallback not found: {proc_path}")
    return pd.DataFrame()


def _coerce_numeric_allowlist(df: pd.DataFrame) -> pd.DataFrame:
    numeric_allowlist = {
        "pos", "len", "hits", "abun",
        "pval_h_f", "N_f", "X_f", "pval_r_f", "pval_corr_f",
        "pval_h_r", "N_r", "X_r", "pval_r_r", "pval_corr_r",
    }
    for col in numeric_allowlist.intersection(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _read_cached_phas_to_detect(output_file: str, cache: MemCache, input_sig: str | None) -> Optional[pd.DataFrame]:
    """Return cached PHAS_to_detect table if cache hit; else None."""
    section_name = "PHAS_TO_DETECT"
    if cache.hit(section_name, output_file, input_sig):
        print(f"  - Output up-to-date (hash+sig match). Skipping processing: {output_file}")
        physical_output_file = resolve_artifact_path(output_file) or output_file
        try:
            df = pd.read_csv(physical_output_file, sep="\t", engine="python")
        except Exception:
            df = pd.read_csv(physical_output_file, sep="\t")
        df = _coerce_numeric_allowlist(df)
        return df
    return None


def process_chromosome_data(loci_group: Sequence[Sequence]) -> pd.DataFrame:
    """
    Process data for a single chromosome-library group.

    STRICT: expects 20-column per-read/per-alignment rows with REQUIRED_20_COLS.
    Returns a dataframe with REQUIRED_20_COLS + ["identifier"].

    Any row that cannot be mapped clusterID -> universal "identifier" is dropped.
    """
    if not loci_group:
        return pd.DataFrame(columns=REQUIRED_20_COLS + ["identifier"])

    # Guard against accidental wrong payloads (e.g. 6-col merged-candidates)
    width = len(loci_group[0])
    if width != len(REQUIRED_20_COLS):
        return pd.DataFrame(columns=REQUIRED_20_COLS + ["identifier"])

    df = pd.DataFrame(loci_group, columns=REQUIRED_20_COLS)

    # light dtype normalization used later downstream
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
    df["len"] = pd.to_numeric(df["len"], errors="coerce")
    df["abun"] = pd.to_numeric(df["abun"], errors="coerce")
    df = df.dropna(subset=["pos", "len"]).reset_index(drop=True)

    # Attach universal identifier (ensure mergedClusterDict has been prepared earlier)
    df["identifier"] = df["clusterID"].astype(str).map(ids.getUniversalID)

    # Drop rows we can't map
    df = df.dropna(subset=["identifier"]).reset_index(drop=True)
    return df


def process_phas_cluster_group(group) -> pd.DataFrame:
    """
    Worker: ((chromosome, alib), loci_group-as-list) -> DataFrame
    Adds 'chromosome' and 'alib' columns to the processed DataFrame.
    """
    (chromosome, alib), loci_group = group
    processed_df = process_chromosome_data(loci_group)
    # Ensure these columns exist (even if empty), and are consistent for the group
    processed_df["chromosome"] = chromosome
    processed_df["alib"] = alib
    return processed_df


def fishers(pvals: Iterable[float]) -> float:
    """
    Combine p-values using Fisher's method.
    Returns the combined p-value.
    """
    apval = combine_pvalues(list(pvals), method="fisher", weights=None)
    return float(apval[1])


def build_and_save_phas_clusters(
    allClusters: Optional[pd.DataFrame],
    *,
    phase: Optional[int] = None,
    memFile: Optional[str] = None,
    concat_libs: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Build per-(chromosome, library) PHAS cluster rows in parallel and write to TSV.

    Skips recomputation if output exists and matches the hash stored in memFile.

    Robust to accidentally receiving the 6-col merged-candidates frame: falls back to
    {phase}_processed_clusters.tab (20-col per-read/per-alignment schema).

    Runtime-first:
      - prefers phasis.runtime for phase/memFile/concat_libs
      - allows explicit args if run_phase2() wants to pass them
    """
    print("### Step: Build PHAS clusters per (chromosome, library) — parallel ###")

    # Prefer explicit args; fall back to runtime snapshot
    phase_local = phase if phase is not None else getattr(rt, "phase", None)
    memfile_local = memFile if memFile is not None else (getattr(rt, "memFile", None) or default_memfile_path())
    concat_local = concat_libs if concat_libs is not None else getattr(rt, "concat_libs", False)

    output_file = phase2_basename("PHAS_to_detect.tab")

    cache = MemCache.load(memfile_local)

    # Signature from upstream Phase II inputs
    proc_path = phase2_basename("processed_clusters.tab")
    dict_tab = phase2_basename("mergedClusterDict.tab")
    input_sig = stage_signature(
        files=[proc_path, dict_tab],
        params={"phase": phase_local, "concat_libs": bool(concat_local)},
    )

    # ---- Early cache check ----
    cached = _read_cached_phas_to_detect(output_file, cache, input_sig)
    if cached is not None:
        return cached

    # ---- Accept only the 20-col per-read schema; else load the processed tab ----
    if not (isinstance(allClusters, pd.DataFrame) and REQUIRED_20_SET.issubset(set(allClusters.columns))):
        allClusters = load_processed_clusters_fallback()

    # ---- If still empty, bail cleanly ----
    if allClusters is None or getattr(allClusters, "empty", True):
        print("  - Found 0 (chromosome, library) groups (empty input). Returning empty DataFrame.")
        return pd.DataFrame(columns=REQUIRED_20_COLS + ["identifier"])

    # ---- Ensure grouping columns exist / normalize ----
    if "chromosome" not in allClusters.columns and "chr" in allClusters.columns:
        allClusters = allClusters.rename(columns={"chr": "chromosome"})

    if "alib" not in allClusters.columns:
        if bool(concat_local):
            allClusters = allClusters.copy()
            allClusters["alib"] = "ALL_LIBS"
        else:
            print("[WARN] 'alib' column missing and not in concat mode; returning empty DataFrame.")
            return pd.DataFrame(columns=REQUIRED_20_COLS + ["identifier"])

    # ---- Enforce EXACT 20-column payload (drop extras like 'identifier') ----
    if not REQUIRED_20_SET.issubset(set(allClusters.columns)):
        allClusters = load_processed_clusters_fallback()
        if allClusters is None or getattr(allClusters, "empty", True):
            print("  - Input invalid and fallback empty; returning empty DataFrame.")
            return pd.DataFrame(columns=REQUIRED_20_COLS + ["identifier"])

    allClusters = allClusters.loc[:, REQUIRED_20_COLS].copy()

    # ---- Ensure universal ID mapping is READY BEFORE spawning workers ----
    # (macOS spawn): each worker can re-load from mergedClusterDict.tab if needed,
    # but we want the parent to validate the mapping is non-empty to avoid silent empties.
    try:
        ids.ensure_mergedClusterDict(phase=str(phase_local) if phase_local is not None else None)
    except Exception:
        pass

    # Quick sanity check — if mapping fails completely, parallel work will be empty
    try:
        sample = allClusters["clusterID"].astype(str).head(50).tolist()
        ok = sum(1 for cid in sample if ids.getUniversalID(cid) is not None)
        if ok == 0 and sample:
            print(
                "[WARN] Universal ID mapping returned 0/50 hits in parent process. "
                "Workers will likely return empty. Check mergedClusterDict/reverse map wiring."
            )
    except Exception:
        pass

    # ---- Group → ((chromosome, library), loci_list) ----
    cluster_groups = [
        ((chromosome, alib), df.values.tolist())
        for (chromosome, alib), df in allClusters.groupby(["chromosome", "alib"], sort=False)
    ]
    print(f"  - Found {len(cluster_groups)} (chromosome, library) groups")

    if not cluster_groups:
        print("  - No groups to process. Returning empty DataFrame.")
        return pd.DataFrame(columns=REQUIRED_20_COLS + ["identifier"])

    processed_results = run_parallel_with_progress(
        process_phas_cluster_group,
        cluster_groups,
        desc="Building PHAS cluster groups",
        min_chunk=1,
        unit="lib-chr",
    )

    if not processed_results:
        print("  - Worker returned no results. Returning empty DataFrame.")
        return pd.DataFrame(columns=REQUIRED_20_COLS + ["identifier"])

    # Surface worker failures if they were wrapped (keep behavior: filter and continue)
    worker_errors = [r for r in processed_results if isinstance(r, RuntimeError)]
    if worker_errors:
        print("[WARN] One or more worker tasks failed; filtering to successful results. First error:")
        print(worker_errors[0])

    processed_frames = [r for r in processed_results if isinstance(r, pd.DataFrame) and not r.empty]
    if not processed_frames:
        print("  - All worker results empty. Returning empty DataFrame.")
        return pd.DataFrame(columns=REQUIRED_20_COLS + ["identifier"])

    clusters_data = pd.concat(processed_frames, ignore_index=True)

    # ---- Write + update md5 cache (best effort) ----
    clusters_data.to_csv(output_file, sep="\t", encoding="utf-8", index=False)
    fp = finalize_text_artifact(cache, "PHAS_TO_DETECT", output_file, input_sig)
    if fp:
        print(f"  - Wrote {output_file} (md5: {fp})")
    else:
        print(f"  - Wrote {output_file}")

    return clusters_data
