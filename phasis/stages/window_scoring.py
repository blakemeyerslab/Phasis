from __future__ import annotations

import os
import sys
import configparser

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues

import phasis.runtime as rt
from phasis.parallel import run_parallel_with_progress
from phasis.cache import getmd5, phase2_basename, MEM_FILE_DEFAULT, compute_cache_signature, sig_key

from .. import state as st

# Keep same constant as legacy
WINDOW_MULTIPLIER = 10  # 10 cycles

# Optional alias (nice for consistency/debug)
WIN_SCORE_LOOKUP = st.WIN_SCORE_LOOKUP


def fishers(pvals):
    """
    Combine p-values using Fisher's method.

    Notes
    -----
    SciPy's combine_pvalues computes log(p). If any p==0, you'll see
    RuntimeWarning: divide by zero encountered in log. That is mathematically
    fine (it implies an extremely significant combined p-value), but it is
    noisy and can trigger numerical edge cases on some platforms.

    We defensively clip p-values into (0, 1] before combining.
    """
    if pvals is None:
        return 1.0

    arr = np.asarray(list(pvals), dtype=float)

    # Drop NaNs/Infs
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0

    # Clip into (0, 1]; keep 1.0 as-is, push 0.0 up to a tiny positive float.
    # 1e-300 is below typical double precision min-normal (~1e-308) but still
    # representable; also avoids returning exactly 0.0 from combine_pvalues.
    tiny = 1e-300
    arr = np.clip(arr, tiny, 1.0)

    stat, p = combine_pvalues(arr, method="fisher", weights=None)
    return float(p)

def compute_scores_for_group(chromosome_data_group):
    """
    Worker: compute PHASIS scores for one (chromosome, library) group.

    Input: ((chromosome, library), rows-as-list) where each row has columns:
      ['cluster_id','window_n','fw_pval_corr','rv_pval_corr',
       'combined_window_p_value','chromosome','library']

    Output: list of [cID, phasis_score, combined_fishers]
    """
    (chromosome, lib), data_list = chromosome_data_group

    cols = ['cluster_id','window_n','fw_pval_corr','rv_pval_corr',
            'combined_window_p_value','chromosome','library']
    if not data_list:
        return []

    df = pd.DataFrame(data_list, columns=cols)

    # Re-assert group labels (defensive)
    df['chromosome'] = chromosome
    df['library'] = lib

    # Coerce numerics safely
    for c in ['window_n', 'fw_pval_corr', 'rv_pval_corr', 'combined_window_p_value']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    out = []
    for cID, aclust in df.groupby('cluster_id', sort=False):
        vals = aclust['combined_window_p_value'].dropna()
        if vals.empty:
            # ---- DEBUG LINE (requested) ----
            print(f"[DBG] no combined_window_p_value for {cID} in (chr={chromosome}, lib={lib}); rows={len(aclust)}")
            combined_fishers = 1.0
            phasis_score = 0.0
        else:
            combined_fishers = fishers(vals.tolist())
            if not np.isfinite(combined_fishers) or combined_fishers <= 0.0:
                combined_fishers = max(float(combined_fishers) if np.isfinite(combined_fishers) else 0.0, 1e-300)
            phasis_score = -np.log10(combined_fishers)
            if not np.isfinite(phasis_score):
                phasis_score = 300.0

        if phasis_score > 300.0:
            phasis_score = 300.0

        out.append([str(cID), float(phasis_score), float(combined_fishers)])

    return out

def _record_clusters_scored_tsv_path(path: str) -> None:
    """
    Persist scored TSV path into runtime + snapshot (spawn-safe).

    macOS spawn workers must rebuild WIN_SCORE_LOOKUP from the scored TSV.
    This stores the absolute path in rt and updates the runtime snapshot file.
    """
    try:
        if not path:
            return
        p = os.path.abspath(os.path.expanduser(path))
        rt.clusters_scored_tsv = p

        # Save back to the same snapshot file if already set, else create default
        snap = getattr(rt, "runtime_snapshot", None)
        if hasattr(rt, "save_snapshot"):
            rt.save_snapshot(snap)
    except Exception:
        # Never fail legacy execution because of snapshot bookkeeping
        return

def infer_library_from_cluster_id(cid: str, phase_value: int) -> str:
    """
    Infer library from cluster_id (compact IDs), with fallbacks for old IDs.

    Rules (same as your nested _infer_lib):
      1) If '-' exists: library is everything before the last '-'
      2) Else: split on '{phase}-PHAS' or '{swap_phase}-PHAS'
      3) Else: UNKNOWN
    """
    s = str(cid)

    if "-" in s:
        return s.rsplit("-", 1)[0]

    swap_phase = (21 if phase_value == 24 else 24 if phase_value == 21 else phase_value)
    tag_main = f"{phase_value}-PHAS"
    tag_alt = f"{swap_phase}-PHAS"

    if tag_main in s:
        return s.split(tag_main)[0]
    if tag_alt in s:
        return s.split(tag_alt)[0]

    return "UNKNOWN"

def compute_and_save_phasis_scores(clusters: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PHASIS scores per (chromosome, library) group in parallel.
    Writes {phase}_clusters_scored.tsv and caches its hash in memFile.

    Spawn-safety:
      - Records rt.clusters_scored_tsv and saves runtime snapshot so that
        macOS spawn workers can rebuild WIN_SCORE_LOOKUP from this TSV.
    """
    print("### Step: Compute PHASIS scores per (chromosome, library) ###")
    # Pull run-specific settings from runtime (spawn/fork safe)
    phase = getattr(rt, "phase", None)
    memFile = getattr(rt, "memFile", MEM_FILE_DEFAULT)
    outfname = phase2_basename("clusters_scored.tsv")

    windows_path = phase2_basename("clusters_windows_to_score.tsv")
    extra_sig = []
    try:
        extra_sig.append(f"rows={len(clusters)}")
    except Exception:
        extra_sig = []

    input_sig = compute_cache_signature(
        files=[windows_path],
        params={"phase": phase},
        extra=extra_sig,
    )

    # --- Early hash+sig check ---
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(memFile)
    section = "CLUSTERS_SCORED"
    if not cfg.has_section(section):
        cfg.add_section(section)

    if os.path.isfile(outfname):
        _, cur_md5 = getmd5(outfname)
        prev_md5 = cfg[section].get(outfname)
        prev_sig = cfg[section].get(sig_key(outfname))
        if prev_md5 and prev_md5 == cur_md5 and prev_sig and prev_sig == input_sig:
            print(f"  - Output up-to-date (hash+sig match). Skipping computation: {outfname}")
            df_cached = pd.read_csv(outfname, sep="\t")
            for c in ("phasis_score", "combined_fishers"):
                if c in df_cached.columns:
                    df_cached[c] = pd.to_numeric(df_cached[c], errors="coerce").fillna(0.0)

            # >>> spawn fix: persist scored TSV path for workers
            _record_clusters_scored_tsv_path(outfname)
            return df_cached

    # --- Guard input ---
    if clusters is None or getattr(clusters, "empty", True):
        print("[INFO] compute_and_save_phasis_scores: empty input; writing empty file.")
        pd.DataFrame(columns=["cID", "phasis_score", "combined_fishers"]).to_csv(
            outfname, sep="\t", index=False
        )

        if os.path.isfile(outfname):
            _, out_md5 = getmd5(outfname)
            cfg[section][outfname] = out_md5
            cfg[section][sig_key(outfname)] = input_sig
            with open(memFile, "w") as fh:
                cfg.write(fh)

        # >>> spawn fix: persist scored TSV path for workers (even if empty)
        _record_clusters_scored_tsv_path(outfname)
        return pd.DataFrame(columns=["cID", "phasis_score", "combined_fishers"])

    # --- Derive chromosome & library from compact IDs (robust) ---
    clusters = clusters.copy()
    if "cluster_id" not in clusters.columns:
        raise KeyError("compute_and_save_phasis_scores: input must contain 'cluster_id' column")

    clusters["cluster_id"] = clusters["cluster_id"].astype(str)

    # Chromosome = last underscore chunk
    clusters["chromosome"] = clusters["cluster_id"].str.rsplit("_", n=1).str[-1]

    # Library inference (no nested function)
    phase_value = int(phase) if phase is not None else 21
    clusters["library"] = [
        infer_library_from_cluster_id(cid, phase_value) for cid in clusters["cluster_id"].tolist()
    ]

    # Debug visibility on inference
    total_rows = len(clusters)
    unknown_libs = int((clusters["library"] == "UNKNOWN").sum())
    if unknown_libs:
        ex = clusters.loc[clusters["library"] == "UNKNOWN", "cluster_id"].head(5).tolist()
        print(f"[WARN] {unknown_libs}/{total_rows} cluster_ids have UNKNOWN library. Examples: {ex}")

    # Ensure numeric columns (they might be strings if read from TSV)
    for c in ("window_n", "fw_pval_corr", "rv_pval_corr", "combined_window_p_value"):
        if c in clusters.columns:
            clusters[c] = pd.to_numeric(clusters[c], errors="coerce")

    # Group and ship to workers (dropna=False to avoid dropping UNKNOWN)
    groups = [
        (
            (chrom, lib),
            df[
                [
                    "cluster_id",
                    "window_n",
                    "fw_pval_corr",
                    "rv_pval_corr",
                    "combined_window_p_value",
                    "chromosome",
                    "library",
                ]
            ].values.tolist(),
        )
        for (chrom, lib), df in clusters.groupby(["chromosome", "library"], sort=False, dropna=False)
    ]
    print(f"  - Found {len(groups)} (chromosome, library) groups")

    if not groups:
        print(
            f"[ERR] No groups formed. Unique cluster_ids={clusters['cluster_id'].nunique()}, "
            f"chromosomes={clusters['chromosome'].nunique()}, libraries={clusters['library'].nunique()}"
        )

    # --- Parallel compute ---
    # On Linux/HPC, 'fork' can occasionally deadlock with NumPy/SciPy (esp. if any
    # native libs have background threads). Prefer forkserver/spawn for this stage.
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

    # Flatten and write
    flat = [item for sub in (results or []) for item in (sub or [])]
    win_phasis_score = pd.DataFrame(flat, columns=["cID", "phasis_score", "combined_fishers"])

    win_phasis_score.to_csv(outfname, sep="\t", index=False)
    if os.path.isfile(outfname):
        _, out_md5 = getmd5(outfname)
        cfg[section][outfname] = out_md5
        cfg[section][sig_key(outfname)] = input_sig
        with open(memFile, "w") as fh:
            cfg.write(fh)
        print(f"  - Wrote {outfname} (md5: {out_md5})")

    # >>> spawn fix: persist scored TSV path for workers
    _record_clusters_scored_tsv_path(outfname)
    return win_phasis_score

