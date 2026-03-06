from __future__ import annotations

"""
phasis.stages.window_selection
------------------------------

Phase II stage: select scoring windows per chromosome (and per library if present).

This stage is **resume-safe**:
- Each lib-chr group is written to a chunk TSV under a parameterized cache dir:
    {phase}_windows_sl{sliding}_wl{window_len}_mcl{minClusterLength}/<lib>__chr<id>.tsv
- If the chunk file already exists and is non-empty, it is reused (existence-only; no md5
  checks for speed).

The final merged output is written to:
    phase2_basename('clusters_windows_to_score.tsv')
and its md5 is recorded in memFile under section "WINDOWS_TO_SCORE" (best effort).

Constraints:
- spawn-safe (top-level functions only)
- no nested functions; no imports inside functions
- runtime-first: defaults come from phasis.runtime, but explicit args are supported
"""

import configparser
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import phasis.runtime as rt
from phasis.cache import getmd5, phase2_basename, compute_cache_signature, sig_key
from phasis.parallel import run_parallel_with_progress

WINDOWS_COLUMNS: List[str] = [
    "cluster_id",
    "window_n",
    "fw_pval_corr",
    "rv_pval_corr",
    "combined_window_p_value",
]


def _safe_key(akey: str) -> str:
    """Normalize an akey to a filesystem-safe basename."""
    s = str(akey)
    # Drop any path components to avoid directory traversal
    s = os.path.basename(s)
    # Avoid accidental separators on Windows-like paths (harmless on macOS/Linux)
    s = s.replace(os.sep, "_")
    return s


def _load_final_if_fresh(outfname: str, memFile: Optional[str], input_sig: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Return cached final dataframe if outfname md5 matches memFile record; else None."""
    if not os.path.isfile(outfname):
        return None
    if not memFile or not os.path.isfile(memFile):
        return None

    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    try:
        cfg.read(memFile)
    except Exception:
        return None

    sec = "WINDOWS_TO_SCORE"
    try:
        prev = cfg.get(sec, outfname, fallback=None)
    except Exception:
        prev = None
    if not prev:
        return None

    try:
        _, cur = getmd5(outfname)
    except Exception:
        return None

    if cur != prev:
        return None

    if input_sig:
        try:
            prev_sig = cfg.get(sec, sig_key(outfname), fallback=None)
        except Exception:
            prev_sig = None
        if not prev_sig or prev_sig != input_sig:
            return None

    print(f"  - Output up-to-date (hash match). Skipping computation: {outfname}")
    try:
        df = pd.read_csv(outfname, sep="\t", engine="python")
    except Exception:
        df = pd.read_csv(outfname, sep="\t")

    # enforce numeric columns
    for c in ("window_n", "fw_pval_corr", "rv_pval_corr", "combined_window_p_value"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _write_final_md5(outfname: str, memFile: Optional[str], input_sig: Optional[str] = None) -> None:
    """Best-effort: store outfname md5 in memFile under section WINDOWS_TO_SCORE."""
    if not memFile:
        print("[WARN] memFile not set; skipping WINDOWS_TO_SCORE md5 update.")
        return

    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    try:
        if os.path.isfile(memFile):
            cfg.read(memFile)
    except Exception:
        pass

    sec = "WINDOWS_TO_SCORE"
    if not cfg.has_section(sec):
        try:
            cfg.add_section(sec)
        except Exception:
            return

    try:
        _, out_md5 = getmd5(outfname)
        cfg.set(sec, outfname, out_md5)
        if input_sig:
            try:
                cfg.set(sec, sig_key(outfname), input_sig)
            except Exception:
                pass
    except Exception:
        return

    try:
        with open(memFile, "w") as fh:
            cfg.write(fh)
        print(f"  - Wrote {outfname} (md5: {out_md5})")
    except Exception:
        print(f"  - Wrote {outfname}")


def select_scoring_windows(
    clusters_data: pd.DataFrame,
    *,
    window_len: Optional[int] = None,
    sliding: Optional[int] = None,
    minClusterLength: Optional[int] = None,
    memFile: Optional[str] = None,
) -> pd.DataFrame:
    """
    For each chromosome (and library, if present), slide a fixed-length window across each
    cluster (>= minClusterLength) and record the best corrected p-values per window
    (forward/reverse) and their product.

    Inputs:
      clusters_data: DataFrame with columns:
        - clusterID, pos, pval_corr_f, pval_corr_r, chromosome
        - optional: alib

    Runtime-first defaults:
      - window_len: rt.window_len
      - sliding: rt.sliding
      - minClusterLength: rt.minClusterLength
      - memFile: rt.memFile
    """
    print("### Step: select scoring windows per chromosome ###")

    wl = int(window_len if window_len is not None else getattr(rt, "window_len", 0) or 0)
    sl = int(sliding if sliding is not None else getattr(rt, "sliding", 0) or 0)
    mcl = int(
        minClusterLength if minClusterLength is not None else getattr(rt, "minClusterLength", 0) or 0
    )
    memFile_local = memFile if memFile is not None else getattr(rt, "memFile", None)

    if wl <= 0 or sl <= 0:
        raise ValueError(f"Invalid window_len/sliding: window_len={wl}, sliding={sl}")

    outfname = phase2_basename("clusters_windows_to_score.tsv")

    # Encode key runtime params in the directory name to segregate caches across settings
    outdir = phase2_basename(f"windows_sl{sl}_wl{wl}_mcl{mcl}")
    os.makedirs(outdir, exist_ok=True)

    # Signature from upstream PHAS_to_detect + key parameters
    try:
        phas_tab = phase2_basename("PHAS_to_detect.tab")
        input_sig = compute_cache_signature(
            files=[phas_tab],
            params={"window_len": wl, "sliding": sl, "minClusterLength": mcl},
        )
    except Exception:
        input_sig = None

    # Early return on final up-to-date file (hash match)
    cached = _load_final_if_fresh(outfname, memFile_local, input_sig)
    if cached is not None:
        return cached

    # --- Normalize/guard input ---
    required_in = ["clusterID", "pos", "pval_corr_f", "pval_corr_r", "chromosome"]

    if clusters_data is None or getattr(clusters_data, "empty", True):
        print("[INFO] No clusters to select windows from; writing empty output.")
        empty_out = pd.DataFrame(columns=WINDOWS_COLUMNS)
        empty_out.to_csv(outfname, sep="\t", index=False)
        _write_final_md5(outfname, memFile_local, input_sig)
        return empty_out

    if "chromosome" not in clusters_data.columns and "chr" in clusters_data.columns:
        clusters_data = clusters_data.rename(columns={"chr": "chromosome"})

    missing = [c for c in required_in if c not in clusters_data.columns]
    if missing:
        raise ValueError(f"select_scoring_windows(): missing required columns: {missing}")

    keep_cols = required_in + (["alib"] if "alib" in clusters_data.columns else [])
    clusters_data = clusters_data.loc[:, keep_cols].copy()

    if clusters_data.empty:
        print("[INFO] Input empty after column filtering; writing empty output.")
        empty_out = pd.DataFrame(columns=WINDOWS_COLUMNS)
        empty_out.to_csv(outfname, sep="\t", index=False)
        _write_final_md5(outfname, memFile_local, input_sig)
        return empty_out

    # --- Build lib‑chr groups ---
    grouping = ["chromosome"] + (["alib"] if "alib" in clusters_data.columns else [])
    groups = [(k, df) for k, df in clusters_data.groupby(grouping, sort=False)]
    print(f"  - Found {len(groups)} group(s) by {grouping}")

    # --- Plan tasks with cache checks (existence-only resume) ---
    tasks: List[Dict[str, Any]] = []
    kept_paths: List[str] = []  # chunk paths to merge (cached + newly written)

    for key_tuple, gdf in groups:
        # key normalization
        if isinstance(key_tuple, tuple):
            chrom = key_tuple[0]
            libid = key_tuple[1] if len(key_tuple) > 1 else "concat"
        else:
            chrom = key_tuple
            libid = "concat"

        key = f"{libid}__chr{chrom}"
        outp = os.path.join(outdir, f"{_safe_key(key)}.tsv")

        if os.path.isfile(outp) and os.path.getsize(outp) > 0:
            kept_paths.append(outp)
            continue

        tasks.append(
            {
                "key": key,
                "df": gdf,
                "outpath": outp,
                "window_len": wl,
                "sliding": sl,
                "minClusterLength": mcl,
            }
        )

    print(
        f"  - {len(kept_paths)} cached chunk(s) will be reused; "
        f"{len(tasks)} chunk(s) to compute"
    )

    results: List[object] = []
    if tasks:
        results = (
            run_parallel_with_progress(
                select_windows_task_worker,
                tasks,
                desc="Selecting windows (resume‑safe)",
                min_chunk=1,
                batch_factor=5,
                unit="lib-chr",
            )
            or []
        )

    # Fail fast on worker errors (prevents confusing downstream TypeErrors)
    worker_errors = [r for r in results if isinstance(r, RuntimeError)]
    if worker_errors:
        raise worker_errors[0]

    # Update bookkeeping for newly produced chunks
    for r in results:
        if not r:
            continue
        if isinstance(r, dict):
            outp = r.get("outpath")
            if outp:
                kept_paths.append(outp)

    # Merge all chunk files (order by path for reproducibility)
    kept_paths = sorted(set(kept_paths))
    frames: List[pd.DataFrame] = []
    for p in kept_paths:
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            try:
                frames.append(pd.read_csv(p, sep="\t", engine="python"))
            except Exception:
                frames.append(pd.read_csv(p, sep="\t"))

    if frames:
        to_score = pd.concat(frames, ignore_index=True)
        for c in ("window_n", "fw_pval_corr", "rv_pval_corr", "combined_window_p_value"):
            if c in to_score.columns:
                to_score[c] = pd.to_numeric(to_score[c], errors="coerce")

        sort_cols = [c for c in ("cluster_id", "window_n") if c in to_score.columns]
        if sort_cols:
            to_score = to_score.sort_values(sort_cols, kind="mergesort")
    else:
        to_score = pd.DataFrame(columns=WINDOWS_COLUMNS)

    # --- Write final + hash ---
    to_score.to_csv(outfname, sep="\t", index=False)
    _write_final_md5(outfname, memFile_local, input_sig)

    print(f"    Cached chunks directory: {outdir}")
    return to_score


def select_windows_task_worker(task: Dict[str, Any]) -> Dict[str, str]:
    """
    Worker wrapper: computes windows for a task and writes TSV to task['outpath'].
    Returns {'outpath', 'key'} for bookkeeping.

    Task fields:
      - df: group DataFrame
      - outpath: output chunk path
      - window_len, sliding, minClusterLength: ints
    """
    outpath = str(task["outpath"])
    df = task["df"]

    wl = int(task["window_len"])
    sl = int(task["sliding"])
    mcl = int(task["minClusterLength"])

    rows = select_windows_for_chromosome(df, window_len=wl, sliding=sl, minClusterLength=mcl)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    if not rows:
        pd.DataFrame(columns=WINDOWS_COLUMNS).to_csv(outpath, sep="\t", index=False)
    else:
        pd.DataFrame(rows, columns=WINDOWS_COLUMNS).to_csv(outpath, sep="\t", index=False)

    return {"outpath": outpath, "key": str(task.get("key", ""))}


def select_windows_for_chromosome(
    chromosome_df: pd.DataFrame,
    *,
    window_len: int,
    sliding: int,
    minClusterLength: int,
) -> List[List[Any]]:
    """
    Select windows for a single (lib, chr) group.

    Expected columns: clusterID, pos, pval_corr_f, pval_corr_r
    Returns rows:
      [cluster_id, window_n, best_f, best_r, best_f*best_r]
    """
    df = chromosome_df.loc[:, ["clusterID", "pos", "pval_corr_f", "pval_corr_r"]].copy()

    # Ensure numeric types for computations (once per group)
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
    df["pval_corr_f"] = pd.to_numeric(df["pval_corr_f"], errors="coerce")
    df["pval_corr_r"] = pd.to_numeric(df["pval_corr_r"], errors="coerce")

    df = df.dropna(subset=["pos"])
    if df.empty:
        return []

    wl = int(window_len)
    sl = int(sliding)
    mcl = int(minClusterLength)

    to_score: List[List[Any]] = []
    append = to_score.append  # micro-opt

    for cID, aclust in df.groupby("clusterID", sort=False):
        if aclust.empty:
            continue

        pos = aclust["pos"].to_numpy()
        if pos.size == 0:
            continue

        # Only sort if needed (stable sort preserves deterministic order)
        if not (pos[:-1] <= pos[1:]).all():
            aclust = aclust.sort_values("pos", kind="mergesort")
            pos = aclust["pos"].to_numpy()

        fw = aclust["pval_corr_f"].to_numpy()
        rv = aclust["pval_corr_r"].to_numpy()

        fw = np.where(np.isfinite(fw), fw, np.inf)
        rv = np.where(np.isfinite(rv), rv, np.inf)

        cluster_start = int(pos[0])
        cluster_end = int(pos[-1])
        cluster_len = cluster_end - cluster_start

        if cluster_len < mcl or cluster_len < wl:
            continue

        nwin = 1 + (cluster_len - wl) // sl
        if nwin <= 0:
            continue

        w_starts = cluster_start + np.arange(0, nwin * sl, sl, dtype=np.int64)
        w_ends = w_starts + wl

        left_idx = np.searchsorted(pos, w_starts, side="left")
        right_idx = np.searchsorted(pos, w_ends, side="left")  # half-open [start, end)

        for w_i in range(nwin):
            li = int(left_idx[w_i])
            ri = int(right_idx[w_i])
            if li >= ri:
                continue

            best_f = float(fw[li:ri].min())
            best_r = float(rv[li:ri].min())

            if not np.isfinite(best_f) or not np.isfinite(best_r):
                continue

            append([cID, w_i, best_f, best_r, best_f * best_r])

    return to_score
