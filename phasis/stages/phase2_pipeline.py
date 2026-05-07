from __future__ import annotations

"""
phasis.phase2_pipeline
----------------------

Orchestrator for Phase II ("class") that sequences already-extracted stages:

1) cluster_aggregation   -> {phase}_processed_clusters.tab
2) candidates_merge      -> {phase}_candidate.loci_table.tab (+ merged_candidates.tab in concat mode)
3) ids                   -> mergedClusterDict.tab + reverse map cache
4) phas_clusters         -> {phase}_PHAS_to_detect.tab
5) window_selection      -> {phase}_clusters_windows_to_score.tsv (resume-safe)
6) window_scoring        -> {phase}_clusters_scored.tsv + score lookup
7) feature_assembly      -> features table (stage-owned naming)
8) classify              -> KNN/GMM calls + downstream outputs (stage-owned)

Design constraints:
- spawn-safe (top-level functions only)
- no imports inside functions
- runtime-first via Phase2Config.from_runtime(), but explicit cfg supported
- minimal behavior drift: keeps the same early-exit behavior used in legacy run_phase2()
"""

import os
from typing import Any, Optional, Sequence, Union

import pandas as pd

import phasis.runtime as rt
import phasis.ids as ids
from phasis.cache import phase2_basename
from phasis.config import Phase2Config
from phasis.state import set_win_score_lookup

from phasis.stages import cluster_aggregation as st_cluster_aggregation
from phasis.stages import candidates_merge as st_cmerge
from phasis.stages import phas_clusters as st_phas_clusters
from phasis.stages import window_selection as st_winsel
from phasis.stages import window_scoring as st_winscore
from phasis.stages import feature_assembly as st_feat
from phasis.stages import classify as st_classify
from phasis.stages import locus_plots as st_locus_plots
from phasis.stages import output as st_output


def _normalize_cluster_df(df: pd.DataFrame, is_concat: bool) -> pd.DataFrame:
    """
    Normalize column names/required cols for downstream grouping.
    - Ensures 'chromosome' column exists (renames 'chr' -> 'chromosome' if present).
    - Ensures 'alib' exists (sets to 'ALL_LIBS' in concat mode if missing).
    Returns the same DataFrame (mutated) for convenience.
    """
    if df is None:
        return pd.DataFrame()

    # Rename chr -> chromosome if needed
    if "chromosome" not in df.columns and "chr" in df.columns:
        df = df.rename(columns={"chr": "chromosome"})

    # Ensure alib
    if "alib" not in df.columns:
        if is_concat:
            df = df.copy()
            df["alib"] = "ALL_LIBS"
        else:
            print("[WARN] DataFrame missing 'alib' in non-concat mode.")

    return df


def run_phase2_pipeline(
    clusterFilePaths: Union[Sequence[str], str, None],
    *,
    cfg: Phase2Config | None = None,
) -> None:
    """
    Phase II (class): merge candidates, ensure universal IDs, build clusters,
    select windows, score, feature assembly, classify, and write outputs/plots.

    clusterFilePaths:
      - in -steps both: list of per-library cluster files returned by Phase I
      - in -steps class: usually overridden by cfg.class_cluster_file
    """
    print("######            Starting Phase II          #########")

    if cfg is None:
        cfg = Phase2Config.from_runtime()

    # If running 'class' only, take precomputed cluster files
    if cfg.steps == "class":
        clusterFilePaths = cfg.class_cluster_file

    # 1) Aggregate (writes processed_clusters.tab; returns dataframe)
    agg_df = st_cluster_aggregation.aggregate_and_write_processed_clusters(
        clusterFilePaths, memFile=cfg.memFile
    )

    # Build "allClusters" baseline from aggregator result or file fallback
    if isinstance(agg_df, pd.DataFrame) and not agg_df.empty:
        allClusters = agg_df
    else:
        proc_path = phase2_basename("processed_clusters.tab")
        allClusters = (
            pd.read_csv(proc_path, sep="\t") if os.path.isfile(proc_path) else pd.DataFrame()
        )

    # Normalize for downstream grouping
    allClusters = _normalize_cluster_df(allClusters, is_concat=cfg.concat_libs)

    # 2) ALWAYS emit loci table BEFORE merge (guarantees the input exists for merge)
    loci_table_df = st_cmerge.loci_table_from_clusters(allClusters)
    loci_table_path = phase2_basename("candidate.loci_table.tab")

    # 3) Merge candidates (concat only). Non-concat continues with the pre-merge representation.
    merged_out_path = phase2_basename("merged_candidates.tab")
    if cfg.concat_libs:
        # Cache-aware merge: returns a DataFrame (loads TSV on cache hit)
        _ = st_cmerge.merge_candidate_clusters_across_libs(loci_table_path, merged_out_path)

    # 3.5) Always ensure universal-ID dict (used by ids.getUniversalID)
    mcd = ids.ensure_mergedClusterDict_always(
        concat_libs=cfg.concat_libs,
        phase=str(cfg.phase),
        merged_out_path=merged_out_path,
        loci_table_df=loci_table_df,
        allClusters_df=allClusters,
        memFile=cfg.memFile,
    )

    # Optional: surface this in runtime for any remaining legacy compatibility
    try:
        rt.mergedClusterDict = mcd
    except Exception:
        pass

    print(f"[INFO] mergedClusterDict ready with {len(mcd)} universal IDs.")

    # 4) Build PHAS clusters (handles empty input)
    clusters_data = st_phas_clusters.build_and_save_phas_clusters(
        allClusters,
        phase=int(cfg.phase) if str(cfg.phase).isdigit() else None,
        memFile=cfg.memFile,
        concat_libs=cfg.concat_libs,
    )

    # 5) If there are no clusters, short-circuit cleanly
    if clusters_data is None or getattr(clusters_data, "empty", True):
        print("[INFO] No PHAS clusters to score; exiting classification early.")
        return

    # 6) Select windows (explicit args; no legacy globals)
    clusters_windows = st_winsel.select_scoring_windows(
        clusters_data,
        window_len=cfg.window_len,
        sliding=cfg.sliding,
        minClusterLength=cfg.minClusterLength,
        memFile=cfg.memFile,
    )
    if clusters_windows is None or getattr(clusters_windows, "empty", True):
        print("[INFO] No scoring windows found; exiting classification early.")
        return

    # 7) Score windows, expose compact lookup to workers, extract features, classify
    win_phasis_score = st_winscore.compute_and_save_phasis_scores(clusters_windows)
    set_win_score_lookup(win_phasis_score)

    features = st_feat.features_to_detection(
        clusters_data,
        phase=int(cfg.phase) if str(cfg.phase).isdigit() else cfg.phase,
        outdir=cfg.outdir,
        concat_libs=cfg.concat_libs,
        memFile=cfg.memFile,
    )

    # 8) Classify (stage returns labeled DF), then finalize outputs (output stage)
    if cfg.classifier == "GMM":
        labeled = st_classify.gmm_classify(
            features,
            phasisScoreCutoff=float(cfg.phasisScoreCutoff),
            min_Howell_score=float(cfg.min_Howell_score),
            max_complexity=float(cfg.max_complexity),
            n_clusters=int(getattr(cfg, "n_clusters", 2) or 2),
        )
        labeled = st_classify.apply_evidence_classification(
            labeled,
            phase=cfg.phase,
            legacy_classification=bool(getattr(cfg, "legacy_classification", False)),
            overrides_path=getattr(cfg, "classification_overrides", None),
        )
        st_locus_plots.write_individual_phas_locus_plots(
            "GMM",
            labeled,
            clusters_data,
            job_outdir=cfg.outdir,
            job_phase=cfg.phase,
        )
        st_output.finalize_and_write_results(
            "GMM",
            labeled,
            job_outdir=cfg.outdir,
            job_phase=cfg.phase,
        )
    elif cfg.classifier == "KNN":
        labeled = st_classify.knn_classify(
            features,
            phasisScoreCutoff=float(cfg.phasisScoreCutoff),
            min_Howell_score=float(cfg.min_Howell_score),
            max_complexity=float(cfg.max_complexity),
        )
        labeled = st_classify.apply_evidence_classification(
            labeled,
            phase=cfg.phase,
            legacy_classification=bool(getattr(cfg, "legacy_classification", False)),
            overrides_path=getattr(cfg, "classification_overrides", None),
        )
        st_locus_plots.write_individual_phas_locus_plots(
            "KNN",
            labeled,
            clusters_data,
            job_outdir=cfg.outdir,
            job_phase=cfg.phase,
        )
        st_output.finalize_and_write_results(
            "KNN",
            labeled,
            job_outdir=cfg.outdir,
            job_phase=cfg.phase,
        )
    else:
        raise ValueError(f"Unknown classifier: {cfg.classifier!r}")
