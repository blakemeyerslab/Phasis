from __future__ import annotations

"""
phasis.stages.phase1_pipeline
-----------------------------

Phase I ("cfind") orchestrator.

This stage now calls the extracted Phase I stage modules directly and no longer
depends on any temporary compatibility module.

Design constraints:
- spawn-safe (top-level functions only)
- no nested functions; no imports inside functions
- runtime-first (worker-safe via phasis.runtime)
- minimal behavior drift: preserves legacy's execution order and runtime log file
"""

import os
from typing import Any, Optional, Sequence

import phasis.runtime as rt
from phasis import stage_timer

from phasis.stages import cluster_build as st_cluster_build
from phasis.stages import cluster_scoring as st_cluster_scoring
from phasis.stages import folder_setup as st_folder_setup
from phasis.stages import indexing as st_indexing
from phasis.stages import library_processing as st_library_processing
from phasis.stages import mapping as st_mapping
from phasis.stages import sam_parsing as st_sam_parsing


def run_phase1_pipeline(
    libs: Sequence[str],
    *,
    cfg: Optional[Any] = None,
    workdir: Optional[str] = None,
) -> Any:
    """
    Phase I (cfind): preprocess → map → parse → cluster → window scoring.
    Returns clusterFilePaths (same object the old monolithic entrypoint used).

    Parameters
    ----------
    libs:
        List of library inputs (already validated by input validation).
    cfg:
        Reserved for future Phase I config. Currently unused (kept for API symmetry).
    workdir:
        Optional override of working directory (defaults to os.getcwd()).
    """
    _ = cfg

    # Preserve legacy behavior: print header + write runtime log
    print("######            Starting Phase I           #########")

    # The runtime file holds the index time and one line per finished stage.
    runLog = stage_timer.open_run_log()
    fh_run = open(runLog, "a")

    clusterFilePaths = None
    try:
        wd = workdir if workdir is not None else os.getcwd()

        # Create folders, build/reuse index
        clustfolder = st_folder_setup.createfolders(wd)
        with stage_timer.stage("Index and reference"):
            genoIndex = st_indexing.getindex(fh_run)
            fh_run.flush()

        # Preprocess → map → parse
        with stage_timer.stage("Library processing"):
            libs_processed = st_library_processing.libraryprocess(libs)

        ncores_local = getattr(rt, "ncores", None)
        if ncores_local is None:
            ncores_local = getattr(rt, "cores", None)

        with stage_timer.stage("Mapping"):
            _ = st_mapping.mapprocess(
                libs_processed,
                genoIndex,
                ncores_local=ncores_local,
            )
        with stage_timer.stage("Parsing alignments"):
            libs_nestdict, libs_poscountdict = st_sam_parsing.parserprocess(libs_processed)

        # Find clusters
        with stage_timer.stage("Finding clusters"):
            libs_clustdicts = st_cluster_build.clusterprocess(libs_poscountdict, clustfolder)

        # Score (bucketed hashing will skip when cached)
        with stage_timer.stage("Cluster scoring"):
            clusterFilePaths = st_cluster_scoring.scoringprocess(
                libs=libs_processed,
                libs_clustdicts=libs_clustdicts,
                libs_nestdict=libs_nestdict,
                clustfolder=clustfolder,
            )
    finally:
        try:
            fh_run.close()
        except Exception:
            pass

    return clusterFilePaths
