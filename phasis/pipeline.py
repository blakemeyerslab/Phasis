# phasis/pipeline.py
from __future__ import annotations

import os

from . import parallel as parallel_utils


def run_pipeline() -> int:
    # Ensure headless plotting (safe on macOS/HPC)
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Heavy imports happen here (run path only)
    from . import runtime as rt
    from .stages import dependency_check as st_dependency_check
    from .stages import input_validation as st_input_validation
    from .stages.phase1_pipeline import run_phase1_pipeline
    from .stages.phase2_pipeline import run_phase2_pipeline

    # Reserve worker cores and publish to runtime for pool creation
    rt.ncores = parallel_utils.coreReserve(rt.cores)

    # startup sequence (same order as before)
    st_dependency_check.checkDependency()
    libs_checked = st_input_validation.checkLibs()

    # keep validated runtime inputs consistent
    rt.libs = libs_checked

    cleanup_requested = bool(getattr(rt, "cleanup", False) or getattr(rt, "cleanup_all", False))
    if cleanup_requested:
        print("[ERROR] Cleanup modes are standalone. Use them without any other arguments.")
        return 2

    # ---- dispatcher ----
    steps_local = getattr(rt, "steps", None) or "both"
    steps_local = str(steps_local).strip().lower()

    if steps_local == "cfind":
        run_phase1_pipeline(libs_checked)
    elif steps_local == "class":
        # run_phase2_pipeline() will pull cfg from runtime and override clusterFilePaths
        run_phase2_pipeline([])
    elif steps_local == "both":
        clusterFilePaths = run_phase1_pipeline(libs_checked)
        run_phase2_pipeline(clusterFilePaths)
    else:
        raise ValueError(
            f"Unknown steps value: {steps_local!r} (expected 'cfind', 'class', or 'both')"
        )

    return 0
