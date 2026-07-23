# phasis/runtime.py
# Central home for globals (Phase 2 refactor). Keep stdlib-only.

import os
import json

# --- NEW: execution context ---
run_dir = None                 # directory where intermediates live (your ".")
outdir = None                  # already present in your file, but keep it here too
memFile = None                 # path to phasis.mem (NOT inside outdir)
runtime_snapshot = None        # path to .phasis.runtime.json
# --- NEW: spawn-safe scoring lookup ---
clusters_scored_tsv = None      # absolute path to *_clusters_scored.tsv
# --- NEW: include missing config knobs you already set in CLI ---
cleanup = None
cleanup_all = None
cluster_build_initial_worker_cap = None
cluster_build_max_worker_cap = None
cluster_scoring_initial_worker_cap = None
cluster_scoring_max_worker_cap = None
feature_assembly_worker_cap = None
feature_assembly_batch_rows = None
plot_staging = None
plot_staging_mode = None
plot_staging_root = None
legacy_classification = None
classification_overrides = None
locus_plot_mode = None
reference_id_mode = None
compress_intermediates = True
samtools_path = None
samtools_version = None
parallel_lib_worker_cap = None
fastq_chunk_unique_tags = None

# (keep your existing globals below; I’m not repeating them all)

RUNTIME_SNAPSHOT_NAME = ".phasis.runtime.json"

# Only persist lightweight config values (do NOT persist huge dicts like mergedClusterDict)
_RUNTIME_KEYS = [
    "libs","reference","norm","norm_factor","maxhits","runtype","reference_id_mode","mindepth","uniqueRatioCut","mismat",
    "libformat","phase","phase2","phaseLen","clustbuffer","phasisScoreCutoff","minClusterLength","window_len","sliding",
    "cores","classifier","steps","class_cluster_file","max_complexity","min_Howell_score","concat_libs",
    "outdir","run_dir","memFile","clusters_scored_tsv","cleanup","cleanup_all",
    "cluster_build_initial_worker_cap","cluster_build_max_worker_cap",
    "cluster_scoring_initial_worker_cap","cluster_scoring_max_worker_cap",
    "feature_assembly_worker_cap",
    "feature_assembly_batch_rows",
    "plot_staging","plot_staging_mode","plot_staging_root",
    "legacy_classification","classification_overrides","locus_plot_mode",
    "compress_intermediates","samtools_path","samtools_version","parallel_lib_worker_cap","fastq_chunk_unique_tags",
]

def _snapshot_path(run_dir_override: str | None = None) -> str:
    rd = run_dir_override or run_dir or os.getcwd()
    return os.path.join(rd, RUNTIME_SNAPSHOT_NAME)

def to_dict() -> dict:
    d = {}
    g = globals()
    for k in _RUNTIME_KEYS:
        if k in g:
            d[k] = g[k]
    return d

def apply_dict(d: dict) -> None:
    g = globals()
    for k, v in d.items():
        if k in _RUNTIME_KEYS:
            g[k] = v

def save_snapshot(path: str | None = None) -> str:
    p = path or _snapshot_path()
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(to_dict(), fh, indent=2, sort_keys=True)
    globals()["runtime_snapshot"] = p
    return p

def load_snapshot(path: str | None = None) -> bool:
    p = path or _snapshot_path()
    if not os.path.isfile(p):
        return False
    with open(p, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    apply_dict(d)
    globals()["runtime_snapshot"] = p
    return True
