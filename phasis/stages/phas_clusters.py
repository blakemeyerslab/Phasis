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

import multiprocessing
import os
import tempfile
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
from phasis.env import getenv


# ---- required 20-col schema (ORDER MATTERS) ----
REQUIRED_20_COLS: List[str] = [
    "alib", "clusterID", "chromosome", "strand", "pos", "len", "hits", "abun",
    "pval_h_f", "N_f", "X_f", "pval_r_f", "pval_corr_f",
    "pval_h_r", "N_r", "X_r", "pval_r_r", "pval_corr_r",
    "tag_id", "tag_seq",
]
REQUIRED_20_SET = set(REQUIRED_20_COLS)

PHAS_CLUSTER_INITIAL_WORKER_CAP = 2
PHAS_CLUSTER_DEFAULT_MAX_CPU_FRACTION_NUMERATOR = 7
PHAS_CLUSTER_DEFAULT_MAX_CPU_FRACTION_DENOMINATOR = 10
PHAS_CLUSTER_DEFAULT_BATCH_ROWS = 100_000

PHAS_TO_DETECT_TEXT_COLUMNS = (
    "alib",
    "clusterID",
    "chromosome",
    "strand",
    "tag_id",
    "tag_seq",
    "identifier",
)


def _empty_phas_clusters_frame() -> pd.DataFrame:
    """Return the canonical empty PHAS-to-detect schema."""
    return pd.DataFrame(columns=REQUIRED_20_COLS + ["identifier"])


def _coerce_positive_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        value = int(value)
        if value <= 0:
            return default
        return value
    except Exception:
        return default


def _phas_cluster_ncores() -> int:
    """Return the Phasis allocation, falling back safely outside a run."""
    ncores = _coerce_positive_int(getattr(rt, "ncores", None), None)
    if ncores is None:
        ncores = multiprocessing.cpu_count()
    return int(max(1, ncores))


def _phas_cluster_worker_cap() -> int:
    """Resolve the bounded maximum concurrency for PHAS-cluster batches."""
    ncores = _phas_cluster_ncores()
    default_cap = max(
        1,
        (ncores * PHAS_CLUSTER_DEFAULT_MAX_CPU_FRACTION_NUMERATOR)
        // PHAS_CLUSTER_DEFAULT_MAX_CPU_FRACTION_DENOMINATOR,
    )
    configured = _coerce_positive_int(
        getattr(rt, "phas_cluster_worker_cap", None),
        _coerce_positive_int(
            getenv("Phasis_PHAS_CLUSTER_WORKER_CAP"),
            default_cap,
        ),
    )
    return int(max(1, min(ncores, configured)))


def _phas_cluster_batch_rows() -> int:
    """Return the maximum input rows materialized for one PHAS work task."""
    return int(
        _coerce_positive_int(
            getattr(rt, "phas_cluster_batch_rows", None),
            _coerce_positive_int(
                getenv("Phasis_PHAS_CLUSTER_BATCH_ROWS"),
                PHAS_CLUSTER_DEFAULT_BATCH_ROWS,
            ),
        )
    )


def _phas_cluster_parallel_kwargs(task_count: int) -> dict:
    """Bound workers and in-flight DataFrame batches for this stage."""
    max_worker_cap = _phas_cluster_worker_cap()
    initial_worker_cap = min(PHAS_CLUSTER_INITIAL_WORKER_CAP, max_worker_cap)
    initial_task_window = max(1, min(int(task_count), initial_worker_cap))
    max_task_window = max(1, min(int(task_count), max_worker_cap))
    return {
        "initial_worker_cap": initial_worker_cap,
        "max_worker_cap": max_worker_cap,
        "initial_chunk_size": initial_task_window,
        "max_chunk_size": max_task_window,
        "adaptive_recovery": True,
    }


def load_processed_clusters_fallback() -> pd.DataFrame:
    """
    Load {phase}_processed_clusters.tab or return empty DF.

    NOTE: the actual filename comes from phasis.cache.phase2_basename, which uses rt.phase/rt.concat_libs.
    """
    proc_path = phase2_basename("processed_clusters.tab")
    if artifact_exists(proc_path):
        print(f"  - Detected non-20-col input; loading processed-clusters fallback: {proc_path}")
        physical_proc_path = resolve_artifact_path(proc_path) or proc_path
        text_columns = {
            "alib",
            "clusterID",
            "chromosome",
            "strand",
            "tag_id",
            "tag_seq",
        }
        numeric_columns = set(REQUIRED_20_COLS) - text_columns
        try:
            return pd.read_csv(
                physical_proc_path,
                sep="\t",
                engine="python",
                dtype={column: str for column in text_columns},
                keep_default_na=False,
                na_values={column: [""] for column in numeric_columns},
            )
        except Exception:
            return pd.read_csv(
                physical_proc_path,
                sep="\t",
                dtype={column: str for column in text_columns},
                keep_default_na=False,
                na_values={column: [""] for column in numeric_columns},
            )
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


def load_phas_to_detect_output(
    output_file: str,
    *,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load a completed PHAS-to-detect table without coercing text identifiers.

    Phase II normally calls this only after releasing the much larger processed
    clusters DataFrame.  Keeping the text columns explicit is important for
    biological IDs such as ``001`` and literal sequence/tag values such as
    ``NA``.  Passing ``columns`` is an internal memory-saving option that
    leaves the complete on-disk artifact unchanged.
    """
    physical_output_file = resolve_artifact_path(output_file) or output_file
    selected_columns = None
    if columns is not None:
        requested_columns = list(dict.fromkeys(str(column) for column in columns))
        # Read only the header first so a manually supplied legacy table that
        # lacks an optional field remains usable rather than failing a usecols
        # check before downstream validation can give a useful error.
        available_columns = pd.read_csv(physical_output_file, sep="\t", nrows=0).columns
        selected_columns = [
            column for column in requested_columns if column in available_columns
        ]

    text_columns = (
        PHAS_TO_DETECT_TEXT_COLUMNS
        if selected_columns is None
        else tuple(
            column for column in PHAS_TO_DETECT_TEXT_COLUMNS
            if column in selected_columns
        )
    )
    try:
        df = pd.read_csv(
            physical_output_file,
            sep="\t",
            engine="python",
            usecols=selected_columns,
            dtype={column: str for column in text_columns},
            keep_default_na=False,
        )
    except Exception:
        df = pd.read_csv(
            physical_output_file,
            sep="\t",
            usecols=selected_columns,
            dtype={column: str for column in text_columns},
            keep_default_na=False,
        )
    return _coerce_numeric_allowlist(df)


def _read_cached_phas_to_detect(output_file: str, cache: MemCache, input_sig: str | None) -> Optional[pd.DataFrame]:
    """Return cached PHAS_to_detect table if cache hit; else None."""
    section_name = "PHAS_TO_DETECT"
    if cache.hit(section_name, output_file, input_sig):
        print(f"  - Output up-to-date (hash+sig match). Skipping processing: {output_file}")
        return load_phas_to_detect_output(output_file)
    return None


def process_chromosome_data(loci_group: Sequence[Sequence] | pd.DataFrame) -> pd.DataFrame:
    """
    Process data for a single chromosome-library group.

    STRICT: expects 20-column per-read/per-alignment rows with REQUIRED_20_COLS.
    Returns a dataframe with REQUIRED_20_COLS + ["identifier"].

    Any row that cannot be mapped clusterID -> universal "identifier" is dropped.
    """
    if isinstance(loci_group, pd.DataFrame):
        if loci_group.empty or not REQUIRED_20_SET.issubset(loci_group.columns):
            return _empty_phas_clusters_frame()
        # ``_PhasClusterBatchSequence`` has already produced exactly these
        # columns in exactly this order.  Reuse that bounded task DataFrame
        # rather than making another full batch copy in the worker.
        if list(loci_group.columns) == REQUIRED_20_COLS:
            df = loci_group
        else:
            df = loci_group.loc[:, REQUIRED_20_COLS].copy()
    else:
        if not loci_group:
            return _empty_phas_clusters_frame()

        # Guard against accidental wrong payloads (e.g. 6-col merged-candidates)
        width = len(loci_group[0])
        if width != len(REQUIRED_20_COLS):
            return _empty_phas_clusters_frame()

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


def process_phas_cluster_batch(batch) -> pd.DataFrame:
    """Worker for one bounded piece of a chromosome/library group.

    A group can span several batches, but each task stays within one group so
    the original filtering and identifier logic remains unchanged.  Results are
    committed in task order by ``run_parallel_with_progress``.
    """
    (chromosome, alib), batch_df = batch
    processed_df = process_chromosome_data(batch_df)
    processed_df["chromosome"] = chromosome
    processed_df["alib"] = alib
    return processed_df


class _PhasClusterBatchSequence:
    """Lazily materialize fixed-row PHAS tasks from one source DataFrame.

    The task plan contains only group keys and integer row-position views.  The
    parallel runner slices this object, so at most its current bounded task
    window is materialized and queued for workers rather than every chromosome
    group being copied into Python lists up front.
    """

    def __init__(self, source: pd.DataFrame, column_positions, tasks):
        self._source = source
        # pandas ``.iloc`` accepts a list/array as its column selector; a tuple
        # is interpreted as nested indexing on current pandas releases.
        self._column_positions = [int(position) for position in column_positions]
        self._tasks = list(tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        key, positions = self._tasks[index]
        # Advanced integer indexing gives a standalone, bounded task frame.
        # Do not call ``.copy()`` again: that would transiently double every
        # in-flight batch without adding worker isolation.
        batch_df = self._source.iloc[positions, self._column_positions]
        return key, batch_df


def _build_phas_cluster_batch_sequence(
    all_clusters: pd.DataFrame,
    *,
    batch_rows: int,
) -> tuple[_PhasClusterBatchSequence, int]:
    """Plan group-major fixed-row batches without copying group DataFrames."""
    grouping_columns = ["chromosome", "alib"]
    column_positions = all_clusters.columns.get_indexer(REQUIRED_20_COLS)
    if (column_positions < 0).any():
        raise ValueError("allClusters is missing required PHAS-cluster columns")

    target_rows = max(1, int(batch_rows))
    grouped = all_clusters.groupby(grouping_columns, sort=False)

    # ``grouped.indices`` is not guaranteed to follow groupby iteration order
    # for a multi-key grouping.  ``ngroup`` labels do, so selecting each first
    # non-null group code preserves the exact legacy group-major output order.
    group_codes = grouped.ngroup()
    first_group_rows = all_clusters.loc[
        group_codes.notna() & ~group_codes.duplicated(), grouping_columns
    ]

    tasks = []
    group_count = 0
    group_indices = grouped.indices
    for key in first_group_rows.itertuples(index=False, name=None):
        positions = group_indices[key]
        group_count += 1
        for start in range(0, len(positions), target_rows):
            tasks.append((key, positions[start:start + target_rows]))

    return _PhasClusterBatchSequence(all_clusters, column_positions, tasks), group_count


class _PhasClusterOutputWriter:
    """Ordered, disk-backed result consumer for PHAS-cluster worker batches."""

    def __init__(self, output_file: str):
        output_dir = os.path.dirname(os.path.abspath(output_file)) or os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".phasis_phas_cluster_",
            suffix=".tmp",
            dir=output_dir,
        )
        self.output_file = output_file
        self.temporary_path = temporary_path
        self._handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="")
        self.wrote_rows = False
        self.error_count = 0
        self.first_error = None
        self._closed = False

    def __call__(self, result) -> None:
        if isinstance(result, RuntimeError):
            self.error_count += 1
            if self.first_error is None:
                self.first_error = result
            return
        if not isinstance(result, pd.DataFrame):
            self.error_count += 1
            if self.first_error is None:
                self.first_error = RuntimeError(
                    f"Unexpected PHAS-cluster worker result: {type(result).__name__}"
                )
            return
        if result.empty:
            return
        result.to_csv(
            self._handle,
            sep="\t",
            encoding="utf-8",
            index=False,
            header=not self.wrote_rows,
        )
        self.wrote_rows = True

    def finish(self) -> bool:
        """Close and atomically publish a nonempty temporary result table."""
        if not self._closed:
            self._handle.close()
            self._closed = True
        if not self.wrote_rows:
            self.discard()
            return False
        os.replace(self.temporary_path, self.output_file)
        return True

    def discard(self) -> None:
        """Close and remove a partial temporary table."""
        if not self._closed:
            self._handle.close()
            self._closed = True
        if os.path.exists(self.temporary_path):
            try:
                os.remove(self.temporary_path)
            except OSError:
                pass


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
    return_dataframe: bool = True,
) -> pd.DataFrame | str | None:
    """
    Build per-(chromosome, library) PHAS cluster rows in parallel and write to TSV.

    Skips recomputation if output exists and matches the hash stored in memFile.

    ``return_dataframe=False`` is the memory-safe pipeline path: it writes the
    table and returns its logical path, allowing the caller to release the raw
    processed-cluster DataFrame before loading the PHAS table needed by
    later Phase II stages.  The default remains a DataFrame for API
    compatibility with existing callers.

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
    if cache.hit("PHAS_TO_DETECT", output_file, input_sig):
        print(f"  - Output up-to-date (hash+sig match). Skipping processing: {output_file}")
        # Avoid loading the completed table here for the memory-safe pipeline
        # path; it first releases the much larger raw candidate table.
        if not return_dataframe:
            return output_file
        return load_phas_to_detect_output(output_file)

    # ---- Accept only the 20-col per-read schema; else load the processed tab ----
    if not (isinstance(allClusters, pd.DataFrame) and REQUIRED_20_SET.issubset(set(allClusters.columns))):
        allClusters = load_processed_clusters_fallback()

    # ---- If still empty, bail cleanly ----
    if allClusters is None or getattr(allClusters, "empty", True):
        print("  - Found 0 (chromosome, library) groups (empty input). Returning empty DataFrame.")
        return _empty_phas_clusters_frame() if return_dataframe else None

    # ---- Ensure grouping columns exist / normalize ----
    if "chromosome" not in allClusters.columns and "chr" in allClusters.columns:
        allClusters = allClusters.rename(columns={"chr": "chromosome"})

    if "alib" not in allClusters.columns:
        if bool(concat_local):
            allClusters = allClusters.copy()
            allClusters["alib"] = "ALL_LIBS"
        else:
            print("[WARN] 'alib' column missing and not in concat mode; returning empty DataFrame.")
            return _empty_phas_clusters_frame() if return_dataframe else None

    # ---- Enforce EXACT 20-column payload (drop extras like 'identifier') ----
    if not REQUIRED_20_SET.issubset(set(allClusters.columns)):
        allClusters = load_processed_clusters_fallback()
        if allClusters is None or getattr(allClusters, "empty", True):
            print("  - Input invalid and fallback empty; returning empty DataFrame.")
            return _empty_phas_clusters_frame() if return_dataframe else None

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

    # ---- Group-major, fixed-row tasks (lazily materialized) ----
    batch_rows = _phas_cluster_batch_rows()
    cluster_batches, group_count = _build_phas_cluster_batch_sequence(
        allClusters,
        batch_rows=batch_rows,
    )
    print(
        f"  - Found {group_count} (chromosome, library) groups; planned "
        f"{len(cluster_batches)} fixed-row batch(es) of up to {batch_rows:,} rows"
    )

    if not cluster_batches:
        print("  - No groups to process. Returning empty DataFrame.")
        return _empty_phas_clusters_frame() if return_dataframe else None

    parallel_kwargs = _phas_cluster_parallel_kwargs(len(cluster_batches))
    print(
        "  - PHAS-cluster assembly starts with "
        f"{parallel_kwargs['initial_worker_cap']} concurrent batch(es) and can grow to "
        f"{parallel_kwargs['max_worker_cap']}; set PHASIS_PHAS_CLUSTER_WORKER_CAP or "
        "PHASIS_PHAS_CLUSTER_BATCH_ROWS to override."
    )

    # Stream ordered worker output directly to a temporary table.  This avoids
    # retaining a list of every result frame and avoids the final concat peak.
    writer = _PhasClusterOutputWriter(output_file)
    try:
        run_parallel_with_progress(
            process_phas_cluster_batch,
            cluster_batches,
            desc="Building PHAS cluster batches",
            min_chunk=1,
            unit="lib-chr",
            on_result=writer,
            return_results=False,
            **parallel_kwargs,
        )
        wrote_output = writer.finish()
    except Exception:
        writer.discard()
        raise

    # Surface worker failures while preserving the prior behavior of retaining
    # successfully processed rows from other tasks.
    if writer.error_count:
        print("[WARN] One or more worker tasks failed; filtering to successful results. First error:")
        print(writer.first_error)

    if not wrote_output:
        print("  - All worker results empty. Returning empty DataFrame.")
        return _empty_phas_clusters_frame() if return_dataframe else None

    # ---- Write + update md5 cache (best effort) ----
    fp = finalize_text_artifact(cache, "PHAS_TO_DETECT", output_file, input_sig)
    if fp:
        print(f"  - Wrote {output_file} (md5: {fp})")
    else:
        print(f"  - Wrote {output_file}")

    if not return_dataframe:
        return output_file
    return load_phas_to_detect_output(output_file)
