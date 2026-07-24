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

For large Phase II runs, ``select_scoring_windows_from_path`` consumes the
completed PHAS-to-detect TSV through fixed-size, whole-cluster batches instead
of first materializing every per-read PHAS row in one DataFrame.

Constraints:
- spawn-safe (top-level functions only)
- no nested functions; no imports inside functions
- runtime-first: defaults come from phasis.runtime, but explicit args are supported
"""

import multiprocessing
import os
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import phasis.runtime as rt
from phasis.cache import (
    MemCache,
    default_memfile_path,
    finalize_text_artifact,
    phase2_basename,
    resolve_artifact_path,
    stage_signature,
)
from phasis.env import getenv
from phasis.parallel import run_parallel_with_progress

WINDOWS_COLUMNS: List[str] = [
    "cluster_id",
    "window_n",
    "fw_pval_corr",
    "rv_pval_corr",
    "combined_window_p_value",
]

# The path-backed Phase II route reads only the fields used by this stage.  It
# intentionally does not load the full PHAS-to-detect table just to select
# scoring windows.
WINDOW_SELECTION_INPUT_COLUMNS: Tuple[str, ...] = (
    "clusterID",
    "pos",
    "pval_corr_f",
    "pval_corr_r",
    "chromosome",
)
WINDOW_SELECTION_STREAM_BATCH_ROWS_DEFAULT = 100_000
WINDOW_SELECTION_INITIAL_WORKER_CAP = 2
WINDOW_SELECTION_DEFAULT_MAX_CPU_FRACTION_NUMERATOR = 7
WINDOW_SELECTION_DEFAULT_MAX_CPU_FRACTION_DENOMINATOR = 10


def load_window_chunk_file(path: str):
    physical_path = resolve_artifact_path(path)
    if not physical_path or os.path.getsize(physical_path) <= 0:
        return (path, None)

    try:
        frame = pd.read_csv(physical_path, sep="\t", engine="python")
    except Exception:
        frame = pd.read_csv(physical_path, sep="\t")
    return (path, frame)


def _safe_key(akey: str) -> str:
    """Normalize an akey to a filesystem-safe basename."""
    s = str(akey)
    # Drop any path components to avoid directory traversal
    s = os.path.basename(s)
    # Avoid accidental separators on Windows-like paths (harmless on macOS/Linux)
    s = s.replace(os.sep, "_")
    return s


def _load_final_if_cached(
    cache: MemCache, outfname: str, input_sig: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Return cached final dataframe if cache hit; else None."""
    if not cache.hit("WINDOWS_TO_SCORE", outfname, input_sig):
        return None

    print(f"  - Output up-to-date (hash+sig match). Skipping computation: {outfname}")
    physical_outfname = resolve_artifact_path(outfname) or outfname
    try:
        df = pd.read_csv(physical_outfname, sep="\t", engine="python")
    except Exception:
        df = pd.read_csv(physical_outfname, sep="\t")

    for c in ("window_n", "fw_pval_corr", "rv_pval_corr", "combined_window_p_value"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _record_final(cache: MemCache, outfname: str, input_sig: Optional[str] = None) -> None:
    """Record outfname fingerprint and (optional) signature into phasis.mem."""
    fp = finalize_text_artifact(cache, "WINDOWS_TO_SCORE", outfname, input_sig)
    if fp:
        print(f"  - Wrote {outfname} (md5: {fp})")
    else:
        print(f"  - Wrote {outfname}")


def _coerce_positive_int(value, default: int) -> int:
    """Return a positive integer, falling back without raising on bad knobs."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return int(parsed) if parsed > 0 else int(default)


def _window_selection_batch_rows(explicit: Optional[int] = None) -> int:
    """Resolve the bounded input-row count for one path-backed window task."""
    if explicit is not None:
        return _coerce_positive_int(explicit, WINDOW_SELECTION_STREAM_BATCH_ROWS_DEFAULT)
    configured = getattr(rt, "window_selection_batch_rows", None)
    if configured is None:
        configured = getenv("Phasis_WINDOW_SELECTION_BATCH_ROWS")
    return _coerce_positive_int(configured, WINDOW_SELECTION_STREAM_BATCH_ROWS_DEFAULT)


def _window_selection_ncores() -> int:
    """Return the allocated Phasis CPU count with a safe fallback."""
    return _coerce_positive_int(getattr(rt, "ncores", None), multiprocessing.cpu_count())


def _window_selection_worker_cap() -> int:
    """Bound path-backed window tasks like the other memory-sensitive Phase II stages."""
    ncores = _window_selection_ncores()
    default_cap = max(
        1,
        (ncores * WINDOW_SELECTION_DEFAULT_MAX_CPU_FRACTION_NUMERATOR)
        // WINDOW_SELECTION_DEFAULT_MAX_CPU_FRACTION_DENOMINATOR,
    )
    configured = getattr(rt, "window_selection_worker_cap", None)
    if configured is None:
        configured = getenv("Phasis_WINDOW_SELECTION_WORKER_CAP")
    return min(ncores, _coerce_positive_int(configured, default_cap))


def _window_selection_parallel_kwargs(task_count: int) -> Dict[str, Any]:
    """Keep only a bounded window of fixed-size task DataFrames in flight."""
    max_worker_cap = _window_selection_worker_cap()
    initial_worker_cap = min(WINDOW_SELECTION_INITIAL_WORKER_CAP, max_worker_cap)
    initial_task_window = max(1, min(int(task_count), initial_worker_cap))
    max_task_window = max(1, min(int(task_count), max_worker_cap))
    return {
        "initial_worker_cap": initial_worker_cap,
        "max_worker_cap": max_worker_cap,
        "initial_chunk_size": initial_task_window,
        "max_chunk_size": max_task_window,
        "adaptive_recovery": True,
    }


def _window_group_key_name(group_key: Tuple[str, str]) -> str:
    """Return the legacy cache-key stem for a ``(chromosome, library)`` pair."""
    chromosome, library = group_key
    return f"{library}__chr{chromosome}"


def _window_group_outpath(outdir: str, group_key: Tuple[str, str]) -> str:
    return os.path.join(outdir, f"{_safe_key(_window_group_key_name(group_key))}.tsv")


def _nonempty_artifact(path: str) -> bool:
    """Check an existing logical artifact without treating an empty file as cached."""
    physical_path = resolve_artifact_path(path)
    if not physical_path:
        return False
    try:
        return os.path.getsize(physical_path) > 0
    except OSError:
        return False


def _streaming_window_input_layout(phas_path: str) -> Dict[str, Any]:
    """Inspect a PHAS TSV header and describe its minimal window-selection projection."""
    physical_path = resolve_artifact_path(phas_path) or str(phas_path)
    if not os.path.isfile(physical_path):
        raise FileNotFoundError(f"PHAS table not found: {phas_path}")

    header = pd.read_csv(physical_path, sep="\t", nrows=0)
    available = set(header.columns)
    chromosome_source = "chromosome" if "chromosome" in available else "chr"
    required_source = ["clusterID", "pos", "pval_corr_f", "pval_corr_r", chromosome_source]
    missing = [column for column in required_source if column not in available]
    if missing:
        raise ValueError(
            "select_scoring_windows_from_path(): missing required columns: "
            f"{missing}"
        )

    has_alib = "alib" in available
    return {
        "physical_path": physical_path,
        "chromosome_source": chromosome_source,
        "has_alib": has_alib,
    }


def _iter_streaming_window_input_frames(
    layout: Dict[str, Any],
    *,
    read_rows: int,
    keys_only: bool = False,
) -> Iterator[pd.DataFrame]:
    """Yield a bounded key-only or six-field window projection from the PHAS TSV."""
    chromosome_source = str(layout["chromosome_source"])
    has_alib = bool(layout["has_alib"])
    text_columns = {"clusterID": str, chromosome_source: str}
    if has_alib:
        text_columns["alib"] = str
    usecols = ["clusterID", chromosome_source]
    if not keys_only:
        usecols.extend(["pos", "pval_corr_f", "pval_corr_r"])
    if has_alib:
        usecols.append("alib")

    reader = pd.read_csv(
        str(layout["physical_path"]),
        sep="\t",
        usecols=usecols,
        dtype=text_columns,
        keep_default_na=False,
        chunksize=max(1, int(read_rows)),
        # The historical Python parser effectively used round-trip conversion
        # for p-values. Keep that behavior while using the much leaner C parser.
        float_precision="round_trip",
    )
    try:
        for frame in reader:
            if chromosome_source != "chromosome":
                frame = frame.rename(columns={chromosome_source: "chromosome"})
            if not has_alib:
                frame["alib"] = "concat"
            if keys_only:
                yield frame.loc[:, ["clusterID", "chromosome", "alib"]]
            else:
                yield frame.loc[:, list(WINDOW_SELECTION_INPUT_COLUMNS) + ["alib"]]
    finally:
        # A malformed manually supplied table can make the first planning pass
        # raise early. Explicitly close TextFileReader in that case so repeated
        # retries do not leak file descriptors on a long-lived HPC worker.
        close = getattr(reader, "close", None)
        if callable(close):
            close()


def _iter_contiguous_window_runs(
    frame: pd.DataFrame,
) -> Iterator[Tuple[Tuple[str, str], str, pd.DataFrame]]:
    """Yield contiguous ``(chromosome, library, cluster)`` runs from one reader chunk."""
    if frame.empty:
        return

    chromosomes = frame["chromosome"].to_numpy(copy=False)
    libraries = frame["alib"].to_numpy(copy=False)
    cluster_ids = frame["clusterID"].to_numpy(copy=False)
    row_count = len(frame)
    boundaries = np.empty(row_count, dtype=bool)
    boundaries[0] = True
    if row_count > 1:
        boundaries[1:] = (
            (chromosomes[1:] != chromosomes[:-1])
            | (libraries[1:] != libraries[:-1])
            | (cluster_ids[1:] != cluster_ids[:-1])
        )

    starts = np.flatnonzero(boundaries)
    ends = np.append(starts[1:], row_count)
    for start, end in zip(starts, ends):
        group_key = (str(chromosomes[start]), str(libraries[start]))
        cluster_id = str(cluster_ids[start])
        yield group_key, cluster_id, frame.iloc[int(start):int(end)]


class _StreamingWindowTaskPlan:
    """First-pass, constant-memory task plan for a group-major PHAS table."""

    def __init__(self, *, outdir: str, batch_rows: int) -> None:
        self.outdir = str(outdir)
        self.batch_rows = max(1, int(batch_rows))
        self.group_count = 0
        self.task_count = 0
        self.oversized_clusters = 0
        self.kept_paths: List[str] = []
        self.new_paths: List[str] = []
        self.outpaths: Dict[Tuple[str, str], str] = {}
        self.cached_groups = set()
        self._seen_groups = set()
        self._closed_cluster_ids = set()
        self._current_group: Optional[Tuple[str, str]] = None
        self._current_cluster: Optional[str] = None
        self._current_cluster_rows = 0
        self._current_batch_rows = 0
        self._current_group_cached = False

    def _start_group(self, group_key: Tuple[str, str]) -> None:
        if group_key in self._seen_groups:
            raise ValueError(
                "PHAS-to-detect records are not contiguous by chromosome/library; "
                "cannot safely stream scoring-window batches."
            )
        self._seen_groups.add(group_key)
        self.group_count += 1
        outpath = _window_group_outpath(self.outdir, group_key)
        self.outpaths[group_key] = outpath
        self._current_group = group_key
        self._current_group_cached = _nonempty_artifact(outpath)
        if self._current_group_cached:
            self.cached_groups.add(group_key)
            self.kept_paths.append(outpath)
        else:
            self.new_paths.append(outpath)
        self._closed_cluster_ids = set()
        self._current_cluster = None
        self._current_cluster_rows = 0
        self._current_batch_rows = 0

    def _finish_cluster(self) -> None:
        if self._current_cluster is None:
            return
        if not self._current_group_cached:
            cluster_rows = int(self._current_cluster_rows)
            if cluster_rows > self.batch_rows:
                self.oversized_clusters += 1
            if self._current_batch_rows and (
                self._current_batch_rows + cluster_rows > self.batch_rows
            ):
                self.task_count += 1
                self._current_batch_rows = 0
            self._current_batch_rows += cluster_rows
            if self._current_batch_rows >= self.batch_rows:
                self.task_count += 1
                self._current_batch_rows = 0
        self._closed_cluster_ids.add(self._current_cluster)
        self._current_cluster = None
        self._current_cluster_rows = 0

    def _finish_group(self) -> None:
        if self._current_group is None:
            return
        self._finish_cluster()
        if not self._current_group_cached and self._current_batch_rows:
            self.task_count += 1
        self._current_group = None
        self._current_batch_rows = 0

    def consume_run(
        self,
        group_key: Tuple[str, str],
        cluster_id: str,
        row_count: int,
    ) -> None:
        if group_key != self._current_group:
            self._finish_group()
            self._start_group(group_key)

        if self._current_cluster is None:
            if cluster_id in self._closed_cluster_ids:
                raise ValueError(
                    "PHAS-to-detect records are not contiguous by cluster within "
                    f"{_window_group_key_name(group_key)!r}; cannot safely stream "
                    "whole-cluster scoring-window batches."
                )
            self._current_cluster = cluster_id
        elif cluster_id != self._current_cluster:
            self._finish_cluster()
            if cluster_id in self._closed_cluster_ids:
                raise ValueError(
                    "PHAS-to-detect records are not contiguous by cluster within "
                    f"{_window_group_key_name(group_key)!r}; cannot safely stream "
                    "whole-cluster scoring-window batches."
                )
            self._current_cluster = cluster_id
        self._current_cluster_rows += int(row_count)

    def finish(self) -> None:
        self._finish_group()


def _plan_streaming_window_tasks(
    layout: Dict[str, Any],
    *,
    outdir: str,
    batch_rows: int,
) -> _StreamingWindowTaskPlan:
    """Count fixed whole-cluster tasks without retaining PHAS rows in memory."""
    plan = _StreamingWindowTaskPlan(outdir=outdir, batch_rows=batch_rows)
    for frame in _iter_streaming_window_input_frames(
        layout,
        read_rows=batch_rows,
        keys_only=True,
    ):
        for group_key, cluster_id, run_frame in _iter_contiguous_window_runs(frame):
            plan.consume_run(group_key, cluster_id, len(run_frame))
    plan.finish()
    return plan


class _WholeClusterWindowBatchEmitter:
    """Second-pass state machine that materializes only one fixed-size task at a time."""

    def __init__(
        self,
        *,
        plan: _StreamingWindowTaskPlan,
        window_len: int,
        sliding: int,
        min_cluster_length: int,
    ) -> None:
        self.plan = plan
        self.window_len = int(window_len)
        self.sliding = int(sliding)
        self.min_cluster_length = int(min_cluster_length)
        self._seen_groups = set()
        self._closed_cluster_ids = set()
        self._current_group: Optional[Tuple[str, str]] = None
        self._current_cluster: Optional[str] = None
        self._current_group_cached = False
        self._cluster_parts: List[pd.DataFrame] = []
        self._batch_parts: List[pd.DataFrame] = []
        self._batch_rows = 0

    def _start_group(self, group_key: Tuple[str, str]) -> None:
        if group_key in self._seen_groups:
            raise ValueError(
                "PHAS-to-detect records are not contiguous by chromosome/library; "
                "cannot safely stream scoring-window batches."
            )
        if group_key not in self.plan.outpaths:
            raise RuntimeError("Streaming window-task plan does not match its PHAS input.")
        self._seen_groups.add(group_key)
        self._current_group = group_key
        self._current_group_cached = group_key in self.plan.cached_groups
        self._closed_cluster_ids = set()
        self._current_cluster = None
        self._cluster_parts = []
        self._batch_parts = []
        self._batch_rows = 0

    def _emit_batch(self) -> List[Dict[str, Any]]:
        if self._current_group_cached or not self._batch_parts:
            return []
        batch_frame = pd.concat(self._batch_parts, ignore_index=True, copy=False)
        task = {
            "key": _window_group_key_name(self._current_group),
            "df": batch_frame,
            "outpath": self.plan.outpaths[self._current_group],
            "window_len": self.window_len,
            "sliding": self.sliding,
            "minClusterLength": self.min_cluster_length,
        }
        self._batch_parts = []
        self._batch_rows = 0
        return [task]

    def _finish_cluster(self) -> List[Dict[str, Any]]:
        if self._current_cluster is None:
            return []
        emitted: List[Dict[str, Any]] = []
        if not self._current_group_cached:
            cluster_frame = pd.concat(self._cluster_parts, ignore_index=True, copy=False)
            cluster_rows = len(cluster_frame)
            if self._batch_rows and self._batch_rows + cluster_rows > self.plan.batch_rows:
                emitted.extend(self._emit_batch())
            self._batch_parts.append(cluster_frame)
            self._batch_rows += cluster_rows
            if self._batch_rows >= self.plan.batch_rows:
                emitted.extend(self._emit_batch())
        self._closed_cluster_ids.add(self._current_cluster)
        self._current_cluster = None
        self._cluster_parts = []
        return emitted

    def _finish_group(self) -> List[Dict[str, Any]]:
        if self._current_group is None:
            return []
        emitted = self._finish_cluster()
        emitted.extend(self._emit_batch())
        self._current_group = None
        return emitted

    def consume_run(
        self,
        group_key: Tuple[str, str],
        cluster_id: str,
        run_frame: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        emitted: List[Dict[str, Any]] = []
        if group_key != self._current_group:
            emitted.extend(self._finish_group())
            self._start_group(group_key)

        if self._current_cluster is None:
            if cluster_id in self._closed_cluster_ids:
                raise ValueError(
                    "PHAS-to-detect records are not contiguous by cluster within "
                    f"{_window_group_key_name(group_key)!r}; cannot safely stream "
                    "whole-cluster scoring-window batches."
                )
            self._current_cluster = cluster_id
        elif cluster_id != self._current_cluster:
            emitted.extend(self._finish_cluster())
            if cluster_id in self._closed_cluster_ids:
                raise ValueError(
                    "PHAS-to-detect records are not contiguous by cluster within "
                    f"{_window_group_key_name(group_key)!r}; cannot safely stream "
                    "whole-cluster scoring-window batches."
                )
            self._current_cluster = cluster_id

        if not self._current_group_cached:
            self._cluster_parts.append(run_frame)
        return emitted

    def finish(self) -> List[Dict[str, Any]]:
        return self._finish_group()


class _StreamingWindowTaskSequence:
    """Lazy, retry-safe sequence consumed by ``run_parallel_with_progress``.

    The shared parallel runner asks for monotonically increasing slices, but it
    may retry a slice with fewer tasks.  Retaining only the current uncommitted
    slice lets retries reuse the same bounded DataFrames without preloading all
    PHAS records or spilling a duplicate copy to disk.
    """

    def __init__(
        self,
        *,
        layout: Dict[str, Any],
        plan: _StreamingWindowTaskPlan,
        window_len: int,
        sliding: int,
        min_cluster_length: int,
    ) -> None:
        self.layout = layout
        self.plan = plan
        self.window_len = int(window_len)
        self.sliding = int(sliding)
        self.min_cluster_length = int(min_cluster_length)
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._next_index = 0
        self._task_iterator = self._iter_tasks()

    def __len__(self) -> int:
        return int(self.plan.task_count)

    def _iter_tasks(self) -> Iterator[Dict[str, Any]]:
        emitter = _WholeClusterWindowBatchEmitter(
            plan=self.plan,
            window_len=self.window_len,
            sliding=self.sliding,
            min_cluster_length=self.min_cluster_length,
        )
        for frame in _iter_streaming_window_input_frames(
            self.layout,
            read_rows=self.plan.batch_rows,
        ):
            for group_key, cluster_id, run_frame in _iter_contiguous_window_runs(frame):
                for task in emitter.consume_run(group_key, cluster_id, run_frame):
                    yield task
        for task in emitter.finish():
            yield task

    def _discard_before(self, index: int) -> None:
        stale = [item for item in self._cache if item < int(index)]
        for item in stale:
            del self._cache[item]

    def _ensure_through(self, stop: int) -> None:
        while self._next_index < int(stop):
            try:
                self._cache[self._next_index] = next(self._task_iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "Streaming window-task plan ended before its expected task count."
                ) from exc
            self._next_index += 1

    def _verify_exhausted(self) -> None:
        if self._next_index != len(self):
            return
        try:
            next(self._task_iterator)
        except StopIteration:
            return
        raise RuntimeError(
            "Streaming window-task plan produced more tasks than its first-pass count."
        )

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return [self[item] for item in range(start, stop, step)]
            self._discard_before(start)
            self._ensure_through(stop)
            if stop == len(self):
                self._verify_exhausted()
            return [self._cache[item] for item in range(start, stop)]

        item = int(index)
        if item < 0:
            item += len(self)
        if item < 0 or item >= len(self):
            raise IndexError(item)
        self._discard_before(item)
        self._ensure_through(item + 1)
        if item + 1 == len(self):
            self._verify_exhausted()
        return self._cache[item]


def select_windows_stream_task_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Compute one fixed whole-cluster batch; the parent streams its rows to disk."""
    rows = select_windows_for_chromosome(
        task["df"],
        window_len=int(task["window_len"]),
        sliding=int(task["sliding"]),
        minClusterLength=int(task["minClusterLength"]),
    )
    return {
        "outpath": str(task["outpath"]),
        "key": str(task.get("key", "")),
        "rows": rows,
    }


class _StreamingWindowChunkWriter:
    """Atomically publish per-group chunk files without retaining worker rows."""

    def __init__(self, outpaths: Sequence[str]) -> None:
        self.outpaths = list(dict.fromkeys(str(path) for path in outpaths))
        self.temporary_paths: Dict[str, str] = {}
        self.first_error = None
        self.bad_results = 0

    def prepare(self) -> None:
        for outpath in self.outpaths:
            output_dir = os.path.dirname(os.path.abspath(outpath)) or os.getcwd()
            os.makedirs(output_dir, exist_ok=True)
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=".phasis_window_selection_",
                suffix=".tmp",
                dir=output_dir,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                pd.DataFrame(columns=WINDOWS_COLUMNS).to_csv(handle, sep="\t", index=False)
            self.temporary_paths[outpath] = temporary_path

    def __call__(self, result) -> None:
        if isinstance(result, RuntimeError):
            self.bad_results += 1
            if self.first_error is None:
                self.first_error = result
            return
        if not isinstance(result, dict):
            self.bad_results += 1
            if self.first_error is None:
                self.first_error = RuntimeError(
                    f"Unexpected streaming window worker result: {type(result).__name__}"
                )
            return

        outpath = str(result.get("outpath") or "")
        rows = result.get("rows")
        temporary_path = self.temporary_paths.get(outpath)
        if temporary_path is None or not isinstance(rows, list):
            self.bad_results += 1
            if self.first_error is None:
                self.first_error = RuntimeError("Malformed streaming window worker result.")
            return
        if not rows:
            return
        pd.DataFrame(rows, columns=WINDOWS_COLUMNS).to_csv(
            temporary_path,
            sep="\t",
            index=False,
            mode="a",
            header=False,
        )

    def discard(self) -> None:
        for temporary_path in self.temporary_paths.values():
            try:
                os.remove(temporary_path)
            except OSError:
                pass
        self.temporary_paths = {}

    def finish(self) -> None:
        if self.bad_results:
            error = self.first_error or RuntimeError("Streaming window worker failed.")
            self.discard()
            raise error
        for outpath, temporary_path in self.temporary_paths.items():
            os.replace(temporary_path, outpath)
        self.temporary_paths = {}


def _merge_window_chunk_paths(kept_paths: Sequence[str]) -> pd.DataFrame:
    """Apply the historical chunk merge and final ordering contract."""
    paths = sorted(set(str(path) for path in kept_paths))
    frames: List[pd.DataFrame] = []
    if paths:
        print(f"  - Loading {len(paths)} cached/new window chunk(s) for merge")
        loaded_chunks = run_parallel_with_progress(
            load_window_chunk_file,
            paths,
            desc="Loading window chunks",
            min_chunk=1,
            unit="file",
        ) or []
        worker_errors = [result for result in loaded_chunks if isinstance(result, RuntimeError)]
        if worker_errors:
            raise worker_errors[0]
        for _path, frame in loaded_chunks:
            if frame is not None:
                frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=WINDOWS_COLUMNS)

    to_score = pd.concat(frames, ignore_index=True)
    for column in ("window_n", "fw_pval_corr", "rv_pval_corr", "combined_window_p_value"):
        if column in to_score.columns:
            to_score[column] = pd.to_numeric(to_score[column], errors="coerce")
    sort_cols = [column for column in ("cluster_id", "window_n") if column in to_score.columns]
    if sort_cols:
        to_score = to_score.sort_values(sort_cols, kind="mergesort")
    return to_score


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
    memFile_local = memFile_local or default_memfile_path()
    cache = MemCache.load(memFile_local)

    if wl <= 0 or sl <= 0:
        raise ValueError(f"Invalid window_len/sliding: window_len={wl}, sliding={sl}")

    outfname = phase2_basename("clusters_windows_to_score.tsv")

    # Encode key runtime params in the directory name to segregate caches across settings
    outdir = phase2_basename(f"windows_sl{sl}_wl{wl}_mcl{mcl}")
    os.makedirs(outdir, exist_ok=True)

    # Signature from upstream PHAS_to_detect + key parameters
    phas_tab = phase2_basename("PHAS_to_detect.tab")
    input_sig = stage_signature(
        files=[phas_tab],
        params={"window_len": wl, "sliding": sl, "minClusterLength": mcl},
    )

    # Early return on final up-to-date file (hash+sig match)
    cached = _load_final_if_cached(cache, outfname, input_sig)
    if cached is not None:
        return cached

    # --- Normalize/guard input ---
    required_in = ["clusterID", "pos", "pval_corr_f", "pval_corr_r", "chromosome"]

    if clusters_data is None or getattr(clusters_data, "empty", True):
        print("[INFO] No clusters to select windows from; writing empty output.")
        empty_out = pd.DataFrame(columns=WINDOWS_COLUMNS)
        empty_out.to_csv(outfname, sep="\t", index=False)
        _record_final(cache, outfname, input_sig)
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
        _record_final(cache, outfname, input_sig)
        return empty_out

    # --- Build lib‑chr groups ---
    grouping = ["chromosome"] + (["alib"] if "alib" in clusters_data.columns else [])
    groups = [
        (k, df)
        for k, df in clusters_data.groupby(grouping, sort=False, observed=True)
    ]
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

        physical_outp = resolve_artifact_path(outp)
        if physical_outp and os.path.getsize(physical_outp) > 0:
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
    if kept_paths:
        print(f"  - Loading {len(kept_paths)} cached/new window chunk(s) for merge")
        loaded_chunks = run_parallel_with_progress(
            load_window_chunk_file,
            kept_paths,
            desc="Loading window chunks",
            min_chunk=1,
            unit="file",
        ) or []
        worker_errors = [r for r in loaded_chunks if isinstance(r, RuntimeError)]
        if worker_errors:
            raise worker_errors[0]
        for path, frame in loaded_chunks:
            if frame is not None:
                frames.append(frame)

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
    _record_final(cache, outfname, input_sig)

    print(f"    Cached chunks directory: {outdir}")
    return to_score


def select_scoring_windows_from_path(
    phas_path: str,
    *,
    window_len: Optional[int] = None,
    sliding: Optional[int] = None,
    minClusterLength: Optional[int] = None,
    memFile: Optional[str] = None,
    batch_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Select scoring windows from a disk-backed PHAS-to-detect table.

    This is the memory-safe counterpart to :func:`select_scoring_windows`.
    It scans only six input fields, carries a cluster across CSV-reader chunk
    boundaries, and dispatches fixed-size batches that never split a cluster.
    The PHAS-cluster builder emits chromosome/library-major, cluster-major rows;
    the two lightweight passes validate that invariant so a malformed manual
    table cannot silently lose rows.

    The per-library/chromosome chunk filenames, final TSV schema, cache section,
    sorting, and returned ``DataFrame`` remain the same as the DataFrame API.
    ``batch_rows`` (or ``Phasis_WINDOW_SELECTION_BATCH_ROWS``) bounds one worker
    input; ``Phasis_WINDOW_SELECTION_WORKER_CAP`` bounds concurrency.
    """
    print("### Step: select scoring windows from disk-backed PHAS clusters ###")

    wl = int(window_len if window_len is not None else getattr(rt, "window_len", 0) or 0)
    sl = int(sliding if sliding is not None else getattr(rt, "sliding", 0) or 0)
    mcl = int(
        minClusterLength if minClusterLength is not None else getattr(rt, "minClusterLength", 0) or 0
    )
    if wl <= 0 or sl <= 0:
        raise ValueError(f"Invalid window_len/sliding: window_len={wl}, sliding={sl}")

    memFile_local = memFile if memFile is not None else getattr(rt, "memFile", None)
    memFile_local = memFile_local or default_memfile_path()
    cache = MemCache.load(memFile_local)
    outfname = phase2_basename("clusters_windows_to_score.tsv")
    outdir = phase2_basename(f"windows_sl{sl}_wl{wl}_mcl{mcl}")
    os.makedirs(outdir, exist_ok=True)
    input_sig = stage_signature(
        files=[str(phas_path)],
        params={"window_len": wl, "sliding": sl, "minClusterLength": mcl},
    )

    cached = _load_final_if_cached(cache, outfname, input_sig)
    if cached is not None:
        return cached

    resolved_batch_rows = _window_selection_batch_rows(batch_rows)
    layout = _streaming_window_input_layout(str(phas_path))
    print(
        "[INFO] Streaming PHAS records into whole-cluster scoring-window batches "
        f"({resolved_batch_rows:,} rows per batch)."
    )
    plan = _plan_streaming_window_tasks(
        layout,
        outdir=outdir,
        batch_rows=resolved_batch_rows,
    )

    if not plan.group_count:
        print("[INFO] No clusters to select windows from; writing empty output.")
        empty_out = pd.DataFrame(columns=WINDOWS_COLUMNS)
        empty_out.to_csv(outfname, sep="\t", index=False)
        _record_final(cache, outfname, input_sig)
        return empty_out

    print(
        f"  - Found {plan.group_count} group(s); {len(plan.kept_paths)} cached chunk(s) "
        f"will be reused; {plan.task_count} fixed whole-cluster batch(es) to compute"
    )
    if plan.oversized_clusters:
        print(
            "[WARN] "
            f"{plan.oversized_clusters} cluster(s) exceed the {resolved_batch_rows:,}-row "
            "batch size and must be processed intact."
        )

    if plan.task_count:
        parallel_kwargs = _window_selection_parallel_kwargs(plan.task_count)
        print(
            "  - Window selection starts with "
            f"{parallel_kwargs['initial_worker_cap']} concurrent batch(es) and can grow to "
            f"{parallel_kwargs['max_worker_cap']}; set "
            "PHASIS_WINDOW_SELECTION_WORKER_CAP or "
            "PHASIS_WINDOW_SELECTION_BATCH_ROWS to override."
        )
        task_sequence = _StreamingWindowTaskSequence(
            layout=layout,
            plan=plan,
            window_len=wl,
            sliding=sl,
            min_cluster_length=mcl,
        )
        writer = _StreamingWindowChunkWriter(plan.new_paths)
        writer.prepare()
        try:
            run_parallel_with_progress(
                select_windows_stream_task_worker,
                task_sequence,
                desc="Selecting windows (disk-backed)",
                min_chunk=1,
                unit="lib-chr",
                on_result=writer,
                return_results=False,
                **parallel_kwargs,
            )
            writer.finish()
        except Exception:
            writer.discard()
            raise

    to_score = _merge_window_chunk_paths(plan.kept_paths + plan.new_paths)
    to_score.to_csv(outfname, sep="\t", index=False)
    _record_final(cache, outfname, input_sig)
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

    for cID, aclust in df.groupby("clusterID", sort=False, observed=True):
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
