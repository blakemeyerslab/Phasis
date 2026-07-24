from __future__ import annotations

"""
phasis.stages.cluster_aggregation
--------------------------------

Phase II helper stage: read one or more *.PHAS.candidate.clusters files and
write {phase}_processed_clusters.tab.

Design constraints:
- spawn-safe (top-level functions only)
- no imports inside functions
- minimal behavior drift vs legacy implementation
"""

import csv
import heapq
import math
import os
import re
import shutil
import tempfile
from typing import Iterator, List, Sequence, Tuple

import pandas as pd

import phasis.runtime as rt
from phasis.cache import (
    MemCache,
    artifact_exists,
    default_memfile_path,
    finalize_text_artifact,
    open_text_artifact,
    phase2_basename,
    resolve_artifact_path,
    stage_signature,
)
from phasis.env import getenv
from phasis.parallel import run_parallel_with_progress


# Canonical column order for processed cluster rows
PROCESSED_CLUSTER_COLUMNS: List[str] = [
    "alib",
    "clusterID",
    "chromosome",
    "strand",
    "pos",
    "len",
    "hits",
    "abun",
    "pval_h_f",
    "N_f",
    "X_f",
    "pval_r_f",
    "pval_corr_f",
    "pval_h_r",
    "N_r",
    "X_r",
    "pval_r_r",
    "pval_corr_r",
    "tag_id",
    "tag_seq",
]
PROCESSED_CLUSTER_TEXT_COLUMNS = (
    "alib",
    "clusterID",
    "chromosome",
    "strand",
    "tag_id",
    "tag_seq",
)
PROCESSED_CLUSTER_NUMERIC_COLUMNS = tuple(
    column for column in PROCESSED_CLUSTER_COLUMNS
    if column not in PROCESSED_CLUSTER_TEXT_COLUMNS
)

CLUSTER_AGGREGATION_CHUNK_ROWS_DEFAULT = 100_000
CLUSTER_AGGREGATION_MERGE_FAN_IN_DEFAULT = 64


def _coerce_positive_int(value, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return int(parsed) if parsed > 0 else int(default)


def _cluster_aggregation_chunk_rows() -> int:
    """Return the bounded number of records held by one sort worker."""
    return _coerce_positive_int(
        getenv("Phasis_CLUSTER_AGGREGATION_CHUNK_ROWS"),
        CLUSTER_AGGREGATION_CHUNK_ROWS_DEFAULT,
    )


def _cluster_aggregation_merge_fan_in() -> int:
    """Return the maximum number of sorted spill files opened by one merge."""
    return max(
        2,
        _coerce_positive_int(
            getenv("Phasis_CLUSTER_AGGREGATION_MERGE_FAN_IN"),
            CLUSTER_AGGREGATION_MERGE_FAN_IN_DEFAULT,
        ),
    )


def _library_name_from_candidate_path(filename: str) -> str:
    """Return the legacy library identifier encoded in a candidate filename."""
    base = os.path.basename(filename)
    return re.sub(r"\.\d+-PHAS\.candidate\.clusters$", "", base)


def _iter_single_lib_cluster_rows(filename: str) -> Iterator[Tuple]:
    """Stream candidate-cluster rows without materializing a whole library."""
    if not artifact_exists(filename):
        raise FileNotFoundError(f"Cluster file not found: {filename}")

    alib = _library_name_from_candidate_path(filename)
    aid = None
    with open_text_artifact(filename, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                # Header example: ">cluster = lobe_3_nocontam-1_3894_1"
                match = re.search(r"cluster\s*=\s*([^\s]+)", line)
                aid = match.group(1).strip() if match else None
                continue

            if not aid:
                continue

            ent = line.rstrip("\n").split("\t")
            if len(ent) < 18:
                continue

            yield (
                alib,
                aid,
                str(ent[0]),
                str(ent[1]),
                int(ent[2]),
                int(ent[3]),
                int(ent[4]),
                int(ent[5]),
                float(ent[6]),
                int(ent[7]),
                int(ent[8]),
                float(ent[9]),
                float(ent[10]),
                float(ent[11]),
                int(ent[12]),
                int(ent[13]),
                float(ent[14]),
                float(ent[15]),
                str(ent[16]),
                str(ent[17]),
            )


def process_single_lib_cluster(filename: str) -> List[Tuple]:
    """
    Parse a single *.PHAS.candidate.clusters file into a list of tuples matching
    PROCESSED_CLUSTER_COLUMNS.
    """
    return list(_iter_single_lib_cluster_rows(filename))


def _spill_row_sort_key(row: Tuple) -> Tuple[str, int, int, int]:
    """Sort by legacy output keys plus source order for deterministic ties."""
    return str(row[1]), int(row[4]), int(row[20]), int(row[21])


def _write_sorted_spill(rows: List[Tuple], path: str) -> None:
    rows.sort(key=_spill_row_sort_key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        # pandas.DataFrame.to_csv(), used by the legacy implementation, writes
        # floating NaN values as empty fields.  Preserve that on-disk contract
        # rather than emitting csv.writer's literal "nan" spelling.
        writer.writerows(
            tuple(
                "" if isinstance(value, float) and math.isnan(value) else value
                for value in row
            )
            for row in rows
        )


def spill_single_lib_cluster_to_sorted_runs(job: Tuple[int, str, str, int]) -> dict:
    """Parse one library in bounded chunks and spill sorted runs to disk.

    This top-level worker is spawn-safe.  It returns only filenames and counts,
    never the parsed candidate rows themselves.
    """
    source_index, filename, spill_root, chunk_rows = job
    library_dir = os.path.join(spill_root, f"source_{int(source_index):06d}")
    os.makedirs(library_dir, exist_ok=True)

    runs: List[str] = []
    rows: List[Tuple] = []
    row_index = 0
    run_index = 0
    for row in _iter_single_lib_cluster_rows(filename):
        rows.append(tuple(row) + (int(source_index), int(row_index)))
        row_index += 1
        if len(rows) >= int(chunk_rows):
            run_path = os.path.join(library_dir, f"run_{run_index:06d}.tsv")
            _write_sorted_spill(rows, run_path)
            runs.append(run_path)
            rows = []
            run_index += 1

    if rows:
        run_path = os.path.join(library_dir, f"run_{run_index:06d}.tsv")
        _write_sorted_spill(rows, run_path)
        runs.append(run_path)

    return {
        "source_index": int(source_index),
        "row_count": int(row_index),
        "runs": runs,
    }


def _read_spill_row(reader) -> List[str] | None:
    try:
        return next(reader)
    except StopIteration:
        return None


def _spill_text_sort_key(row: Sequence[str]) -> Tuple[str, int, int, int]:
    return str(row[1]), int(row[4]), int(row[20]), int(row[21])


def _merge_sorted_spills(
    spill_paths: Sequence[str],
    destination: str,
    *,
    include_header: bool,
) -> None:
    """K-way merge already sorted spill files using one row per open input."""
    heap = []
    handles = []
    try:
        for source_number, path in enumerate(spill_paths):
            handle = open(path, "r", encoding="utf-8", newline="")
            reader = csv.reader(handle, delimiter="\t")
            handles.append((handle, reader))
            row = _read_spill_row(reader)
            if row is not None:
                heapq.heappush(heap, (_spill_text_sort_key(row), source_number, row))

        with open(destination, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            if include_header:
                writer.writerow(PROCESSED_CLUSTER_COLUMNS)
            while heap:
                _, source_number, row = heapq.heappop(heap)
                writer.writerow(row[: len(PROCESSED_CLUSTER_COLUMNS)] if include_header else row)
                next_row = _read_spill_row(handles[source_number][1])
                if next_row is not None:
                    heapq.heappush(
                        heap,
                        (_spill_text_sort_key(next_row), source_number, next_row),
                    )
    finally:
        for handle, _ in handles:
            handle.close()


def _merge_spills_with_bounded_fan_in(
    spill_paths: Sequence[str],
    *,
    spill_root: str,
    destination: str,
    fan_in: int,
) -> None:
    """Merge sorted runs in passes, avoiding an unbounded open-file count."""
    active_paths = list(spill_paths)
    pass_index = 0
    while len(active_paths) > int(fan_in):
        next_paths: List[str] = []
        for group_index, start in enumerate(range(0, len(active_paths), int(fan_in))):
            group = active_paths[start:start + int(fan_in)]
            merged_path = os.path.join(
                spill_root,
                f"merge_{pass_index:03d}_{group_index:06d}.tsv",
            )
            _merge_sorted_spills(group, merged_path, include_header=False)
            next_paths.append(merged_path)
        for path in active_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        active_paths = next_paths
        pass_index += 1

    _merge_sorted_spills(active_paths, destination, include_header=True)


def _spill_results_to_paths(results: Sequence[dict]) -> Tuple[List[str], int]:
    spill_paths: List[str] = []
    row_count = 0
    for result in sorted(results, key=lambda item: int(item["source_index"])):
        row_count += int(result["row_count"])
        spill_paths.extend(str(path) for path in result["runs"])
    return spill_paths, row_count


def _coerce_paths(clusterFiles: Sequence[str] | str) -> List[str]:
    if isinstance(clusterFiles, str):
        return [clusterFiles]
    return [str(x) for x in clusterFiles if str(x).strip()]


def _load_processed_clusters(path: str) -> pd.DataFrame:
    """Load the consolidated table without changing legacy value semantics."""
    # Candidate parsing yields these six fields as strings.  Do not let pandas
    # turn IDs such as "001" into integers or turn biological text such as "NA"
    # into missing values.  Empty numeric fields remain NaN, matching the legacy
    # DataFrame.to_csv()/read_csv behavior for floating NaNs.
    return pd.read_csv(
        path,
        sep="\t",
        dtype={column: str for column in PROCESSED_CLUSTER_TEXT_COLUMNS},
        keep_default_na=False,
        na_values={column: [""] for column in PROCESSED_CLUSTER_NUMERIC_COLUMNS},
    )


def _raise_on_parallel_errors(results: List[object]) -> None:
    errs = [x for x in results if isinstance(x, RuntimeError)]
    if errs:
        raise errs[0]


def aggregate_and_write_processed_clusters(
    clusterFiles: Sequence[str] | str,
    *,
    memFile: str | None = None,
) -> pd.DataFrame:
    """
    Aggregate candidate cluster files and write {phase}_processed_clusters.tab.

    Returns: allClusters dataframe (sorted by clusterID, pos)
    """
    print("### Aggregating and processing candidate cluster files per library ###")

    paths = _coerce_paths(clusterFiles)
    if not paths:
        raise ValueError("No cluster files provided to aggregate_and_write_processed_clusters().")

    missing = [p for p in paths if not artifact_exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing cluster file(s): {missing}")

    # Stable ordering for signatures and processing
    paths = sorted(paths, key=os.path.basename)

    outfname = phase2_basename("processed_clusters.tab")
    mem_path = memFile or getattr(rt, "memFile", None) or default_memfile_path()

    cache = MemCache.load(str(mem_path))
    input_sig = stage_signature(
        files=paths,
        params={
            "phase": getattr(rt, "phase", None),
            "concat_libs": bool(getattr(rt, "concat_libs", False)),
        },
    )

    if cache.hit("PROCESSED", outfname, input_sig):
        print(f"  - Output up-to-date (hash+sig match). Skipping aggregation: {outfname}")
        df_cached = _load_processed_clusters(resolve_artifact_path(outfname) or outfname)
        print(f"Processed clusters written to {outfname}")
        return df_cached

    chunk_rows = _cluster_aggregation_chunk_rows()
    fan_in = _cluster_aggregation_merge_fan_in()
    output_dir = os.path.dirname(os.path.abspath(outfname)) or os.getcwd()
    spill_root = tempfile.mkdtemp(prefix=".phasis_cluster_aggregation_", dir=output_dir)
    try:
        jobs = [
            (source_index, path, spill_root, chunk_rows)
            for source_index, path in enumerate(paths)
        ]
        print(
            "[INFO] Streaming candidate records into disk-backed sorted runs "
            f"({chunk_rows:,} rows per run; parallel across {len(jobs)} library file(s)).",
            flush=True,
        )
        spill_results = run_parallel_with_progress(
            spill_single_lib_cluster_to_sorted_runs,
            jobs,
            desc="Aggregating cluster files",
            min_chunk=1,
            unit="lib",
        )
        _raise_on_parallel_errors(spill_results)

        spill_paths, parsed_row_count = _spill_results_to_paths(spill_results)
        print(
            "[INFO] Merging "
            f"{len(spill_paths):,} sorted spill file(s) for {parsed_row_count:,} "
            "candidate-cluster records. Aggregation uses one bounded chunk per active "
            f"worker and a {fan_in}-file merge buffer; downstream then loads one table.",
            flush=True,
        )
        temporary_output = os.path.join(spill_root, "processed_clusters.tab")
        _merge_spills_with_bounded_fan_in(
            spill_paths,
            spill_root=spill_root,
            destination=temporary_output,
            fan_in=fan_in,
        )
        os.replace(temporary_output, outfname)
    finally:
        shutil.rmtree(spill_root, ignore_errors=True)

    print(
        "[INFO] Loading consolidated processed clusters once for downstream Phase II: "
        f"{outfname}",
        flush=True,
    )
    allClusters = _load_processed_clusters(outfname)

    fp = finalize_text_artifact(cache, "PROCESSED", outfname, input_sig)
    if fp:
        print(f"Hash for {outfname}: {fp}")

    print(f"Processed clusters written to {outfname}")
    return allClusters
