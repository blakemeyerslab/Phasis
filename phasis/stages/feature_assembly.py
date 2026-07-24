from __future__ import annotations


from .. import state as st
from .. import ids
import os
import re
import multiprocessing
import tempfile
import gc
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

DCL_OVERHANG = 2          # canonical 2-nt 3' overhang used for opposite-strand duplex pairing
WINDOW_MULTIPLIER = 10    # 10 cycles per window
FEATURE_SCHEMA_VERSION = 19
FEATURE_ASSEMBLY_INITIAL_WORKER_CAP = 2
FEATURE_ASSEMBLY_DEFAULT_MAX_CPU_FRACTION_NUMERATOR = 7
FEATURE_ASSEMBLY_DEFAULT_MAX_CPU_FRACTION_DENOMINATOR = 10
FEATURE_ASSEMBLY_DEFAULT_BATCH_ROWS = 100_000
HOWELL_AMBIGUITY_FRACTION = 0.90
HOWELL_CROWDING_SCORE_GAP = 4.0
ALTERNATIVE_MIN_SHARED_CYCLES = 3
PROMOTED_ALT_MIN_RELATIVE_SCORE = 0.66
PROMOTED_ALT_MIN_NON_GREY_ROWS = 2
PROMOTED_ALT_MIN_EXACT_ROWS = 1
MAIN_UNIT_BRIDGE_MIN_RATIO = 0.50
MAIN_UNIT_BRIDGE_MAX_ZERO_RUN = 2
CROSS_STRAND_BRIDGE_MAX_ZERO_RUN = 4
CROSS_STRAND_BRIDGE_RELAXED_MIN_RATIO = 0.35
MAIN_UNIT_MIN_SUPPORTED_POSITIONS = 2
MAIN_UNIT_MIN_EXACT_POSITIONS = 1
MAIN_PARTNER_TRACE_DEFAULT_NAME = "main_partner_trace.tsv"
MAIN_PARTNER_TRACE_ENV = "Phasis_MAIN_PARTNER_DEBUG"
MAIN_PARTNER_TRACE_OUT_ENV = "Phasis_MAIN_PARTNER_DEBUG_OUT"
MAIN_PARTNER_TRACE_COLS = [
    "identifier",
    "alib",
    "winner_strand",
    "winner_score",
    "winner_register_origin",
    "attempt_stage",
    "candidate_source",
    "candidate_category",
    "candidate_strand",
    "candidate_register_origin_tested",
    "observed_shift_nt",
    "normalized_shift_nt",
    "duplex_orientation_ok",
    "canonical_compatible",
    "candidate_tier",
    "shared_cycles",
    "bridge_support_ratio",
    "exact_support_count",
    "supported_position_count",
    "max_unsupported_run",
    "candidate_peak_score",
    "candidate_non_grey_row_count",
    "first_reject_reason",
    "accept_route",
    "final_route",
]


# ---- legacy schema (classifier-compatible) --------------------------------
FEATURE_COLS = ['identifier',
 'cID',
 'alib',
 'complexity',
 'strand_bias',
 'log_clust_len_norm_counts',
 'ratio_abund_len_phase',
 'phasis_score',
 'combined_fishers',
 'total_abund',
 'w_Howell_score',
 'w_window_start',
 'w_window_end',
 'c_Howell_score',
 'c_window_start',
 'c_window_end',
 'Peak_Howell_score',
 'Howell_exact_support_score',
 'Howell_ambiguity_count',
 'Howell_alt_register_count',
 'Howell_overlap_margin',
 'Howell_extension_window_count',
 'Howell_extension_span_nt',
 'Howell_origin_window_count',
 'Howell_origin_frame_count',
 'Howell_origin_margin',
 'Howell_origin_class',
 'Howell_additional_peak_count',
 'Howell_additional_peak_best_score',
 'Howell_overlapping_alt_count',
 'Howell_overlapping_alt_best_score',
 'Howell_overlapping_alt_best_shift_nt',
 'Howell_crowding_window_count',
 'Howell_crowding_best_score',
 'Howell_crowding_score_gap',
 'w_Howell_score_strict',
 'w_window_start_strict',
 'w_window_end_strict',
 'c_Howell_score_strict',
 'c_window_start_strict',
 'c_window_end_strict',
 'Peak_Howell_score_strict']

# Numeric columns (all except id-like/text fields)
NUMERIC_COLS = set(FEATURE_COLS) - {"identifier", "cID", "alib", "Howell_origin_class"}
FEATURE_TEXT_COLS = ("identifier", "cID", "alib", "Howell_origin_class")


def _truthy_env(value) -> bool:
    return str(value or "").strip().lower() not in {"", "0", "false", "no", "off", "none"}


def _main_partner_trace_enabled() -> bool:
    return _truthy_env(getenv(MAIN_PARTNER_TRACE_ENV)) or bool(
        str(getenv(MAIN_PARTNER_TRACE_OUT_ENV) or "").strip()
    )


def _main_partner_trace_outpath(*, phase: int | str | None, outdir: str | None, concat_libs: bool) -> str | None:
    explicit = str(getenv(MAIN_PARTNER_TRACE_OUT_ENV) or "").strip()
    if explicit:
        return explicit
    if not _main_partner_trace_enabled() or not outdir:
        return None
    prefix = "concat_" if bool(concat_libs) else ""
    phase_local = _phase_value() if phase is None else int(phase)
    return os.path.join(str(outdir), f"{prefix}{phase_local}_{MAIN_PARTNER_TRACE_DEFAULT_NAME}")


def _coerce_trace_row(row: dict) -> dict:
    payload = {col: np.nan for col in MAIN_PARTNER_TRACE_COLS}
    payload.update({key: value for key, value in dict(row or {}).items() if key in payload})
    return payload


def _emit_main_partner_trace(debug_rows, debug_context: dict | None, **fields) -> None:
    if debug_rows is None or debug_context is None:
        return
    payload = {
        "identifier": debug_context.get("identifier"),
        "alib": debug_context.get("alib"),
        "winner_strand": debug_context.get("winner_strand"),
        "winner_score": debug_context.get("winner_score"),
        "winner_register_origin": debug_context.get("winner_register_origin"),
    }
    payload.update(fields)
    debug_rows.append(_coerce_trace_row(payload))


def _phase_value(default: int = 21) -> int:
    """Return rt.phase as an int (module-level, spawn-safe)."""
    try:
        v = getattr(rt, "phase", None)
        if v is None:
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def _min_howell_score_value(default: float = 12.5) -> float:
    """Return rt.min_Howell_score as a float, falling back safely."""
    try:
        value = getattr(rt, "min_Howell_score", None)
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _get_memfile() -> str:
    """Return runtime memFile; default to the run directory if missing."""
    mem = getattr(rt, "memFile", None)
    if mem:
        return str(mem)
    mem = default_memfile_path()
    rt.memFile = mem
    return mem


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


def _feature_assembly_ncores() -> int:
    """Return the resolved Phasis core allocation, with a safe fallback."""
    ncores = _coerce_positive_int(getattr(rt, "ncores", None), None)
    if ncores is None:
        ncores = multiprocessing.cpu_count()
    return int(max(1, ncores))


def _feature_assembly_worker_cap() -> int:
    """Resolve the maximum feature-assembly concurrency for this allocation."""
    ncores = _feature_assembly_ncores()
    default_cap = max(
        1,
        (ncores * FEATURE_ASSEMBLY_DEFAULT_MAX_CPU_FRACTION_NUMERATOR)
        // FEATURE_ASSEMBLY_DEFAULT_MAX_CPU_FRACTION_DENOMINATOR,
    )
    configured = _coerce_positive_int(
        getattr(rt, "feature_assembly_worker_cap", None),
        _coerce_positive_int(
            getenv("Phasis_FEATURE_ASSEMBLY_WORKER_CAP"),
            default_cap,
        ),
    )
    return int(max(1, min(ncores, configured)))


def _feature_assembly_batch_rows() -> int:
    """Return the maximum input rows per indivisible feature-assembly task."""
    return int(
        _coerce_positive_int(
            getattr(rt, "feature_assembly_batch_rows", None),
            _coerce_positive_int(
                getenv("Phasis_FEATURE_ASSEMBLY_BATCH_ROWS"),
                FEATURE_ASSEMBLY_DEFAULT_BATCH_ROWS,
            ),
        )
    )


def _build_feature_assembly_batch_positions(
    clusters_data: pd.DataFrame,
    *,
    batch_rows: int,
) -> tuple[list[np.ndarray], int]:
    """Plan bounded whole-cluster batches without copying their DataFrames."""
    target_rows = max(1, int(batch_rows))
    batch_positions: list[np.ndarray] = []
    current_parts = []
    current_rows = 0
    oversized_clusters = 0

    # ``indices`` holds integer row positions, avoiding a DataFrame copy for
    # every cluster while preserving the prior chromosome -> cluster grouping.
    cluster_indices = clusters_data.groupby(
        ["chromosome", "clusterID"],
        sort=False,
        observed=True,
    ).indices
    for positions in cluster_indices.values():
        cluster_rows = len(positions)
        if current_parts and current_rows + cluster_rows > target_rows:
            batch_positions.append(np.concatenate(current_parts))
            current_parts = []
            current_rows = 0

        current_parts.append(positions)
        current_rows += cluster_rows
        if cluster_rows > target_rows:
            oversized_clusters += 1

        # Flush a full batch immediately. An oversized cluster is therefore
        # isolated rather than being combined with another cluster.
        if current_rows >= target_rows:
            batch_positions.append(np.concatenate(current_parts))
            current_parts = []
            current_rows = 0

    if current_parts:
        batch_positions.append(np.concatenate(current_parts))

    return batch_positions, oversized_clusters


class _LazyFeatureAssemblyBatches:
    """Sequence-like feature batches that materializes only a runner slice.

    ``run_parallel_with_progress`` accesses its input with ``len()`` and slices
    it once per bounded scheduling window. Keeping only positional plans here
    prevents the parent process from holding one copied DataFrame per batch.
    The worker still receives the same independent, nine-column DataFrame that
    the legacy eager batch builder produced.
    """

    def __init__(
        self,
        clusters_data: pd.DataFrame,
        *,
        required_cols: list[str],
        batch_positions: list[np.ndarray],
    ) -> None:
        self._clusters_data = clusters_data
        self._column_positions = [
            int(clusters_data.columns.get_loc(column))
            for column in required_cols
        ]
        self._batch_positions = batch_positions

    def __len__(self) -> int:
        return len(self._batch_positions)

    def _materialize(self, index: int) -> pd.DataFrame:
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("feature batch index out of range")
        positions = self._batch_positions[index]
        # One advanced positional selection gives a standalone, bounded task
        # frame in the desired nine-column order.  Avoid selecting all source
        # columns first, which would add a second batch-sized temporary copy.
        return self._clusters_data.iloc[positions, self._column_positions]

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self._materialize(item) for item in range(start, stop, step)]
        return self._materialize(index)


def _build_feature_assembly_batches(
    clusters_data: pd.DataFrame,
    *,
    batch_rows: int,
) -> tuple[list[pd.DataFrame], int]:
    """Pack complete clusters into bounded-row work batches.

    A feature calculation is intrinsically per cluster, so a cluster must never
    be split between tasks. The former chromosome-sized work units could make a
    single enriched chromosome consume most of a worker's RAM. This keeps that
    cluster boundary while allowing batches to span chromosomes and stay near a
    predictable input size. The returned oversized count identifies clusters
    which cannot be made smaller without changing feature semantics.

    This eager compatibility helper remains available for callers and tests;
    production feature assembly uses :class:`_LazyFeatureAssemblyBatches`.
    """
    batch_positions, oversized_clusters = _build_feature_assembly_batch_positions(
        clusters_data,
        batch_rows=batch_rows,
    )
    batches = [clusters_data.iloc[positions].copy() for positions in batch_positions]
    return batches, oversized_clusters


def _feature_assembly_parallel_kwargs(group_count: int) -> dict:
    """Bound both workers and in-flight feature-batch DataFrames for this stage."""
    max_worker_cap = _feature_assembly_worker_cap()
    initial_worker_cap = min(FEATURE_ASSEMBLY_INITIAL_WORKER_CAP, max_worker_cap)
    initial_task_window = max(1, min(int(group_count), initial_worker_cap))
    max_task_window = max(1, min(int(group_count), max_worker_cap))
    return {
        "initial_worker_cap": initial_worker_cap,
        "max_worker_cap": max_worker_cap,
        "initial_chunk_size": initial_task_window,
        "max_chunk_size": max_task_window,
        # A caught worker failure drops to one worker; successful later batches
        # can recover only to this hard cap, never to all requested CPU cores.
        "adaptive_recovery": True,
    }


def _coerce_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Restore the legacy numeric schema after reading a feature TSV."""
    for col in df.columns:
        if col in NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure exact column order if all expected fields are available. This is
    # deliberately the same tolerant behavior used by the historical cache
    # reader, which can still surface a useful warning for a manually supplied
    # legacy file.
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if not missing:
        return df[FEATURE_COLS]
    print(f"[WARN] Existing file lacks expected columns: {missing}")
    return df


def _read_feature_frame(path: str) -> pd.DataFrame:
    physical = resolve_artifact_path(path) or path
    return _coerce_feature_frame(
        pd.read_csv(
            physical,
            sep="\t",
            # The default parser may round a decimal by one ULP.  This table is
            # re-serialized after disk-backed streaming, so retain the exact
            # binary float represented in the TSV to preserve legacy output
            # bytes and avoid needless downstream classification jitter.
            float_precision="round_trip",
            # A freshly assembled table keeps these fields as strings. Pinning
            # their dtype here avoids turning an all-numeric-looking cluster ID
            # into an integer only because the streamed file was re-read.
            dtype={column: object for column in FEATURE_TEXT_COLS},
        )
    )


def _feature_assembly_temp_path(outpath: str, label: str, *, suffix: str = ".tsv") -> str:
    """Create a sibling temporary path so final replacement stays atomic."""
    absolute = os.path.abspath(str(outpath))
    directory = os.path.dirname(absolute)
    basename = os.path.basename(absolute)
    os.makedirs(directory, exist_ok=True)
    fd, path = tempfile.mkstemp(
        prefix=f".phasis_feature_assembly_{basename}.{label}.",
        suffix=suffix,
        dir=directory,
    )
    os.close(fd)
    return path


class _FeatureResultStreamWriter:
    """Write one ordered worker result without retaining previous results."""

    def __init__(self, feature_handle, *, trace_handle=None) -> None:
        self.feature_handle = feature_handle
        self.trace_handle = trace_handle
        self.streamed_rows = 0
        self.bad_chunks = 0

    def __call__(self, sub) -> None:
        chunk_rows = None
        chunk_trace_rows = None
        if isinstance(sub, dict):
            chunk_rows = sub.get("rows")
            chunk_trace_rows = sub.get("debug_rows")
        elif isinstance(sub, list):
            chunk_rows = sub

        if not isinstance(chunk_rows, list):
            self.bad_chunks += 1
            return

        valid_rows = []
        for row in chunk_rows:
            if isinstance(row, (list, tuple)) and len(row) == len(FEATURE_COLS):
                valid_rows.append(list(row))
            else:
                self.bad_chunks += 1
        if valid_rows:
            # This per-result DataFrame is bounded by one task; numeric
            # coercion mirrors the historical final DataFrame before it is
            # serialized to the stream.
            _coerce_feature_frame(
                pd.DataFrame(valid_rows, columns=FEATURE_COLS)
            ).to_csv(
                self.feature_handle,
                sep="\t",
                index=False,
                header=(self.streamed_rows == 0),
            )
            self.streamed_rows += len(valid_rows)

        if self.trace_handle is not None and isinstance(chunk_trace_rows, list):
            valid_trace_rows = [
                _coerce_trace_row(row)
                for row in chunk_trace_rows
                if isinstance(row, dict)
            ]
            if valid_trace_rows:
                pd.DataFrame(
                    valid_trace_rows,
                    columns=MAIN_PARTNER_TRACE_COLS,
                ).to_csv(
                    self.trace_handle,
                    sep="\t",
                    index=False,
                    header=False,
                )


def ensure_win_score_lookup_ready() -> None:
    """
    Spawn-safe: ensure st.WIN_SCORE_LOOKUP is populated in *this* process.
    If empty and rt.clusters_scored_tsv exists, load it.
    """
    try:
        if st.WIN_SCORE_LOOKUP:
            return
        p = getattr(rt, "clusters_scored_tsv", None)
        physical = resolve_artifact_path(p) if p else None
        if physical:
            st.load_win_score_lookup_from_tsv(physical)
    except Exception:
        # keep feature assembly robust; caller will fall back to defaults
        return

def features_to_detection(clusters_data: pd.DataFrame,*,phase: str | int | None = None,outdir: str | None = None,concat_libs: bool | None = None,memFile: str | None = None,outfname: str | None = None,) -> pd.DataFrame:
    """
    Assemble per-cluster feature set (parallel), write TSV, and memoize via md5.
    Uses legacy column names compatible with downstream classification.
    Expects process_chromosome_features() to return rows in FEATURE_COLS order.
    """
    print("### Step: assemble per-cluster features ###")

    # Resolve defaults from runtime unless explicitly provided
    if phase is None:
        phase = getattr(rt, "phase", None)
    if outdir is None:
        outdir = getattr(rt, "outdir", None)
    if concat_libs is None:
        try:
            concat_libs = bool(getattr(rt, "concat_libs", False))
        except Exception:
            concat_libs = False

    if memFile is None:
        memFile = _get_memfile()
    else:
        # Keep runtime consistent so other cache helpers still behave
        rt.memFile = memFile
        if outdir is not None:
            rt.outdir = outdir

    if outfname is None:
        prefix = "concat_" if concat_libs else ""
        outfname = f"{prefix}{phase}_cluster_set_features.tsv"

    # Input signature: only reuse cached features when upstream inputs match.
    phas_path = phase2_basename("PHAS_to_detect.tab")
    scored_path = phase2_basename("clusters_scored.tsv")

    input_sig = stage_signature(
        files=[phas_path, scored_path, __file__],
        params={
            "phase": phase,
            "concat_libs": bool(concat_libs),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        },
    )

    cache = MemCache.load(memFile)
    section = "CLUSTER_FEATURES"

    # ---------- Early cache check ----------
    if cache.hit(section, outfname, input_sig):
        print(f"  - Output up-to-date (hash+sig match). Skipping assembly: {outfname}")
        return _read_feature_frame(outfname)

    # ---------- Validate input ----------
    required_cols = ['clusterID', 'chromosome', 'strand', 'pos', 'len', 'abun', 'identifier', 'tag_seq', 'alib']
    missing_in_input = [c for c in required_cols if c not in clusters_data.columns]
    if missing_in_input:
        raise ValueError(f"clusters_data missing required columns: {missing_in_input}")

    # Split into bounded batches of whole clusters. A chromosome can therefore
    # use several tasks instead of forcing one potentially huge DataFrame into a
    # worker, while small chromosomes are efficiently packed together. Keep only
    # positional batch plans in the parent; each copied input DataFrame is made
    # lazily for the scheduler's current bounded window.
    batch_rows = _feature_assembly_batch_rows()
    batch_positions, oversized_clusters = _build_feature_assembly_batch_positions(
        clusters_data,
        batch_rows=batch_rows,
    )
    feature_batches = _LazyFeatureAssemblyBatches(
        clusters_data,
        required_cols=required_cols,
        batch_positions=batch_positions,
    )
    chromosome_count = clusters_data.groupby(
        "chromosome",
        sort=False,
        observed=True,
    ).ngroups
    print(
        f"  - Found {chromosome_count} chromosome groups; built "
        f"{len(feature_batches)} feature batches of up to {batch_rows:,} input rows"
    )
    if oversized_clusters:
        print(
            "[WARN] "
            f"{oversized_clusters} cluster(s) exceed the feature batch size and "
            "must be processed intact."
        )

    feature_parallel_kwargs = _feature_assembly_parallel_kwargs(len(feature_batches))
    print(
        "  - Feature assembly starts with "
        f"{feature_parallel_kwargs['initial_worker_cap']} concurrent batch(es) and "
        f"can grow to {feature_parallel_kwargs['max_worker_cap']}; set "
        "PHASIS_FEATURE_ASSEMBLY_WORKER_CAP or PHASIS_FEATURE_ASSEMBLY_BATCH_ROWS "
        "to override."
    )

    trace_enabled = _main_partner_trace_enabled()
    trace_outpath = _main_partner_trace_outpath(
        phase=phase,
        outdir=outdir,
        concat_libs=bool(concat_libs),
    )

    # ---------- Parallel processing + bounded result streaming ----------
    # Results are committed by the shared runner in input order. Stream each
    # accepted work result to a TSV so neither a global ``results`` list nor a
    # global flattened feature-row list is retained in RAM.
    streamed_path = _feature_assembly_temp_path(outfname, "stream")
    normalized_suffix = ".tsv.gz" if str(outfname).endswith(".gz") else ".tsv"
    normalized_path = _feature_assembly_temp_path(
        outfname,
        "normalized",
        suffix=normalized_suffix,
    )
    trace_streamed_path = None
    trace_final_path = None
    if trace_enabled and trace_outpath:
        trace_streamed_path = _feature_assembly_temp_path(trace_outpath, "stream")

    stream_writer = None
    try:
        with open(streamed_path, "w", encoding="utf-8", newline="") as feature_handle:
            trace_handle = None
            try:
                if trace_streamed_path:
                    trace_handle = open(trace_streamed_path, "w", encoding="utf-8", newline="")
                    # Match the former empty-trace behavior: debug mode always
                    # emits a header even if no feature batch produced a trace.
                    pd.DataFrame(columns=MAIN_PARTNER_TRACE_COLS).to_csv(
                        trace_handle,
                        sep="\t",
                        index=False,
                    )

                stream_writer = _FeatureResultStreamWriter(
                    feature_handle,
                    trace_handle=trace_handle,
                )

                run_parallel_with_progress(
                    process_chromosome_features,
                    feature_batches,
                    desc="Assemble features",
                    min_chunk=1,
                    unit="lib-chr",
                    on_result=stream_writer,
                    return_results=False,
                    **feature_parallel_kwargs,
                )
            finally:
                if trace_handle is not None:
                    trace_handle.close()

        # The scheduler no longer needs its positional plans or its local input
        # reference. Release them before loading the one returned feature table.
        del feature_batches
        del batch_positions
        del clusters_data
        gc.collect()

        if stream_writer is not None and stream_writer.bad_chunks:
            print(
                "[WARN] Skipped "
                f"{stream_writer.bad_chunks} malformed/failed rows or chunks during feature assembly."
            )

        if stream_writer is None or not stream_writer.streamed_rows:
            raise RuntimeError("No features assembled; all chunks failed or returned empty results.")

        # Read the finished stream once to preserve the old return-value API and
        # its complete-table numeric dtype inference. Rewriting from this one
        # final DataFrame also keeps legacy TSV formatting stable across worker
        # batch boundaries.
        collected_features = _read_feature_frame(streamed_path)
        collected_features.to_csv(
            normalized_path,
            sep="\t",
            index=False,
            compression=("gzip" if str(outfname).endswith(".gz") else None),
        )
        os.replace(normalized_path, outfname)
        normalized_path = None

        if trace_streamed_path and trace_outpath:
            if str(trace_outpath).endswith(".gz"):
                trace_final_path = _feature_assembly_temp_path(
                    trace_outpath,
                    "normalized",
                    suffix=".tsv.gz",
                )
                trace_frame = pd.read_csv(trace_streamed_path, sep="\t")
                trace_frame.to_csv(trace_final_path, sep="\t", index=False, compression="gzip")
                os.replace(trace_final_path, trace_outpath)
                trace_final_path = None
            else:
                os.replace(trace_streamed_path, trace_outpath)
            trace_streamed_path = None
            print(f"  - Wrote {trace_outpath}")

        fp = finalize_text_artifact(cache, section, outfname, input_sig)
        if fp:
            print(f"  - Wrote {outfname} (md5: {fp})")
        return collected_features
    finally:
        for temporary_path in (
            streamed_path,
            normalized_path,
            trace_streamed_path,
            trace_final_path,
        ):
            if temporary_path and os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass


def _strand_masks(df: pd.DataFrame):
    """Return boolean masks for W and C strands accepting several encodings."""
    s = df['strand'].astype(str).str.lower()
    w_mask = s.isin(['w', '+', 'watson', '1', 'true'])
    c_mask = s.isin(['c', '-', 'crick', '0', 'false'])
    return w_mask, c_mask

def _build_pos_abun_exact_phase(df: pd.DataFrame, seq_start: int, seq_end: int, phase: int):
    """
    Build {position -> abundance} using ONLY reads with length == phase
    and positions within [seq_start, seq_end].
    """
    ph = int(phase)
    d: dict[int, float] = {}
    # small speed-up: filter by length first
    df_ph = df.loc[pd.to_numeric(df['len'], errors='coerce') == ph]
    if df_ph.empty:
        return d
    for _, row in df_ph.iterrows():
        pos = int(row['pos'])
        if seq_start <= pos <= seq_end:
            d[pos] = d.get(pos, 0.0) + float(row['abun'])
    return d


def _compute_howell_score_from_register(
    in_phase_sum: float,
    effective_total: float,
    n_filled: int,
    num_cycles: int,
):
    out_of_phase = max(0.0, float(effective_total) - float(in_phase_sum))
    numerator = float(in_phase_sum)
    denominator = 1.0 + out_of_phase
    if numerator <= 0.0 or not (int(n_filled) > 3):
        return 0.0, out_of_phase

    log_arg = 1.0 + 10.0 * (numerator / denominator)
    if log_arg <= 0.0 or log_arg != log_arg:
        return 0.0, out_of_phase

    scale = max(min(int(n_filled), int(num_cycles)) - 2, 0)
    return float(scale * (0.0 if log_arg <= 0 else np.log(log_arg))), float(out_of_phase)


def _score_window_registers(window_positions, pos_abun, win_start, win_end, phase, *, forward=True, exact_only: bool = False):
    ph = int(phase)
    if not window_positions:
        return {
            "score": 0.0,
            "best_register": None,
            "best_in_phase_sum": 0.0,
            "best_out_of_phase": 0.0,
            "best_filled": 0,
            "register_scores": [0.0] * ph,
        }

    num_cycles = max(0, (win_end - win_start + 1) // ph)
    if num_cycles < 4:
        return {
            "score": 0.0,
            "best_register": None,
            "best_in_phase_sum": 0.0,
            "best_out_of_phase": 0.0,
            "best_filled": 0,
            "register_scores": [0.0] * ph,
        }

    best_score = -float("inf")
    best_reg_sum = 0.0
    best_reg_out = 0.0
    best_reg_filled = 0
    best_reg = None
    register_scores: list[float] = []
    evaluator = _evaluate_register_strict_exact if exact_only else _evaluate_register

    for reg in range(ph):
        in_sum, eff_total, n_filled = evaluator(
            window_positions, pos_abun, win_start, win_end, ph, reg, forward=forward
        )
        reg_score, out_of_phase = _compute_howell_score_from_register(
            in_sum,
            eff_total,
            n_filled,
            num_cycles,
        )
        register_scores.append(float(reg_score))
        if reg_score > best_score:
            best_score = reg_score
            best_reg_sum = float(in_sum)
            best_reg_out = float(out_of_phase)
            best_reg_filled = int(n_filled)
            best_reg = reg

    return {
        "score": float(0.0 if best_score == -float("inf") else best_score),
        "best_register": best_reg,
        "best_in_phase_sum": float(best_reg_sum),
        "best_out_of_phase": float(best_reg_out),
        "best_filled": int(best_reg_filled),
        "register_scores": register_scores,
    }


def _windows_overlap(start_a, end_a, start_b, end_b) -> bool:
    return int(max(start_a, start_b)) <= int(min(end_a, end_b))


def _window_scan_bounds(positions, win_size, seq_start=None, seq_end=None):
    lower_bound = seq_start if seq_start is not None else positions[0]
    upper_bound = (seq_end - win_size + 1) if seq_end is not None else positions[-1] - win_size + 1
    if upper_bound < lower_bound:
        lower_bound = positions[0]
        upper_bound = lower_bound
    return int(lower_bound), int(upper_bound)


def _canonical_exact_frame(
    *,
    window_start: int | None,
    window_end: int | None,
    exact_best_register,
    strand_code: str | None,
    phase: int,
) -> int | None:
    try:
        reg_value = int(exact_best_register)
    except Exception:
        return None

    try:
        phase_local = int(phase)
    except Exception:
        return None
    if phase_local <= 0:
        return None

    strand_text = str(strand_code or "").strip().lower()
    try:
        if strand_text == "c":
            return int((int(window_end) - reg_value) % phase_local)
        return int((int(window_start) + reg_value) % phase_local)
    except Exception:
        return None


def _exact_register_origin(
    *,
    window_start: int | None,
    window_end: int | None,
    exact_best_register,
    strand_code: str | None,
):
    try:
        reg_value = int(exact_best_register)
    except Exception:
        return None

    try:
        strand_text = str(strand_code or "").strip().lower()
        if strand_text == "c":
            return int(window_end) - reg_value
        return int(window_start) + reg_value
    except Exception:
        return None


def _enumerate_relaxed_candidate_windows(
    pos_abun,
    phase,
    win_size,
    seq_start=None,
    seq_end=None,
    *,
    forward=True,
):
    positions = sorted(pos_abun.keys())
    if not positions:
        return [], None

    lower_bound, upper_bound = _window_scan_bounds(
        positions,
        win_size,
        seq_start=seq_start,
        seq_end=seq_end,
    )

    best_score = -float("inf")
    best_detail = None
    candidate_windows = []
    phase_local = int(phase)

    for win_start in range(lower_bound, upper_bound + 1):
        win_end = win_start + win_size - 1
        window_positions = [p for p in positions if win_start <= p <= win_end]
        relaxed_summary = _score_window_registers(
            window_positions,
            pos_abun,
            win_start,
            win_end,
            phase_local,
            forward=forward,
        )
        exact_summary = _score_window_registers(
            window_positions,
            pos_abun,
            win_start,
            win_end,
            phase_local,
            forward=forward,
            exact_only=True,
        )
        exact_best_register = exact_summary.get("best_register")
        strand_code = "w" if forward else "c"
        candidate = {
            "strand": strand_code,
            "anchor_position": int(win_start if forward else win_end),
            "window_start": int(win_start),
            "window_end": int(win_end),
            "score": float(relaxed_summary["score"]),
            "best_register": relaxed_summary.get("best_register"),
            "exact_score": float(exact_summary["score"]),
            "exact_best_register": exact_best_register,
            "exact_frame": _canonical_exact_frame(
                window_start=win_start,
                window_end=win_end,
                exact_best_register=exact_best_register,
                strand_code=strand_code,
                phase=phase_local,
            ),
            "exact_register_position": _exact_register_origin(
                window_start=win_start,
                window_end=win_end,
                exact_best_register=exact_best_register,
                strand_code=strand_code,
            ),
        }
        candidate_windows.append(candidate)

        if candidate["score"] > best_score:
            best_score = candidate["score"]
            best_detail = {
                "strand": strand_code,
                "anchor_position": candidate["anchor_position"],
                "window_start": candidate["window_start"],
                "window_end": candidate["window_end"],
                "score": candidate["score"],
                "best_register": candidate["best_register"],
                "register_scores": list(relaxed_summary.get("register_scores") or []),
                "exact_score": candidate["exact_score"],
                "exact_best_register": candidate["exact_best_register"],
                "exact_register_scores": list(exact_summary.get("register_scores") or []),
                "exact_frame": candidate["exact_frame"],
                "exact_register_position": candidate["exact_register_position"],
            }

    return candidate_windows, best_detail


def _summarize_peak_howell_ambiguity(
    best_window_detail: dict | None,
    candidate_windows,
    *,
    threshold_fraction: float = HOWELL_AMBIGUITY_FRACTION,
):
    if not best_window_detail:
        return None

    exact_support_score = float(best_window_detail.get("exact_score", 0.0) or 0.0)
    winner_register = best_window_detail.get("exact_best_register")
    register_scores = best_window_detail.get("exact_register_scores") or []
    result = {
        "winner_strand": best_window_detail.get("strand"),
        "winner_window_start": best_window_detail.get("window_start"),
        "winner_window_end": best_window_detail.get("window_end"),
        "winner_register": best_window_detail.get("best_register"),
        "exact_winner_register": winner_register,
        "winner_exact_frame": best_window_detail.get("exact_frame"),
        "Howell_exact_support_score": float(exact_support_score),
        "best_overlapping_competitor_score": np.nan,
        "Howell_ambiguity_count": np.nan,
        "Howell_alt_register_count": np.nan,
        "Howell_overlap_margin": np.nan,
        "Howell_extension_window_count": np.nan,
        "Howell_extension_span_nt": np.nan,
        "Howell_origin_window_count": np.nan,
        "Howell_origin_frame_count": np.nan,
        "Howell_origin_margin": np.nan,
        "Howell_origin_class": "insufficient_exact_support",
    }

    winner_exact_frame = result.get("winner_exact_frame")
    if exact_support_score <= 0.0 or winner_exact_frame is None:
        return result

    threshold_score = float(threshold_fraction) * exact_support_score
    overlap_best_score = None
    overlap_count = 0
    same_frame_count = 0
    competing_frame_count = 0
    competing_frames = set()
    best_competing_frame_score = None
    extension_min_start = int(best_window_detail.get("window_start"))
    extension_max_end = int(best_window_detail.get("window_end"))
    for candidate in candidate_windows or []:
        if str(candidate.get("strand")) != str(best_window_detail.get("strand")):
            continue
        if (
            int(candidate.get("window_start")) == int(best_window_detail.get("window_start"))
            and int(candidate.get("window_end")) == int(best_window_detail.get("window_end"))
        ):
            continue
        if not _windows_overlap(
            candidate.get("window_start"),
            candidate.get("window_end"),
            best_window_detail.get("window_start"),
            best_window_detail.get("window_end"),
        ):
            continue
        candidate_score = float(candidate.get("exact_score", 0.0) or 0.0)
        if candidate_score < threshold_score:
            continue
        overlap_count += 1
        if overlap_best_score is None or candidate_score > overlap_best_score:
            overlap_best_score = candidate_score
        candidate_exact_frame = candidate.get("exact_frame")
        if candidate_exact_frame == winner_exact_frame:
            same_frame_count += 1
            extension_min_start = min(extension_min_start, int(candidate.get("window_start")))
            extension_max_end = max(extension_max_end, int(candidate.get("window_end")))
        else:
            competing_frame_count += 1
            if candidate_exact_frame is not None:
                competing_frames.add(int(candidate_exact_frame))
            if best_competing_frame_score is None or candidate_score > best_competing_frame_score:
                best_competing_frame_score = candidate_score

    alt_register_count = 0
    if winner_register is not None:
        for reg_idx, reg_score in enumerate(register_scores):
            if int(reg_idx) == int(winner_register):
                continue
            if float(reg_score) >= threshold_score:
                alt_register_count += 1

    result["best_overlapping_competitor_score"] = (
        np.nan if overlap_best_score is None else float(overlap_best_score)
    )
    result["Howell_ambiguity_count"] = int(overlap_count)
    result["Howell_alt_register_count"] = int(alt_register_count)
    result["Howell_overlap_margin"] = (
        np.nan if overlap_best_score is None else float(exact_support_score - float(overlap_best_score))
    )
    result["Howell_extension_window_count"] = int(same_frame_count)
    result["Howell_extension_span_nt"] = int(extension_max_end - extension_min_start + 1)
    result["Howell_origin_window_count"] = int(competing_frame_count)
    result["Howell_origin_frame_count"] = int(len(competing_frames))
    result["Howell_origin_margin"] = (
        np.nan
        if best_competing_frame_score is None
        else float(exact_support_score - float(best_competing_frame_score))
    )
    if overlap_count == 0:
        result["Howell_origin_class"] = "unique_origin"
    elif same_frame_count > 0 and competing_frame_count == 0:
        result["Howell_origin_class"] = "coherent_extension"
    elif same_frame_count == 0 and competing_frame_count > 0:
        result["Howell_origin_class"] = "ambiguous_origin"
    else:
        result["Howell_origin_class"] = "mixed_extension_and_ambiguity"
    return result

# ---------- RELAXED Howell (positional ±1 wobble allowed) ----------
def _best_sliding_window_score_generic(
    pos_abun,
    phase,
    win_size,
    seq_start=None,
    seq_end=None,
    forward=True,
    *,
    return_detail: bool = False,
):
    """
    Generic window scan (relaxed Howell with ±1 wobble).
    Expects pos_abun already filtered to len == phase.
    """
    positions = sorted(pos_abun.keys())
    if not positions:
        if return_detail:
            return 0.0, None, None, None
        return 0.0, None, None

    lower_bound, upper_bound = _window_scan_bounds(
        positions,
        win_size,
        seq_start=seq_start,
        seq_end=seq_end,
    )

    if return_detail:
        candidate_windows, best_detail = _enumerate_relaxed_candidate_windows(
            pos_abun,
            int(phase),
            win_size,
            seq_start=lower_bound,
            seq_end=upper_bound + win_size - 1,
            forward=forward,
        )
        if best_detail is None:
            return 0.0, None, None, None
        detail = _summarize_peak_howell_ambiguity(best_detail, candidate_windows)
        return (
            float(best_detail["score"]),
            int(best_detail["window_start"]),
            int(best_detail["window_end"]),
            detail,
        )

    best_score = -float("inf")
    best_window = (None, None)

    for win_start in range(lower_bound, upper_bound + 1):
        win_end = win_start + win_size - 1
        window_positions = [p for p in positions if win_start <= p <= win_end]
        relaxed_summary = _score_window_registers(
            window_positions,
            pos_abun,
            win_start,
            win_end,
            int(phase),
            forward=forward,
        )
        score = float(relaxed_summary["score"])
        if score > best_score:
            best_score = score
            best_window = (win_start, win_end)

    resolved_score = best_score if best_score != -float("inf") else 0.0
    return resolved_score, best_window[0], best_window[1]


def collect_exact_only_peak_competitors(
    aclust: pd.DataFrame,
    *,
    phase: int | None = None,
    threshold_fraction: float = HOWELL_AMBIGUITY_FRACTION,
) -> dict:
    ph = int(phase) if phase is not None else _phase_value()
    win_size = WINDOW_MULTIPLIER * int(ph)
    seq_start = int(aclust["pos"].min())
    seq_end = int(aclust["pos"].max())
    w_mask, c_mask = _strand_masks(aclust)

    all_candidates = []
    winner_detail = None
    winner_score = -float("inf")

    for strand_code, mask, forward in (("w", w_mask, True), ("c", c_mask, False)):
        if not mask.any():
            continue
        pos_abun = _build_pos_abun_exact_phase(aclust.loc[mask], seq_start, seq_end, int(ph))
        if not pos_abun:
            continue
        candidates, best_detail = _enumerate_relaxed_candidate_windows(
            pos_abun,
            int(ph),
            win_size,
            seq_start=seq_start,
            seq_end=seq_end,
            forward=forward,
        )
        all_candidates.extend(candidates)
        if best_detail is None:
            continue
        score = float(best_detail.get("score", 0.0) or 0.0)
        if score >= winner_score:
            winner_score = score
            winner_detail = best_detail

    summary = _summarize_peak_howell_ambiguity(
        winner_detail,
        all_candidates,
        threshold_fraction=threshold_fraction,
    )
    if not winner_detail or not summary:
        return {
            "winner_detail": winner_detail,
            "summary": summary,
            "competing_windows": [],
        }

    exact_support_score = float(summary.get("Howell_exact_support_score", 0.0) or 0.0)
    winner_exact_frame = summary.get("winner_exact_frame")
    threshold_score = float(threshold_fraction) * exact_support_score
    competing_windows = []
    for candidate in all_candidates:
        if str(candidate.get("strand")) != str(winner_detail.get("strand")):
            continue
        if (
            int(candidate.get("window_start")) == int(winner_detail.get("window_start"))
            and int(candidate.get("window_end")) == int(winner_detail.get("window_end"))
        ):
            continue
        if not _windows_overlap(
            candidate.get("window_start"),
            candidate.get("window_end"),
            winner_detail.get("window_start"),
            winner_detail.get("window_end"),
        ):
            continue
        candidate_score = float(candidate.get("exact_score", 0.0) or 0.0)
        if candidate_score < threshold_score:
            continue
        if candidate.get("exact_frame") == winner_exact_frame:
            continue
        competing_windows.append(dict(candidate))

    return {
        "winner_detail": winner_detail,
        "summary": summary,
        "competing_windows": competing_windows,
    }

def best_sliding_window_score_forward(pos_abun, phase, win_size, seq_start=None, seq_end=None):
    return _best_sliding_window_score_generic(
        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=True
    )

def best_sliding_window_score_reverse(pos_abun, phase, win_size, seq_start=None, seq_end=None):
    return _best_sliding_window_score_generic(
        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=False
    )

def _pick_peak_howell_detail(w_detail: dict | None, c_detail: dict | None):
    if w_detail is None:
        return c_detail
    if c_detail is None:
        return w_detail

    w_score = float(w_detail.get("score", 0.0) or 0.0)
    c_score = float(c_detail.get("score", 0.0) or 0.0)
    if c_score > w_score:
        return c_detail
    return w_detail


def compute_phasing_score_Howell(aclust: pd.DataFrame, *, return_detail: bool = False):
    """
    Howell-like phasing WITH positional wobble (±1), but ONLY len == phase reads.
    Returns: (w_score,(w_start,w_end), c_score,(c_start,c_end))
    When return_detail=True, appends the peak-window ambiguity summary.
    """
    ph = _phase_value()
    win_size  = WINDOW_MULTIPLIER * int(ph)
    seq_start = int(aclust['pos'].min()); seq_end = int(aclust['pos'].max())
    w_mask, c_mask = _strand_masks(aclust)

    # Forward “w”
    if w_mask.any():
        w_pos_abun = _build_pos_abun_exact_phase(aclust.loc[w_mask], seq_start, seq_end, int(ph))
        if w_pos_abun:
            if return_detail:
                w_score, w_s, w_e, w_detail = _best_sliding_window_score_generic(
                    w_pos_abun,
                    int(ph),
                    win_size,
                    seq_start=seq_start,
                    seq_end=seq_end,
                    forward=True,
                    return_detail=True,
                )
            else:
                w_score, w_s, w_e = best_sliding_window_score_forward(
                    w_pos_abun, int(ph), win_size, seq_start, seq_end
                )
                w_detail = None
        else:
            w_score, w_s, w_e, w_detail = 0.0, None, None, None
    else:
        w_score, w_s, w_e, w_detail = None, None, None, None

    # Reverse “c”
    if c_mask.any():
        c_pos_abun = _build_pos_abun_exact_phase(aclust.loc[c_mask], seq_start, seq_end, int(ph))
        if c_pos_abun:
            if return_detail:
                c_score, c_s, c_e, c_detail = _best_sliding_window_score_generic(
                    c_pos_abun,
                    int(ph),
                    win_size,
                    seq_start=seq_start,
                    seq_end=seq_end,
                    forward=False,
                    return_detail=True,
                )
            else:
                c_score, c_s, c_e = best_sliding_window_score_reverse(
                    c_pos_abun, int(ph), win_size, seq_start, seq_end
                )
                c_detail = None
        else:
            c_score, c_s, c_e, c_detail = 0.0, None, None, None
    else:
        c_score, c_s, c_e, c_detail = None, None, None, None

    if not return_detail:
        return (w_score, (w_s, w_e), c_score, (c_s, c_e))

    peak_detail = _pick_peak_howell_detail(w_detail, c_detail)
    return (w_score, (w_s, w_e), c_score, (c_s, c_e), peak_detail)

def _evaluate_register_strict_exact(window_positions, pos_abun, win_start, win_end, phase, reg, forward=True):
    """Count ONLY exact register hits (no ±1). Returns: (in_phase_sum, total_in_window, n_filled_cycles)"""
    positions_set = set(window_positions)
    num_cycles = max(0, (win_end - win_start + 1) // int(phase))

    in_phase_sum = 0.0
    n_filled = 0
    for c in range(num_cycles):
        expected_pos = (win_start + reg + c * int(phase)) if forward else (win_end - reg - c * int(phase))
        if expected_pos in positions_set:
            in_phase_sum += pos_abun[expected_pos]
            n_filled += 1

    total_in_window = sum(pos_abun[p] for p in window_positions)
    return in_phase_sum, total_in_window, n_filled

def _evaluate_register(window_positions, pos_abun, win_start, win_end, phase, reg, forward=True):
    """
    Wobble-tolerant register evaluation (±1 positional wobble).
    Returns: (in_phase_sum, effective_total, n_filled_cycles)
    Semantics:
      - If the exact expected position exists, count it for in-phase and quarantine its ±1 neighbors.
      - Else, pick the better of the ±1 neighbors (if any) and quarantine the sibling neighbor.
      - Each genomic position is counted at most once in-phase.
      - Effective total excludes quarantined neighbors so they don't inflate U.
    """
    ph = int(phase)
    positions_set = set(window_positions)
    num_cycles = max(0, (win_end - win_start + 1) // ph)

    used_positions = set()     # positions used as in-phase
    ignored_positions = set()  # neighbors to exclude from effective_total
    in_phase_sum = 0.0
    n_filled = 0

    for c in range(num_cycles):
        expected_pos = (win_start + reg + c * ph) if forward else (win_end - reg - c * ph)

        # Case 1: exact exists -> use and quarantine neighbors
        if expected_pos in positions_set and expected_pos not in used_positions:
            in_phase_sum += pos_abun[expected_pos]
            used_positions.add(expected_pos)
            n_filled += 1
            for off in (-1, 1):
                npos = expected_pos + off
                if npos in positions_set and npos not in used_positions:
                    ignored_positions.add(npos)
        else:
            # Case 2: consider ±1; choose best if present; quarantine sibling neighbor
            left = expected_pos - 1
            right = expected_pos + 1
            candidates = []
            if left in positions_set and left not in used_positions:
                candidates.append(left)
            if right in positions_set and right not in used_positions:
                candidates.append(right)
            if candidates:
                best = max(candidates, key=lambda p: pos_abun[p])
                in_phase_sum += pos_abun[best]
                used_positions.add(best)
                n_filled += 1
                sibling = right if best == left else left
                if sibling in positions_set and sibling not in used_positions:
                    ignored_positions.add(sibling)

    # Effective total excludes quarantined neighbors of exact/selected hits
    effective_positions = [p for p in window_positions if p not in ignored_positions]
    effective_total = sum(pos_abun[p] for p in effective_positions)
    return in_phase_sum, effective_total, n_filled


def _score_relaxed_window(window_positions, pos_abun, win_start, win_end, phase, forward=True):
    summary = _score_window_registers(
        window_positions,
        pos_abun,
        win_start,
        win_end,
        phase,
        forward=forward,
    )
    return (
        float(summary["score"]),
        float(summary["best_in_phase_sum"]),
        float(summary["best_out_of_phase"]),
        int(summary["best_filled"]),
        summary["best_register"],
    )


def _enumerate_relaxed_trace_for_strand(pos_abun, phase, win_size, *, forward=True):
    positions = sorted(pos_abun.keys())
    if not positions:
        return []

    if forward:
        anchors = positions
    else:
        anchors = list(reversed(positions))

    trace = []
    for anchor in anchors:
        if forward:
            win_start = int(anchor)
            win_end = int(anchor) + int(win_size) - 1
            anchor_position = int(anchor)
        else:
            win_end = int(anchor)
            win_start = int(anchor) - int(win_size) + 1
            anchor_position = int(anchor)

        window_positions = [p for p in positions if win_start <= p <= win_end]
        score, in_phase_sum, out_of_phase, n_filled, best_reg = _score_relaxed_window(
            window_positions,
            pos_abun,
            win_start,
            win_end,
            int(phase),
            forward=forward,
        )
        trace.append(
            {
                "anchor_position": anchor_position,
                "window_start": int(win_start),
                "window_end": int(win_end),
                "score": float(score),
                "in_phase_abund": float(in_phase_sum),
                "out_phase_abund": float(out_of_phase),
                "occupied_cycles": int(n_filled),
                "best_register": best_reg,
            }
        )
    return trace


def _find_relaxed_trace_peak_row(trace_rows) -> dict | None:
    best_row = None
    best_score = float("-inf")
    for row in trace_rows or []:
        try:
            score = float(row.get("score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        if score >= best_score:
            best_score = score
            best_row = row
    return best_row


def _same_trace_window(a: dict | None, b: dict | None) -> bool:
    if not a or not b:
        return False
    try:
        return (
            int(a.get("anchor_position")) == int(b.get("anchor_position"))
            and int(a.get("window_start")) == int(b.get("window_start"))
            and int(a.get("window_end")) == int(b.get("window_end"))
        )
    except Exception:
        return False


def _normalize_trace_strand_code(strand_code) -> str:
    text = str(strand_code or "").strip().lower()
    if text in {"c", "-", "crick", "0", "false"}:
        return "c"
    return "w"


def _is_forward_trace_strand(strand_code) -> bool:
    return _normalize_trace_strand_code(strand_code) == "w"


def _build_relaxed_trace_register_origin(peak_row: dict | None, phase: int, strand_code) -> int | None:
    if not peak_row:
        return None
    best_register = peak_row.get("best_register")
    if best_register is None:
        return None

    # `best_register` is a 0-based cycle index within the phased window.
    if _is_forward_trace_strand(strand_code):
        return int(peak_row["window_start"]) + int(best_register)
    return int(peak_row["window_end"]) - int(best_register)


def _classify_relaxed_trace_relation(anchor_position: int, register_origin: int | None, phase: int):
    if register_origin is None:
        return "other", None

    delta = int(anchor_position) - int(register_origin)
    remainder = delta % int(phase)
    if remainder == 0:
        return "exact", int(anchor_position)
    if remainder == 1:
        return "offset", int(anchor_position) - 1
    if remainder == int(phase) - 1:
        return "offset", int(anchor_position) + 1
    return "other", None


def _compute_phase_shift_nt(
    reference_origin: int | None,
    candidate_origin: int | None,
    phase: int,
) -> int | None:
    if reference_origin is None or candidate_origin is None:
        return None

    try:
        phase_local = int(phase)
        if phase_local <= 0:
            return None
        shift_value = (int(candidate_origin) - int(reference_origin)) % phase_local
    except Exception:
        return None

    if shift_value == 0:
        return None
    half_phase = phase_local / 2.0
    if shift_value > half_phase:
        shift_value -= phase_local
    return int(shift_value)


def _expected_duplex_partner_shift_nt(reference_strand, candidate_strand) -> int | None:
    ref_local = _normalize_trace_strand_code(reference_strand)
    cand_local = _normalize_trace_strand_code(candidate_strand)
    if ref_local == cand_local:
        return 0
    # The trace/register abstraction stores strand-specific lattice origins, and
    # the expected cross-strand duplex partner projects with the same canonical
    # 2-nt 3' overhang offset in either direction.
    return int(DCL_OVERHANG)


def _expected_duplex_partner_register_origin(
    source_register_origin: int | None,
    *,
    source_strand,
    target_strand,
    phase: int,
) -> int | None:
    if source_register_origin is None:
        return None
    source_local = _normalize_trace_strand_code(source_strand)
    target_local = _normalize_trace_strand_code(target_strand)
    if source_local == target_local:
        return int(source_register_origin)

    cycle_span = (int(WINDOW_MULTIPLIER) - 1) * int(phase)
    if source_local == "w":
        return int(source_register_origin) + int(cycle_span) + int(DCL_OVERHANG)
    return int(source_register_origin) - int(cycle_span) + int(DCL_OVERHANG)


def _duplex_shift_is_orientation_compatible(
    reference_strand,
    candidate_strand,
    shift_nt: int | None,
) -> bool:
    expected_shift = _expected_duplex_partner_shift_nt(reference_strand, candidate_strand)
    if expected_shift is None:
        return False
    return abs(int(shift_nt or 0)) == abs(int(expected_shift))


def _canonical_partner_shift_equivalent(shift_nt: int | None, *, cross_strand: bool) -> bool:
    if not bool(cross_strand):
        return False
    if shift_nt is None:
        return False
    try:
        return abs(int(shift_nt)) == int(DCL_OVERHANG)
    except Exception:
        return False


def _normalized_public_partner_shift_nt(detail: dict | None) -> int | None:
    if not detail or not bool(detail.get("cross_strand")):
        return None
    if not bool(detail.get("canonical_compatible")):
        return None
    shift_nt = detail.get("shift_nt")
    if shift_nt is None:
        return None
    try:
        shift_local = int(shift_nt)
    except Exception:
        return None
    if shift_local == 0:
        return None
    return int(DCL_OVERHANG) if shift_local > 0 else -int(DCL_OVERHANG)


def _partner_candidate_tier(detail: dict | None) -> str:
    if not detail:
        return "rejected"
    if not bool(detail.get("cross_strand")):
        return "same_strand_extension"
    if bool(detail.get("canonical_compatible")):
        return "canonical"
    return "fallback_noncanonical"


def _partner_match_ranking(detail: dict | None, peak_score: float | int | None) -> tuple:
    if not detail:
        return (9, 9, 0, 0.0, 0, 0.0, 9999, 9999)
    shift_nt = detail.get("shift_nt", 0)
    try:
        shift_local = int(shift_nt or 0)
    except Exception:
        shift_local = 0
    try:
        peak_local = float(peak_score or 0.0)
    except Exception:
        peak_local = 0.0
    return (
        0 if bool(detail.get("cross_strand")) else 1,
        0 if bool(detail.get("canonical_compatible")) else 1,
        -int(detail.get("shared_cycles", 0) or 0),
        -float(detail.get("support_ratio", 0.0) or 0.0),
        -int(detail.get("candidate_exact_positions", 0) or 0),
        -peak_local,
        abs(int(shift_local)),
        int(shift_local),
    )


def _member_geometry_positions(
    member: dict | None,
    *,
    phase: int,
) -> list[int]:
    positions = _member_supported_positions(member)
    if positions:
        return [int(value) for value in positions]
    if not member:
        return []
    return _build_candidate_register_positions(
        member.get("register_origin"),
        int(phase),
        member.get("strand"),
    )


def _cross_strand_duplex_projected_overlap(
    reference_member: dict | None,
    candidate_member: dict | None,
    *,
    phase: int,
) -> int:
    if not reference_member or not candidate_member:
        return 0

    reference_origin = reference_member.get("register_origin")
    candidate_origin = candidate_member.get("register_origin")
    if reference_origin is None or candidate_origin is None:
        return 0

    reference_positions = _member_geometry_positions(reference_member, phase=int(phase))
    candidate_positions = _member_geometry_positions(candidate_member, phase=int(phase))
    if not reference_positions or not candidate_positions:
        return 0

    projected_positions = _project_register_positions_to_strand(
        reference_positions,
        source_strand=reference_member.get("strand"),
        target_strand=candidate_member.get("strand"),
        source_register_origin=reference_origin,
        target_register_origin=candidate_origin,
        phase=int(phase),
    )
    if not projected_positions:
        return 0
    return int(len(set(int(value) for value in projected_positions) & set(int(value) for value in candidate_positions)))


def _duplex_geometry_match_detail(
    reference_member: dict | None,
    candidate_member: dict | None,
    *,
    phase: int,
) -> dict:
    result = {
        "cross_strand": False,
        "shift_nt": 0,
        "raw_orientation_ok": False,
        "canonical_compatible": False,
        "projected_shared_cycles": 0,
        "compatible": False,
    }
    if not reference_member or not candidate_member:
        return result

    reference_origin = reference_member.get("register_origin")
    candidate_origin = candidate_member.get("register_origin")
    if reference_origin is None or candidate_origin is None:
        return result

    reference_strand = _normalize_trace_strand_code(reference_member.get("strand"))
    candidate_strand = _normalize_trace_strand_code(candidate_member.get("strand"))
    raw_shift = _compute_phase_shift_nt(reference_origin, candidate_origin, int(phase))
    shift_nt = 0 if raw_shift is None else int(raw_shift)
    cross_strand = candidate_strand != reference_strand

    result["cross_strand"] = bool(cross_strand)
    result["shift_nt"] = int(shift_nt)
    if not cross_strand:
        result["raw_orientation_ok"] = int(shift_nt) == 0
        result["compatible"] = bool(result["raw_orientation_ok"])
        return result

    raw_orientation_ok = _duplex_shift_is_orientation_compatible(
        reference_strand,
        candidate_strand,
        shift_nt,
    )
    projected_shared_cycles = _cross_strand_duplex_projected_overlap(
        reference_member,
        candidate_member,
        phase=int(phase),
    )
    result["raw_orientation_ok"] = bool(raw_orientation_ok)
    result["canonical_compatible"] = bool(
        _canonical_partner_shift_equivalent(shift_nt, cross_strand=True)
    )
    result["projected_shared_cycles"] = int(projected_shared_cycles)
    result["compatible"] = bool(raw_orientation_ok or projected_shared_cycles > 0)
    return result


def _project_register_positions_to_strand(
    positions,
    *,
    source_strand,
    target_strand,
    source_register_origin: int | None = None,
    target_register_origin: int | None = None,
    phase: int | None = None,
) -> list[int]:
    source_local = _normalize_trace_strand_code(source_strand)
    target_local = _normalize_trace_strand_code(target_strand)
    coords = sorted({int(value) for value in (positions or [])})
    if not coords:
        return []
    if source_local == target_local:
        return coords

    if source_register_origin is None or target_register_origin is None or phase is None:
        return []
    phase_local = int(phase)
    if phase_local <= 0:
        return []

    source_origin_local = int(source_register_origin)
    target_origin_local = int(target_register_origin)
    projected = []
    for value in coords:
        if source_local == "w":
            cycle_index = (int(value) - source_origin_local) // phase_local
        else:
            cycle_index = (source_origin_local - int(value)) // phase_local
        if target_local == "w":
            projected.append(target_origin_local + cycle_index * phase_local)
        else:
            projected.append(target_origin_local - cycle_index * phase_local)
    return sorted({int(value) for value in projected})


def _summarize_group_register_support_rows(
    rows,
    *,
    register_origin: int | None,
    phase: int,
) -> dict:
    exact_row_count = 0
    offset_row_count = 0
    other_row_count = 0
    relation_rows = []
    for row in rows or []:
        try:
            anchor_position = int(row.get("anchor_position"))
        except Exception:
            continue
        relation, expected_position = _classify_relaxed_trace_relation(
            anchor_position,
            register_origin,
            int(phase),
        )
        relation_rows.append(
            {
                **dict(row),
                "phase_relation": relation,
                "expected_position": expected_position,
            }
        )
        if relation == "exact":
            exact_row_count += 1
        elif relation == "offset":
            offset_row_count += 1
        else:
            other_row_count += 1
    return {
        "exact_row_count": int(exact_row_count),
        "offset_row_count": int(offset_row_count),
        "other_row_count": int(other_row_count),
        "non_grey_row_count": int(exact_row_count + offset_row_count),
        "relation_rows": relation_rows,
    }


def _describe_biogenesis_match(
    group_a: dict,
    group_b: dict,
    *,
    phase: int,
) -> dict | None:
    if not _windows_overlap(
        group_a.get("min_start"),
        group_a.get("max_end"),
        group_b.get("min_start"),
        group_b.get("max_end"),
    ):
        return None

    register_origin_a = group_a.get("register_origin")
    register_origin_b = group_b.get("register_origin")
    if register_origin_a is None or register_origin_b is None:
        return None

    strand_a = _normalize_trace_strand_code(group_a.get("strand"))
    strand_b = _normalize_trace_strand_code(group_b.get("strand"))
    geometry_detail = _duplex_geometry_match_detail(
        {
            "strand": strand_a,
            "register_origin": register_origin_a,
            "relation_rows": list(group_a.get("relation_rows") or []),
        },
        {
            "strand": strand_b,
            "register_origin": register_origin_b,
            "relation_rows": list(group_b.get("relation_rows") or []),
        },
        phase=int(phase),
    )
    cross_strand = bool(geometry_detail.get("cross_strand"))
    shift_local = int(geometry_detail.get("shift_nt", 0) or 0)

    if not bool(geometry_detail.get("compatible")):
        return None

    positions_a = _build_candidate_register_positions(
        register_origin_a,
        int(phase),
        strand_a,
    )
    positions_b = _build_candidate_register_positions(
        register_origin_b,
        int(phase),
        strand_b,
    )
    if not positions_a or not positions_b:
        return None

    if cross_strand and not bool(geometry_detail.get("raw_orientation_ok")):
        shared_cycles = int(geometry_detail.get("projected_shared_cycles", 0) or 0)
    else:
        shared_cycles = int(
            _count_shifted_register_matches(
                positions_a,
                positions_b,
                shift_nt=int(shift_local),
            )
        )
    if shared_cycles < int(ALTERNATIVE_MIN_SHARED_CYCLES):
        return None

    return {
        "shared_cycles": int(shared_cycles),
        "shift_nt": int(shift_local),
        "cross_strand": bool(cross_strand),
        "paired_unit": bool(cross_strand),
    }


def classify_browser_style_relaxed_trace(
    trace: dict,
    *,
    phase: int | None = None,
    crowded_gap: float = HOWELL_CROWDING_SCORE_GAP,
) -> dict:
    """
    Classify relaxed trace windows using browser-style semantics.

    Returns a dict with per-strand classified rows plus crowding metrics derived
    from non-in-phase windows on the winning strand that overlap the relaxed
    HPSP window and stay within the requested Howell score gap.
    """
    ph = int(phase) if phase is not None else _phase_value()
    result = {
        "w": [],
        "c": [],
        "strand_hpsp_rows": {"w": None, "c": None},
        "strand_register_origins": {"w": None, "c": None},
        "winner_strand": None,
        "winner_row": None,
        "Howell_crowding_window_count": 0,
        "Howell_crowding_best_score": np.nan,
        "Howell_crowding_score_gap": np.nan,
        "crowding_rows": [],
    }

    winner_strand = None
    winner_row = None
    winner_score = float("-inf")
    for strand_code in ("w", "c"):
        peak_row = _find_relaxed_trace_peak_row(trace.get(strand_code, []) or [])
        result["strand_hpsp_rows"][strand_code] = peak_row
        register_origin = _build_relaxed_trace_register_origin(peak_row, ph, strand_code)
        result["strand_register_origins"][strand_code] = register_origin
        if peak_row is None:
            continue
        try:
            peak_score = float(peak_row.get("score", 0.0) or 0.0)
        except Exception:
            peak_score = 0.0
        if peak_score >= winner_score:
            winner_score = peak_score
            winner_strand = strand_code
            winner_row = peak_row

    result["winner_strand"] = winner_strand
    result["winner_row"] = winner_row

    for strand_code in ("w", "c"):
        peak_row = result["strand_hpsp_rows"][strand_code]
        register_origin = result["strand_register_origins"][strand_code]
        strand_rows = []
        for row in trace.get(strand_code, []) or []:
            try:
                anchor_position = int(row.get("anchor_position"))
            except Exception:
                continue
            relation, expected_position = _classify_relaxed_trace_relation(
                anchor_position,
                register_origin,
                ph,
            )
            classified = dict(row)
            classified["strand"] = strand_code
            classified["phase_relation"] = relation
            classified["expected_register_position"] = expected_position
            classified["register_origin"] = register_origin
            classified["is_hpsp"] = _same_trace_window(row, peak_row)
            strand_rows.append(classified)
        result[strand_code] = strand_rows

    crowding_rows = []
    if winner_row is not None and winner_strand is not None:
        for row in result.get(winner_strand, []) or []:
            if row.get("is_hpsp"):
                continue
            if str(row.get("phase_relation")) != "other":
                continue
            if not _windows_overlap(
                row.get("window_start"),
                row.get("window_end"),
                winner_row.get("window_start"),
                winner_row.get("window_end"),
            ):
                continue
            try:
                candidate_score = float(row.get("score", 0.0) or 0.0)
            except Exception:
                continue
            if abs(float(winner_score) - candidate_score) > float(crowded_gap):
                continue
            crowding_rows.append(dict(row))

    result["crowding_rows"] = crowding_rows
    result["Howell_crowding_window_count"] = int(len(crowding_rows))
    if crowding_rows:
        best_crowding_score = max(float(row.get("score", 0.0) or 0.0) for row in crowding_rows)
        result["Howell_crowding_best_score"] = float(best_crowding_score)
        result["Howell_crowding_score_gap"] = float(float(winner_score) - best_crowding_score)
    return result


def _group_trace_rows_by_overlap(trace_rows, *, score_cutoff: float) -> list[dict]:
    qualifying_rows = []
    for row in trace_rows or []:
        try:
            score = float(row.get("score", 0.0) or 0.0)
        except Exception:
            continue
        if score_cutoff is not None and score < float(score_cutoff):
            continue
        qualifying_rows.append(row)

    qualifying_rows.sort(
        key=lambda row: (
            int(row.get("window_start")),
            int(row.get("window_end")),
            int(row.get("anchor_position")),
        )
    )

    groups = []
    for row in qualifying_rows:
        window_start = int(row.get("window_start"))
        window_end = int(row.get("window_end"))
        if not groups or window_start > groups[-1]["max_end"]:
            groups.append(
                {
                    "rows": [row],
                    "min_start": window_start,
                    "max_end": window_end,
                }
            )
        else:
            groups[-1]["rows"].append(row)
            groups[-1]["min_start"] = min(groups[-1]["min_start"], window_start)
            groups[-1]["max_end"] = max(groups[-1]["max_end"], window_end)
    return groups


def _build_relaxed_group_summary(
    group: dict,
    *,
    strand_code: str,
    category: str,
    phase: int | None = None,
    winner_register_origin: int | None = None,
    winner_window: tuple[int, int] | None = None,
    register_origin_override: int | None = None,
) -> dict | None:
    rows = list(group.get("rows") or [])
    if not rows:
        return None

    peak_row = _find_relaxed_trace_peak_row(rows)
    if peak_row is None:
        return None

    ph = int(phase) if phase is not None else None
    register_origin = None
    shift_nt = None
    if ph is not None:
        if register_origin_override is not None:
            register_origin = int(register_origin_override)
        else:
            register_origin = _build_relaxed_trace_register_origin(peak_row, ph, strand_code)
        shift_nt = _compute_phase_shift_nt(winner_register_origin, register_origin, ph)
    support_summary = _summarize_group_register_support_rows(
        rows,
        register_origin=register_origin,
        phase=(1 if ph is None else ph),
    )

    try:
        peak_score = float(peak_row.get("score", 0.0) or 0.0)
    except Exception:
        peak_score = 0.0

    return {
        "category": str(category),
        "strand": str(strand_code),
        "rows": rows,
        "peak_row": peak_row,
        "peak_score": float(peak_score),
        "register_origin": register_origin,
        "shift_nt": shift_nt,
        "min_start": int(group.get("min_start")),
        "max_end": int(group.get("max_end")),
        "exact_row_count": int(support_summary.get("exact_row_count", 0)),
        "offset_row_count": int(support_summary.get("offset_row_count", 0)),
        "other_row_count": int(support_summary.get("other_row_count", 0)),
        "non_grey_row_count": int(support_summary.get("non_grey_row_count", 0)),
        "relation_rows": list(support_summary.get("relation_rows") or []),
        "overlaps_winner_window": (
            False
            if winner_window is None
            else _windows_overlap(
                int(group.get("min_start")),
                int(group.get("max_end")),
                int(winner_window[0]),
                int(winner_window[1]),
            )
        ),
    }


def _trace_row_key(row: dict | None):
    if not row:
        return None
    try:
        return (
            _normalize_trace_strand_code(row.get("strand")),
            int(row.get("anchor_position")),
            int(row.get("window_start")),
            int(row.get("window_end")),
            None if row.get("best_register") is None else int(row.get("best_register")),
        )
    except Exception:
        return None


def _relation_support_weight(relation: str | None) -> float:
    relation_local = str(relation or "").strip()
    if relation_local == "exact":
        return 1.0
    if relation_local == "offset":
        return 0.5
    return 0.0


def _member_support_weight_map(member: dict | None) -> dict[int, float]:
    support_map: dict[int, float] = {}
    for row in list((member or {}).get("relation_rows") or []):
        expected_position = row.get("expected_position")
        if expected_position is None:
            continue
        weight = _relation_support_weight(row.get("phase_relation"))
        if weight <= 0.0:
            continue
        expected_local = int(expected_position)
        support_map[expected_local] = max(float(weight), float(support_map.get(expected_local, 0.0)))
    return support_map


def _member_supported_positions(member: dict | None) -> list[int]:
    return sorted(_member_support_weight_map(member).keys())


def _trace_support_weight_map(
    trace_rows,
    *,
    register_origin: int | None,
    phase: int,
) -> dict[int, float]:
    if register_origin is None:
        return {}

    support_map: dict[int, float] = {}
    for row in trace_rows or []:
        try:
            anchor_position = int(row.get("anchor_position"))
        except Exception:
            continue
        relation, expected_position = _classify_relaxed_trace_relation(
            anchor_position,
            register_origin,
            int(phase),
        )
        if expected_position is None:
            continue
        weight = _relation_support_weight(relation)
        if weight <= 0.0:
            continue
        expected_local = int(expected_position)
        support_map[expected_local] = max(float(weight), float(support_map.get(expected_local, 0.0)))
    return support_map


def _supported_position_counts(member: dict | None) -> tuple[int, int]:
    support_map = _member_support_weight_map(member)
    exact_count = 0
    for row in list((member or {}).get("relation_rows") or []):
        if row.get("expected_position") is None:
            continue
        if str(row.get("phase_relation")) == "exact":
            exact_count += 1
    return int(len(support_map)), int(exact_count)


def _build_lattice_positions_between(positions, *, phase: int) -> list[int]:
    coords = sorted({int(value) for value in (positions or [])})
    if not coords:
        return []
    if len(coords) == 1:
        return [coords[0]]
    start = int(coords[0])
    end = int(coords[-1])
    return list(range(start, end + 1, int(phase)))


def _max_consecutive_unsupported(weights) -> int:
    longest = 0
    current = 0
    for weight in weights or []:
        if float(weight) <= 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _trim_positions_to_supported_core(positions, weights) -> tuple[list[int], list[float]]:
    coords = [int(value) for value in (positions or [])]
    weight_values = [float(value) for value in (weights or [])]
    if not coords or not weight_values or len(coords) != len(weight_values):
        return coords, weight_values
    supported_indexes = [index for index, weight in enumerate(weight_values) if float(weight) > 0.0]
    if not supported_indexes:
        return coords, weight_values
    start_index = int(min(supported_indexes))
    end_index = int(max(supported_indexes))
    return coords[start_index : end_index + 1], weight_values[start_index : end_index + 1]


def _candidate_bridge_reference_positions(
    candidate: dict,
    main_members,
    *,
    phase: int,
) -> list[int]:
    candidate_strand = _normalize_trace_strand_code(candidate.get("strand"))
    candidate_origin = candidate.get("register_origin")
    projected = []
    for member in main_members or []:
        projected.extend(
            _project_register_positions_to_strand(
                _member_supported_positions(member),
                source_strand=member.get("strand"),
                target_strand=candidate_strand,
                source_register_origin=member.get("register_origin"),
                target_register_origin=candidate_origin,
                phase=int(phase),
            )
        )
    return sorted({int(value) for value in projected})


def _opposite_trace_strand(strand_code) -> str:
    return "c" if _normalize_trace_strand_code(strand_code) == "w" else "w"


def _synthesize_projected_partner_candidate(
    winner_member: dict | None,
    main_members,
    trace_rows_by_strand: dict,
    *,
    phase: int,
    require_canonical_cross_strand: bool = False,
    debug_context: dict | None = None,
    debug_rows=None,
) -> tuple[dict | None, dict | None]:
    if winner_member is None:
        return None, None

    winner_origin = winner_member.get("register_origin")
    if winner_origin is None:
        return None, None

    candidate_strand = _opposite_trace_strand(winner_member.get("strand"))
    candidate_origin = _expected_duplex_partner_register_origin(
        winner_origin,
        source_strand=winner_member.get("strand"),
        target_strand=candidate_strand,
        phase=int(phase),
    )
    candidate_rows = list(trace_rows_by_strand.get(candidate_strand, []) or [])
    if candidate_origin is None or not candidate_rows:
        return None, None

    origin_candidates = [int(candidate_origin)]
    seen_origins = {int(candidate_origin)}
    scored_rows = []
    for row in candidate_rows:
        row_origin = _build_relaxed_trace_register_origin(row, int(phase), candidate_strand)
        if row_origin is None:
            continue
        try:
            row_score = float(row.get("score", 0.0) or 0.0)
        except Exception:
            row_score = 0.0
        scored_rows.append((float(row_score), int(row_origin)))
    for _, row_origin in sorted(scored_rows, key=lambda item: item[0], reverse=True):
        if row_origin in seen_origins:
            continue
        seen_origins.add(row_origin)
        origin_candidates.append(int(row_origin))
        if len(origin_candidates) >= 64:
            break

    best_member = None
    best_detail = None
    for origin_local in origin_candidates:
        reference_positions = _candidate_bridge_reference_positions(
            {"strand": candidate_strand, "register_origin": int(origin_local)},
            main_members,
            phase=int(phase),
        )
        if not reference_positions:
            continue

        span_positions = _build_lattice_positions_between(reference_positions, phase=int(phase))
        expected_positions = set(int(value) for value in (span_positions or reference_positions))
        selected_rows = []
        for row in candidate_rows:
            try:
                anchor_position = int(row.get("anchor_position"))
            except Exception:
                continue
            relation, expected_position = _classify_relaxed_trace_relation(
                anchor_position,
                int(origin_local),
                int(phase),
            )
            if relation not in {"exact", "offset"} or expected_position is None:
                continue
            if int(expected_position) not in expected_positions:
                continue
            selected_rows.append(dict(row))

        if len(selected_rows) < int(MAIN_UNIT_MIN_SUPPORTED_POSITIONS):
            _emit_main_partner_trace(
                debug_rows,
                debug_context,
                attempt_stage="synthetic_projection",
                candidate_source="synthetic_projected",
                candidate_category="main_partner_candidate",
                candidate_strand=candidate_strand,
                candidate_register_origin_tested=int(origin_local),
                observed_shift_nt=_compute_phase_shift_nt(
                    winner_member.get("register_origin"),
                    int(origin_local),
                    int(phase),
                ),
                duplex_orientation_ok=np.nan,
                canonical_compatible=np.nan,
                candidate_tier="synthetic_projected",
                shared_cycles=0,
                bridge_support_ratio=0.0,
                exact_support_count=0,
                supported_position_count=int(len(selected_rows)),
                max_unsupported_run=0,
                candidate_peak_score=np.nan,
                candidate_non_grey_row_count=int(len(selected_rows)),
                first_reject_reason="support_ratio_below_threshold",
                final_route="rejected",
            )
            continue

        synthetic_group = {
            "rows": selected_rows,
            "min_start": min(int(row.get("window_start")) for row in selected_rows),
            "max_end": max(int(row.get("window_end")) for row in selected_rows),
        }
        synthetic_member = _build_relaxed_group_summary(
            synthetic_group,
            strand_code=candidate_strand,
            category="main_partner_candidate",
            phase=int(phase),
            winner_register_origin=int(winner_origin),
            register_origin_override=int(origin_local),
        )
        if synthetic_member is None:
            continue

        match_detail = _evaluate_main_unit_candidate_match(
            winner_member,
            synthetic_member,
            main_members,
            trace_rows_by_strand,
            phase=int(phase),
        )
        if (
            bool(require_canonical_cross_strand)
            and bool(match_detail.get("accepted"))
            and bool(match_detail.get("cross_strand"))
            and not bool(match_detail.get("canonical_compatible"))
        ):
            match_detail = dict(match_detail)
            match_detail["accepted"] = False
            match_detail["reject_reason"] = "noncanonical_partner_rerouted_to_secondary"
        _emit_main_partner_trace(
            debug_rows,
            debug_context,
            attempt_stage="synthetic_projection",
            candidate_source="synthetic_projected",
            candidate_category="main_partner_candidate",
            candidate_strand=candidate_strand,
            candidate_register_origin_tested=int(origin_local),
            observed_shift_nt=match_detail.get("shift_nt"),
            normalized_shift_nt=_normalized_public_partner_shift_nt(match_detail),
            duplex_orientation_ok=match_detail.get("duplex_orientation_ok"),
            canonical_compatible=match_detail.get("canonical_compatible"),
            candidate_tier=match_detail.get("candidate_tier"),
            shared_cycles=match_detail.get("shared_cycles"),
            bridge_support_ratio=match_detail.get("support_ratio"),
            exact_support_count=match_detail.get("candidate_exact_positions"),
            supported_position_count=match_detail.get("candidate_supported_positions"),
            max_unsupported_run=match_detail.get("max_zero_run"),
            candidate_peak_score=synthetic_member.get("peak_score"),
            candidate_non_grey_row_count=synthetic_member.get("non_grey_row_count"),
            first_reject_reason=match_detail.get("reject_reason"),
            accept_route=("candidate_match" if bool(match_detail.get("accepted")) else np.nan),
            final_route=("accepted_match" if bool(match_detail.get("accepted")) else "rejected"),
        )
        if not bool(match_detail.get("accepted")) or not bool(match_detail.get("cross_strand")):
            continue

        ranking = _partner_match_ranking(
            match_detail,
            synthetic_member.get("peak_score", 0.0),
        )
        if best_detail is None or ranking < _partner_match_ranking(
            best_detail,
            None if best_member is None else best_member.get("peak_score", 0.0),
        ):
            best_member = dict(synthetic_member)
            best_detail = dict(match_detail)

    return best_member, best_detail


def _member_origin_candidates(member: dict | None, *, phase: int) -> list[int]:
    if member is None:
        return []
    origins = []
    seen = set()

    current_origin = member.get("register_origin")
    if current_origin is not None:
        current_local = int(current_origin)
        origins.append(current_local)
        seen.add(current_local)

    for row in list((member or {}).get("rows") or []):
        row_origin = _build_relaxed_trace_register_origin(
            row,
            int(phase),
            member.get("strand"),
        )
        if row_origin is None:
            continue
        row_local = int(row_origin)
        if row_local in seen:
            continue
        origins.append(row_local)
        seen.add(row_local)
    return origins


def _reanchor_member_register_origin(
    member: dict | None,
    *,
    register_origin: int,
    phase: int,
) -> dict | None:
    if member is None:
        return None
    rows = [dict(row) for row in list((member or {}).get("rows") or [])]
    if not rows:
        return None
    group = {
        "rows": rows,
        "min_start": min(int(row.get("window_start")) for row in rows),
        "max_end": max(int(row.get("window_end")) for row in rows),
    }
    summary = _build_relaxed_group_summary(
        group,
        strand_code=str(member.get("strand", "w")),
        category=str(member.get("category", "trace_segment")),
        phase=int(phase),
        register_origin_override=int(register_origin),
    )
    if summary is None:
        return None
    for key in (
        "overlaps_winner_window",
        "paired_unit",
        "member_count",
        "member_strands",
        "raw_categories",
    ):
        if key in member:
            summary[key] = member.get(key)
    return summary


def _evaluate_main_unit_bridge(
    candidate: dict,
    main_members,
    trace_rows_by_strand: dict,
    *,
    phase: int,
    cross_strand: bool = False,
) -> dict | None:
    candidate_supported_count, candidate_exact_count = _supported_position_counts(candidate)
    if candidate_supported_count < int(MAIN_UNIT_MIN_SUPPORTED_POSITIONS):
        return {
            "support_ratio": 0.0,
            "max_zero_run": 0,
            "shared_cycles": 0,
            "candidate_supported_positions": int(candidate_supported_count),
            "candidate_exact_positions": int(candidate_exact_count),
            "passes": False,
            "reject_reason": "support_ratio_below_threshold",
        }
    if candidate_exact_count < int(MAIN_UNIT_MIN_EXACT_POSITIONS):
        return {
            "support_ratio": 0.0,
            "max_zero_run": 0,
            "shared_cycles": 0,
            "candidate_supported_positions": int(candidate_supported_count),
            "candidate_exact_positions": int(candidate_exact_count),
            "passes": False,
            "reject_reason": "exact_support_below_threshold",
        }

    reference_positions = _candidate_bridge_reference_positions(
        candidate,
        main_members,
        phase=int(phase),
    )
    candidate_positions = _member_supported_positions(candidate)
    if not reference_positions or not candidate_positions:
        return {
            "support_ratio": 0.0,
            "max_zero_run": 0,
            "shared_cycles": 0,
            "candidate_supported_positions": int(candidate_supported_count),
            "candidate_exact_positions": int(candidate_exact_count),
            "passes": False,
            "reject_reason": "support_ratio_below_threshold",
        }

    span_positions = _build_lattice_positions_between(
        list(reference_positions) + list(candidate_positions),
        phase=int(phase),
    )
    if bool(cross_strand):
        overlap_start = max(min(reference_positions), min(candidate_positions))
        overlap_end = min(max(reference_positions), max(candidate_positions))
        if overlap_start <= overlap_end:
            overlap_positions = [
                int(position)
                for position in span_positions
                if int(overlap_start) <= int(position) <= int(overlap_end)
            ]
            if overlap_positions:
                span_positions = overlap_positions
    if not span_positions:
        return {
            "support_ratio": 0.0,
            "max_zero_run": 0,
            "shared_cycles": 0,
            "candidate_supported_positions": int(candidate_supported_count),
            "candidate_exact_positions": int(candidate_exact_count),
            "passes": False,
            "reject_reason": "support_ratio_below_threshold",
        }

    candidate_strand = _normalize_trace_strand_code(candidate.get("strand"))
    strand_support = _trace_support_weight_map(
        trace_rows_by_strand.get(candidate_strand, []) or [],
        register_origin=candidate.get("register_origin"),
        phase=int(phase),
    )
    support_weights = [float(strand_support.get(int(pos), 0.0)) for pos in span_positions]
    if bool(cross_strand):
        span_positions, support_weights = _trim_positions_to_supported_core(span_positions, support_weights)
    if not span_positions or not support_weights:
        return {
            "support_ratio": 0.0,
            "max_zero_run": 0,
            "shared_cycles": 0,
            "candidate_supported_positions": int(candidate_supported_count),
            "candidate_exact_positions": int(candidate_exact_count),
            "passes": False,
            "reject_reason": "support_ratio_below_threshold",
        }
    support_ratio = float(sum(support_weights) / max(len(support_weights), 1))
    max_zero_run = _max_consecutive_unsupported(support_weights)
    shared_cycles = int(len(set(reference_positions) & set(candidate_positions)))
    reject_reason = None
    max_zero_run_limit = int(MAIN_UNIT_BRIDGE_MAX_ZERO_RUN)
    if bool(cross_strand):
        max_zero_run_limit = int(CROSS_STRAND_BRIDGE_MAX_ZERO_RUN)
    if max_zero_run > int(max_zero_run_limit):
        reject_reason = "unsupported_run_too_long"
    elif support_ratio < float(MAIN_UNIT_BRIDGE_MIN_RATIO):
        reject_reason = "support_ratio_below_threshold"
    elif (
        bool(cross_strand)
        and max_zero_run > int(MAIN_UNIT_BRIDGE_MAX_ZERO_RUN)
        and support_ratio < float(CROSS_STRAND_BRIDGE_RELAXED_MIN_RATIO)
    ):
        reject_reason = "unsupported_run_too_long"

    return {
        "support_ratio": float(support_ratio),
        "max_zero_run": int(max_zero_run),
        "shared_cycles": int(shared_cycles),
        "candidate_supported_positions": int(candidate_supported_count),
        "candidate_exact_positions": int(candidate_exact_count),
        "passes": bool(
            reject_reason is None
        ),
        "reject_reason": reject_reason,
    }


def _evaluate_main_unit_candidate_match(
    winner_member: dict | None,
    candidate: dict,
    main_members,
    trace_rows_by_strand: dict,
    *,
    phase: int,
) -> dict:
    result = {
        "accepted": False,
        "reject_reason": None,
        "shift_nt": 0,
        "cross_strand": False,
        "shared_cycles": 0,
        "support_ratio": 0.0,
        "max_zero_run": 0,
        "candidate_supported_positions": int(candidate.get("non_grey_row_count", 0) or 0),
        "candidate_exact_positions": int(candidate.get("exact_row_count", 0) or 0),
        "canonical_compatible": False,
        "candidate_tier": "rejected",
    }
    if winner_member is None:
        result["reject_reason"] = "row_origin_not_tested"
        return result

    winner_origin = winner_member.get("register_origin")
    candidate_origin = candidate.get("register_origin")
    if winner_origin is None or candidate_origin is None:
        result["reject_reason"] = "row_origin_not_tested"
        return result

    geometry_detail = _duplex_geometry_match_detail(
        winner_member,
        candidate,
        phase=int(phase),
    )
    result["shift_nt"] = int(geometry_detail.get("shift_nt", 0) or 0)
    result["cross_strand"] = bool(geometry_detail.get("cross_strand"))
    result["duplex_orientation_ok"] = bool(geometry_detail.get("compatible"))
    result["canonical_compatible"] = bool(geometry_detail.get("canonical_compatible"))
    result["candidate_tier"] = _partner_candidate_tier(result)
    cross_strand = bool(result["cross_strand"])

    if not bool(result["duplex_orientation_ok"]):
        result["reject_reason"] = "duplex_geometry_mismatch"
        return result

    bridge = _evaluate_main_unit_bridge(
        candidate,
        main_members,
        trace_rows_by_strand,
        phase=int(phase),
        cross_strand=cross_strand,
    )
    if bridge is None:
        result["reject_reason"] = "support_ratio_below_threshold"
        return result
    result.update(
        {
            "shared_cycles": int(bridge.get("shared_cycles", 0)),
            "support_ratio": float(bridge.get("support_ratio", 0.0)),
            "max_zero_run": int(bridge.get("max_zero_run", 0)),
            "candidate_supported_positions": int(bridge.get("candidate_supported_positions", 0)),
            "candidate_exact_positions": int(bridge.get("candidate_exact_positions", 0)),
        }
    )
    if not bool(bridge.get("passes")):
        result["reject_reason"] = str(bridge.get("reject_reason") or "support_ratio_below_threshold")
        return result
    if cross_strand and int(bridge.get("candidate_supported_positions", 0)) < int(ALTERNATIVE_MIN_SHARED_CYCLES):
        result["reject_reason"] = "support_ratio_below_threshold"
        return result

    result["accepted"] = True
    return result


def _describe_main_unit_candidate_match(
    winner_member: dict | None,
    candidate: dict,
    main_members,
    trace_rows_by_strand: dict,
    *,
    phase: int,
) -> dict | None:
    detail = _evaluate_main_unit_candidate_match(
        winner_member,
        candidate,
        main_members,
        trace_rows_by_strand,
        phase=int(phase),
    )
    return None if not bool(detail.get("accepted")) else detail


def _build_all_trace_segment_candidates(
    trace: dict,
    winner_row: dict | None,
    winner_strand: str | None,
    *,
    phase: int,
) -> list[dict]:
    candidates = []
    winner_register_origin = (
        None
        if winner_row is None or winner_strand is None
        else _build_relaxed_trace_register_origin(winner_row, int(phase), winner_strand)
    )
    winner_window = (
        None
        if winner_row is None
        else (int(winner_row.get("window_start")), int(winner_row.get("window_end")))
    )

    for strand_code in ("w", "c"):
        groups = _group_trace_rows_by_overlap(
            trace.get(strand_code, []) or [],
            score_cutoff=0.0,
        )
        for group in groups:
            summary = _build_relaxed_group_summary(
                group,
                strand_code=strand_code,
                category="trace_segment",
                phase=int(phase),
                winner_register_origin=winner_register_origin,
                winner_window=winner_window,
            )
            if summary is None:
                continue
            peak_score = float(summary.get("peak_score", 0.0) or 0.0)
            non_grey_rows = int(summary.get("non_grey_row_count", 0) or 0)
            if peak_score <= 0.0 and non_grey_rows <= 0:
                continue
            candidates.append(summary)

    candidates.sort(
        key=lambda item: (
            -float(item.get("peak_score", 0.0) or 0.0),
            0 if bool(item.get("overlaps_winner_window")) else 1,
            int(item.get("min_start", 0) or 0),
        )
    )
    return candidates


def _unit_row_keys(unit: dict | None) -> set:
    row_keys = set()
    if unit is None:
        return row_keys
    for member in _unit_members(unit):
        for row in list(member.get("rows") or []):
            key = _trace_row_key(row)
            if key is not None:
                row_keys.add(key)
        peak_key = _trace_row_key(member.get("peak_row"))
        if peak_key is not None:
            row_keys.add(peak_key)
    return row_keys


def _main_unit_row_keys(main_unit: dict | None) -> set:
    row_keys = set()
    if not main_unit:
        return row_keys
    for member in list(main_unit.get("members") or []):
        row_keys.update(_unit_row_keys(member))
    return row_keys


def _filter_secondary_units_against_main_unit(
    candidate_units,
    main_unit: dict | None,
    *,
    claimed_row_keys=None,
) -> list[dict]:
    if not main_unit:
        return [dict(unit) for unit in (candidate_units or [])]

    main_keys = _main_unit_row_keys(main_unit)
    if claimed_row_keys:
        main_keys.update(set(claimed_row_keys))
    if not main_keys:
        return [dict(unit) for unit in (candidate_units or [])]

    remaining = []
    for unit in candidate_units or []:
        if _unit_row_keys(unit) & main_keys:
            continue
        remaining.append(dict(unit))
    return remaining


def _refine_main_unit_member(
    member: dict | None,
    *,
    phase: int,
    unit_role: str,
    category: str,
) -> dict | None:
    if member is None:
        return None

    def _finalize(member_local: dict) -> dict:
        output = dict(member_local)
        output["unit_role"] = str(unit_role)
        output["category"] = str(category)
        output["shift_nt"] = member.get("shift_nt", output.get("shift_nt", 0))
        output["overlaps_winner_window"] = bool(member.get("overlaps_winner_window"))
        return output

    relation_rows = [
        dict(row)
        for row in list((member or {}).get("relation_rows") or [])
        if str(row.get("phase_relation")) in {"exact", "offset"}
    ]
    if not relation_rows:
        return _finalize(dict(member))

    group = {
        "rows": relation_rows,
        "min_start": min(int(row.get("window_start")) for row in relation_rows),
        "max_end": max(int(row.get("window_end")) for row in relation_rows),
    }
    summary = _build_relaxed_group_summary(
        group,
        strand_code=str(member.get("strand", "w")),
        category=str(category),
        phase=int(phase),
        register_origin_override=member.get("register_origin"),
    )
    if summary is None:
        return _finalize(dict(member))

    return _finalize(summary)


def _best_main_unit_match_for_candidate(
    candidate: dict,
    winner_member: dict | None,
    main_members,
    trace_rows_by_strand: dict,
    *,
    phase: int,
    require_canonical_cross_strand: bool = False,
    debug_context: dict | None = None,
    debug_rows=None,
    attempt_stage: str = "candidate_match",
) -> tuple[dict | None, dict | None]:
    best_member = None
    best_detail = None
    for member in _unit_members(candidate):
        member_variants = [dict(member)]
        winner_strand = None if winner_member is None else _normalize_trace_strand_code(winner_member.get("strand"))
        member_strand = _normalize_trace_strand_code(member.get("strand"))
        if winner_strand is not None and member_strand != winner_strand:
            origin_candidates = []
            seen_origins = set()
            projected_origin = _expected_duplex_partner_register_origin(
                None if winner_member is None else winner_member.get("register_origin"),
                source_strand=None if winner_member is None else winner_member.get("strand"),
                target_strand=member.get("strand"),
                phase=int(phase),
            )
            if projected_origin is not None:
                projected_local = int(projected_origin)
                origin_candidates.append(projected_local)
                seen_origins.add(projected_local)
            for origin_local in _member_origin_candidates(member, phase=int(phase)):
                if int(origin_local) in seen_origins:
                    continue
                origin_candidates.append(int(origin_local))
                seen_origins.add(int(origin_local))
            for origin_local in origin_candidates:
                if member.get("register_origin") is not None and int(origin_local) == int(member.get("register_origin")):
                    continue
                variant = _reanchor_member_register_origin(
                    member,
                    register_origin=int(origin_local),
                    phase=int(phase),
                )
                if variant is not None:
                    member_variants.append(variant)

        for member_variant in member_variants:
            variant_source = "reanchored_unit" if member_variant is not member and (
                member_variant.get("register_origin") != member.get("register_origin")
            ) else "existing_unit"
            match_detail = _evaluate_main_unit_candidate_match(
                winner_member,
                member_variant,
                main_members,
                trace_rows_by_strand,
                phase=int(phase),
            )
            if (
                bool(require_canonical_cross_strand)
                and bool(match_detail.get("accepted"))
                and bool(match_detail.get("cross_strand"))
                and not bool(match_detail.get("canonical_compatible"))
            ):
                match_detail = dict(match_detail)
                match_detail["accepted"] = False
                match_detail["reject_reason"] = "noncanonical_partner_rerouted_to_secondary"
            _emit_main_partner_trace(
                debug_rows,
                debug_context,
                attempt_stage=attempt_stage,
                candidate_source=variant_source,
                candidate_category=str(candidate.get("category", member_variant.get("category", ""))),
                candidate_strand=member_variant.get("strand"),
                candidate_register_origin_tested=member_variant.get("register_origin"),
                observed_shift_nt=match_detail.get("shift_nt"),
                normalized_shift_nt=_normalized_public_partner_shift_nt(match_detail),
                duplex_orientation_ok=match_detail.get("duplex_orientation_ok"),
                canonical_compatible=match_detail.get("canonical_compatible"),
                candidate_tier=match_detail.get("candidate_tier"),
                shared_cycles=match_detail.get("shared_cycles"),
                bridge_support_ratio=match_detail.get("support_ratio"),
                exact_support_count=match_detail.get("candidate_exact_positions"),
                supported_position_count=match_detail.get("candidate_supported_positions"),
                max_unsupported_run=match_detail.get("max_zero_run"),
                candidate_peak_score=member_variant.get("peak_score"),
                candidate_non_grey_row_count=member_variant.get("non_grey_row_count"),
                first_reject_reason=match_detail.get("reject_reason"),
                accept_route=("candidate_match" if bool(match_detail.get("accepted")) else np.nan),
                final_route=("accepted_match" if bool(match_detail.get("accepted")) else "rejected"),
            )
            if not bool(match_detail.get("accepted")):
                continue
            ranking = _partner_match_ranking(
                match_detail,
                member_variant.get("peak_score", 0.0),
            )
            if best_detail is None or ranking < _partner_match_ranking(
                best_detail,
                None if best_member is None else best_member.get("peak_score", 0.0),
            ):
                best_member = dict(member_variant)
                best_detail = dict(match_detail)
    return best_member, best_detail


def _absorb_secondary_candidates_into_main_unit(
    winner_member: dict | None,
    candidate_units,
    trace_rows_by_strand: dict,
    *,
    phase: int,
    main_members,
    debug_context: dict | None = None,
    debug_rows=None,
) -> tuple[list[dict], set, set]:
    members = [dict(member) for member in (main_members or []) if member]
    consumed_row_keys = set()
    claimed_partner_row_keys = set()
    for member in members:
        consumed_row_keys.update(_unit_row_keys(member))

    expanded = True
    while expanded:
        expanded = False
        best_member = None
        best_detail = None
        best_candidate_keys = None

        for candidate in candidate_units or []:
            candidate_keys = _unit_row_keys(candidate)
            if not candidate_keys or candidate_keys & consumed_row_keys:
                continue
            member, detail = _best_main_unit_match_for_candidate(
                candidate,
                winner_member,
                members,
                trace_rows_by_strand,
                phase=int(phase),
                require_canonical_cross_strand=True,
                debug_context=debug_context,
                debug_rows=debug_rows,
                attempt_stage="secondary_absorption",
            )
            if member is None or detail is None:
                continue
            ranking = _partner_match_ranking(
                detail,
                member.get("peak_score", 0.0),
            )
            if best_detail is None or ranking < _partner_match_ranking(
                best_detail,
                None if best_member is None else best_member.get("peak_score", 0.0),
            ):
                best_member = dict(member)
                best_detail = dict(detail)
                best_candidate_keys = set(candidate_keys)

        if best_member is None or best_detail is None or best_candidate_keys is None:
            continue

        unit_role = "main_partner" if bool(best_detail.get("cross_strand")) else "main_extension"
        member_copy = _refine_main_unit_member(
            dict(best_member),
            phase=int(phase),
            unit_role=unit_role,
            category=unit_role,
        )
        if member_copy is None:
            member_copy = dict(best_member)
            member_copy["unit_role"] = unit_role
            member_copy["category"] = unit_role
        raw_shift_nt = int(best_detail.get("shift_nt", 0))
        member_copy["raw_shift_nt"] = raw_shift_nt
        member_copy["shift_nt"] = (
            _normalized_public_partner_shift_nt(best_detail)
            if str(unit_role) == "main_partner"
            else raw_shift_nt
        )
        if str(member_copy.get("unit_role") or "") != str(unit_role):
            _emit_main_partner_trace(
                debug_rows,
                debug_context,
                attempt_stage="secondary_absorption",
                candidate_source="existing_unit",
                candidate_category=str(best_member.get("category", "")),
                candidate_strand=best_member.get("strand"),
                candidate_register_origin_tested=best_member.get("register_origin"),
                observed_shift_nt=best_detail.get("shift_nt"),
                normalized_shift_nt=_normalized_public_partner_shift_nt(best_detail),
                duplex_orientation_ok=True,
                canonical_compatible=best_detail.get("canonical_compatible"),
                candidate_tier=best_detail.get("candidate_tier"),
                shared_cycles=best_detail.get("shared_cycles"),
                bridge_support_ratio=best_detail.get("support_ratio"),
                exact_support_count=best_detail.get("candidate_exact_positions"),
                supported_position_count=best_detail.get("candidate_supported_positions"),
                max_unsupported_run=best_detail.get("max_zero_run"),
                candidate_peak_score=best_member.get("peak_score"),
                candidate_non_grey_row_count=best_member.get("non_grey_row_count"),
                first_reject_reason="role_lost_after_refinement",
                final_route="rejected",
            )
        _emit_main_partner_trace(
            debug_rows,
            debug_context,
            attempt_stage="secondary_absorption",
            candidate_source="existing_unit",
            candidate_category=str(best_member.get("category", "")),
            candidate_strand=best_member.get("strand"),
            candidate_register_origin_tested=best_member.get("register_origin"),
            observed_shift_nt=best_detail.get("shift_nt"),
            normalized_shift_nt=_normalized_public_partner_shift_nt(best_detail),
            duplex_orientation_ok=True,
            canonical_compatible=best_detail.get("canonical_compatible"),
            candidate_tier=best_detail.get("candidate_tier"),
            shared_cycles=best_detail.get("shared_cycles"),
            bridge_support_ratio=best_detail.get("support_ratio"),
            exact_support_count=best_detail.get("candidate_exact_positions"),
            supported_position_count=best_detail.get("candidate_supported_positions"),
            max_unsupported_run=best_detail.get("max_zero_run"),
            candidate_peak_score=best_member.get("peak_score"),
            candidate_non_grey_row_count=best_member.get("non_grey_row_count"),
            accept_route=unit_role,
            final_route=unit_role,
        )
        members.append(member_copy)
        consumed_row_keys.update(best_candidate_keys)
        if str(unit_role) == "main_partner":
            claimed_partner_row_keys.update(best_candidate_keys)
        expanded = True

    return members, consumed_row_keys, claimed_partner_row_keys


def _build_candidate_register_positions(
    register_origin: int | None,
    phase: int,
    strand_code,
    *,
    cycles: int = WINDOW_MULTIPLIER,
) -> list[int]:
    if register_origin is None:
        return []
    origin = int(register_origin)
    phase_local = int(phase)
    count = int(cycles)
    if count <= 0 or phase_local <= 0:
        return []
    if _is_forward_trace_strand(strand_code):
        return [origin + cycle * phase_local for cycle in range(count)]
    return [origin - cycle * phase_local for cycle in range(count)]


def _count_shifted_register_matches(
    positions_a,
    positions_b,
    *,
    shift_nt: int = 0,
) -> int:
    pos_a = [int(value) for value in positions_a or []]
    pos_b = {int(value) for value in positions_b or []}
    if not pos_a or not pos_b:
        return 0
    shift_local = int(shift_nt)
    return sum(1 for value in pos_a if int(value) + shift_local in pos_b)


def _alternative_groups_share_biogenesis_unit(
    group_a: dict,
    group_b: dict,
    *,
    phase: int,
) -> bool:
    return _describe_biogenesis_match(group_a, group_b, phase=int(phase)) is not None


def _merge_relaxed_candidate_groups_into_units(
    candidate_groups,
    *,
    phase: int,
) -> list[dict]:
    groups = [dict(group) for group in candidate_groups or [] if group]
    if not groups:
        return []

    parent = list(range(len(groups)))
    pair_matches: dict[tuple[int, int], dict] = {}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            match_detail = _describe_biogenesis_match(groups[left], groups[right], phase=int(phase))
            if match_detail is None:
                continue
            pair_matches[(left, right)] = dict(match_detail)
            union(left, right)

    clustered = {}
    for index, group in enumerate(groups):
        clustered.setdefault(find(index), []).append(group)

    merged_units = []
    for root_index, members in clustered.items():
        members_local = sorted(
            members,
            key=lambda item: (
                -float(item.get("peak_score", 0.0) or 0.0),
                0 if bool(item.get("overlaps_winner_window")) else 1,
                0 if str(item.get("category", "")).strip() == "overlapping_alternative" else 1,
            ),
        )
        representative = dict(members_local[0])
        member_indexes = [idx for idx in range(len(groups)) if find(idx) == root_index]
        cross_strand_matches = []
        for left_pos in range(len(member_indexes)):
            for right_pos in range(left_pos + 1, len(member_indexes)):
                left_idx = member_indexes[left_pos]
                right_idx = member_indexes[right_pos]
                pair_key = (min(left_idx, right_idx), max(left_idx, right_idx))
                match_detail = pair_matches.get(pair_key)
                if match_detail and bool(match_detail.get("cross_strand")):
                    cross_strand_matches.append(dict(match_detail))
        best_cross_match = None
        if cross_strand_matches:
            best_cross_match = max(
                cross_strand_matches,
                key=lambda item: (
                    int(item.get("shared_cycles", 0)),
                    -abs(int(item.get("shift_nt", 0))),
                    int(item.get("shift_nt", 0)),
                ),
            )
        overlapping_member_present = any(
            bool(member.get("overlaps_winner_window"))
            or str(member.get("category", "")).strip() == "overlapping_alternative"
            for member in members_local
        )
        unit_category = "overlapping_alternative" if overlapping_member_present else "other_local_peak"
        best_shift_member = None
        shift_members = [member for member in members_local if member.get("shift_nt") is not None]
        if shift_members:
            best_shift_member = max(
                shift_members,
                key=lambda item: float(item.get("peak_score", 0.0) or 0.0),
            )
        paired_unit = bool(best_cross_match is not None)
        merged_units.append(
            {
                **representative,
                "category": unit_category,
                "members": [dict(member) for member in members_local],
                "member_count": int(len(members_local)),
                "member_strands": sorted(
                    {
                        _normalize_trace_strand_code(member.get("strand"))
                        for member in members_local
                    }
                ),
                "raw_categories": sorted(
                    {
                        str(member.get("category", "")).strip()
                        for member in members_local
                    }
                ),
                "peak_row": representative.get("peak_row"),
                "peak_score": float(representative.get("peak_score", 0.0) or 0.0),
                "register_origin": representative.get("register_origin"),
                "shift_nt": (
                    None
                    if best_shift_member is None
                    else int(best_shift_member.get("shift_nt"))
                ),
                "paired_unit": bool(paired_unit),
                "has_opposite_strand_match": bool(best_cross_match is not None),
                "best_cross_strand_shared_cycles": (
                    0 if best_cross_match is None else int(best_cross_match.get("shared_cycles", 0))
                ),
                "best_cross_strand_shift_nt": (
                    None if best_cross_match is None else int(best_cross_match.get("shift_nt", 0))
                ),
                "exact_row_count": int(sum(int(member.get("exact_row_count", 0) or 0) for member in members_local)),
                "offset_row_count": int(sum(int(member.get("offset_row_count", 0) or 0) for member in members_local)),
                "other_row_count": int(sum(int(member.get("other_row_count", 0) or 0) for member in members_local)),
                "non_grey_row_count": int(sum(int(member.get("non_grey_row_count", 0) or 0) for member in members_local)),
                "min_start": min(int(member.get("min_start")) for member in members_local),
                "max_end": max(int(member.get("max_end")) for member in members_local),
                "overlaps_winner_window": overlapping_member_present,
            }
        )

    merged_units.sort(
        key=lambda item: (
            -float(item.get("peak_score", 0.0) or 0.0),
            0 if str(item.get("category", "")).strip() == "overlapping_alternative" else 1,
            int(item.get("shift_nt") or 0),
        )
    )
    return merged_units


def _group_overlapping_alternative_candidates(
    main_group_rows,
    winner_row: dict | None,
    *,
    phase: int,
    strand_code: str,
    score_cutoff: float,
) -> list[dict]:
    if winner_row is None:
        return []

    winner_register_origin = _build_relaxed_trace_register_origin(winner_row, phase, strand_code)
    if winner_register_origin is None:
        return []

    grouped = {}
    for row in main_group_rows or []:
        if _same_trace_window(row, winner_row):
            continue
        if not _windows_overlap(
            row.get("window_start"),
            row.get("window_end"),
            winner_row.get("window_start"),
            winner_row.get("window_end"),
        ):
            continue

        try:
            candidate_score = float(row.get("score", 0.0) or 0.0)
            anchor_position = int(row.get("anchor_position"))
        except Exception:
            continue
        if candidate_score < float(score_cutoff):
            continue

        relation_to_winner, _ = _classify_relaxed_trace_relation(
            anchor_position,
            winner_register_origin,
            int(phase),
        )
        if relation_to_winner != "other":
            continue

        candidate_register_origin = _build_relaxed_trace_register_origin(row, int(phase), strand_code)
        shift_nt = _compute_phase_shift_nt(winner_register_origin, candidate_register_origin, int(phase))
        if shift_nt is None:
            continue

        shift_group = grouped.setdefault(
            int(shift_nt),
            {
                "rows": [],
                "min_start": int(row.get("window_start")),
                "max_end": int(row.get("window_end")),
            },
        )
        shift_group["rows"].append(dict(row))
        shift_group["min_start"] = min(shift_group["min_start"], int(row.get("window_start")))
        shift_group["max_end"] = max(shift_group["max_end"], int(row.get("window_end")))

    summaries = []
    for shift_nt, group in grouped.items():
        summary = _build_relaxed_group_summary(
            group,
            strand_code=strand_code,
            category="overlapping_alternative",
            phase=int(phase),
            winner_register_origin=winner_register_origin,
        )
        if summary is None:
            continue
        summary["shift_nt"] = int(shift_nt)
        summary["overlaps_winner_window"] = True
        summaries.append(summary)

    summaries.sort(key=lambda item: (-float(item.get("peak_score", 0.0) or 0.0), int(item.get("shift_nt") or 0)))
    return summaries


def _unit_members(unit: dict) -> list[dict]:
    members = list(unit.get("members") or [])
    if members:
        return [dict(member) for member in members]
    return [dict(unit)]


def _synthetic_main_unit_member(
    winner_row: dict | None,
    *,
    phase: int,
    strand_code: str,
) -> dict | None:
    if winner_row is None:
        return None
    group = {
        "rows": [dict(winner_row)],
        "min_start": int(winner_row.get("window_start")),
        "max_end": int(winner_row.get("window_end")),
    }
    summary = _build_relaxed_group_summary(
        group,
        strand_code=strand_code,
        category="main_hpsp",
        phase=int(phase),
        winner_register_origin=_build_relaxed_trace_register_origin(winner_row, int(phase), strand_code),
        winner_window=(int(winner_row.get("window_start")), int(winner_row.get("window_end"))),
    )
    if summary is None:
        return None
    summary["category"] = "main_hpsp"
    summary["unit_role"] = "main_hpsp"
    summary["shift_nt"] = 0
    return summary


def _select_main_biogenesis_partner(
    winner_member: dict | None,
    candidate_units,
    trace_rows_by_strand: dict,
    *,
    phase: int,
    debug_context: dict | None = None,
    debug_rows=None,
) -> tuple[int | None, dict | None, dict | None]:
    if winner_member is None:
        return None, None, None

    best_index = None
    best_member = None
    best_detail = None
    for index, unit in enumerate(candidate_units or []):
        member, match_detail = _best_main_unit_match_for_candidate(
            unit,
            winner_member,
            [winner_member],
            trace_rows_by_strand,
            phase=int(phase),
            require_canonical_cross_strand=True,
            debug_context=debug_context,
            debug_rows=debug_rows,
            attempt_stage="partner_selection",
        )
        if member is None or match_detail is None or not bool(match_detail.get("cross_strand")):
            continue
        ranking = _partner_match_ranking(
            match_detail,
            member.get("peak_score", 0.0),
        )
        if best_detail is None or ranking < _partner_match_ranking(
            best_detail,
            None if best_member is None else best_member.get("peak_score", 0.0),
        ):
            best_index = int(index)
            best_member = dict(member)
            best_detail = dict(match_detail)

    if best_member is None or best_detail is None:
        synthetic_member, synthetic_detail = _synthesize_projected_partner_candidate(
            winner_member,
            [winner_member],
            trace_rows_by_strand,
            phase=int(phase),
            require_canonical_cross_strand=True,
            debug_context=debug_context,
            debug_rows=debug_rows,
        )
        if synthetic_member is not None and synthetic_detail is not None:
            best_index = None
            best_member = dict(synthetic_member)
            best_detail = dict(synthetic_detail)
    if best_member is None or best_detail is None:
        if list(trace_rows_by_strand.get(_opposite_trace_strand(winner_member.get("strand")), []) or []):
            _emit_main_partner_trace(
                debug_rows,
                debug_context,
                attempt_stage="partner_selection",
                candidate_source="existing_unit",
                candidate_category="main_partner_candidate",
                candidate_strand=_opposite_trace_strand(winner_member.get("strand")),
                first_reject_reason="no_cross_strand_candidate_rows",
                final_route="rejected",
            )
    else:
        _emit_main_partner_trace(
            debug_rows,
            debug_context,
            attempt_stage="partner_selection",
            candidate_source=("synthetic_projected" if best_index is None else "existing_unit"),
            candidate_category=str(best_member.get("category", "main_partner_candidate")),
            candidate_strand=best_member.get("strand"),
            candidate_register_origin_tested=best_member.get("register_origin"),
            observed_shift_nt=best_detail.get("shift_nt"),
            normalized_shift_nt=_normalized_public_partner_shift_nt(best_detail),
            duplex_orientation_ok=True,
            canonical_compatible=best_detail.get("canonical_compatible"),
            candidate_tier=best_detail.get("candidate_tier"),
            shared_cycles=best_detail.get("shared_cycles"),
            bridge_support_ratio=best_detail.get("support_ratio"),
            exact_support_count=best_detail.get("candidate_exact_positions"),
            supported_position_count=best_detail.get("candidate_supported_positions"),
            max_unsupported_run=best_detail.get("max_zero_run"),
            candidate_peak_score=best_member.get("peak_score"),
            candidate_non_grey_row_count=best_member.get("non_grey_row_count"),
            accept_route="main_partner",
            final_route="main_partner",
        )
    return best_index, best_member, best_detail


def _expand_main_biogenesis_members(
    winner_member: dict | None,
    trace_candidates,
    trace_rows_by_strand: dict,
    *,
    phase: int,
    initial_members=None,
    consumed_indexes=None,
    debug_context: dict | None = None,
    debug_rows=None,
) -> tuple[list[dict], set[int]]:
    if winner_member is None:
        return [], set()

    members = [dict(member) for member in (initial_members or []) if member]
    if not members:
        members = [dict(winner_member)]
    used_indexes = {int(index) for index in (consumed_indexes or set())}
    used_row_keys = set()
    for member in members:
        used_row_keys.update(_unit_row_keys(member))

    expanded = True
    while expanded:
        expanded = False
        best_index = None
        best_detail = None
        best_candidate = None
        for index, unit in enumerate(trace_candidates or []):
            if int(index) in used_indexes:
                continue
            candidate_keys = _unit_row_keys(unit)
            if candidate_keys & used_row_keys:
                continue
            member, match_detail = _best_main_unit_match_for_candidate(
                unit,
                winner_member,
                members,
                trace_rows_by_strand,
                phase=int(phase),
                require_canonical_cross_strand=True,
                debug_context=debug_context,
                debug_rows=debug_rows,
                attempt_stage="main_extension_expansion",
            )
            if member is None or match_detail is None:
                continue
            ranking = _partner_match_ranking(
                match_detail,
                member.get("peak_score", 0.0),
            )
            if best_detail is None or ranking < _partner_match_ranking(
                best_detail,
                None if best_candidate is None else best_candidate.get("peak_score", 0.0),
            ):
                best_index = int(index)
                best_detail = dict(match_detail)
                best_candidate = dict(member)

        if best_index is None or best_detail is None or best_candidate is None:
            continue

        unit_role = (
            "main_partner"
            if bool(best_detail.get("cross_strand"))
            else "main_extension"
        )
        member_copy = _refine_main_unit_member(
            dict(best_candidate),
            phase=int(phase),
            unit_role=unit_role,
            category=unit_role,
        )
        if member_copy is None:
            member_copy = dict(best_candidate)
            member_copy["unit_role"] = unit_role
        raw_shift_nt = int(best_detail.get("shift_nt", 0))
        member_copy["raw_shift_nt"] = raw_shift_nt
        member_copy["shift_nt"] = (
            _normalized_public_partner_shift_nt(best_detail)
            if str(unit_role) == "main_partner"
            else raw_shift_nt
        )
        _emit_main_partner_trace(
            debug_rows,
            debug_context,
            attempt_stage="main_extension_expansion",
            candidate_source="existing_unit",
            candidate_category=str(best_candidate.get("category", "")),
            candidate_strand=best_candidate.get("strand"),
            candidate_register_origin_tested=best_candidate.get("register_origin"),
            observed_shift_nt=best_detail.get("shift_nt"),
            normalized_shift_nt=_normalized_public_partner_shift_nt(best_detail),
            duplex_orientation_ok=True,
            canonical_compatible=best_detail.get("canonical_compatible"),
            candidate_tier=best_detail.get("candidate_tier"),
            shared_cycles=best_detail.get("shared_cycles"),
            bridge_support_ratio=best_detail.get("support_ratio"),
            exact_support_count=best_detail.get("candidate_exact_positions"),
            supported_position_count=best_detail.get("candidate_supported_positions"),
            max_unsupported_run=best_detail.get("max_zero_run"),
            candidate_peak_score=best_candidate.get("peak_score"),
            candidate_non_grey_row_count=best_candidate.get("non_grey_row_count"),
            accept_route=unit_role,
            final_route=unit_role,
        )
        members.append(member_copy)
        used_indexes.add(int(best_index))
        used_row_keys.update(_unit_row_keys(member_copy))
        expanded = True

    return members, used_indexes


def _main_biogenesis_unit(
    winner_row: dict | None,
    winner_strand: str | None,
    candidate_units,
    trace_candidates,
    trace: dict,
    *,
    phase: int,
    debug_context: dict | None = None,
    debug_rows=None,
) -> tuple[dict | None, list[dict]]:
    trace_candidates_local = [dict(unit) for unit in (trace_candidates or [])]
    winner_member = None
    winner_key = _trace_row_key(winner_row)
    seed_index = None
    for index, unit in enumerate(trace_candidates_local):
        if winner_key is None or winner_key not in _unit_row_keys(unit):
            continue
        winner_member = _refine_main_unit_member(
            dict(unit),
            phase=int(phase),
            unit_role="main_hpsp",
            category="main_hpsp",
        )
        if winner_member is None:
            winner_member = dict(unit)
            winner_member["category"] = "main_hpsp"
            winner_member["unit_role"] = "main_hpsp"
            winner_member["shift_nt"] = 0
        seed_index = int(index)
        break

    if winner_member is None:
        winner_member = _synthetic_main_unit_member(
            winner_row,
            phase=int(phase),
            strand_code=("w" if winner_strand is None else winner_strand),
        )
    if winner_member is None:
        return None, list(candidate_units or [])

    trace_rows_by_strand = {
        "w": list((trace or {}).get("w") or []),
        "c": list((trace or {}).get("c") or []),
    }
    partner_index, partner_member, partner_detail = _select_main_biogenesis_partner(
        winner_member,
        trace_candidates_local,
        trace_rows_by_strand,
        phase=int(phase),
        debug_context=debug_context,
        debug_rows=debug_rows,
    )

    main_members = [dict(winner_member)]
    if partner_member is not None:
        partner_unit = _refine_main_unit_member(
            dict(partner_member),
            phase=int(phase),
            unit_role="main_partner",
            category="main_partner",
        )
        if partner_unit is None:
            partner_unit = dict(partner_member)
            partner_unit["unit_role"] = "main_partner"
        partner_unit["raw_shift_nt"] = int(partner_detail.get("shift_nt", 0))
        partner_unit["shift_nt"] = _normalized_public_partner_shift_nt(partner_detail)
        main_members.append(partner_unit)

    consumed_indexes = set()
    if seed_index is not None:
        consumed_indexes.add(int(seed_index))
    if partner_index is not None:
        consumed_indexes.add(int(partner_index))

    main_members, consumed_indexes = _expand_main_biogenesis_members(
        winner_member,
        trace_candidates_local,
        trace_rows_by_strand,
        phase=int(phase),
        initial_members=main_members,
        consumed_indexes=consumed_indexes,
        debug_context=debug_context,
        debug_rows=debug_rows,
    )

    main_members, claimed_secondary_row_keys, claimed_partner_row_keys = _absorb_secondary_candidates_into_main_unit(
        winner_member,
        candidate_units,
        trace_rows_by_strand,
        phase=int(phase),
        main_members=main_members,
        debug_context=debug_context,
        debug_rows=debug_rows,
    )

    remaining_units = _filter_secondary_units_against_main_unit(
        candidate_units,
        {"members": main_members},
        claimed_row_keys=claimed_secondary_row_keys,
    )
    if _main_partner_trace_enabled() and claimed_partner_row_keys:
        leaked_units = [
            dict(unit)
            for unit in remaining_units
            if _unit_row_keys(unit) & claimed_partner_row_keys
        ]
        if leaked_units:
            leaked_summary = [
                {
                    "category": str(unit.get("category", "")),
                    "strand": str(unit.get("strand", "")),
                    "register_origin": unit.get("register_origin"),
                    "peak_score": unit.get("peak_score"),
                }
                for unit in leaked_units
            ]
            raise AssertionError(
                f"Claimed main-partner candidates leaked into secondary routing: {leaked_summary}"
            )
    for unit in remaining_units:
        if (
            bool(unit.get("has_opposite_strand_match"))
            and bool(unit.get("overlaps_winner_window"))
            and bool(_unit_row_keys(unit) & claimed_partner_row_keys)
        ):
            _emit_main_partner_trace(
                debug_rows,
                debug_context,
                attempt_stage="secondary_routing",
                candidate_source="existing_unit",
                candidate_category=str(unit.get("category", "")),
                candidate_strand=unit.get("strand"),
                candidate_register_origin_tested=unit.get("register_origin"),
                observed_shift_nt=unit.get("best_cross_strand_shift_nt", unit.get("shift_nt")),
                normalized_shift_nt=np.nan,
                duplex_orientation_ok=True,
                shared_cycles=unit.get("best_cross_strand_shared_cycles"),
                bridge_support_ratio=np.nan,
                exact_support_count=unit.get("exact_row_count"),
                supported_position_count=unit.get("non_grey_row_count"),
                max_unsupported_run=np.nan,
                candidate_peak_score=unit.get("peak_score"),
                candidate_non_grey_row_count=unit.get("non_grey_row_count"),
                first_reject_reason="accepted_then_diverted_to_secondary",
                final_route=str(unit.get("category", "overlapping_alternative")),
            )

    partner_members = [
        member
        for member in main_members
        if str(member.get("unit_role") or "") == "main_partner"
    ]
    best_partner_detail = None
    if partner_members:
        partner_details = [
            _describe_main_unit_candidate_match(
                winner_member,
                member,
                [winner_member],
                trace_rows_by_strand,
                phase=int(phase),
            )
            for member in partner_members
        ]
        partner_details = [item for item in partner_details if item is not None]
        if partner_details:
            best_partner_detail = min(
                partner_details,
                key=lambda item: _partner_match_ranking(item, 0.0),
            )

    main_unit = {
        "category": "main_biogenesis_unit",
        "members": main_members,
        "peak_row": winner_row,
        "peak_score": float(winner_row.get("score", 0.0) or 0.0),
        "register_origin": winner_member.get("register_origin"),
        "shift_nt": 0,
        "member_count": int(len(main_members)),
        "member_strands": sorted(
            {
                _normalize_trace_strand_code(member.get("strand"))
                for member in main_members
            }
        ),
        "main_partner_present": bool(partner_members),
        "main_partner_raw_shift_nt": (
            None if best_partner_detail is None else int(best_partner_detail.get("shift_nt", 0))
        ),
        "main_partner_shift_nt": (
            _normalized_public_partner_shift_nt(best_partner_detail)
        ),
        "best_cross_strand_shared_cycles": (
            0 if best_partner_detail is None else int(best_partner_detail.get("shared_cycles", 0))
        ),
    }
    return main_unit, remaining_units


def _secondary_unit_is_promotable(
    unit: dict,
    *,
    main_peak_score: float | None,
    score_cutoff: float,
) -> bool:
    try:
        peak_score = float(unit.get("peak_score", 0.0) or 0.0)
    except Exception:
        peak_score = 0.0
    if peak_score < float(score_cutoff):
        return False

    if bool(unit.get("promote_as_noncanonical_secondary")):
        return True

    if bool(unit.get("has_opposite_strand_match")):
        return int(unit.get("best_cross_strand_shared_cycles", 0) or 0) >= int(ALTERNATIVE_MIN_SHARED_CYCLES)

    if main_peak_score is None or not np.isfinite(float(main_peak_score)):
        return False
    relative_cutoff = max(
        float(score_cutoff),
        float(main_peak_score) * float(PROMOTED_ALT_MIN_RELATIVE_SCORE),
    )
    if peak_score < relative_cutoff:
        return False
    if int(unit.get("non_grey_row_count", 0) or 0) < int(PROMOTED_ALT_MIN_NON_GREY_ROWS):
        return False
    if int(unit.get("exact_row_count", 0) or 0) < int(PROMOTED_ALT_MIN_EXACT_ROWS):
        return False
    return True


def _annotate_secondary_units_against_main_unit(
    candidate_units,
    main_unit: dict | None,
    trace_rows_by_strand: dict,
    *,
    phase: int,
) -> list[dict]:
    if not candidate_units:
        return []
    if not main_unit:
        return [dict(unit) for unit in candidate_units]

    main_members = [dict(member) for member in list((main_unit or {}).get("members") or []) if member]
    if not main_members:
        return [dict(unit) for unit in candidate_units]

    winner_member = None
    for member in main_members:
        if str(member.get("unit_role") or "") == "main_hpsp":
            winner_member = dict(member)
            break
    if winner_member is None:
        winner_member = dict(main_members[0])

    annotated_units = []
    for unit in candidate_units or []:
        unit_copy = dict(unit)
        member, detail = _best_main_unit_match_for_candidate(
            unit_copy,
            winner_member,
            main_members,
            trace_rows_by_strand,
            phase=int(phase),
            require_canonical_cross_strand=False,
            debug_context=None,
            debug_rows=None,
            attempt_stage="secondary_promotion_probe",
        )
        if member is None or detail is None or not bool(detail.get("cross_strand")):
            annotated_units.append(unit_copy)
            continue

        unit_copy["main_cross_strand_candidate"] = True
        unit_copy["main_cross_strand_canonical"] = bool(detail.get("canonical_compatible"))
        unit_copy["main_cross_strand_shift_nt"] = int(detail.get("shift_nt", 0))
        unit_copy["main_cross_strand_normalized_shift_nt"] = _normalized_public_partner_shift_nt(detail)
        unit_copy["main_cross_strand_support_ratio"] = float(detail.get("support_ratio", 0.0) or 0.0)
        unit_copy["main_cross_strand_shared_cycles"] = int(detail.get("shared_cycles", 0) or 0)
        unit_copy["main_cross_strand_exact_support"] = int(detail.get("candidate_exact_positions", 0) or 0)
        unit_copy["main_cross_strand_supported_positions"] = int(detail.get("candidate_supported_positions", 0) or 0)

        if bool(detail.get("accepted")) and not bool(detail.get("canonical_compatible")):
            unit_copy["promote_as_noncanonical_secondary"] = True
            unit_copy["paired_unit"] = True
            unit_copy["has_opposite_strand_match"] = True
            unit_copy["best_cross_strand_shared_cycles"] = max(
                int(unit_copy.get("best_cross_strand_shared_cycles", 0) or 0),
                int(detail.get("shared_cycles", 0) or 0),
            )
            unit_copy["best_cross_strand_shift_nt"] = int(detail.get("shift_nt", 0))
            unit_copy["shift_nt"] = int(detail.get("shift_nt", unit_copy.get("shift_nt", 0)) or 0)

        annotated_units.append(unit_copy)

    return annotated_units


def _annotate_promoted_secondary_units(
    candidate_units,
    *,
    main_peak_score: float | None,
    score_cutoff: float,
) -> tuple[list[dict], list[dict]]:
    promoted_units = []
    unpromoted_units = []
    for unit in candidate_units or []:
        unit_copy = dict(unit)
        promotable = _secondary_unit_is_promotable(
            unit_copy,
            main_peak_score=main_peak_score,
            score_cutoff=score_cutoff,
        )
        unit_copy["promoted_secondary"] = bool(promotable)
        if promotable:
            promoted_units.append(unit_copy)
        else:
            unpromoted_units.append(unit_copy)

    promoted_units.sort(
        key=lambda item: (
            -float(item.get("peak_score", 0.0) or 0.0),
            0 if str(item.get("category", "")).strip() == "overlapping_alternative" else 1,
            abs(int(item.get("shift_nt") or 0)),
        )
    )
    return promoted_units, unpromoted_units


def summarize_relaxed_trace_subregions(
    trace: dict,
    *,
    score_cutoff: float | None = None,
    phase: int | None = None,
    debug_context: dict | None = None,
) -> dict:
    """
    Summarize additional relaxed Howell peak regions across the whole locus.

    These are intentionally separate from the exact-only HPSP ambiguity metrics:
    we group supra-threshold relaxed trace windows by genomic overlap on each
    strand, identify the region containing the relaxed trace HPSP as the main
    region, and count any other supra-threshold regions as alternative
    phased-like subregions.
    """
    cutoff = _min_howell_score_value() if score_cutoff is None else float(score_cutoff)
    phase_local = int(phase) if phase is not None else _phase_value()
    debug_rows = [] if debug_context is not None else None
    raw_candidate_groups = []
    global_peak_strand = None
    global_peak_row = None
    global_peak_score = float("-inf")

    for strand_code in ("w", "c"):
        strand_peak = _find_relaxed_trace_peak_row(trace.get(strand_code, []) or [])
        if strand_peak is None:
            continue
        try:
            strand_peak_score = float(strand_peak.get("score", 0.0) or 0.0)
        except Exception:
            strand_peak_score = 0.0
        if strand_peak_score >= global_peak_score:
            global_peak_score = strand_peak_score
            global_peak_strand = strand_code
            global_peak_row = strand_peak

    for strand_code in ("w", "c"):
        strand_rows = list(trace.get(strand_code, []) or [])
        if not strand_rows:
            continue

        groups = _group_trace_rows_by_overlap(strand_rows, score_cutoff=cutoff)
        if not groups:
            continue

        main_group_index = None
        for idx, group in enumerate(groups):
            if strand_code == global_peak_strand and any(_same_trace_window(row, global_peak_row) for row in group["rows"]):
                main_group_index = idx
                break

        if strand_code == global_peak_strand and main_group_index is not None:
            raw_candidate_groups.extend(
                _group_overlapping_alternative_candidates(
                    groups[main_group_index]["rows"],
                    global_peak_row,
                    phase=phase_local,
                    strand_code=strand_code,
                    score_cutoff=cutoff,
                )
            )

        for idx, group in enumerate(groups):
            if main_group_index is not None and idx == main_group_index:
                continue
            group_summary = _build_relaxed_group_summary(
                group,
                strand_code=strand_code,
                category="other_local_peak",
                phase=phase_local,
                winner_register_origin=(
                    _build_relaxed_trace_register_origin(global_peak_row, phase_local, global_peak_strand)
                    if global_peak_row is not None and global_peak_strand is not None
                    else None
                ),
                winner_window=(
                    (int(global_peak_row.get("window_start")), int(global_peak_row.get("window_end")))
                    if global_peak_row is not None
                    else None
                ),
            )
            if group_summary is None:
                continue
            raw_candidate_groups.append(group_summary)

    merged_candidate_groups = _merge_relaxed_candidate_groups_into_units(
        raw_candidate_groups,
        phase=phase_local,
    )
    debug_context_local = None if debug_context is None else dict(debug_context)
    if debug_context_local is not None:
        debug_context_local["winner_strand"] = global_peak_strand
        debug_context_local["winner_score"] = (
            np.nan if not np.isfinite(global_peak_score) else float(global_peak_score)
        )
        debug_context_local["winner_register_origin"] = (
            None
            if global_peak_row is None or global_peak_strand is None
            else _build_relaxed_trace_register_origin(global_peak_row, phase_local, global_peak_strand)
        )
    trace_segment_candidates = _build_all_trace_segment_candidates(
        trace,
        global_peak_row,
        global_peak_strand,
        phase=phase_local,
    )
    main_unit, secondary_units = _main_biogenesis_unit(
        global_peak_row,
        global_peak_strand,
        merged_candidate_groups,
        trace_segment_candidates,
        trace,
        phase=phase_local,
        debug_context=debug_context_local,
        debug_rows=debug_rows,
    )
    secondary_units = _annotate_secondary_units_against_main_unit(
        secondary_units,
        main_unit,
        trace_rows_by_strand={
            "w": list((trace or {}).get("w") or []),
            "c": list((trace or {}).get("c") or []),
        },
        phase=phase_local,
    )
    promoted_secondary_units, unpromoted_secondary_units = _annotate_promoted_secondary_units(
        secondary_units,
        main_peak_score=global_peak_score if np.isfinite(global_peak_score) else None,
        score_cutoff=cutoff,
    )

    additional_peak_groups = [
        group
        for group in secondary_units
        if str(group.get("category", "")).strip() == "other_local_peak"
    ]
    promoted_additional_peak_groups = [
        group
        for group in promoted_secondary_units
        if str(group.get("category", "")).strip() == "other_local_peak"
    ]
    overlapping_alt_groups = [
        group
        for group in promoted_secondary_units
        if str(group.get("category", "")).strip() == "overlapping_alternative"
    ]
    additional_region_scores = [
        float(group.get("peak_score", 0.0) or 0.0)
        for group in additional_peak_groups
    ]

    best_overlap_group = None
    if overlapping_alt_groups:
        best_overlap_group = max(
            overlapping_alt_groups,
            key=lambda item: float(item.get("peak_score", 0.0) or 0.0),
        )
    best_overlap_shift_nt = (
        None if best_overlap_group is None else best_overlap_group.get("shift_nt")
    )

    return {
        "Howell_additional_peak_count": int(len(additional_region_scores)),
        "Howell_additional_peak_best_score": (
            np.nan if not additional_region_scores else float(max(additional_region_scores))
        ),
        "Howell_overlapping_alt_count": int(len(overlapping_alt_groups)),
        "Howell_overlapping_alt_best_score": (
            np.nan if best_overlap_group is None else float(best_overlap_group.get("peak_score", 0.0) or 0.0)
        ),
        "Howell_overlapping_alt_best_shift_nt": (
            np.nan if best_overlap_shift_nt is None else float(best_overlap_shift_nt)
        ),
        "main_biogenesis_unit": main_unit,
        "additional_peak_groups": additional_peak_groups,
        "promoted_additional_peak_groups": promoted_additional_peak_groups,
        "overlapping_alt_groups": overlapping_alt_groups,
        "promoted_secondary_units": promoted_secondary_units,
        "unpromoted_secondary_units": unpromoted_secondary_units,
        "main_partner_debug_rows": ([] if debug_rows is None else debug_rows),
    }


def enumerate_relaxed_howell_trace(aclust: pd.DataFrame, phase: int | None = None):
    ph = int(phase) if phase is not None else _phase_value()
    win_size = WINDOW_MULTIPLIER * int(ph)
    seq_start = int(aclust["pos"].min())
    seq_end = int(aclust["pos"].max())
    w_mask, c_mask = _strand_masks(aclust)

    w_trace = []
    if w_mask.any():
        w_pos_abun = _build_pos_abun_exact_phase(aclust.loc[w_mask], seq_start, seq_end, int(ph))
        if w_pos_abun:
            w_trace = _enumerate_relaxed_trace_for_strand(
                w_pos_abun,
                int(ph),
                int(win_size),
                forward=True,
            )

    c_trace = []
    if c_mask.any():
        c_pos_abun = _build_pos_abun_exact_phase(aclust.loc[c_mask], seq_start, seq_end, int(ph))
        if c_pos_abun:
            c_trace = _enumerate_relaxed_trace_for_strand(
                c_pos_abun,
                int(ph),
                int(win_size),
                forward=False,
            )

    return {"w": w_trace, "c": c_trace}


def _best_sliding_window_score_generic_strict(pos_abun, phase, win_size, seq_start=None, seq_end=None, forward=True):
    positions = sorted(pos_abun.keys())
    if not positions:
        return 0.0, None, None

    lower_bound = seq_start if seq_start is not None else positions[0]
    upper_bound = (seq_end - win_size + 1) if seq_end is not None else positions[-1] - win_size + 1
    if upper_bound < lower_bound:
        lower_bound = positions[0]
        upper_bound = lower_bound

    best_score = -float("inf")
    best_window = (None, None)

    for win_start in range(lower_bound, upper_bound + 1):
        win_end = win_start + win_size - 1
        window_positions = [p for p in positions if win_start <= p <= win_end]
        if not window_positions:
            score = 0.0
        else:
            num_cycles = max(0, (win_end - win_start + 1) // int(phase))
            if num_cycles < 4:
                score = 0.0
            else:
                best_reg_sum = 0.0
                best_reg_total = 0.0
                best_reg_filled = 0

                for reg in range(int(phase)):
                    in_sum, total, n_filled = _evaluate_register_strict_exact(
                        window_positions, pos_abun, win_start, win_end, int(phase), reg, forward=forward
                    )
                    if in_sum > best_reg_sum:
                        best_reg_sum = in_sum
                        best_reg_total = total
                        best_reg_filled = n_filled

                out_of_phase = max(0.0, best_reg_total - best_reg_sum)
                numerator = best_reg_sum
                denominator = 1.0 + out_of_phase
                if numerator <= 0.0 or not (best_reg_filled > 3):
                    score = 0.0
                else:
                    log_arg = 1.0 + 10.0 * (numerator / denominator)
                    if log_arg <= 0.0 or log_arg != log_arg:
                        score = 0.0
                    else:
                        scale = max(min(best_reg_filled, num_cycles) - 2, 0)
                        score = scale * (0.0 if log_arg <= 0 else np.log(log_arg))

        if score > best_score:
            best_score = score
            best_window = (win_start, win_end)

    return best_score if best_score != -float("inf") else 0.0, best_window[0], best_window[1]

def best_sliding_window_score_forward_strict(pos_abun, phase, win_size, seq_start=None, seq_end=None):
    return _best_sliding_window_score_generic_strict(
        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=True
    )

def best_sliding_window_score_reverse_strict(pos_abun, phase, win_size, seq_start=None, seq_end=None):
    return _best_sliding_window_score_generic_strict(
        pos_abun, phase, win_size, seq_start=seq_start, seq_end=seq_end, forward=False
    )

def compute_phasing_score_Howell_strict(aclust: pd.DataFrame):
    """
    Classic Howell phasing WITHOUT positional wobble.
    Uses ONLY len == phase reads.
    Returns: (w_score,(w_start,w_end), c_score,(c_start,c_end))
    """
    ph = _phase_value()
    win_size  = WINDOW_MULTIPLIER * int(ph)
    seq_start = int(aclust['pos'].min()); seq_end = int(aclust['pos'].max())
    w_mask, c_mask = _strand_masks(aclust)

    # Forward “w”
    if w_mask.any():
        w_pos_abun = _build_pos_abun_exact_phase(aclust.loc[w_mask], seq_start, seq_end, int(ph))
        w_score, w_s, w_e = (
            best_sliding_window_score_forward_strict(w_pos_abun, int(ph), win_size, seq_start, seq_end)
            if w_pos_abun else (0.0, None, None)
        )
    else:
        w_score, w_s, w_e = None, None, None

    # Reverse “c”
    if c_mask.any():
        c_pos_abun = _build_pos_abun_exact_phase(aclust.loc[c_mask], seq_start, seq_end, int(ph))
        c_score, c_s, c_e = (
            best_sliding_window_score_reverse_strict(c_pos_abun, int(ph), win_size, seq_start, seq_end)
            if c_pos_abun else (0.0, None, None)
        )
    else:
        c_score, c_s, c_e = None, None, None

    return (w_score, (w_s, w_e), c_score, (c_s, c_e))

def process_chromosome_features(chromosome_df: pd.DataFrame):
    """
    Build per-cluster feature rows (wobble + strict Howell) for one work batch.
    Returns rows matching FEATURE_COLS order.
    Expected columns:
      ['clusterID','chromosome','strand','pos','len','abun','identifier','tag_seq','alib']
    """
    ph = _phase_value()
    ensure_win_score_lookup_ready()

    df = chromosome_df[['clusterID','chromosome','strand','pos','len','abun','identifier','tag_seq','alib']].copy()
    df['pos']  = pd.to_numeric(df['pos'], errors='coerce')
    df['len']  = pd.to_numeric(df['len'], errors='coerce')
    df['abun'] = pd.to_numeric(df['abun'], errors='coerce').fillna(0)
    df = df.dropna(subset=['pos', 'len'])

    # Normalize score lookup keys once per chromosome chunk
    raw_lookup = st.WIN_SCORE_LOOKUP or {}
    score_lookup = {}
    for k, v in raw_lookup.items():
        try:
            nk = ids.normalize_cluster_id_for_lookup(str(k), phase=ph)
        except Exception:
            nk = str(k).strip()
        score_lookup[nk] = v
        # keep raw key as a fallback too
        score_lookup[str(k).strip()] = v

    rows = []
    debug_rows = [] if _main_partner_trace_enabled() else None
    for cID, aclust in df.groupby('clusterID', sort=False, observed=True):
        if aclust.empty:
            continue

        cid_raw = str(cID).strip()
        try:
            cid_norm = ids.normalize_cluster_id_for_lookup(cid_raw, phase=ph)
        except Exception:
            cid_norm = cid_raw

        # derive genomic span from this cluster's rows (works even if lookup misses)
        achr  = str(aclust['chromosome'].iloc[0])
        start = int(aclust['pos'].min()); end = int(aclust['pos'].max())

        # Universal identifier (prefer mergedClusterDict mapping; fallback to coords)
        uid = None
        try:
            uid = ids.getUniversalID(cid_raw)
        except Exception:
            uid = None
        if not uid:
            try:
                uid = ids.getUniversalID(cid_norm)
            except Exception:
                uid = None

        if not uid or ":" not in uid or ".." not in uid:
            uid = f"{achr}:{start}..{end}"  # hard fallback: always coordinate-style

        identifier = uid
        alib = str(aclust['alib'].iloc[0])

        # Normalize strand labels
        s_norm = aclust['strand'].astype(str).str.lower()
        w_mask = s_norm.isin(['w', '+', 'watson', '1', 'true'])
        c_mask = s_norm.isin(['c', '-', 'crick', '0', 'false'])

        # Strand bias
        total_w = int(w_mask.sum())
        total_c = int(c_mask.sum())
        denom = total_w + total_c
        strand_bias = (total_w / denom) if denom > 0 else 1.0

        # Abundance ratios
        sum_abun_len_phase = aclust.loc[aclust['len'] == int(ph), 'abun'].sum()
        sum_abun_other_len = aclust.loc[aclust['len'] != int(ph), 'abun'].sum()
        ratio_abund_len_phase = (sum_abun_len_phase / sum_abun_other_len) if sum_abun_other_len > 0 else 1.0

        # Totals and cluster length
        total_abund = float(aclust['abun'].sum())
        aclust_len = max(end - start, 0)

        # CLNC for phase-length reads (legacy):
        #CLNC = (phase-length abundance) / (cluster_length - phase)
        w_sum_abun_len_phase = aclust.loc[(aclust['len'] == int(ph)) & w_mask, 'abun'].sum()
        c_sum_abun_len_phase = aclust.loc[(aclust['len'] == int(ph)) & c_mask, 'abun'].sum()
        denom_len = max(aclust_len - int(ph), 0)
        clnc = ((w_sum_abun_len_phase + c_sum_abun_len_phase) / denom_len) if denom_len > 0 else 0.0
        log_CLNC = float(np.log10(clnc + 1.0))
        # Complexity
        #   distinct tag sequences / total abundance (all reads)
        distinct_tags = int(aclust['tag_seq'].nunique(dropna=True))
        complexity = (distinct_tags / total_abund) if total_abund > 0 else 0.0

        # Default scores
        aclust_phasis_score = 0.0
        aclust_fishers_combined = 1.0

        tup = score_lookup.get(cid_norm)
        if tup is None:
            # Sometimes scores might be keyed by the universal ID already; try that too.
            tup = score_lookup.get(uid)
        if tup is None:
            tup = score_lookup.get(cid_raw)

        if tup is not None:
            ps, cf = tup
            if ps is not None:
                try:
                    aclust_phasis_score = float(ps)
                except Exception:
                    pass
            if cf is not None:
                try:
                    aclust_fishers_combined = float(cf)
                except Exception:
                    pass

        # Howell (wobble-tolerant)
        (w_Howell, (w_s, w_e),
         c_Howell, (c_s, c_e),
         peak_howell_detail) = compute_phasing_score_Howell(aclust, return_detail=True)
        Peak_Howell = None if (w_Howell is None and c_Howell is None) else max([x for x in (w_Howell, c_Howell) if x is not None])
        relaxed_trace = enumerate_relaxed_howell_trace(aclust, phase=ph)
        browser_trace_summary = classify_browser_style_relaxed_trace(relaxed_trace, phase=ph)
        additional_peak_summary = summarize_relaxed_trace_subregions(
            relaxed_trace,
            score_cutoff=_min_howell_score_value(),
            phase=ph,
            debug_context=(
                None
                if debug_rows is None
                else {
                    "identifier": identifier,
                    "alib": alib,
                    "winner_score": float(Peak_Howell) if Peak_Howell is not None else np.nan,
                }
            ),
        )
        if debug_rows is not None:
            debug_rows.extend(list(additional_peak_summary.get("main_partner_debug_rows") or []))
        if peak_howell_detail:
            exact_support_score = peak_howell_detail.get("Howell_exact_support_score", np.nan)
            ambiguity_count = peak_howell_detail.get("Howell_ambiguity_count", np.nan)
            alt_register_count = peak_howell_detail.get("Howell_alt_register_count", np.nan)
            overlap_margin = peak_howell_detail.get("Howell_overlap_margin", np.nan)
            extension_window_count = peak_howell_detail.get("Howell_extension_window_count", np.nan)
            extension_span_nt = peak_howell_detail.get("Howell_extension_span_nt", np.nan)
            origin_window_count = peak_howell_detail.get("Howell_origin_window_count", np.nan)
            origin_frame_count = peak_howell_detail.get("Howell_origin_frame_count", np.nan)
            origin_margin = peak_howell_detail.get("Howell_origin_margin", np.nan)
            origin_class = peak_howell_detail.get("Howell_origin_class", np.nan)
        else:
            exact_support_score = np.nan
            ambiguity_count = np.nan
            alt_register_count = np.nan
            overlap_margin = np.nan
            extension_window_count = np.nan
            extension_span_nt = np.nan
            origin_window_count = np.nan
            origin_frame_count = np.nan
            origin_margin = np.nan
            origin_class = np.nan
        additional_peak_count = additional_peak_summary.get("Howell_additional_peak_count", np.nan)
        additional_peak_best_score = additional_peak_summary.get("Howell_additional_peak_best_score", np.nan)
        overlapping_alt_count = additional_peak_summary.get("Howell_overlapping_alt_count", 0)
        overlapping_alt_best_score = additional_peak_summary.get("Howell_overlapping_alt_best_score", np.nan)
        overlapping_alt_best_shift_nt = additional_peak_summary.get("Howell_overlapping_alt_best_shift_nt", np.nan)
        crowding_window_count = browser_trace_summary.get("Howell_crowding_window_count", 0)
        crowding_best_score = browser_trace_summary.get("Howell_crowding_best_score", np.nan)
        crowding_score_gap = browser_trace_summary.get("Howell_crowding_score_gap", np.nan)

        # Howell (classic strict)
        (w_Howell_strict, (w_s_strict, w_e_strict),
         c_Howell_strict, (c_s_strict, c_e_strict)) = compute_phasing_score_Howell_strict(aclust)
        Peak_Howell_strict = None if (w_Howell_strict is None and c_Howell_strict is None)                              else max([x for x in (w_Howell_strict, c_Howell_strict) if x is not None])

        rows.append([
            identifier, cid_raw, alib,
            float(complexity), float(strand_bias), float(log_CLNC),
            float(ratio_abund_len_phase), aclust_phasis_score, aclust_fishers_combined,
            float(total_abund),
            # wobble-tolerant
            w_Howell, w_s, w_e, c_Howell, c_s, c_e, Peak_Howell,
            exact_support_score, ambiguity_count, alt_register_count, overlap_margin,
            extension_window_count, extension_span_nt, origin_window_count, origin_frame_count, origin_margin, origin_class,
            additional_peak_count, additional_peak_best_score,
            overlapping_alt_count, overlapping_alt_best_score, overlapping_alt_best_shift_nt,
            crowding_window_count, crowding_best_score, crowding_score_gap,
            # classic (strict)
            w_Howell_strict, w_s_strict, w_e_strict, c_Howell_strict, c_s_strict, c_e_strict, Peak_Howell_strict
        ])

    return {
        "rows": rows,
        "debug_rows": ([] if debug_rows is None else debug_rows),
    }
