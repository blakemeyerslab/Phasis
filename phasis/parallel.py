import os, multiprocessing, gc, traceback, sys, re, tempfile
from tqdm import tqdm
import phasis.runtime as rt
from phasis.env import getenv


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


def _extract_first_positive_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        match = re.search(r"(\d+)", str(value))
        if not match:
            return default
        parsed = int(match.group(1))
        if parsed <= 0:
            return default
        return parsed
    except Exception:
        return default


def _scheduler_cpu_limit(env=None):
    env = os.environ if env is None else env
    for key in (
        "SLURM_CPUS_PER_TASK",
        "PBS_NP",
        "NSLOTS",
        "LSB_DJOB_NUMPROC",
        "SLURM_CPUS_ON_NODE",
    ):
        parsed = _extract_first_positive_int(env.get(key), None)
        if parsed is not None:
            return parsed
    return None


def _effective_visible_cpu_count(env=None):
    totalcores = int(multiprocessing.cpu_count())
    scheduler_limit = _scheduler_cpu_limit(env=env)
    if scheduler_limit is None:
        return totalcores
    return int(max(1, min(totalcores, scheduler_limit)))


def _effective_start_method(start_method=None):
    if start_method is None:
        start_method = getattr(rt, "mp_start_method", None) or getenv("Phasis_MP_START_METHOD")
    if start_method is None:
        start_method = "spawn" if sys.platform == "darwin" else "forkserver"
    return start_method


def _disabled_env_value(value) -> bool:
    return str(value or "").strip().lower() in {"0", "false", "no", "n", "off", "none", "disable", "disabled"}


def _default_pycache_root(env=None) -> str:
    env = os.environ if env is None else env
    for key in ("SLURM_TMPDIR", "TMPDIR", "TMP", "TEMP"):
        value = str(env.get(key) or "").strip()
        if value:
            return value
    return tempfile.gettempdir()


def _ensure_worker_pycache_prefix(env=None) -> str | None:
    """Route worker bytecode caches away from shared source/install trees.

    Forkserver/spawn workers start fresh Python interpreters and import Phasis
    before worker initializers run. On shared filesystems, a truncated or
    concurrently written ``.pyc`` can make those workers fail during import.
    Setting ``PYTHONPYCACHEPREFIX`` before pool creation makes child
    interpreters use a job-local cache prefix instead.
    """
    env = os.environ if env is None else env
    configured = getenv("Phasis_PYCACHE_PREFIX", env=env)
    if _disabled_env_value(configured):
        return None

    prefix = str(configured or env.get("PYTHONPYCACHEPREFIX") or "").strip()
    if not prefix:
        try:
            user_token = str(os.getuid())
        except Exception:
            user_token = "user"
        prefix = os.path.join(
            _default_pycache_root(env=env),
            f"phasis_pycache_{user_token}_{os.getpid()}",
        )

    try:
        prefix = os.path.abspath(os.path.expanduser(prefix))
        os.makedirs(prefix, exist_ok=True)
        env["PYTHONPYCACHEPREFIX"] = prefix
        sys.pycache_prefix = prefix
        return prefix
    except Exception:
        return None


def _resolve_lib_worker_cap(ncores_local: int) -> int:
    configured = _coerce_positive_int(
        getattr(rt, "parallel_lib_worker_cap", None),
        _coerce_positive_int(getenv("Phasis_LIB_WORKER_CAP"), None),
    )
    if configured is None:
        # Streaming FASTQ conversion can still retain many valid unique tags.
        # Keep it sequential unless the user explicitly accepts more per-lib RAM.
        configured = 1 if str(getattr(rt, "libformat", "")).upper() == "Q" else int(max(1, ncores_local))
    return int(max(1, min(ncores_local, configured)))


def resolve_library_worker_cap(ncores_local: int) -> tuple[int, str]:
    """Return the effective library-worker cap and whether it was explicitly configured."""
    configured = _coerce_positive_int(
        getattr(rt, "parallel_lib_worker_cap", None),
        _coerce_positive_int(getenv("Phasis_LIB_WORKER_CAP"), None),
    )
    cap = _resolve_lib_worker_cap(ncores_local)
    if configured is not None:
        return cap, "PHASIS_LIB_WORKER_CAP"
    if str(getattr(rt, "libformat", "")).upper() == "Q":
        return cap, "conservative FASTQ default"
    return cap, "default"


def _resolve_worker_cap(cap_value, ncores_local: int) -> int:
    configured = _coerce_positive_int(cap_value, None)
    if configured is None:
        configured = int(max(1, ncores_local))
    return int(max(1, min(ncores_local, configured)))


def _resolve_maxtasksperchild(
    maxtasksperchild,
    *,
    unit: str,
    n_data: int,
    kind: str,
    start_method=None,
):
    explicit = _coerce_positive_int(maxtasksperchild, None)
    if explicit is not None:
        return explicit

    configured = _coerce_positive_int(
        getattr(rt, "parallel_maxtasksperchild", None),
        _coerce_positive_int(getenv("Phasis_MAXTASKSPERCHILD"), None),
    )
    if configured is not None:
        return configured

    method = _effective_start_method(start_method)
    if kind == "plot":
        return 8 if method == "spawn" else 4
    if unit == "lib":
        if n_data <= 64:
            return 32 if method == "spawn" else 16
        return 24 if method == "spawn" else 12
    if n_data <= 8:
        return 32 if method == "spawn" else 12
    if n_data <= 64:
        return 24 if method == "spawn" else 10
    return 16 if method == "spawn" else 8


def _ordered_buffered_results(buffered_results):
    return [buffered_results[key] for key in sorted(buffered_results.keys())]


def _commit_buffered_results(buffered_results, *, on_result, keep_results, results, pbar):
    if not buffered_results:
        return
    if on_result is not None:
        for res in buffered_results:
            on_result(res)
    if keep_results:
        results.extend(buffered_results)
    pbar.update(len(buffered_results))


def _run_serial_chunk(func, chunk):
    buffered_results = {}
    for item_id, arg in chunk:
        retry = safe_worker((func, arg))
        if isinstance(retry, RuntimeError):
            print(f"[ERROR] Serial retry failed for arg: {arg}\n{retry}")
        buffered_results[item_id] = retry
    return buffered_results


def safe_worker_indexed(args):
    item_id, func, arg = args
    return item_id, arg, safe_worker((func, arg))


def run_parallel_with_progress(
    func,
    data,
    desc=None,
    min_chunk=1,
    batch_factor=0.1,
    unit="lib",
    on_result=None,        # Optional: callable(result) -> None (avoid storing results)
    return_results=True,   # If False and on_result provided, we won’t keep a results list
    start_method=None,     # Optional: 'spawn' / 'fork' / 'forkserver'
    kind="compute",        # Passed to make_pool (e.g., 'plot' -> sets MPLBACKEND=Agg)
    snapshot_path=None,    # Optional explicit runtime snapshot path
    maxtasksperchild=None, # Auto-tuned worker reuse; callers can still override explicitly
    initial_worker_cap=None,
    max_worker_cap=None,
    adaptive_recovery=True,
    recovery_success_slices=2,
    recovery_progress_fraction=0.05,
    recovery_growth_factor=2.0,
):

    """
    Parallel, streaming, and adaptive:
      - Streams results via imap_unordered and buffers one slice at a time so
        retries only commit accepted work.
      - On any pool failure, automatically retries the current slice with
        smaller (chunk_size, nworkers): [proposed] -> 10 -> 5 -> 1; workers n->8->4->2->1.
      - Optional adaptive recovery can re-expand chunk_size/workers after a
        stretch of successful reduced-mode slices.
      - Callers can start conservatively with `initial_worker_cap` while still
        allowing recovery to grow toward `max_worker_cap`.
      - maxtasksperchild is auto-tuned unless callers override it.
      - BLAS single-threaded to avoid hidden fan-out.

    Tips:
      * If results are large, pass an `on_result` consumer and set return_results=False.
      * Keep `chunksize=1` to avoid big internal queues in the pool.
    """
    n_data = len(data)
    if n_data == 0:
        return []
    ncores = rt.ncores
    if rt.ncores is None or rt.ncores <= 0:
        ncores = multiprocessing.cpu_count()
    if unit == "lib":
        library_worker_cap = _resolve_lib_worker_cap(ncores)
        if initial_worker_cap is None:
            initial_worker_cap = library_worker_cap
        if max_worker_cap is None:
            max_worker_cap = library_worker_cap

    resolved_maxtasksperchild = _resolve_maxtasksperchild(
        maxtasksperchild,
        unit=unit,
        n_data=n_data,
        kind=kind,
        start_method=start_method,
    )
    resolved_max_worker_cap = _resolve_worker_cap(max_worker_cap, ncores)
    resolved_initial_worker_cap = _resolve_worker_cap(initial_worker_cap, ncores)
    resolved_initial_worker_cap = min(resolved_initial_worker_cap, resolved_max_worker_cap)

    # Initial chunk size & workers
    initial_chunk_size = _compute_initial_chunk_size(n_data, ncores, unit, min_chunk, batch_factor)
    chunk_size = initial_chunk_size
    initial_nworkers = min(ncores, chunk_size, resolved_initial_worker_cap) or 1
    nworkers = initial_nworkers
    success_slices_since_failure = 0
    items_since_failure = 0
    recovery_progress_items = _compute_recovery_progress_items(
        n_data,
        recovery_progress_fraction,
    )

    # Decide whether to accumulate results or stream-only
    keep_results = (on_result is None) or return_results
    results = [] if keep_results else None

    i = 0
    with tqdm(total=n_data, desc=desc, unit=unit) as pbar:
        while i < n_data:
            current_worker_target = min(ncores, chunk_size, resolved_max_worker_cap) or 1
            if adaptive_recovery and (
                chunk_size < initial_chunk_size or nworkers < current_worker_target
            ):
                if nworkers >= current_worker_target and chunk_size >= initial_chunk_size:
                    success_slices_since_failure = 0
                    items_since_failure = 0
                else:
                    if (
                        success_slices_since_failure >= max(1, int(recovery_success_slices))
                        and items_since_failure >= recovery_progress_items
                    ):
                        prev_chunk_size = chunk_size
                        prev_nworkers = nworkers
                        chunk_size = _grow_parallel_setting(
                            chunk_size,
                            initial_chunk_size,
                            growth_factor=recovery_growth_factor,
                        )
                        worker_growth_target = min(
                            ncores,
                            chunk_size,
                            resolved_max_worker_cap,
                        ) or 1
                        nworkers = min(
                            ncores,
                            worker_growth_target,
                            _grow_parallel_setting(
                                nworkers,
                                worker_growth_target,
                                growth_factor=recovery_growth_factor,
                            ),
                        ) or 1
                        success_slices_since_failure = 0
                        items_since_failure = 0
                        if chunk_size != prev_chunk_size or nworkers != prev_nworkers:
                            print(
                                f"[INFO] Re-expanding parallelism to chunk_size={chunk_size}, workers={nworkers}."
                            )

            start = i
            end = min(i + chunk_size, n_data)
            proposed = end - start if end > start else 1

            # Build retry ladder for sizes
            try_sizes = []
            for s in (proposed, 16, 12, 10, 8, 4, 2, 1):
                s = int(max(1, min(s, n_data - start)))
                if s not in try_sizes:
                    try_sizes.append(s)

            slice_completed = False
            last_exception = None
            slice_had_failure = False

            for local_chunk_size in try_sizes:
                end = min(start + local_chunk_size, n_data)
                chunk = data[start:end]

                # Worker trials, decreasing
                worker_trials = _worker_trial_ladder(nworkers, local_chunk_size, ncores)

                for nw in worker_trials:
                    # Try streaming this chunk with nw workers
                    try:
                        buffered_results = {}
                        with make_pool(
                            nw,
                            start_method=start_method,
                            kind=kind,
                            snapshot_path=snapshot_path,
                            maxtasksperchild=resolved_maxtasksperchild,
                        ) as pool:
                            # Stream results; avoid big intermediate lists
                            indexed_chunk = list(enumerate(chunk, start=start))
                            for item_id, arg, res in pool.imap_unordered(
                                safe_worker_indexed,
                                ((item_id, func, arg) for item_id, arg in indexed_chunk),
                                chunksize=1,
                            ):
                                if isinstance(res, RuntimeError):
                                    slice_had_failure = True
                                    retry = safe_worker((func, arg))
                                    if isinstance(retry, RuntimeError):
                                        print(f"[ERROR] Serial retry failed for arg: {arg}\n{retry}")
                                    buffered_results[item_id] = retry
                                    continue
                                buffered_results[item_id] = res

                            # If we reached here without exceptions, the chunk is done
                            slice_completed = True

                        _commit_buffered_results(
                            _ordered_buffered_results(buffered_results),
                            on_result=on_result,
                            keep_results=keep_results,
                            results=results,
                            pbar=pbar,
                        )

                        # Adopt smaller settings if they worked
                        if local_chunk_size < chunk_size:
                            chunk_size = local_chunk_size
                            #print(f"[INFO] Lowering ongoing chunk size to {chunk_size}.")
                        if nw < nworkers:
                            nworkers = nw
                            #print(f"[INFO] Lowering worker count to {nworkers}.")

                        break  # worker_trials loop
                    except MemoryError as e:
                        last_exception = e
                        slice_had_failure = True
                        print(f"\n[WARN] MemoryError on slice [{start}:{end}] size={local_chunk_size}, nworkers={nw}. Trying smaller.\n")
                    except Exception as e:
                        last_exception = e
                        slice_had_failure = True
                        print(f"\n[WARN] Pool error on slice [{start}:{end}] size={local_chunk_size}, nworkers={nw}: {e}\nTrying smaller.\n")

                if slice_completed:
                    if slice_had_failure:
                        success_slices_since_failure = 0
                        items_since_failure = 0
                    else:
                        success_slices_since_failure += 1
                        items_since_failure += (end - start)
                    break  # size loop

            # If pool attempts all failed for this slice, do serial for this slice
            if not slice_completed:
                print(f"[WARN] Running slice [{start}:{end}] serially after pool failures.")
                indexed_chunk = list(enumerate(data[start:end], start=start))
                serial_results = _run_serial_chunk(func, indexed_chunk)
                _commit_buffered_results(
                    _ordered_buffered_results(serial_results),
                    on_result=on_result,
                    keep_results=keep_results,
                    results=results,
                    pbar=pbar,
                )

                # If even 1-worker pool attempts failed, move to the most
                # conservative mode and let adaptive recovery climb back up later.
                chunk_size = 1
                nworkers = 1
                success_slices_since_failure = 0
                items_since_failure = 0

            # Advance window and do some housekeeping
            i = end
            gc.collect()

    return results if keep_results else None

def _compute_initial_chunk_size(n_data: int, ncores_local: int, unit: str, min_chunk: int, batch_factor: float):
    #print(f"batch factor set to {batch_factor}")
    if unit == "lib":
        worker_cap_for_lib = _resolve_lib_worker_cap(ncores_local)
        return min(ncores_local, worker_cap_for_lib) or 1
    #if n_data <= ncores_local:
        #print("n_data <= ncores_local: return 1")
    #    return 1
    n_batches = int(ncores_local * batch_factor) or 1
    #print(f"n_data is {n_data}")
    #print(f"n_batches set to {n_batches}")
    chunk_size = max(min_chunk, int(n_data / n_batches), int(ncores_local))
    #print(f" Initial chunk_size set to {chunk_size}")
    if n_data > 300:
        #print("n_data > 300")
        max_chunk_size = max(min_chunk, int(ncores_local))
        chunk_size = max(chunk_size, max_chunk_size)
        #print(f" Initial chunk_size set to {chunk_size}")
    return max(1, chunk_size)


def _compute_recovery_progress_items(n_data: int, recovery_progress_fraction: float) -> int:
    try:
        fraction = float(recovery_progress_fraction)
    except Exception:
        fraction = 0.05
    fraction = min(max(fraction, 0.0), 1.0)
    return max(1, int(n_data * fraction))


def _grow_parallel_setting(current: int, target: int, *, growth_factor: float = 2.0) -> int:
    current = int(max(1, current))
    target = int(max(1, target))
    if current >= target:
        return target
    try:
        grown = int(round(current * float(growth_factor)))
    except Exception:
        grown = current * 2
    return min(target, max(current + 1, grown))


def _worker_trial_ladder(preferred_workers: int, local_chunk_size: int, ncores: int) -> list[int]:
    ceiling = int(max(1, min(preferred_workers, local_chunk_size, ncores)))
    worker_trials = [ceiling]
    for w in (16, 12, 10, 8, 4, 2, 1):
        w = int(max(1, min(w, ceiling)))
        if w not in worker_trials:
            worker_trials.append(w)
    return worker_trials


def _infer_runtime_snapshot_path():
    # Prefer an explicit snapshot path if runtime.py defines one
    p = getattr(rt, "runtime_snapshot", None)
    if p and os.path.isfile(p):
        return p

    # Fallback: look in run_dir, then CWD
    run_dir = getattr(rt, "run_dir", None) or os.getcwd()
    cand = os.path.join(run_dir, ".phasis.runtime.json")
    if os.path.isfile(cand):
        return cand

    cand = os.path.join(os.getcwd(), ".phasis.runtime.json")
    if os.path.isfile(cand):
        return cand

    return None


def _pool_initializer(snapshot_path, kind):
    # Keep BLAS single-threaded in workers
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Plot pools on macOS must avoid GUI backends
    if kind == "plot":
        os.environ.setdefault("MPLBACKEND", "Agg")

    # Load runtime snapshot if available (spawn-safe)
    try:
        if snapshot_path and hasattr(rt, "load_snapshot"):
            rt.load_snapshot(snapshot_path)
    except Exception:
        pass

    # Ensure workers operate from the run directory where intermediates live
    try:
        rd = getattr(rt, "run_dir", None)
        if rd:
            os.chdir(rd)
    except Exception:
        pass

    # No compatibility module is needed here anymore.
    # Workers read phasis.runtime directly (or via stage-local sync helpers).


def make_pool(nworkers: int | None = None, *, processes: int | None = None, start_method: str | None = None,
             kind: str = "compute", snapshot_path: str | None = None, maxtasksperchild: int | None = None):
    """
    Pool with safer defaults to limit RAM spikes.

    - BLAS threads set to 1.
    - maxtasksperchild is auto-tuned unless callers override it.
    - macOS: spawn by default (safe for ObjC/matplotlib).
    - Linux: forkserver by default.

    NEW:
    - Supports `processes=` kwarg as alias for nworkers.
    - Loads runtime snapshot in workers (spawn-safe).
    - kind="plot" sets MPLBACKEND=Agg inside workers.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    _ensure_worker_pycache_prefix()

    if processes is not None:
        nworkers = processes
    nworkers = int(max(1, nworkers or 1))
    if maxtasksperchild is not None:
        maxtasksperchild = int(max(1, maxtasksperchild))

    start_method = _effective_start_method(start_method)


    if snapshot_path is None:
        snapshot_path = _infer_runtime_snapshot_path()

    if hasattr(multiprocessing, "get_context"):
        try:
            ctx = multiprocessing.get_context(start_method)
        except ValueError:
            ctx = multiprocessing.get_context()
    else:
        ctx = multiprocessing

    return ctx.Pool(
        processes=nworkers,
        maxtasksperchild=maxtasksperchild,
        initializer=_pool_initializer,
        initargs=(snapshot_path, kind),
    )

def safe_worker(args):
    """Run func(arg), catching exceptions; return RuntimeError sentinel on failure."""
    func, arg = args
    try:
        return func(arg)
    except Exception as e:
        import traceback  # allowed here (small, unavoidable for nice trace)
        return RuntimeError(f"Error in {func.__name__} with arg={arg}: {e}\n{traceback.format_exc()}")




def coreReserve(cores):
    """
    Decide how many CPU cores Phasis should reserve for the active run.

    Kept as a canonical helper outside legacy.py so startup can reserve cores
    without routing the real logic through the compatibility layer.

    ``-cores 0`` means all CPU cores visible to this process (respecting a
    scheduler allocation such as ``SLURM_CPUS_PER_TASK``). A positive request
    is exact unless it exceeds that visible allocation.
    """
    totalcores = _effective_visible_cpu_count()
    requested = int(cores)
    if requested < 0:
        raise ValueError("cores must be zero or a positive integer")
    if requested == 0:
        return int(max(1, totalcores))
    return int(max(1, min(requested, totalcores)))



def optimize(ncores: int, nfiles: int):
    '''
    Optimization of total processes and cores per process.

    Returns:
      (nproc, nspread)
    '''
    if nfiles <= 0:
        return 1, max(1, int(ncores) if ncores else 1)

    ncores = int(ncores) if ncores else 1
    nspread = int(ncores / nfiles)  # cores per process
    if nspread < 3:
        nspread = 3
        nproc = int(ncores / 3) if ncores >= 3 else 1
    else:
        nproc = nfiles

    if nproc < 1:
        nproc = 1

    print(f"\n#### {ncores} computing core(s) reserved for analysis")
    print(f"#### {nspread} computing core(s) assigned to each lib #\n")
    return nproc, nspread


def PPBalance(module, alist, *, n_workers: int | None = None, start_method: str | None = None, kind: str = "compute",
             maxtasksperchild: int | None = None):
    '''
    Parallel runner used by legacy mapping: run `module(arg)` for each arg in alist.

    - Uses the safer `make_pool()` defaults (spawn on macOS, forkserver on Linux).
    - Returns a list of results (usually ignored by callers).
    - Raises on pool-level exceptions.
    - If an item fails, returns a RuntimeError sentinel for that item (via safe_worker).
    '''
    print("##    FN PPBalance   ######")

    if n_workers is None:
        # default: use rt.ncores when available, else 1
        try:
            n_workers = int(getattr(rt, "ncores", 1) or 1)
        except Exception:
            n_workers = 1

    n_workers = int(max(1, n_workers))
    resolved_maxtasksperchild = _resolve_maxtasksperchild(
        maxtasksperchild,
        unit="task",
        n_data=len(alist),
        kind=kind,
        start_method=start_method,
    )

    results = []
    try:
        with make_pool(
            n_workers,
            start_method=start_method,
            kind=kind,
            maxtasksperchild=resolved_maxtasksperchild,
        ) as pool:
            for res in pool.imap_unordered(safe_worker, ((module, arg) for arg in alist), chunksize=1):
                results.append(res)
        return results
    except Exception as e:
        print(f"[PPBalance] Error in parallel processing: {e}")
        traceback.print_exc()
        raise
