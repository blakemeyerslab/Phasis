
from __future__ import annotations

import os
import re
import gc
import pickle
import shutil
import configparser
import multiprocessing
from collections import OrderedDict

import phasis.runtime as rt
from phasis.cache import MEM_FILE_DEFAULT, MemCache, getmd5, stage_signature
from phasis.parallel import run_parallel_with_progress, make_pool


def _safe_key(s: str) -> str:
    """Filesystem-safe key for cache files."""
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
    return "".join(ch if ch in allowed else "_" for ch in str(s))


def canonicalize_akey(s: str) -> str:
    """Return 'LIB-CHR' from any akey-like string without destroying dots in LIB."""
    base = os.path.basename(str(s)).strip()

    for suf in (".lclust", ".sclust"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break

    m = re.search(r"([A-Za-z0-9._+-]+-\d+)$", base)
    if m:
        return m.group(1)

    return base


def _flush_prev_cluster(prev_merged, clustid, clustlen_cutoff, clustdict_long):
    """Finalize one merged cluster into the long-cluster dict only."""
    if prev_merged:
        clustid += 1
        leftx = prev_merged[0]
        rightx = prev_merged[-1]
        if (rightx - leftx) + 1 > clustlen_cutoff:
            clustdict_long[clustid] = prev_merged
        prev_merged = []
    return prev_merged, clustid


CLUSTER_BUILD_SECTION = "CLUSTER_BUILD"
CLUSTER_BUILD_DEFAULT_INITIAL_WORKER_CAP = 20
CLUSTER_BUILD_BATCH_LIMIT_MAX = 96
CLUSTER_BUILD_RECOVERY_SUCCESS_SLICES = 2
CLUSTER_BUILD_RECOVERY_PROGRESS_FRACTION = 0.05
CLUSTER_BUILD_MAXTASKSPERCHILD = 32
CLUSTER_SCAN_PROGRESS_INTERVAL = 250


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


def _cluster_build_ncores() -> int:
    ncores = getattr(rt, "ncores", None)
    try:
        ncores = int(ncores) if ncores is not None else 0
    except Exception:
        ncores = 0
    if ncores <= 0:
        ncores = multiprocessing.cpu_count()
    return int(max(1, ncores))


def _resolve_cluster_build_worker_cap(runtime_attr: str, env_name: str, default=None) -> int:
    configured = _coerce_positive_int(
        getattr(rt, runtime_attr, None),
        _coerce_positive_int(os.environ.get(env_name), default),
    )
    ncores = _cluster_build_ncores()
    if configured is None:
        configured = ncores
    return int(max(1, min(ncores, configured)))


def _cluster_build_initial_worker_cap() -> int:
    return _resolve_cluster_build_worker_cap(
        "cluster_build_initial_worker_cap",
        "PHASIS_CLUSTER_BUILD_INITIAL_WORKER_CAP",
        CLUSTER_BUILD_DEFAULT_INITIAL_WORKER_CAP,
    )


def _cluster_build_max_worker_cap() -> int:
    return _resolve_cluster_build_worker_cap(
        "cluster_build_max_worker_cap",
        "PHASIS_CLUSTER_BUILD_MAX_WORKER_CAP",
        None,
    )


def _ensure_cluster_sections(cfg: configparser.ConfigParser) -> None:
    for section in (CLUSTER_BUILD_SECTION, "CLUSTERED"):
        if not cfg.has_section(section):
            cfg.add_section(section)


def _cluster_batch_limit() -> int:
    """
    Use larger clustering batches on high-core systems so scaffold-heavy runs
    can keep more lib-chr jobs in flight.
    """
    ncores = _cluster_build_ncores()
    initial_cap = _cluster_build_initial_worker_cap()
    return max(initial_cap * 2, min(max(initial_cap * 3, ncores * 2), CLUSTER_BUILD_BATCH_LIMIT_MAX))


def _cluster_output_paths_for_akey(clustfolder: str, akey: str) -> tuple[str, None]:
    a_safe = _safe_key(akey)
    lclust_path = os.path.realpath(os.path.join(clustfolder, f"{a_safe}.lclust"))
    return lclust_path, None


def _cluster_build_input_signature(count_path: str, *, phase: int, clustbuffer: int) -> str:
    return stage_signature(
        files=[count_path],
        params={
            "stage": "cluster_build",
            "phase": int(phase),
            "clustbuffer": int(clustbuffer),
        },
    )


def _cluster_source_label(src, source_index: int, source_count: int) -> str:
    if isinstance(src, str):
        return os.path.basename(src) or str(src)
    return f"in-memory source {source_index}/{source_count}"


def _maybe_report_cluster_scan_progress(
    source_label: str,
    scanned: int,
    total: int,
    cache_hits: int,
    queued: int,
    *,
    force: bool = False,
) -> None:
    if not force and scanned % CLUSTER_SCAN_PROGRESS_INTERVAL != 0:
        return
    if total > 0:
        print(
            f"[scan] {source_label}: planned {scanned}/{total} lib-chr groups "
            f"(cache hits: {cache_hits}, queued: {queued})",
            flush=True,
        )
    else:
        print(
            f"[scan] {source_label}: planned {scanned} lib-chr groups "
            f"(cache hits: {cache_hits}, queued: {queued})",
            flush=True,
        )


def _record_compat_cluster_md5(
    cfg: configparser.ConfigParser,
    lclust_path: str,
    md5hex: str | None = None,
) -> bool:
    if not lclust_path or not os.path.isfile(lclust_path):
        return False

    if md5hex is None:
        _, md5hex = getmd5(lclust_path, wait_stable=False)
    md5hex = str(md5hex or "")
    if not md5hex:
        return False

    prev = (cfg["CLUSTERED"].get(lclust_path) or "").strip()
    if prev == md5hex:
        return False

    cfg["CLUSTERED"][lclust_path] = md5hex
    return True




def getclusters(args):
    """
    Compute clusters for one (akey, acounter) and write to disk immediately.

    Returns: (akey, lclust_file, lclust_md5)

    NOTE: spawn-safe: derives all parameters from rt.* inside the worker.
    """
    akey, acounter, clustfolder = args

    phase = int(rt.phase)
    clustbuffer = int(rt.clustbuffer)
    clustsplit = phase + 1 + 3
    clustlen_cutoff = phase * 4 + 3 + 1

    akey_safe = canonicalize_akey(akey)
    lclust_file = os.path.join(clustfolder, f"{akey_safe}.lclust")

    if isinstance(acounter, OrderedDict):
        keys_iter = acounter.keys()
    else:
        try:
            keys_iter = sorted(acounter.keys(), key=int)
        except Exception:
            keys_iter = sorted(acounter.keys())

    it = iter(keys_iter)
    try:
        first_key = next(it)
    except StopIteration:
        with open(lclust_file, "wb") as f1:
            pickle.dump({}, f1, protocol=pickle.HIGHEST_PROTOCOL)
        _, lclust_md5 = getmd5(lclust_file)
        return (akey, lclust_file, lclust_md5)

    try:
        first_pos = int(first_key)
    except Exception:
        first_pos = first_key

    clustdict_long = {}
    clustid = 0

    prev_merged = []
    curr_pre = [first_pos]
    last_pos = first_pos

    for k in it:
        try:
            pos = int(k)
        except Exception:
            pos = k

        if pos - last_pos > clustsplit:
            if not prev_merged:
                prev_merged = curr_pre
            else:
                if curr_pre[0] <= (prev_merged[-1] + clustbuffer):
                    prev_merged.extend(curr_pre)
                else:
                    prev_merged, clustid = _flush_prev_cluster(
                        prev_merged, clustid, clustlen_cutoff, clustdict_long
                    )
                    prev_merged = curr_pre
            curr_pre = [pos]
        else:
            curr_pre.append(pos)

        last_pos = pos

    if not prev_merged:
        prev_merged = curr_pre
    else:
        if curr_pre[0] <= (prev_merged[-1] + clustbuffer):
            prev_merged.extend(curr_pre)
        else:
            prev_merged, clustid = _flush_prev_cluster(
                prev_merged, clustid, clustlen_cutoff, clustdict_long
            )
            prev_merged = curr_pre

    prev_merged, clustid = _flush_prev_cluster(
        prev_merged, clustid, clustlen_cutoff, clustdict_long
    )

    with open(lclust_file, "wb") as f1:
        pickle.dump(clustdict_long, f1, protocol=pickle.HIGHEST_PROTOCOL)

    _, lclust_md5 = getmd5(lclust_file)

    del clustdict_long, prev_merged, curr_pre
    gc.collect()

    return (akey, lclust_file, lclust_md5)


def alt_parallel_process(func, data_chunks):
    """Fallback clustering runner using make_pool (spawn-safe initializer)."""
    max_workers = _cluster_build_initial_worker_cap()

    with make_pool(max_workers, maxtasksperchild=CLUSTER_BUILD_MAXTASKSPERCHILD) as pool:
        for result in pool.imap_unordered(func, data_chunks, chunksize=1):
            yield result


def process_cluster_batch(batch, batch_id):
    """Run one clustering batch and return results."""
    initial_worker_cap = _cluster_build_initial_worker_cap()
    max_worker_cap = _cluster_build_max_worker_cap()
    try:
        return run_parallel_with_progress(
            getclusters,
            batch,
            desc=f"Clustering batch {batch_id}",
            unit="lib-chr",
            maxtasksperchild=CLUSTER_BUILD_MAXTASKSPERCHILD,
            initial_worker_cap=initial_worker_cap,
            max_worker_cap=max_worker_cap,
            adaptive_recovery=True,
            recovery_success_slices=CLUSTER_BUILD_RECOVERY_SUCCESS_SLICES,
            recovery_progress_fraction=CLUSTER_BUILD_RECOVERY_PROGRESS_FRACTION,
        )
    except Exception as e:
        print(f"[WARN] run_parallel_with_progress failed on batch {batch_id}: {e}")
        print("[INFO] Falling back to alternative chunked parallel_process...")
        try:
            return list(alt_parallel_process(getclusters, batch))
        except Exception as ee:
            print(f"[ERROR] Fallback failed for batch {batch_id}: {ee}")
            return []


def _prune_old_clustered_entries(cfg: configparser.ConfigParser, basename: str, keep_abs: str) -> int:
    """Remove stale [CLUSTERED] entries with same basename but different path."""
    if not cfg.has_section("CLUSTERED"):
        return 0
    to_delete = [
        k
        for k in cfg["CLUSTERED"].keys()
        if os.path.basename(k) == basename and os.path.realpath(k) != os.path.realpath(keep_abs)
    ]
    for k in to_delete:
        try:
            cfg.remove_option("CLUSTERED", k)
        except Exception:
            pass
    return len(to_delete)


def flush_cluster_batch(
    batch_items,
    idx,
    *,
    clustfolder,
    cfg,
    clustered_md5,
    results,
    new_hashes,
    processed_akeys,
    cache,
    sig_by_output,
):
    """Top-level (non-nested) flush helper to satisfy no-nested-functions rule."""
    if not batch_items:
        return False

    compat_dirty = False
    chunk_results = process_cluster_batch(batch_items, idx)
    print(
        f"[scan] Finalizing clustering batch {idx} ({len(chunk_results)} outputs)...",
        flush=True,
    )
    for res in chunk_results:
        if isinstance(res, RuntimeError):
            # run_parallel_with_progress returns RuntimeError sentinel on worker failure
            raise res

        if len(res) == 3:
            a, lfile, lmd5 = res
            sfile = None
        elif len(res) == 4:
            a, lfile, sfile, lmd5 = res
        else:
            a, lfile = res[0], res[1]
            sfile = None
            _, lmd5 = getmd5(lfile)

        want_l, want_s = _cluster_output_paths_for_akey(clustfolder, a)

        # If worker wrote elsewhere, move into clustfolder
        if os.path.isfile(lfile) and os.path.realpath(lfile) != want_l:
            try:
                os.replace(lfile, want_l)
            except Exception:
                shutil.copy2(lfile, want_l)
                try:
                    os.remove(lfile)
                except Exception:
                    pass
        if sfile and want_s and os.path.isfile(sfile) and os.path.realpath(sfile) != want_s:
            try:
                os.replace(sfile, want_s)
            except Exception:
                shutil.copy2(sfile, want_s)
                try:
                    os.remove(sfile)
                except Exception:
                    pass

        lmd5_final = str(lmd5 or "")
        if not lmd5_final:
            _, lmd5_final = getmd5(want_l)
        new_hashes[want_l] = lmd5_final

        pruned = _prune_old_clustered_entries(cfg, os.path.basename(want_l), want_l)
        if pruned:
            for k in list(clustered_md5.keys()):
                if os.path.basename(k) == os.path.basename(want_l) and os.path.realpath(k) != want_l:
                    clustered_md5.pop(k, None)

        clustered_md5[want_l] = lmd5_final

        l_sig = sig_by_output.get(want_l)
        if cache is not None and l_sig is not None:
            cache.record(CLUSTER_BUILD_SECTION, want_l, l_sig)

        compat_dirty = (
            _record_compat_cluster_md5(cfg, want_l, md5hex=lmd5_final) or compat_dirty
        )
        results.append((a, want_l, None))
        processed_akeys.append(a)

    return compat_dirty


def clusterprocess(libs_poscountdict, clustfolder):
    """
    Phase I clustering (spawn-safe, no nested functions).

    - Writes per-lib-chr clusters to <clustfolder>/<akey>.lclust
    - Centralizes cache reuse in [CLUSTER_BUILD] for lclust outputs
    - Preserves legacy [CLUSTERED] md5 bookkeeping for compatibility
    - Returns: [(akey, lclust_file, None), ...]
    """
    print("#### Fn: Find Clusters #######################")

    os.makedirs(clustfolder, exist_ok=True)

    # Normalize sources
    if isinstance(libs_poscountdict, (dict, str)):
        sources = [libs_poscountdict]
    else:
        sources = list(libs_poscountdict)

    mem_file = rt.memFile or MEM_FILE_DEFAULT
    phase = int(rt.phase)
    clustbuffer = int(rt.clustbuffer)

    cache = MemCache.load(mem_file)
    cfg = cache.cfg
    _ensure_cluster_sections(cfg)

    clustered_md5 = dict(cfg["CLUSTERED"])

    results: list[tuple[str, str, str | None]] = []
    new_hashes: dict[str, str] = {}
    processed_akeys: list[str] = []
    sig_by_output: dict[str, str] = {}
    compat_dirty = False
    total_cache_hits = 0
    total_queued = 0

    CLUSTER_CHUNK_MAX = _cluster_batch_limit()
    batch = []
    batch_index = 0
    print(
        f"[scan] Cluster batches will start near {_cluster_build_initial_worker_cap()} worker(s) "
        f"and can grow to {_cluster_build_max_worker_cap()} across slices; "
        f"queued misses flush in batches of up to {CLUSTER_CHUNK_MAX}.",
        flush=True,
    )

    for source_index, src in enumerate(sources, start=1):
        input_sig = None
        source_label = _cluster_source_label(src, source_index, len(sources))
        print(f"[scan] Loading cluster inputs from {source_label}...", flush=True)
        if isinstance(src, str):
            try:
                with open(src, "rb") as fh:
                    libdict = pickle.load(fh)
            except Exception:
                print(f"[WARN] Failed to load count file {src}")
                continue
            loaded_from_path = True
            input_sig = _cluster_build_input_signature(
                src,
                phase=phase,
                clustbuffer=clustbuffer,
            )
        elif isinstance(src, dict):
            libdict = src
            loaded_from_path = False
        else:
            print(f"[WARN] Unexpected type in libs_poscountdict: {type(src)} (skipping)")
            continue

        source_total = len(libdict)
        source_scanned = 0
        source_cache_hits = 0
        source_queued = 0
        print(
            f"[scan] Loaded {source_total} lib-chr groups from {source_label}; checking cache and queuing work...",
            flush=True,
        )

        for akey, positions in libdict.items():
            lclust_path, _ = _cluster_output_paths_for_akey(clustfolder, akey)

            if input_sig is not None:
                sig_by_output[lclust_path] = input_sig

                cached_l_md5 = cache.get(CLUSTER_BUILD_SECTION, lclust_path, "") or ""
                l_hit = cache.hit(
                    CLUSTER_BUILD_SECTION,
                    lclust_path,
                    input_sig,
                    wait_stable=False,
                )
                if l_hit:
                    _prune_old_clustered_entries(cfg, os.path.basename(lclust_path), lclust_path)
                    compat_dirty = (
                        _record_compat_cluster_md5(cfg, lclust_path, md5hex=cached_l_md5) or compat_dirty
                    )
                    results.append((akey, lclust_path, None))
                    processed_akeys.append(akey)
                    source_scanned += 1
                    source_cache_hits += 1
                    total_cache_hits += 1
                    _maybe_report_cluster_scan_progress(
                        source_label,
                        source_scanned,
                        source_total,
                        source_cache_hits,
                        source_queued,
                    )
                    continue

                if os.path.isfile(lclust_path):
                    _, cur_md5 = getmd5(lclust_path, wait_stable=False)
                    prev_md5 = cfg["CLUSTERED"].get(lclust_path, "")
                    if prev_md5 and cur_md5 and prev_md5 == cur_md5:
                        print(f"Legacy cache matches for clustered library-chr {akey}")
                        cache.record(
                            CLUSTER_BUILD_SECTION,
                            lclust_path,
                            input_sig,
                            output_fp=cur_md5,
                        )
                        _prune_old_clustered_entries(cfg, os.path.basename(lclust_path), lclust_path)
                        results.append((akey, lclust_path, None))
                        processed_akeys.append(akey)
                        source_scanned += 1
                        source_cache_hits += 1
                        total_cache_hits += 1
                        _maybe_report_cluster_scan_progress(
                            source_label,
                            source_scanned,
                            source_total,
                            source_cache_hits,
                            source_queued,
                        )
                        continue
            else:
                if os.path.isfile(lclust_path):
                    _, cur_md5 = getmd5(lclust_path, wait_stable=False)
                    prev_md5 = cfg["CLUSTERED"].get(lclust_path, "")
                    if prev_md5 and cur_md5 and prev_md5 == cur_md5:
                        _prune_old_clustered_entries(cfg, os.path.basename(lclust_path), lclust_path)
                        results.append((akey, lclust_path, None))
                        processed_akeys.append(akey)
                        source_scanned += 1
                        source_cache_hits += 1
                        total_cache_hits += 1
                        _maybe_report_cluster_scan_progress(
                            source_label,
                            source_scanned,
                            source_total,
                            source_cache_hits,
                            source_queued,
                        )
                        continue

            batch.append((akey, positions, clustfolder))
            source_scanned += 1
            source_queued += 1
            total_queued += 1
            _maybe_report_cluster_scan_progress(
                source_label,
                source_scanned,
                source_total,
                source_cache_hits,
                source_queued,
            )
            if len(batch) >= CLUSTER_CHUNK_MAX:
                batch_index += 1
                compat_dirty = flush_cluster_batch(
                    batch,
                    batch_index,
                    clustfolder=clustfolder,
                    cfg=cfg,
                    clustered_md5=clustered_md5,
                    results=results,
                    new_hashes=new_hashes,
                    processed_akeys=processed_akeys,
                    cache=cache,
                    sig_by_output=sig_by_output,
                ) or compat_dirty
                batch = []
                gc.collect()

        _maybe_report_cluster_scan_progress(
            source_label,
            source_scanned,
            source_total,
            source_cache_hits,
            source_queued,
            force=True,
        )

        if loaded_from_path:
            del libdict
            gc.collect()

    if batch:
        batch_index += 1
        compat_dirty = flush_cluster_batch(
            batch,
            batch_index,
            clustfolder=clustfolder,
            cfg=cfg,
            clustered_md5=clustered_md5,
            results=results,
            new_hashes=new_hashes,
            processed_akeys=processed_akeys,
            cache=cache,
            sig_by_output=sig_by_output,
        ) or compat_dirty
        batch = []
        gc.collect()

    # Persist legacy [CLUSTERED] updates
    if not cfg.has_section("CLUSTERED"):
        cfg.add_section("CLUSTERED")
    for lpath, md5 in new_hashes.items():
        cfg["CLUSTERED"][lpath] = md5
    if new_hashes or compat_dirty:
        cache.flush()

    # Save akeys list for downstream
    try:
        with open(os.path.join(clustfolder, "libchr-keys.p"), "wb") as pf:
            pickle.dump(processed_akeys, pf)
    except Exception as e:
        print(f"[WARN] Could not write libchr-keys.p: {e}")

    print(
        f"[scan] Cluster planning finished with {total_cache_hits} cache hits and {total_queued} queued lib-chr groups.",
        flush=True,
    )

    return results
