"""Cluster scoring stage extracted from legacy.py (Phase I refactor).

This module keeps the original scoring behavior while making the orchestration
available as a stage function. It remains compatible with macOS spawn and Linux
fork by pulling lightweight settings from phasis.runtime and by passing the
scored chunk folder explicitly to worker tasks.
"""

import configparser
import gc
import multiprocessing
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

from scipy.stats import combine_pvalues, hypergeom, mannwhitneyu

import phasis.runtime as rt
from phasis.cache import (
    MemCache,
    assemble_candidate_from_chunks,
    compute_cache_signature_from_file_manifest,
    default_memfile_path,
    getmd5,
    md5_file_worker,
    sanitize_mem_md5s,
)
from phasis.parallel import run_parallel_with_progress

# Match legacy advanced defaults
UNIQRATIO_HIT = 2
DOMSIZE_CUT = 0.50
WINDOW_SIZE = 15

# Stage-local mirrors of runtime values (refreshed by sync_from_runtime)
mismat = None
maxhits = None
clustbuffer = None
phase = None
uniqueRatioCut = None
memFile = None
scoredClustFolder = None


CLUSTER_SCORING_SECTION = "CLUSTER_SCORING"
CLUSTER_SCORING_DEFAULT_INITIAL_WORKER_CAP = 20
CLUSTER_SCORING_BATCH_LIMIT = 256
CLUSTER_SCORING_HASH_MAXTASKSPERCHILD = 64
CLUSTER_SCORING_LOAD_MAXTASKSPERCHILD = 24
CLUSTER_SCORING_SCORE_MAXTASKSPERCHILD = 8
CLUSTER_SCORING_NESTDICT_MAXTASKSPERCHILD = 16
CLUSTER_SCORING_ASSEMBLY_MAXTASKSPERCHILD = 16
CLUSTER_SCORING_RECOVERY_SUCCESS_SLICES = 2
CLUSTER_SCORING_RECOVERY_PROGRESS_FRACTION = 0.05


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


def _cluster_scoring_ncores() -> int:
    ncores = getattr(rt, "ncores", None)
    try:
        ncores = int(ncores) if ncores is not None else 0
    except Exception:
        ncores = 0
    if ncores <= 0:
        ncores = multiprocessing.cpu_count()
    return int(max(1, ncores))


def _resolve_cluster_scoring_worker_cap(runtime_attr: str, env_name: str, default=None) -> int:
    configured = _coerce_positive_int(
        getattr(rt, runtime_attr, None),
        _coerce_positive_int(os.environ.get(env_name), default),
    )
    ncores = _cluster_scoring_ncores()
    if configured is None:
        configured = ncores
    return int(max(1, min(ncores, configured)))


def _cluster_scoring_initial_worker_cap() -> int:
    return _resolve_cluster_scoring_worker_cap(
        "cluster_scoring_initial_worker_cap",
        "PHASIS_CLUSTER_SCORING_INITIAL_WORKER_CAP",
        CLUSTER_SCORING_DEFAULT_INITIAL_WORKER_CAP,
    )


def _cluster_scoring_max_worker_cap() -> int:
    return _resolve_cluster_scoring_worker_cap(
        "cluster_scoring_max_worker_cap",
        "PHASIS_CLUSTER_SCORING_MAX_WORKER_CAP",
        None,
    )


def _cluster_scoring_parallel_kwargs(
    *,
    maxtasksperchild: int,
    initial_worker_cap: int,
    max_worker_cap: int,
) -> dict:
    return {
        "maxtasksperchild": maxtasksperchild,
        "initial_worker_cap": initial_worker_cap,
        "max_worker_cap": max_worker_cap,
        "adaptive_recovery": True,
        "recovery_success_slices": CLUSTER_SCORING_RECOVERY_SUCCESS_SLICES,
        "recovery_progress_fraction": CLUSTER_SCORING_RECOVERY_PROGRESS_FRACTION,
    }


def _cluster_scoring_batch_size(n_data: int, initial_worker_cap: int) -> int:
    if n_data <= 0:
        return 1
    ncores = _cluster_scoring_ncores()
    target = max(initial_worker_cap * 6, ncores * 4, 128)
    target = min(CLUSTER_SCORING_BATCH_LIMIT, target)
    return max(1, min(n_data, max(initial_worker_cap, target)))


def _cluster_scoring_needed_nest_keys(inputs) -> set[str]:
    keys = set()
    for akey, _ in inputs or []:
        akey_text = str(akey)
        keys.add(akey_text)
        keys.add(canonicalize_akey(akey_text))
    return keys


def _verify_existing_candidate_outputs(
    cfg: configparser.ConfigParser,
    expected_outfiles,
    *,
    initial_worker_cap: int,
    max_worker_cap: int,
) -> dict[str, bool]:
    existing_paths = []
    outfile_realpaths = {}
    for _, outfile in expected_outfiles:
        if os.path.isfile(outfile):
            realpath = os.path.realpath(outfile)
            outfile_realpaths[outfile] = realpath
            existing_paths.append(realpath)

    if not existing_paths:
        return {}

    md5_results = run_parallel_with_progress(
        md5_file_worker,
        existing_paths,
        desc="Verifying existing candidate outputs",
        min_chunk=1,
        unit="file",
        **_cluster_scoring_parallel_kwargs(
            maxtasksperchild=CLUSTER_SCORING_HASH_MAXTASKSPERCHILD,
            initial_worker_cap=initial_worker_cap,
            max_worker_cap=max_worker_cap,
        ),
    )
    current_md5 = {path: md5 for path, md5 in md5_results if md5}

    verified = {}
    for _, outfile in expected_outfiles:
        realpath = outfile_realpaths.get(outfile)
        current = current_md5.get(realpath)
        previous = cfg["CLUSTERS"].get(os.path.basename(outfile))
        verified[outfile] = bool(realpath and current and previous and current == previous)
    return verified


def inspect_lclust_input_for_scoring(arg):
    """
    Read-only .lclust inspection used to build a one-time scoring manifest.

    Returns:
      (akey, absolute_realpath, md5hex_or_none)
    """
    akey, lclust_path = arg
    try:
        lclust_real = os.path.realpath(lclust_path)
    except Exception:
        lclust_real = lclust_path

    if not os.path.isfile(lclust_real):
        return (akey, lclust_real, None)

    _, md5hex = getmd5(lclust_real, wait_stable=False)
    if not md5hex:
        md5hex = None
    return (akey, lclust_real, md5hex)


def _inspect_cluster_scoring_inputs(
    filtered_inputs,
    cfg: configparser.ConfigParser,
    sect_lclust: str,
    *,
    initial_worker_cap: int,
    max_worker_cap: int,
    precomputed_md5_map=None,
):
    if not filtered_inputs:
        return {}, {}, set()

    precomputed_md5_map = precomputed_md5_map or {}
    manifest_results = []
    inspect_jobs = []
    for akey, lclust_path in filtered_inputs:
        lclust_real = os.path.realpath(lclust_path)
        precomputed_md5 = str(precomputed_md5_map.get(lclust_real) or "")
        if precomputed_md5:
            manifest_results.append((akey, lclust_real, precomputed_md5))
        else:
            inspect_jobs.append((akey, lclust_real))

    if inspect_jobs:
        inspected = run_parallel_with_progress(
            inspect_lclust_input_for_scoring,
            inspect_jobs,
            desc="Inspecting .lclust inputs",
            min_chunk=1,
            unit="file",
            **_cluster_scoring_parallel_kwargs(
                maxtasksperchild=CLUSTER_SCORING_HASH_MAXTASKSPERCHILD,
                initial_worker_cap=initial_worker_cap,
                max_worker_cap=max_worker_cap,
            ),
        )
        manifest_results.extend(inspected)

    manifest_by_path = {}
    lclust_md5_updates = {}
    changed_inputs = set()
    unchanged = 0

    for akey, lclust_path, md5hex in manifest_results:
        md5_text = str(md5hex or "")
        prev_md5 = (cfg[sect_lclust].get(lclust_path) or "").strip()
        is_changed = (not md5_text) or (not prev_md5) or (prev_md5 != md5_text)
        manifest_by_path[lclust_path] = (akey, md5_text, is_changed)
        if md5_text:
            lclust_md5_updates[lclust_path] = md5_text
        if is_changed:
            changed_inputs.add(akey)
        else:
            unchanged += 1

    print(
        f"[scan] .lclust manifest classified {len(changed_inputs)} changed and {unchanged} unchanged input(s).",
        flush=True,
    )
    return manifest_by_path, lclust_md5_updates, changed_inputs


def _ensure_cluster_scoring_sections(cfg: configparser.ConfigParser) -> None:
    for sect in ("CLUSTERS", "CLUSTERED", "SCORED_CHUNKS", CLUSTER_SCORING_SECTION):
        if not cfg.has_section(sect):
            cfg.add_section(sect)


def _normalize_cache_file_inputs(sources):
    if sources is None:
        return []
    if isinstance(sources, str):
        iterable = [sources]
    else:
        iterable = list(sources)

    norm = []
    for src in iterable:
        if not isinstance(src, str):
            return None
        norm.append(os.path.realpath(src))

    return sorted(set(norm))


def load_filtered_nestdict_source(job):
    idx, src, needed_keys = job
    try:
        with open(src, "rb") as fh:
            loaded = pickle.load(fh)
    except Exception as e:
        return idx, None, f"[WARN] Could not load dict file '{src}': {e}"

    if not isinstance(loaded, dict):
        return idx, None, f"[WARN] Dict file '{src}' did not contain a dict; got {type(loaded)}"

    if needed_keys is None:
        return idx, loaded, None

    subset = {}
    for k in needed_keys:
        if k in loaded:
            subset[k] = loaded[k]
    return idx, subset, None


def _nestdict_pickle_sources(sources):
    """
    Return ordered pickle paths when every nestdict source is path-like.

    Dict objects are still supported by the legacy global-loader path.  The
    batched path is intentionally limited to parser pickle files so it can load
    and release one library at a time without changing the public parser format.
    """
    if sources is None:
        return None
    if isinstance(sources, str):
        iterable = [sources]
    else:
        try:
            iterable = list(sources)
        except TypeError:
            return None

    ordered = []
    seen = set()
    for src in iterable:
        if not isinstance(src, str):
            return None
        path = os.path.realpath(src)
        if path not in seen:
            ordered.append(path)
            seen.add(path)
    return ordered


def _load_nestdict_source_direct(src):
    try:
        with open(src, "rb") as fh:
            loaded = pickle.load(fh)
    except Exception as e:
        print(f"[WARN] Could not load dict file '{src}': {e}")
        return None

    if not isinstance(loaded, dict):
        print(f"[WARN] Dict file '{src}' did not contain a dict; got {type(loaded)}")
        return None

    return loaded


def _nestdict_source_label(src):
    base = os.path.basename(str(src))
    return base[:-5] if base.endswith(".dict") else _basename_no_ext(base)


def _select_inputs_for_nestdict_source(inputs, loaded_nestdict, unresolved_keys):
    """
    Pick still-unresolved .lclust inputs present in the loaded library nestdict.

    Parser and cluster keys should normally match exactly.  Canonical matching
    is kept for compatibility with historical path-like keys.
    """
    selected = []
    for akey, lclust_path in inputs:
        akey_text = str(akey)
        akey_can = canonicalize_akey(akey_text)
        if akey_can not in unresolved_keys:
            continue
        if akey_text in loaded_nestdict or akey_can in loaded_nestdict:
            selected.append((akey, lclust_path))
            unresolved_keys.discard(akey_can)
    return selected


def inspect_cluster_scoring_signature_input(path):
    try:
        p = os.path.abspath(os.path.expanduser(str(path)))
    except Exception:
        p = str(path)

    if not os.path.isfile(p):
        return (p, "", -1, False)

    _, fp = getmd5(p, wait_stable=False)
    fp = str(fp or "")
    try:
        size = os.path.getsize(p)
    except Exception:
        size = -1
    return (p, fp, size, True)


def _cluster_scoring_stage_signature(
    lclust_paths,
    nestdict_sources,
    *,
    phase,
    unique_ratio_cut,
    concat_mode,
    merged_name,
    expected_outputs,
    initial_worker_cap,
    max_worker_cap,
):
    nest_paths = _normalize_cache_file_inputs(nestdict_sources)
    if nest_paths is None:
        return None, {}

    lclust_norm = [os.path.realpath(p) for p in lclust_paths]
    files = sorted(set(lclust_norm + nest_paths))
    extras = sorted({os.path.basename(str(p)) for p in expected_outputs})

    print(
        f"[scan] Building cluster scoring stage signature from {len(files)} input file(s)...",
        flush=True,
    )
    file_manifest = run_parallel_with_progress(
        inspect_cluster_scoring_signature_input,
        files,
        desc="Inspecting cluster scoring stage inputs",
        min_chunk=1,
        unit="file",
        **_cluster_scoring_parallel_kwargs(
            maxtasksperchild=CLUSTER_SCORING_HASH_MAXTASKSPERCHILD,
            initial_worker_cap=initial_worker_cap,
            max_worker_cap=max_worker_cap,
        ),
    )

    lclust_set = set(lclust_norm)
    lclust_md5_map = {}
    for path, fp, _size, exists in file_manifest:
        if exists and path in lclust_set and fp:
            lclust_md5_map[path] = fp

    return compute_cache_signature_from_file_manifest(
        file_manifest=file_manifest,
        params={
            "stage": "cluster_scoring",
            "phase": int(phase),
            "uniqueRatioCut": float(unique_ratio_cut),
            "concat_mode": bool(concat_mode),
            "merged_name": str(merged_name),
            "window_size": int(WINDOW_SIZE),
            "uniqratio_hit": int(UNIQRATIO_HIT),
            "domsize_cut": float(DOMSIZE_CUT),
        },
        extra=extras,
    ), lclust_md5_map


def _record_compat_cluster_output_md5(cfg: configparser.ConfigParser, section: str, path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False

    _, md5hex = md5_file_worker(path)
    md5hex = str(md5hex or "")
    if not md5hex:
        return False

    key = os.path.basename(path) if section == "CLUSTERS" else os.path.realpath(path)
    prev = (cfg[section].get(key) or "").strip()
    if prev == md5hex:
        return False

    cfg[section][key] = md5hex
    return True


def sync_from_runtime() -> None:
    """Refresh stage-local globals from phasis.runtime (spawn-safe with snapshot)."""
    global mismat, maxhits, clustbuffer, phase, uniqueRatioCut, memFile

    mismat = getattr(rt, "mismat", mismat)
    maxhits = getattr(rt, "maxhits", maxhits)
    clustbuffer = getattr(rt, "clustbuffer", clustbuffer)
    phase = getattr(rt, "phase", phase)
    uniqueRatioCut = getattr(rt, "uniqueRatioCut", uniqueRatioCut)

    mem_override = getattr(rt, "memFile", None)
    if mem_override:
        memFile = mem_override
    else:
        memFile = default_memfile_path()


def candidate_output_needs_rebuild(path: str) -> bool:
    return (not os.path.isfile(path)) or (os.path.getsize(path) == 0)

def resolve_lclust_path(path_like, clustfolder):
    """
    Resolve .lclust path to an existing ABS REALPATH.
    Preference: given path → clustfolder/basename → CWD/basename fallback.
    """
    p = str(path_like)
    if os.path.isfile(p):
        return os.path.realpath(p)
    b = os.path.basename(p)
    cand = os.path.join(clustfolder, b) if clustfolder else None
    if cand and os.path.isfile(cand):
        return os.path.realpath(cand)
    # last-ditch: interpret relative to CWD
    cand2 = os.path.join(os.getcwd(), b)
    if os.path.isfile(cand2):
        return os.path.realpath(cand2)
    # return normalized original (may not exist; caller will check)
    return os.path.realpath(p)

def _basename_no_ext(p):
    # '/a/b/ALL_LIBS.fas' -> 'ALL_LIBS'
    return os.path.basename(str(p)).rsplit('.', 1)[0]

def canonicalize_akey(s: str) -> str:
    """
    Return 'LIB-CHR' from any akey-like string without destroying dots in LIB.
    Handles paths and known suffixes (.lclust/.sclust).
    """
    base = os.path.basename(str(s)).strip()

    # Only strip known file suffixes (do NOT rsplit('.') blindly)
    for suf in (".lclust", ".sclust"):
        if base.endswith(suf):
            base = base[:-len(suf)]
            break

    # Prefer trailing "<lib>-<digits>" (chr) when present
    m = re.search(r'([A-Za-z0-9._+-]+-\d+)$', base)
    if m:
        return m.group(1)

    return base

def build_libchrs_nestdict(
    sources,
    needed_keys=None,
    *,
    initial_worker_cap: int | None = None,
    max_worker_cap: int | None = None,
):
    """
    Build {akey: value} from a mix of:
      - dict objects (each mapping akey -> value), or
      - '*.dict' pickle file paths containing such dicts.

    If needed_keys is provided (set of akeys), only those keys are loaded.
    """
    result = {}
    # Normalize input
    if isinstance(sources, (dict, str)):
        iterable = [sources]
    else:
        iterable = list(sources)

    needed_keys_seq = None
    if needed_keys is not None:
        needed_keys_seq = tuple(sorted(str(k) for k in needed_keys))

    path_jobs = []
    path_results = {}
    for idx, src in enumerate(iterable):
        if isinstance(src, dict):
            # Direct merge, but only required keys if specified
            if needed_keys is None:
                result.update(src)
            else:
                for k, v in src.items():
                    if k in needed_keys:
                        result[k] = v
        elif isinstance(src, str):
            path_jobs.append((idx, src, needed_keys_seq))
        else:
            print(f"[WARN] Unexpected libs_nestdict element type at index {idx}: {type(src)}; skipping")

    if path_jobs:
        path_entries = run_parallel_with_progress(
            load_filtered_nestdict_source,
            path_jobs,
            desc="Loading nestdict sources",
            min_chunk=1,
            unit="file",
            **_cluster_scoring_parallel_kwargs(
                maxtasksperchild=CLUSTER_SCORING_NESTDICT_MAXTASKSPERCHILD,
                initial_worker_cap=(
                    initial_worker_cap if initial_worker_cap is not None else _cluster_scoring_initial_worker_cap()
                ),
                max_worker_cap=(
                    max_worker_cap if max_worker_cap is not None else _cluster_scoring_max_worker_cap()
                ),
            ),
        )
        for idx, loaded, warn in path_entries:
            if warn:
                print(warn)
            path_results[idx] = loaded

        for idx, src in enumerate(iterable):
            if not isinstance(src, str):
                continue
            loaded = path_results.get(idx)
            if not isinstance(loaded, dict):
                continue
            result.update(loaded)
            gc.collect()

    return result

def iter_batches(seq, size):
    for i in range(0, len(seq), size):
        yield i // size + 1, (len(seq) + size - 1) // size, seq[i:i+size]

def load_lclust_for_scoring(arg):
    """
    arg = (akey_expected, lclust_path)
    Returns a tuple keyed by the file path (order-agnostic):
        (lclust_path, loaded_akey, ldict)
    - loaded_akey is derived from the filename stem, e.g. '<akey>.lclust' -> '<akey>'
    - on failure: (lclust_path, None, None)
    """
    akey_expected, lclust_path = arg
    try:
        fname = os.path.basename(lclust_path)
        loaded_akey = os.path.splitext(fname)[0]
        with open(lclust_path, "rb") as fh:
            ldict = pickle.load(fh)
        return (lclust_path, loaded_akey, ldict)
    except Exception as e:
        print(f"[WARN] Could not load {lclust_path}: {e}")
        return (lclust_path, None, None)


def assemble_candidate_output_job(job):
    lib_prefix, outfile, scored_dir, phase_value = job

    if os.path.isfile(outfile):
        try:
            os.remove(outfile)
        except Exception:
            pass

    n_chunks, n_bytes = assemble_candidate_from_chunks(
        scored_dir,
        lib_prefix,
        phase_value,
        outfile,
    )
    exists = os.path.isfile(outfile) and os.path.getsize(outfile) > 0
    md5hex = ""
    if exists:
        _, md5hex = getmd5(outfile, wait_stable=False)
    return {
        "outfile": outfile,
        "lib_prefix": lib_prefix,
        "n_chunks": int(n_chunks),
        "n_bytes": int(n_bytes),
        "exists": bool(exists),
        "md5": str(md5hex or ""),
    }

def getPhasedIndexes(WINDOW_SIZE):
    '''
    generates phased indexes for position labelling
    '''
    ## generate phase position maps for
    ## forward/reverse and for +/- strands
    regs    = list(range(0, WINDOW_SIZE))   ## a template for positions coefficiants - OK
    sens    = [i*int(phase) for i in regs]  ## forward direction to use with sRNAs on 'w' strand; for negative strand subtract - OK
    ## add dicer offsets
    dicer_off_left  = [x-1 for x in sens]
    dicer_off_right = [x+1 for x in sens]
    sens.extend(dicer_off_left)
    sens.extend(dicer_off_right)
    sens.sort()
    asens   = [i-3 for i in sens]           ## forward direction to use with sRNAs on 'c' strand; for negative strand subtract - OK
    return sens,asens

def median(lst):
    n = len(lst)
    if n < 1:
            return None
    if n % 2 == 1:
            return sorted(lst)[n//2]
    else:
            return sum(sorted(lst)[n//2-1:n//2+1])/2.0

def compute_domsize(lenlist,abundict,abuncount):
    '''
    compute the dominant sRNA size in clusters
    '''
    ## combine len, counts and abundance in one
    ## list for sorting on counts and abun
    lencounter   = Counter(lenlist)
    sizeinfo_l   = []
    for alen,acount in lencounter.items():
        aabun    = sum(abundict[alen])
        sizeinfo_l.append((alen,acount,aabun))
    sizeinfo_ls  = sorted(sizeinfo_l, key=lambda x: (-x[1],-x[2]))
    ## dominant size class based on distribution of counts
    if len(lencounter.keys()) <= 4:
        ## just four size classes
        DOMCOUNT_CUT = int(0.25*abuncount)
    else:
        DOMCOUNT_CUT = median(lencounter.values())
    ## dominant size class based on counts - very stringent
    domsize      = sizeinfo_ls[0][0] ## first element of list has most counts, store size
    domcount     = sizeinfo_ls[0][1] ## first element of list has most counts, store count
    domabun      = sizeinfo_ls[0][2] ## first element of list has most counts, store abun
    samecounts_l = [x for x in sizeinfo_ls     if  x[1] == domcount]  ## get size classes that have same counts as domsize
    domsize_l    = [x[0] for x in samecounts_l if  x[2] == domabun]   ## get size classes that have same abundance (alongwith counts) as domsize
    return domsize_l,lencounter,DOMCOUNT_CUT

def clustfilter(tempL1_s, tempL2_s, abundict, uniqcount, tagcount, abuncount):
    """
    filters a lib-chr list of cluster
    Requires >= 10 tag entries in the cluster.
    """
    # Minimum evidence: at least 10 tag records in this cluster
    if len(tempL1_s) < 10:
        return False

    # Basic sanity checks to avoid division by zero
    if tagcount <= 0 or abuncount <= 0:
        return False

    ph = int(phase)

    # Compute dominant size metrics
    domsize_l, lencounter, DOMCOUNT_CUT = compute_domsize(tempL2_s, abundict, abuncount)

    # Uniqueness ratio and phase-length abundance ratio
    uniqratio = round(uniqcount / tagcount, 5)
    ph_abuns = abundict.get(ph, [])
    domsize_ratio = round((sum(ph_abuns) / abuncount), 5) if abuncount else 0.0

    # Count of reads at the phase length
    phaslen_counts = lencounter.get(ph, 0)

    # Final decision
    if (phaslen_counts > DOMCOUNT_CUT) and (uniqratio >= uniqueRatioCut) and (domsize_ratio >= DOMSIZE_CUT):
        return True
    else:
        # the cluster size class doesn't match the phase or it has lots of multihit sRNAs
        return False

def mapPhaseSites(posinfo, poslist, direction, sens):
    """
    Label positions as phased ('p'), non-phased within window ('np'),
    or outside window ('na') relative to a reference position/strand.

    Notes:
      - Uses set membership for speed.
      - Ignores counts/abundances in poslist for labeling (only pos & strand).
      - Expects `sens` = list of integer offsets (includes wobble if desired).
      - Returns tuples: (apos, astrand, poscount, abun_all, abun_phase, label)
    """
    list_labeled = []
    refpos, refstrand = posinfo[0], posinfo[1]

    if direction == "F":
        if refstrand == "w":
            sens_p_set  = {refpos + o for o in sens}          # 'w' strand
            asens_p_set = {refpos + o - 3 for o in sens}      # 'c' strand (−3 shift)
            window_end  = max(sens_p_set)                     # 5'-most end
            for bent in poslist:
                bpos, bstrand = bent[0], bent[1]
                if (bstrand == "w" and bpos in sens_p_set) or (bstrand == "c" and bpos in asens_p_set):
                    label = "p"
                elif bpos <= window_end:
                    label = "np"
                else:
                    label = "na"
                list_labeled.append((*bent, label))
        else:  # refstrand == "c"
            sens_p_set  = {refpos + o for o in sens}          # 'c' strand
            asens_p_set = {refpos + o + 3 for o in sens}      # 'w' strand (+3 shift)
            window_end  = max(asens_p_set)                    # 5'-most end
            for bent in poslist:
                bpos, bstrand = bent[0], bent[1]
                if (bstrand == "w" and bpos in asens_p_set) or (bstrand == "c" and bpos in sens_p_set):
                    label = "p"
                elif bpos <= window_end:
                    label = "np"
                else:
                    label = "na"
                list_labeled.append((*bent, label))

    elif direction == "R":
        # Reverse direction (3' -> 5'): mirror offsets; order not needed for sets.
        if refstrand == "w":
            sens_p_set  = {refpos - o for o in sens}          # 'w' strand
            asens_p_set = {p - 3 for p in sens_p_set}         # 'c' strand (−3 shift)
            window_end  = min(asens_p_set)                    # 3'-most end
            for bent in poslist:
                bpos, bstrand = bent[0], bent[1]
                if (bstrand == "w" and bpos in sens_p_set) or (bstrand == "c" and bpos in asens_p_set):
                    label = "p"
                elif bpos >= window_end:
                    label = "np"
                else:
                    label = "na"
                list_labeled.append((*bent, label))
        else:  # refstrand == "c"
            sens_p_set  = {refpos - o for o in sens}          # 'c' strand
            asens_p_set = {p + 3 for p in sens_p_set}         # 'w' strand (+3 shift)
            window_end  = min(sens_p_set)                     # 3'-most end
            for bent in poslist:
                bpos, bstrand = bent[0], bent[1]
                if (bstrand == "w" and bpos in asens_p_set) or (bstrand == "c" and bpos in sens_p_set):
                    label = "p"
                elif bpos >= window_end:
                    label = "np"
                else:
                    label = "na"
                list_labeled.append((*bent, label))
    else:
        print(f"Unexpected input for scoring direction:{direction}")
        print("PHASIS will exit now, please contact authors")
        sys.exit()

    return list_labeled

def collectstats(labeled_list):
    '''
    takes a labelled list i.e. with 'p' and 'np'
    tags for a cluster and specific to a reference position;
    input list has follwing elements:
    apos, astrand, poscount(number of sRNAs at one positions),
    posabun_a_strand(abundance from all sRNAs),posabun_p_strand (abundance of sRNAs mathcing phase) and label (p and np)
    '''
    test_values_list = []
    count_p  = len([i for i in labeled_list if i[-1] == "p"])
    count_np = len([i for i in labeled_list if i[-1] == "np"])
    count_all = count_p + count_np
    test_values_list.extend((count_p, count_np, count_all))
    ## WHAT STATS YOU NEED FOR RANK SUM?
    ## Mann Whitney test requires two lists with values
    ## Here we choose all abundances from non-phased and
    ## all abundances from  phased positions; we could have
    ## also selected the abundnaces from phased positions
    ## correposnding to sRNAs that match phase length (since
    ## it's already included in the input list) but anyhow
    ## we are going to put a filter on the ratio of abundanced from
    ## phased position for phase len against abundances from non-phased
    ## postions.
    abun_p_all   =  [int(i[3]) for i in labeled_list if i[-1] == "p"] ## selecting abundances for all sizes at phased psoitions
    abun_np_all  =  [int(i[3]) for i in labeled_list if i[-1] == "np"] ## selecting abundances for all sizes from non-phased positions
    abun_p_phase =  [int(i[4]) for i in labeled_list if i[-1] == "p"]
    test_values_list.extend((abun_p_all,abun_np_all,abun_p_phase))
    # ## compute phased sRNAs (of phaselen) vs. all ratio i.e.
    # ## ph_size_prop using nesteddict fom clusterassemble
    # pos_ph_size_prop = round(sum(abun_p_phase)/(sum(abun_p_all)+sum(abun_np_all)),5)
    # test_values_list.extend(pos_ph_size_prop)
    return test_values_list

def hypertest(test_values_list):
    '''
    input: N,K,n,k
    Computes probability from hypergeometric disribution
    https://alexlenail.medium.com/understanding-and-implementing-the-hypergeometric-test-in-python-a7db688a7458
    '''
    M       = WINDOW_SIZE*int(phase)*2  ## count of all the positions from both strands in a window i.e. 'N'
    n       = WINDOW_SIZE*2*3           ## count of expected phased position from both strands, including +1 and -1
                                        ## dicer offsets in a window i.e 'K' i.e. max successes
    N       = test_values_list[2]       ## count of all sRNAs filled positions (count_all) i.e. 'n'
    X       = test_values_list[0]       ## count of positions labelled as phased(count_p) i.e. 'k' the observed successes
    #print("Hyper pval variables - N:%s | K:%s | n:%s | k:%s" % (M,n,N,X))
    #For the hypergeomtric test the count of all sRNA filled positions (N) cannot be higher
    #than the possible number of positions from both strands in a window (M). It is biologically possible
    #because many smallRNAs fragments can overlap, but doing so they are not occuping more positions that the should
    #that why if N is bigger than N the value of N will be automatilly set to M
    hyperp  = round(hypergeom.sf(X-1, M, n, N),10)
    return hyperp,N,X

def ranksumtest(test_values_list):
    '''
    computes probability from ranksum test
    https://stats.stackexchange.com/questions/299733/how-to-interpret-wilcoxon-rank-sum-result
    Mann-Whitney with tie and continuity correction is better option but can be applied only when
    both populations have more than 20 postitiob-speciifc abundances
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.mannwhitneyu.html#scipy.stats.mannwhitneyu
    https://www.youtube.com/watch?v=BT1FKd1Qzjw&t=0s
    Can' mix both tests for same cluster so Wilcoxon rank sum used
    '''
    abun_p_all  = test_values_list[3]
    abun_np_all = test_values_list[4]
    if not abun_p_all:
        abun_p_all = [0]
    if not abun_np_all:
        abun_np_all =  [0]
    ## Sanity check
    aset = set(abun_p_all+abun_np_all)
    if len(aset) == 1:
        rankp           = 1.0 
    else:
        mannu_res       = mannwhitneyu(abun_p_all, abun_np_all, alternative='greater') ## testing if x ranks are greater then y ranks
        rankp           = round(mannu_res[1],10)
    return rankp

def stouffer(pvals):
    '''
    combine pvals using stouffer's method
    '''
    apval = combine_pvalues(pvals, method='stouffer', weights=None)
    return apval[1]

def compute_p_vals(test_values_list_f,test_values_list_r):
    '''
    takes a list of values to compute p_values from forward and reverse direction
    of a cluster
    '''
    ## apply statistical test - hyp and ranksum
    ## to compute a pval for each position in cluster
    ## do this in both forward and reverse direction,
    ## for the latter used sens and asen in negative
    pval_h_f,N_f,X_f     = hypertest(test_values_list_f)
    pval_r_f             = ranksumtest(test_values_list_f)
    pval_corr_f          = round(stouffer([pval_h_f,pval_r_f]),15)
    pval_h_r,N_r,X_r     = hypertest(test_values_list_r)
    pval_r_r             = ranksumtest(test_values_list_r)
    pval_corr_r          = round(stouffer([pval_h_r,pval_r_r]),15)
    return pval_h_f,N_f,X_f,pval_r_f,pval_corr_f,pval_h_r,N_r,X_r,pval_r_r,pval_corr_r

def clustscore(aclust, poslist, sens, asens=None):
    """
    Scores an individual cluster and appends p-values.
    `asens` kept for signature compatibility but not used.
    """
    scoredclust = []
    cluster_pvals_f = []
    cluster_pvals_r = []
    pos_dict = {}

    for ind in range(0, len(poslist)):
        posinfo = poslist[ind]
        flist   = poslist[ind:]     # forward slice
        rlist   = poslist[:ind+1]   # reverse slice
        apos    = posinfo[0]

        # Fast labeling: set membership; no use of counts/abun for label
        flist_labeled = mapPhaseSites(posinfo, flist, "F", sens)
        rlist_labeled = mapPhaseSites(posinfo, rlist, "R", sens)

        test_values_list_f = collectstats(flist_labeled)
        test_values_list_r = collectstats(rlist_labeled)

        pval_h_f, N_f, X_f, pval_r_f, pval_corr_f, pval_h_r, N_r, X_r, pval_r_r, pval_corr_r = \
            compute_p_vals(test_values_list_f, test_values_list_r)

        pos_dict[int(apos)] = (pval_h_f, N_f, X_f, pval_r_f, pval_corr_f,
                               pval_h_r, N_r, X_r, pval_r_r, pval_corr_r)
        cluster_pvals_f.append(pval_corr_f)
        cluster_pvals_r.append(pval_corr_r)

    for taginfo in aclust:
        apos = int(taginfo[5])
        astats = pos_dict[apos]
        taginfo.extend(astats)
        scoredclust.append(taginfo)

    return scoredclust

def clustwrite(akey, clustlist, scored_clust_folder=None):
    '''
    writes all clusters for a fragment
    '''
    #print("writting clusters")
    target_folder = scored_clust_folder if scored_clust_folder else scoredClustFolder
    outfile = "%s/%s.sRNA_%s.cluster" % (target_folder,akey,phase)
    fh_out  = open(outfile,'a')
    for aclust in clustlist:
        #print(f"aclust is {aclust}")
        aid,taglist = aclust
        fh_out.write(">cluster = %s_%s_%s\n" % (akey,aid, taglist[0][0]))
        for taginfo in taglist:
            achr, astrand , ahits, atag, aname, apos, alen, abun, pval_h_f, N_f, X_f, pval_r_f, pval_corr_f, pval_h_r, N_r, X_r, pval_r_r,pval_corr_r = taginfo ## N and X stats are coming from hypergeometric test
            fh_out.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (achr,astrand,apos,alen,ahits,abun,pval_h_f,N_f,X_f,pval_r_f,pval_corr_f,pval_h_r,N_r,X_r,pval_r_r,pval_corr_r,aname,atag))
    fh_out.close()
    return None

def clustassemble(aninput):
    """
    gather full info for clusters i.e. all tags for cluster positions
    and write full clusters to a file
    """
    sync_from_runtime()
    akey, lclustdict, nesteddict, sens, asens, scored_clust_folder = aninput
    nesteddict = nesteddict[0]
    clustlist  = []
    phasedlist = []

    for aid, aclust in lclustdict.items():
        tempL1 = []   # full cluster tag records
        tempL2 = []   # tag lengths
        tempL3 = []   # (apos, strand, poscount, abun_all, abun_phase)

        tagcount = 0
        uniqcount = 0
        abuncount = 0
        abundict = defaultdict(list)

        for apos in aclust:
            tagslist = nesteddict[apos]
            poscount_w = poscount_c = 0
            posabun_a_w = posabun_p_w = 0
            posabun_a_c = posabun_p_c = 0

            for taginfo in tagslist:
                taghits  = int(taginfo[2])
                taglen   = int(taginfo[6])
                tagabun  = int(taginfo[7])
                tagstrand= taginfo[1]

                tempL1.append(taginfo)
                tempL2.append(taglen)

                abuncount += tagabun
                tagcount  += 1
                uniqcount += 1 if taghits <= UNIQRATIO_HIT else 0
                abundict[taglen].append(tagabun)

                if tagstrand == "w":
                    poscount_w  += 1
                    posabun_a_w += tagabun
                    posabun_p_w += tagabun if taglen == int(phase) else 0
                elif tagstrand == "c":
                    poscount_c  += 1
                    posabun_a_c += tagabun
                    posabun_p_c += tagabun if taglen == int(phase) else 0

            if poscount_w > 0:
                tempL3.append((apos, "w", poscount_w, posabun_a_w, posabun_p_w))
            if poscount_c > 0:
                tempL3.append((apos, "c", poscount_c, posabun_a_c, posabun_p_c))

        tempL1_s = sorted(tempL1, key=lambda x: x[5])
        tempL2_s = sorted(tempL2)
        tempL3_s = sorted(tempL3, key=lambda x: x[0])

        signal = clustfilter(tempL1_s, tempL2_s, abundict, uniqcount, tagcount, abuncount)
        if signal:
            # `asens` is accepted but not used by clustscore
            scoredclust = clustscore(tempL1_s, tempL3_s, sens, asens)
            clustlist.append((aid, scoredclust))

    clustwrite(akey, clustlist, scored_clust_folder)
    return None

def scoringprocess(
    libs,
    libs_clustdicts,
    libs_nestdict,
    clustfolder,
    force_rescore=False,
    verify_outputs=True,
    scored_dir=None,
    purge_existing=False,
    # Back-compat: discover concat from global if not given
    concat_mode=None,
    merged_name="ALL_LIBS",
):
    """
    Cluster Scorer with concat-aware akey handling and robust mem/md5 bookkeeping.
    Produces *.cluster chunks (in <phase>_scoredClusters) and assembles
    <lib>.<phase>-PHAS.candidate.clusters outputs.

    Returns: list[str] of assembled cluster file paths.
    """
    sync_from_runtime()
    print("#### Fn: Cluster Scorer ######################")

    # -------------------- Normalize inputs --------------------
    # Expect a list like [(akey, lclust_path), ...]
    libchrs_clust_toscore = []
    for tpl in (libs_clustdicts or []):
        if len(tpl) >= 2:
            akey, lclust_path = tpl[0], tpl[1]
            lclust_path = resolve_lclust_path(lclust_path, clustfolder)
            libchrs_clust_toscore.append((akey, lclust_path))
        else:
            print(f"[WARN] Unexpected libs_clustdicts element (len={len(tpl)}): {tpl}")

    if not libchrs_clust_toscore:
        print("[WARN] No .lclust inputs to score; returning empty list.")
        return []

    # -------------------- mem file & sections --------------------
    cache = MemCache.load(memFile)
    config = cache.cfg
    _ensure_cluster_scoring_sections(config)

    sect_clusters = "CLUSTERS"       # outputs: bare filename -> md5
    sect_lclust   = "CLUSTERED"      # inputs: absolute realpath -> md5
    sect_chunks   = "SCORED_CHUNKS"  # chunk *.cluster absolute paths -> md5

    removed = sanitize_mem_md5s(config, (sect_clusters, sect_lclust, sect_chunks))
    if any(removed.values()):
        cache.flush()

    initial_worker_cap = _cluster_scoring_initial_worker_cap()
    max_worker_cap = _cluster_scoring_max_worker_cap()

    # -------------------- Filter: existing .lclust files ----------------------
    existing_inputs, missing_files = [], []
    for akey, lclust_path in libchrs_clust_toscore:
        if not os.path.isfile(lclust_path):
            missing_files.append(lclust_path)
            continue
        existing_inputs.append((akey, lclust_path))

    if missing_files:
        print(f"[WARN] Missing .lclust files for {len(missing_files)}; e.g.: {missing_files[:3]}")

    if not existing_inputs:
        print("[WARN] No valid inputs after filtering; returning empty list.")
        return []

    # -------------------- Collect nest dict availability ----------------------
    # Normal parserprocess output is a list of per-library .dict pickle paths.
    # Use the memory-light batched path for that case and keep the old merged
    # dict path for explicit dict objects or unusual callers.
    batched_nest_sources = _nestdict_pickle_sources(libs_nestdict)
    libchrs_nestdict = None

    if batched_nest_sources is None:
        needed_nest_keys = _cluster_scoring_needed_nest_keys(existing_inputs)
        print(
            f"[scan] Loading parser nestdict sources for {len(needed_nest_keys)} cluster key(s)...",
            flush=True,
        )
        libchrs_nestdict = build_libchrs_nestdict(
            libs_nestdict,
            needed_keys=needed_nest_keys,
            initial_worker_cap=initial_worker_cap,
            max_worker_cap=max_worker_cap,
        )
        print(
            f"[scan] Loaded nestdict entries for {len(libchrs_nestdict)} cluster key(s).",
            flush=True,
        )

        filtered_inputs, missing_akeys = [], []
        for akey, lclust_path in existing_inputs:
            if (akey in libchrs_nestdict) or (canonicalize_akey(akey) in libchrs_nestdict):
                filtered_inputs.append((akey, lclust_path))
            else:
                missing_akeys.append(akey)

        if missing_akeys:
            print(f"[WARN] Missing nestdict for {len(missing_akeys)} akeys; skipped. Example: {missing_akeys[:5]}")

        if not filtered_inputs:
            print("[WARN] No valid inputs after filtering; returning empty list.")
            return []
    else:
        filtered_inputs = existing_inputs
        print(
            f"[scan] Using per-library batched parser loading for "
            f"{len(batched_nest_sources)} source file(s) and {len(filtered_inputs)} cluster key(s).",
            flush=True,
        )

    # -------------------- Concat mode detection/target lib --------------------
    if concat_mode is None:
        # Concat if we detect the merged lib name as a basename, or if only one lib provided
        concat_mode = any(_basename_no_ext(alib) == merged_name for alib in (libs or [])) or (len(libs or []) == 1)

    concat_target_lib = None
    if concat_mode and libs:
        # Prefer the lib whose basename matches merged_name, fall back to first
        for alib in libs:
            if _basename_no_ext(alib) == merged_name:
                concat_target_lib = alib
                break
        if concat_target_lib is None:
            concat_target_lib = libs[0]

    # -------------------- Per-lib expected outfiles ---------------------------
    expected_outfiles = []
    if concat_mode and concat_target_lib:
        bname = _basename_no_ext(concat_target_lib)
        expected_outfiles.append((concat_target_lib, f"{bname}.{phase}-PHAS.candidate.clusters"))
    else:
        for alib in libs:
            bname = _basename_no_ext(alib)
            expected_outfiles.append((alib, f"{bname}.{phase}-PHAS.candidate.clusters"))

    # -------------------- Centralized stage-cache fast path --------------------
    stage_sig, stage_sig_lclust_md5 = _cluster_scoring_stage_signature(
        [p for _, p in filtered_inputs],
        libs_nestdict,
        phase=phase,
        unique_ratio_cut=uniqueRatioCut,
        concat_mode=concat_mode,
        merged_name=merged_name,
        expected_outputs=[outf for _, outf in expected_outfiles],
        initial_worker_cap=initial_worker_cap,
        max_worker_cap=max_worker_cap,
    )

    if stage_sig is not None and not force_rescore and not purge_existing:
        stage_hits = []
        all_stage_hits = True
        for _, outfile in expected_outfiles:
            if cache.hit(CLUSTER_SCORING_SECTION, outfile, stage_sig):
                stage_hits.append(outfile)
            else:
                all_stage_hits = False
                break

        if all_stage_hits:
            compat_dirty = False
            for outfile in stage_hits:
                compat_dirty = _record_compat_cluster_output_md5(config, sect_clusters, outfile) or compat_dirty
            if compat_dirty:
                cache.flush()
            print(f"cluster files are {stage_hits}")
            return stage_hits

    verified_outputs = {}
    if verify_outputs and not force_rescore and expected_outfiles:
        print(
            f"[scan] Preflighting {len(expected_outfiles)} candidate output(s) before scoring...",
            flush=True,
        )
        verified_outputs = _verify_existing_candidate_outputs(
            config,
            expected_outfiles,
            initial_worker_cap=initial_worker_cap,
            max_worker_cap=max_worker_cap,
        )
        verified_count = sum(1 for ok in verified_outputs.values() if ok)
        print(
            f"[scan] Candidate output preflight verified {verified_count}/{len(expected_outfiles)} output(s).",
            flush=True,
        )

    # -------------------- Map akey (basename) -> lib path ---------------------
    akey_to_lib = {}
    if concat_mode and concat_target_lib:
        for akey, _ in filtered_inputs:
            akey_to_lib[os.path.basename(str(akey))] = concat_target_lib
    else:
        # Greedy longest-prefix match against per-lib basename
        lib_prefixes = [(_basename_no_ext(alib), alib) for alib in libs]
        for akey, _ in filtered_inputs:
            akey_base = os.path.basename(str(akey))
            best_lib, best_len = None, -1
            for pref, alib in lib_prefixes:
                if akey_base.startswith(pref) or (pref in akey_base):
                    if len(pref) > best_len:
                        best_lib, best_len = alib, len(pref)
            akey_to_lib[akey_base] = best_lib

    input_manifest_by_path, lclust_md5_updates, changed_inputs_global = _inspect_cluster_scoring_inputs(
        filtered_inputs,
        config,
        sect_lclust,
        initial_worker_cap=initial_worker_cap,
        max_worker_cap=max_worker_cap,
        precomputed_md5_map=stage_sig_lclust_md5,
    )
    all_outputs_verified = bool(expected_outfiles) and all(
        verified_outputs.get(outf, False) for _, outf in expected_outfiles
    )
    if all_outputs_verified and not changed_inputs_global and not force_rescore and not purge_existing:
        print(
            "[scan] Candidate outputs verified and all .lclust inputs are unchanged; "
            "batch scoring work will be skipped.",
            flush=True,
        )

    # -------------------- Scored chunks folder (global for clustwrite) --------
    currdir = os.getcwd()
    base_scored = os.path.join(currdir, f"{phase}_scoredClusters")
    os.makedirs(base_scored, exist_ok=True)

    global scoredClustFolder
    scoredClustFolder = scored_dir if scored_dir else base_scored
    os.makedirs(scoredClustFolder, exist_ok=True)

    if purge_existing and os.path.isdir(scoredClustFolder):
        # Purge only .cluster files; keep the folder
        for fn in os.listdir(scoredClustFolder):
            if fn.endswith(f".sRNA_{phase}.cluster"):
                try:
                    os.remove(os.path.join(scoredClustFolder, fn))
                except Exception:
                    pass

    # -------------------- Batch planning -------------------------------------
    n_data = len(filtered_inputs)
    batch_size = _cluster_scoring_batch_size(n_data, initial_worker_cap)
    batches = list(iter_batches(filtered_inputs, batch_size))
    print(
        f"[scan] Cluster scoring will start near {initial_worker_cap} worker(s), "
        f"can grow to {max_worker_cap}, and will process batches of up to {batch_size} lib-chr inputs.",
        flush=True,
    )
    hash_parallel_kwargs = _cluster_scoring_parallel_kwargs(
        maxtasksperchild=CLUSTER_SCORING_HASH_MAXTASKSPERCHILD,
        initial_worker_cap=initial_worker_cap,
        max_worker_cap=max_worker_cap,
    )
    load_parallel_kwargs = _cluster_scoring_parallel_kwargs(
        maxtasksperchild=CLUSTER_SCORING_LOAD_MAXTASKSPERCHILD,
        initial_worker_cap=initial_worker_cap,
        max_worker_cap=max_worker_cap,
    )
    score_parallel_kwargs = _cluster_scoring_parallel_kwargs(
        maxtasksperchild=CLUSTER_SCORING_SCORE_MAXTASKSPERCHILD,
        initial_worker_cap=initial_worker_cap,
        max_worker_cap=max_worker_cap,
    )

    # Precompute phased indexes
    sens, asens = getPhasedIndexes(WINDOW_SIZE)

    # Track md5 updates + libs that need re-assembly
    libs_marked_stale = set()
    if purge_existing or force_rescore:
        libs_marked_stale.update(alib for (alib, _) in expected_outfiles)

    def _process_scoring_batch(batch, nestdict_for_batch, label, progress_desc):
        # 1) Determine affected libs for this batch
        batch_libs = set()
        for akey, _ in batch:
            alib = akey_to_lib.get(os.path.basename(str(akey)))
            if alib is not None:
                batch_libs.add(alib)

        # Batch outfiles (concat: single; normal: those touched)
        if concat_mode and concat_target_lib:
            batch_outfiles = [(concat_target_lib, expected_outfiles[0][1])]
        else:
            filtered = [(alib, outf) for (alib, outf) in expected_outfiles if alib in batch_libs]
            batch_outfiles = filtered or list(expected_outfiles)

        # 2) Output verification (preflighted once per stage)
        batch_outputs_ok = bool(batch_outfiles) and all(
            verified_outputs.get(outf, False) for _, outf in batch_outfiles
        )

        # 3) Input MD5 check (absolute realpaths in [CLUSTERED])
        batch_pairs = []
        for akey, p in batch:
            p_abs = os.path.realpath(p)
            batch_pairs.append((akey, p_abs))

        changed_inputs = set()
        if batch_pairs:
            print(
                f"[scan] {label}: consulting precomputed .lclust manifest for {len(batch_pairs)} input(s)...",
                flush=True,
            )
            for akey, p_abs in batch_pairs:
                _, _, is_changed = input_manifest_by_path.get(p_abs, (akey, "", True))
                if force_rescore or is_changed:
                    changed_inputs.add(akey)

        # If outputs are OK and there are no changed inputs, we can skip scoring
        if batch_outputs_ok and not changed_inputs and not force_rescore and not purge_existing:
            print(
                f"[scan] {label}: candidate outputs verified and inputs unchanged; skipping scoring.",
                flush=True,
            )
            return

        # 4) Load lclust dicts for the batch
        to_process = [(akey, p_abs) for (akey, p_abs) in batch_pairs]
        print(
            f"[scan] {label}: loading {len(to_process)} .lclust file(s)...",
            flush=True,
        )
        loaded = run_parallel_with_progress(
            load_lclust_for_scoring,
            to_process,
            desc=f"Loading .lclust ({progress_desc}, {len(to_process)} akeys)",
            min_chunk=1,
            unit="lib-chr",
            **load_parallel_kwargs,
        )

        # Build lookup tables using canonical keys for robust matching
        by_path = {}
        by_akey = defaultdict(list)
        for path_abs, loaded_akey, ldict in loaded:
            loaded_akey_can = canonicalize_akey(loaded_akey)
            by_path[path_abs] = (loaded_akey_can, ldict)
            if loaded_akey_can is not None:
                by_akey[loaded_akey_can].append((path_abs, ldict))

        # 5) Build rawinputs for scoring (match loader/nest; handle mismatches)
        rawinputs = []
        mismatches = 0
        missing_in_loader = 0
        missing_nest = 0

        for akey_exp, lclust_path in to_process:
            akey_can = canonicalize_akey(akey_exp)
            entry = by_path.get(lclust_path)

            if entry is not None:
                loaded_akey, ldict = entry
                if ldict is None:
                    continue
                if loaded_akey != akey_can:
                    cand = by_akey.get(akey_can, [])
                    if len(cand) == 1:
                        _, ldict2 = cand[0]
                        ldict = ldict2
                        mismatches += 1
                    else:
                        print(f"[WARN] Key mismatch ({akey_can} vs {loaded_akey}); skipping.")
                        mismatches += 1
                        continue
            else:
                cand = by_akey.get(akey_can, [])
                if len(cand) == 1:
                    _, ldict = cand[0]
                    missing_in_loader += 1
                else:
                    print(f"[WARN] Loaded results missing path {lclust_path}")
                    missing_in_loader += 1
                    continue

            # Probe the nest dict using expected key, then canonical
            if akey_exp in nestdict_for_batch:
                nest_key = akey_exp
            elif akey_can in nestdict_for_batch:
                nest_key = akey_can
            else:
                missing_nest += 1
                continue

            rawinputs.append((akey_can, ldict, nestdict_for_batch[nest_key], sens, asens, scoredClustFolder))

        if mismatches or missing_in_loader or missing_nest:
            print(
                f"[INFO] {label}: fixups={mismatches}, "
                f"path_fallbacks={missing_in_loader}, missing_nest={missing_nest}"
            )
            # Any of these conditions means touched libs should be re-assembled
            libs_marked_stale.update(batch_libs)

        # 6) Score clusters -> *.cluster chunks
        if rawinputs:
            print(
                f"[scan] {label}: scoring {len(rawinputs)} lib-chr cluster set(s)...",
                flush=True,
            )
            run_parallel_with_progress(
                clustassemble,
                rawinputs,
                desc=f"Scoring clusters ({progress_desc})",
                min_chunk=1,
                unit="lib-chr",
                **score_parallel_kwargs,
            )
            libs_marked_stale.update(batch_libs)
        else:
            print(f"[INFO] {label}: nothing to score after loader/nest checks.")

        # Free batch data
        del loaded, by_path, by_akey, rawinputs
        gc.collect()

    # -------------------- Process batches ------------------------------------
    skip_all_scoring = all_outputs_verified and not changed_inputs_global and not force_rescore and not purge_existing

    if skip_all_scoring:
        print("[scan] Skipping cluster scoring batches after cache/output preflight.", flush=True)
    elif batched_nest_sources is None:
        for b_idx, b_tot, batch in batches:
            _process_scoring_batch(
                batch,
                libchrs_nestdict,
                f"Batch {b_idx}/{b_tot}",
                f"batch {b_idx}/{b_tot}",
            )
    else:
        unresolved_keys = {canonicalize_akey(akey) for akey, _ in filtered_inputs}
        scored_input_count = 0
        for src_idx, src in enumerate(batched_nest_sources, 1):
            label = _nestdict_source_label(src)
            print(
                f"[scan] Loading nestdict for library {label} "
                f"({src_idx}/{len(batched_nest_sources)})...",
                flush=True,
            )
            nestdict_for_source = _load_nestdict_source_direct(src)
            if not isinstance(nestdict_for_source, dict):
                continue

            source_inputs = _select_inputs_for_nestdict_source(
                filtered_inputs,
                nestdict_for_source,
                unresolved_keys,
            )
            print(
                f"[scan] Loaded nestdict for library {label}; matched "
                f"{len(source_inputs)} cluster key(s) for scoring.",
                flush=True,
            )

            if not source_inputs:
                del nestdict_for_source
                gc.collect()
                continue

            scored_input_count += len(source_inputs)
            source_batch_size = _cluster_scoring_batch_size(len(source_inputs), initial_worker_cap)
            for b_idx, b_tot, batch in iter_batches(source_inputs, source_batch_size):
                _process_scoring_batch(
                    batch,
                    nestdict_for_source,
                    f"Library {label} batch {b_idx}/{b_tot}",
                    f"{label} batch {b_idx}/{b_tot}",
                )

            del nestdict_for_source
            gc.collect()

        if unresolved_keys:
            examples = sorted(unresolved_keys)[:5]
            print(
                f"[WARN] Missing nestdict for {len(unresolved_keys)} akeys; skipped. "
                f"Example: {examples}",
                flush=True,
            )
        if scored_input_count == 0:
            print("[WARN] No valid inputs after batched nestdict loading; returning empty list.")
            return []

    # -------------------- Assemble per-lib outputs ----------------------------
    clusterFiles = []
    regenerated = []
    reused_outputs = 0


    if libs:
        rebuild_jobs = []
        for alib, outfile in expected_outfiles:
            lib_prefix = _basename_no_ext(alib)
            must_rebuild = (
                force_rescore or purge_existing or
                (alib in libs_marked_stale) or
                candidate_output_needs_rebuild(outfile)
            )

            if not must_rebuild:
                clusterFiles.append(outfile)
                reused_outputs += 1
                continue

            rebuild_jobs.append((lib_prefix, outfile, scoredClustFolder, phase))

        if rebuild_jobs:
            print(
                f"[scan] Rebuilding {len(rebuild_jobs)} candidate output(s) and reusing {reused_outputs}.",
                flush=True,
            )
            assembly_results = run_parallel_with_progress(
                assemble_candidate_output_job,
                rebuild_jobs,
                desc="Assembling candidate outputs",
                min_chunk=1,
                unit="lib",
                **_cluster_scoring_parallel_kwargs(
                    maxtasksperchild=CLUSTER_SCORING_ASSEMBLY_MAXTASKSPERCHILD,
                    initial_worker_cap=initial_worker_cap,
                    max_worker_cap=max_worker_cap,
                ),
            )
            assembly_errors = [r for r in assembly_results if isinstance(r, RuntimeError)]
            if assembly_errors:
                raise assembly_errors[0]
            for result in assembly_results:
                outfile = result["outfile"]
                lib_prefix = result["lib_prefix"]
                if result["exists"]:
                    clusterFiles.append(outfile)
                    regenerated.append(outfile)
                    md5hex = result.get("md5", "")
                    if md5hex:
                        config[sect_clusters][os.path.basename(outfile)] = md5hex
                else:
                    print(
                        f"[WARN] No non-empty chunks aggregated for {lib_prefix}; {os.path.basename(outfile)} is empty."
                    )
                    if os.path.isfile(outfile):
                        clusterFiles.append(outfile)
            print(
                f"[scan] Candidate output assembly rebuilt {len(regenerated)} output(s) and reused {reused_outputs}.",
                flush=True,
            )
        else:
            print(
                f"[scan] Candidate output assembly reused all {reused_outputs} output(s).",
                flush=True,
            )
    else:
        print("[WARN] No libraries passed for output assembly step.")

    # -------------------- Hash regenerated outputs -> [CLUSTERS] --------------
    # -------------------- Hash chunk files -> [SCORED_CHUNKS] -----------------
    if os.path.isdir(scoredClustFolder):
        chunk_paths = [
            os.path.realpath(os.path.join(scoredClustFolder, f))
            for f in os.listdir(scoredClustFolder)
            if f.endswith(f".sRNA_{phase}.cluster")
        ]
        if chunk_paths:
            md5_chunks = run_parallel_with_progress(
                md5_file_worker,
                chunk_paths,
                desc="Hashing .cluster chunks",
                min_chunk=1,
                unit="file",
                **hash_parallel_kwargs,
            )
            for p_abs, md5 in md5_chunks:
                if md5:
                    config[sect_chunks][p_abs] = md5

    # -------------------- Persist input md5 -> [CLUSTERED] --------------------
    for p_abs, md5 in lclust_md5_updates.items():
        if md5 is not None:
            config[sect_lclust][p_abs] = md5

    for outfile in clusterFiles:
        if stage_sig is not None and os.path.isfile(outfile):
            cache.record(CLUSTER_SCORING_SECTION, outfile, stage_sig)

    cache.flush()

    print(f"cluster files are {clusterFiles}")
    return clusterFiles

__all__ = [
    "scoringprocess",
    "clustassemble",
    "sync_from_runtime",
]
