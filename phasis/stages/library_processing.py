import os
import sys
import gzip
import shutil
import hashlib

from phasis import libprep
from phasis import runtime as rt
from phasis.cache import MEM_FILE_DEFAULT, MemCache, compute_md5_str, stage_signature
from phasis.parallel import run_parallel_with_progress

# Stage-local globals (only what libraryprocess needs)
mindepth = None
libformat = None
concat_libs = None
outdir = None
memFile = MEM_FILE_DEFAULT

LIBRARY_PROCESSING_SECTION = "LIBRARY_PROCESSING"


def _runtime_errors(results):
    return [res for res in results if isinstance(res, RuntimeError)]


def _valid_bool_results(results):
    return [bool(res) for res in results if not isinstance(res, RuntimeError)]


def _existing_path_results(results):
    out = []
    for res in results:
        if isinstance(res, RuntimeError):
            continue
        if isinstance(res, (str, bytes, os.PathLike)) and os.path.exists(res):
            out.append(res)
    return out


def _input_stem(alib):
    base = os.path.basename(str(alib))
    if base.lower().endswith(".gz"):
        base = base[:-3]
    for ext in (".fastq", ".fq", ".fasta", ".fa", ".tag"):
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    stem, dot, _ = base.rpartition(".")
    return stem if dot else base


def _processed_libraries_root():
    run_dir = getattr(rt, "run_dir", None) or os.getcwd()
    root = os.path.join(os.path.abspath(os.path.expanduser(str(run_dir))), "processed_libraries")
    os.makedirs(root, exist_ok=True)
    return root


def _processed_subdir_for_input(alib):
    src = os.path.abspath(os.path.expanduser(str(alib)))
    digest = hashlib.blake2s(src.encode("utf-8"), digest_size=6).hexdigest()
    outdir = os.path.join(_processed_libraries_root(), f"src_{digest}")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _fas_output_for_input(alib):
    return os.path.join(_processed_subdir_for_input(alib), f"{_input_stem(alib)}.fas")


def _sum_output_for_fas(fas_path):
    return f"{fas_path.rpartition('.')[0]}.sum"

def _materialize_fas_from_gz_if_needed(fas_path):
    """
    Ensure `fas_path` exists in plain form if either:
      - `fas_path` already exists, or
      - `fas_path + ".gz"` exists

    Returns the plain `fas_path` when available/resolved.
    Returns None if neither form exists.

    Conservative behavior:
      - plain .fas wins if already present
      - .fas.gz is only inflated when plain .fas is absent
      - uses tmp + os.replace for atomic finalization
    """
    if os.path.isfile(fas_path):
        return fas_path

    gz_path = f"{fas_path}.gz"
    if not os.path.isfile(gz_path):
        return None

    tmp_path = f"{fas_path}.tmp"

    try:
        with gzip.open(gz_path, "rb") as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(tmp_path, fas_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return fas_path


def _archive_fas_to_gz(fas_path):
    """
    Compress a plain .fas into deterministic .fas.gz bytes and remove the
    plain file. Returns the archived path when available, else None.
    """
    gz_path = f"{fas_path}.gz"
    if not os.path.isfile(fas_path):
        if os.path.isfile(gz_path):
            return gz_path
        return None

    tmp_path = f"{gz_path}.tmp"

    try:
        with open(fas_path, "rb") as src, open(tmp_path, "wb") as raw_dst:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_dst, mtime=0) as dst:
                shutil.copyfileobj(src, dst)
        os.replace(tmp_path, gz_path)
        os.remove(fas_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return gz_path


def _logical_fas_available(fas_path):
    return os.path.isfile(fas_path) or os.path.isfile(f"{fas_path}.gz")

def _processing_mode_name():
    if libformat == "F":
        return "dedup_process"
    if libformat == "Q":
        return "fastq_process"
    return "filter_process"


def _ensure_sections(cfg):
    for sect in ("ADVANCED", "LIBRARIES", "FASTAS", LIBRARY_PROCESSING_SECTION):
        if not cfg.has_section(sect):
            cfg.add_section(sect)


def _library_input_signature(alib):
    return stage_signature(
        files=[alib],
        params={
            "stage": "library_processing",
            "libformat": libformat,
            "mindepth": mindepth,
            "mode": _processing_mode_name(),
        },
    )


def _signature_fas_artifact(fas_path):
    gz_path = f"{fas_path}.gz"
    if os.path.isfile(gz_path):
        return gz_path
    return fas_path


def _merged_input_signature(fas_paths):
    sig_files = [_signature_fas_artifact(p) for p in fas_paths]
    return stage_signature(
        files=sig_files,
        params={
            "stage": "library_processing_merge",
            "mindepth": mindepth,
            "libformat": libformat,
            "concat_libs": bool(concat_libs),
        },
        extra=[os.path.basename(p) for p in fas_paths],
    )


def _legacy_input_cache_hit(cfg, alib, fas_path, cur_input_md5):
    prev_input_md5 = (cfg["LIBRARIES"].get(alib) or "").strip()
    if not prev_input_md5 or not cur_input_md5:
        return False
    if prev_input_md5 != cur_input_md5:
        return False
    if not os.path.isfile(fas_path):
        return False
    return True


def _record_compat_input_md5(cfg, alib, md5hex):
    if not md5hex:
        return False
    prev = (cfg["LIBRARIES"].get(alib) or "").strip()
    if prev == md5hex:
        return False
    cfg["LIBRARIES"][alib] = md5hex
    return True


def _record_compat_fasta_md5(cfg, fas_path):
    if not fas_path.endswith(".fas"):
        return False
    if not os.path.isfile(fas_path):
        return False
    fas_md5 = compute_md5_str(fas_path) or ""
    if not fas_md5:
        return False
    prev = (cfg["FASTAS"].get(fas_path) or "").strip()
    if prev == fas_md5:
        return False
    cfg["FASTAS"][fas_path] = fas_md5
    return True


def _check_input_formats(libs_to_process):
    if libformat == "F":
        check_func = libprep.isfasta
    elif libformat == "Q":
        check_func = libprep.isfastq
    else:
        check_func = libprep.isfiletagcount
    print("Checking format:")
    format_results = run_parallel_with_progress(
        check_func, libs_to_process, desc="Checking format"
    )
    if _runtime_errors(format_results):
        sys.exit("One or more libraries failed format check; see errors above.")
    if not any(_valid_bool_results(format_results)):
        sys.exit("No libraries passed format check.")



def _process_single_library_job(job):
    alib, out_fas = job
    out_sum = _sum_output_for_fas(out_fas)

    if libformat == "F":
        return libprep.dedup_process(alib, out_fas=out_fas, out_sum=out_sum)
    if libformat == "Q":
        return libprep.fastq_process(alib, out_fas=out_fas, out_sum=out_sum)
    return libprep.filter_process(alib, out_fas=out_fas, out_sum=out_sum)


def _process_input_libraries(libs_to_process):
    jobs = [(alib, _fas_output_for_input(alib)) for alib in libs_to_process]
    print("Processing libraries:")
    proc_results = run_parallel_with_progress(
        _process_single_library_job, jobs, desc="Filtering/Converting"
    )
    if _runtime_errors(proc_results):
        sys.exit("One or more libraries failed during filtering/conversion; see errors above.")
    return _existing_path_results(proc_results)


def sync_from_runtime() -> None:
    """
    Populate library-processing stage globals from phasis.runtime.
    Keep minimal and spawn-safe.
    """
    global mindepth, libformat, concat_libs, outdir, memFile

    mindepth = rt.mindepth
    libformat = rt.libformat
    concat_libs = rt.concat_libs
    outdir = rt.outdir

    if outdir:
        outdir_abs = os.path.abspath(os.path.expanduser(outdir))
        if outdir_abs != outdir:
            outdir = outdir_abs
            rt.outdir = outdir_abs
        os.makedirs(outdir, exist_ok=True)

    mem_override = getattr(rt, "memFile", None)
    if mem_override:
        memFile = mem_override
    else:
        if outdir:
            memFile = os.path.join(outdir, MEM_FILE_DEFAULT)
        else:
            memFile = MEM_FILE_DEFAULT
        rt.memFile = memFile



def libraryprocess(libs):
    """
    Stage version of libraryprocess().
    Phase I cache-centralized version:
      - uses MemCache + stage_signature for .fas outputs
      - preserves legacy [LIBRARIES]/[FASTAS] bookkeeping for compatibility
    """
    global mindepth, libformat, concat_libs, memFile

    sync_from_runtime()

    print("#### Fn: Lib Processor #######################")

    cache = MemCache.load(memFile)
    config = cache.cfg
    _ensure_sections(config)

    expected_fas = [_fas_output_for_input(alib) for alib in libs]
    legacy_mindepth_matches = str(mindepth) == str(config["ADVANCED"].get("mindepth", ""))

    libs_to_process = []
    input_md5s = {}
    trusted_fas = set()
    compat_dirty = False

    for alib, fas_path in zip(libs, expected_fas):
        _materialize_fas_from_gz_if_needed(fas_path)
        input_sig = _library_input_signature(alib)
        if cache.hit(LIBRARY_PROCESSING_SECTION, fas_path, input_sig):
            print(f"Cache hit for processed library: {alib}")
            cur_input_md5 = compute_md5_str(alib) or ""
            input_md5s[alib] = cur_input_md5
            trusted_fas.add(fas_path)
            compat_dirty = _record_compat_input_md5(config, alib, cur_input_md5) or compat_dirty
            compat_dirty = _record_compat_fasta_md5(config, fas_path) or compat_dirty
            continue

        cur_input_md5 = compute_md5_str(alib) or ""
        input_md5s[alib] = cur_input_md5

        if legacy_mindepth_matches and _legacy_input_cache_hit(config, alib, fas_path, cur_input_md5):
            print(f"Legacy cache matches for library: {alib}")
            cache.record(LIBRARY_PROCESSING_SECTION, fas_path, input_sig)
            trusted_fas.add(fas_path)
            compat_dirty = _record_compat_fasta_md5(config, fas_path) or compat_dirty
            continue

        if cur_input_md5:
            prev_input_md5 = (config["LIBRARIES"].get(alib) or "").strip()
            if prev_input_md5 and prev_input_md5 != cur_input_md5:
                print(f"MD5 doesn't match (or missing) for library: {alib}")
        libs_to_process.append(alib)

    libs_processed = []

    if libs_to_process:
        print("\nLibraries to be processed: %s" % (", ".join(libs_to_process)))
        _check_input_formats(libs_to_process)
        proc_outputs = _process_input_libraries(libs_to_process)

        libs_all = [p for p in expected_fas if os.path.exists(p)]
        for p in proc_outputs:
            if p not in libs_all:
                libs_all.append(p)
        libs_processed = libs_all
    else:
        print("\nNo new libraries to process this time.")
        libs_processed = [p for p in expected_fas if os.path.exists(p)]

    libs_processed = [p for p in expected_fas if _logical_fas_available(p)]

    processed_now = set(proc_outputs) if libs_to_process else set()

    for alib, fas_path in zip(libs, expected_fas):
        if fas_path in processed_now:
            trusted_fas.add(fas_path)
        if not os.path.isfile(fas_path):
            continue
        input_md5 = input_md5s.get(alib)
        if input_md5 is None:
            input_md5 = compute_md5_str(alib) or ""
            input_md5s[alib] = input_md5
        compat_dirty = _record_compat_input_md5(config, alib, input_md5) or compat_dirty
        compat_dirty = _record_compat_fasta_md5(config, fas_path) or compat_dirty
        if fas_path in trusted_fas:
            cache.record(LIBRARY_PROCESSING_SECTION, fas_path, _library_input_signature(alib))

    if compat_dirty:
        cache.flush()

    for fas_path in expected_fas:
        _archive_fas_to_gz(fas_path)

    if concat_libs:
        if not libs_processed:
            sys.exit("No processed libraries available to concatenate.")
        merged_dir = _processed_libraries_root()
        merged_basename = "ALL_LIBS"
        merged_path = os.path.join(merged_dir, f"{merged_basename}.fas")
        merged_sig = _merged_input_signature(libs_processed)

        merge_inputs = []
        for fas_path in libs_processed:
            resolved = _materialize_fas_from_gz_if_needed(fas_path)
            if resolved and os.path.isfile(resolved):
                merge_inputs.append(resolved)

        _materialize_fas_from_gz_if_needed(merged_path)
        if cache.hit(LIBRARY_PROCESSING_SECTION, merged_path, merged_sig):
            print(f"[concat_libs] Reusing merged library: {merged_path}")
            compat_dirty = _record_compat_fasta_md5(config, merged_path) or compat_dirty
            if compat_dirty:
                cache.flush()
            _archive_fas_to_gz(merged_path)
            for fas_path in libs_processed:
                _archive_fas_to_gz(fas_path)
            return [merged_path]

        merged_path = libprep.merge_processed_fastas(
            fas_paths=merge_inputs,
            out_dir=merged_dir,
            out_basename=merged_basename,
            mindepth=mindepth,
        )
        print(f"[concat_libs] Created merged library: {merged_path}")

        compat_dirty = _record_compat_fasta_md5(config, merged_path) or compat_dirty
        cache.record(LIBRARY_PROCESSING_SECTION, merged_path, merged_sig)
        if compat_dirty:
            cache.flush()
        _archive_fas_to_gz(merged_path)
        for fas_path in libs_processed:
            _archive_fas_to_gz(fas_path)
        return [merged_path]

    return [p for p in expected_fas if _logical_fas_available(p)]
