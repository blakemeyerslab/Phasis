import os
import subprocess
import sys
import time
import gzip
import shutil

from phasis import runtime as rt
from phasis.cache import MemCache, compute_md5_str, default_memfile_path, sig_key, stage_signature
from phasis.parallel import PPBalance, optimize, run_parallel_with_progress


# Stage-local globals (populated by sync_from_runtime)
mismat = None
maxhits = None
clustbuffer = None
phase = None
runtype = None
outdir = None
memFile = default_memfile_path()

MAPPING_SECTION = "MAPPING"
MAPPING_IO_MAXTASKSPERCHILD = 64
MAPPING_PPBALANCE_MAXTASKSPERCHILD = 8


def resolve_plain_or_gz_path(path):
    """
    Ensure the plain file at `path` exists if either:
      - `path` exists, or
      - `path + ".gz"` exists

    Returns (`path`, False) when the plain FASTA already exists.
    Returns (`path`, True) when the plain FASTA had to be materialized from
    the canonical `.fas.gz` artifact.
    Returns (None, False) if neither plain nor gzipped form exists.
    """
    if os.path.isfile(path):
        return path, False

    gz_path = f"{path}.gz"
    if not os.path.isfile(gz_path):
        return None, False

    tmp_path = f"{path}.tmp"

    try:
        with gzip.open(gz_path, "rb") as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return path, True


def canonical_fas_artifact(path):
    gz_path = f"{path}.gz"
    if os.path.isfile(gz_path):
        return gz_path
    if os.path.isfile(path):
        return path
    return None


def _compat_fasta_keys(path):
    path = str(path or "")
    if not path:
        return []
    if path.endswith(".fas.gz"):
        return _unique_paths([path, path[:-3]])
    if path.endswith(".fas"):
        return _unique_paths([f"{path}.gz", path])
    return [path]


def _compat_fasta_fp(cfg, path):
    if not cfg.has_section("FASTAS"):
        return ""
    for key in _compat_fasta_keys(path):
        fp = (cfg["FASTAS"].get(key) or "").strip()
        if fp:
            return fp
    return ""


def _unique_paths(paths):
    seen = set()
    out = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _resolve_plain_or_gz_job(path):
    resolved_path, materialized = resolve_plain_or_gz_path(path)
    return path, resolved_path, materialized


def _parallel_resolve_plain_or_gz(paths, *, desc):
    unique_paths = _unique_paths(paths)
    if not unique_paths:
        return {}

    results = run_parallel_with_progress(
        _resolve_plain_or_gz_job,
        unique_paths,
        desc=desc,
        unit="lib",
        maxtasksperchild=MAPPING_IO_MAXTASKSPERCHILD,
    )
    runtime_errors = [res for res in results if isinstance(res, RuntimeError)]
    if runtime_errors:
        sys.exit("One or more FASTA materialization jobs failed; see errors above.")

    resolved_by_path = {}
    for path, resolved, materialized in results:
        resolved_by_path[path] = {
            "resolved_path": resolved,
            "materialized": bool(materialized),
        }
    return resolved_by_path


def _cleanup_materialized_fas(path):
    if not path:
        return False
    if not os.path.isfile(path):
        return False
    try:
        os.remove(path)
    except OSError:
        return False
    return True


def _cleanup_materialized_fas_job(path):
    return path, _cleanup_materialized_fas(path)


def _parallel_cleanup_materialized_fastas(paths, *, desc):
    unique_paths = _unique_paths(paths)
    if not unique_paths:
        return {}

    results = run_parallel_with_progress(
        _cleanup_materialized_fas_job,
        unique_paths,
        desc=desc,
        unit="lib",
        maxtasksperchild=MAPPING_IO_MAXTASKSPERCHILD,
    )
    runtime_errors = [res for res in results if isinstance(res, RuntimeError)]
    if runtime_errors:
        sys.exit("One or more temporary FASTA cleanup jobs failed; see errors above.")

    cleaned_by_path = {}
    for path, cleaned in results:
        cleaned_by_path[path] = bool(cleaned)
    return cleaned_by_path


def _inspect_mapping_cache_job(job):
    fas_path, bam_path, genoIndex = job
    sync_from_runtime()

    artifact_path = canonical_fas_artifact(fas_path)
    artifact_fp = compute_md5_str(artifact_path) or "" if artifact_path and os.path.isfile(artifact_path) else ""
    bam_fp = compute_md5_str(bam_path) or "" if os.path.isfile(bam_path) else ""

    legacy_sam_path = _legacy_sam_output_for_fas(fas_path)
    legacy_sam_fp = compute_md5_str(legacy_sam_path) or "" if os.path.isfile(legacy_sam_path) else ""

    return {
        "fas_path": fas_path,
        "bam_path": bam_path,
        "artifact_path": artifact_path,
        "artifact_fp": artifact_fp,
        "bam_fp": bam_fp,
        "legacy_sam_path": legacy_sam_path,
        "legacy_sam_fp": legacy_sam_fp,
        "input_sig": _mapping_input_signature(fas_path, genoIndex),
    }


def _parallel_inspect_mapping_cache(jobs, *, desc):
    if not jobs:
        return {}

    results = run_parallel_with_progress(
        _inspect_mapping_cache_job,
        jobs,
        desc=desc,
        unit="lib",
        maxtasksperchild=MAPPING_IO_MAXTASKSPERCHILD,
    )
    runtime_errors = [res for res in results if isinstance(res, RuntimeError)]
    if runtime_errors:
        sys.exit("One or more mapping-cache inspection jobs failed; see errors above.")

    inspection_by_fas = {}
    for entry in results:
        inspection_by_fas[entry["fas_path"]] = entry
    return inspection_by_fas


def _ensure_sections(cfg):
    for sect in (MAPPING_SECTION, "MAPS", "FASTAS"):
        if not cfg.has_section(sect):
            cfg.add_section(sect)


def _bam_output_for_fas(fas_path):
    return f"{fas_path.rpartition('.')[0]}.bam"


def _legacy_sam_output_for_fas(fas_path):
    return f"{fas_path.rpartition('.')[0]}.sam"


def _summary_output_for_fas(fas_path):
    return f"{fas_path.rpartition('.')[0]}.sum"


def _sorted_bam_temp_for_fas(fas_path):
    return f"{fas_path.rpartition('.')[0]}.sorted.bam"


def _temp_sam_for_fas(fas_path):
    return f"{fas_path.rpartition('.')[0]}.temp.sam"


def _index_files_for_signature(genoIndex):
    prefix_dir = os.path.dirname(genoIndex) or os.getcwd()
    prefix_base = os.path.basename(genoIndex)
    out = []

    if not os.path.isdir(prefix_dir):
        return out

    for name in sorted(os.listdir(prefix_dir)):
        if not name.startswith(f"{prefix_base}."):
            continue
        if ".ht2" not in name:
            continue
        out.append(os.path.join(prefix_dir, name))

    return out


def _mapping_input_signature(fas_path, genoIndex):
    artifact_path = canonical_fas_artifact(fas_path) or fas_path
    sig_files = [artifact_path]

    ref_path = getattr(rt, "reference", None)
    if ref_path:
        sig_files.append(ref_path)

    sig_files.extend(_index_files_for_signature(genoIndex))

    return stage_signature(
        files=sig_files,
        params={
            "stage": "mapping",
            "maxhits": maxhits,
            "runtype": runtype,
        },
        extra=[
            os.path.abspath(os.path.expanduser(str(genoIndex))),
        ],
    )


def _record_compat_fasta_md5(cfg, artifact_path):
    if not artifact_path:
        return False
    if not os.path.isfile(artifact_path):
        return False
    md5hex = compute_md5_str(artifact_path) or ""
    if not md5hex:
        return False
    prev = (cfg["FASTAS"].get(artifact_path) or "").strip()
    if prev == md5hex:
        return False
    cfg["FASTAS"][artifact_path] = md5hex
    return True


def _record_compat_fasta_fp(cfg, artifact_path, fphex):
    fphex = str(fphex or "")
    if not artifact_path or not fphex:
        return False
    prev = (cfg["FASTAS"].get(artifact_path) or "").strip()
    if prev == fphex:
        return False
    cfg["FASTAS"][artifact_path] = fphex
    return True


def _record_compat_map_md5(cfg, bam_path):
    if not bam_path:
        return False
    if not os.path.isfile(bam_path):
        return False
    md5hex = compute_md5_str(bam_path) or ""
    if not md5hex:
        return False
    prev = (cfg["MAPS"].get(bam_path) or "").strip()
    if prev == md5hex:
        return False
    cfg["MAPS"][bam_path] = md5hex
    return True


def _record_compat_map_fp(cfg, bam_path, fphex):
    fphex = str(fphex or "")
    if not bam_path or not fphex:
        return False
    prev = (cfg["MAPS"].get(bam_path) or "").strip()
    if prev == fphex:
        return False
    cfg["MAPS"][bam_path] = fphex
    return True


def _legacy_map_cache_hit(cfg, bam_path):
    prev = (cfg["MAPS"].get(bam_path) or "").strip()
    if not prev:
        return False
    if not os.path.isfile(bam_path):
        return False
    cur = compute_md5_str(bam_path) or ""
    if not cur:
        return False
    return prev == cur


def _legacy_sam_ready_for_upgrade(cfg, artifact_path, sam_path):
    if not artifact_path or not os.path.isfile(artifact_path):
        return False
    if not os.path.isfile(sam_path):
        return False

    fas_prev = (cfg["FASTAS"].get(artifact_path) or "").strip()
    fas_cur = compute_md5_str(artifact_path) or ""
    if not fas_prev or not fas_cur or fas_prev != fas_cur:
        return False

    sam_prev = (cfg["MAPS"].get(sam_path) or "").strip()
    sam_cur = compute_md5_str(sam_path) or ""
    if not sam_prev or not sam_cur or sam_prev != sam_cur:
        return False

    return True


def _stabilize_outputs(paths):
    for out_path in paths:
        tries = 0
        last_size = -1
        while tries < 3:
            if os.path.isfile(out_path):
                try:
                    sz = os.path.getsize(out_path)
                except Exception:
                    sz = -1
                if sz > 0 and sz == last_size:
                    break
                last_size = sz
            time.sleep(0.5)
            tries += 1


def sync_from_runtime() -> None:
    """
    Populate mapping-stage globals from phasis.runtime.
    Keep this minimal and spawn-safe.
    """
    global mismat, maxhits, clustbuffer, phase, runtype, outdir, memFile

    mismat = rt.mismat
    maxhits = rt.maxhits
    clustbuffer = rt.clustbuffer
    phase = rt.phase
    runtype = rt.runtype
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
        memFile = default_memfile_path()
        rt.memFile = memFile


def mapper(aninput):
    """
    Map one FASTA with HISAT2, then sort and convert to a BAM that excludes
    unmapped reads. The final BAM preserves the sorted order.
    """
    alib, genoIndex, nspread, maxhits_local, runtype_local = aninput

    asam_temp = _temp_sam_for_fas(alib)
    abam_sorted = _sorted_bam_temp_for_fas(alib)
    abam_final = _bam_output_for_fas(alib)
    asum = _summary_output_for_fas(alib)
    nspread = str(nspread)

    if runtype_local == "G" or runtype_local == "S":
        retcode = subprocess.call(
            [
                "hisat2",
                "--no-softclip",
                "--no-spliced-alignment",
                "-k",
                str(maxhits_local),
                "-p",
                nspread,
                "-x",
                genoIndex,
                "-f",
                alib,
                "-S",
                asam_temp,
                "--summary-file",
                asum,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    elif runtype_local == "T":
        retcode = subprocess.call(
            [
                "hisat2",
                "--no-softclip",
                "--no-spliced-alignment",
                "-k",
                str(maxhits_local),
                "-p",
                nspread,
                "-x",
                genoIndex,
                "-f",
                alib,
                "-S",
                asam_temp,
                "--summary-file",
                asum,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        print("Please input the correct setting for 'runtype'")
        print("Script will exit for now\n")
        sys.exit()

    if retcode != 0:
        print(f"Error: HISAT2 mapping of '{alib}' to reference index failed.")
        sys.exit()

    retcode = subprocess.call(
        ["samtools", "sort", "-@", str(nspread), "-O", "BAM", "-o", abam_sorted, asam_temp],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if retcode != 0:
        print(f"Error: Samtools sorting of '{asam_temp}' failed.")
        sys.exit()

    retcode = subprocess.call(
        ["samtools", "view", "-@", str(nspread), "-b", "-F", "4", "-o", abam_final, abam_sorted],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if retcode != 0:
        print(f"Error: BAM conversion/filtering of '{abam_sorted}' failed.")
        sys.exit()

    for tmp_path in (asam_temp, abam_sorted):
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return abam_final


def _convert_legacy_sam_to_bam(aninput):
    sam_path, bam_path, nspread, reference = aninput
    nspread = str(nspread)

    retcode = subprocess.call(
        ["samtools", "view", "-@", str(nspread), "-b", "-T", reference, "-F", "4", "-o", bam_path, sam_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if retcode != 0:
        print(f"Error: Legacy SAM -> BAM conversion failed for '{sam_path}'.")
        sys.exit()

    if os.path.exists(sam_path):
        try:
            os.remove(sam_path)
        except OSError:
            pass

    return bam_path


def mapprocess(
    libs,
    genoIndex,
    *,
    ncores_local,
):
    """
    Map the libs to the reference index with centralized cache validation.

    INPUT: libs are logical FASTA paths (usually *.fas from libraryprocess;
    merged in --concat_libs). The canonical stored FASTA artifact is .fas.gz
    when available, but HISAT2 still receives a temporary plain .fas.

    OUTPUT: list of mapped BAM files for the libs that required work.

    Guarantees:
      - MAPPING uses MemCache + stage_signature.
      - Legacy [FASTAS]/[MAPS] hashes are still written for compatibility.
      - Final mapping artifacts are *.bam, not *.sam.
      - BAM excludes unmapped reads and preserves sorted order.
    """
    global maxhits, runtype

    sync_from_runtime()

    print("#### Fn: Lib Mapper ##########################")

    cache = MemCache.load(memFile)
    config = cache.cfg
    _ensure_sections(config)

    bases = [alib.rpartition(".")[0] for alib in libs]
    fas_inputs = [f"{b}.fas" for b in bases]
    bam_expected = [_bam_output_for_fas(fas) for fas in fas_inputs]

    libs_to_map = []
    legacy_sams_to_upgrade = []
    compat_dirty = False
    materialized_fastas = []

    cache_jobs = [(fas_path, bam_path, genoIndex) for fas_path, bam_path in zip(fas_inputs, bam_expected)]
    cache_inspection = _parallel_inspect_mapping_cache(
        cache_jobs,
        desc="Inspecting mapping cache",
    )

    for fas_path, bam_path in zip(fas_inputs, bam_expected):
        inspect = cache_inspection.get(fas_path, {})
        artifact_path = inspect.get("artifact_path")
        artifact_fp = inspect.get("artifact_fp", "")
        bam_fp = inspect.get("bam_fp", "")
        legacy_sam_path = inspect.get("legacy_sam_path")
        legacy_sam_fp = inspect.get("legacy_sam_fp", "")
        input_sig = inspect.get("input_sig") or _mapping_input_signature(fas_path, genoIndex)

        compat_dirty = _record_compat_fasta_fp(config, artifact_path, artifact_fp) or compat_dirty

        prev_stage_fp = cache.get(MAPPING_SECTION, bam_path, "") or ""
        prev_stage_sig = cache.get(MAPPING_SECTION, sig_key(bam_path), "") or ""
        if bam_fp and prev_stage_fp == bam_fp and prev_stage_sig == input_sig:
            print(f"Cache hit for mapped library: {fas_path}")
            compat_dirty = _record_compat_map_fp(config, bam_path, bam_fp) or compat_dirty
            continue

        prev_legacy_bam_fp = (config["MAPS"].get(bam_path) or "").strip()
        if bam_fp and prev_legacy_bam_fp and prev_legacy_bam_fp == bam_fp:
            print(f"Legacy cache matches for mapped library: {fas_path}")
            cache.record(
                MAPPING_SECTION,
                bam_path,
                input_sig,
                output_fp=bam_fp,
                wait_stable=False,
            )
            compat_dirty = _record_compat_map_fp(config, bam_path, bam_fp) or compat_dirty
            continue

        reference_path = getattr(rt, "reference", None)
        prev_legacy_artifact_fp = _compat_fasta_fp(config, artifact_path or fas_path)
        prev_legacy_sam_fp = (config["MAPS"].get(legacy_sam_path) or "").strip() if legacy_sam_path else ""
        legacy_upgrade_ready = (
            bool(reference_path and os.path.isfile(reference_path))
            and bool(artifact_path and artifact_fp and prev_legacy_artifact_fp == artifact_fp)
            and bool(legacy_sam_path and legacy_sam_fp and prev_legacy_sam_fp == legacy_sam_fp)
        )
        if legacy_upgrade_ready:
            legacy_sams_to_upgrade.append((fas_path, legacy_sam_path, bam_path, input_sig))
            continue

        libs_to_map.append((fas_path, bam_path, input_sig))

    libs_mapped = []

    if legacy_sams_to_upgrade:
        print(
            "Legacy SAMs to upgrade to BAM: %s"
            % ", ".join([item[1] for item in legacy_sams_to_upgrade])
        )
        nproc, nspread = optimize(ncores_local, len(legacy_sams_to_upgrade))
        rawinputs = [
            (sam_path, bam_path, nspread, getattr(rt, "reference", None))
            for _, sam_path, bam_path, _ in legacy_sams_to_upgrade
        ]
        PPBalance(
            _convert_legacy_sam_to_bam,
            rawinputs,
            n_workers=nproc,
            maxtasksperchild=MAPPING_PPBALANCE_MAXTASKSPERCHILD,
        )
        upgraded_paths = [bam_path for _, _, bam_path, _ in legacy_sams_to_upgrade]
        _stabilize_outputs(upgraded_paths)
        for _, _, bam_path, input_sig in legacy_sams_to_upgrade:
            cache.record(MAPPING_SECTION, bam_path, input_sig)
            compat_dirty = _record_compat_map_md5(config, bam_path) or compat_dirty
            libs_mapped.append(bam_path)

    if libs_to_map:
        print("Libraries to be mapped: %s" % (", ".join([item[0] for item in libs_to_map])))

        try:
            resolved_inputs = _parallel_resolve_plain_or_gz(
                [fas_path for fas_path, _, _ in libs_to_map],
                desc="Materializing FASTAs for mapping",
            )
            materialized_inputs = []
            for fas_path, bam_path, input_sig in libs_to_map:
                resolved_info = resolved_inputs.get(fas_path, {})
                resolved = resolved_info.get("resolved_path")
                if not resolved or not os.path.isfile(resolved):
                    print(f"Error: input FASTA missing for mapping: {fas_path}")
                    sys.exit()
                if resolved_info.get("materialized"):
                    materialized_fastas.append(resolved)
                materialized_inputs.append((resolved, bam_path, input_sig))

            nproc, nspread = optimize(ncores_local, len(materialized_inputs))
            rawinputs = [(alib, genoIndex, nspread, maxhits, runtype) for alib, _, _ in materialized_inputs]
            PPBalance(
                mapper,
                rawinputs,
                n_workers=nproc,
                maxtasksperchild=MAPPING_PPBALANCE_MAXTASKSPERCHILD,
            )

            produced_bams = [bam_path for _, bam_path, _ in materialized_inputs]
            _stabilize_outputs(produced_bams)

            for _, bam_path, input_sig in materialized_inputs:
                cache.record(MAPPING_SECTION, bam_path, input_sig)
                compat_dirty = _record_compat_map_md5(config, bam_path) or compat_dirty
                libs_mapped.append(bam_path)
        finally:
            _parallel_cleanup_materialized_fastas(
                materialized_fastas,
                desc="Cleaning temporary FASTAs",
            )
    else:
        if not legacy_sams_to_upgrade:
            print("\nNo new libraries to map this time")

    if compat_dirty:
        cache.flush()

    return libs_mapped
