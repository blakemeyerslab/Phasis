import os
import subprocess
import sys
import time
import gzip
import shutil

from phasis import runtime as rt
from phasis.cache import MEM_FILE_DEFAULT, MemCache, compute_md5_str, stage_signature
from phasis.parallel import PPBalance, optimize, run_parallel_with_progress


# Stage-local globals (populated by sync_from_runtime)
mismat = None
maxhits = None
clustbuffer = None
phase = None
runtype = None
outdir = None
memFile = MEM_FILE_DEFAULT

MAPPING_SECTION = "MAPPING"
MAPPING_IO_MAXTASKSPERCHILD = 64
MAPPING_PPBALANCE_MAXTASKSPERCHILD = 8


def resolve_plain_or_gz_path(path):
    """
    Ensure the plain file at `path` exists if either:
      - `path` exists, or
      - `path + ".gz"` exists

    Returns the plain path (`path`) when resolved.
    Returns None if neither plain nor gzipped form exists.
    """
    if os.path.isfile(path):
        return path

    gz_path = f"{path}.gz"
    if not os.path.isfile(gz_path):
        return None

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

    return path


def canonical_fas_artifact(path):
    gz_path = f"{path}.gz"
    if os.path.isfile(gz_path):
        return gz_path
    if os.path.isfile(path):
        return path
    return None


def archive_fas_to_gz(path):
    """
    Compress a plain .fas into deterministic .fas.gz bytes and remove the
    plain file. Returns the archived path when available, else None.
    """
    gz_path = f"{path}.gz"
    if not os.path.isfile(path):
        if os.path.isfile(gz_path):
            return gz_path
        return None

    tmp_path = f"{gz_path}.tmp"

    try:
        with open(path, "rb") as src, open(tmp_path, "wb") as raw_dst:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_dst, mtime=0) as dst:
                shutil.copyfileobj(src, dst)
        os.replace(tmp_path, gz_path)
        os.remove(path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return gz_path


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
    return path, resolve_plain_or_gz_path(path)


def _archive_fas_job(path):
    return path, archive_fas_to_gz(path)


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
    for path, resolved in results:
        resolved_by_path[path] = resolved
    return resolved_by_path


def _parallel_archive_fas(paths, *, desc):
    unique_paths = _unique_paths(paths)
    if not unique_paths:
        return {}

    results = run_parallel_with_progress(
        _archive_fas_job,
        unique_paths,
        desc=desc,
        unit="lib",
        maxtasksperchild=MAPPING_IO_MAXTASKSPERCHILD,
    )
    runtime_errors = [res for res in results if isinstance(res, RuntimeError)]
    if runtime_errors:
        sys.exit("One or more FASTA archive jobs failed; see errors above.")

    archived_by_path = {}
    for path, artifact_path in results:
        archived_by_path[path] = artifact_path
    return archived_by_path


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
        if outdir:
            memFile = os.path.join(outdir, MEM_FILE_DEFAULT)
        else:
            memFile = MEM_FILE_DEFAULT
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

    for fas_path, bam_path in zip(fas_inputs, bam_expected):
        artifact_path = canonical_fas_artifact(fas_path)
        compat_dirty = _record_compat_fasta_md5(config, artifact_path) or compat_dirty
        input_sig = _mapping_input_signature(fas_path, genoIndex)

        if cache.hit(MAPPING_SECTION, bam_path, input_sig):
            print(f"Cache hit for mapped library: {fas_path}")
            compat_dirty = _record_compat_map_md5(config, bam_path) or compat_dirty
            continue

        if _legacy_map_cache_hit(config, bam_path):
            print(f"Legacy cache matches for mapped library: {fas_path}")
            cache.record(MAPPING_SECTION, bam_path, input_sig)
            compat_dirty = _record_compat_map_md5(config, bam_path) or compat_dirty
            continue

        legacy_sam_path = _legacy_sam_output_for_fas(fas_path)
        reference_path = getattr(rt, "reference", None)
        if reference_path and os.path.isfile(reference_path) and _legacy_sam_ready_for_upgrade(config, artifact_path, legacy_sam_path):
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

        resolved_inputs = _parallel_resolve_plain_or_gz(
            [fas_path for fas_path, _, _ in libs_to_map],
            desc="Preparing FASTAs for mapping",
        )
        materialized_inputs = []
        for fas_path, bam_path, input_sig in libs_to_map:
            resolved = resolved_inputs.get(fas_path)
            if not resolved or not os.path.isfile(resolved):
                print(f"Error: input FASTA missing for mapping: {fas_path}")
                sys.exit()
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
    else:
        if not legacy_sams_to_upgrade:
            print("\nNo new libraries to map this time")

    archived_fastas = _parallel_archive_fas(fas_inputs, desc="Archiving mapped FASTAs")
    for fas_path in fas_inputs:
        artifact_path = archived_fastas.get(fas_path)
        compat_dirty = _record_compat_fasta_md5(config, artifact_path) or compat_dirty

    if compat_dirty:
        cache.flush()

    return libs_mapped
