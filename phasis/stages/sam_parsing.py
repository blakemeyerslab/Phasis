# phasis/stages/sam_parsing.py
from __future__ import annotations

import os
import gc
import pickle
import configparser
import subprocess
from collections import defaultdict, OrderedDict, Counter
from typing import Iterable, List, Sequence, Tuple

import phasis.runtime as rt
from phasis.parallel import run_parallel_with_progress
from phasis.cache import MEM_FILE_DEFAULT, MemCache, compute_md5_str, getmd5, sig_key, stage_signature


def _resolve_mem_file() -> str:
    mem = getattr(rt, "memFile", None)
    return mem if mem else MEM_FILE_DEFAULT


def updatedsets(config: configparser.ConfigParser) -> List[str]:
    """
    Determine which settings changed since the last run by comparing the mem/settings
    file values to current runtime (rt.*).

    Defensive: if sections/keys are missing or malformed, we skip that check.
    """
    updated: List[str] = []

    # ADVANCED knobs
    try:
        if config.has_section("ADVANCED"):
            if "mismat" in config["ADVANCED"]:
                try:
                    if int(config["ADVANCED"]["mismat"]) != int(getattr(rt, "mismat", 0)):
                        updated.append("mismat")
                except Exception:
                    pass
            if "maxhits" in config["ADVANCED"]:
                try:
                    if int(config["ADVANCED"]["maxhits"]) != int(getattr(rt, "maxhits", 0)):
                        updated.append("maxhits")
                except Exception:
                    pass
            if "clustbuffer" in config["ADVANCED"]:
                try:
                    if int(config["ADVANCED"]["clustbuffer"]) != int(getattr(rt, "clustbuffer", 0)):
                        updated.append("clustbuffer")
                except Exception:
                    pass
    except Exception:
        pass

    # BASIC: phase length (optional; older configs may not have it)
    try:
        if config.has_section("BASIC") and "phaselen" in config["BASIC"]:
            try:
                if int(config["BASIC"]["phaselen"]) != int(getattr(rt, "phase", 0)):
                    updated.append("phaselen")
            except Exception:
                # Some historical configs stored non-int tokens here; ignore.
                pass
    except Exception:
        pass

    return updated


def _cached_file_matches(config: configparser.ConfigParser, section: str, path: str) -> bool:
    """
    Reuse cache only when:
      - section exists
      - key exists with a non-empty stored hash
      - file exists on disk
      - current hash matches stored hash
    """
    if not config.has_section(section):
        return False

    prev = (config[section].get(path) or "").strip()
    if not prev:
        return False

    if not os.path.isfile(path):
        return False

    _, cur = getmd5(path)
    return bool(cur and cur == prev)


def _alignment_path_for_lib(alib: str) -> str:
    base = alib.rpartition(".")[0]
    bam_path = f"{base}.bam"
    if os.path.isfile(bam_path):
        return bam_path
    return f"{base}.sam"


def _iter_alignment_lines(alib: str):
    """
    Yield SAM-format alignment lines from either:
      - a plain SAM file on disk, or
      - a BAM file streamed through `samtools view`
    """
    if alib.endswith(".bam"):
        proc = subprocess.Popen(
            ["samtools", "view", alib],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    yield line
        finally:
            stderr_text = ""
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                stderr_text = proc.stderr.read()
                proc.stderr.close()
            retcode = proc.wait()
            if retcode != 0:
                raise RuntimeError(
                    f"samtools view failed for '{alib}' with exit code {retcode}: {stderr_text.strip()}"
                )
        return

    with open(alib, "r") as fh:
        for line in fh:
            yield line


def libstoset(alist: Iterable[Tuple[str, str]], akey: str) -> None:
    """Write (path, md5) entries to the mem/settings file under section `akey`."""
    mem_file = _resolve_mem_file()

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(mem_file)

    if not config.has_section(akey):
        config.add_section(akey)

    for alib, ahash in alist:
        config[akey][str(alib)] = str(ahash)

    with open(mem_file, "w") as fh_out:
        config.write(fh_out)


SAM_PARSING_SECTION = "SAM_PARSING"


def _ensure_sections(cfg: configparser.ConfigParser) -> None:
    for section in (SAM_PARSING_SECTION, "PARSED", "COUNTERS"):
        if not cfg.has_section(section):
            cfg.add_section(section)


def _parser_output_paths_for_lib(alib: str, phase: str) -> Tuple[str, str]:
    stem = alib.rpartition(".")[0]
    return (f"{stem}_{phase}.dict", f"{stem}_{phase}.count")


def _parser_input_signature(
    alignment_path: str,
    *,
    phase: str,
    maxhits: int,
    mismat: int,
    norm: bool,
    norm_factor: float,
) -> str:
    return stage_signature(
        files=[alignment_path],
        params={
            "stage": "sam_parsing",
            "phase": phase,
            "maxhits": maxhits,
            "mismat": mismat,
            "norm": bool(norm),
            "norm_factor": norm_factor,
        },
    )


def _legacy_parser_cache_hit(
    config: configparser.ConfigParser,
    dict_path: str,
    count_path: str,
) -> bool:
    dict_ok = _cached_file_matches(config, "PARSED", dict_path)
    count_ok = _cached_file_matches(config, "COUNTERS", count_path)
    return bool(dict_ok and count_ok)


def _record_compat_md5(
    config: configparser.ConfigParser,
    section: str,
    path: str,
) -> bool:
    if not path or not os.path.isfile(path):
        return False

    md5hex = compute_md5_str(path) or ""
    if not md5hex:
        return False

    prev = (config[section].get(path) or "").strip()
    if prev == md5hex:
        return False

    config[section][path] = md5hex
    return True


def _record_compat_fp(
    config: configparser.ConfigParser,
    section: str,
    path: str,
    fphex: str,
) -> bool:
    fphex = str(fphex or "")
    if not path or not fphex:
        return False

    prev = (config[section].get(path) or "").strip()
    if prev == fphex:
        return False

    config[section][path] = fphex
    return True


def _inspect_parser_cache_job(job):
    alib, phase, maxhits, mismat, norm, norm_factor = job

    alignment_path = _alignment_path_for_lib(alib)
    dict_path, count_path = _parser_output_paths_for_lib(alib, phase)
    input_sig = _parser_input_signature(
        alignment_path,
        phase=phase,
        maxhits=maxhits,
        mismat=mismat,
        norm=norm,
        norm_factor=norm_factor,
    )

    dict_fp = compute_md5_str(dict_path) or "" if os.path.isfile(dict_path) else ""
    count_fp = compute_md5_str(count_path) or "" if os.path.isfile(count_path) else ""

    return {
        "alib": alib,
        "alignment_path": alignment_path,
        "dict_path": dict_path,
        "count_path": count_path,
        "input_sig": input_sig,
        "dict_fp": dict_fp,
        "count_fp": count_fp,
    }


def _parallel_inspect_parser_cache(jobs, *, desc):
    if not jobs:
        return {}

    results = run_parallel_with_progress(
        _inspect_parser_cache_job,
        jobs,
        desc=desc,
        unit="lib",
    )
    runtime_errors = [res for res in results if isinstance(res, RuntimeError)]
    if runtime_errors:
        raise RuntimeError("One or more parser-cache inspection jobs failed; see errors above.")

    by_lib = {}
    for entry in results:
        by_lib[entry["alib"]] = entry
    return by_lib


def _load_parsed_outputs(
    dict_paths: Sequence[str],
    count_paths: Sequence[str],
    *,
    load_dicts: bool,
):
    if not load_dicts:
        return list(dict_paths), list(count_paths)

    libs_nestdict = []
    libs_poscountdict = []
    for dp, cp in zip(dict_paths, count_paths):
        try:
            with open(dp, "rb") as f1:
                obj = pickle.load(f1)
                if isinstance(obj, dict):
                    libs_nestdict.append(obj)
            with open(cp, "rb") as f2:
                obj = pickle.load(f2)
                if isinstance(obj, dict):
                    libs_poscountdict.append(obj)
        except FileNotFoundError:
            print(f"Warning: Missing parsed file for {dp}")
        gc.collect()

    return libs_nestdict, libs_poscountdict



def parserprocess(libs: Sequence[str], load_dicts: bool = False):
    """
    Parse mapped libraries (BAM or SAM) in parallel.

    Default: return only file paths to avoid RAM blow-ups.
    If load_dicts=True, load them back SEQUENTIALLY (low peak RAM).

    Spawn-safe: reads runtime knobs from rt.* and does not rely on legacy globals.

    Phase I cache-centralized version:
      - uses MemCache + stage_signature for .dict/.count outputs
      - preserves legacy [PARSED]/[COUNTERS] bookkeeping for compatibility
    """
    print("#### Fn: Lib Parser ##########################")

    mem_file = _resolve_mem_file()
    phase = str(getattr(rt, "phase", ""))
    maxhits = int(getattr(rt, "maxhits", 0))
    mismat = int(getattr(rt, "mismat", 0))
    norm = bool(getattr(rt, "norm", False))
    norm_factor = float(getattr(rt, "norm_factor", 0.0))

    cache = MemCache.load(mem_file)
    config = cache.cfg
    _ensure_sections(config)

    updatedsetL = updatedsets(config)
    force_reparse = "mismat" in updatedsetL
    if force_reparse:
        print("Setting update detected for 'mismat' parameter")
    elif config.has_section("PARSED") or config.has_section("COUNTERS"):
        print("Subsequent run for parserprocess; parsing only remapped libraries")

    parse_jobs = []
    dict_paths: List[str] = []
    count_paths: List[str] = []
    compat_dirty = False

    parser_cache_inspection = {}
    if not force_reparse:
        inspect_jobs = [
            (alib, phase, maxhits, mismat, norm, norm_factor)
            for alib in libs
        ]
        parser_cache_inspection = _parallel_inspect_parser_cache(
            inspect_jobs,
            desc="Inspecting parser cache",
        )

    for alib in libs:
        inspect = parser_cache_inspection.get(alib, {})
        alignment_path = inspect.get("alignment_path") or _alignment_path_for_lib(alib)
        dict_path = inspect.get("dict_path")
        count_path = inspect.get("count_path")
        if not dict_path or not count_path:
            dict_path, count_path = _parser_output_paths_for_lib(alib, phase)
        input_sig = inspect.get("input_sig")
        if not input_sig:
            input_sig = _parser_input_signature(
                alignment_path,
                phase=phase,
                maxhits=maxhits,
                mismat=mismat,
                norm=norm,
                norm_factor=norm_factor,
            )
        dict_fp = inspect.get("dict_fp", "")
        count_fp = inspect.get("count_fp", "")

        dict_paths.append(dict_path)
        count_paths.append(count_path)

        if not force_reparse:
            prev_dict_fp = (config[SAM_PARSING_SECTION].get(dict_path) or "").strip()
            prev_count_fp = (config[SAM_PARSING_SECTION].get(count_path) or "").strip()
            prev_dict_sig = (config[SAM_PARSING_SECTION].get(sig_key(dict_path)) or "").strip()
            prev_count_sig = (config[SAM_PARSING_SECTION].get(sig_key(count_path)) or "").strip()
            dict_hit = bool(dict_fp and prev_dict_fp == dict_fp and prev_dict_sig == input_sig)
            count_hit = bool(count_fp and prev_count_fp == count_fp and prev_count_sig == input_sig)
            if dict_hit and count_hit:
                print(f"Cache hit for parsed library: {alib.rpartition('.')[0]}")
                compat_dirty = _record_compat_fp(config, "PARSED", dict_path, dict_fp) or compat_dirty
                compat_dirty = _record_compat_fp(config, "COUNTERS", count_path, count_fp) or compat_dirty
                continue

            legacy_dict_fp = (config["PARSED"].get(dict_path) or "").strip()
            legacy_count_fp = (config["COUNTERS"].get(count_path) or "").strip()
            legacy_hit = bool(
                dict_fp
                and count_fp
                and legacy_dict_fp == dict_fp
                and legacy_count_fp == count_fp
            )
            if legacy_hit:
                print(f"Legacy cache matches for parsed library {alib.rpartition('.')[0]}")
                cache.record(
                    SAM_PARSING_SECTION,
                    dict_path,
                    input_sig,
                    output_fp=dict_fp,
                    wait_stable=False,
                )
                cache.record(
                    SAM_PARSING_SECTION,
                    count_path,
                    input_sig,
                    output_fp=count_fp,
                    wait_stable=False,
                )
                compat_dirty = _record_compat_fp(config, "PARSED", dict_path, dict_fp) or compat_dirty
                compat_dirty = _record_compat_fp(config, "COUNTERS", count_path, count_fp) or compat_dirty
                continue

        print(f"Added {alignment_path} to libs_to_parse")
        parse_jobs.append((alignment_path, dict_path, count_path, input_sig))

    if parse_jobs:
        print(f"Libraries to be parsed: {', '.join(job[0] for job in parse_jobs)}")
        rawinputs = [(alignment_path, maxhits, mismat) for alignment_path, _, _, _ in parse_jobs]

        out_pairs = run_parallel_with_progress(
            samparser_streaming, rawinputs, desc="Parsing alignments", unit="lib"
        )

        sig_by_output = {}
        for _, expected_dp, expected_cp, input_sig in parse_jobs:
            sig_by_output[expected_dp] = input_sig
            sig_by_output[expected_cp] = input_sig

        for dp, cp in out_pairs:
            dict_sig = sig_by_output[dp]
            count_sig = sig_by_output[cp]

            cache.record(SAM_PARSING_SECTION, dp, dict_sig)
            cache.record(SAM_PARSING_SECTION, cp, count_sig)
            compat_dirty = _record_compat_md5(config, "PARSED", dp) or compat_dirty
            compat_dirty = _record_compat_md5(config, "COUNTERS", cp) or compat_dirty

            if dict_sig != count_sig:
                raise RuntimeError(f"Signature mismatch between paired parser outputs: {dp} / {cp}")

    if compat_dirty:
        cache.flush()

    return _load_parsed_outputs(dict_paths, count_paths, load_dicts=load_dicts)


def samparser_streaming(aninput):
    """
    Parse one alignment file -> write:
      - <lib>_<phase>.dict (pickle of nestdict)
      - <lib>_<phase>.count (pickle of poscountdict)
    Return only (outfile1, outfile2) to keep RAM low.
    """
    alib, maxhits, mismat = aninput

    phase = str(getattr(rt, "phase", ""))
    norm = bool(getattr(rt, "norm", False))
    norm_factor = float(getattr(rt, "norm_factor", 0.0))

    outfile1 = f"{alib.rpartition('.')[0]}_{phase}.dict"
    outfile2 = f"{alib.rpartition('.')[0]}_{phase}.count"
    asum = f"{alib.rpartition('.')[0]}.sum"

    total_abund = None
    if norm:
        total_abund = 0
        for line in _iter_alignment_lines(alib):
            if line.startswith("@"):
                continue
            ent = line.rstrip("\n").split("\t")
            aflag = int(ent[1])
            if aflag not in {0, 256, 16, 272}:
                continue
            aname = ent[0].strip()
            aabun = int(aname.split("|")[-1])
            total_abund += aabun

    tempdict1 = defaultdict(list)
    posdict = defaultdict(list)

    reads_passed = 0
    for line in _iter_alignment_lines(alib):
        if line.startswith("@"):
            continue
        ent = line.rstrip("\n").split("\t")
        aflag = int(ent[1])
        if aflag not in {0, 256, 16, 272}:
            continue

        aname = ent[0].strip()
        achr = ent[2]
        apos = int(ent[3])
        atag = ent[9].strip()
        alen = len(atag)
        aabun = int(aname.split("|")[-1])
        astrand = "w" if aflag in {0, 256} else "c"
        try:
            amismat = int(ent[-7].rpartition(":")[-1])
            ahits = int(ent[-1].rpartition(":")[-1])
        except Exception:
            continue

        if ahits < maxhits and amismat <= mismat:
            reads_passed += 1
            anid = make_akey(lib_stem(alib), achr)

            adj_abun = aabun
            if norm and total_abund and total_abund > 0:
                adj_abun = max(round((aabun / total_abund) * norm_factor), 1)

            taginfo = [achr, astrand, ahits, atag, aname, apos, alen, adj_abun]
            tempdict1[anid].append((apos, taginfo))
            posdict[anid].append(apos)

    nestdict = defaultdict(list)
    for akey, aval in tempdict1.items():
        tmp = defaultdict(list)
        for p, tinfo in aval:
            tmp[p].append(tinfo)
        nestdict[akey].append(tmp)

    poscountdict = {
        akey: OrderedDict(sorted(Counter(aval).items(), key=lambda x: int(x[0])))
        for akey, aval in posdict.items()
    }

    with open(outfile1, "wb") as f1:
        pickle.dump(nestdict, f1, protocol=pickle.HIGHEST_PROTOCOL)
    with open(outfile2, "wb") as f2:
        pickle.dump(poscountdict, f2, protocol=pickle.HIGHEST_PROTOCOL)
    with open(asum, "a") as fsum:
        fsum.write(f"Reads passed filters for {alib}:\t{reads_passed}\n")

    del tempdict1, posdict, nestdict, poscountdict
    gc.collect()

    return outfile1, outfile2


def lib_stem(p: str) -> str:
    """'.../ALL_LIBS.bam' -> 'ALL_LIBS'; '.../ALL_LIBS.sam' -> 'ALL_LIBS'."""
    return os.path.splitext(os.path.basename(p))[0]


def make_akey(lib_id: str, chr_id) -> str:
    """Consistent akey constructor."""
    return f"{lib_id}-{chr_id}"
