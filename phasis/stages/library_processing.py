import configparser
import os
import sys

from phasis import libprep
from phasis import runtime as rt
from phasis.cache import MEM_FILE_DEFAULT, compute_md5_str
from phasis.parallel import run_parallel_with_progress

# Stage-local globals (only what libraryprocess needs)
mindepth = None
libformat = None
concat_libs = None
outdir = None
memFile = MEM_FILE_DEFAULT


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
    Keeps legacy behavior and cache semantics.
    """
    global mindepth, libformat, concat_libs, memFile

    sync_from_runtime()

    print("#### Fn: Lib Processor #######################")
    libs_to_process = []
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(memFile)

    # Ensure sections exist
    for sect in ("ADVANCED", "LIBRARIES", "FASTAS"):
        if not config.has_section(sect):
            config.add_section(sect)

    # Determine which original libs need processing (MD5-based)
    if str(mindepth) == str(config["ADVANCED"].get("mindepth", "")) and config.has_section("LIBRARIES"):
        libkeys = [key for key in config["LIBRARIES"]]
        for alib in libs:
            if alib in libkeys:
                cur = compute_md5_str(alib)
                prev = (config["LIBRARIES"].get(alib) or "").strip()
                if cur and prev and (cur == prev):
                    print(f"MD5 matches for library: {alib}")
                else:
                    print(f"MD5 doesn't match (or missing) for library: {alib}")
                    libs_to_process.append(alib)
            else:
                libs_to_process.append(alib)
    else:
        libs_to_process = libs.copy()

    libs_processed = []

    if libs_to_process:
        print("\nLibraries to be processed: %s" % (", ".join(libs_to_process)))

        # Step 1: Check library format (FASTA vs tag-count)
        check_func = libprep.isfasta if libformat == "F" else libprep.isfiletagcount
        print("Checking format:")
        format_results = run_parallel_with_progress(
            check_func, libs_to_process, desc="Checking format"
        )
        if _runtime_errors(format_results):
            sys.exit("One or more libraries failed format check; see errors above.")
        if not any(_valid_bool_results(format_results)):
            sys.exit("No libraries passed format check.")

        # Step 2: Convert or filter to .fas
        proc_func = libprep.dedup_process if libformat == "F" else libprep.filter_process
        print("Processing libraries:")
        proc_results = run_parallel_with_progress(
            proc_func, libs_to_process, desc="Filtering/Converting"
        )
        if _runtime_errors(proc_results):
            sys.exit("One or more libraries failed during filtering/conversion; see errors above.")
        libs_processed = _existing_path_results(proc_results)

        # IMPORTANT: even if only some inputs needed processing, return the full
        # expected .fas list for ALL requested libraries (existing + newly generated).
        expected_fas = [f"{alib.rpartition(".")[0]}.fas" for alib in libs]
        libs_all = [p for p in expected_fas if os.path.exists(p)]
        # preserve deterministic order (expected_fas order) and include any extra outputs
        for p in libs_processed:
            if p not in libs_all:
                libs_all.append(p)
        libs_processed = libs_all

        # Step 3: Record MD5 of original files to memory
        print("Recording MD5:")
        for alib in libs_to_process:
            md5 = compute_md5_str(alib) or ""
            config["LIBRARIES"][alib] = md5
        with open(memFile, "w") as fh:
            config.write(fh)

    else:
        # IMPORTANT: keep your missing-.fas regeneration fix here
        expected_fas = [f"{alib.rpartition('.')[0]}.fas" for alib in libs]
        missing_fas_libs = [
            alib for alib, fas in zip(libs, expected_fas) if not os.path.exists(fas)
        ]

        if missing_fas_libs:
            print("\nNo input-library MD5 changes, but missing processed .fas detected.")
            print("Reprocessing libraries to regenerate missing .fas: %s" % (", ".join(missing_fas_libs)))

            check_func = libprep.isfasta if libformat == "F" else libprep.isfiletagcount
            print("Checking format:")
            format_results = run_parallel_with_progress(
                check_func, missing_fas_libs, desc="Checking format"
            )
            if _runtime_errors(format_results):
                sys.exit("One or more libraries failed format check; see errors above.")
            if not any(_valid_bool_results(format_results)):
                sys.exit("No libraries passed format check.")

            proc_func = libprep.dedup_process if libformat == "F" else libprep.filter_process
            print("Processing libraries:")
            proc_results = run_parallel_with_progress(
                proc_func, missing_fas_libs, desc="Filtering/Converting"
            )
            if _runtime_errors(proc_results):
                sys.exit("One or more libraries failed during filtering/conversion; see errors above.")

            libs_processed = [p for p in expected_fas if os.path.exists(p)]
        else:
            print("\nNo new libraries to process this time.")
            libs_processed = [p for p in expected_fas if os.path.exists(p)]

    # Guard: filter out any Nones or missing files
    libs_processed = _existing_path_results(libs_processed)

    # Concatenation path
    if concat_libs:
        if not libs_processed:
            sys.exit("No processed libraries available to concatenate.")
        merged_dir = os.path.dirname(libs_processed[0]) or os.getcwd()
        merged_basename = "ALL_LIBS"
        merged_path = libprep.merge_processed_fastas(
            fas_paths=libs_processed,
            out_dir=merged_dir,
            out_basename=merged_basename,
            mindepth=mindepth,
        )
        print(f"[concat_libs] Created merged library: {merged_path}")

        fas_md5 = compute_md5_str(merged_path) or ""
        config["FASTAS"][merged_path] = fas_md5
        with open(memFile, "w") as fh:
            config.write(fh)

        return [merged_path]

    # Non-concat: record MD5 for each produced .fas in [FASTAS]
    updated = False
    for fas in libs_processed:
        if fas.endswith(".fas") and os.path.isfile(fas):
            fas_md5 = compute_md5_str(fas) or ""
            if (config["FASTAS"].get(fas) or "") != fas_md5:
                config["FASTAS"][fas] = fas_md5
                updated = True
    if updated:
        with open(memFile, "w") as fh:
            config.write(fh)

    return libs_processed
