import shutil
import sys

from phasis.samtools import SamtoolsError, validate_runtime_samtools


def _has_executable(cmd_name):
    """
    Return True if executable is found in PATH.
    """
    return shutil.which(cmd_name) is not None


def _print_status(label, found):
    """
    Keep legacy-style status prints.
    """
    status = "found" if found else "missing"
    print(f"--{label:<30}: {status}")


def checkDependency():
    """
    Validate required runtime dependencies (legacy-compatible behavior).
    Exits on failure.
    """
    # Python version check (legacy says 3.10+)
    py_ok = sys.version_info >= (3, 10)
    _print_status("Python v3.10 or higher", py_ok)

    # External tools used in current pipeline
    hisat_ok = _has_executable("hisat2")
    _print_status("Hisat (v2)", hisat_ok)

    # The CLI has already validated the absolute runtime path. This re-check
    # deliberately validates that same path rather than resolving PATH again.
    try:
        validate_runtime_samtools(announce=False)
        samtools_ok = True
    except SamtoolsError as exc:
        samtools_ok = False
        samtools_error = str(exc)
    _print_status("Samtools", samtools_ok)

    # Optional: if your legacy checkDependency includes more tools, add them here
    # and preserve names/prints exactly.

    if not py_ok:
        print("Please use Python 3.10 or higher")
        sys.exit()

    if not hisat_ok:
        print("HISAT2 not found in PATH. Please install or activate the correct environment.")
        sys.exit()

    if not samtools_ok:
        print(samtools_error)
        sys.exit()
