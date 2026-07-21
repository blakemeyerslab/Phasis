import os
import sys

from phasis import runtime as rt
from phasis.fastq import FastqFormatError, RawFastqInputError, preflight_fastq


# Stage-local globals (minimal)
libs = []
reference = None


def sync_from_runtime() -> None:
    """
    Populate input-validation stage globals from phasis.runtime.
    """
    global libs, reference
    libs = rt.libs
    reference = rt.reference


def fileexists(afile):
    """
    Test if file exists (keeps legacy prints for parity).
    """
    print("checking if file exists:%s" % (afile))
    if os.path.isfile(afile):
        abool = True
    else:
        abool = False
    print(f"File available:{abool}")
    return abool


def checkLibs():
    """
    Validate that requested libraries and reference exist.
    Returns libs (legacy-compatible behavior).
    """
    global libs, reference

    sync_from_runtime()

    notfound = []
    for alibs in libs:
        if fileexists(alibs) is False:
            notfound.append(alibs)

    if notfound:
        print("\nERROR:These sRNA libraries not found   : %s" % (",".join(notfound)))
        print("------Please check file exists at specified location")
        sys.exit()

    if fileexists(reference) is False:
        print("\nERROR:Reference genome or transcriptome not found:%s" % (reference))
        print("------Please check file exists at specified location")
        sys.exit()

    if str(getattr(rt, "libformat", "")).upper() == "Q":
        print("Preflighting FASTQ input before index construction:")
        for alib in libs:
            try:
                stats = preflight_fastq(alib)
            except (FastqFormatError, RawFastqInputError) as exc:
                sys.exit(f"FASTQ preflight failed: {exc}")
            print(
                f"  {alib}: checked {stats.sampled_reads} read(s); "
                f"{stats.sampled_long_reads} exceed {35} nt."
            )

    return libs
