# Installation validation: 2026-10-27

## Candidate and scope

Unpublished Phasis 2.8.4 working-tree candidate based on development commit
`1415236b6da3ac78e2e3cf9703e560b3506b1ca3`, with approved documentation edits
and local packaging changes. The staged ZIP and `BUILD_PROVENANCE.json`, not
the base commit alone, identify the tested source. No scientific code changed
in this installation task. No commit, push, tag, release, or recipe submission.

## Changes

- README now describes both Git and ZIP download, extraction, entering the
  source directory, editable installation, dependency checks, and the real
  example. Bioconda availability is explicitly pending.
- `pyproject.toml` declares Artistic-2.0 and `LICENSE.txt`, using setuptools
  >=77.0.3 for modern license metadata. The original license text is unchanged.
- `MANIFEST.in` includes public guides, supporting scripts, and tests in the
  Python source distribution. The wheel still installs only the package and
  its two existing CLI entry points; no auxiliary BAM CLI was added.
- A local draft Bioconda recipe declares Python/scientific dependencies,
  HISAT2, SAMtools, license, CLI entry points, imports, and runtime checks.
- A separate Hive handoff specifies independent Linux installation and usage
  checks against the checksummed candidate, not an older public checkout.

## Completed checks

Local host: macOS arm64. Fresh pip environments use Python 3.11.4; the Conda
recipe build/test uses Python 3.12. These are local checks, not Linux validation.

| Check | Result |
| --- | --- |
| Build source distribution and wheel with current setuptools | PASS |
| Wheel license expression, license-file metadata, and exact license bytes | PASS |
| Supporting scripts/tests/docs included in source distribution, not installed as auxiliary commands in wheel | PASS |
| Clean wheel installation including optional BAM dependency; CLI help/version and dependency check outside checkout | PASS |
| Extract validation ZIP; editable installation in second clean environment; version, import path, `pip check` | PASS |
| Documented FASTA and gzip-FASTQ converters on known repeated sequences | PASS; expected counts 2 and 1 |
| Unit/regression suite in fresh pip environment | PASS; 277 tests |
| Small real HISAT2/Bowtie 1/BWA-aln importer and classification integration | PASS; 12 calls per arm, native/HISAT2 importer parity, standard outputs |
| Font-warning scan of integration logs | No findfont or missing-glyph warnings found |
| Local Conda recipe build using source archive and real checksum instead of pending release URL | PASS; `phasis-2.8.4-py_0.conda` |
| Conda build's isolated test environment | PASS; imports, both CLI help commands, version and executable checks |
| Fresh environment installation of locally built Conda artifact | PASS |
| Conda-installed native 24-PHAS pipeline on synthetic tag-count input, from indexing through plots | PASS; 12 loci, standard tables/GFF/phasiRNA exports/PDFs; no font warnings |
| Public documentation shell syntax and local file links | PASS; 36 shell blocks parse |
| Whitespace/diff validation | PASS |

The initial full native CLI smoke run using the integration test's collapsed
FASTA completed with zero calls. That input is not a count-preserving native
FASTA library: its header counts are for the external importer. It is not
evidence of biological parity. The separate tag-count native smoke run
preserved the original abundances and successfully produced 12 calls.

The Conda HISAT2 2.2.3 `--version` text contains a compiler/linker diagnostic
embedded in its build information, despite exiting successfully. Actual
indexing/mapping must be checked rather than treating the version command as
proof of mapper usability. The local native smoke test exercises that path.

## Hive evidence accepted after local validation

The returned `LOCAL_CODEX_HANDOFF_21_VALIDATED.md` and supporting evidence
supersede the earlier pending status for Linux checks. See the
[Hive validation record](HIVE_VALIDATION_21.md) for immutable candidate hashes,
measured results, recovered environment/harness issues, and review provenance.

On Linux x86_64/glibc 2.35 with Python 3.12.16, the candidate passed source,
supplied-wheel, Linux-built-wheel, and local Conda installation checks;
277 tests; three-aligner integration; and the full four-library 21-*PHAS*
example. The example produced 349 distinct locus identifiers and 1,084 call
rows, with no warnings reported. Its analysis exited successfully before a
validation-script error incorrectly expected locus PDFs instead of PNGs;
correcting that check verified the existing outputs without rerunning it.

The user accepts this evidence to proceed without waiting for the full
24-*PHAS* example. Its completion was **unverified at handoff** (job 25169918),
not a measured PASS or FAIL. Synthetic 24-*PHAS* tests remain separate evidence.

## Remaining checks and prerequisites

- Full four-library 21-*PHAS* Linux validation is complete. Full 24-*PHAS*
  completion and the automatic combined report were unverified at handoff;
  neither is required to proceed under the user's accepted validation scope.
- Python 3.10 and Intel macOS were not separately tested in this task.
- The README's public `main` download currently retrieves the existing public
  version, not these uncommitted changes. Retest public download routes after
  the approved release source is available.
- The submission recipe still has an explicit release-checksum placeholder.
  Do not submit it until the approved `v2.8.4` tag archive is accessible and its
  checksum recorded. A GitHub Release page is not technically necessary, but
  this recipe requires its tagged source URL to exist.
- Local Conda testing is not official Bioconda lint/CI or maintainer approval.
  Submission and publication are not authorized by this validation task.

Comment 3's documentation changes are prepared and tested locally and on Hive.
Comment 1's recipe is prepared and locally built/tested on both platforms, but public installation remains
pending release source finalization, approved submission, and external review.

See [recipe instructions](bioconda/README.md) and
[Hive handoff](HIVE_INSTALLATION_HANDOFF.md).
