# Hive validation and release review

## Accepted scope

The checksummed Phasis 2.8.4 candidate passed Linux/Hive source, wheel and local
Conda installation checks, 277 tests, three-aligner integration, and the full
four-library 21-*PHAS* example. The user accepts this evidence to proceed with
release review without waiting for the full 24-*PHAS* run.

At handoff, job 25169918 was reported still running; full 24-*PHAS* completion
was **unverified**. The automatic combined report (25169920) was also pending.
Neither is recorded as PASS or FAIL. This record does not assert their current
remote status. No new remote jobs were started for this local evidence review.

## Immutable tested candidate

Base development commit: `1415236b6da3ac78e2e3cf9703e560b3506b1ca3`.
The candidate includes uncommitted documentation/packaging changes; the commit
alone does not identify it. Hive verified all 94 source-file hashes in
`incoming/BUILD_PROVENANCE.json`. The local review independently confirmed
that all 94 files matched before the post-Hive documentation edits.

| Artifact | SHA256 |
| --- | --- |
| Tested source ZIP | `ad2a24cd548ec357dc3e805b1b88017b05db60edbef05e904d8ee6ba77a1341e` |
| Supplied sdist | `2b94c55dce397e057779a7f7cb91cfc13502a99aaf5656ab3160dfa7ebd7fc64` |
| Supplied wheel | `e3fb6fecfc38b3c281c198f08d2ea60046b5145d7b85e332d60b2f0a93be0644` |
| Hive-built wheel | `d0c1c550bd26978b4911ea5d11d08a56e4fb1ba7dd9cd3ac3314c35f13317afa` |
| Hive-built sdist | `f8c25d87731caf7f115b166d5346bdf9b80abd137f7cf1182cb39b99013195e6` |
| Returned evidence tarball | `9e04f4b2d5bbdd68a2901c3564fded26f0f9fce0e5e79816463afb39d33439d6` |

The original local bundle remains unchanged at
`dist/installation_validation_20261027/`. The returned evidence remains at
`~/Downloads/phasis-v284-hive-validation/`; its tarball checksum and the five
main 21-*PHAS* output checksums were independently verified locally.

Persistent Hive root:
`/quobyte/bcmeyersgrp/thales/phasis_submission_workflows/source/revision_runs/phasis_v284_install_validation_20261027`.

## Verified evidence

Linux x86_64, glibc 2.35, Python 3.12.16. Main source/wheel environments used
HISAT2 2.2.3 and SAMtools 1.24; the local Conda package test used SAMtools 1.25.
Exact package freezes and explicit Conda specifications remain in `reports/`.

| Check | Evidence relative to returned bundle | Outcome |
| --- | --- | --- |
| Source provenance | `reports/SOURCE_VERIFIED.json` | PASS; 94 files |
| Source installation | `reports/source_installation.json` | PASS |
| Supplied and Linux-built wheels | `reports/supplied_wheel_installation.json`, `reports/wheel_installation.json` | PASS; imports outside source |
| License, archive contents, entry points | `reports/ARTIFACT_CHECKS.json` | PASS; supporting scripts in sdist, not wheel |
| FASTA/FASTQ/gzip FASTQ conversion | `reports/CONVERTERS.json` | PASS |
| Regression suite | `logs/unit_tests_25165700.log`, `reports/auxiliary_exit_codes.tsv` | PASS; 277 tests, exit 0 |
| Three-aligner integration | `hive-bam-smoke/INTEGRATION_SUCCESS.json` | PASS; native/HISAT2 parity, 12 calls per arm |
| Local Conda build/tests | `logs/recipe_build_25166589.log`, `reports/recipe_build.PASS` | PASS |
| Local Conda artifact installation | `reports/recipe_installation.json`, `reports/RECIPE_SUCCESS` | PASS; job 25169919 |
| Four-library 21-*PHAS* example | `reports/README_PHASE21.json`, `reports/readme_phase21.time.txt` | PASS; analysis exit 0 |

### Full 21-PHAS example

The original analysis job was 25165701; corrected output verification was
performed by job 25169918. The command used fresh indexing/intermediates,
`-mindepth 1 -phase 21 -libformat T -cores 12 -maxhits 25`, the B73 RefGen v2
reference and the four README libraries. Exact argv and working directory are
in `reports/readme_phase21_command.json`.

| Library | Call-table rows |
| --- | ---: |
| W23_2.0_1 | 202 |
| W23_2.0_2 | 214 |
| sTP_dcl5_1_2.0 | 330 |
| sTR_dcl5_1_2.0 | 338 |
| Total | 1,084 |

There are 349 distinct locus identifiers. The phasiRNA export contains 27,593
assignment rows, not 27,593 unique phasiRNA sequences. Outputs include the
calls/all-clusters/evidence tables, GFF, phasiRNA TSV, four top-level PDFs,
and 996 locus PNGs. The local review independently reproduced the call counts
and plot-file counts from the downloaded outputs. No warnings or font warnings
were reported for this run.

Measured wall time was 1:27:12 and maximum resident set size reported by
`/usr/bin/time` was 12,013,132 KiB (about 11.46 GiB). These are observations from
this run, not universal runtime or scheduler-memory guarantees.

This installation example uses the README's four libraries; it does not
replace the frozen manuscript benchmarks or the homozygous-anther experiment.

## Resolved issues: environment and validation harness only

| Issue | Resolution and attribution |
| --- | --- |
| Personal Miniconda conflicted with cluster module | Isolated the batch shell from inherited Conda variables/functions before loading the module. No personal environment uninstalled. |
| Conda cache verification failures | Retried with a fresh private Quobyte cache; original logs retained. The handoff highlights SciPy; logs also include other cached packages. Not a Phasis source defect. |
| Disposable conda-build launcher selected the wrong Python after PATH changes | Backed up the launcher and pinned it to its own environment Python. `fix_build_launcher.py` records the fix; not a Phasis or recipe patch. Do not apply this globally to users' installations. |
| Output verifier expected locus PDFs | Corrected the harness to check PNGs. The already-completed 21 run passed re-verification; no scientific rerun or output-format change. |
| Conda test-install command mixed a package filename and Python spec | Used a local channel and `phasis=2.8.4=py_0` instead. Job 25169919 passed. |

Historical `.FAILED` markers are retained evidence of recovered attempts, not
unresolved software failures. The full 24-*PHAS* outcome is a separate,
unverified item; it is not inferred from these recoveries.

## Release-review boundary

Post-Hive changes in this review are documentation and its packaging: validation
records, release notes, HPC installation/build guidance, and a `MANIFEST.in`
entry to include the linked packaging guides/recipe in the source distribution.
Scientific source, supporting scripts, dependencies, and the recipe are not changed. The new
review snapshot must have a new hash and reference the tested ZIP above as its
parent; do not overwrite or relabel the original archives as newly tested.

For documentation-only repackaging, repeat Markdown link/shell syntax checks,
package builds, license/content checks, and installation smoke checks; compare
all software files to the tested candidate. The existing 277-test and full
21-run evidence applies to those unchanged software files, not to a claim
that the new archive itself ran on Hive. Any later runtime/dependency/recipe
change requires targeted revalidation before release approval.

Local post-Hive review checks passed: shell syntax and local documentation
links, source/wheel builds, license and source-archive contents, and rebuilt
wheel CLI/import/dependency smoke tests. Every member of the rebuilt wheel is
byte-identical to the tested supplied wheel. This is not a new clean Linux
test; it confirms unchanged installed package content after documentation edits.
The separate review bundle is `dist/release_review_hive21_20261027/`, with
new archive checksums, parent-candidate provenance, and retained small evidence
reports. The old bundle and downloaded evidence were not modified.

Remaining publication steps require separate approval: finalize source/tag
selection, verify newly built artifact hashes, review the documentation, then
authorize the release/tag and any Bioconda PR. The draft recipe's public
source checksum remains a placeholder. Successful local Conda builds do not
replace official Bioconda CI or maintainer approval. Frozen manuscript
archives are untouched; nothing has been published or uploaded in this review.
