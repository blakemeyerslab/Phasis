# Changelog

## 2.8.8 - 2026-11-17

A speed and memory release. The detected *PHAS* and *PHAS*-like loci are the
same as in 2.8.4: on the README maize example (21- and 24-nt runs) every result
table is identical to 2.8.4, and only some intermediate tables list the same
rows in a different order. The same held with `-norm`, with FASTQ input, and
on macOS (see Validation below).

### Speed and memory compared with 2.8.4

README maize example (four tag-count libraries, B73 RefGen v2), 12 cores,
`-mindepth 1 -maxhits 25`, Linux:

| Run | 2.8.4 | 2.8.8 |
| --- | --- | --- |
| 21-nt run, whole (includes building the index) | 91.6 min | 52.5 min |
| 24-nt run, whole (index and mapping reused) | 490 min | 48 min |
| Peak memory of the largest process, 21-nt run | 11.5 GB | 5.9 GB |
| Peak memory of the largest process, 24-nt run | 12.4 GB | 7.8 GB |

The 24-nt run of 2.8.8 shared the machine with another job using about 54 of
its 80 cores; its Phase II alone took 28 min when rerun on a quiet machine.

Stages that changed most (24-nt run, minutes):

| Stage | 2.8.4 | 2.8.8 |
| --- | --- | --- |
| Cluster scoring | 76.9 | 1.3 |
| Aggregation and scoring windows | 30.3 | 13.1 |
| Feature assembly | 367.8 | 13.7 |

Other measurements:

- macOS (Apple M5 Pro, 18 cores, maize chromosomes 9 and 10, two libraries,
  `-norm`): 24-nt run 27.8 min in 2.8.4 against 3.9 min in 2.8.8, and
  21-nt run 7.0 against 6.0 min; maximum resident memory 11.8 GB against
  7.2 GB.
- FASTQ input (Linux, 40 cores, maize chromosomes 9 and 10, two libraries of
  18 and 20 million reads, `-norm`): 21-nt run 11.6 against 7.5 min, 24-nt
  run 13.6 against 2.9 min; peak memory of Phasis and its workers 4.5 against
  2.5 GB.

Times depend on the machine and its load; the ratios are the useful part.

### Validation

- README maize example, Linux, 21 and 24 nt: 146 output files compared with
  2.8.4, no differences in content.
- Maize chromosomes 9 and 10 with `-norm` and with FASTQ input: result
  tables identical to 2.8.4. `--pool_libraries` was checked the same way on
  an earlier development build (dev/v2.8.6).
- macOS: unit tests pass and outputs match 2.8.4 on the chromosome 9 and 10
  test with `-norm`.
- Not tested: the optional external-BAM importer (its 48 unit tests are
  skipped unless `Phasis[bam]` is installed).

### Changed

- Per-position cluster scoring computes the same hypergeometric,
  Mann-Whitney and Stouffer values without one SciPy call per test, and labels
  phased positions inside the scoring window only. Scored-cluster files and all
  result tables were identical to 2.8.4 on the README maize example for 21- and
  24-nt runs. Set `Phasis_CLUSTER_SCORING_LEGACY_STATS=1` to use the former
  code path.
- The alignment parser stores the parsed tags of each library in one file per
  chromosome (`<library>_<phase>.dict.shards/`) with a small index at the
  former `.dict` path, and every cluster-scoring worker reads its own
  chromosome. The main process no longer loads a library's tags or sends them
  to the workers, which shortens scoring and lowers peak memory; scored
  clusters are unchanged. A `.dict` written by 2.8.4 is still accepted, but a
  run directory parsed by this version cannot be reused by 2.8.4.
- The Howell phasing-score scans in feature assembly build each window's
  position set and total once instead of once per register. Feature rows are
  unchanged and the step runs about twice as fast on 24-nt data.
- The Howell phasing-score scans in feature assembly score all windows and
  registers of a cluster strand with array operations instead of Python loops
  over windows, registers and cycles. The array form is used when the
  abundances are non-negative whole numbers, which is the case with and
  without `-norm` (normalised abundances are rounded); any other input keeps
  the former loops. Feature rows are unchanged and feature
  assembly ran 5 to 9 times as fast on maize 21- and 24-nt test batches. Set
  `Phasis_HOWELL_ARRAY_SCAN=0` to use the loops.
- PHAS-cluster building, window selection and feature assembly still start
  with two concurrent batches, but after the first clean group they go to
  their full worker count at once when the measured peak memory of a worker
  shows that the full count fits in half of the available memory. When it does
  not fit, or the memory cannot be read (non-Linux systems), the former
  stepwise doubling applies. Set `Phasis_MEMORY_AWARE_RAMP=0` to always use
  the stepwise doubling.
- Completed Phase II tables, the cleaned reference and the processed
  libraries are compressed at gzip level 6 instead of 9. On
  a maize test table this is about three times as fast for a file about 2%
  larger; the tables inside are unchanged. Set
  `Phasis_INTERMEDIATE_GZIP_LEVEL` (1-9) to choose another level.
- Feature assembly reports its work while it runs: batches, input rows done
  and total, clusters, feature rows written, rate and estimated time left. Set
  `Phasis_FEATURE_ASSEMBLY_COUNT_ROWS=0` to skip counting the input rows.

### Added

- The `runtime_*` file of each run now has one line per finished stage with
  its start, end and duration in seconds, and the log prints a
  `[stage-time]` line when a stage ends. This covers the stages that had no
  timing before (mapping, classification, writing results) and the parts of
  Phase II between its progress bars.
- `--no_locus_plots` skips rendering the individual PHAS and PHAS-like locus
  plots. All result tables, including the phasiRNA tables, the GFF files and
  the summary PDFs are still written.

### Fixed

- `phasis -cleanup` and `-cleanup_all` now also remove the gzip-compressed
  Phase II tables (`*_PHAS_to_detect.tab.gz` and the other `.tab.gz` and
  `.tsv.gz` intermediates), which are the default since intermediates are
  compressed.

## 2.8.4 - 2026-10-27

This release has passed macOS and Linux/Hive validation within the scope
described below.

### Added

- Optional external-BAM import through the auxiliary
  `support_scripts/bam_to_candidates.py` tool. Supported HISAT2, Bowtie 1, and
  BWA-aln/samse evidence can feed native candidate generation and subsequent
  `phasis -steps class` classification without native HISAT2 mapping.
- Exact-match import by default, with an opt-in substitution limit, configurable
  reported-hit cutoff, and all-reported or unique-only retention policy.
  These policies act on reported evidence, not unknown genomic placements.
- Import manifests, abundance and reference validation, and regression and
  end-to-end synthetic checks. Clipped, gapped, spliced, paired-end, and
  otherwise incompatible BAMs remain unsupported. See the external-BAM guide.

### Fixed

- Reference preparation now distinguishes sampled fingerprinting from whole
  cleaned-FASTA compression, with elapsed times and byte-based compression
  progress. Compression level is unchanged.
- Replaced semibold legend headings with bold and scoped all Phasis plotting
  entry points to Matplotlib's bundled DejaVu Sans fonts and math text, avoiding
  reliance on optional system fonts or external TeX. Caller settings are
  restored afterward; font warnings are not suppressed.
- Corrected an indexing error message's issue-tracker link to the maintained
  repository.

### Documentation

- Reorganized the README from installation and an executable example to output
  interpretation and advanced workflows; added Git and ZIP installation,
  HPC guidance, and generic Bowtie 1/BWA tutorials.
- Added explanations of Howell scoring, phased units, and the phasiRNA export,
  including sequence-orientation and repeated-assignment cautions.
- Clarified the relationship between the original PHASIS and the completely
  rewritten Phasis 2.

### Packaging

- Declared Artistic-2.0 license metadata and included the unchanged license
  text in built packages.
- Included supporting scripts, tests, and public guides in the source
  distribution. Auxiliary scripts remain source-checkout tools, not installed
  Phasis commands.
- Prepared and locally tested a Bioconda recipe on macOS and Linux, now
  [submitted for review](https://github.com/bioconda/bioconda-recipes/pull/70852).
  Public Bioconda installation remains pending CI and maintainer approval.

### Validation

- Source, wheel, and local Conda installation checks passed on Linux/Hive.
- All 277 regression tests passed, along with real HISAT2/Bowtie 1/BWA-aln
  integration and native/importer parity checks on synthetic data.
- The complete four-library 21-*PHAS* README example passed on Hive:
  349 distinct locus identifiers and 1,084 library-specific call rows, with
  standard outputs and no warnings reported.

These results apply to the checksummed candidate recorded in the
[validation record](packaging/HIVE_VALIDATION_21.md). Documentation revisions
after testing do not change the tested software, but require separately
identified release artifacts. No Phasis source fixes were needed for the
recovered Hive environment and validation-script issues.
