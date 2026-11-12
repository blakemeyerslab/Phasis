# Changelog

## Unreleased (development branches dev/v2.8.5 and dev/v2.8.6)

Not yet released; the package version is still 2.8.4.

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
- Feature assembly reports its work while it runs: batches, input rows done
  and total, clusters, feature rows written, rate and estimated time left. Set
  `Phasis_FEATURE_ASSEMBLY_COUNT_ROWS=0` to skip counting the input rows.

### Added

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
